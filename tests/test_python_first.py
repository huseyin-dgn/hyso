"""Real bounded Python workflows and source model preservation after web retirement."""

import copy
import importlib.util
import json
from pathlib import Path
import ast
import subprocess
import sys

import pytest
import torch

from hyso.models import GraphModelBuilder, ModelBuilder
from hyso.models.output import GraphTensorOutput
from hyso.model_checks import count_parameters, preflight_model, run_model_checks
from hyso.architecture_comparison import compare_architectures
from hyso.model_observations import observe_result, diagnose_results, explain_diagnostic


@pytest.fixture
def config():
    return (
        GraphModelBuilder({"image": [1, 2, 2]})
        .add("head", "classification_head", ["image"], in_features=1, num_classes=2)
        .outputs(logits="head", features="image")
        .to_config()
    )


def test_adapter_shares_parameters_gradients_dtype_and_strict_state(config):
    graph = GraphModelBuilder.from_config(config).build().double()
    wrapped = GraphTensorOutput(graph, "logits", task="multiclass")
    assert list(wrapped.parameters()) == list(graph.parameters())
    x = torch.ones(2, 1, 2, 2, dtype=torch.float64, requires_grad=True)
    wrapped(x).sum().backward()
    assert x.grad is not None and all(p.grad is not None for p in graph.parameters())
    clone = GraphTensorOutput(
        GraphModelBuilder.from_config(config).build().double(), "logits"
    )
    clone.load_state_dict(wrapped.state_dict(), strict=True)
    assert torch.equal(wrapped(x), clone(x))
    assert wrapped(x).dtype == torch.float64


@pytest.mark.parametrize(
    "output,task",
    [
        ("missing", None),
        ("features", "multiclass"),
        ("logits", "regression"),
        ("logits", "binary"),
    ],
)
def test_invalid_explicit_output_training_guards(config, output, task):
    with pytest.raises(ValueError):
        GraphTensorOutput(
            GraphModelBuilder.from_config(config).build(), output, task=task
        )


def test_multi_input_adapter_rejected():
    graph = (
        GraphModelBuilder({"a": [1, 2, 2], "b": [1, 2, 2]})
        .add("sum", "add", ["a", "b"])
        .outputs(value="sum")
        .build()
    )
    with pytest.raises(ValueError, match="exactly one"):
        GraphTensorOutput(graph, "value")


def test_preflight_and_comparison_do_not_allocate(config, monkeypatch):
    def fail(*a, **kw):
        raise AssertionError("allocated")

    monkeypatch.setattr(torch.nn.Module, "__init__", fail)
    monkeypatch.setattr(torch, "empty", fail)
    assert preflight_model(config)["valid"]
    result = compare_architectures([config, copy.deepcopy(config)])
    assert result["equivalent_groups"] == [[0, 1]]
    assert run_model_checks(config, tests=["structure"])["cases"][0]["status"] == "PASS"


@pytest.mark.parametrize("schema", [1, 2, 3])
def test_real_checks_all_schemas_source_state_and_rng(config, schema):
    if schema == 1:
        config = (
            ModelBuilder((1, 2, 2), 2)
            .add("conv2d", in_channels=1, out_channels=2, kernel_size=1)
            .to_config()
        )
    if schema == 2:
        config = (
            ModelBuilder.sequential((1, 2, 2), num_classes=2)
            .add("conv2d", in_channels=1, out_channels=1, kernel_size=1, padding=0)
            .to_config()
        )
    builder = GraphModelBuilder if schema == 3 else ModelBuilder
    model = builder.from_config(config).build().double().train()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    states = {k: v.clone() for k, v in model.state_dict().items()}
    grads = [p.grad.clone() for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    result = run_model_checks(
        config,
        model=model,
        model_id="test-model",
        tests=["structure", "forward", "backward", "serialization", "parameters"],
    )
    assert all(c["status"] == "PASS" for c in result["cases"]), result
    assert model.training and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, states[k]) for k, v in model.state_dict().items())
    assert all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), grads))
    assert result["parameter_count"] == count_parameters(config, model=model)
    assert result["model_id"] == "test-model" and len(result["state_sha256"]) == 64


def test_invalid_graph_diagnostics_without_mutation(config):
    bad = copy.deepcopy(config)
    bad["nodes"][0]["params"]["in_features"] = 9
    before = copy.deepcopy(bad)
    result = run_model_checks(bad)
    assert result["cases"][0]["status"] == "FAIL"
    assert all(c["status"] == "SKIP" for c in result["cases"][1:])
    diagnostic = result["preflight"]["diagnostics"][0]
    explanation = explain_diagnostic(diagnostic, bad, result["preflight"])
    assert explanation["observed_condition"]["en"] and bad == before


def test_actual_failed_forward_has_no_fake_pass(config, monkeypatch):
    from hyso.models.graph import GraphModel

    monkeypatch.setattr(
        GraphModel,
        "forward",
        lambda self, x: (_ for _ in ()).throw(RuntimeError("actual forward failure")),
    )
    result = run_model_checks(config, tests=["forward", "backward", "serialization"])
    assert [r["status"] for r in result["cases"]] == ["FAIL", "SKIP", "SKIP"]
    assert "actual forward failure" in result["error"]


def test_exact_architecture_comparison_is_lossless(config):
    second = copy.deepcopy(config)
    second["nodes"][0]["params"]["num_classes"] = 3
    originals = copy.deepcopy([config, second])
    result = compare_architectures([config, second], exact_parameters=True)
    assert [c["parameter_count"] for c in result["columns"]] == [4, 6]
    assert result["columns"][1]["parameter_delta"] == 2
    assert [config, second] == originals


def test_large_fixture_is_skipped_without_build(config, monkeypatch):
    huge = (
        GraphModelBuilder({"image": [1, 512, 512]})
        .add("head", "classification_head", ["image"], in_features=1, num_classes=2)
        .outputs(value="head")
        .to_config()
    )
    monkeypatch.setattr(
        GraphModelBuilder,
        "build",
        lambda self: (_ for _ in ()).throw(AssertionError("allocated")),
    )
    result = run_model_checks(huge, tests=["forward"])
    assert (
        result["cases"][0]["status"] == "SKIP"
        and "65536" in result["cases"][0]["reason"]
    )


def test_python_example_a_to_i_and_native_observations(tmp_path):
    p = Path(__file__).resolve().parents[1] / "examples/models/python_first.py"
    spec = importlib.util.spec_from_file_location("python_first_example", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run(tmp_path)
    assert result["experiment_runs"] == 1
    assert all(c["status"] == "PASS" for c in result["checks"]["cases"])
    assert result["evaluation"]["sample_count"] == 4

    # Real result from the completed example, no forged TrainingResult constructor.
    data = json.loads((tmp_path / "report.json").read_text())
    assert data
    from hyso.evaluation import EvaluationResult

    evaluation = EvaluationResult(
        json.loads((tmp_path / "evaluation.json").read_text())
    )
    observation = observe_result(
        evaluation,
        config=module.architecture(),
        model_id="A",
        dataset="synthetic-demo",
        split="synthetic-demo",
    )
    assert diagnose_results([observation]) == []


def test_native_training_thresholds_are_preserved(config):
    from hyso.training import TrainingConfig
    from torch.utils.data import TensorDataset

    model = GraphTensorOutput(
        GraphModelBuilder.from_config(config).build(), "logits", task="multiclass"
    )
    settings = TrainingConfig(epochs=4, batch_size=2, shuffle=False, learning_rate=1e-9)
    train, _ = settings.make_loaders(
        TensorDataset(torch.zeros(2, 1, 2, 2), torch.tensor([0, 1]))
    )
    result = settings.prepare(model, train).fit()
    observation = observe_result(result, config=config, model_id="training-A")
    rules = diagnose_results([observation])
    assert any(r["rule"] == "LOSS_PLATEAU" for r in rules)
    assert all(
        r["category"] in ("HYPOTHESIS", "INSUFFICIENT_EVIDENCE") and not r["action"]
        for r in rules
    )


def test_check_failure_regression_and_candidate_do_not_modify_model(
    config, monkeypatch
):
    from hyso.models.graph import GraphModel
    from hyso.model_observations import observe_model_checks, suggest_model_corrections

    passed = run_model_checks(config, tests=["forward"])
    monkeypatch.setattr(
        GraphModel,
        "forward",
        lambda self, x: (_ for _ in ()).throw(RuntimeError("failure")),
    )
    failed = run_model_checks(config, tests=["forward"])
    before = observe_model_checks(
        passed, model_id="A", observed_at="2026-11-14T00:00:00+00:00"
    )
    after = observe_model_checks(
        failed, model_id="A", observed_at="2026-11-14T01:00:00+00:00"
    )
    rules = diagnose_results([before, after])
    assert {"TEST_FAILURE", "TEST_REGRESSION"} <= {r["rule"] for r in rules}
    bad = (
        GraphModelBuilder({"image": [1, 2, 2]})
        .add("conv", "conv2d", ["image"], in_channels=1, out_channels=2, kernel_size=1)
        .outputs(value="conv")
        .to_config()
    )
    bad["nodes"][0]["params"]["in_channels"] = 3
    original = copy.deepcopy(bad)
    candidates = suggest_model_corrections(bad)
    assert candidates[0]["candidate_status"] == "STATICALLY_VALIDATED"
    assert bad == original


def test_preserved_component_json_examples_preflight():
    configs = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "examples/models/component-configs.json"
        ).read_text()
    )
    assert len(configs) == 60
    for name, config in configs.items():
        assert preflight_model(config)["valid"], name


def test_web_free_runtime_imports_and_cli():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "hyso/studio").exists()
    assert importlib.util.find_spec("hyso.studio") is None
    for path in (root / "hyso").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("hyso.studio")
            if isinstance(node, ast.Import):
                assert all(not a.name.startswith("hyso.studio") for a in node.names)
    for group in [
        "",
        "models",
        "train",
        "analysis",
        "compare",
        "advisor",
        "experiments",
        "reports",
    ]:
        args = [sys.executable, "-m", "hyso"] + ([group] if group else []) + ["--help"]
        result = subprocess.run(
            args, cwd=root, capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr
    removed = subprocess.run(
        [sys.executable, "-m", "hyso.studio"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert removed.returncode != 0 and "hyso.studio" in removed.stderr
