"""Stage 13.5.14 independent arithmetic and bounded evidence acceptance.

All data are deterministic QA tensors. No retained Digits sample is evaluated.
The declared-scope test records the historical P26 policy boundary. Stronger
dataset/preprocessing rejection probes have their own explicit acceptance module.
"""

import copy
import gc
import hashlib
import json
from dataclasses import FrozenInstanceError
from fractions import Fraction
from pathlib import Path
import os
import subprocess
import sys
import weakref

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, default_collate

from hyso import EvaluationResult, GraphModelBuilder, ModelBuilder
from hyso import evaluate_classification, evaluate_predictions, ExperimentManager
from hyso._evaluation_evidence import checkpoint_evidence, model_hash
from hyso.analysis import (
    AnalysisContext, EvidenceError, EvidenceIndex, EvidenceRecord, EvidenceScope,
    ModelComparisonSource, compare_models,
)
from hyso.analysis.graph_evidence import (
    GraphEvaluationResult, compare_graph_evaluations, evaluate_graph_classification,
)
from hyso.architecture_comparison import compare_architectures
from hyso.checkpoint import load_training_checkpoint, save_training_checkpoint
from hyso.evaluation import _hash
from hyso.reporting import create_report
from hyso.structure import capture_model_structure
from hyso.structure_comparison import compare_model_structures


@pytest.fixture(autouse=True)
def bounded_cpu_work(monkeypatch):
    from hyso.benchmark import BenchmarkRunner
    from hyso.training import Trainer

    def forbidden(*args, **kwargs):
        raise AssertionError("Training, benchmarking and CUDA are outside this QA")

    monkeypatch.setattr(Trainer, "fit", forbidden)
    monkeypatch.setattr(BenchmarkRunner, "run", forbidden)
    for name in ("init", "is_available", "device_count", "current_device"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def rational_metrics(targets, predictions, classes, zero):
    """Independent counts and rational arithmetic, without production helpers."""
    matrix = [
        [sum(t == c and p == d for t, p in zip(targets, predictions))
         for d in range(classes)]
        for c in range(classes)
    ]
    rows = []
    for c in range(classes):
        true_positive = sum(t == p == c for t, p in zip(targets, predictions))
        support = targets.count(c)
        predicted = predictions.count(c)
        numerators = (true_positive, true_positive, 2 * true_positive)
        denominators = (predicted, support, predicted + support)
        row = {name: float(Fraction(n, d)) if d else float(zero)
               for name, n, d in zip(("precision", "recall", "f1"), numerators, denominators)}
        rows.append(dict(row, support=support, predicted=predicted))
    accuracy = float(Fraction(sum(t == p for t, p in zip(targets, predictions)), len(targets)))
    return matrix, rows, accuracy


METRIC_CASES = [
    ("perfect", [0, 1, 2, 0], [0, 1, 2, 0], 3),
    ("wrong", [0, 1, 2, 0], [1, 2, 0, 2], 3),
    ("collapse", [0, 1, 2, 0], [1, 1, 1, 1], 3),
    ("imbalanced", [0] * 9 + [1], [0] * 8 + [1, 0], 3),
    ("missing_prediction", [0, 1, 2, 2], [0, 1, 1, 1], 3),
    ("zero_support", [0, 0, 1, 1], [0, 2, 1, 2], 4),
    ("ten_classes", list(range(10)) * 2, list(range(10)) + [0] * 10, 10),
]


@pytest.mark.parametrize("zero", [0, 1])
@pytest.mark.parametrize("name,targets,predictions,classes", METRIC_CASES, ids=[r[0] for r in METRIC_CASES])
def test_independent_metrics_on_actual_receiver(name, targets, predictions, classes, zero):
    logits = torch.full((len(targets), classes), -4.0)
    logits[torch.arange(len(targets)), predictions] = 4.0
    result = evaluate_classification(
        nn.Identity(), DataLoader(TensorDataset(logits, torch.tensor(targets)), batch_size=3),
        class_names=[f"class-{i}" for i in range(classes)], split="qa",
        zero_division=zero, capture_evidence=True,
    ).to_dict()
    metrics = result["metrics"]
    matrix, rows, accuracy = rational_metrics(targets, predictions, classes, zero)
    assert result["raw"]["targets"] == targets
    assert torch.tensor(result["raw"]["predictions"]).argmax(1).tolist() == predictions
    assert metrics["confusion_matrix"] == matrix
    assert metrics["sample_count"] == sum(map(sum, matrix)) == len(targets)
    correct = sum(matrix[i][i] for i in range(classes))
    assert correct == sum(t == p for t, p in zip(targets, predictions))
    assert len(targets) - correct == sum(t != p for t, p in zip(targets, predictions))
    assert metrics["accuracy"] == accuracy
    assert sum(result["accounting"]["batch_sizes"]) == len(targets)
    for c, reference in enumerate(rows):
        assert sum(matrix[c]) == reference["support"] == metrics["per_class"][c]["support"]
        assert sum(row[c] for row in matrix) == reference["predicted"]
        for metric in ("precision", "recall", "f1"):
            assert metrics["per_class"][c][metric] == pytest.approx(reference[metric], abs=1e-15, rel=1e-14)
    for metric in ("precision", "recall", "f1"):
        assert metrics["averages"]["macro"][metric] == pytest.approx(sum(r[metric] for r in rows) / classes, abs=1e-15)
        assert metrics["averages"]["weighted"][metric] == pytest.approx(sum(r[metric] * r["support"] for r in rows) / len(targets), abs=1e-15)
        assert metrics["averages"]["micro"][metric] == accuracy


@pytest.mark.parametrize("task,logits,targets,expected", [
    ("binary", [[-1.0], [0.0], [1.0]], [0.0, 1.0, 1.0], [[1, 0], [0, 2]]),
    ("multiclass", [[1.0, 1.0], [0.0, 2.0]], [0, 1], [[1, 0], [0, 1]]),
])
def test_explicit_binary_and_two_class_multiclass(task, logits, targets, expected):
    result = evaluate_classification(
        nn.Identity(), DataLoader(TensorDataset(torch.tensor(logits), torch.tensor(targets)), batch_size=2),
        class_names=["negative", "positive"], split="qa", task=task,
    ).to_dict()
    assert result["configuration"]["task"] == task
    assert result["metrics"]["confusion_matrix"] == expected


@pytest.mark.parametrize("targets,logits,task", [
    ([0, 1], [[1.0, 0.0], [0.0, 1.0]], "binary"),
    ([0], [[1.0, 0.0, 0.0]], "multiclass"),
    ([2], [[1.0, 0.0]], "multiclass"),
    ([-1], [[1.0, 0.0]], "multiclass"),
    ([0], [[float("nan"), 0.0]], "multiclass"),
    ([0], [[float("inf"), 0.0]], "multiclass"),
])
def test_invalid_task_width_targets_and_nonfinite(targets, logits, task):
    labels = torch.tensor(targets).float() if task == "binary" else torch.tensor(targets)
    with pytest.raises(ValueError):
        evaluate_classification(
            nn.Identity(), DataLoader(TensorDataset(torch.tensor(logits), labels), batch_size=2),
            class_names=["a", "b"], split="qa", task=task,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan-logit", "inf-logit"])
def test_nonfinite_output_with_finite_input_is_rejected(value):
    model = nn.Linear(1, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(torch.tensor([value, 0.0]))
    with pytest.raises(ValueError, match="logits"):
        evaluate_classification(model, DataLoader(TensorDataset(
            torch.ones(2, 1), torch.tensor([0, 1])), batch_size=2), class_names=["a", "b"], split="qa")


@pytest.mark.parametrize("extra", [-1, 1])
def test_collation_cannot_under_or_over_account(extra):
    def collate(items):
        x, y = default_collate(items)
        return (x[:-1], y[:-1]) if extra < 0 else (torch.cat([x, x[:1]]), torch.cat([y, y[:1]]))

    with pytest.raises(ValueError):
        evaluate_classification(nn.Identity(), DataLoader(
            TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long)), batch_size=2,
            collate_fn=collate), class_names=["a", "b"], split="qa", capture_evidence=True)


def test_empty_and_class_order_are_explicit():
    result = evaluate_classification(nn.Identity(), DataLoader(
        TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)), batch_size=2),
        class_names=["z", "a"], split="qa", capture_evidence=True).to_dict()
    assert result["status"] == "empty" and result["metrics"]["accuracy"] is None
    assert result["accounting"] == dict(expected_samples=0, batch_sizes=[])
    assert all(r["precision"] is None and r["support"] == 0 for r in result["metrics"]["per_class"])
    ordered = evaluate_predictions([0, 1], [1, 1], class_names=["z", "a"], split="qa").to_dict()
    assert [r["class_name"] for r in ordered["metrics"]["per_class"]] == ["z", "a"]


def classifier(threshold=0.0, *, wrapped=False, binary=False):
    model = nn.Linear(1, 1 if binary else 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0]] if binary else [[0.0], [1.0]]))
        model.bias.copy_(torch.tensor([-threshold] if binary else [0.0, -threshold]))
    return nn.Sequential(model, nn.Identity()) if wrapped else model


def produce(model=None, *, x=None, y=None, batch=2, **options):
    config = dict(class_names=["a", "b"], split="test", capture_evidence=True)
    config.update(options)
    if x is None:
        x = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    if y is None:
        y = torch.tensor([0, 0, 1, 1])
    if config.get("task") == "binary":
        y = y.float()
    return evaluate_classification(classifier() if model is None else model,
        DataLoader(TensorDataset(x, y), batch_size=batch), **config)


def source(model_id, evaluation, **scope):
    native = evaluation.to_dict() if isinstance(evaluation, EvaluationResult) else evaluation
    record = EvidenceRecord.from_native(native, feature_id="P26", scope=EvidenceScope(model_id=model_id, **scope))
    report = EvidenceIndex([record]).analyze(context=AnalysisContext(
        workload_id="stage13-5-14-small-qa", task="classification", model={"id": model_id}, dataset={"kind": "qa"}))
    return ModelComparisonSource(model_id, report, records={"P26": record.record_id})


def compare(left, right, **kwargs):
    return compare_models(source("A", left), source("B", right), features="P26", **kwargs)


def assert_withheld(result, status):
    data = result.to_dict()
    assert all(r["status"] == status and r["difference"] is None for r in data["metrics"])
    assert data["claims"]["global_winner"] is None


@pytest.mark.parametrize("structure", [False, True])
def test_actual_native_chain_compares_changed_architecture(structure):
    left = produce(classifier(1.5), capture_structure=structure)
    right = produce(classifier(wrapped=True), capture_structure=structure)
    result = compare(left, right, include_structure=structure).to_dict()
    accuracy = next(r for r in result["metrics"] if r["metric"] == "test_accuracy")
    assert accuracy["status"] == "COMPARABLE"
    assert accuracy["difference"]["difference"] == 0.25
    assert accuracy["difference"]["percentage_point_change"] == 25.0
    assert result["claims"] == dict(global_winner=None, statistical_superiority="not_established", acceptance="NOT_EVALUATED")
    if structure:
        assert result["structure"]["status"] == "COMPARABLE"


def test_equal_accuracy_different_per_class_has_no_winner():
    left = produce(classifier(1.5))
    right = produce(classifier(-1.5, wrapped=True))
    assert left.to_dict()["metrics"]["per_class"] != right.to_dict()["metrics"]["per_class"]
    result = compare(left, right).to_dict()
    assert all(r["status"] == "COMPARABLE" for r in result["metrics"])
    assert result["metrics"][0]["difference"]["difference"] == 0.0
    assert result["claims"]["global_winner"] is None


@pytest.mark.parametrize("change", ["input", "order", "target", "count", "classes", "vocabulary", "split", "task", "batch", "zero"])
def test_actual_incompatible_evidence_withholds_numbers(change):
    x = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    y = torch.tensor([0, 0, 1, 1])
    options = {}
    model = classifier()
    if change == "input":
        x[0] -= 1
    elif change == "order":
        x, y = x.flip(0), y.flip(0)
    elif change == "target":
        y[0] = 1
    elif change == "count":
        x, y = x[:3], y[:3]
    elif change == "classes":
        options["class_names"] = ["b", "a"]
    elif change == "vocabulary":
        options["class_names"] = ["a", "other"]
    elif change == "split":
        options["split"] = "validation"
    elif change == "task":
        options["task"] = "binary"
        model = classifier(binary=True)
    elif change == "batch":
        options["batch"] = 3
    elif change == "zero":
        options["zero_division"] = 1
    left, right = produce(classifier(1.5)), produce(model, x=x, y=y, **options)
    assert_withheld(compare(left, right), "INCOMPATIBLE")
    if change == "batch":
        assert left.to_dict()["producer_evidence"]["evaluated_stream"] == right.to_dict()["producer_evidence"]["evaluated_stream"]


@pytest.mark.parametrize("kind", ["schema1", "supplied"])
def test_missing_native_producer_is_insufficient(kind):
    right = produce(capture_evidence=False) if kind == "schema1" else evaluate_predictions(
        [0, 0, 1, 1], [0, 0, 1, 1], class_names=["a", "b"], split="test")
    assert_withheld(compare(produce(classifier(1.5)), right), "INSUFFICIENT_EVIDENCE")


@pytest.mark.parametrize("field", ["dataset_id", "preprocessing_id"])
def test_declared_scope_historical_policy_boundary(field):
    left = source("A", produce(classifier(1.5)), **{field: "declared-left"})
    right = source("B", produce(), **{field: "declared-right"})
    result = compare_models(left, right, features="P26").to_dict()
    assert all(r["status"] == "COMPARABLE" for r in result["metrics"])
    assert all(r["difference"] is not None for r in result["metrics"])
    assert result["claims"]["global_winner"] is None
    assert "preprocessing_not_verified" in " ".join(result["limitations"])


def test_copied_evaluation_identity_is_rejected():
    left, right = produce(classifier(1.5)), produce().to_dict()
    right["evaluation_id"] = left.to_dict()["evaluation_id"]
    right["source_sha256"] = _hash({k: v for k, v in right.items() if k not in ("source_sha256", "metrics", "status")})
    assert_withheld(compare(left, EvaluationResult(right)), "INCOMPATIBLE")


@pytest.mark.parametrize("damage", ["metric", "tiny_metric", "raw", "count", "batch", "producer", "checkpoint", "stream", "structure"])
def test_incoherent_json_rejected_with_rehashed_envelope(damage):
    data = produce(capture_structure=True).to_dict()
    if damage == "metric":
        data["metrics"]["accuracy"] = 0.5
    elif damage == "tiny_metric":
        data["metrics"]["accuracy"] -= 1e-15
    elif damage == "raw":
        data["raw"]["targets"][0] = 1
    elif damage == "count":
        data["accounting"]["expected_samples"] += 1
    elif damage == "batch":
        data["accounting"]["batch_sizes"] = [1, 1]
    elif damage == "producer":
        data["producer_evidence"]["model_state"]["status"] = "declared"
    elif damage == "checkpoint":
        data["producer_evidence"]["checkpoint"] = dict(status="native", sha256="a" * 64,
            model_state_sha256="b" * 64, source="trusted_checkpoint_load", reason=None)
    elif damage == "stream":
        data["producer_evidence"]["evaluated_stream"]["sample_count"] -= 1
    elif damage == "structure":
        data["structure_binding"]["evaluation_id"] = "20000000-0000-0000-0000-000000000000"
    data["source_sha256"] = _hash({k: v for k, v in data.items() if k not in ("source_sha256", "metrics", "status")})
    with pytest.raises(ValueError):
        EvaluationResult(data)


@pytest.mark.parametrize("schema", [1, 2, 3])
def test_schema_roundtrip_immutability_and_artifact_hash(tmp_path, schema):
    result = produce(capture_evidence=schema > 1, capture_structure=schema == 3)
    data = result.to_dict()
    assert data["schema_version"] == schema
    path = tmp_path / "evaluation.json"
    result.to_json(path)
    imported = EvaluationResult(json.loads(path.read_text(encoding="utf8")))
    assert imported.to_dict() == data
    assert (imported.structure_snapshot is not None) == (schema == 3)
    with pytest.raises(FrozenInstanceError):
        result._payload = "forged"
    detached = result.to_dict()
    detached["raw"]["targets"].clear()
    assert result.to_dict() == data
    EvidenceRecord.from_json(path, feature_id="P26", expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(EvidenceError):
        EvidenceRecord.from_json(path, feature_id="P26", expected_sha256="0" * 64)


def test_checkpoint_receipt_is_object_state_and_lifetime_bound(tmp_path):
    model = classifier()
    path = tmp_path / "checkpoint.pth"
    save_training_checkpoint(path, model, include_rng=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        load_training_checkpoint(path, model, expected_sha256="0" * 64, capture_evidence=True)
    load_training_checkpoint(path, model, expected_sha256=digest, capture_evidence=True)
    assert produce(model).to_dict()["producer_evidence"]["checkpoint"]["sha256"] == digest
    assert checkpoint_evidence(copy.deepcopy(model), model_hash(model))["status"] == "missing"
    from hyso._evaluation_evidence import _CHECKPOINTS
    identity = id(model)
    ref = weakref.ref(model)
    assert identity in _CHECKPOINTS
    with torch.no_grad():
        model.bias.add_(1)
    assert produce(model).to_dict()["producer_evidence"]["checkpoint"]["status"] == "missing"
    del model
    gc.collect()
    assert ref() is None and identity not in _CHECKPOINTS


@pytest.mark.parametrize("field", ["model_state_sha256", "checkpoint_sha256"])
def test_wrong_native_model_or_checkpoint_scope_is_rejected(tmp_path, field):
    model = classifier()
    path = tmp_path / "checkpoint.pth"
    save_training_checkpoint(path, model, include_rng=False)
    load_training_checkpoint(path, model, capture_evidence=True)
    result = produce(model)
    with pytest.raises(EvidenceError):
        source("A", result, **{field: "0" * 64})


def graph_builder(topology="add", activation="silu", classes=2):
    return (GraphModelBuilder({"image": [1, 4, 4]})
        .add("stem", "conv_norm_act", ["image"], in_channels=1, out_channels=2, norm="batch", activation=activation)
        .add("branch", "conv2d", ["stem"], in_channels=2, out_channels=2, kernel_size=1, padding=0)
        .add("merge", topology, ["stem", "branch"])
        .add("head", "classification_head", ["merge"], in_features=2, num_classes=classes)
        .outputs(logits="head"))


def graph_capture(tmp_path, *, receiver=True, context=None):
    from hyso.models.output import GraphTensorOutput
    model = GraphTensorOutput(graph_builder().build(), "logits", task="multiclass").eval()
    path = tmp_path / "graph.pth"
    save_training_checkpoint(path, model, graph_contract=True, include_rng=False)
    load_training_checkpoint(path, model, capture_evidence=True)
    data = DataLoader(TensorDataset(torch.arange(64).reshape(4, 1, 4, 4).float() / 64,
        torch.tensor([0, 1, 0, 1])), batch_size=2)
    result = evaluate_graph_classification(model, data, class_names=["a", "b"], split="test",
        context=context or dict(dataset_id="small-qa", preprocessing_id="identity", split="test"),
        capture_receiver_runtime=receiver)
    return model, result


@pytest.mark.parametrize("receiver", [False, True])
def test_graph_actual_receiver_default_and_process_loss(tmp_path, receiver):
    model, result = graph_capture(tmp_path, receiver=receiver)
    data = result.to_dict()
    assert data["verification"]["graph_v2"] == "verified"
    assert data["relation"]["status"] == ("verified" if receiver else "limited")
    assert data["evaluation"]["producer_evidence"]["checkpoint"]["status"] == "missing"
    path = tmp_path / "association.json"
    result.to_json(path)
    imported = GraphEvaluationResult.from_json(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert imported.to_dict()["verification"]["graph_v2"] == "unverified"
    assert imported.to_dict()["relation"]["status"] == "unavailable"
    if receiver:
        assert imported.to_dict()["verification"]["receiver_runtime"] == "unverified"
    del model
    gc.collect()
    assert result.to_dict()["relation"]["status"] == "unavailable"


def test_graph_json_in_second_interpreter_cannot_restore_authority(tmp_path):
    model, result = graph_capture(tmp_path)
    path = tmp_path / "association.json"
    result.to_json(path)
    import hyso
    package_root = str(Path(hyso.__file__).resolve().parent.parent)
    script = (
        "import sys,json;sys.path.insert(0,sys.argv[1]);"
        "from hyso.analysis.graph_evidence import GraphEvaluationResult;"
        "print(json.dumps(GraphEvaluationResult.from_json(sys.argv[2]).to_dict()['verification']))"
    )
    process = subprocess.run([sys.executable, "-I", "-B", "-c", script, package_root, str(path)],
        cwd=tmp_path, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"), capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    verification = json.loads(process.stdout.strip().splitlines()[-1])
    assert verification["graph_v2"] == verification["receiver_runtime"] == "unverified"
    assert verification["inference_binding"] == "unavailable"
    assert result.to_dict()["relation"]["status"] == "verified"


@pytest.mark.parametrize("field", ["dataset_id", "preprocessing_id"])
def test_graph_context_comparison_rejects_declared_mismatch(tmp_path, field):
    left_model, left = graph_capture(tmp_path)
    context = dict(dataset_id="small-qa", preprocessing_id="identity", split="test")
    context[field] = "different-declaration"
    right_dir = tmp_path / "right"
    right_dir.mkdir()
    right_model, right = graph_capture(right_dir, context=context)
    result = compare_graph_evaluations(left, right).to_dict()
    assert result["status"] == "INCOMPATIBLE"
    assert all(r["difference"] is None for r in result["metrics"])
    # The retained nested legacy projection keeps its historical narrow scope.
    assert all(r["status"] == "COMPARABLE" for r in result["legacy"]["metrics"])


def test_structure_equality_survives_weight_change_without_receiver_authority():
    left, right = classifier(), classifier(1.5)
    assert model_hash(left) != model_hash(right)
    structure = compare_model_structures(capture_model_structure(left), capture_model_structure(right))
    assert not structure.has_observed_changes
    assert produce(left).to_dict()["metrics"]["accuracy"] != produce(right).to_dict()["metrics"]["accuracy"]


def test_architecture_metadata_separates_weights_activation_head_and_topology():
    configs = [graph_builder().to_config(), graph_builder("multiply").to_config(),
        graph_builder(activation="relu").to_config(), graph_builder(classes=3).to_config()]
    columns = compare_architectures(configs, exact_parameters=True)["columns"]
    assert all(c["valid"] for c in columns)
    assert len({c["config_sha256"] for c in columns}) == 4
    assert columns[0]["parameter_count"] == columns[1]["parameter_count"] == columns[2]["parameter_count"]
    assert columns[0]["parameter_count"] != columns[3]["parameter_count"]
    builder = ModelBuilder.sequential((1, 4, 4), num_classes=2)
    repeated = compare_architectures([builder.to_config(), builder.to_config()])["columns"]
    assert repeated[0]["config_sha256"] == repeated[1]["config_sha256"]


def test_reporting_and_experiment_artifacts_in_fresh_qa_store(tmp_path):
    left, right = produce(classifier(1.5), capture_structure=True), produce(capture_structure=True)
    a, b = source("A", left), source("B", right)
    comparison = compare_models(a, b, features="P26", include_structure=True)
    from hyso.analysis import UnifiedAnalysisReport
    report = create_report(UnifiedAnalysisReport(a.to_dict()["report"]),
        UnifiedAnalysisReport(b.to_dict()["report"]), comparison, language="en", level="technical")
    paths = [tmp_path / ("report." + suffix) for suffix in ("json", "md", "html")]
    report.to_json(paths[0])
    report.to_markdown(paths[1])
    report.to_html(paths[2])
    restored = json.loads(paths[0].read_text(encoding="utf8"))
    assert restored == report.to_dict()
    payload = json.dumps(restored)
    assert "COMPARABLE" in payload and "test_accuracy" in payload
    for path in paths[1:]:
        text = path.read_text(encoding="utf8")
        assert "0.75" in text and "COMPARABLE" in text
    left.to_json(tmp_path / "evaluation.json")
    comparison.to_json(tmp_path / "comparison.json")
    manager = ExperimentManager(tmp_path / "store")
    manager.close()
    ExperimentManager.migrate_store(tmp_path / "store")
    manager = ExperimentManager(tmp_path / "store")
    try:
        experiment = manager.create_experiment("Stage 13.5.14 bounded QA")
        run = manager.create_run(experiment["id"], "evaluation", config=dict(dataset="small-qa"))
        run.start()
        for path in [*paths, tmp_path / "evaluation.json", tmp_path / "comparison.json"]:
            artifact = run.add_artifact(path, kind="report", ownership="copy")
            assert manager.verify_artifact(artifact["id"])["status"] == "verified"
        wrong = tmp_path / "wrong.json"
        wrong.write_text("{}", encoding="utf8")
        artifact = run.add_artifact(tmp_path / "comparison.json", kind="report", ownership="reference")
        assert manager.verify_artifact(artifact["id"], path=wrong)["status"] == "mismatch"
        assert manager.verify_artifact(artifact["id"], path=tmp_path / "comparison.json")["status"] == "verified"
        run.finish()
    finally:
        manager.close()


def test_graph_reporting_preserves_limited_and_imported_authority(tmp_path):
    model, live = graph_capture(tmp_path, receiver=False)
    path = tmp_path / "graph.json"
    live.to_json(path)
    imported = GraphEvaluationResult.from_json(path)
    report = create_report(live, imported, language="en", level="technical")
    sources = report.to_dict()["sources"]
    assert sources[0]["graph_evidence"]["cross_evidence_binding"] == "limited"
    assert sources[1]["graph_evidence"]["cross_evidence_binding"] == "unavailable"
    assert report.to_dict()["authoritative_evidence"] is False
    report.to_json(tmp_path / "graph-report.json")
    report.to_markdown(tmp_path / "graph-report.md")
    report.to_html(tmp_path / "graph-report.html")
    for suffix in ("md", "html"):
        text = (tmp_path / ("graph-report." + suffix)).read_text(encoding="utf8")
        assert "limited" in text and "unavailable" in text
