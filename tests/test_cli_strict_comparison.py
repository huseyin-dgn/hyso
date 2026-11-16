"""CLI policy selection over retained typed P26 evidence; no live workloads."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from hyso import ModelBuilder, GraphModelBuilder
from hyso.analysis import (
    AnalysisContext, EvidenceIndex, EvidenceRecord, EvidenceScope,
    ModelComparisonSource, UnifiedAnalysisReport, compare_models,
)
from hyso.cli import main
from hyso.cli.main import build_parser

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = dict(dataset_id="retained-cli-qa", split="test", preprocessing_id="identity")


@pytest.fixture(autouse=True)
def no_workloads(monkeypatch):
    from hyso import benchmark, evaluation, training
    def forbidden(*args, **kwargs):
        pytest.fail("retained CLI comparison attempted model execution, CUDA or checkpoint loading")
    for obj, name in ((torch.nn.Module, "_call_impl"), (torch, "load"),
                      (ModelBuilder, "build"), (GraphModelBuilder, "build"),
                      (benchmark.BenchmarkRunner, "run"), (evaluation, "evaluate_classification"),
                      (training.Trainer, "fit"), (training.TrainingSession, "fit")):
        monkeypatch.setattr(obj, name, forbidden)
    for name in ("init", "_lazy_init", "is_available", "device_count", "current_device",
                 "synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def retained_source(model_id, *, context=None, scope=None):
    payload = json.loads((ROOT / "tests/fixtures/model-comparison-bound-v1.json").read_text(encoding="utf8"))["payloads"][model_id]
    declared = dict(CONTEXT if context is None else context)
    scope = dict(declared if scope is None else scope)
    record = EvidenceRecord.from_native(payload, feature_id="P26",
        scope=EvidenceScope(model_id=model_id, **scope))
    report = EvidenceIndex([record]).analyze(context=AnalysisContext(
        workload_id="retained-cli-policy", task="classification",
        model={"id": model_id}, dataset=declared))
    return ModelComparisonSource(model_id, report)


def saved_pair(tmp_path, *, context=None, scope=None, left_context=None):
    a = retained_source("A", context=left_context)
    b = retained_source("B", context=context, scope=scope)
    paths = [tmp_path / "baseline.json", tmp_path / "candidate.json"]
    for path, source in zip(paths, (a, b)):
        UnifiedAnalysisReport(source.to_dict()["report"]).to_json(path)
    return a, b, paths


def arguments(paths, *extra):
    return ["compare", *map(str, paths), "--baseline-model", "A", "--candidate-model", "B",
            "--features", "P26", *extra]


def invoke(capsys, args):
    code = main([*args, "--json"])
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope["exit_code"] == code
    return code, envelope, captured.err


def assert_quality_withheld(value, path=()):
    if "structure" in path:
        return
    if isinstance(value, dict):
        if value.get("domain") == "registered_structure":
            return
        for key, item in value.items():
            if key in ("difference", "percent_change", "percentage_point_change", "metric_effect", "global_winner"):
                assert item is None, (path + (key,), item)
            assert_quality_withheld(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_quality_withheld(item, path + (index,))


@pytest.mark.parametrize("flags,expected", [([], "legacy"), (["--context-policy", "legacy"], "legacy"),
                                           (["--context-policy", "strict"], "strict")])
def test_parser_policy_selection(flags, expected):
    args = build_parser().parse_args(arguments(["left.json", "right.json"], *flags))
    assert args.context_policy == expected


def test_compare_help_exposes_policy_without_building(capsys):
    assert main(["compare", "--help"]) == 0
    assert "--context-policy {legacy,strict}" in capsys.readouterr().out


@pytest.mark.parametrize("policy", ["unknown", "STRICT"])
def test_invalid_policy_rejected_before_reading_sources(capsys, monkeypatch, policy):
    import hyso.cli.presentation as command
    monkeypatch.setattr(command, "read_result", lambda *a: pytest.fail("invalid policy reached source reader"))
    code, envelope, error = invoke(capsys, arguments(["left.json", "right.json"], "--context-policy", policy))
    assert code == 2 and envelope["status"] == "error"
    assert "invalid choice" in error and policy in error


@pytest.mark.parametrize("policy", ["legacy", "strict"])
def test_cli_delegates_exact_sources_and_policy(capsys, tmp_path, monkeypatch, policy):
    import hyso.analysis as api
    a, b, paths = saved_pair(tmp_path)
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    calls = []
    native = api.compare_models
    def observe(left, right, **kwargs):
        calls.append((left.to_dict(), right.to_dict(), kwargs))
        return native(left, right, **kwargs)
    monkeypatch.setattr(api, "compare_models", observe)
    code, envelope, _ = invoke(capsys, arguments(paths, "--context-policy", policy))
    assert code == 0 and len(calls) == 1
    assert calls[0] == (a.to_dict(), b.to_dict(), dict(features=["P26"], context_policy=policy))
    assert envelope["result"] == native(a, b, features="P26", context_policy=policy).to_dict()
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]


@pytest.mark.parametrize("format", ["json", "markdown"])
def test_strict_compatible_reuses_native_result(capsys, tmp_path, format):
    a, b, paths = saved_pair(tmp_path)
    output = tmp_path / ("strict." + format)
    code, envelope, _ = invoke(capsys, arguments(paths, "--context-policy", "strict",
        "--format", format, "--language", "en", "--output", str(output)))
    expected = compare_models(a, b, features="P26", context_policy="strict")
    assert code == 0 and envelope["result"] == expected.to_dict()
    assert envelope["result"]["schema_version"] == 2
    assert all(row["status"] == "COMPARABLE" for row in envelope["result"]["metrics"])
    assert envelope["result"]["metrics"][0]["difference"]["difference"] == .25
    assert envelope["result"]["claims"]["global_winner"] is None
    text = expected.to_json() if format == "json" else expected.to_markdown(language="en")
    assert output.read_bytes() == text.encode("utf8")


@pytest.mark.parametrize("field", ["dataset_id", "split", "preprocessing_id"])
@pytest.mark.parametrize("format", ["json", "markdown"])
def test_strict_mismatch_withholds_nested_quality(capsys, tmp_path, field, format):
    context = dict(CONTEXT, **{field: "other"})
    _, _, paths = saved_pair(tmp_path, context=context, scope=CONTEXT)
    output = tmp_path / ("incompatible." + format)
    code, envelope, _ = invoke(capsys, arguments(paths, "--context-policy", "strict",
        "--format", format, "--output", str(output)))
    assert code == 0  # Valid comparison outcome; not a CLI invocation error.
    assert all(row["status"] == "INCOMPATIBLE" for row in envelope["result"]["metrics"])
    assert_quality_withheld(envelope["result"])
    assert "INCOMPATIBLE" in output.read_text(encoding="utf8")


@pytest.mark.parametrize("field", ["dataset_id", "preprocessing_id"])
@pytest.mark.parametrize("side", ["candidate", "both"])
def test_missing_context_stays_insufficient(capsys, tmp_path, field, side):
    missing = dict(CONTEXT)
    missing.pop(field)
    _, _, paths = saved_pair(tmp_path, context=missing,
        left_context=missing if side == "both" else None)
    code, envelope, _ = invoke(capsys, arguments(paths, "--context-policy", "strict"))
    assert code == 0
    assert all(row["status"] == "INSUFFICIENT_EVIDENCE" for row in envelope["result"]["metrics"])
    assert all("missing_declared_condition:" + field in row["reasons"] for row in envelope["result"]["metrics"])
    assert_quality_withheld(envelope["result"])
    records = envelope["result"]["sources"]["candidate"]["report"]["index"]["records"]
    assert all(record["scope"][field] is None for record in records)


def test_strict_without_p26_retains_native_validation(capsys, tmp_path):
    _, _, paths = saved_pair(tmp_path)
    args = arguments(paths, "--context-policy", "strict")
    args[args.index("P26")] = "P01"
    code, envelope, error = invoke(capsys, args)
    assert code == 2 and envelope["status"] == "error"
    assert "strict_context_requires_P26" in error


@pytest.mark.parametrize("format", ["json", "markdown"])
def test_default_and_explicit_legacy_preserve_bytes(capsys, tmp_path, format):
    a, b, paths = saved_pair(tmp_path, context=dict(CONTEXT, dataset_id="other"))
    native = compare_models(a, b, features="P26")
    expected = native.to_json() if format == "json" else native.to_markdown(language="tr")
    for policy, flags in (("default", []), ("legacy", ["--context-policy", "legacy"])):
        output = tmp_path / (policy + "." + format)
        code, envelope, _ = invoke(capsys, arguments(paths, *flags, "--format", format,
            "--language", "tr", "--output", str(output)))
        assert code == 0 and envelope["result"] == native.to_dict()
        assert envelope["result"]["schema_version"] == 1
        assert output.read_bytes() == expected.encode("utf8")


@pytest.mark.parametrize("format", ["json", "markdown", "html"])
@pytest.mark.parametrize("language", ["tr", "en"])
def test_saved_strict_render_keeps_source_and_nulls(capsys, tmp_path, format, language):
    _, _, paths = saved_pair(tmp_path, context=dict(CONTEXT, dataset_id="other"))
    saved = tmp_path / "strict.json"
    assert invoke(capsys, arguments(paths, "--context-policy", "strict", "--output", str(saved)))[0] == 0
    raw = json.loads(saved.read_text(encoding="utf8"))
    output = tmp_path / ("rendered." + format)
    code, envelope, _ = invoke(capsys, ["reports", "render", str(saved), "--format", format,
        "--language", language, "--level", "expert", "--output", str(output)])
    assert code == 0
    assert envelope["result"]["sources"][0]["raw"] == raw
    assert envelope["result"]["presentation_only"] and not envelope["result"]["authoritative_evidence"]
    assert_quality_withheld(envelope["result"])
    text = output.read_text(encoding="utf8")
    assert "INCOMPATIBLE" in text and "strict_declared_context_v1" in text
