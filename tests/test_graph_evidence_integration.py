"""Opt-in associations never transfer a source receipt to the isolated receiver."""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from hyso.checkpoint import save_training_checkpoint, load_training_checkpoint
from hyso.analysis._codec import digest
from hyso.analysis.graph_evidence import (
    GraphEvaluationResult, GraphEvidenceError, evaluate_graph_classification,
    compare_graph_evaluations,
)
from hyso.reporting import create_report
from test_graph_checkpoint_contract import model


def loader(*, shift=0, batch=2, dtype=torch.float32):
    x = torch.arange(64, dtype=dtype).reshape(4, 1, 4, 4) / 64 + shift
    return DataLoader(TensorDataset(x, torch.tensor([0, 1, 0, 1])), batch_size=batch)


def capture(tmp_path, name="a", *, trusted=True, version=2, data=None, context=None):
    m = model().eval()
    if trusted:
        path = tmp_path / (name + ".pth")
        save_training_checkpoint(path, m, graph_contract=version == 2, include_rng=False)
        load_training_checkpoint(path, m, capture_evidence=True)
    result = evaluate_graph_classification(m, loader() if data is None else data,
        class_names=["a", "b"], split="test", source_id="fixture",
        context=context or dict(dataset_id="fixture", split="test", preprocessing_id="identity"))
    return m, result


def test_native_source_and_isolated_inference_have_separate_scopes(tmp_path):
    m, result = capture(tmp_path)
    data = result.to_dict()
    assert data["verification"]["graph_v2"] == "verified"
    assert data["evaluation"]["producer_evidence"]["checkpoint"]["status"] == "missing"
    assert data["relation"]["status"] == "limited"
    assert data["relation"]["isolated_receiver_receipt"] == "not_transferred"
    assert data["relation"]["runtime_equivalence"] == "not_observed"
    assert data["evaluation"]["producer_evidence"]["evaluated_stream"]["sample_count"] == 4


@pytest.mark.parametrize("version,trusted", [(1, True), (2, False)])
def test_missing_or_v1_source_still_evaluates_without_trust(tmp_path, version, trusted):
    m, result = capture(tmp_path, version=version, trusted=trusted)
    assert result.to_dict()["verification"]["graph_v2"] == "missing"
    assert result.to_dict()["relation"]["status"] == "unavailable"


@pytest.mark.parametrize("expected", ["expected_checkpoint_sha256", "expected_model_state_sha256"])
def test_wrong_expected_identity_rejected(tmp_path, expected):
    m, result = capture(tmp_path)
    with pytest.raises(GraphEvidenceError):
        evaluate_graph_classification(m, loader(), class_names=["a", "b"], split="test",
            **{expected: "0" * 64})


@pytest.mark.parametrize("change", ["topology", "activation", "output", "geometry", "dtype", "device", "parameter", "buffer"])
def test_source_runtime_or_tensor_mutation_never_keeps_verified(tmp_path, change):
    m, result = capture(tmp_path)
    if change == "topology":
        c = m.graph.to_config()
        next(row for row in c["nodes"] if row["id"] == "merge")["component"] = "multiply"
        m.graph._architecture = json.dumps(c, sort_keys=True)
    elif change == "activation":
        m.graph.nodes["node_stem"].activation = torch.nn.ReLU()
    elif change == "output":
        m.output_name = "features"
    elif change == "geometry":
        c = m.graph.to_config()
        c["inputs"]["image"]["shape"] = [1, 5, 5]
        m.graph._architecture = json.dumps(c, sort_keys=True)
    elif change == "dtype":
        m.double()
    elif change == "device":
        m.graph._contract = torch.empty(0, device="meta")
    elif change == "parameter":
        with torch.no_grad():
            next(m.parameters()).add_(1)
    else:
        next(v for n, v in m.named_buffers() if n.endswith("running_mean")).add_(1)
    assert result.to_dict()["verification"]["graph_v2"] == "missing"
    assert result.to_dict()["relation"]["status"] == "unavailable"


def test_deepcopy_other_object_and_json_cannot_recreate_trust(tmp_path):
    m, result = capture(tmp_path)
    assert copy.deepcopy(result).to_dict()["verification"]["graph_v2"] == "unverified"
    other = copy.deepcopy(m)
    run = evaluate_graph_classification(other, loader(), class_names=["a", "b"], split="test")
    assert run.to_dict()["verification"]["graph_v2"] == "missing"
    data = result.to_dict()
    data["verification"]["graph_v2"] = "verified"
    imported = GraphEvaluationResult(data)
    assert imported.to_dict()["verification"]["graph_v2"] == "unverified"
    report = create_report(imported, language="en").to_dict()
    assert report["sources"][0]["graph_evidence"]["graph_v2"] == "unverified"


@pytest.mark.parametrize("field,value", [("dataset_id", "wrong"), ("split", "train")])
def test_wrong_declared_association_is_rejected(tmp_path, field, value):
    m = model()
    with pytest.raises(GraphEvidenceError):
        evaluate_graph_classification(m, loader(), class_names=["a", "b"], split="test",
            source_id="fixture", context={"dataset_id": "fixture", "split": "test", field: value})


@pytest.mark.parametrize("condition", ["stream", "batch", "preprocessing", "dataset"])
def test_verified_sources_do_not_override_incompatible_measurements(tmp_path, condition):
    a, left = capture(tmp_path, "a")
    context = dict(dataset_id="fixture", split="test", preprocessing_id="other" if condition == "preprocessing" else "identity")
    b, right = capture(tmp_path, "b", data=loader(shift=1 if condition == "stream" else 0,
        batch=4 if condition == "batch" else 2), context=context)
    if condition == "dataset":
        right = evaluate_graph_classification(b, loader(), class_names=["a", "b"], split="test",
            source_id="other", context=dict(dataset_id="other", split="test", preprocessing_id="identity"))
    result = compare_graph_evaluations(left, right).to_dict()
    assert result["status"] == "INCOMPATIBLE"
    assert all(row["difference"] is None for row in result["metrics"])


def test_compatible_pair_and_one_missing_preserve_legacy_comparison(tmp_path):
    a, left = capture(tmp_path, "a")
    b, right = capture(tmp_path, "b", trusted=False)
    result = compare_graph_evaluations(left, right)
    data = result.to_dict()
    assert data["status"] == "COMPARABLE"
    assert all(row["difference"] is not None for row in data["metrics"])
    assert data["verification"]["baseline"]["graph_v2"] == "verified"
    assert data["verification"]["candidate"]["graph_v2"] == "missing"
    c, third = capture(tmp_path, "c")
    both = compare_graph_evaluations(left, third)
    assert both.to_dict()["status"] == "COMPARABLE"
    for language in ["en", "tr"]:
        report = create_report(left, both, language=language)
        assert "limited" in report.to_json()
        assert "missing" in report.to_markdown()
        assert "not_transferred" in report.to_html()
    assert data["legacy"]["claims"]["global_winner"] is None


@pytest.mark.parametrize("problem", ["missing", "bad_version", "bad_evaluation", "bad_metadata"])
def test_broken_report_metadata_is_rejected(tmp_path, problem):
    m, result = capture(tmp_path)
    data = result.to_dict()
    if problem == "missing":
        del data["context"]
    elif problem == "bad_version":
        data["schema_version"] = True
    elif problem == "bad_evaluation":
        data["evaluation"]["source"]["model_state_sha256"] = "0" * 64
    else:
        data["context"] = "verified"
    with pytest.raises((GraphEvidenceError, ValueError)):
        GraphEvaluationResult(data)


def test_experiment_artifact_integrity_does_not_create_native_receipt(tmp_path):
    from hyso import ExperimentManager
    m, result = capture(tmp_path)
    store = tmp_path / "experiments"
    manager = ExperimentManager(store)
    manager.close()
    ExperimentManager.migrate_store(store)
    manager = ExperimentManager(store)
    try:
        experiment = manager.create_experiment("graph evidence")
        run = manager.create_run(experiment["id"], "evaluation", config={"scope": "retained_artifact_bytes"})
        run.start()
        path = tmp_path / "association.json"
        result.to_json(path)
        artifact = run.add_artifact(path, kind="graph_evaluation_association", ownership="reference")
        assert manager.verify_artifact(artifact["id"], path=path)["status"] == "verified"
        assert GraphEvaluationResult.from_json(path).to_dict()["verification"]["graph_v2"] == "unverified"
        path.write_text(path.read_text() + " ")
        assert manager.verify_artifact(artifact["id"], path=path)["status"] == "mismatch"
        run.finish()
    finally:
        manager.close()


def legacy_golden():
    from hyso.analysis import EvidenceRecord, EvidenceScope, EvidenceIndex, AnalysisContext, ModelComparisonSource, compare_models
    from hyso.training import TrainingResult
    class Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 14, 0, 0, tzinfo=timezone.utc)
    root = Path(__file__).resolve().parents[1]
    payloads = json.loads((root / "tests/fixtures/model-comparison-bound-v1.json").read_text())["payloads"]
    sources = []
    with patch("hyso.analysis.evidence.datetime", Fixed):
        for key in ["A", "B"]:
            record = EvidenceRecord.from_native(payloads[key], feature_id="P26", scope=EvidenceScope(model_id=key))
            report = EvidenceIndex([record]).analyze(context=AnalysisContext(workload_id="legacy-golden",
                task="classification", model={"id": key}, dataset={}))
            sources.append(ModelComparisonSource(key, report))
        comparison = compare_models(*sources, features="P26")
        training = TrainingResult((root / "examples/reporting/fixtures/training-result-v1.json").read_text())
        data = dict(comparison=comparison.to_dict(), report=create_report(training, comparison,
            language="en", level="technical").to_dict())
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def test_legacy_comparison_and_reporting_golden():
    assert legacy_golden() == "eb66ee58d77155529d83db358cb5e0567fa8882c44ce9d7d9ee10e6664cc7b6e"


def test_forged_snapshot_and_comparison_json_remain_unverified(tmp_path):
    from hyso.analysis.graph_evidence import GraphComparisonResult
    a, left = capture(tmp_path, "a")
    b, right = capture(tmp_path, "b")
    data = left.to_dict()
    data["source_receipt"]["checkpoint_sha256"] = "f" * 64
    data["content_sha256"] = digest({k: v for k, v in data.items()
        if k not in ("content_sha256", "verification", "relation")})
    assert GraphEvaluationResult(data).to_dict()["verification"]["graph_v2"] == "unverified"
    comparison = compare_graph_evaluations(left, right)
    imported = GraphComparisonResult(comparison.to_dict()).to_dict()
    assert all(v["graph_v2"] == "unverified" for v in imported["verification"].values())


def test_missing_preprocessing_limits_only_opt_in_not_legacy(tmp_path):
    a, left = capture(tmp_path, "a", context=dict(dataset_id="fixture", split="test"))
    b, right = capture(tmp_path, "b", context=dict(dataset_id="fixture", split="test"))
    data = compare_graph_evaluations(left, right).to_dict()
    assert data["status"] == "INSUFFICIENT_EVIDENCE"
    assert all(r["difference"] is None for r in data["metrics"])
    assert all(r["status"] == "COMPARABLE" for r in data["legacy"]["metrics"])


def test_new_interpreter_cannot_restore_native_authority(tmp_path):
    import os
    import subprocess
    import sys
    import hyso
    m, result = capture(tmp_path)
    path = tmp_path / "association.json"
    result.to_json(path)
    code = ("import sys,json;sys.path.insert(0,sys.argv[2]);"
        "from hyso.analysis.graph_evidence import GraphEvaluationResult;"
        "print(json.dumps(GraphEvaluationResult.from_json(sys.argv[1]).to_dict()['verification']))")
    child = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(path),
        str(Path(hyso.__file__).resolve().parent.parent)], cwd=tmp_path, capture_output=True,
        text=True, timeout=60, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1"))
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout)["graph_v2"] == "unverified"


def test_existing_evaluator_called_once_comparison_reporting_do_not_execute(tmp_path, monkeypatch):
    import hyso.analysis.graph_evidence as api
    actual, calls = api.evaluate_classification, []
    def counted(*args, **kwargs):
        calls.append(1)
        return actual(*args, **kwargs)
    monkeypatch.setattr(api, "evaluate_classification", counted)
    a, left = capture(tmp_path, "a")
    b, right = capture(tmp_path, "b")
    assert len(calls) == 2
    def forbidden(*args, **kwargs):
        raise AssertionError("presentation/comparison executed inference or checkpoint load")
    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    comparison = compare_graph_evaluations(left, right)
    assert create_report(left, comparison, language="en").to_dict()["presentation_only"] is True
