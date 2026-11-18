"""Strict declared-context policy, Graph export boundaries and legacy compatibility."""

import json

import pytest
import torch

from hyso.analysis import (
    AnalysisContext, EvidenceIndex, EvidenceRecord, EvidenceScope,
    ModelComparisonResult, ModelComparisonSource, compare_models,
)
from hyso.analysis._codec import digest
from hyso.analysis.graph_evidence import (
    GraphComparisonResult, GraphEvidenceError, compare_graph_evaluations,
    evaluate_graph_classification,
)
from hyso.analysis.model_comparison import ModelComparisonError
from hyso.reporting import create_report
from test_evaluation_comparison_acceptance import (
    bounded_cpu_work, classifier, graph_capture, produce,
)


CONTEXT = dict(dataset_id="small-qa", preprocessing_id="identity", split="test")


def source(model_id, evaluation, *, context=None, scope=None):
    declared = dict(CONTEXT if scope is None else scope)
    record = EvidenceRecord.from_native(evaluation.to_dict(), feature_id="P26",
        scope=EvidenceScope(model_id=model_id, **declared))
    report = EvidenceIndex([record]).analyze(context=AnalysisContext(
        workload_id="strict-context-qa", task="classification", model={"id": model_id},
        dataset=dict(CONTEXT if context is None else context)))
    return ModelComparisonSource(model_id, report, records={"P26": record.record_id})


def compare(left, right):
    return compare_models(left, right, features="P26", context_policy="strict")


def assert_quality_withheld(value, path=()):
    """Inspect nested comparison and report payloads, including numeric zero."""
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


@pytest.mark.parametrize("structure", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_strict_positive_reuses_native_math_for_weights_and_architecture(structure, wrapped):
    a = source("A", produce(classifier(1.5), capture_structure=structure))
    b = source("B", produce(classifier(wrapped=wrapped), capture_structure=structure))
    legacy = compare_models(a, b, features="P26", include_structure=structure).to_dict()
    strict = compare_models(a, b, features="P26", include_structure=structure, context_policy="strict")
    data = strict.to_dict()
    assert data["schema_version"] == 2
    assert data["context_policy"] == dict(id="strict_declared_context_v1", version=1,
        required_fields=["dataset_id", "split", "preprocessing_id"], authority="caller_declared_consistency")
    assert data["metrics"] == legacy["metrics"]
    assert data["metrics"][0]["status"] == "COMPARABLE"
    assert data["metrics"][0]["difference"]["difference"] == 0.25
    assert data["claims"]["global_winner"] is None
    assert ModelComparisonResult(data).to_dict() == data
    assert strict.tracking_config()["policy"].startswith("strict_declared_context_v1;")


@pytest.mark.parametrize("field", ["dataset_id", "preprocessing_id", "split"])
def test_known_context_mismatch_withholds_all_nested_quality_fields(field):
    context = dict(CONTEXT, **{field: "other"})
    evaluation = produce(split="other") if field == "split" else produce()
    a = source("A", produce(classifier(1.5)))
    b = source("B", evaluation, context=context, scope=context)
    data = compare(a, b).to_dict()
    assert all(row["status"] == "INCOMPATIBLE" for row in data["metrics"])
    assert_quality_withheld(data)


@pytest.mark.parametrize("field,side", [
    ("dataset_id", "both"), ("preprocessing_id", "both"),
    ("dataset_id", "candidate"), ("preprocessing_id", "candidate"),
])
def test_missing_declarations_are_insufficient(field, side):
    missing = dict(CONTEXT)
    missing.pop(field)
    left = missing if side == "both" else CONTEXT
    a = source("A", produce(classifier(1.5)), context=left, scope=left)
    b = source("B", produce(), context=missing, scope=missing)
    data = compare(a, b).to_dict()
    assert all(row["status"] == "INSUFFICIENT_EVIDENCE" for row in data["metrics"])
    assert all("missing_declared_condition:" + field in row["reasons"] for row in data["metrics"])
    assert_quality_withheld(data)


def test_known_mismatch_takes_precedence_over_missing_metadata():
    a_context = dict(dataset_id="left", split="test")
    b_context = dict(dataset_id="right", split="test")
    data = compare(source("A", produce(), context=a_context, scope=a_context),
        source("B", produce(), context=b_context, scope=b_context)).to_dict()
    assert all(row["status"] == "INCOMPATIBLE" for row in data["metrics"])
    assert_quality_withheld(data)


@pytest.mark.parametrize("field", ["dataset_id", "preprocessing_id", "split"])
def test_record_and_report_declaration_conflict_is_incompatible(field):
    a = source("A", produce())
    b = source("B", produce(), context=dict(CONTEXT, **{field: "conflicting"}))
    data = compare(a, b).to_dict()
    assert all(row["status"] == "INCOMPATIBLE" for row in data["metrics"])
    assert any("conflicting_declared_condition:candidate:" + field in row["reasons"] for row in data["metrics"])
    assert_quality_withheld(data)


@pytest.mark.parametrize("field", ["dataset_id", "preprocessing_id", "split"])
def test_invalid_report_context_values_are_rejected(field):
    a = source("A", produce())
    b = source("B", produce(), context=dict(CONTEXT, **{field: 1}))
    with pytest.raises(ModelComparisonError, match="invalid_strict_context_declaration"):
        compare(a, b)


@pytest.mark.parametrize("change", ["input", "target", "order", "classes", "vocabulary", "task", "count", "batch"])
def test_equal_declarations_preserve_native_rejection(change):
    x = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    y = torch.tensor([0, 0, 1, 1])
    options = {}
    model = classifier()
    if change == "input":
        x[0] -= 1
    elif change == "target":
        y[0] = 1
    elif change == "order":
        x, y = x.flip(0), y.flip(0)
    elif change == "classes":
        options["class_names"] = ["b", "a"]
    elif change == "vocabulary":
        options["class_names"] = ["a", "other"]
    elif change == "task":
        model = classifier(binary=True)
        options["task"] = "binary"
    elif change == "count":
        x, y = x[:3], y[:3]
    elif change == "batch":
        options["batch"] = 3
    left, right = produce(), produce(model, x=x, y=y, **options)
    data = compare(source("A", left), source("B", right)).to_dict()
    assert all(row["status"] == "INCOMPATIBLE" for row in data["metrics"])
    assert_quality_withheld(data)
    if change == "batch":
        assert left.to_dict()["producer_evidence"]["evaluated_stream"] == right.to_dict()["producer_evidence"]["evaluated_stream"]


def test_missing_native_producer_stays_insufficient_with_matching_context():
    data = compare(source("A", produce()), source("B", produce(capture_evidence=False))).to_dict()
    assert all(row["status"] == "INSUFFICIENT_EVIDENCE" for row in data["metrics"])
    assert_quality_withheld(data)


@pytest.mark.parametrize("policy", [None, True, "STRICT", "", 1])
def test_invalid_policy_is_rejected(policy):
    with pytest.raises(ModelComparisonError, match="unsupported_context_policy"):
        compare_models(source("A", produce()), source("B", produce()), features="P26", context_policy=policy)


@pytest.mark.parametrize("field", ["dataset_id", "preprocessing_id"])
def test_tampered_context_json_is_rejected_even_after_outer_rehash(field):
    data = compare(source("A", produce()), source("B", produce())).to_dict()
    data["sources"]["candidate"]["report"]["context"]["dataset"][field] = "changed-in-json"
    data["content_sha256"] = digest({key: value for key, value in data.items() if key != "content_sha256"})
    with pytest.raises(ModelComparisonError):
        ModelComparisonResult(data)


@pytest.mark.parametrize("damage", ["version", "policy", "difference", "native_comparison"])
def test_rehashed_inconsistent_strict_envelope_is_rejected(damage):
    a = source("A", produce(classifier(1.5)))
    b = source("B", produce(), context=dict(CONTEXT, dataset_id="other"), scope=dict(CONTEXT, dataset_id="other"))
    data = compare(a, b).to_dict()
    if damage == "version":
        data["schema_version"] = 1
    elif damage == "policy":
        data["context_policy"]["version"] = 2
    elif damage == "difference":
        data["metrics"][0]["difference"] = 0.25
    else:
        data["metrics"][0]["comparison"] = compare_models(a, b, features="P26").to_dict()["metrics"][0]["comparison"]
    data["content_sha256"] = digest({key: value for key, value in data.items() if key != "content_sha256"})
    with pytest.raises(ModelComparisonError):
        ModelComparisonResult(data)


def graph_pair(tmp_path, context=None, *, receiver=True):
    left_dir, right_dir = tmp_path / "left", tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    left_model, left = graph_capture(left_dir, receiver=receiver)
    right_model, right = graph_capture(right_dir, context=context, receiver=receiver)
    return (left_model, right_model), (left, right)


@pytest.mark.parametrize("receiver", [False, True])
def test_strict_graph_positive_and_imported_authority(tmp_path, receiver):
    models, pair = graph_pair(tmp_path, receiver=receiver)
    legacy = compare_graph_evaluations(*pair)
    strict = compare_graph_evaluations(*pair, context_policy="strict")
    data = strict.to_dict()
    assert data["schema_version"] == 3 and data["status"] == "COMPARABLE"
    assert strict.effective_comparison.to_dict()["schema_version"] == 2
    assert strict.legacy_comparison is None
    assert data["legacy"]["comparison_sha256"] == legacy.legacy_comparison.to_dict()["content_sha256"]
    assert len(data["legacy"]["metrics"]) == 2
    assert all(row["status"] == "NOT_EXPORTED" and row["difference"] is None for row in data["legacy"]["metrics"])
    assert data["metrics"] == legacy.to_dict()["metrics"]
    assert all(view["graph_v2"] == "verified" for view in data["verification"].values())
    if receiver:
        assert all(view["inference_binding"] == "verified" for view in data["verification"].values())
    else:
        assert all(association.to_dict()["relation"]["status"] == "limited" for association in pair)
    path = tmp_path / "strict.json"
    strict.to_json(path)
    imported = GraphComparisonResult.from_json(path)
    assert imported.to_dict()["schema_version"] == 3
    assert all(view["graph_v2"] == "unverified" for view in imported.to_dict()["verification"].values())
    assert all(a.to_dict()["evaluation"]["producer_evidence"]["checkpoint"]["status"] == "missing" for a in pair)


@pytest.mark.parametrize("field", ["dataset_id", "preprocessing_id"])
def test_graph_strict_recursive_export_and_reporting_reject_numeric_differences(tmp_path, field):
    models, pair = graph_pair(tmp_path, context=dict(CONTEXT, **{field: "different"}))
    legacy = compare_graph_evaluations(*pair)
    # Demonstrate the recursive assertion detects the actual old disclosure.
    with pytest.raises(AssertionError):
        assert_quality_withheld(legacy.to_dict())
    legacy.to_json(tmp_path / "historical-legacy.json")
    strict = compare_graph_evaluations(*pair, context_policy="strict")
    data = strict.to_dict()
    assert data["status"] == "INCOMPATIBLE"
    assert_quality_withheld(data)
    assert data["legacy"]["comparison_sha256"] == legacy.legacy_comparison.to_dict()["content_sha256"]
    assert len(data["legacy"]["metrics"]) == 2
    assert_quality_withheld(strict.effective_comparison.to_dict())
    report = create_report(strict, language="en", level="technical")
    payload = report.to_dict()
    assert_quality_withheld(payload)
    assert payload["sources"][0]["charts"] == []
    for suffix, method in (("json", report.to_json), ("md", report.to_markdown), ("html", report.to_html)):
        path = tmp_path / ("report." + suffix)
        method(path)
        text = path.read_text(encoding="utf8")
        assert "INCOMPATIBLE" in text
    assert_quality_withheld(json.loads((tmp_path / "report.json").read_text(encoding="utf8")))


@pytest.mark.parametrize("field", ["dataset_id", "preprocessing_id"])
def test_graph_missing_metadata_is_insufficient_without_numeric_disclosure(tmp_path, field):
    context = dict(CONTEXT, **{field: None})
    models, pair = graph_pair(tmp_path, context=context)
    result = compare_graph_evaluations(*pair, context_policy="strict").to_dict()
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert_quality_withheld(result)


@pytest.mark.parametrize("change", ["input", "order", "classes", "batch"])
def test_graph_strict_context_preserves_native_stream_and_protocol_gates(tmp_path, change):
    from torch.utils.data import DataLoader, TensorDataset
    models, pair = graph_pair(tmp_path)
    x = torch.arange(64).reshape(4, 1, 4, 4).float() / 64
    y = torch.tensor([0, 1, 0, 1])
    if change == "input":
        x += 1
    elif change == "order":
        x, y = x.flip(0), y.flip(0)
    classes = ["b", "a"] if change == "classes" else ["a", "b"]
    batch = 3 if change == "batch" else 2
    right = evaluate_graph_classification(models[1], DataLoader(TensorDataset(x, y), batch_size=batch),
        class_names=classes, split="test", context=CONTEXT, capture_receiver_runtime=True)
    result = compare_graph_evaluations(pair[0], right, context_policy="strict").to_dict()
    assert result["status"] == "INCOMPATIBLE"
    assert_quality_withheld(result)


@pytest.mark.parametrize("damage", ["context", "policy", "reference", "legacy_payload", "effective", "metric"])
def test_rehashed_graph_inconsistency_is_rejected(tmp_path, damage):
    models, pair = graph_pair(tmp_path, context=dict(CONTEXT, dataset_id="different"))
    legacy = compare_graph_evaluations(*pair)
    data = compare_graph_evaluations(*pair, context_policy="strict").to_dict()
    if damage == "context":
        association = data["sources"]["candidate"]
        association["context"]["dataset_id"] = "small-qa"
        association["content_sha256"] = digest({k: v for k, v in association.items() if k != "content_sha256"})
    elif damage == "policy":
        data["context_policy"]["version"] = 2
    elif damage == "reference":
        data["legacy"]["comparison_sha256"] = "0" * 64
    elif damage == "legacy_payload":
        data["legacy"] = legacy.legacy_comparison.to_dict()
    elif damage == "effective":
        data["effective"] = legacy.legacy_comparison.to_dict()
    else:
        data["metrics"][0]["difference"] = 0.0
    data.pop("verification", None)
    data["content_sha256"] = digest({key: value for key, value in data.items() if key != "content_sha256"})
    with pytest.raises(GraphEvidenceError):
        GraphComparisonResult(data)


def test_strict_generic_reporting_with_structure_keeps_registered_counts(tmp_path):
    a = source("A", produce(classifier(1.5), capture_structure=True))
    b = source("B", produce(classifier(wrapped=True), capture_structure=True),
        context=dict(CONTEXT, dataset_id="other"), scope=dict(CONTEXT, dataset_id="other"))
    comparison = compare_models(a, b, features="P26", include_structure=True, context_policy="strict")
    assert_quality_withheld(comparison.to_dict())
    assert comparison.structure_comparison is not None
    report = create_report(comparison, language="en")
    assert_quality_withheld(report.to_dict())
    report.to_json(tmp_path / "structured-report.json")
    assert any(m["domain"] == "registered_structure" for m in report.to_dict()["sources"][0]["metrics"])


@pytest.mark.parametrize("compatible", [False, True])
def test_strict_typed_result_roundtrips_through_fresh_experiment_store(tmp_path, compatible):
    from hyso import ExperimentManager
    context = dict(CONTEXT, dataset_id="small-qa" if compatible else "other")
    result = compare(source("A", produce(classifier(1.5))),
        source("B", produce(), context=context, scope=context))
    root = tmp_path / "store"
    manager = ExperimentManager(root)
    manager.close()
    ExperimentManager.migrate_store(root)
    manager = ExperimentManager(root)
    try:
        experiment = manager.create_experiment("Strict comparison QA")
        run = manager.create_workflow_run(experiment["id"], "comparison", model_ids=["A", "B"], comparison=result)
        run.start()
        artifact = run.record_result(result)
        assert run.record_result(result)["id"] == artifact["id"]
        run.finish()
        loaded = manager.open_result(artifact["id"])
        assert loaded["status"] == "verified"
        assert loaded["result"].to_json() == result.to_json()
        assert loaded["result"].to_dict()["context_policy"]["version"] == 1
        if not compatible:
            assert_quality_withheld(loaded["result"].to_dict())
    finally:
        manager.close()
