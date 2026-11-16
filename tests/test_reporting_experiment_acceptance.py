"""Stage 13.5.15 public Reporting and ExperimentManager acceptance."""

import json
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
import sys
import uuid

import pytest
import torch

from hyso import ExperimentManager
from hyso.analysis import compare_models
from hyso.analysis._codec import digest
from hyso.analysis.graph_evidence import GraphEvaluationResult, compare_graph_evaluations
from hyso.cli import main as cli_main
from hyso.experiments import ExperimentError
from hyso.reporting import PresentationReport, ReportError, create_report, report_experiment
from examples.reporting.reporting_experiment_acceptance import (
    CONTEXT, analysis_source, evaluation, graph_evaluation, graph_model, linear, run, saved_training,
)
from test_evaluation_comparison_acceptance import bounded_cpu_work
from test_strict_comparison_context import assert_quality_withheld


class Document(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.pre = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        assert tag not in ("script", "iframe", "object", "embed", "link")
        for key, value in attrs:
            assert not key.startswith("on")
            if key in ("href", "src"):
                assert value.startswith("#")
        if tag == "pre":
            self.current = []

    def handle_data(self, data):
        if self.current is not None:
            self.current.append(data)

    def handle_endtag(self, tag):
        if tag == "pre" and self.current is not None:
            self.pre.append("".join(self.current))
            self.current = None


def comparison(*, incompatible=False, empty=False, policy="strict"):
    left = evaluation(linear(1.5), empty=empty)
    right = evaluation(linear(wrapped=True), empty=empty)
    a, ra = analysis_source("A", left)
    b, rb = analysis_source("B", right, context=dict(CONTEXT, dataset_id="other") if incompatible else None)
    return compare_models(a, b, features="P26", context_policy=policy), (left, right), (ra, rb)


def manager_at(root):
    manager = ExperimentManager(root)
    manager.close()
    ExperimentManager.migrate_store(root)
    return ExperimentManager(root)


@pytest.fixture
def manager(tmp_path):
    current = manager_at(tmp_path / "store")
    yield current
    current.close()


def artifact_run(manager):
    result = manager.create_run(manager.create_experiment("QA")["id"], "evaluation", {})
    result.start()
    return result


def invoke(capsys, *arguments):
    code = cli_main([*map(str, arguments), "--json"])
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope["exit_code"] == code
    return code, envelope


@pytest.mark.parametrize("marker", ["*", "_"])
def test_report_title_preserves_literal_markdown_delimiters(marker):
    title = f"Name {marker}literal{marker}"
    report = create_report(title=title, language="en")
    heading = report.to_markdown().splitlines()[0]
    assert f"{marker}literal{marker}" not in heading
    assert report.to_dict()["title"] == title
    assert title in report.to_html()


@pytest.mark.parametrize("language", ["tr", "en"])
def test_expert_formats_retain_complete_native_sources_and_exact_values(tmp_path, language):
    result, values, reports = comparison()
    presentation = create_report(*reports, result, language=language, level="expert")
    data = presentation.to_dict()
    assert [source["raw"] for source in data["sources"]] == [reports[0].to_dict(), reports[1].to_dict(), result.to_dict()]
    for section, evaluation_result in zip(data["sources"][:2], values):
        original = evaluation_result.to_dict()
        assert section["raw"]["index"]["records"][0]["native_payload"] == original
        assert sum(map(sum, original["metrics"]["confusion_matrix"])) == 4
        for metric in section["metrics"]:
            expected = original
            for part in metric["key"].split("."):
                expected = expected[int(part)] if isinstance(expected, list) else expected[part]
            assert metric["value"] == expected
    assert data["sources"][2]["metrics"][0]["value"] == 0.25
    path = tmp_path / "report.json"
    presentation.to_json(path)
    assert PresentationReport.from_json(path).to_json() == presentation.to_json()
    markdown = presentation.to_markdown(tmp_path / "report.md")
    markdown_raw = [json.loads(block)["source"] for block in re.findall(r"```json\n(.*?)\n```", markdown, re.S)]
    assert markdown_raw == [section["raw"] for section in data["sources"]]
    document = Document()
    document.feed(presentation.to_html(tmp_path / "report.html"))
    html_raw = []
    for block in document.pre:
        try:
            value = json.loads(block)
        except ValueError:
            continue
        if isinstance(value, dict) and "source" in value:
            html_raw.append(value["source"])
    assert html_raw == markdown_raw
    assert all(v.to_dict()["evaluation_id"] in markdown for v in values)
    assert [s["source_id"] for s in data["sources"]] == [s["source_id"] for s in create_report(*reports, result, language=language, level="expert").to_dict()["sources"]]
    assert len(create_report(reports[0], reports[0]).to_dict()["sources"]) == 1


def test_empty_evaluation_preserves_null_and_zero():
    value = evaluation(empty=True)
    _, report = analysis_source("empty", value)
    presentation = create_report(report, language="en", level="expert")
    native = presentation.to_dict()["sources"][0]["raw"]["index"]["records"][0]["native_payload"]
    assert native == value.to_dict()
    assert native["status"] == "empty" and native["metrics"]["sample_count"] == 0
    assert native["metrics"]["accuracy"] is None
    assert all(row["support"] == 0 and row["f1"] is None for row in native["metrics"]["per_class"])
    assert "Not recorded" in presentation.to_markdown() and "Not recorded" in presentation.to_html()


@pytest.mark.parametrize("value", ["dictionary", "evaluation", "presentation"])
def test_unsupported_direct_report_inputs_are_rejected(value):
    source = {"a": 1} if value == "dictionary" else evaluation() if value == "evaluation" else create_report()
    with pytest.raises(ReportError, match="supported_retained_result"):
        create_report(source)


@pytest.mark.parametrize("kind", ["json", "md", "html"])
def test_export_no_clobber_and_explicit_overwrite(tmp_path, kind):
    report = create_report(title="one")
    writer = {"json": report.to_json, "md": report.to_markdown, "html": report.to_html}[kind]
    path = tmp_path / ("with spaces." + kind)
    raw = writer(path)
    assert path.read_text(encoding="utf8") == raw
    with pytest.raises(FileExistsError):
        writer(path)
    assert path.read_text(encoding="utf8") == raw
    assert writer(path, overwrite=True) == raw


def test_html_and_markdown_special_labels_are_literal():
    label = '<script>alert("x")</script> | [name] `x` *star* _under_ \\ ~~strike~~ &'
    result = evaluation()
    _, report = analysis_source("A", result, context=dict(CONTEXT, dataset_id=label))
    presentation = create_report(report, title=label, language="en", level="expert")
    document = Document()
    document.feed(presentation.to_html())
    heading = presentation.to_markdown().splitlines()[0]
    assert "<script>" not in heading and "*star*" not in heading and "_under_" not in heading
    assert "&#92;" in heading and "&#126;" in heading and "&#124;" in heading
    raw = presentation.to_dict()["sources"][0]["raw"]
    assert raw["context"]["dataset"]["dataset_id"] == label


def test_special_model_identity_is_preserved_in_order_and_escaped():
    name = '<b>model</b> | [x] `t` *star* _id_ $label$'
    a, _ = analysis_source(name, evaluation(linear(1.5)))
    b, _ = analysis_source("B", evaluation(linear(wrapped=True)))
    result = compare_models(a, b, features="P26", context_policy="strict")
    report = create_report(result, language="en", level="expert")
    raw = report.to_dict()["sources"][0]["raw"]
    assert list(raw["sources"]) == ["baseline", "candidate"]
    assert raw["sources"]["baseline"]["model_id"] == name
    Document().feed(report.to_html())
    assert "<b>model</b>" not in report.to_markdown().split("```json")[0]


@pytest.mark.parametrize("damage", ["json", "schema", "metric", "authority"])
def test_invalid_presentation_json_rejected(tmp_path, damage):
    result, _, _ = comparison()
    data = create_report(result).to_dict()
    if damage == "schema":
        data["schema_version"] = 99
    elif damage == "metric":
        data["sources"][0]["metrics"][0]["value"] = 999
    elif damage == "authority":
        data["authoritative_evidence"] = True
        data["content_sha256"] = digest({k: v for k, v in data.items() if k != "content_sha256"})
    path = tmp_path / "bad.json"
    path.write_text("{" if damage == "json" else json.dumps(data), encoding="utf8")
    with pytest.raises(ValueError):
        PresentationReport.from_json(path)


def test_rehashed_presentation_retains_presentation_authority_only():
    report = create_report(title="historical presentation")
    data = report.to_dict()
    data["title"] = "independently supplied presentation"
    data["content_sha256"] = digest({k: v for k, v in data.items() if k != "content_sha256"})
    loaded = PresentationReport(data).to_dict()
    assert loaded["presentation_only"] and not loaded["authoritative_evidence"]


def test_strict_and_legacy_reporting_remain_distinct():
    strict, _, _ = comparison(incompatible=True)
    legacy, _, _ = comparison(incompatible=True, policy="legacy")
    assert strict.to_dict()["schema_version"] == 2 and legacy.to_dict()["schema_version"] == 1
    assert all(row["status"] == "COMPARABLE" for row in legacy.to_dict()["metrics"])
    assert all(row["status"] == "INCOMPATIBLE" for row in strict.to_dict()["metrics"])
    report = create_report(strict, language="en", level="expert")
    assert_quality_withheld(report.to_dict())
    assert "strict_declared_context_v1" in report.to_markdown()
    assert "INCOMPATIBLE" in report.to_html()
    assert "strict_declared_context_v1" not in create_report(legacy).to_json()


def test_graph_three_authority_views_and_second_interpreter(tmp_path):
    from hyso.checkpoint import save_training_checkpoint, load_training_checkpoint
    from examples.reporting.reporting_experiment_acceptance import sha
    model = graph_model()
    checkpoint = tmp_path / "graph.pth"
    save_training_checkpoint(checkpoint, model, graph_contract=True, include_rng=False)
    load_training_checkpoint(checkpoint, model, capture_evidence=True)
    verified, limited = graph_evaluation(model), graph_evaluation(model, receiver=False)
    path = tmp_path / "association.json"
    verified.to_json(path)
    imported = GraphEvaluationResult.from_json(path, expected_sha256=sha(path))
    data = create_report(verified, limited, imported, language="en", level="expert").to_dict()
    assert [s["graph_evidence"]["cross_evidence_binding"] for s in data["sources"]] == ["verified", "limited", "unavailable"]
    assert all(s["graph_evidence"]["legacy_p26"] == "missing" for s in data["sources"])
    changed = graph_evaluation(model, context=dict(CONTEXT, preprocessing_id="different"))
    compared = compare_graph_evaluations(verified, changed, context_policy="strict")
    assert_quality_withheld(create_report(compared, language="en", level="expert").to_dict())
    import hyso
    package_root = str(Path(hyso.__file__).resolve().parent.parent)
    code = "import sys,json;sys.path.insert(0,sys.argv[1]);from hyso.analysis.graph_evidence import GraphEvaluationResult;print(json.dumps(GraphEvaluationResult.from_json(sys.argv[2]).to_dict()['verification']))"
    child = subprocess.run([sys.executable, "-I", "-B", "-c", code, package_root, str(path)], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stderr
    status = json.loads(child.stdout.strip().splitlines()[-1])
    assert status["graph_v2"] == status["receiver_runtime"] == "unverified"


@pytest.mark.parametrize("state", ["created", "running", "failed"])
def test_reopen_never_completes_unfinished_or_failed_run(manager, state):
    experiment = manager.create_experiment("incomplete")
    current = manager.create_run(experiment["id"], "evaluation", {})
    if state != "created":
        current.start()
    if state == "failed":
        current.fail("fixture-failure")
    if state != "running":
        with pytest.raises(ExperimentError, match="invalid_transition"):
            current.finish()
    before = current.to_dict()
    root = manager.root_dir
    manager.close()
    reopened = ExperimentManager(root)
    try:
        assert reopened.get_run(current.id).to_dict() == before
        report = report_experiment(reopened, experiment["id"], language="en")
        assert report.to_dict()["experiment"]["status"] == state
        assert report.to_dict()["experiment"]["status_counts"]["completed"] == 0
    finally:
        reopened.close()


@pytest.mark.parametrize("ownership", ["copy", "reference"])
@pytest.mark.parametrize("damage", ["changed", "missing"])
def test_artifact_failure_status_and_expected_hash_are_persistent(manager, tmp_path, ownership, damage):
    source = tmp_path / "source"
    source.write_bytes(b"original")
    current = artifact_run(manager)
    item = current.add_artifact(source, ownership=ownership, retain_locator=True)
    expected = item["expected_sha256"]
    path = manager.root_dir / item["managed_relative_path"] if ownership == "copy" else source
    assert manager.verify_artifact(item["id"])["status"] == "verified"
    if damage == "changed":
        path.write_bytes(b"changed")
    else:
        path.unlink()
    assert manager.verify_artifact(item["id"])["status"] == ("mismatch" if damage == "changed" else "missing")
    assert manager.get_artifact(item["id"])["expected_sha256"] == expected


def test_empty_same_named_relative_and_idempotent_artifacts(manager, tmp_path, monkeypatch):
    current = artifact_run(manager)
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir(); second.mkdir()
    (first / "same").write_bytes(b"")
    (second / "same").write_bytes(b"different")
    monkeypatch.chdir(first)
    request = uuid.uuid4().hex
    empty = current.add_artifact("same", ownership="copy", request_id=request)
    assert current.add_artifact("same", ownership="copy", request_id=request)["id"] == empty["id"]
    other = current.add_artifact(second / "same", ownership="copy")
    assert empty["id"] != other["id"] and empty["expected_sha256"] != other["expected_sha256"]
    assert empty["expected_size_bytes"] == 0
    assert manager.verify_artifact(empty["id"])["status"] == "verified"
    with pytest.raises(ExperimentError, match="request_conflict"):
        current.add_artifact(second / "same", ownership="copy", request_id=request)


@pytest.mark.parametrize("entity", ["experiment", "run", "artifact"])
def test_unknown_identifiers_fail(manager, entity):
    function = {"experiment": manager.get_experiment, "run": manager.get_run, "artifact": manager.verify_artifact}[entity]
    with pytest.raises(ExperimentError, match="not_found"):
        function("f" * 32)


@pytest.mark.parametrize("damage", ["json", "schema", "metric", "evaluation", "partial"])
def test_hash_valid_invalid_typed_artifact_is_rejected(manager, tmp_path, damage):
    result, values, _ = comparison()
    data = result.to_dict()
    if damage == "schema":
        data["schema_version"] = 99
    elif damage == "metric":
        data["metrics"][0]["difference"]["difference"] = 123
    elif damage == "evaluation":
        data = values[0].to_dict()
    elif damage == "partial":
        data["sources"]["baseline"]["report"]["index"]["records"][0]["native_payload"]["accounting"]["expected_samples"] = 5
    path = tmp_path / "result.json"
    path.write_text("not-json" if damage == "json" else json.dumps(data), encoding="utf8")
    current = artifact_run(manager)
    artifact = current.add_artifact(path, kind="model_comparison", ownership="copy")
    assert manager.verify_artifact(artifact["id"])["status"] == "verified"
    opened = manager.open_result(artifact["id"])
    assert opened["status"] == "invalid_result" and opened["result"] is None


def test_result_type_context_and_null_scalar_fail_before_writes(manager):
    result, values, _ = comparison()
    experiment = manager.create_experiment("typed")
    current = manager.create_workflow_run(experiment["id"], "comparison", model_ids=["A", "B"], comparison=result)
    current.start()
    for bad in (values[0], create_report(result)):
        with pytest.raises(ExperimentError, match="typed_workflow_result_required"):
            current.record_result(bad)
    different, _, _ = comparison(incompatible=True)
    with pytest.raises(ExperimentError, match="comparison_result_mismatch"):
        current.record_result(different)
    with pytest.raises(ExperimentError, match="invalid_metric"):
        current.log_metric(name="empty/accuracy", value=None, unit="fraction", source="qa", step=0)
    assert manager.list_artifacts(current.id) == []


@pytest.mark.parametrize("format", ["json", "markdown", "html"])
def test_cli_renders_saved_strict_result(capsys, tmp_path, format):
    result, _, _ = comparison(incompatible=True)
    path = tmp_path / "strict.json"
    result.to_json(path)
    output = tmp_path / ("rendered." + format)
    code, envelope = invoke(capsys, "reports", "render", path, "--language", "en", "--level", "expert", "--format", format, "--output", output)
    assert code == 0
    assert_quality_withheld(envelope["result"])
    text = output.read_text(encoding="utf8")
    assert "INCOMPATIBLE" in text and "strict_declared_context_v1" in text


@pytest.mark.parametrize("kind", ["missing", "bad-json", "unsupported", "no-context", "strict-flag"])
def test_cli_invalid_inputs_and_missing_features_are_explicit(capsys, tmp_path, kind):
    path = tmp_path / "input.json"
    path.write_text("{" if kind == "bad-json" else "{}", encoding="utf8")
    if kind == "missing":
        path = tmp_path / "absent.json"
    elif kind == "no-context":
        value = evaluation()
        from hyso.analysis import EvidenceIndex, EvidenceRecord
        EvidenceIndex([EvidenceRecord.from_native(value.to_dict(), feature_id="P26")]).to_json(path, overwrite=True)
    if kind == "strict-flag":
        args = ("compare", path, path, "--baseline-model", "A", "--candidate-model", "B", "--features", "P26", "--context-policy", "strict")
    else:
        args = ("reports", "render", path)
    code, envelope = invoke(capsys, *args)
    assert code == 2 and envelope["status"] == "error"


def test_cli_artifact_mismatch_and_unsupported_kind_are_distinct(capsys, manager, tmp_path):
    source = tmp_path / "checkpoint.pth"
    source.write_bytes(b"opaque-checkpoint-bytes")
    current = artifact_run(manager)
    artifact = current.add_artifact(source, retain_locator=True)
    code, result = invoke(capsys, "experiments", "verify", "--root", manager.root_dir, artifact["id"])
    assert code == 0 and result["result"]["status"] == "verified"
    code, result = invoke(capsys, "experiments", "open", "--root", manager.root_dir, artifact["id"])
    assert code == 4 and result["result"]["status"] == "unsupported"
    source.write_bytes(b"changed")
    code, result = invoke(capsys, "experiments", "verify", "--root", manager.root_dir, artifact["id"])
    assert code == 3 and result["result"]["status"] == "mismatch"


def test_retained_training_and_advisor_typed_inputs():
    from hyso.advisor import advise, recommend
    _, _, reports = comparison()
    training = saved_training()
    presentation = create_report(training, advise(reports[0]), recommend(reports[0]), language="en")
    assert [s["kind"] for s in presentation.to_dict()["sources"]] == ["training", "advisor", "recommendations"]
    assert presentation.to_dict()["sources"][0]["raw"] == training.to_dict()


def test_full_qa_lifecycle(tmp_path):
    summary = run(tmp_path / "acceptance")
    assert summary["initial_schema"] == 1 and summary["migration"]["status"] == "migrated"
    assert len(summary["runs"]) == 4 and summary["history"]["status"] == "completed"
    assert all(row["status"] == "verified" for row in summary["before_checks"] + summary["after_checks"])
    assert summary["graph_verified"]["receiver_runtime"] == "verified"
    assert summary["limited"]["status"] == "limited" and summary["imported"]["graph_v2"] == "unverified"
