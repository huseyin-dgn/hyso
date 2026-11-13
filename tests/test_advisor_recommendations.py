"""Stage 9 retained fixtures only; live workloads and CUDA queries are forbidden."""
import ast
from dataclasses import FrozenInstanceError
from html.parser import HTMLParser
import json
from pathlib import Path
import runpy
import sys
import typing

import pytest
import torch

from hyso import analysis, advisor, reporting
from hyso.analysis._codec import canonical, digest
from test_model_comparison import source as comparison_source, throughput

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_workload(monkeypatch):
    from hyso import benchmark, training, performance, quality, evaluation, structure
    from hyso.training.session import TrainingSession
    def forbidden(*args, **kwargs):
        pytest.fail("recommendation attempted workload, device query or checkpoint access")
    for target, name in ((benchmark.BenchmarkRunner, "run"), (training.Trainer, "fit"),
                         (TrainingSession, "fit"), (torch.nn.Module, "_call_impl"), (torch, "load"),
                         (quality, "compare_performance_quality"), (evaluation, "evaluate_classification"),
                         (structure, "capture_model_structure"), (analysis, "execute_analysis")):
        monkeypatch.setattr(target, name, forbidden)
    for name in ("init", "_lazy_init", "is_available", "device_count", "current_device"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    for name in ("measure_throughput", "measure_memory", "measure_footprint"):
        monkeypatch.setattr(performance, name, forbidden)


def context():
    return analysis.AnalysisContext(workload_id="stage9-fixture", task="classification", model={}, dataset={})


def report(records=()):
    return analysis.EvidenceIndex(records).analyze(context=context())


def native(feature="P02", *, missing=None, identity=None):
    data = json.loads((ROOT / "tests/fixtures/advisor-native-v1.json").read_text(encoding="utf-8"))["payloads"][feature]
    if feature == "P02":
        data["provenance"]["environment"]["backend"] = "fixture-native-backend"
        if missing == "backend":
            data["provenance"]["environment"].pop("backend", None)
        if missing == "model":
            for name in ("weights_sha256", "state_sha256"):
                data["provenance"]["model"].pop(name, None)
    if identity:
        data["result_id"] = identity
    return analysis.EvidenceRecord.from_native(data, feature_id=feature, origin="reused",
        scope=analysis.EvidenceScope(model_id="fixture-model", dataset_id="fixture-data",
                                    workload_id="fixture-workload", preprocessing_id="fixture-preprocessing"))


def candidates(result):
    return [r for r in result.to_dict()["recommendations"] if r["action_kind"] == "conditional_experiment"]


def goal_request(source, record, *, target=1e12):
    scope, protocol = advisor.goal_context(advisor.NativeMetricResolver(source), record.record_id,
                                           advisor.Metric.IMAGE_THROUGHPUT)
    return advisor.AdvisorRequest([advisor.UserGoal("rate", advisor.Metric.IMAGE_THROUGHPUT, ">=", target,
                                                   "images/s", scope=scope, protocol=protocol)])


def test_empty_index_requires_context_and_has_no_invented_measurement():
    with pytest.raises(advisor.AdvisorError, match="analysis_context_required"):
        advisor.recommend(analysis.EvidenceIndex())
    result = advisor.recommend(analysis.EvidenceIndex(), context=context())
    data = result.to_dict()
    assert {a["area"] for a in data["areas"]} == {
        "inference", "training", "memory", "model_architecture", "deployment", "data_pipeline"}
    assert all(r["action_kind"] == "measurement_request" and not r["observations"] and not r["evidence"]
               and not r["retained_record_ids"] for r in data["recommendations"])
    assert data["claims"] == dict(global_winner=None, numerical_improvement=None, automatically_applied=False)
    assert set(f for a in data["areas"] for f in a["features"]) == {f"P{i:02d}" for i in range(1, 36)}
    assert candidates(result) == []


def test_original_batch_rule_is_reused_and_candidate_stays_blocked():
    source = report([native()])
    before = canonical(source.to_dict())
    old = advisor.advise(source)
    result = advisor.recommend(source)
    card, = candidates(result)
    assert card["rule_id"] == "plan.batch-throughput"
    assert card["original_finding"] in [f.to_dict() for f in old.findings]
    assert card["plan"]["change_parameter"] == "batch_size"
    assert card["plan"]["changed_variables"] == ["input.batch_size"]
    assert (card["authorization"], card["readiness"], card["expected_effect"], card["estimated_improvement"]
            ) == ("not_authorized", "blocked", "unknown", None)
    assert result.to_dict()["sources"][0]["advisor"] == old.to_dict()
    assert result.to_json() == advisor.recommend(source).to_json()
    assert canonical(source.to_dict()) == before
    assert advisor.validate_advisor_report(old, source) is not None


@pytest.mark.parametrize("missing", ["backend", "model"])
def test_missing_native_binding_prevents_batch_prescription(missing):
    result = advisor.recommend(report([native(missing=missing)]))
    assert not candidates(result)
    assert any(r["rule_id"] == "plan.batch-throughput" and r["action_kind"] == "evidence_review"
               for r in result.to_dict()["recommendations"])


@pytest.mark.parametrize("status", ["unsupported", "not_run", "invalid", "unknown"])
def test_unavailable_evidence_preserves_status_and_never_becomes_zero(status):
    record = analysis.EvidenceRecord.unavailable("P09", status=status, reason="fixture unavailable")
    result = advisor.recommend(report([record]))
    assert not candidates(result)
    slots = [s for c in result.to_dict()["recommendations"] for s in c["coverage"] if s["feature_id"] == "P09"]
    assert slots and all(s["status"] == status for s in slots)
    assert not any(o["reference"]["record_id"] == record.record_id for c in result.to_dict()["recommendations"] for o in c["observations"])


def test_ambiguous_captures_require_selection_without_averaging():
    a, b = native(), native(identity="90000000-0000-0000-0000-000000000000")
    source = report([a, b])
    result = advisor.recommend(source)
    assert not candidates(result)
    cards = [r for r in result.to_dict()["recommendations"] if r["blocked_by"]]
    assert cards and all("ambiguous_captures:select_record_and_scope" in r["blocked_by"] for r in cards)
    # An explicit, uniquely resolved goal may retain the selected local candidate.
    selected = advisor.recommend(source, goal_request(source, a))
    # Equal native contexts across both captures remain ambiguous to the evaluator.
    assert not candidates(selected)


def test_disagreeing_capture_values_are_not_averaged_or_ranked_as_a_bottleneck():
    payloads = [throughput("1", 2.), throughput("2", 1.)]
    records = []
    for payload in payloads:
        payload["provenance"]["environment"]["backend"] = "fixture-native-backend"
        records.append(analysis.EvidenceRecord.from_native(payload, feature_id="P02", origin="reused",
            scope=analysis.EvidenceScope(model_id="disagreeing-fixture")))
    source = report(records)
    assert any(f.kind == advisor.FindingKind.EXPERIMENT_CANDIDATE for f in advisor.advise(source).findings)
    result = advisor.recommend(source)
    assert not candidates(result)
    values = {m["value"] for c in result.to_dict()["recommendations"] for m in c["observations"]}
    assert values == {p["measurements"]["value"] for p in payloads} and len(values) == 2
    assert all(c["priority"] is None and c["estimated_improvement"] is None for c in result.to_dict()["recommendations"])


def test_conflicting_hard_goals_only_request_consistency_review():
    record = native(); source = report([record])
    base = goal_request(source, record).goals[0]
    other = advisor.UserGoal("upper", base.metric, "<=", 1, base.unit, scope=base.scope, protocol=base.protocol)
    result = advisor.recommend(source, advisor.AdvisorRequest([base, other]))
    assert not candidates(result)
    cards = result.to_dict()["recommendations"]
    assert any(r["rule_id"] == "plan.hard-goal-consistency" for r in cards)
    assert all(r["recipe"] in ("goals", "bindings") for r in cards)


@pytest.mark.parametrize("known", [True, False])
def test_met_and_unknown_goals_do_not_create_new_optimization_objectives(known):
    record = native(); source = report([record])
    request = goal_request(source, record, target=1) if known else advisor.AdvisorRequest([
        advisor.UserGoal("unknown", advisor.Metric.IMAGE_THROUGHPUT, ">=", 1e12, "images/s")])
    result = advisor.recommend(source, request)
    assert not candidates(result)
    assert not any(r["rule_id"] == "coverage.review" for r in result.to_dict()["recommendations"])


@pytest.mark.parametrize("compatible", [True, False])
def test_comparison_preserves_gates_without_winner_or_unsupported_prescription(compatible):
    a, b = throughput("1", 2.), throughput("2", 1.)
    if not compatible:
        b["provenance"]["comparison_capture"]["dataset_id"] = "other-data"
    comparison = analysis.compare_models(comparison_source("A", "P02", a), comparison_source("B", "P02", b), features="P02")
    before = comparison.to_json()
    result = advisor.recommend(comparison)
    assert result.to_dict()["source"]["data"] == comparison.to_dict()
    assert result.to_dict()["claims"]["global_winner"] is None
    if not compatible:
        assert not candidates(result)
        assert all(m["status"] == "INCOMPATIBLE" for m in comparison.to_dict()["metrics"])
    assert comparison.to_json() == before


def test_insufficient_comparison_provenance_blocks_otherwise_eligible_local_candidate():
    payloads = [throughput("1", 2.), throughput("2", 1.)]
    for payload in payloads:
        payload["provenance"]["environment"]["backend"] = "fixture-native-backend"
    payloads[0]["provenance"].pop("comparison_capture")
    a, b = [comparison_source(model, "P02", payload) for model, payload in zip(("A", "B"), payloads)]
    assert any(f.kind == advisor.FindingKind.EXPERIMENT_CANDIDATE for f in advisor.advise(
        analysis.UnifiedAnalysisReport(a.to_dict()["report"])).findings)
    comparison = analysis.compare_models(a, b, features=("P02", "P09"))
    assert {m["status"] for m in comparison.to_dict()["metrics"]} == {
        "INSUFFICIENT_EVIDENCE", "NOT_SUPPORTED_IN_COMPARISON"}
    result = advisor.recommend(comparison)
    assert not candidates(result)
    assert any("comparison_not_comparable:no_cross_model_prescription" in c["blocked_by"]
               for c in result.to_dict()["recommendations"])


def test_precision_and_memory_reviews_do_not_prescribe_unproven_optimizations():
    result = advisor.recommend(report([native("P08"), native("P10"), native("P11")]))
    assert not candidates(result)
    cards = result.to_dict()["recommendations"]
    memory = next(c for c in cards if c["rule_id"] == "plan.memory-domain")
    assert {r["domain"] for r in memory["evidence"]} == {"process_rss", "model_logical_bytes"}
    assert memory["plan"]["change_parameter"] is None
    assert all(c["estimated_improvement"] is None and c["priority"] is None for c in cards)
    precision = next(c for c in cards if c["recipe"] == "precision")
    assert any("unsupported" in s for s in precision["missing_evidence"])


@pytest.mark.parametrize("language", ["tr", "en"])
@pytest.mark.parametrize("level", ["beginner", "technical", "expert"])
def test_all_languages_and_levels_have_manual_fields_and_evidence_links(language, level):
    result = advisor.recommend(report([native()]))
    display = result.to_display_dict(language=language, level=level)
    for card in display["recommendations"]:
        assert all(card[k] for k in ("title", "detected", "change", "why", "risks", "steps", "verification"))
        for m in card["observations"]:
            assert isinstance(m["value"], (float, int)) and m["reference"]["unit"] and m["label"]
    html = result.to_html(language=language, level=level)
    md = result.to_markdown(language=language, level=level)
    for card in display["recommendations"]:
        for key in ("evidence_anchor", "verification_anchor"):
            assert "href='#" + card[key] + "'" in html
            assert "id='" + card[key] + "'" in html
            assert "](#" + card[key] + ")" in md and 'id="' + card[key] + '"' in md
    assert "Uygulama adımları" in html if language == "tr" else "Manual steps" in html
    combined = reporting.create_report(report(), result, language=language, level=level)
    assert [s["kind"] for s in combined.to_dict()["sources"]] == ["analysis", "recommendations"]


@pytest.mark.parametrize("change", ["gain", "kind", "priority", "winner", "source"])
def test_import_rederives_and_rejects_forged_advice_even_with_rehashed_json(change):
    data = advisor.recommend(report()).to_dict()
    if change == "gain": data["recommendations"][0]["estimated_improvement"] = 50
    if change == "kind": data["recommendations"][0]["action_kind"] = "conditional_experiment"
    if change == "priority": data["recommendations"][0]["priority"] = 1
    if change == "winner": data["claims"]["global_winner"] = "candidate"
    if change == "source": data["sources"][0]["advisor"]["source_report_id"] = "0" * 64
    data["content_sha256"] = digest({k: v for k, v in data.items() if k != "content_sha256"})
    with pytest.raises(advisor.AdvisorError, match="derivation_or_integrity_mismatch"):
        advisor.RecommendationReport(data)


def test_roundtrip_no_clobber_immutable_and_invalid_options(tmp_path):
    result = advisor.recommend(report())
    path = tmp_path / "advice.json"
    result.to_json(path)
    assert advisor.RecommendationReport.from_json(path).to_json() == result.to_json()
    with pytest.raises(FileExistsError): result.to_json(path)
    with pytest.raises(FrozenInstanceError): result._payload = "forged"
    for kw in (dict(language="de"), dict(level="advanced")):
        with pytest.raises(advisor.AdvisorError): result.to_display_dict(**kw)
    with pytest.raises(advisor.AdvisorError): advisor.recommend({})
    with pytest.raises(advisor.AdvisorError): advisor.recommend(report(), request={})
    assert typing.get_type_hints(advisor.recommend)["return"] is advisor.RecommendationReport


def test_html_escapes_native_context_and_fragment_ids_are_unique():
    unsafe = analysis.EvidenceRecord.unavailable("P09", reason="<script>alert('bad')</script>")
    html = advisor.recommend(report([unsafe])).to_html()
    assert "<script>" not in html and "&lt;script&gt;" in html
    class Links(HTMLParser):
        def __init__(self): super().__init__(); self.ids = []; self.links = []
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if "id" in attrs: self.ids.append(attrs["id"])
            if attrs.get("href", "").startswith("#"): self.links.append(attrs["href"][1:])
    parsed = Links(); parsed.feed(html)
    assert len(parsed.ids) == len(set(parsed.ids))
    assert set(parsed.links) <= set(parsed.ids)


@pytest.mark.parametrize("input_kind", ["empty", "analysis", "comparison"])
def test_executable_example_uses_retained_results_only(tmp_path, monkeypatch, input_kind):
    output = tmp_path / "exports"
    args = ["retained_recommendations.py", "--output", str(output), "--language", "en", "--level", "expert"]
    if input_kind != "empty":
        source = report([native()]) if input_kind == "analysis" else analysis.compare_models(
            comparison_source("A", "P02", throughput("1", 2.)),
            comparison_source("B", "P02", throughput("2", 1.)), features="P02")
        path = tmp_path / "source.json"; source.to_json(path); args.extend(["--input", str(path)])
    monkeypatch.setattr(sys, "argv", args)
    runpy.run_path(str(ROOT / "examples/advisor/retained_recommendations.py"), run_name="__main__")
    assert {p.name for p in output.iterdir()} == {"recommendations.json", "report.json", "report.md", "report.html"}
    result = advisor.RecommendationReport.from_json(output / "recommendations.json")
    presentation = reporting.PresentationReport.from_json(output / "report.json")
    assert presentation.to_dict()["sources"][0]["raw"] == result.to_dict()
    assert (output / "report.md").read_text(encoding="utf-8") == presentation.to_markdown()
    assert (output / "report.html").read_text(encoding="utf-8") == presentation.to_html()


def test_readme_advisor_example_and_guide_commands(tmp_path):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    import re
    blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.S)
    block = next(b for b in blocks if "def explain_performance" in b)
    namespace = {}; exec(compile(ast.parse(block), "README.md", "exec"), namespace)
    source = report([native()]); source.to_json(tmp_path / "analysis.json")
    result = namespace["explain_performance"](tmp_path / "analysis.json", tmp_path)
    assert isinstance(result, advisor.RecommendationReport)
    guide = (ROOT / "docs/guides/performance-advisor.md").read_text(encoding="utf-8")
    assert "python examples/advisor/retained_recommendations.py --output" in guide
    assert "--language en --level expert" in guide


def test_guide_python_example_executes_without_workloads(tmp_path, monkeypatch):
    import re
    monkeypatch.chdir(tmp_path)
    report([native()]).to_json(tmp_path / "saved-analysis.json")
    text = (ROOT / "docs/guides/performance-advisor.md").read_text(encoding="utf-8")
    block, = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    namespace = {}; exec(compile(ast.parse(block), "performance-advisor.md", "exec"), namespace)
    assert namespace["restored"].to_json() == namespace["advice"].to_json()


def test_training_coverage_retains_native_timing_without_observer_or_optimizer_claims():
    payload = json.loads((ROOT / "tests/fixtures/training-p19-schema1.json").read_text(encoding="utf-8"))
    record = analysis.EvidenceRecord.from_native(payload, feature_id="P19", origin="reused")
    source = report([record]); before = canonical(source.to_dict())
    result = advisor.recommend(source)
    card = next(c for c in result.to_dict()["recommendations"] if c["recipe"] == "training")
    assert card["action_kind"] == "evidence_review" and card["plan"] is None
    assert card["retained_record_ids"] == [record.record_id]
    assert not card["observations"]  # The existing metric resolver does not resolve P19.
    assert card["coverage"][0]["status"] == "partial"  # Golden fixture retains partial stage scope.
    assert not candidates(result) and canonical(source.to_dict()) == before
    assert "UX observer" in result.to_html(language="en")


def test_invalid_native_payload_is_rejected_before_recommendation():
    from hyso.analysis._codec import EvidenceError
    payload = native().to_dict()["native_payload"]
    payload["measurements"]["value"] = -1
    with pytest.raises(EvidenceError):
        analysis.EvidenceRecord.from_native(payload, feature_id="P02")


def test_combined_reports_link_metric_to_cards_and_namespace_shared_card_ids():
    source = report([native()])
    one = advisor.recommend(source)
    two = advisor.recommend(source, advisor.AdvisorRequest(priority="memory"))
    combined = reporting.create_report(source, one, two)
    data = combined.to_dict()
    links = [link for m in data["sources"][0]["metrics"] for link in m["recommendation_links"]]
    anchors = [c["recommendation_anchor"] for s in data["sources"][1:] for c in s["recommendations"]["recommendations"]]
    assert len(anchors) == len(set(anchors)) and any(link[1:] in anchors for link in links)
    assert all("id='" + a + "'" in combined.to_html() for a in anchors)


def test_presentation_import_cannot_inject_html_through_recommendation_anchors():
    data = reporting.create_report(advisor.recommend(report())).to_dict()
    data["sources"][0]["recommendations"]["recommendations"][0]["evidence_anchor"] = "x'><script>alert(1)</script>"
    data["content_sha256"] = digest({k: v for k, v in data.items() if k != "content_sha256"})
    presentation = reporting.PresentationReport(data)
    for writer in (presentation.to_html, presentation.to_markdown):
        with pytest.raises(reporting.ReportError, match="invalid_recommendation_anchor"):
            writer()


@pytest.mark.parametrize("language,first_limit", [("tr", "Yalnız manuel inceleme"), ("en", "Manual review only")])
def test_new_presentation_limits_are_localized_without_translating_native_sources(language, first_limit):
    result = advisor.recommend(report())
    before = result.to_json()
    view = result.to_display_dict(language=language, level="beginner")
    presentation = reporting.create_report(result, language=language, level="beginner")
    assert view["limitations"][0].startswith(first_limit)
    assert presentation.to_dict()["sources"][0]["limits"] == view["limitations"]
    assert first_limit in presentation.to_html() and first_limit in presentation.to_markdown()
    assert result.to_json() == before
