"""One presentation contract over existing result, evidence and Advisor APIs."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from ..analysis import AnalysisContext, EvidenceIndex, EvidenceRecord, UnifiedAnalysisReport
from ..analysis.execution import AnalysisExecutionResult
from ..analysis.model_comparison import ModelComparisonResult
from ..analysis._codec import canonical, digest, export, plain, read_json
from ..advisor import advise
from ..advisor.contracts import AdvisorReport
from ..advisor.presentation import to_display_dict as advisor_display
from ..experiments import ExperimentManager
from ..training import TrainingResult
from .catalog import feature_explanations


class ReportError(ValueError):
    """Unsupported source, presentation option, size or serialized report."""


def _text(language, tr, en):
    return tr if language == "tr" else en


def _options(language, level):
    if language not in ("tr", "en"):
        raise ReportError("unsupported_language")
    if level not in ("beginner", "technical", "expert"):
        raise ReportError("unsupported_explanation_level")


def _metric(key, label, value, unit, domain, status, scope, description, limits, *, feature=None, record_id=None):
    return dict(key=key, label=label, value=value, unit=unit, domain=domain, status=status,
                scope=scope, description=description, limits=limits, feature_id=feature,
                record_id=record_id, recommendation_links=[])


def _chart(kind, title, unit, domain, series, scope, references):
    return dict(kind=kind, title=title, unit=unit, domain=domain, series=series,
                scope=scope, references=references)


def _analysis_charts(record, language, labels):
    native = record["native_payload"]
    if native is None:
        return []
    feature = record["feature_id"]
    scope = dict(record_id=record["record_id"], scope=record["scope"], bindings=record["bindings"],
                 protocol=native.get("protocol"), limitations=record["limitations"])
    charts = []
    if feature == "P01":
        samples = (native.get("raw_evidence", {}).get("latency_samples_ms")
                   if native.get("format") == "hyso.performance.latency_analysis"
                   else native.get("timing", {}).get("latency_samples_ms"))
        if samples and any(samples):
            charts.append(_chart("histogram", _text(language, "Kayıtlı model gecikmesi dağılımı", "Retained model latency distribution"),
                "ms", "latency", [dict(label=_text(language, "Tekrar ", "Repeat ") + str(i + 1), x=[], y=row) for i, row in enumerate(samples) if row],
                scope, [record["record_id"]]))
    if feature in ("P02", "P09", "P10", "P11", "P19", "P22"):
        groups = {}
        for metric in record["metrics"]:
            if metric["value"] is None or metric["status"] not in ("measured", "derived"):
                continue
            if feature in ("P09", "P10", "P11") and metric["unit"] != "byte":
                continue
            if feature in ("P02", "P22") and metric["unit"] != "images/s":
                continue
            if feature == "P19" and metric["unit"] != "s":
                continue
            groups.setdefault((metric["domain"], metric["unit"]), []).append(metric)
        for (domain, unit), metrics in groups.items():
            charts.append(_chart("bar", feature + " · " + domain, unit, domain,
                [dict(label=labels.get((record["record_id"], metric["path"]), metric["path"]),
                      x=[metric["path"]], y=[metric["value"]]) for metric in metrics],
                scope, [record["record_id"]]))
    return charts


def _analysis_section(report, language, level, catalog):
    data = UnifiedAnalysisReport(report.to_dict()).to_dict()
    view = advisor_display(advise(report), language=language, audience="summary" if level == "beginner" else "technical")
    labels = {(m["record_id"], m["metric_path"]): m["label"] for m in view["text"]["metrics"]}
    metrics, charts = [], []
    for record in data["index"]["records"]:
        info = catalog[record["feature_id"]]
        for item in record["metrics"]:
            metric = _metric(item["path"], labels.get((record["record_id"], item["path"]), info["name"] + " · " + item["path"]),
                item["value"], item["unit"], item["domain"], item["status"],
                dict(scope=record["scope"], bindings=record["bindings"]), info["explanations"][level],
                [info["limits"], *record["limitations"]], feature=record["feature_id"], record_id=record["record_id"])
            metric["reason"] = item["reason"]
            metrics.append(metric)
        charts.extend(_analysis_charts(record, language, labels))
    return dict(kind="analysis", title=data["analysis"]["title"], raw=data, metrics=metrics, charts=charts,
                features=data["analysis"]["feature_results"], advisor=view, limits=data["analysis"]["risks"])


def _training_section(result, language, level):
    from ..experiments.workflow import _restore_result
    data = _restore_result("training_result", result.to_dict()).to_dict()
    metrics, charts = [], []
    groups = (
        ("loss", "mean_loss", "training_loss", ("train_loss", "val_loss"),
         _text(language, "Eğitim ve doğrulama loss", "Training and validation loss"),
         _text(language, "Loss, kayıtlı hata fonksiyonunun değeridir. Eğitim sürecindeki hata değişimini inceleyin.",
               "Loss is the recorded objective value. Inspect its changes across training.")),
        ("accuracy", "percent", "training_accuracy", ("train_acc", "val_acc"),
         _text(language, "Eğitim ve doğrulama accuracy", "Training and validation accuracy"),
         _text(language, "Accuracy, Trainer geçmişinde yüzde olarak kayıtlı doğru tahmin oranıdır.",
               "Accuracy is the correct-prediction rate recorded as a percentage in Trainer history.")),
    )
    labels = {"train_loss": _text(language, "Eğitim loss", "Training loss"),
              "val_loss": _text(language, "Doğrulama loss", "Validation loss"),
              "train_acc": _text(language, "Eğitim accuracy", "Training accuracy"),
              "val_acc": _text(language, "Doğrulama accuracy", "Validation accuracy")}
    scope = dict(settings=data["settings"], history_scope=data.get("history_scope"),
                 completed_history_epochs=data["completed_history_epochs"])
    limits = [_text(language, "Eğitim geçmişi geri yüklenmiş epoch'lar içerebilir. Kalite yorumu kayıtlı veri ve split kapsamına bağlıdır.",
                    "Training history may include restored epochs. Quality interpretation depends on recorded data and split scope.")]
    if data.get("fixture_scope"):
        limits.append(data["fixture_scope"])
    for _, unit, domain, keys, title, explanation in groups:
        series = []
        for key in keys:
            values = data["history"][key]
            observed = [(i + 1, value) for i, value in enumerate(values) if value is not None]
            if observed:
                epoch, value = observed[-1]
                description = explanation
                if level == "technical":
                    description += _text(language, " Noktalar kayıtlı epoch ve birimle gösterilir. Eksik validation aralıkları eğride boş bırakılır.",
                                         " Points use recorded epochs and units. Missing validation intervals remain gaps in the curve.")
                elif level == "expert":
                    description += " " + str(data.get("history_scope") or limits[0])
                metrics.append(_metric(key, labels[key], value, unit, domain, "recorded", dict(scope, last_observed_epoch=epoch), description, limits))
                series.append(dict(label=labels[key], x=list(range(1, len(values) + 1)), y=values))
        if series:
            charts.append(_chart("line", title, unit, domain, series, scope, list(keys)))
    observer = data["epoch_events"]
    if observer:
        explanation = _text(language, "Observer aralığı, validation ve gözlemci ek yükünü içeren UX süresidir.",
                            "An observer interval is a UX duration including validation and observer overhead.")
        observer_limits = [_text(language, "P19 native eğitim zaman defteri ve GPU işlem süresi ayrı ölçümlerdir.",
                                 "P19 native training ledgers and GPU compute time have separate measurement scopes.")]
        metrics.append(_metric("epoch_observer_interval", _text(language, "UX observer aralığı", "UX observer interval"),
            observer[-1]["interval_seconds"], "s", "ux_observer_interval", "recorded", dict(epoch=observer[-1]["epoch"]), explanation, observer_limits))
        charts.append(_chart("line", _text(language, "UX observer aralıkları", "UX observer intervals"), "s", "ux_observer_interval",
            [dict(label="UX observer", x=[r["epoch"] for r in observer], y=[r["interval_seconds"] for r in observer])],
            dict(scope="observer_intervals", limitations=observer_limits), ["epoch_events"]))
    return dict(kind="training", title=_text(language, "Kayıtlı eğitim sonucu", "Retained training result"), raw=data,
                metrics=metrics, charts=charts, features=[], advisor=None, limits=limits)


def _comparison_section(result, language, level, catalog):
    data = result.to_dict()
    view = result.to_display_dict(language=language)
    metrics, charts = [], []
    for native, row in zip(data["metrics"], view["metrics"]):
        feature = row["feature_id"]
        info = catalog[feature]
        difference = native["difference"]
        metric = _metric(row["metric"], row["label"], None if difference is None else difference["difference"],
            row["unit"], row["domain"], row["status"], dict(scope=row["scope"], models=view["models"]),
            row["explanation"] + " " + info["explanations"][level], [info["limits"], *data["limitations"]], feature=feature)
        metric.update(comparison=native, reason=row["reasons"], percent_change=row["percent_change"],
                      percentage_point_change=row["percentage_point_change"])
        metrics.append(metric)
        if row["status"] == "COMPARABLE" and difference is not None:
            charts.append(_chart("bar", row["label"], row["unit"], row["domain"],
                [dict(label=view["models"][side], x=[view["models"][side]], y=[row[side]]) for side in ("baseline", "candidate")],
                dict(comparison_contract=native["comparison"]["contract"], limitations=[info["limits"], *data["limitations"]]),
                [row["metric"]]))
    structure = data["structure"]
    if structure["comparison"] is not None:
        labels = {
            "parameter_elements": _text(language, "Kayıtlı parametre elemanı", "Registered parameter elements"),
            "buffer_elements": _text(language, "Kayıtlı buffer elemanı", "Registered buffer elements"),
            "parameter_logical_bytes": _text(language, "Yapı kaydındaki parametre baytları", "Registered structure parameter bytes"),
            "buffer_logical_bytes": _text(language, "Yapı kaydındaki buffer baytları", "Registered structure buffer bytes"),
            "tensor_union_logical_bytes": _text(language, "Yapı kaydındaki tensor birleşim baytları", "Registered structure tensor-union bytes"),
        }
        limits = [_text(language, "Yapı snapshot kapsamındaki kayıtlı sayılardır. P11 ölçümü ve runtime bellek karşılaştırması ayrı kanıtlardır.",
                        "Counts belong to the registered structure snapshot. P11 measurements and runtime-memory comparisons have separate evidence.")]
        for row in structure["comparison"]["summary_differences"]:
            label = labels.get(row["metric"], row["metric"])
            metric = _metric(row["metric"], label, row["difference"], row["unit"], "registered_structure", row["status"],
                dict(scope=structure.get("scope")), limits[0], limits)
            metric.update(comparison=row, percent_change=row["percent_change"], percentage_point_change=None)
            metrics.append(metric)
            if row["status"] == "computed" and row["metric"] in labels:
                charts.append(_chart("bar", label, row["unit"], "registered_structure",
                    [dict(label=view["models"][side], x=[view["models"][side]], y=[row[side]["value"]]) for side in ("baseline", "candidate")],
                    dict(scope=structure.get("scope"), limitations=limits), [row["metric"]]))
    return dict(kind="comparison", title=_text(language, "İki model karşılaştırması", "Two-model comparison"), raw=data,
                metrics=metrics, charts=charts, features=[], advisor=view, limits=data["limitations"])


def _advisor_section(report, language, level):
    view = advisor_display(report, language=language, audience="summary" if level == "beginner" else "technical")
    metrics = []
    labels = {(m["record_id"], m["metric_path"]): m["label"] for m in view["text"]["metrics"]}
    for row in view["data"]["metrics"]:
        ref = row["reference"]
        metrics.append(_metric(row["metric"], labels[(ref["record_id"], ref["metric_path"])], row["value"],
            ref["unit"], ref["domain"], "recorded", json.loads(row["native_context_json"]),
            _text(language, "Advisor'ın mevcut kayıt üzerindeki ölçüm gözlemidir.", "An existing Advisor observation over retained evidence."),
            row["limitations"], record_id=ref["record_id"]))
    return dict(kind="advisor", title="Advisor", raw=report.to_dict(), metrics=metrics, charts=[], features=[],
                advisor=view, limits=view["data"]["original_evidence_limitations"])


def _source(result, context, language, level, catalog):
    from ..analysis.graph_evidence import GraphEvaluationResult, GraphComparisonResult, report_section
    if type(result) in (GraphEvaluationResult, GraphComparisonResult):
        return report_section(result, language, level, catalog)
    from ..advisor.recommendations import RecommendationReport
    if type(result) is RecommendationReport:
        view = result.to_display_dict(language=language, level=level)
        return dict(kind="recommendations", title=_text(language, "Performance Advisor önerileri", "Performance Advisor recommendations"),
                    raw=result.to_dict(), metrics=[], charts=[], features=[], advisor=None,
                    recommendations=view, limits=view["limitations"])
    if type(result) is TrainingResult:
        return _training_section(result, language, level)
    if type(result) is ModelComparisonResult:
        return _comparison_section(result, language, level, catalog)
    if type(result) is AdvisorReport:
        return _advisor_section(result, language, level)
    if type(result) is EvidenceIndex:
        if type(context) is not AnalysisContext:
            raise ReportError("analysis_context_required_for_index")
        result = result.analyze(context=context)
    execution = None
    if type(result) is AnalysisExecutionResult:
        if type(context) is not AnalysisContext:
            raise ReportError("analysis_context_required_for_execution")
        from ..experiments.workflow import _restore_result
        execution = _restore_result("analysis_execution", result.to_dict()).to_dict()
        # Existing records retain their original identities. Unavailable execution
        # rows stay in their original envelope without generating new import times.
        index = EvidenceIndex(EvidenceRecord(r["evidence_record"]) for r in execution["features"] if r["evidence_record"] is not None)
        result = index.analyze(context=context)
    if type(result) is UnifiedAnalysisReport:
        section = _analysis_section(result, language, level, catalog)
        if execution is not None:
            section["execution"] = execution
        return section
    raise ReportError("supported_retained_result_required")


@dataclass(frozen=True, init=False)
class PresentationReport:
    """Immutable presentation JSON, Markdown and portable HTML over retained results.

    JSON includes exact source snapshots, native statuses and chart data. Its digest
    checks presentation integrity. Native evidence retains its original authority.
    """
    _payload: str

    def __init__(self, data: dict[str, Any]) -> None:
        body = plain(data)
        fields = {"format", "schema_version", "presentation_only", "authoritative_evidence", "language", "level",
                  "title", "catalog", "sources", "experiment", "content_sha256"}
        if (type(body) is not dict or set(body) != fields
                or body.get("format") != "hyso.presentation_report" or type(body.get("schema_version")) is not int
                or body["schema_version"] != 1 or body.get("presentation_only") is not True
                or body.get("authoritative_evidence") is not False):
            raise ReportError("unsupported_presentation_format")
        _options(body["language"], body["level"])
        expected = digest({k: v for k, v in body.items() if k != "content_sha256"})
        if body.get("content_sha256") != expected:
            raise ReportError("presentation_integrity_mismatch")
        raw = canonical(body)
        if len(raw.encode("utf-8")) > 128 * 1024**2:
            raise ReportError("presentation_size_limit")
        object.__setattr__(self, "_payload", raw)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._payload)

    def to_json(self, path: str | Path | None = None, *, overwrite: bool = False) -> str:
        """Export deterministic presentation JSON with all source and chart data."""
        raw = self._payload + "\n"
        if path is not None:
            export(path, raw, overwrite)
        return raw

    @classmethod
    def from_json(cls, path: str | Path) -> PresentationReport:
        """Read bounded presentation JSON and verify its presentation digest."""
        data, _, _ = read_json(path, max_bytes=128 * 1024**2)
        return cls(data)

    def to_markdown(self, path: str | Path | None = None, *, overwrite: bool = False) -> str:
        """Render explanatory metric tables and native statuses."""
        from .render import markdown
        raw = markdown(self.to_dict())
        if path is not None:
            export(path, raw, overwrite)
        return raw

    def to_html(self, path: str | Path | None = None, *, overwrite: bool = False) -> str:
        """Render a self-contained HTML document with embedded Matplotlib SVG charts."""
        from .render import html_report
        raw = html_report(self.to_dict())
        if path is not None:
            export(path, raw, overwrite)
        return raw


def create_report(*results: Any, language: str = "tr", level: str = "technical",
                  title: str = "HYSo", context: AnalysisContext | None = None) -> PresentationReport:
    """Present retained Training/Analysis/Comparison/Advisor results without execution.

    EvidenceIndex and AnalysisExecutionResult require an explicit AnalysisContext.
    RecommendationReport adds manual recommendations and native evidence links.
    Levels are beginner, technical, expert. Missing values keep their source statuses.
    """
    _options(language, level)
    if type(title) is not str or not title.strip() or len(title) > 512:
        raise ReportError("invalid_report_title")
    metadata = feature_explanations(language=language)
    catalog = {row["feature_id"]: row for row in metadata}
    sources, seen = [], set()
    for result in results:
        section = _source(result, context, language, level, catalog)
        identity = digest(dict(kind=section["kind"], raw=section["raw"], execution=section.get("execution")))
        if identity in seen:
            continue
        seen.add(identity)
        section["source_id"] = identity
        advice = section["advisor"]
        if advice and advice.get("text", {}).get("actions"):
            for metric in section["metrics"]:
                metric["recommendation_links"].append("#advisor-" + identity)
        sources.append(section)
    # Explicit recommendation sources add links to their exact retained records.
    # Existing presentation output is unchanged when no such source is supplied.
    links = {}
    for section in sources:
        for card in section.get("recommendations", {}).get("recommendations", []):
            for record_id in card["retained_record_ids"]:
                links.setdefault(record_id, set()).add("#" + card["recommendation_anchor"])
    for section in sources:
        for metric in section["metrics"]:
            metric["recommendation_links"].extend(sorted(links.get(metric["record_id"], ())))
    body = dict(format="hyso.presentation_report", schema_version=1, presentation_only=True,
                authoritative_evidence=False, language=language, level=level, title=title,
                catalog=metadata, sources=sources, experiment=None)
    body["content_sha256"] = digest(body)
    return PresentationReport(body)


def report_experiment(manager: ExperimentManager, experiment_id: str, *, language: str = "tr",
                      level: str = "technical", title: str = "HYSo") -> PresentationReport:
    """Read all history pages and verified workflow results using the existing Manager.

    Unavailable/invalid artifacts stay explicit. No database state is changed.
    Retained execution artifacts use the experiment's recorded caller context.
    """
    if type(manager) is not ExperimentManager:
        raise ReportError("experiment_manager_required")
    _options(language, level)
    history = manager.get_experiment_history(experiment_id)
    runs = list(history["runs"])
    cursor = history["next_after_seq"]
    while cursor is not None:
        page = manager.get_experiment_history(experiment_id, after_seq=cursor)
        runs.extend(page["runs"])
        cursor = page["next_after_seq"]
    details, sources, seen = [], [], set()
    run_configs = {run["id"]: run["config"] for run in runs}
    for run in runs:
        detail = manager.get_run_details(run["id"])
        for cursor_key, data_key, argument in (("next_metric_seq", "metrics", "after_metric_seq"),
                                               ("next_artifact_seq", "artifacts", "after_artifact_seq")):
            cursor = detail[cursor_key]
            while cursor is not None:
                page = manager.get_run_details(run["id"], **{argument: cursor})
                detail[data_key].extend(page[data_key])
                cursor = page[cursor_key]
            detail[cursor_key] = None
        for item in detail["artifacts"]:
            record = item["artifact"]
            if record["id"] in seen:
                continue
            seen.add(record["id"])
            if record["kind"] in ("training_result", "analysis_execution", "model_comparison"):
                opened = manager.open_result(record["id"])
                item["result_status"] = opened["status"]
                item["result_reason"] = opened["reason"]
                if opened["result"] is not None:
                    caller = run_configs.get(record["producer_run_id"], {}).get("context", {})
                    context = AnalysisContext(workload_id=experiment_id, task="unspecified",
                        model={"ids": caller.get("model_ids")}, dataset=caller.get("dataset") or {})
                    presented = create_report(opened["result"], language=language, level=level, context=context).to_dict()
                    for section in presented["sources"]:
                        section["artifact_ids"] = [record["id"]]
                    sources.append(presented)
        details.append(detail)
    body = create_report(language=language, level=level, title=title).to_dict()
    unique_sources = {}
    for report in sources:
        for section in report["sources"]:
            identity = section["source_id"]
            if identity in unique_sources:
                unique_sources[identity]["artifact_ids"].extend(section["artifact_ids"])
            else:
                unique_sources[identity] = section
    body["sources"] = list(unique_sources.values())
    body["experiment"] = dict(experiment=history["experiment"], status=history["status"],
        status_counts=history["status_counts"], status_scope=history["status_scope"], runs=details)
    body.pop("content_sha256")
    body["content_sha256"] = digest(body)
    return PresentationReport(body)
