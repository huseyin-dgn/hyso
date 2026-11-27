"""Manual recommendations derived from existing, validated Advisor decisions.

This module never executes measurements or edits a model. Native evidence remains
authoritative; explanations, coverage reviews and candidate plans are not proof
of an improvement. Serialized recommendations are rederived on import.
"""
from __future__ import annotations
import typing as _typing

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from ..analysis import AnalysisContext, EvidenceIndex, UnifiedAnalysisReport
from ..analysis._codec import canonical, digest, export, plain, read_json
from ..analysis.model_comparison import ModelComparisonResult, ModelComparisonSource, _selected_report
from ._json import AdvisorError
from .contracts import AdvisorRequest, FindingKind
from .engine import advise
from .planning import PlannedFinding
from .resolver import NativeObservation
from ._recommendation_text import AREAS, AREA_FEATURES, RECIPES, recipe_text


# Presentation-only translations. Canonical rule identifiers, metrics, evidence and
# stored limitations are not translated or modified.
_CARD_LOCALE = {
    'de': {
        'areas': {'inference':'Inferenz', 'training':'Training', 'memory':'Arbeitsspeicher',
                  'model_architecture':'Modellarchitektur', 'deployment':'Bereitstellung',
                  'data_pipeline':'Datenpipeline'},
        'coverage':'Dies ist eine Prüfung der Nachweisabdeckung, kein gemessener Engpass oder nachgewiesener Leistungsgewinn.',
        'blocked':'Klären Sie zunächst die Auswahl der Nachweise und die Vergleichbarkeit; wählen Sie noch keine Änderung.',
        'step':'Wählen Sie die Kennung des Quelldatensatzes, prüfen Sie dessen Kontext und bewerten Sie die Empfehlungen erneut.',
        'limitations': [
            'Es wird nur eine manuelle Prüfung angeboten; Modell, Optimierer, Datenpipeline und Trainingsablauf werden nicht verändert.',
            'Eine evidenzbasierte Nutzen-Kosten-Rangfolge wurde nicht ermittelt; die Priorität der Nutzenden bleibt separat.',
            'Eine Abdeckungsangabe ist keine Optimierungsfreigabe. Resolver v1 unterstützt nur P01/P02/P03/P08/P10/P11/P26.',
            'Bei der Quellenvalidierung werden externe Artefakte nicht erneut gelesen und keine unabhängigen Verbesserungsnachweise erzeugt.',
        ],
    },
    'es': {
        'areas': {'inference':'Inferencia', 'training':'Entrenamiento', 'memory':'Memoria',
                  'model_architecture':'Arquitectura del modelo', 'deployment':'Despliegue',
                  'data_pipeline':'Canalización de datos'},
        'coverage':'Esta es una revisión de cobertura de evidencias, no un cuello de botella medido ni una mejora demostrada.',
        'blocked':'Aclare primero la selección de evidencias y la comparabilidad; no elija todavía ningún cambio.',
        'step':'Seleccione el identificador del registro de origen, valide su contexto y vuelva a revisar las recomendaciones.',
        'limitations': [
            'Solo se ofrece una revisión manual; no se modifican el modelo, el optimizador, la canalización de datos ni el entrenamiento.',
            'No se ha establecido una clasificación de beneficios y costes basada en evidencias; la prioridad del usuario permanece aparte.',
            'La cobertura no implica autorización para optimizar. Resolver v1 solo admite P01/P02/P03/P08/P10/P11/P26.',
            'La validación de origen no vuelve a leer artefactos externos ni crea pruebas independientes de mejora.',
        ],
    },
}


def _area_label(area, language):
    if language in _CARD_LOCALE:
        return _CARD_LOCALE[language]['areas'][area]
    return AREAS[area][0 if language == 'tr' else 1]


def _card_message(kind, language):
    if language in _CARD_LOCALE:
        return _CARD_LOCALE[language][kind]
    original = {
        'coverage': ('Bu bir kapsam incelemesidir; ölçülmüş darboğaz veya iyileşme tespiti değildir.',
                     'This is a coverage review, not a measured bottleneck or demonstrated improvement.'),
        'blocked': ('Önce kanıt seçimini ve karşılaştırma uygunluğunu açıklığa kavuşturun; değişiklik seçmeyin.',
                    'Clarify evidence selection and comparison eligibility first; do not select a change.'),
        'step': ('Kaynak kayıt kimliğini seçin, bağlamı doğrulayın ve önerileri yeniden inceleyin.',
                 'Select the source record identity, validate its context and review recommendations again.'),
    }
    return original[kind][0 if language == 'tr' else 1]

_MAX_BYTES = 128 * 1024**2
_RULE_RECIPES = {
    "plan.latency-stage": "latency", "plan.batch-throughput": "batch",
    "plan.precision-quality": "precision", "plan.memory-domain": "memory",
    "plan.accuracy-preservation": "quality", "plan.hard-goal-consistency": "goals",
}
_COVERAGE_RECIPES = dict(inference="latency", training="training", memory="memory",
                         model_architecture="architecture", deployment="deployment", data_pipeline="pipeline")


def _snapshot(source):
    if type(source) is UnifiedAnalysisReport:
        return dict(kind="analysis", data=source.to_dict())
    if type(source) is ModelComparisonResult:
        return dict(kind="comparison", data=source.to_dict())
    raise AdvisorError("validated_analysis_or_comparison_required")


def _restore(snapshot):
    if type(snapshot) is not dict or set(snapshot) != {"kind", "data"}:
        raise AdvisorError("invalid_recommendation_source")
    if snapshot["kind"] == "analysis":
        return UnifiedAnalysisReport(snapshot["data"])
    if snapshot["kind"] == "comparison":
        return ModelComparisonResult(snapshot["data"])
    raise AdvisorError("invalid_recommendation_source")


def _derive(source, request):
    snapshot = _snapshot(source)
    comparison = snapshot["data"] if snapshot["kind"] == "comparison" else None
    if comparison is None:
        inputs = [("source", source)]
    else:
        inputs = [(side, _selected_report(ModelComparisonSource._restore(comparison["sources"][side]),
                                         comparison["selection"]["features"]))
                  for side in ("baseline", "candidate")]
    sources, items = [], []
    for side, report in inputs:
        data = report.to_dict()
        advice = advise(report, request)
        sources.append(dict(side=side, report=data, advisor=advice.to_dict()))
        records = {r["record_id"]: r for r in data["index"]["records"]}
        slots = {s["feature_id"]: s for s in data["analysis"]["feature_results"]}
        observations = [f for f in advice.findings if type(f) is NativeObservation]

        def emit(code, finding=None):
            recipe = RECIPES[code]
            plan = finding.plan if type(finding) is PlannedFinding else None
            refs = [] if finding is None else [r.to_dict() for r in finding.evidence]
            ids = set(r["record_id"] for r in refs)
            if plan is not None:
                ids.update(plan.capture_records)
            retained = [r for r in records.values() if r["feature_id"] in recipe["features"]
                        and r["validation"]["status"] == "native_validated"]
            if finding is None:
                ids.update(r["record_id"] for r in retained)
            blocked = []
            kind = "evidence_review"
            if finding is not None and finding.kind == FindingKind.EXPERIMENT_CANDIDATE:
                kind = "conditional_experiment"
                # Multiple captures are not a population or a causal contradiction.
                # Preserve the original per-record plan, but require disambiguation
                # unless the user's evaluated goal selected the capture explicitly.
                if not plan.goal_ids:
                    for rid in sorted(ids):
                        record = records[rid]
                        peers = [r for r in records.values() if r["feature_id"] == record["feature_id"]
                                 and r["scope"]["model_id"] == record["scope"]["model_id"]]
                        if len(peers) > 1:
                            blocked.append("ambiguous_captures:select_record_and_scope")
                if comparison is not None:
                    features = {records[rid]["feature_id"] for rid in ids}
                    gates = [m for m in comparison["metrics"] if m["feature_id"] in features]
                    if not gates or any(m["status"] != "COMPARABLE" for m in gates):
                        blocked.append("comparison_not_comparable:no_cross_model_prescription")
                if blocked:
                    kind = "evidence_review"
            coverage = [slots[f] for f in recipe["features"]]
            missing = [f["feature_id"] + ":" + f["status"] + ":" + f["reason"]
                       for f in coverage if f["status"] != "validated"]
            if plan is not None:
                missing.extend(plan.missing_evidence)
            if finding is None and not ids:
                kind = "measurement_request"
            original = None if finding is None else finding.to_dict()
            item = dict(area=recipe["area"], recipe=code, side=side, source_report_id=data["report_id"],
                        action_kind=kind, original_finding=original, plan=None if plan is None else plan.to_dict(),
                        rule_id="coverage.review" if finding is None else finding.rule_id,
                        rule_version="1" if finding is None else finding.rule_version,
                        evidence=refs, retained_record_ids=sorted(ids), coverage=coverage,
                        observations=[o.resolved_metric.to_dict() for o in observations
                                      if o.resolved_metric.reference.record_id in ids],
                        missing_evidence=missing, blocked_by=sorted(set(blocked)),
                        priority=None, expected_effect="unknown", estimated_improvement=None,
                        verified_effect="not_measured", manual_only=True,
                        authorization="not_authorized", readiness="blocked")
            item["recommendation_id"] = digest(item)
            items.append(item)

        for finding in advice.findings:
            if type(finding) is PlannedFinding:
                emit(_RULE_RECIPES[finding.rule_id], finding)
            elif finding.kind == FindingKind.MEASURE_NEXT:
                emit("bindings", finding)
        # Coverage prompts are questions, not new optimization rules or objectives.
        # Explicit goals retain the original goal evaluator's scope.
        if not request.goals:
            covered = {i["recipe"] for i in items if i["side"] == side}
            for area in AREAS:
                if _COVERAGE_RECIPES[area] not in covered:
                    emit(_COVERAGE_RECIPES[area])
    body = dict(format="hyso.advisor.recommendations", schema_version=1,
                source=snapshot, request=request.to_dict(), sources=sources, recommendations=items,
                areas=[dict(area=area, features=list(AREA_FEATURES[area]),
                            recommendation_ids=[i["recommendation_id"] for i in items if i["area"] == area])
                       for area in AREAS],
                priority_status="not_established", requested_priority=None if request.priority is None else request.priority.value,
                claims=dict(global_winner=None, numerical_improvement=None, automatically_applied=False),
                limitations=["Manual review only; no model, optimizer, data pipeline or training flow is changed.",
                             "No evidence-based benefit/cost ranking is established; user priority is retained separately.",
                             "Coverage records are not optimization eligibility. Resolver v1 supports P01/P02/P03/P08/P10/P11/P26 only.",
                             "Source validation does not reread external artifacts or establish independent improvement."])
    body["content_sha256"] = digest(body)
    return body


@dataclass(frozen=True, init=False)
class RecommendationReport:
    """Immutable, read-only recommendations with original rules and native sources.

    Import rederives all decisions, including comparison gates. This verifies
    derivation from the supplied snapshot, not the authenticity of external files.

    Args:
        data (dict[str, Any]): Detached plain-data payload accepted by the existing version/schema
            validator. Required.

    Returns:
        An instance of RecommendationReport with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        RecommendationReport(data)

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en
    """
    _payload: str

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize RecommendationReport with the supplied settings.

        Args:
            data (dict[str, Any]): Detached plain-data payload accepted by the existing version/schema
                validator. Required.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            RecommendationReport(data)

        HYSo Guide: advisor.recommendations
        Offline: hyso docs show advisor.recommendations --lang en
        Open: hyso docs open advisor.recommendations --lang en
        """
        try:
            body = plain(data)
            if len(canonical(body).encode("utf-8")) > _MAX_BYTES:
                raise AdvisorError("recommendation_size_limit")
            if (body.get("format") != "hyso.advisor.recommendations"
                    or type(body.get("schema_version")) is not int or body["schema_version"] != 1):
                raise AdvisorError("unsupported_recommendation_format")
            expected = _derive(_restore(body["source"]), AdvisorRequest.from_dict(body["request"]))
            if canonical(body) != canonical(expected):
                raise AdvisorError("recommendation_derivation_or_integrity_mismatch")
        except AdvisorError:
            raise
        except (ValueError, TypeError, KeyError, AttributeError, IndexError, RecursionError, OverflowError) as exc:
            raise AdvisorError("invalid_recommendation_evidence") from exc
        object.__setattr__(self, "_payload", canonical(expected))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached neutral snapshot, including original Advisor reports.

        Returns:
            A detached plain-data view of this validated object; native units and unavailable values are
            retained.

        Raises:
            No additional validation for a valid object; corrupt retained state or native copy/serialization
            failures propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_dict()

        HYSo Guide: advisor.recommendations
        Offline: hyso docs show advisor.recommendations --lang en
        Open: hyso docs open advisor.recommendations --lang en
        """
        return json.loads(self._payload)

    def to_json(self, path: str | Path | None = None, *, overwrite: bool = False) -> str:
        """Export deterministic neutral JSON; preserve existing files by default.

        Args:
            path (str | Path | None): Filesystem path for the specific reader/exporter; no configuration
                code is executed. Default: None.
            overwrite (bool): Whether an existing output file may be replaced explicitly; inputs remain
                protected. Default: False.

        Returns:
            A JSON string when path is None, otherwise None after writing the file.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_json()

        HYSo Guide: advisor.recommendations
        Offline: hyso docs show advisor.recommendations --lang en
        Open: hyso docs open advisor.recommendations --lang en
        """
        raw = self._payload + "\n"
        if path is not None:
            export(path, raw, overwrite)
        return raw

    @classmethod
    def from_json(cls, path: str | Path) -> RecommendationReport:
        """Reopen bounded JSON without executing a measurement or applying changes.

        Args:
            path (str | Path): Filesystem path for the specific reader/exporter; no configuration code is
                executed. Required.

        Returns:
            A validated RecommendationReport instance reconstructed from the supplied data.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            RecommendationReport.from_json(path)

        HYSo Guide: advisor.recommendations
        Offline: hyso docs show advisor.recommendations --lang en
        Open: hyso docs open advisor.recommendations --lang en
        """
        data, _, _ = read_json(path, max_bytes=_MAX_BYTES)
        return cls(data)

    def to_display_dict(self, *, language: _typing.Literal['tr', 'en', 'de', 'es'] = "tr", level: _typing.Literal['beginner', 'technical', 'expert'] = "technical") -> dict[str, Any]:
        """Explain decisions in tr/en/de/es at beginner/technical/expert levels.

        Native numbers, units, protocol bindings and original rule text remain
        unchanged. Coverage prompts explicitly distinguish gaps from measurements.

        Args:
            language (str): Presentation language supported by this API, tr/en/de/es for this presentation;
                guide languages do not expand unrelated APIs. Default: 'tr'.
            level (str): Existing beginner, technical or expert presentation detail level. Default:
                'technical'.

        Returns:
            Localized presentation data; canonical comparison values and units are retained.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_display_dict()

        HYSo Guide: advisor.recommendations
        Offline: hyso docs show advisor.recommendations --lang en
        Open: hyso docs open advisor.recommendations --lang en
        """
        from .presentation import to_display_dict
        if language not in ("tr", "en", "de", "es") or level not in ("beginner", "technical", "expert"):
            raise AdvisorError("unsupported_recommendation_presentation")
        data = self.to_dict()
        from ..reporting.catalog import feature_explanations
        metadata = {f["feature_id"]: f for f in feature_explanations(language=language)}
        record_features = {(s["side"], r["record_id"]): r["feature_id"] for s in data["sources"]
                           for r in s["report"]["index"]["records"]}
        labels = {}
        from .contracts import AdvisorReport
        for source in data["sources"]:
            view = to_display_dict(AdvisorReport.from_dict(source["advisor"]), language=language,
                                   audience="summary" if level == "beginner" else "technical")
            labels.update({(source["side"], row["record_id"], row["metric_path"]): row["label"]
                           for row in view["text"]["metrics"]})
        cards = []
        for item in data["recommendations"]:
            card = dict(item, **recipe_text(item["recipe"], language, level))
            if item["original_finding"] is None:
                card["detected"] = _card_message('coverage', language)
            if item["blocked_by"]:
                card["change"] = _card_message('blocked', language)
                card["steps"] = (_card_message('step', language),)
            card["area_label"] = _area_label(item["area"], language)
            identity = data["content_sha256"] + "-" + item["recommendation_id"]
            card["recommendation_anchor"] = "recommendation-" + identity
            card["evidence_anchor"] = "evidence-" + identity
            card["verification_anchor"] = "verify-" + identity
            card["observations"] = [dict(m, label=labels[(item["side"], m["reference"]["record_id"],
                                                          m["reference"]["metric_path"])],
                                         description=metadata[record_features[(item["side"], m["reference"]["record_id"])]]
                                                             ["explanations"][level]) for m in item["observations"]]
            cards.append(card)
        comparison_observations = []
        if data["source"]["kind"] == "comparison":
            comparison_observations = ModelComparisonResult(data["source"]["data"]).to_display_dict(
                language=language)["comparison_observations"]
        return dict(format="hyso.advisor.recommendations.display", language=language, level=level,
                    comparison_observations=comparison_observations,
                    recommendations=cards, areas=[dict(a, label=_area_label(a["area"], language))
                                                   for a in data["areas"]],
                    sources=data["sources"], comparison=data["source"]["data"] if data["source"]["kind"] == "comparison" else None,
                    priority_status=data["priority_status"], limitations=(_CARD_LOCALE[language]["limitations"] if language in _CARD_LOCALE else data["limitations"] if language == "en" else [
                        "Yalnız manuel inceleme sunulur; model, optimizer, veri hattı veya eğitim akışı değiştirilmez.",
                        "Kanıta dayalı fayda/maliyet sıralaması belirlenmedi; kullanıcı önceliği ayrı korunur.",
                        "Kapsam kaydı optimizasyon uygunluğu değildir. Resolver v1 yalnız P01/P02/P03/P08/P10/P11/P26'yı destekler.",
                        "Kaynak doğrulaması dış artifact dosyalarını yeniden okumaz ve bağımsız iyileşme kanıtı oluşturmaz."]))

    def to_markdown(self, path: str | Path | None = None, *, language: _typing.Literal['tr', 'en', 'de', 'es'] = "tr",
                    level: _typing.Literal['beginner', 'technical', 'expert'] = "technical", overwrite: bool = False) -> str:
        """Render through the existing Stage 8 presentation contract.

        Args:
            path (str | Path | None): Filesystem path for the specific reader/exporter; no configuration
                code is executed. Default: None.
            language (str): Presentation language supported by this API, tr/en/de/es for this presentation;
                guide languages do not expand unrelated APIs. Default: 'tr'.
            level (str): Existing beginner, technical or expert presentation detail level. Default:
                'technical'.
            overwrite (bool): Whether an existing output file may be replaced explicitly; inputs remain
                protected. Default: False.

        Returns:
            The localized Markdown report string.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_markdown()

        HYSo Guide: advisor.recommendations
        Offline: hyso docs show advisor.recommendations --lang en
        Open: hyso docs open advisor.recommendations --lang en
        """
        from ..reporting import create_report
        return create_report(self, language=language, level=level).to_markdown(path, overwrite=overwrite)

    def to_html(self, path: str | Path | None = None, *, language: _typing.Literal['tr', 'en', 'de', 'es'] = "tr",
                level: _typing.Literal['beginner', 'technical', 'expert'] = "technical", overwrite: bool = False) -> str:
        """Export portable HTML with source evidence and manual verification links.

        Args:
            path (str | Path | None): Filesystem path for the specific reader/exporter; no configuration
                code is executed. Default: None.
            language (str): Presentation language supported by this API, tr/en/de/es for this presentation;
                guide languages do not expand unrelated APIs. Default: 'tr'.
            level (str): Existing beginner, technical or expert presentation detail level. Default:
                'technical'.
            overwrite (bool): Whether an existing output file may be replaced explicitly; inputs remain
                protected. Default: False.

        Returns:
            The retained to html value, without recomputing measurements.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_html()

        HYSo Guide: advisor.recommendations
        Offline: hyso docs show advisor.recommendations --lang en
        Open: hyso docs open advisor.recommendations --lang en
        """
        from ..reporting import create_report
        return create_report(self, language=language, level=level).to_html(path, overwrite=overwrite)


def recommend(source: UnifiedAnalysisReport | EvidenceIndex | ModelComparisonResult,
              request: AdvisorRequest | None = None, *, context: AnalysisContext | None = None) -> RecommendationReport:
    """Explain existing Advisor plans without running or modifying anything.

    EvidenceIndex requires an explicit AnalysisContext. ModelComparisonResult
    retains its native comparability gates and cannot select a global winner.
    Only the existing eligible CPU batch rule may yield a conditional experiment;
    all candidates remain manual, blocked and without an estimated improvement.
    Other domains receive evidence reviews, not unvalidated optimization recipes.

    Args:
        source (UnifiedAnalysisReport | EvidenceIndex | ModelComparisonResult): Typed native/retained
            source required by this API, not a declaration of producer authority. Required.
        request (AdvisorRequest | None): Validated advisory/selection request; does not authorize
            execution. Default: None.
        context (AnalysisContext | None): Explicit analysis/evaluation context; caller declarations are
            not producer attestation. Default: None.

    Returns:
        The retained recommend value, without recomputing measurements.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        recommend(source)

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en
    """
    if type(source) is EvidenceIndex:
        if type(context) is not AnalysisContext:
            raise AdvisorError("analysis_context_required_for_index")
        source = source.analyze(context=context)
    _snapshot(source)
    request = AdvisorRequest() if request is None else request
    if type(request) is not AdvisorRequest:
        raise AdvisorError("typed_request_required")
    return RecommendationReport(_derive(source, request))
