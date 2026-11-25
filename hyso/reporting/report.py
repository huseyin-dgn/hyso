"""One presentation contract over existing result, evidence and Advisor APIs."""
from __future__ import annotations
import typing as _typing

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
    """Unsupported source, presentation option, size or serialized report.

    Returns:
        An instance of ReportError with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        ReportError()

    HYSo Guide: reporting.outputs
    Offline: hyso docs show reporting.outputs --lang en
    Open: hyso docs open reporting.outputs --lang en
    """


# DE/ES translations for presentation-only labels and explanations.
# Technical evidence keys, status codes, hashes and measurements remain unchanged.
_REPORT_TEXT_DE_ES = {'Retained model latency distribution': ('Verteilung der aufgezeichneten Modelllatenz', 'Distribución de la latencia registrada del modelo'), 'Repeat ': ('Wiederholung ', 'Repetición '), 'Training and validation loss': ('Trainings- und Validierungsverlust', 'Pérdida de entrenamiento y validación'), 'Loss is the recorded objective value. Inspect its changes across training.': ('Der Verlust ist der aufgezeichnete Wert der Zielfunktion. Prüfen Sie seine Entwicklung während des Trainings.', 'La pérdida es el valor registrado de la función objetivo. Examine su evolución durante el entrenamiento.'), 'Training and validation accuracy': ('Trainings- und Validierungsgenauigkeit', 'Exactitud de entrenamiento y validación'), 'Accuracy is the correct-prediction rate recorded as a percentage in Trainer history.': ('Die Genauigkeit ist der in Prozent aufgezeichnete Anteil korrekter Vorhersagen im Trainer-Verlauf.', 'La exactitud es el porcentaje de predicciones correctas registrado en el historial de Trainer.'), 'Training loss': ('Trainingsverlust', 'Pérdida de entrenamiento'), 'Validation loss': ('Validierungsverlust', 'Pérdida de validación'), 'Training accuracy': ('Trainingsgenauigkeit', 'Exactitud de entrenamiento'), 'Validation accuracy': ('Validierungsgenauigkeit', 'Exactitud de validación'), 'Training history may include restored epochs. Quality interpretation depends on recorded data and split scope.': ('Der Trainingsverlauf kann wiederhergestellte Epochen enthalten. Die Qualitätsbewertung hängt von den aufgezeichneten Daten und der Aufteilung ab.', 'El historial de entrenamiento puede incluir épocas restauradas. La interpretación de la calidad depende de los datos registrados y de la partición.'), ' Points use recorded epochs and units. Missing validation intervals remain gaps in the curve.': (' Die Punkte verwenden aufgezeichnete Epochen und Einheiten. Fehlende Validierungsintervalle bleiben als Lücken in der Kurve erhalten.', ' Los puntos usan épocas y unidades registradas. Los intervalos de validación ausentes permanecen como huecos en la curva.'), 'An observer interval is a UX duration including validation and observer overhead.': ('Ein Beobachterintervall ist eine UX-Zeitspanne einschließlich Validierung und Beobachter-Overhead.', 'Un intervalo del observador es una duración de UX que incluye validación y sobrecarga del observador.'), 'P19 native training ledgers and GPU compute time have separate measurement scopes.': ('Native P19-Trainingsprotokolle und GPU-Rechenzeit haben unterschiedliche Messumfänge.', 'Los registros nativos de entrenamiento P19 y el tiempo de cálculo de GPU tienen ámbitos de medición distintos.'), 'UX observer interval': ('UX-Beobachterintervall', 'Intervalo del observador UX'), 'UX observer intervals': ('UX-Beobachterintervalle', 'Intervalos del observador UX'), 'Retained training result': ('Aufgezeichnetes Trainingsergebnis', 'Resultado de entrenamiento registrado'), 'Models and evidence sources': ('Modelle und Nachweisquellen', 'Modelos y fuentes de evidencia'), 'Structure evidence scope': ('Umfang des Strukturnachweises', 'Alcance de la evidencia estructural'), 'Structure sources': ('Strukturquellen', 'Fuentes estructurales'), 'Registered module types': ('Registrierte Modultypen', 'Tipos de módulos registrados'), 'Tensor dtype metadata': ('Tensor-Datentyp-Metadaten', 'Metadatos de tipos de datos de tensores'), 'Added / removed module paths': ('Hinzugefügte / entfernte Modulpfade', 'Rutas de módulos añadidas / eliminadas'), 'Module and configuration differences': ('Modul- und Konfigurationsunterschiede', 'Diferencias de módulos y configuración'), 'State tensor metadata differences': ('Unterschiede der Zustandstensor-Metadaten', 'Diferencias de metadatos de tensores de estado'), 'Structure evidence limits': ('Grenzen des Strukturnachweises', 'Límites de la evidencia estructural'), 'Field': ('Feld', 'Campo'), 'Value': ('Wert', 'Valor'), 'Type': ('Typ', 'Tipo'), 'modules': ('Module', 'módulos'), 'paths': ('Pfade', 'rutas'), 'Registration': ('Registrierung', 'Registro'), 'Metric': ('Metrik', 'Métrica'), 'Status': ('Status', 'Estado'), 'A only': ('Nur A', 'Solo A'), 'B only': ('Nur B', 'Solo B'), 'Path / tensor': ('Pfad / Tensor', 'Ruta / tensor'), 'Explanation': ('Erläuterung', 'Explicación'), 'Registered parameter elements': ('Registrierte Parameterelemente', 'Elementos de parámetros registrados'), 'Trainable parameter elements': ('Trainierbare Parameterelemente', 'Elementos de parámetros entrenables'), 'Frozen parameter elements': ('Eingefrorene Parameterelemente', 'Elementos de parámetros congelados'), 'Unique parameter tensors': ('Eindeutige Parametertensoren', 'Tensores de parámetros únicos'), 'Unique buffer tensors': ('Eindeutige Puffertensoren', 'Tensores de búfer únicos'), 'Unique registered modules': ('Eindeutige registrierte Module', 'Módulos registrados únicos'), 'Registered module paths': ('Registrierte Modulpfade', 'Rutas de módulos registradas'), 'Cross-category tensors': ('Kategorieübergreifende Tensoren', 'Tensores entre categorías'), 'Registered buffer elements': ('Registrierte Pufferelemente', 'Elementos de búfer registrados'), 'Registered structure parameter bytes': ('Parameterbytes der registrierten Struktur', 'Bytes de parámetros de la estructura registrada'), 'Registered structure buffer bytes': ('Pufferbytes der registrierten Struktur', 'Bytes de búfer de la estructura registrada'), 'Registered structure tensor-union bytes': ('Tensor-Vereinigungsbytes der registrierten Struktur', 'Bytes de la unión de tensores de la estructura registrada'), 'Counts belong to the registered structure snapshot. P11 measurements and runtime-memory comparisons have separate evidence.': ('Die Zahlen stammen aus dem registrierten Struktur-Snapshot. P11-Messungen und Laufzeitspeichervergleiche erfordern getrennte Nachweise.', 'Los recuentos pertenecen a la instantánea de estructura registrada. Las mediciones P11 y las comparaciones de memoria en ejecución requieren evidencias separadas.'), 'Retained P12 analytic observations': ('Aufgezeichnete analytische P12-Beobachtungen', 'Observaciones analíticas P12 registradas'), 'Evidence': ('Nachweis', 'Evidencia'), 'Two-model comparison': ('Vergleich zweier Modelle', 'Comparación de dos modelos'), 'An existing Advisor observation over retained evidence.': ('Eine vorhandene Advisor-Beobachtung auf Grundlage aufgezeichneter Nachweise.', 'Una observación existente de Advisor basada en evidencia registrada.'), 'Performance Advisor recommendations': ('Empfehlungen des Performance Advisor', 'Recomendaciones de Performance Advisor')}


def _text(language, tr, en):
    if language == "tr":
        return tr
    if language == "en":
        return en
    if language in ("de", "es"):
        translated = _REPORT_TEXT_DE_ES.get(en)
        if translated is None:
            raise ReportError("missing_report_translation")
        return translated[0 if language == "de" else 1]
    raise ReportError("unsupported_language")


def _options(language, level):
    if type(language) is not str or language not in ("tr", "en", "de", "es"):
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


def _comparison_tables(data, language):
    """Project retained structure fields; never capture or calculate new metadata."""
    tables = []

    def table(title_tr, title_en, headers, rows):
        if rows:
            tables.append(dict(title=_text(language, title_tr, title_en), headers=headers, rows=rows))

    sides = [data['sources'][side] for side in ('baseline', 'candidate')]
    table('Modeller ve kanıt kaynakları', 'Models and evidence sources',
          [_text(language, 'Alan', 'Field'), 'A', 'B'], [
        ['model_id', *[s['model_id'] for s in sides]],
        ['selected_record_ids', *[s['records'] for s in sides]],
        ['report_id', *[s['report']['report_id'] for s in sides]],
        ['selected_features', data['selection']['features'], data['selection']['features']],
    ])
    structure = data['structure']
    table('Yapı kanıtının kapsamı', 'Structure evidence scope',
          [_text(language, 'Alan', 'Field'), _text(language, 'Değer', 'Value')],
          [[key, structure.get(key)] for key in ('status', 'scope', 'reasons', 'original_profile_sha256')])
    comp = structure['comparison']
    if comp is not None:
        table('Yapı kaynakları', 'Structure sources', [_text(language, 'Alan', 'Field'), 'A', 'B'], [
            ['snapshot_reference', comp['baseline'], comp['candidate']],
            ['structure_comparison_sha256', comp['content_sha256'], comp['content_sha256']],
            ['coverage', comp['coverage'], comp['coverage']],
        ])
        table('Kayıtlı modül türleri', 'Registered module types',
            [_text(language, 'Tür', 'Type'), *[prefix + ' ' + _text(language, 'modül', 'modules') for prefix in ('A', 'B', 'B−A')],
             *[prefix + ' ' + _text(language, 'yol', 'paths') for prefix in ('A', 'B', 'B−A')]],
            [[r['name'], r['unique_modules']['baseline']['value'], r['unique_modules']['candidate']['value'],
              r['unique_modules']['difference'], r['registered_paths']['baseline']['value'],
              r['registered_paths']['candidate']['value'], r['registered_paths']['difference']]
             for r in comp['module_type_differences']])
        table('Tensor dtype bilgileri', 'Tensor dtype metadata',
            [_text(language, 'Kayıt', 'Registration'), 'dtype', _text(language, 'Metrik', 'Metric'), 'A', 'B', 'B−A',
             _text(language, 'Durum', 'Status')],
            [[r['kind'], r['dtype'], key, r[key]['baseline'], r[key]['candidate'], r[key]['difference'], r[key]['status']]
             for r in comp['dtype_differences'] for key in ('tensor_count', 'elements', 'logical_bytes')])
        table('Eklenen / kaldırılan modül yolları', 'Added / removed module paths',
            [_text(language, 'Yalnız A', 'A only'), _text(language, 'Yalnız B', 'B only')],
            [[comp['removed_paths'], comp['added_paths']]])
        fields = [_text(language, 'Yol / tensor', 'Path / tensor'), _text(language, 'Alan', 'Field'), 'A', 'B',
                  _text(language, 'Durum', 'Status')]
        table('Modül ve config farkları', 'Module and configuration differences', fields,
            [[r['path'], f['field'], f['baseline'], f['candidate'], f['status']]
             for r in comp['changed_paths'] for f in r['metadata_changes'] + r['configuration_changes']])
        table('State tensor metadata farkları', 'State tensor metadata differences', fields,
            [[r['owner_path'] + ':' + r['kind'] + ':' + r['name'], f['field'], f['baseline'], f['candidate'], f['status']]
             for r in comp['tensor_differences'] for f in r['changes']])
        table('Yapı kanıtının sınırları', 'Structure evidence limits', [_text(language, 'Açıklama', 'Explanation')],
            [[item] for item in comp['limitations']] + [[item] for item in comp['evidence_gaps']])
    return tables


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
        metric.update(comparison=native, baseline=row['baseline'], candidate=row['candidate'],
                      reason=row["reasons"], percent_change=row["percent_change"],
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
            "trainable_elements": _text(language, "Eğitilebilir parametre elemanı", "Trainable parameter elements"),
            "frozen_elements": _text(language, "Dondurulmuş parametre elemanı", "Frozen parameter elements"),
            "unique_parameter_tensors": _text(language, "Benzersiz parametre tensor sayısı", "Unique parameter tensors"),
            "unique_buffer_tensors": _text(language, "Benzersiz buffer tensor sayısı", "Unique buffer tensors"),
            "unique_modules": _text(language, "Benzersiz kayıtlı modül sayısı", "Unique registered modules"),
            "registered_module_paths": _text(language, "Kayıtlı modül yolu sayısı", "Registered module paths"),
            "cross_category_tensors": _text(language, "Ortak parametre/buffer tensor sayısı", "Cross-category tensors"),
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
            metric.update(comparison=row, baseline=row['baseline']['value'], candidate=row['candidate']['value'],
                          percent_change=row["percent_change"], percentage_point_change=None)
            metrics.append(metric)
            if row["status"] == "computed" and row["metric"] in (
                    'parameter_elements', 'buffer_elements', 'parameter_logical_bytes',
                    'buffer_logical_bytes', 'tensor_union_logical_bytes'):
                charts.append(_chart("bar", label, row["unit"], "registered_structure",
                    [dict(label=view["models"][side], x=[view["models"][side]], y=[row[side]["value"]]) for side in ("baseline", "candidate")],
                    dict(scope=structure.get("scope"), limitations=limits), [row["metric"]]))
    tables = _comparison_tables(data, language)
    p12_observations = []
    if 'P12' in data['selection']['features']:
        from ..advisor.p12_observation import observe_p12
        for side, source in data['sources'].items():
            ids = source['records'].get('P12', [])
            if len(ids) != 1:
                continue
            record = next(r for r in source['report']['index']['records'] if r['record_id'] == ids[0])
            if record['native_payload'] is not None:
                p12_observations.append(dict(model_id=source['model_id'], side=side,
                    observation=observe_p12(EvidenceRecord(record), version=1).to_dict()))
        if p12_observations:
            tables.append(dict(title=_text(language, 'P12 kayıtlı analitik gözlemleri',
                                                     'Retained P12 analytic observations'),
                headers=['Model', 'MAC/batch', 'logical_FLOP/batch', _text(language, 'Kanıt', 'Evidence')],
                rows=[[p['model_id'], p['observation']['metrics']['macs_per_batch']['value'],
                       p['observation']['metrics']['logical_flops_per_batch']['value'],
                       dict(record_id=p['observation']['evidence_reference']['record_id'],
                            comparison_eligibility=[dict(metric=r['metric'],status=r['status'],reasons=r['reasons'])
                                                    for r in data['metrics'] if r['feature_id']=='P12'],
                            limitations=p['observation']['limitations'])] for p in p12_observations]))
    return dict(kind="comparison", title=_text(language, "İki model karşılaştırması", "Two-model comparison"), raw=data,
                metrics=metrics, charts=charts, features=[], advisor=view, tables=tables,
                p12_observations=p12_observations, limits=data["limitations"])


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
    from ..analysis.graph_heads import MultiHeadGraphEvaluationResult, report_section as heads_section
    if type(result) is MultiHeadGraphEvaluationResult:
        return heads_section(result,language,level,catalog)
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

    Args:
        data (dict[str, Any]): Detached plain-data payload accepted by the existing version/schema validator. Required.

    Returns:
        An instance of PresentationReport with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        PresentationReport(data)

    HYSo Guide: reporting.outputs
    Offline: hyso docs show reporting.outputs --lang en
    Open: hyso docs open reporting.outputs --lang en
    """
    _payload: str

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize PresentationReport with the supplied settings.

        Args:
            data (dict[str, Any]): Detached plain-data payload accepted by the existing version/schema validator. Required.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            PresentationReport(data)

        HYSo Guide: reporting.outputs
        Offline: hyso docs show reporting.outputs --lang en
        Open: hyso docs open reporting.outputs --lang en
        """
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
        """Return a detached plain-data view of the stored state.

        Returns:
            A detached plain-data view of this validated object; native units and unavailable values are retained.

        Raises:
            No additional validation for a valid object; corrupt retained state or native copy/serialization failures propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_dict()

        HYSo Guide: reporting.outputs
        Offline: hyso docs show reporting.outputs --lang en
        Open: hyso docs open reporting.outputs --lang en
        """
        return json.loads(self._payload)

    def to_json(self, path: str | Path | None = None, *, overwrite: bool = False) -> str:
        """Export deterministic presentation JSON with all source and chart data.

        Args:
            path (str | Path | None): Filesystem path for the specific reader/exporter; no configuration code is executed. Default: None.
            overwrite (bool): Whether an existing output file may be replaced explicitly; inputs remain protected. Default: False.

        Returns:
            A JSON string when path is None, otherwise None after writing the file.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_json()

        HYSo Guide: reporting.outputs
        Offline: hyso docs show reporting.outputs --lang en
        Open: hyso docs open reporting.outputs --lang en
        """
        raw = self._payload + "\n"
        if path is not None:
            export(path, raw, overwrite)
        return raw

    @classmethod
    def from_json(cls, path: str | Path) -> PresentationReport:
        """Read bounded presentation JSON and verify its presentation digest.

        Args:
            path (str | Path): Filesystem path for the specific reader/exporter; no configuration code is executed. Required.

        Returns:
            A validated PresentationReport instance reconstructed from the supplied data.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            PresentationReport.from_json(path)

        HYSo Guide: reporting.outputs
        Offline: hyso docs show reporting.outputs --lang en
        Open: hyso docs open reporting.outputs --lang en
        """
        data, _, _ = read_json(path, max_bytes=128 * 1024**2)
        return cls(data)

    def to_markdown(self, path: str | Path | None = None, *, overwrite: bool = False) -> str:
        """Render explanatory metric tables and native statuses.

        Args:
            path (str | Path | None): Filesystem path for the specific reader/exporter; no configuration code is executed. Default: None.
            overwrite (bool): Whether an existing output file may be replaced explicitly; inputs remain protected. Default: False.

        Returns:
            The localized Markdown report string.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_markdown()

        HYSo Guide: reporting.outputs
        Offline: hyso docs show reporting.outputs --lang en
        Open: hyso docs open reporting.outputs --lang en
        """
        from .render import markdown
        raw = markdown(self.to_dict())
        if path is not None:
            export(path, raw, overwrite)
        return raw

    def to_html(self, path: str | Path | None = None, *, overwrite: bool = False) -> str:
        """Render a self-contained HTML document with embedded Matplotlib SVG charts.

        Args:
            path (str | Path | None): Filesystem path for the specific reader/exporter; no configuration code is executed. Default: None.
            overwrite (bool): Whether an existing output file may be replaced explicitly; inputs remain protected. Default: False.

        Returns:
            The retained to html value, without recomputing measurements.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_html()

        HYSo Guide: reporting.outputs
        Offline: hyso docs show reporting.outputs --lang en
        Open: hyso docs open reporting.outputs --lang en
        """
        from .render import html_report
        raw = html_report(self.to_dict())
        if path is not None:
            export(path, raw, overwrite)
        return raw


def create_report(*results: Any, language: _typing.Literal['tr', 'en', 'de', 'es'] = "tr", level: _typing.Literal['beginner', 'technical', 'expert'] = "technical",
                  title: str = "HYSo", context: AnalysisContext | None = None) -> PresentationReport:
    """Present retained Training/Analysis/Comparison/Advisor results without execution.

    EvidenceIndex and AnalysisExecutionResult require an explicit AnalysisContext.
    RecommendationReport adds manual recommendations and native evidence links.
    Levels are beginner, technical, expert. Missing values keep their source statuses.

    Args:
        language (str): Presentation language: tr/en/de/es; free-form evidence text stays unchanged. Default: 'tr'.
        level (str): Existing beginner, technical or expert presentation detail level. Default: 'technical'.
        title (str): Human-readable report title. Default: 'HYSo'.
        context (AnalysisContext | None): Explicit analysis/evaluation context; caller declarations are not producer attestation. Default: None.
        results (Any): Existing typed retained results to present; no workload is rerun. Required.

    Returns:
        The retained create report value, without recomputing measurements.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        create_report()

    HYSo Guide: reporting.outputs
    Offline: hyso docs show reporting.outputs --lang en
    Open: hyso docs open reporting.outputs --lang en
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


def report_experiment(manager: ExperimentManager, experiment_id: str, *, language: _typing.Literal['tr', 'en', 'de', 'es'] = "tr",
                      level: _typing.Literal['beginner', 'technical', 'expert'] = "technical", title: str = "HYSo") -> PresentationReport:
    """Read all history pages and verified workflow results using the existing Manager.

    Unavailable/invalid artifacts stay explicit. No database state is changed.
    Retained execution artifacts use the experiment's recorded caller context.

    Args:
        manager (ExperimentManager): Open ExperimentManager instance used for existing history and artifacts. Required.
        experiment_id (str): Exact existing experiment identifier. Required.
        language (str): Presentation language: tr/en/de/es; free-form evidence text stays unchanged. Default: 'tr'.
        level (str): Existing beginner, technical or expert presentation detail level. Default: 'technical'.
        title (str): Human-readable report title. Default: 'HYSo'.

    Returns:
        The retained report experiment value, without recomputing measurements.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        report_experiment(manager, experiment_id)

    HYSo Guide: reporting.outputs
    Offline: hyso docs show reporting.outputs --lang en
    Open: hyso docs open reporting.outputs --lang en
    """
    if type(manager) is not ExperimentManager:
        raise ReportError("experiment_manager_required")
    from ..experiments.workflow import _REPORT_RESULT_KINDS
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
            if record["kind"] in _REPORT_RESULT_KINDS:
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
