"""Bilingual views of two-model observations; reuse existing Advisor presentation."""
from __future__ import annotations
import typing as _typing

from typing import Any
from ._json import AdvisorError
from .resolver import NativeMetric
from .presentation import _METRICS, _number, _value, _esc, to_display_dict as advisor_display
from .engine import advise
from ..analysis._codec import export


def to_display_dict(result, *, language: _typing.Literal['tr', 'en', 'de', 'es'] = 'en') -> dict[str,Any]:
    """Translate presentation only; never change eligibility, evidence or policy."""
    from ..analysis.model_comparison import ModelComparisonResult, ModelComparisonSource, _selected_report
    if type(result) is not ModelComparisonResult:
        raise AdvisorError('typed_two_model_comparison_required')
    if language not in ('tr','en','de','es'):
        raise AdvisorError('unsupported_language')
    if language in ('de', 'es'):
        return _localized_two_model(result, language)
    tr = language=='tr'; data = result.to_dict(); index = 0 if tr else 1
    rows = []
    for row in data['metrics']:
        metric = row['metric']; difference = row['difference']
        p12_labels={'macs_per_batch': ('Batch başına analitik MACs', 'Analytic MACs per batch'),
                    'logical_flops_per_batch': ('Batch başına logical FLOPs', 'Logical FLOPs per batch')}
        label = (p12_labels[metric][index] if row['feature_id']=='P12' and metric in p12_labels else
                 _METRICS[NativeMetric(metric)][index] if metric is not None else row['feature_id'])
        view = dict(feature_id=row['feature_id'],metric=metric,label=label,status=row['status'],
            reasons=row['reasons'],baseline=None,candidate=None,difference=None,percent_change=None,
            percentage_point_change=None,preferred_direction=row.get('preferred_direction'),metric_effect=None,
            domain=row.get('domain'),unit=row.get('unit'),scope=row.get('scope'))
        if difference is None:
            text = 'Karşılaştırılabilir sayısal fark yok; eksik/uyumsuz kanıtın nedeni korunuyor.' if tr else 'No comparable numerical difference; missing or incompatible evidence reasons are retained.'
        else:
            view.update(baseline=difference['baseline']['value'],candidate=difference['candidate']['value'],
                difference=difference['difference'],percent_change=difference['percent_change'],
                percentage_point_change=difference['percentage_point_change'],metric_effect=difference['metric_effect'])
            effect = difference['metric_effect']
            text = ({'favorable':'B açısından bu ölçütte olumlu yön gözlendi.',
                     'unfavorable':'B açısından bu ölçütte olumsuz yön gözlendi.',
                     'unchanged':'Bu ölçütte değişim gözlenmedi.'} if tr else
                    {'favorable':'An observed favorable direction for B on this metric.',
                     'unfavorable':'An observed unfavorable direction for B on this metric.',
                     'unchanged':'No observed change on this metric.'})[effect]
            direction = {'increase': ('daha yüksek', 'higher'), 'decrease': ('daha düşük', 'lower'),
                         'unchanged': ('aynı', 'unchanged')}[difference['observed_direction']][index]
            text = (f"Model B ({data['sources']['candidate']['model_id']}): {label} {direction}; karşılaştırılan kayıtlı koşullar kapsamında. " if tr else
                    f"Model B ({data['sources']['candidate']['model_id']}): {direction} observed {label} under the compared recorded conditions. ") + text
            if row['feature_id']=='P10':
                text = ('Süreç genelindeki RSS farkıdır; modelin bellek üstünlüğünü göstermez.' if tr else
                        'Process-wide RSS difference; it does not show model-exclusive memory superiority.')
            if row['feature_id']=='P12':
                text += (' Analitik işlem sayısıdır; daha düşük gecikme, enerji tüketimi veya donanım işlem yükü kanıtı değildir.' if tr else
                         ' These are analytic operation counts; fewer operations do not establish lower latency, energy consumption or hardware work.')
            if difference['percent_change'] is None:
                text += ' Yüzdesel değişim tanımsızdır.' if tr else ' Relative percent is undefined.'
        view['explanation'] = text
        rows.append(view)
    comparison_observations = []
    for native, row in zip(data['metrics'], rows):
        if row['status'] != 'COMPARABLE' or native['difference'] is None:
            continue
        comparison_observations.append(dict(kind='observation', metric=row['metric'],
            statement=row['explanation'],
            guidance=('Model seçimini kendiniz yapmadan önce bu ölçütü kalite, gecikme, throughput ve bellek kanıtlarıyla birlikte inceleyin.' if tr else
                      'Review this metric alongside quality, latency, throughput and memory evidence before manually choosing a model.'),
            evidence=dict(comparison_sha256=data['content_sha256'],
                baseline=native['difference']['baseline']['reference'],
                candidate=native['difference']['candidate']['reference']),
            manual_only=True, automatically_applied=False))
    structure = data['structure']
    if structure['comparison'] is not None:
        for row in structure['comparison']['summary_differences']:
            if row['metric'] != 'parameter_elements' or row['status'] != 'computed' or row['difference'] is None:
                continue
            direction = ('daha fazla' if row['difference'] > 0 else 'daha az' if row['difference'] < 0 else 'aynı sayıda') if tr else (
                'more' if row['difference'] > 0 else 'fewer' if row['difference'] < 0 else 'the same number of')
            comparison_observations.append(dict(kind='observation', metric=row['metric'],
                statement=(f'Model B, kayıtlı snapshot kapsamında {direction} parametre elemanına sahiptir.' if tr else
                           f'Model B has {direction} parameter elements within the retained snapshot scope.'),
                guidance=('Kapasite ve mantıksal depolama ihtiyacını ayrıca değerlendirin; bu sayı accuracy, çalışma süresi veya gerçek dosya boyutunu kanıtlamaz.' if tr else
                          'Review capacity and logical storage needs separately; this count does not establish accuracy, runtime or actual artifact size.'),
                evidence=dict(comparison_sha256=data['content_sha256'],
                    structure_comparison_sha256=structure['comparison']['content_sha256'],
                    baseline=structure['comparison']['baseline'], candidate=structure['comparison']['candidate'],
                    association=structure['scope']), manual_only=True, automatically_applied=False))
    observations = {}
    for side,source in data['sources'].items():
        selected = _selected_report(ModelComparisonSource._restore(source),data['selection']['features'])
        observations[side] = advisor_display(advise(selected),language=language,audience='technical')
    return dict(format='hyso.two_model_comparison.display',schema_version=1,language=language,
        presentation_only=True,authoritative_evidence=False,source_comparison_sha256=data['content_sha256'],
        models={side:source['model_id'] for side,source in data['sources'].items()},metrics=rows,
        structure=data['structure'],advisor_observations=observations,
        comparison_observations=comparison_observations,
        summary=('Genel kazanan belirlenmedi. Farklar gözlemseldir; istatistiksel üstünlük veya kabul PASS sonucu değildir.' if tr else
                 'No overall winner. Differences are observational, not statistical superiority or acceptance PASS.'),
        limitations=data['limitations'])



def _localized_two_model(result, language):
    """Presentation-only translation. The original comparison data is untouched."""
    from .presentation import to_display_dict as advisor_display, _metric_label
    from .engine import advise
    from ..analysis.model_comparison import ModelComparisonSource, _selected_report
    base = to_display_dict(result, language='en')
    data = result.to_dict()
    de = language == 'de'
    def choose(a,b):
        return a if de else b
    for view, row in zip(base['metrics'], data['metrics']):
        metric = row['metric']
        if row['feature_id'] == 'P12' and metric in ('macs_per_batch','logical_flops_per_batch'):
            view['label'] = (choose('Analytische MACs pro Batch','MACs analíticas por lote') if metric=='macs_per_batch'
                             else choose('Logische FLOPs pro Batch','FLOPs lógicas por lote'))
        elif metric is not None:
            view['label'] = _metric_label(NativeMetric(metric), language)
        diff = row['difference']
        if diff is None:
            view['explanation'] = choose(
                'Kein vergleichbarer numerischer Unterschied; Gründe für fehlende oder inkompatible Nachweise bleiben erhalten.',
                'No hay diferencia numérica comparable; se conservan los motivos de evidencia ausente o incompatible.')
        elif row['feature_id'] == 'P10':
            view['explanation'] = choose(
                'Unterschied des gesamten Prozess-RSS; dies belegt keinen exklusiven Speichervorteil des Modells.',
                'Diferencia de RSS de todo el proceso; no demuestra una ventaja de memoria exclusiva del modelo.')
        else:
            effect = diff['metric_effect']
            outcomes = {
                'favorable': choose('günstige beobachtete Richtung','dirección favorable observada'),
                'unfavorable': choose('ungünstige beobachtete Richtung','dirección desfavorable observada'),
                'unchanged': choose('keine beobachtete Änderung','sin cambio observado'),
            }
            directions = {
                'increase': choose('höher','más alto'),
                'decrease': choose('niedriger','más bajo'),
                'unchanged': choose('unverändert','sin cambios'),
            }
            direction = directions[diff['observed_direction']]
            mid = data['sources']['candidate']['model_id']
            view['explanation'] = (
                f"Modell B ({mid}): {view['label']} {direction}, innerhalb der verglichenen aufgezeichneten Bedingungen; {outcomes[effect]}."
                if de else
                f"Modelo B ({mid}): {view['label']} {direction}, dentro de las condiciones registradas comparadas; {outcomes[effect]}.")
            if row['feature_id']=='P12':
                view['explanation'] += choose(
                    ' Dies sind analytische Operationszahlen; eine geringere Latenz, Energieaufnahme oder Hardwarelast ist damit nicht nachgewiesen.',
                    ' Son recuentos analíticos de operaciones; no demuestran menor latencia, consumo energético ni carga de hardware.')
            if diff['percent_change'] is None:
                view['explanation'] += choose(' Die prozentuale Änderung ist nicht definiert.',
                                              ' El cambio porcentual no está definido.')
    from .presentation import to_display_dict as advisor_display
    base['advisor_observations'] = {}
    for side, source in data['sources'].items():
        selected = _selected_report(ModelComparisonSource._restore(source),data['selection']['features'])
        base['advisor_observations'][side] = advisor_display(advise(selected),language=language,audience='technical')
    observations = []
    for row,native in zip(base['metrics'],data['metrics']):
        if native['status'] != 'COMPARABLE' or native['difference'] is None:
            continue
        observations.append(dict(kind='observation',metric=row['metric'],statement=row['explanation'],
            guidance=choose(
                'Prüfen Sie diese Kennzahl zusammen mit Nachweisen zu Qualität, Latenz, Durchsatz und Speicher, bevor Sie ein Modell selbst auswählen.',
                'Revise esta métrica junto con evidencia de calidad, latencia, rendimiento y memoria antes de elegir manualmente un modelo.'),
            evidence=dict(comparison_sha256=data['content_sha256'],
                baseline=native['difference']['baseline']['reference'],
                candidate=native['difference']['candidate']['reference']),
            manual_only=True,automatically_applied=False))
    struct=data['structure']
    if struct['comparison'] is not None:
        for item in struct['comparison']['summary_differences']:
            if item['metric'] != 'parameter_elements' or item['status'] != 'computed' or item['difference'] is None:
                continue
            delta=item['difference']
            d=(choose('mehr','más') if delta>0 else choose('weniger','menos') if delta<0 else choose('gleich viele','igual número de'))
            observations.append(dict(kind='observation',metric=item['metric'],
                statement=(f'Modell B hat im gespeicherten Snapshot {d} Parameterelemente.' if de else
                           f'El modelo B tiene {d} elementos de parámetros en la instantánea conservada.'),
                guidance=choose('Kapazität und logischen Speicherbedarf separat prüfen; die Anzahl belegt weder Genauigkeit noch Laufzeit oder tatsächliche Dateigröße.',
                                'Revise por separado la capacidad y el almacenamiento lógico; este número no demuestra precisión, tiempo de ejecución ni tamaño real de archivo.'),
                evidence=dict(comparison_sha256=data['content_sha256'],
                    structure_comparison_sha256=struct['comparison']['content_sha256'],
                    baseline=struct['comparison']['baseline'],candidate=struct['comparison']['candidate'],
                    association=struct['scope']),manual_only=True,automatically_applied=False))
    base['comparison_observations']=observations
    base['language']=language
    base['summary']=choose(
        'Kein Gesamtsieger bestimmt. Unterschiede sind Beobachtungen, kein Beleg für statistische Überlegenheit oder bestandene Abnahme.',
        'No se ha determinado un ganador global. Las diferencias son observaciones, no prueba de superioridad estadística ni aceptación aprobada.')
    # Raw technical limitation codes and evidence are deliberately retained verbatim.
    return base


def _comp_word(language, tr, en, de, es):
    return {'tr':tr, 'en':en, 'de':de, 'es':es}[language]

def to_markdown(result, path=None, *, language: _typing.Literal['tr', 'en', 'de', 'es'] = 'en', overwrite: bool = False) -> str:
    """Render native differences and existing per-model Advisor observations."""
    view = to_display_dict(result,language=language); tr = language=='tr'
    words = {
        'tr':('HYSo iki model karşılaştırması','Özellik / metrik','Durum','Mutlak fark B−A','Göreli değişim (%)','Açıklama'),
        'en':('HYSo two-model comparison','Feature / metric','Status','Absolute difference B−A','Relative change (%)','Explanation'),
        'de':('HYSo Zwei-Modell-Vergleich','Merkmal / Kennzahl','Status','Absoluter Unterschied B−A','Relative Änderung (%)','Erläuterung'),
        'es':('Comparación HYSo de dos modelos','Función / métrica','Estado','Diferencia absoluta B−A','Cambio relativo (%)','Explicación'),
    }[language]
    lines = ['# '+words[0],'',view['summary'],'',
             'A: '+_esc(view['models']['baseline'])+'; B: '+_esc(view['models']['candidate']),'',
             '| '+words[1]+' | '+words[2]+' | '+_comp_word(language,'Tercih edilen yön','Preferred direction','Bevorzugte Richtung','Dirección preferida')+' | A | B | '+words[3]+' | '+words[4]+' |',
             '|---|---|---|---:|---:|---:|---:|']
    original = result.to_dict()['metrics']
    explanations = []
    for row,native in zip(view['metrics'],original):
        unit = native.get('unit','')
        fmt = lambda value: '—' if value is None else _value(value,unit,language)
        percent = '—' if row['percent_change'] is None else _number(row['percent_change'],language)
        direction = row['preferred_direction'] or '—'
        direction = ({'lower':'düşük','higher':'yüksek'} if tr else
                     {'lower':'niedriger','higher':'höher'} if language=='de' else
                     {'lower':'menor','higher':'mayor'} if language=='es' else {}).get(direction,direction)
        lines.append('| '+_esc(row['feature_id']+' '+row['label'])+' | '+row['status']+' | '+direction+' | '+
            ' | '.join(_esc(fmt(row[key])) for key in ('baseline','candidate','difference'))+' | '+percent+' |')
        explanations += ['',_esc(row['label'])+': '+row['explanation'],
                  ' / '.join(_esc(reason) for reason in row['reasons'])]
        if row['percentage_point_change'] is not None:
            explanations.append(_comp_word(language,'Yüzde puan farkı: ','Percentage-point difference: ','Prozentpunktdifferenz: ','Diferencia en puntos porcentuales: ')+_number(row['percentage_point_change'],language))
    lines += explanations
    structure = view['structure']
    lines += ['', '## '+_comp_word(language,'Kayıtlı yapı bilgisi','Retained structure metadata','Gespeicherte Strukturmetadaten','Metadatos estructurales conservados'),'',structure['status']]
    if structure['comparison'] is not None:
        lines += ['', '| '+_comp_word(language,'Ölçüt','Metric','Kennzahl','Métrica')+' | A | B | B−A | % |','|---|---:|---:|---:|---:|']
        for row in structure['comparison']['summary_differences']:
            def number(v):return '—' if v is None else _number(v,language)
            lines.append('| '+_esc(row['metric'])+' | '+ ' | '.join(number(v) for v in (
                row['baseline']['value'],row['candidate']['value'],row['difference'],row['percent_change']))+' |')
        lines += ['',_comp_word(language,'Boyut ölçütlerinde düşük değer daha küçüktür; bu, kalite veya genel üstünlük değildir.','Lower size quantities mean smaller registered metadata, not quality or overall superiority.','Kleinere Größenwerte bedeuten kleinere erfasste Metadaten, nicht höhere Qualität oder Gesamtüberlegenheit.','Los valores de tamaño menores indican metadatos registrados menores, no calidad ni superioridad global.')]
    for side,observation in view['advisor_observations'].items():
        lines += ['', '## '+_comp_word(language,'Advisor gözlemleri','Advisor observations','Advisor-Beobachtungen','Observaciones de Advisor')+' — '+_esc(view['models'][side]),'']
        for metric in observation['text']['metrics']:
            lines.append('- '+_esc(metric['label'])+': '+_esc(metric['value']))
    lines += ['', '## '+_comp_word(language,'Sınırlar','Limits','Grenzen','Límites'),'']+['- '+_esc(note) for note in view['limitations']]
    lines += ['', 'SHA256: `'+view['source_comparison_sha256']+'`','']
    raw = '\n'.join(lines)
    if path is not None:export(path,raw,overwrite)
    return raw
