"""Read-only export advice and escaped four-language reports."""
from __future__ import annotations
from typing import Any, Literal
import html
import json
from .contracts import ExportResult,ExportError

_TEXT={
 'en':dict(title='Deployment export report',evidence='Recorded evidence',advice='Evidence-based advice',
    precision='Check precision drift and labelled task quality independently.',
    missing='Measure comparable latency and throughput before choosing a faster backend.',
    faster='The candidate was faster under the recorded comparable conditions; this is not a general guarantee.',
    incompatible='Measurement conditions differ; no speedup recommendation is justified.',
    drift='Numerical validation failed. Inspect outputs and tolerances before deployment.',
    quality='Task quality degraded beyond the declared bounds.',
    labels='Labelled task quality has not been tested.',
    memory='Keep the admitted memory and timeout limits; unavailable memory is not zero.',
    portability='TensorRT engines are bound to recorded platform, runtime and GPU compatibility.',
    shapes='Validate every supported dynamic input profile; ONNX symbolic dimensions do not enforce bounds.',
    runtime='Resolve blocked dependencies/runtime stages before claiming deployment verification.'),
 'tr':dict(title='Deployment export raporu',evidence='Kaydedilmiş kanıt',advice='Kanıta bağlı öneriler',
    precision='Precision sapmasını ve etiketli görev kalitesini ayrı değerlendirin.',
    missing='Daha hızlı backend seçmeden önce karşılaştırılabilir latency ve throughput ölçün.',
    faster='Aday, kaydedilmiş karşılaştırılabilir koşullarda daha hızlıydı; bu genel bir garanti değildir.',
    incompatible='Ölçüm koşulları farklı; speedup önerisi desteklenmiyor.',
    drift='Sayısal doğrulama başarısız. Deployment öncesi çıktıları ve toleransları inceleyin.',
    quality='Görev kalitesindeki kayıp belirtilen sınırları aştı.',labels='Etiketli görev kalitesi test edilmedi.',
    memory='Kabul edilmiş bellek ve timeout sınırlarını koruyun; ölçülemeyen bellek sıfır değildir.',
    portability='TensorRT engine uyumluluğu kaydedilmiş platform, runtime ve GPU ile sınırlıdır.',
    shapes='Desteklenen her dinamik girdi profilini doğrulayın; ONNX sembolik boyutları sınırları uygulamaz.',
    runtime='Deployment doğrulaması ilan etmeden önce eksik bağımlılık/runtime aşamalarını çözün.'),
 'de':dict(title='Deployment-Exportbericht',evidence='Gespeicherte Nachweise',advice='Evidenzbasierte Empfehlungen',
    precision='Numerische Abweichung und beschriftete Aufgabenqualität getrennt prüfen.',
    missing='Vergleichbare Latenz und Durchsatz messen, bevor ein schnelleres Backend gewählt wird.',
    faster='Der Kandidat war unter den dokumentierten vergleichbaren Bedingungen schneller; keine allgemeine Garantie.',
    incompatible='Die Messbedingungen unterscheiden sich; keine belastbare Beschleunigungsempfehlung.',
    drift='Numerische Validierung fehlgeschlagen. Ausgaben und Toleranzen vor dem Einsatz prüfen.',
    quality='Der Qualitätsverlust überschreitet die festgelegten Grenzen.',labels='Aufgabenqualität mit Labels wurde nicht getestet.',
    memory='Zugelassene Speicher- und Zeitgrenzen beibehalten; nicht gemessener Speicher ist nicht null.',
    portability='TensorRT-Engines sind an dokumentierte Plattform-, Laufzeit- und GPU-Kompatibilität gebunden.',
    shapes='Jedes unterstützte dynamische Eingabeprofil prüfen; symbolische ONNX-Dimensionen erzwingen keine Grenzen.',
    runtime='Blockierte Abhängigkeiten und Laufzeitphasen vor einer Deployment-Freigabe beheben.'),
 'es':dict(title='Informe de exportación para despliegue',evidence='Evidencia registrada',advice='Recomendaciones basadas en evidencia',
    precision='Evalúe por separado la desviación numérica y la calidad con etiquetas.',
    missing='Mida latencia y rendimiento comparables antes de elegir un backend más rápido.',
    faster='El candidato fue más rápido en las condiciones comparables registradas; no es una garantía general.',
    incompatible='Las condiciones difieren; no se justifica recomendar una aceleración.',
    drift='La validación numérica falló. Revise salidas y tolerancias antes del despliegue.',
    quality='La pérdida de calidad supera los límites declarados.',labels='La calidad con etiquetas no se ha probado.',
    memory='Mantenga los límites admitidos de memoria y tiempo; la memoria no medida no equivale a cero.',
    portability='Los motores TensorRT dependen de la compatibilidad registrada de plataforma, runtime y GPU.',
    shapes='Valide cada perfil dinámico admitido; las dimensiones simbólicas ONNX no imponen límites.',
    runtime='Resuelva las dependencias y fases de runtime bloqueadas antes de declarar el despliegue verificado.')}


def _body(result):
    if isinstance(result,ExportResult):result.verify();return result.to_dict()
    if type(result) is not dict or result.get('format') not in ('hyso.export','hyso.export.validation','hyso.export.benchmark','hyso.export.comparison','hyso.export.advice'):
        raise ExportError('RESULT_SCHEMA','supported saved export result required')
    if result['format']=='hyso.export':
        verified=ExportResult(result);verified.verify();return verified.to_dict()
    if result['format']=='hyso.export.comparison':
        from .deployment import compare
        checked=compare(result['baseline'],result['candidate'])
        if any(result.get(k)!=checked.get(k) for k in ('comparable','speedup','latency_delta_ms','throughput_delta','status')):
            raise ExportError('RESULT_INTEGRITY','comparison differs from retained measurement conditions/samples')
    # Detached JSON only: no models or callbacks can execute through reporting.
    return json.loads(json.dumps(result,allow_nan=False))


def advise(result: ExportResult | dict, *, language: str='en') -> dict[str,Any]:
    """Manual evidence-bound suggestions; never edit a model or run a workload.

    Args:
        result (ExportResult | dict): Retained HYSo result/evidence record; rendering/advice does not
            run measurements. Required.
        language (str): Presentation language: tr, en, de or es. Technical keys/status codes remain
            stable. Default: 'en'.

    Returns:
        Detached manual recommendations with model_modified=False and measurements_executed=False.

    Raises:
        ExportError: Unsupported input, dependency, integrity, resource or evidence requirements fail.
        OSError: Explicit artifact or report file access fails.
    """
    if language not in _TEXT:raise ExportError('LANGUAGE','tr/en/de/es required')
    row=_body(result);keys=['precision','memory','shapes']
    if row.get('format')=='hyso.export.comparison':
        keys.append('incompatible' if not row.get('comparable') else 'faster' if row.get('speedup',0)>1 else 'missing')
    else:keys.append('missing')
    if row.get('numerical',{}).get('status')=='FAIL':keys.append('drift')
    quality=row.get('quality',{})
    if quality.get('status')=='FAIL':keys.append('quality')
    elif quality.get('status')!='PASS':keys.append('labels')
    if any(v.get('status')=='BLOCKED' for v in row.get('stages',{}).values()):keys.append('runtime')
    if 'tensorrt' in json.dumps(row).lower():keys.append('portability')
    return dict(format='hyso.export.advice',schema_version=1,language=language,
                recommendations=[dict(code='export.'+k,text=_TEXT[language][k]) for k in keys],
                source_format=row['format'],model_modified=False,measurements_executed=False)


def render(result: ExportResult | dict, *, format: Literal['json','markdown','html']='markdown', language: str='en') -> str:
    """Render detached records. Technical field names/status codes remain stable.

    Args:
        result (ExportResult | dict): Retained HYSo result/evidence record; rendering/advice does not
            run measurements. Required.
        format (Literal['json', 'markdown', 'html']): Output representation; use the choices declared
            for this renderer. Default: 'markdown'.
        language (str): Presentation language: tr, en, de or es. Technical keys/status codes remain
            stable. Default: 'en'.

    Returns:
        UTF-8-compatible report text; this function does not write a file.

    Raises:
        ExportError: Unsupported input, dependency, integrity, resource or evidence requirements fail.
        OSError: Explicit artifact or report file access fails.
    """
    if language not in _TEXT:raise ExportError('LANGUAGE','tr/en/de/es required')
    row=_body(result);advice=advise(row,language=language);text=_TEXT[language]
    if format=='json':return json.dumps(dict(language=language,title=text['title'],evidence=row,advice=advice),ensure_ascii=False,allow_nan=False,indent=2)+'\n'
    raw=json.dumps(row,ensure_ascii=False,allow_nan=False,indent=2)
    if format=='markdown':
        fence='`'*(max([len(s) for s in __import__('re').findall(r'`+',raw)]+[2])+1)
        return '# '+text['title']+'\n\n'+text['advice']+'\n\n'+'\n'.join('- '+v['text'] for v in advice['recommendations'])+'\n\n'+text['evidence']+'\n\n'+fence+'json\n'+raw+'\n'+fence+'\n'
    if format=='html':return '<!doctype html><html lang="'+language+'"><meta charset="utf-8"><title>'+html.escape(text['title'])+'</title><body><h1>'+html.escape(text['title'])+'</h1><h2>'+html.escape(text['advice'])+'</h2><ul>'+''.join('<li>'+html.escape(v['text'])+'</li>' for v in advice['recommendations'])+'</ul><h2>'+html.escape(text['evidence'])+'</h2><pre>'+html.escape(raw)+'</pre></body></html>'
    raise ExportError('REPORT_FORMAT','json/markdown/html required')

def save(result: ExportResult | dict, path: str, *, overwrite: bool=False) -> str:
    """Save a detached, digested deployment record with no-clobber publication.

    Args:
        result (ExportResult | dict): Retained HYSo result/evidence record; rendering/advice does not
            run measurements. Required.
        path (str): Filesystem location read or written by this explicit operation; output parents must
            already exist. Required.
        overwrite (bool): Permit replacement of an existing destination when True; False retains
            no-clobber publication. Default: False.

    Returns:
        The serialized record text written to path with a canonical evidence digest.

    Raises:
        ExportError: Unsupported input, dependency, integrity, resource or evidence requirements fail.
        OSError: Explicit artifact or report file access fails.
    """
    from ..analysis._codec import digest
    from ..benchmark.result import _write_export
    row=_body(result)
    text=json.dumps(dict(format='hyso.export.record',schema_version=1,evidence=row,content_sha256=digest(row)),
                    ensure_ascii=False,allow_nan=False,indent=2)+'\n'
    if len(text.encode('utf-8'))>16*1024**2:raise ExportError('RESOURCE_BUDGET','saved deployment record exceeds 16 MiB')
    _write_export(path,text,overwrite);return text

def load(path: str) -> dict[str,Any]:
    """Read a digested record; never load/execute its runtime artifact.

    Args:
        path (str): Filesystem location read or written by this explicit operation; output parents must
            already exist. Required.

    Returns:
        Detached validated evidence dictionary; loading metadata never executes a model.

    Raises:
        ExportError: Unsupported input, dependency, integrity, resource or evidence requirements fail.
        OSError: Explicit artifact or report file access fails.
    """
    from pathlib import Path
    from ..analysis._codec import digest
    p=Path(path)
    if p.is_symlink() or not p.is_file() or p.stat().st_size>16*1024**2:raise ExportError('RESULT_INTEGRITY','bounded regular saved record required')
    row=json.loads(p.read_text(encoding='utf-8'))
    if row.get('format')!='hyso.export.record' or row.get('schema_version')!=1 or row.get('content_sha256')!=digest(row.get('evidence')):
        raise ExportError('RESULT_INTEGRITY','deployment record digest/schema mismatch')
    return _body(row['evidence'])
