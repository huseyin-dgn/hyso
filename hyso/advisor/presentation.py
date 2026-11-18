"""Deterministic bilingual views; never authoritative evidence or engine input."""
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re

from ._json import AdvisorError, MAX_BYTES, canonical
from ._locales import BOUNDARIES, LOCALIZATION_VERSION, RULES, STATUS, TEXT
from .contracts import AdvisorReport
from .planning import PlannedFinding
from .resolver import NativeMetric as M, NativeObservation

_METRICS = {
    M.FORWARD_P50: ('Model P50 gecikmesi', 'Resident model P50'),
    M.FORWARD_P95: ('Model P95 gecikmesi', 'Resident model P95'),
    M.FORWARD_P99: ('Model P99 gecikmesi', 'Resident model P99'),
    M.E2E_P50: ('Uçtan uca P50 gecikmesi', 'End-to-end P50'),
    M.E2E_P95: ('Uçtan uca P95 gecikmesi', 'End-to-end P95'),
    M.E2E_P99: ('Uçtan uca P99 gecikmesi', 'End-to-end P99'),
    M.COMPLETED_IMAGES_PER_SECOND: ('Tamamlanan görüntü hızı', 'Completed image throughput'),
    M.PRECISION_MEAN: ('Precision noktasında ortalama model gecikmesi', 'Precision-point mean model latency'),
    M.PRECISION_THROUGHPUT: ('Precision noktasında tamamlanan görüntü hızı', 'Precision-point completed throughput'),
    M.RSS_BASELINE: ('Başlangıç süreç RSS kullanımı', 'Baseline process RSS'),
    M.RSS_CURRENT: ('Mevcut süreç RSS kullanımı', 'Current process RSS'),
    M.RSS_SAMPLED_PEAK: ('Örneklenen tepe süreç RSS kullanımı', 'Sampled peak process RSS'),
    M.RSS_LIFETIME_HWM: ('Süreç ömrü boyunca tepe RSS', 'Lifetime process RSS high-water mark'),
    M.PARAMETER_BYTES: ('Model parametrelerinin mantıksal boyutu', 'Model parameter logical footprint'),
    M.BUFFER_BYTES: ('Model buffer mantıksal boyutu', 'Model buffer logical footprint'),
    M.MODEL_LOGICAL_BYTES: ('Modelin mantıksal boyutu; süreç RAM kullanımı değildir', 'Model logical footprint; not process RAM'),
    M.ARTIFACT_BYTES: ('Kaydedilmiş model dosyasının boyutu', 'Serialized model artifact size'),
    M.TEST_ACCURACY: ('Test doğruluğu', 'Test accuracy'),
    M.TEST_MACRO_F1: ('Test macro F1', 'Test macro F1'),
}
_KIND = {
    'OBSERVATION': ('Ölçüm gözlemi', 'Measurement observation'),
    'MEASURE_NEXT': ('Sonraki kanıt incelemesi', 'Next evidence review'),
    'EXPERIMENT_CANDIDATE': ('Koşullu deney adayı', 'Conditional experiment candidate'),
    'HYPOTHESIS': ('Olası açıklama; nedensel teşhis değildir', 'Hypothesis; not a causal diagnosis'),
    'DIAGNOSIS': ('Kaynakta belirtilen teşhis', 'Source diagnosis'),
    'VERIFIED_IMPROVEMENT': ('Kaynakta belirtilen doğrulama', 'Source verification'),
}
_SUMMARY_METRICS = {M.FORWARD_P95, M.E2E_P95, M.COMPLETED_IMAGES_PER_SECOND,
                    M.PRECISION_MEAN, M.RSS_SAMPLED_PEAK, M.MODEL_LOGICAL_BYTES,
                    M.ARTIFACT_BYTES, M.TEST_ACCURACY, M.TEST_MACRO_F1}


def _options(language, audience):
    if type(language) is not str or language not in TEXT:
        raise AdvisorError('unsupported_language')
    if type(audience) is not str or audience not in ('summary', 'technical'):
        raise AdvisorError('unsupported_audience')


def build_display_model(report):
    """One detached, language-neutral structured snapshot for every view.

    Contract validation is not native evidence revalidation. No source files,
    engine, rule, device or factory are accessed by this presentation operation.
    """
    if type(report) is not AdvisorReport:
        raise AdvisorError('typed_advisor_report_required')
    source = AdvisorReport.from_dict(report.to_dict())
    counts = Counter(f.kind.value for f in source.findings)
    metrics = [f.resolved_metric.to_dict() for f in source.findings if type(f) is NativeObservation]
    findings = []
    for f in source.findings:
        findings.append(dict(finding_id=f.finding_id, kind=f.kind.value, rule_id=f.rule_id,
                             rule_version=f.rule_version, evidence_support=f.evidence_support.value if f.evidence_support else None,
                             verification=f.verification, evidence=[r.to_dict() for r in f.evidence],
                             plan=f.plan.to_dict() if type(f) is PlannedFinding else None,
                             original_technical_record=f.to_dict()))
    evaluations = {g.goal_id:g for g in source.goal_evaluations}
    return dict(source_advisor_report_id=source.report_id,
                source_analysis_report_id=source.source_report_id,
                source_analysis_canonical_sha256=source.source_report_sha256,
                request_id=source.request_id, mode=source.request.mode,
                priority=source.request.priority.value if source.request.priority else None,
                counts={k:counts[k] for k in _KIND}, metrics=metrics, findings=findings,
                goals=[dict(goal=g.to_dict(), evaluation=evaluations[g.goal_id].to_dict() if g.goal_id in evaluations else None)
                       for g in source.request.goals],
                consistency=source.request_validation.consistency.value if source.request_validation else None,
                rule_versions=[r.to_dict() for r in source.rule_versions],
                original_risks=list(source.risks),
                original_evidence_limitations=list(source.evidence_limitations),
                source_text_policy='original_technical_record_not_translated',
                original_technical_record=source.to_dict())


def _number(value, language, *, rate=False, integer=False):
    if rate:
        value = format(value, ',.2f')
    elif integer:
        value = format(value, ',.0f')
    else:
        value = format(value, '.12g')
    if language == 'tr':
        whole, dot, decimals = value.partition('.')
        return whole.replace(',', '.') + (',' + decimals if dot else '')
    return value


def _value(value, unit, language, *, accuracy=False):
    if accuracy:
        number = _number(value * 100, language)
        return '%' + number if language == 'tr' else number + '%'
    units = {'images/s': ('görüntü/sn', 'images/s'), 'byte': ('bayt', 'bytes'),
             'fraction': ('oran', 'fraction'), 'frames/s': ('kare/sn', 'frames/s')}
    displayed = _number(value, language, rate=unit == 'images/s', integer=unit == 'byte')
    label = units.get(unit, (unit, unit))[0 if language == 'tr' else 1]
    return displayed + ' ' + label


def _status(code, language):
    key = code or 'not_evaluated'
    if key not in STATUS:
        raise AdvisorError('unsupported_display_status')
    return STATUS[key][0 if language == 'tr' else 1]


def _localized(data, language, audience):
    t = TEXT[language]; index = 0 if language == 'tr' else 1
    metric_rows = []
    for m in data['metrics']:
        identity = M(m['metric'])  # Closed typed identity; never guess from path/text.
        if identity not in _METRICS:
            raise AdvisorError('unsupported_display_metric')
        context = json.loads(m['native_context_json'])
        parts = []
        if context.get('device') is not None:
            parts.append(str(context['device']).upper())
        if context.get('batch') is not None:
            parts.append('batch ' + str(context['batch']))
        if context.get('precision') is not None:
            parts.append(str(context['precision']).upper())
        metric_rows.append(dict(metric=identity.value, record_id=m['reference']['record_id'],
            metric_path=m['reference']['metric_path'], label=_METRICS[identity][index],
            value=_value(m['value'], m['reference']['unit'], language, accuracy=identity==M.TEST_ACCURACY),
            scope=', '.join(parts), summary=identity in _SUMMARY_METRICS))
    by_ref = {(m['reference']['record_id'],m['reference']['metric_path']):m for m in data['metrics']}
    goal_rows = []
    for row in data['goals']:
        g,e = row['goal'],row['evaluation']
        values = []
        if e:
            for ref in e['evidence']:
                m=by_ref.get((ref['record_id'],ref['metric_path']))
                if m:
                    values.append(dict(record_id=ref['record_id'], value=_value(m['value'],ref['unit'],language,
                        accuracy=m['metric']==M.TEST_ACCURACY.value)))
        goal_rows.append(dict(goal_id=g['goal_id'],
            target=g['operator']+' '+_value(g['target'],g['unit'],language,accuracy=g['metric']=='accuracy'),
            constraint=t[g['constraint']], values=values,
            input=_status(e['input_validity'] if e else None,language),
            support=_status(e['evidence_support'] if e else None,language),
            observed=_status(e['observed'] if e else None,language),
            feasibility=_status(e['feasibility'] if e else None,language)))
    actions=[]; unknown=[]
    for f in data['findings']:
        rule,version=f['rule_id'],f['rule_version']
        if f['kind']=='OBSERVATION':
            if f['original_technical_record']['format']!='hyso.advisor.NativeObservation':
                unknown.append(dict(finding_id=f['finding_id'],message=t['unknown_metric']))
            continue
        known = (version=='1' and rule in RULES and f['plan'] is not None and
                 (f['kind']=='MEASURE_NEXT' or f['kind']=='EXPERIMENT_CANDIDATE' and
                  rule=='plan.batch-throughput' and f['plan']['change_parameter']=='batch_size'))
        if known:
            action,why,missing = RULES[rule][language]
            if f['kind']=='EXPERIMENT_CANDIDATE':
                action=('Önkoşullar sağlanırsa yalnız batch boyutunu değiştiren sınırlı bir deney değerlendirin.' if language=='tr'
                        else 'Consider a bounded, single-variable batch-size experiment only after prerequisites are resolved.')
            actions.append(dict(finding_id=f['finding_id'], rule_id=rule, translation_status='localized_rule_v1',
                label=_KIND[f['kind']][index], action=action, why=why, missing=missing,
                authorization=t['not_authorized'], candidate=f['kind']=='EXPERIMENT_CANDIDATE',
                goal_ids=f['plan']['goal_ids'] if f['plan'] else []))
        elif rule in ('native-observation','record-local-goal') and version=='1' and f['kind']=='MEASURE_NEXT':
            actions.append(dict(finding_id=f['finding_id'],rule_id=rule,translation_status='localized_rule_v1',
                label=t['review'], action=('İlgili kayıttaki kapsamı, örnek yeterliliğini ve native ilişkileri doğrulayın.' if language=='tr'
                else 'Check the source record scope, sample sufficiency and native associations.'),
                why=t['local_goal'],missing=t['partial'],authorization=t['not_authorized'],candidate=False,goal_ids=[]))
        else:
            unknown.append(dict(finding_id=f['finding_id'],message=t['unknown_rule'],
                                rule_id=rule,rule_version=version))
    risks=[BOUNDARIES[r][index] for r in data['original_risks'] if r in BOUNDARIES]
    return dict(title=t['title'], labels=t, metrics=metric_rows, goals=goal_rows, actions=actions,
                unknown_records=unknown, risks=risks,
                consistency=_status(data['consistency'],language),
                kind_labels={k:v[index] for k,v in _KIND.items()})


def to_display_dict(report, *, language='en', audience='summary'):
    _options(language,audience)
    data=build_display_model(report)
    body=dict(format='hyso.advisor.display',schema_version=1,presentation_only=True,
              authoritative_evidence=False,localization_version=LOCALIZATION_VERSION,
              language=language,audience=audience,source_advisor_report_id=report.report_id,
              data=data,text=_localized(data,language,audience))
    result=dict(body,display_content_sha256=hashlib.sha256(canonical(body).encode('utf-8')).hexdigest())
    if len(canonical(result).encode('utf-8'))>MAX_BYTES:
        raise AdvisorError('display_size_limit')
    return result


def _export(path,raw):
    if len(raw.encode('utf-8'))>MAX_BYTES:
        raise AdvisorError('display_size_limit')
    if path is not None:
        with Path(path).open('x',encoding='utf-8',newline='\n') as output:
            output.write(raw)
    return raw


def to_display_json(report,path=None,*,language='en',audience='summary'):
    return _export(path,canonical(to_display_dict(report,language=language,audience=audience))+'\n')


def _esc(value):
    text=html.escape(str(value),quote=False).replace('\r',' ').replace('\n',' ')
    return re.sub(r'([\\`*_\[\]()!#|])',r'\\\1',text)


def to_markdown(report,path=None,*,language='en',audience='summary'):
    view=to_display_dict(report,language=language,audience=audience)
    data,words=view['data'],view['text']; t=words['labels']
    lines=['# '+words['title'],'',t['authority'],'',t['source']+': `'+data['source_advisor_report_id']+'`','']
    lines+=['## '+t['performance'],'']
    selected=[m for m in words['metrics'] if audience=='technical' or m['summary']]
    lines += ['- '+_esc(m['label'])+': '+_esc(m['value'])+(' — '+_esc(m['scope']) if m['scope'] else '') for m in selected]
    if not selected:lines.append(t['no_metrics'])
    lines+=['',t['separate'],'','## '+t['goals'],'']
    if not words['goals']:lines.append(t['no_goals'])
    for g in words['goals']:
        lines+=['- '+_esc(g['goal_id'])+': '+_esc(g['target'])+' ('+g['constraint']+').',
                '  - '+t['observed']+': '+g['observed']+'; '+t['support']+': '+g['support']+'.',
                '  - '+t['input']+': '+g['input']+'; '+t['future']+': '+g['feasibility']+'.']
        for value in g['values']:lines.append('  - '+t['value']+': '+_esc(value['value'])+'; `'+value['record_id']+'`.')
    if words['goals']:lines+=['',t['local_goal'],t['consistency']+': '+words['consistency']+'.']
    lines+=['','## '+t['findings'],'',t['counts']+': '+', '.join(words['kind_labels'][k]+'='+str(v) for k,v in data['counts'].items())+'.']
    if not data['counts']['EXPERIMENT_CANDIDATE']:lines+=['',t['no_candidate']]
    if any(b['authority']!='native' for m in data['metrics'] for b in m['reference']['bindings']):lines+=['',t['partial']]
    for u in words['unknown_records']:lines+=['',_esc(u['message'])+' (`'+u['finding_id']+'`).']
    lines+=['','## '+t['actions'],'']
    if not words['actions']:lines.append(t['no_actions'])
    for i,a in enumerate(words['actions'],1):lines.append(str(i)+'. '+a['action']+' '+a['authorization'])
    lines+=['','## '+t['why'],'']
    for a in words['actions']:lines.append('- '+a['why'])
    lines+=['','## '+t['missing'],'']
    lines+=['- '+m for m in dict.fromkeys(a['missing'] for a in words['actions'])]
    if not words['actions']:lines.append(t['no_actions'])
    if any(a['candidate'] for a in words['actions']):
        lines+=['',t['change']+': batch.',t['fixed']+': '+t['experiment_fixed'],
                t['verify']+': '+t['experiment_verify'],t['accept']+': '+t['experiment_accept'],t['experiment_missing']]
    lines+=['','## '+t['risks'],'']+['- '+r for r in words['risks']]
    lines+=['- '+t['no_risk_translation'],'- '+t['unknown_effect'],'- '+t['unknown_cost']]
    if any(a['candidate'] for a in words['actions']):lines.append('- '+t['experiment_risks'])
    lines+=['','## '+t['execution'],'',t['not_executed'],t['not_authorized'],'']
    if audience=='technical':
        original=json.dumps(data['original_technical_record'],sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)
        longest=max((len(m.group()) for m in re.finditer(r'`+',original)),default=0)
        fence='`'*max(3,longest+1)
        lines+=['## '+t['technical'],'',t['technical_note'],t['bindings'],'',t['original'],'',fence+'json',original,fence,'']
    return _export(path,'\n'.join(lines))
