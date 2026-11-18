"""Pure unified views of validated evidence; no optimization/recommendation rules."""
from dataclasses import dataclass
import html
import json

from ._codec import EvidenceError, canonical, digest, plain, read_json, export
from .evidence import EvidenceIndex, FEATURES

_SECTIONS = (
    ('Training and evaluation', ('P19','P20','P21','P22','P23','P24','P25','P26')),
    ('Inference latency and throughput', ('P01','P02','P03','P04','P05','P06','P07','P08')),
    ('Memory and footprint', ('P09','P10','P11')),
    ('FLOPs and profiling', ('P12','P13','P14','P18')),
    ('Quality, robustness and precision', ('P26','P27','P28','P29')),
    ('Hardware and runtime', ('P15','P16','P17','P34','P35')),
    ('Stability and energy', ('P30','P31','P32','P33')),
)
_EXPLANATIONS = (
    'Training captures are separate runs. A short instrumented epoch must not be substituted for historical full-training epoch time. Evaluation uses its recorded split and averaging policy.',
    'Latency is a batch forward duration. Throughput counts completed images in a wall window. E2E also includes acquisition, queueing and postprocessing; their clocks and denominators stay separate.',
    'Process RSS includes runtime and allocator effects. Registered tensor logical bytes, serialized file size and CUDA allocator counters describe different quantities, not one interchangeable memory total.',
    'Logical FLOPs/MACs describe the declared operation convention and observed coverage. Profiler time includes instrumentation. Payload estimates do not establish native DRAM traffic or a qualified hardware roof.',
    'Quality belongs to the recorded cohort and class mapping. Perturbation/precision experiments use paired samples. Percentage points differ from relative percent; observed Pareto is not independently verified improvement.',
    'Device and backend conditions must match before comparisons. Provider availability is not evidence that a graph node executed there. Missing devices and ambiguous node mappings remain explicit.',
    'A completed short window is not long-duration acceptance. RSS growth alone is not a causal memory leak. Energy requires actual compatible sensor samples, not a TDP or utilization estimate.',
)


@dataclass(frozen=True, init=False)
class AnalysisContext:
    """Caller declarations, explicitly not evidence measurements or attestations."""
    _payload: str

    def __init__(self, *, workload_id, task, model, dataset, training=None,
                 source_revision=None, references=()):
        if any(type(x) is not str or not x.strip() for x in (workload_id,task)):
            raise EvidenceError('invalid_context_identity')
        if type(model) is not dict or type(dataset) is not dict or (training is not None and type(training) is not dict):
            raise EvidenceError('invalid_context_summary')
        data=plain(dict(workload_id=workload_id,task=task,model=model,dataset=dataset,
                        training=training,source_revision=source_revision,references=references,
                        authority='caller_declared; not a measured or verified association'))
        object.__setattr__(self,'_payload',canonical(data))

    def to_dict(self): return json.loads(self._payload)

    @classmethod
    def _restore(cls,data):
        d=dict(data);d.pop('authority',None)
        result=cls(**d)
        if result.to_dict()!=data:raise EvidenceError('invalid_context')
        return result


def _derive(index, context, title, risks):
    if type(title) is not str or not title.strip():raise EvidenceError('invalid_report_title')
    if type(risks) not in (list,tuple) or any(type(s) is not str or not s.strip() for s in risks):
        raise EvidenceError('invalid_report_risks')
    records=index.to_dict()['records'];slots=[]
    for feature in FEATURES:
        items=[r for r in records if r['feature_id']==feature]
        validated=[r for r in items if r['validation']['status']=='native_validated']
        warnings=[]
        for r in validated:
            fields=[k for k,v in r['bindings'].items() if v!='native']
            if fields:warnings.append(dict(record_id=r['record_id'],unproven_associations=fields))
        if not items:status,reason='missing','No evidence was supplied; this is not a zero measurement.'
        elif validated:
            status='partial' if warnings or feature in ('P18','P29','P31','P32') else 'validated'
            reason='Native evidence validated; scope and claim limits remain explicit.'
        else:
            statuses={r['validation']['status'] for r in items}
            status=next(s for s in ('invalid','unknown','unsupported','not_run') if s in statuses)
            reason='; '.join(dict.fromkeys(r['validation']['reason'] for r in items))
        slots.append(dict(feature_id=feature,status=status,reason=reason,
                          record_ids=[r['record_id'] for r in items],
                          new_records=sum(r['origin']=='new' for r in items),
                          reused_records=sum(r['origin']=='reused' for r in items),
                          association_warnings=warnings))
    counts={s:sum(r['status']==s for r in slots) for s in ('validated','partial','missing','unsupported','not_run','invalid','unknown')}
    # Deliberately no merging, delta calculation, winner selection, execution
    # plan, resource mutation or recommendation generation.
    groups={}
    for r in records:
        group=digest(r['scope']);groups.setdefault(group,dict(scope=r['scope'],record_ids=[]))['record_ids'].append(r['record_id'])
    return dict(title=title,coverage=dict(total=35,counts=counts),feature_results=slots,
                sections=[dict(title=name,feature_ids=list(ids),explanation=text)
                          for (name,ids),text in zip(_SECTIONS,_EXPLANATIONS)],
                scope_groups=groups,risks=list(risks)+[
                    'B4 native risk remains OPEN; no device/training authorization is granted.',
                    'Native validation proves consistency, not source authenticity. Caller associations stay declared.',
                    'Process RSS, logical bytes, CUDA allocator and serialized bytes are separate domains.',
                    'Latency, throughput, E2E, training and inference are not interchangeable.',
                    'Missing evidence is never converted to zero.'],
                claims=dict(qualified_stable_roof='not_established_by_this_report',
                            independently_verified_improvement='not_certified',
                            long_run_7200_acceptance='not_certified',causal_memory_leak='not_proven'),
                recommendations=dict(status='not_implemented',items=[]))


@dataclass(frozen=True, init=False)
class UnifiedAnalysisReport:
    _payload: str

    def __init__(self,data):
        d=plain(data)
        if (type(d) is not dict or d.get('format')!='hyso.analysis_report'
                or type(d.get('schema_version')) is not int or d['schema_version']!=1):
            raise EvidenceError('unsupported_report_version')
        if set(d)!={'format','schema_version','report_id','context','index','analysis'}:
            raise EvidenceError('invalid_report_fields')
        idx=d['index']
        if set(idx)!={'format','schema_version','records'} or idx['format']!='hyso.evidence_index' or type(idx['schema_version']) is not int or idx['schema_version']!=1:
            raise EvidenceError('unsupported_index_version')
        from .evidence import EvidenceRecord
        index=EvidenceIndex(EvidenceRecord(r) for r in idx['records'])
        context=AnalysisContext._restore(d['context'])
        # Persist caller risks separately from the five immutable engine cautions.
        risks=d['analysis']['risks'][:-5]
        expected=_derive(index,context,d['analysis']['title'],risks)
        if expected!=d['analysis']:raise EvidenceError('report_derivation_mismatch')
        identity={k:v for k,v in d.items() if k!='report_id'}
        if d['report_id']!=digest(identity):raise EvidenceError('report_integrity_mismatch')
        object.__setattr__(self,'_payload',canonical(d))

    @classmethod
    def build(cls,index,*,context,title='HYSo unified analysis',risks=()):
        if type(index) is not EvidenceIndex or type(context) is not AnalysisContext:
            raise EvidenceError('typed_index_and_context_required')
        d=dict(format='hyso.analysis_report',schema_version=1,context=context.to_dict(),
               index=index.to_dict(),analysis=_derive(index,context,title,risks))
        d['report_id']=digest(d)
        return cls(d)

    def to_dict(self):return json.loads(self._payload)

    def to_json(self,path,*,overwrite=False):export(path,self._payload+'\n',overwrite)

    @classmethod
    def from_json(cls,path):
        d,_,_=read_json(path,max_bytes=128*1024**2)
        return cls(d)

    def to_markdown(self,path,*,overwrite=False):
        """Render human explanations and exact scalar references from this report."""
        d=self.to_dict();a=d['analysis'];c=d['context'];records=d['index']['records']
        def esc(v):return html.escape(str(v)).replace('|','\\|').replace('`','\\`').replace('[','&#91;').replace(']','&#93;').replace('\n',' ')
        lines=['# '+esc(a['title']),'',
               'Generated by HYSo UnifiedAnalysisReport. This is an evidence view, not an Advisor or an execution plan.','',
               '**Model and dataset (caller declarations)**','',
               'Workload: '+esc(c['workload_id'])+'; task: '+esc(c['task'])+'. Source revision: '+esc(c['source_revision']),
               '', '```json',json.dumps(dict(model=c['model'],dataset=c['dataset'],training=c['training']),ensure_ascii=False,indent=2).replace('`','\\u0060'), '```','',
               'These summaries are caller-declared. They do not silently certify missing native checkpoint, split or preprocessing bindings.','',
               '**Coverage and missing evidence**','',
               '| Feature | Evidence status | New | Reused | Explanation |','|---|---|---:|---:|---|']
        for s in a['feature_results']:
            lines.append('| '+s['feature_id']+' | '+s['status']+' | '+str(s['new_records'])+' | '+str(s['reused_records'])+' | '+esc(s['reason'])+' |')
        for section in a['sections']:
            lines += ['', '## '+section['title'],'',section['explanation'],'']
            for feature in section['feature_ids']:
                matching=[r for r in records if r['feature_id']==feature]
                if not matching:
                    lines += [feature+': no evidence supplied; no value was invented.',''];continue
                for r in matching:
                    lines += ['### '+feature+' — '+r['origin'],'']
                    if r['source'] is None:
                        lines += [esc(r['validation']['status'])+': '+esc(r['validation']['reason']),''];continue
                    lines += [esc(r['source']['result_type'])+'; result ID: '+esc(r['source']['result_id']),
                              '', 'Native result status: '+esc(r['source']['native_status'])+'; validation: native internal consistency. This is not an automatic acceptance PASS.',
                              '', 'Artifact: '+esc(r['artifact']['name'])+'; SHA-256: '+r['artifact']['sha256'],
                              '', 'Scope: '+esc(json.dumps(r['scope'],ensure_ascii=False)),
                              '', 'Association status: '+esc(json.dumps(r['bindings'],ensure_ascii=False)),
                              '', '| Native field | Value | Unit | Status | Domain |','|---|---:|---|---|---|']
                    for m in r['metrics'][:12]:
                        lines.append('| '+esc(m['path'])+' | '+('unknown' if m['value'] is None else esc(m['value']))+' | '+esc(m['unit'])+' | '+m['status']+' | '+m['domain']+' |')
                    if len(r['metrics'])>12:lines += ['',str(len(r['metrics'])-12)+' further scalar references are retained in JSON.']
                    lines += ['',*['- '+esc(v) for v in r['limitations']],'']
        lines += ['## Known risks and limits','',*['- '+esc(v) for v in a['risks']],'',
                  'Recommendations: NOT IMPLEMENTED. Observed Pareto is not independently verified improvement; short stability is not 7200-second acceptance.','',
                  'Separate scope groups: '+str(len(a['scope_groups']))+'. No cross-scope numerical merge was performed.','']
        export(path,'\n'.join(lines),overwrite)
