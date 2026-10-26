"""Optional P14 reader for Nsight autograd NVTX + CUDA activity exports.

Recorded push/pop operator nesting owns the HOST launch, never a nearby GPU
timestamp. GPU kernels join that launch by correlationId and process identity.
Only the unique innermost operator owns a kernel; ancestors are context only.
"""
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import uuid


def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def _contains(a,b):return a['start']<=b['start'] and b['end']<=a['end']


def _launch_owner(operators,launch):
    """Exact lexical NVTX stack on the launch thread, no nearest-time fallback."""
    same=[o for o in operators if o['globalTid']==launch['globalTid']]
    if any(o['start']<launch['end'] and launch['start']<o['end'] and not _contains(o,launch) for o in same):
        return None,[],'operator_boundary_crosses_launch'
    chain=sorted((o for o in same if _contains(o,launch)),key=lambda o:(o['start'],-o['end']))
    if not chain:return None,[],'no_recorded_operator_stack'
    for a,b in zip(chain,chain[1:]):
        if not _contains(a,b) or (a['start'],a['end'])==(b['start'],b['end']):
            return None,[],'ambiguous_or_crossing_operator_ranges'
    if any(o['identity_status']!='recorded' for o in chain):
        return None,[],'missing_or_duplicate_operator_identity'
    return chain[-1]['nvtx_id'],[o['nvtx_id'] for o in chain],None


def analyze(sqlite_path,capture_path):
    path=Path(sqlite_path).resolve();receipt_path=Path(capture_path).resolve()
    before=_sha(path);receipt_hash=_sha(receipt_path)
    capture=json.loads(receipt_path.read_text(encoding='utf-8'))
    if (capture.get('format')!='hyso.performance.p14.nsys_capture' or capture.get('version')!=1
            or capture.get('status')!='completed' or capture.get('terminal_completed') is not True
            or type(capture.get('measured_forwards')) is not int or capture['measured_forwards']<1
            or capture.get('output_equal')!=[True]*capture['measured_forwards']):
        raise ValueError('missing completed/validated P14 capture receipt')
    token=capture['trace_token'];uuid.UUID(token);pid=capture['worker_pid']
    if type(pid) is not int or pid<1:raise ValueError('invalid capture PID')
    phases=[];operators=[];kernels=[];launches=[];diagnostics=[];invalid_operator_forwards=set()
    try:
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row
            required={
                'META_DATA_EXPORT':{'name','value'},'StringIds':{'id','value'},
                'NVTX_EVENTS':{'start','end','text','textId','globalTid','eventType'},
                'CUPTI_ACTIVITY_KIND_KERNEL':{'start','end','deviceId','contextId','streamId','correlationId','globalPid','shortName'},
                'CUPTI_ACTIVITY_KIND_RUNTIME':{'start','end','globalTid','correlationId','nameId','returnValue'},
                'DIAGNOSTIC_EVENT':{'text','globalPid'},
                'TARGET_INFO_CUDA_CONTEXT_INFO':{'processId','deviceId','contextId'}}
            for table,columns in required.items():
                if not columns<={r[1] for r in db.execute(f'PRAGMA table_info("{table}")')}:
                    raise ValueError('missing Nsight schema: '+table)
            meta=dict(db.execute("SELECT name,value FROM META_DATA_EXPORT WHERE name IN "
                "('EXPORT_PRODUCT_NAME','EXPORT_PRODUCT_VERSION','EXPORT_SCHEMA_VERSION')"))
            if meta.get('EXPORT_PRODUCT_NAME')!='NVIDIA Nsight Systems' or not meta.get('EXPORT_SCHEMA_VERSION','').startswith('3.'):
                raise ValueError('unsupported Nsight export')
            diagnostics=[dict(r) for r in db.execute('SELECT * FROM DIAGNOSTIC_EVENT')]
            if not any('PyTorch autograd_nvtx injection initialized successfully' in r['text']
                    and ((r['globalPid']>>24)&0xffffff)==pid for r in diagnostics):
                raise ValueError('no matching native autograd NVTX injection evidence')
            names=dict(db.execute('SELECT id,value FROM StringIds'))
            ranges=[]
            for r in db.execute('SELECT rowid AS nvtx_id,* FROM NVTX_EVENTS'):
                row=dict(r);row['label']=r['text'] or names.get(r['textId'],'')
                if ((r['globalTid']>>24)&0xffffff)==pid:ranges.append(row)
            for i in range(capture['measured_forwards']):
                matched=[r for r in ranges if r['label']==f'hyso.p14/{token}/forward/{i}']
                if len(matched)!=1:raise ValueError('missing or duplicate forward marker')
                phase=matched[0]
                if phase['eventType']!=59 or phase['end'] is None or phase['end']<=phase['start']:
                    raise ValueError('incomplete forward push/pop range')
                if phases and phase['start']<phases[-1]['end']:raise ValueError('overlapping forwards')
                phase['forward_index']=i;phases.append(phase)
            for phase in phases:
                for r in ranges:
                    if not r['label'].startswith('aten::') or r['globalTid']!=phase['globalTid']:continue
                    if r['start']>=phase['end'] or r['end'] is not None and r['end']<=phase['start']:continue
                    if r['end'] is None or r['end']<=r['start'] or not _contains(phase,r):
                        invalid_operator_forwards.add(phase['forward_index']);continue
                    match=re.fullmatch(r'(aten::.+?), op_id = (\d+)',r['label'])
                    operators.append(dict(nvtx_id=r['nvtx_id'],name=match[1] if match else r['label'],
                        op_id=int(match[2]) if match else None,start=r['start'],end=r['end'],globalTid=r['globalTid'],
                        forward_index=phase['forward_index'],identity_status='recorded' if match and r['eventType']==59 else 'unsupported'))
            ids=[o['op_id'] for o in operators]
            for op in operators:
                if ids.count(op['op_id'])!=1:op['identity_status']='unsupported'
            tables=['CUPTI_ACTIVITY_KIND_RUNTIME']
            driver_columns={r[1] for r in db.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_DRIVER)')}
            if required['CUPTI_ACTIVITY_KIND_RUNTIME']<=driver_columns:tables.append('CUPTI_ACTIVITY_KIND_DRIVER')
            for table in tables:
                for r in db.execute(f'SELECT rowid AS launch_id,* FROM {table}'):
                    if ((r['globalTid']>>24)&0xffffff)!=pid:continue
                    name=names.get(r['nameId'],'')
                    if not name.startswith(('cudaLaunchKernel','cuLaunchKernel')):continue
                    phase=[p for p in phases if p['globalTid']==r['globalTid'] and _contains(p,r)]
                    if len(phase)!=1:continue
                    launches.append(dict(r,table=table,name=name,forward_index=phase[0]['forward_index']))
            seen=set()
            for r in db.execute('SELECT rowid AS activity_id,* FROM CUPTI_ACTIVITY_KIND_KERNEL'):
                if ((r['globalPid']>>24)&0xffffff)!=pid:continue
                phase=[p for p in phases if _contains(p,r)]
                if len(phase)!=1:raise ValueError('GPU activity outside synchronized forward scope')
                if r['end']<=r['start']:raise ValueError('invalid GPU kernel duration')
                if not db.execute('SELECT 1 FROM TARGET_INFO_CUDA_CONTEXT_INFO WHERE processId=? '
                        'AND deviceId=? AND contextId=?',(pid,r['deviceId'],r['contextId'])).fetchone():
                    raise ValueError('GPU context identity missing')
                key=(r['globalPid'],r['deviceId'],r['contextId'],r['streamId'],r['correlationId'],r['start'],r['end'],r['shortName'])
                if key in seen:raise ValueError('duplicate GPU kernel activity')
                seen.add(key)
                candidates=[a for a in launches if type(r['correlationId']) is int and r['correlationId']>0 and a['correlationId']==r['correlationId']
                    and (a['globalTid'] & ~16777215)==r['globalPid']]
                owner=None;chain=[];reason='missing_or_ambiguous_launch_correlation';launch=None
                if len(candidates)==1:
                    launch=candidates[0]
                    if launch['returnValue']!=0:reason='failed_launch'
                    elif launch['end']<=launch['start']:reason='invalid_launch_interval'
                    elif launch['forward_index']!=phase[0]['forward_index']:reason='launch_forward_mismatch'
                    elif launch['forward_index'] in invalid_operator_forwards:reason='incomplete_operator_ranges'
                    else:owner,chain,reason=_launch_owner([o for o in operators if o['forward_index']==launch['forward_index']],launch)
                kernels.append(dict(activity_id=r['activity_id'],name=names[r['shortName']],
                    demangled_name=names.get(r['demangledName']) if 'demangledName' in r.keys() else names[r['shortName']],
                    forward_index=phase[0]['forward_index'],start_ns=r['start'],end_ns=r['end'],
                    duration_us=(r['end']-r['start'])/1000,device_id=r['deviceId'],context_id=r['contextId'],
                    stream_id=r['streamId'],global_pid=r['globalPid'],correlation_id=r['correlationId'],
                    launch_table=launch['table'] if launch else None,launch_id=launch['launch_id'] if launch else None,
                    launch_name=launch['name'] if launch else None,
                    launch_start_ns=launch['start'] if launch else None,launch_end_ns=launch['end'] if launch else None,
                    operator_nvtx_id=owner,operator_chain_nvtx_ids=chain,
                    status='mapped' if reason is None else 'unsupported',reason=reason))
    except (sqlite3.Error,KeyError,TypeError,UnicodeError) as error:
        raise ValueError('invalid or unsupported Nsight P14 evidence') from error
    if _sha(path)!=before or _sha(receipt_path)!=receipt_hash:raise ValueError('evidence changed during read')
    used={(k['launch_table'],k['launch_id']) for k in kernels if k['launch_id'] is not None}
    missing=[dict(table=a['table'],launch_id=a['launch_id'],correlation_id=a['correlationId'],forward_index=a['forward_index'],
        reason='no_exported_GPU_kernel_for_launch') for a in launches if (a['table'],a['launch_id']) not in used]
    for op in operators:
        owned=[k for k in kernels if k['status']=='mapped' and k['operator_nvtx_id']==op['nvtx_id']]
        uncertain=(not kernels or op['identity_status']!='recorded'
            or any(k['status']!='mapped' and k['forward_index']==op['forward_index'] for k in kernels)
            or any(a['forward_index']==op['forward_index'] for a in missing))
        op.update(exclusive_kernel_count=None if uncertain else len(owned),
            exclusive_kernel_us=None if uncertain else sum(k['duration_us'] for k in owned),
            attribution_status='unsupported' if uncertain else 'measured' if owned else 'no_direct_kernel_observed')
    summary=[]
    for name in sorted({o['name'] for o in operators}):
        calls=[o for o in operators if o['name']==name]
        available=all(o['exclusive_kernel_count'] is not None for o in calls)
        summary.append(dict(name=name,count=len(calls),attribution_status='observed' if available else 'unsupported',
            exclusive_kernel_count=sum(o['exclusive_kernel_count'] for o in calls) if available else None,
            exclusive_kernel_us=sum(o['exclusive_kernel_us'] for o in calls) if available else None))
    kernel_summary=[]
    for name in sorted({k['name'] for k in kernels}):
        selected=[k for k in kernels if k['name']==name]
        kernel_summary.append(dict(name=name,count=len(selected),total_kernel_us=sum(k['duration_us'] for k in selected)))
    mapped=sum(k['status']=='mapped' for k in kernels)
    complete=bool(kernels) and mapped==len(kernels) and not missing and bool(operators)
    return dict(format='hyso.performance.operator_profile.nsys',schema_version=1,feature_ids=['P14'],
        status='complete' if complete else 'partial',source_sqlite=str(path),source_sqlite_sha256=before,
        source_capture=str(receipt_path),source_capture_sha256=receipt_hash,capture=capture,export_metadata=meta,
        kernel_count=len(kernels) if kernels else None,total_kernel_us=sum(k['duration_us'] for k in kernels) if kernels else None,
        cuda_activity_status='measured' if kernels else 'unsupported',mapped_kernel_count=mapped,
        attribution_status='complete' if complete else 'unsupported' if not mapped else 'partial',
        kernels=kernels,operators=operators,operator_summary=summary,kernel_summary=kernel_summary,
        missing_kernel_launches=missing,diagnostics=diagnostics,
        attribution_rule='same_PID_correlation_ID_to_CUDA_launch; recorded_same_thread_NVTX_push_pop_stack; unique_innermost_op_owns_kernel; ancestors_context_only',
        limitations=['GPU duration uses CUPTI kernel start/end only, never host API, NVTX or CUDA Event duration.',
            'Sum of kernel durations is instrumented activity cost, not inference latency or GPU busy union.',
            'Only captured synchronous ATen forward scope; no nearest-time matching or missing-event extrapolation.',
            'Profiler perturbation is unmeasured; instrumented forward wall is not a baseline or kernel time.'])
