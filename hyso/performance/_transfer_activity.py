"""Optional, bounded P15 CUDA activity reader; never launches a profiler.

Supports inspected Nsight SQLite schema major 3 with the columns below and the
existing one-H2D/16-GEMM workload. CPU APIs/NVTX select phases only; all reported
durations come exclusively from GPU MEMCPY/KERNEL start/end columns.
"""
import hashlib
import json
from pathlib import Path
import sqlite3


def _sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def _union(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if type(start) is not int or type(end) is not int or end<=start:
            raise ValueError('invalid GPU activity interval')
        if merged and start<=merged[-1][1]: merged[-1][1]=max(end,merged[-1][1])
        else: merged.append([start,end])
    return merged


def _activity_metrics(copies,kernels):
    copy=_union(copies);compute=_union(kernels);intersection=[]
    i=j=0
    while i<len(copy) and j<len(compute):
        start=max(copy[i][0],compute[j][0]);end=min(copy[i][1],compute[j][1])
        if end>start: intersection.append([start,end])
        if copy[i][1]<=compute[j][1]: i+=1
        else: j+=1
    duration=lambda intervals:sum(end-start for start,end in intervals)
    busy=_union(copy+compute)
    return dict(copy_union_intervals_ns=copy,compute_union_intervals_ns=compute,
        overlap_intervals_ns=intersection,copy_active_ns=duration(copy),
        compute_active_ns=duration(compute),overlap_ns=duration(intersection),
        busy_union_ns=duration(busy),gpu_span_ns=busy[-1][1]-busy[0][0])


_GPU={'start','end','deviceId','contextId','streamId','correlationId','globalPid'}
_COLUMNS={
    'CUPTI_ACTIVITY_KIND_MEMCPY':_GPU|{'bytes','copyKind','srcKind','dstKind'},
    'CUPTI_ACTIVITY_KIND_KERNEL':_GPU|{'shortName'},
    'CUPTI_ACTIVITY_KIND_RUNTIME':{'start','end','globalTid','correlationId','returnValue'},
    'NVTX_EVENTS':{'start','end','text','textId','globalTid'},
    'StringIds':{'id','value'},'META_DATA_EXPORT':{'name','value'},
    'ENUM_CUDA_MEMCPY_OPER':{'id','name'},'ENUM_CUDA_MEM_KIND':{'id','name'},
    'TARGET_INFO_CUDA_CONTEXT_INFO':{'processId','deviceId','contextId'}}


def analyze_nsys_activity(result,sqlite_path):
    data=result.to_dict();config=data['protocol']['configuration']
    if not config.get('profile_capture'): raise ValueError('result has no external capture labels')
    comparison=data['raw_evidence']['overlap_probe']['comparison']
    token=comparison['trace_token'];pid=data['raw_evidence']['worker']['worker_pid']
    path=Path(sqlite_path).resolve();before=_sha256(path)
    try:
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row
            for table,required in _COLUMNS.items():
                if not required<={r[1] for r in db.execute(f'PRAGMA table_info("{table}")')}:
                    raise ValueError(f'missing Nsight GPU evidence schema: {table}')
            metadata=dict(db.execute("SELECT name,value FROM META_DATA_EXPORT WHERE name IN "
                "('EXPORT_PRODUCT_NAME','EXPORT_PRODUCT_VERSION','EXPORT_SCHEMA_VERSION')"))
            if (metadata.get('EXPORT_PRODUCT_NAME')!='NVIDIA Nsight Systems'
                    or not metadata.get('EXPORT_SCHEMA_VERSION','').startswith('3.')):
                raise ValueError('unsupported Nsight export version')
            enums=lambda table:dict(db.execute(f'SELECT name,id FROM {table}'))
            htod=enums('ENUM_CUDA_MEMCPY_OPER')['CUDA_MEMCPY_KIND_HTOD']
            memory=enums('ENUM_CUDA_MEM_KIND')
            pinned=memory['CUDA_MEMOPR_MEMORY_KIND_PINNED'];device=memory['CUDA_MEMOPR_MEMORY_KIND_DEVICE']
            names=dict(db.execute('SELECT id,value FROM StringIds'))
            markers={}
            prefix=f'hyso.p15/{token}/'
            for r in db.execute('SELECT rowid AS activity_id,* FROM NVTX_EVENTS'):
                label=r['text'] or names.get(r['textId'],'')
                if not label.startswith(prefix): continue
                if label in markers: raise ValueError('ambiguous duplicate NVTX phase')
                if ((r['globalTid']>>24)&0xffffff)!=pid: raise ValueError('capture PID mismatch')
                markers[label]=dict(r)
                markers[label]['label']=label
            expected={prefix+f'warmup/{i}/{mode}' for i in range(config['warmup_iterations'])
                for mode in ('serial','concurrent')}
            expected|={prefix+f'measure/{p["index"]}/{mode}' for p in comparison['pairs']
                for mode in ('serial','concurrent')}
            if set(markers)!=expected: raise ValueError('missing or unexpected P15 capture phases')
            phases=[];seen=set();gpu_scope=None;copy_stream=None;compute_stream=None;previous_end=None
            for pair in comparison['pairs']:
                for mode in pair['order']:
                    marker=markers[prefix+f'measure/{pair["index"]}/{mode}']
                    if marker['end'] is None or marker['end']<=marker['start']:
                        raise ValueError('incomplete NVTX phase')
                    if previous_end is not None and marker['start']<previous_end:
                        raise ValueError('overlapping or reordered measurement phases')
                    previous_end=marker['end'];activities=[]
                    for kind,table in (('H2D','CUPTI_ACTIVITY_KIND_MEMCPY'),('kernel','CUPTI_ACTIVITY_KIND_KERNEL')):
                        # The API correlation selects the phase, NOT the GPU interval.
                        query=f'''SELECT a.rowid AS activity_id,a.*,r.rowid AS runtime_id
                            FROM {table} a JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
                            ON a.correlationId=r.correlationId AND a.globalPid=(r.globalTid & ~16777215)
                            WHERE r.globalTid=? AND r.start>=? AND r.end<=? AND r.returnValue=0'''
                        for r in db.execute(query,(marker['globalTid'],marker['start'],marker['end'])):
                            key=(table,r['activity_id'])
                            if key in seen: raise ValueError('duplicate GPU activity attribution')
                            seen.add(key)
                            scope=(r['globalPid'],r['deviceId'],r['contextId'])
                            if gpu_scope is None: gpu_scope=scope
                            if (scope!=gpu_scope or scope[0]!=(marker['globalTid'] & ~16777215)
                                    or not marker['start']<=r['start']<r['end']<=marker['end']):
                                raise ValueError('GPU activity outside capture PID/device/context/phase')
                            if not db.execute('SELECT 1 FROM TARGET_INFO_CUDA_CONTEXT_INFO WHERE '
                                    'processId=? AND deviceId=? AND contextId=?',(pid,r['deviceId'],r['contextId'])).fetchone():
                                raise ValueError('missing GPU context identity')
                            a=dict(kind=kind,table=table,activity_id=r['activity_id'],runtime_id=r['runtime_id'],
                                start_ns=r['start'],end_ns=r['end'],stream_id=r['streamId'],
                                device_id=r['deviceId'],context_id=r['contextId'],
                                global_pid=r['globalPid'],correlation_id=r['correlationId'])
                            if kind=='H2D':
                                if (r['copyKind']!=htod or r['srcKind']!=pinned or r['dstKind']!=device
                                        or r['bytes']!=pair[mode]['transfer_bytes']):
                                    raise ValueError('missing matching pinned H2D activity')
                                a['bytes']=r['bytes']
                            else:
                                a['kernel_name']=names[r['shortName']]
                                if 'gemm' not in a['kernel_name'].lower():
                                    raise ValueError('unexpected non-GEMM kernel in comparison phase')
                            activities.append(a)
                    copies=[a for a in activities if a['kind']=='H2D'];kernels=[a for a in activities if a['kind']=='kernel']
                    if len(copies)!=1 or len(kernels)!=pair[mode]['compute_iterations']:
                        raise ValueError('incomplete one-copy/16-GEMM activity trace')
                    cs=copies[0]['stream_id'];ks={a['stream_id'] for a in kernels}
                    if copy_stream is None: copy_stream=cs
                    if cs!=copy_stream or len(ks)!=1 or ((cs in ks)!=(mode=='serial')):
                        raise ValueError('serial/concurrent GPU stream mapping mismatch')
                    if mode=='concurrent':
                        if compute_stream is None: compute_stream=next(iter(ks))
                        if ks!={compute_stream}: raise ValueError('compute stream changed')
                    metrics=_activity_metrics([(a['start_ns'],a['end_ns']) for a in copies],
                        [(a['start_ns'],a['end_ns']) for a in kernels])
                    phases.append(dict(pair_index=pair['index'],mode=mode,nvtx_label=marker['label'],
                        nvtx_activity_id=marker['activity_id'],activities=sorted(activities,key=lambda a:a['start_ns']),
                        **metrics))
    except (sqlite3.Error,KeyError,TypeError,UnicodeError) as error:
        raise ValueError('invalid or unsupported Nsight GPU activity evidence') from error
    if _sha256(path)!=before: raise ValueError('Nsight export changed during analysis')
    canonical=json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    return dict(format='hyso.performance.transfer.gpu_activity',schema_version=1,status='measured',
        source_result_id=data['result_id'],source_result_sha256=hashlib.sha256(canonical).hexdigest(),
        source_sqlite=str(path),source_sqlite_sha256=before,export_metadata=metadata,
        trace_token=token,worker_pid=pid,clock='Nsight common GPU activity timeline; ns',
        scope='measured_comparison_phases_only; warmup_preparation_validation_excluded',phases=phases,
        limitations=['Profiled timings are not uninstrumented speedup evidence.',
            'NVTX and CPU API intervals select activities only; durations use GPU start/end exclusively.',
            'Bounded phase completeness checked against one H2D and 16 GEMM kernels; no full-process trace completeness claim.',
            'Activity overlap does not establish resource utilization efficiency or a general optimization recommendation.'])
