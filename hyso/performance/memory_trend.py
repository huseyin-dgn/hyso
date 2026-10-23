"""P32 descriptive memory trends and explicit weak-reference tensor witnesses."""
import math
import time
import uuid
import weakref

from .runtime_result import RuntimeResult, _result, _int, _number, _name


class RetentionTracker:
    """Opt-in observed objects, not a heap census. Never owns watched tensors."""
    def __init__(self, *, max_objects=1024):
        _int(max_objects,1,4096);self.max_objects=max_objects;self._entries={}

    def watch(self,tensor,*,label):
        import torch
        _name(label)
        if not isinstance(tensor,torch.Tensor) or tensor.layout!=torch.strided:raise TypeError('dense Tensor required')
        if label in self._entries:raise ValueError('duplicate retention label')
        if len(self._entries)>=self.max_objects:raise ValueError('retention object budget exceeded')
        if any(ref() is tensor for _,ref in self._entries.values()):raise ValueError('same Tensor already tracked')
        self._entries[label]=(str(uuid.uuid4()),weakref.ref(tensor))

    def snapshot(self):
        rows=[]
        for label,(token,ref) in self._entries.items():
            value=ref();alive=value is not None
            rows.append(dict(label=label,object_id=token,alive=alive,logical_bytes=value.numel()*value.element_size() if alive else 0))
        return dict(timestamp_ns=time.perf_counter_ns(),objects=rows,
            alive_objects=sum(r['alive'] for r in rows),logical_live_bytes=sum(r['logical_bytes'] for r in rows),
            scope='explicit_weakrefs_distinct_tensor_objects_not_storage_or_heap_census')


def _retention(snapshot):
    _int(snapshot['timestamp_ns']);objects=snapshot['objects']
    if len(objects)>4096 or len({r['object_id'] for r in objects})!=len(objects):raise ValueError('duplicate/excess retention objects')
    for r in objects:
        _name(r['label']);_name(r['object_id']);_int(r['logical_bytes'])
        if type(r['alive']) is not bool or not r['alive'] and r['logical_bytes']!=0:raise ValueError('invalid retention lifetime')
    if (snapshot['alive_objects']!=sum(r['alive'] for r in objects)
            or snapshot['logical_live_bytes']!=sum(r['logical_bytes'] for r in objects)):raise ValueError('retention snapshot accounting mismatch')
    return {r['object_id'] for r in objects if r['alive']}


def _fit(points):
    if len(points)<3:return dict(status='insufficient_samples',n=len(points),slope_bytes_per_second=None,standard_error=None)
    origin=points[0][0];xs=[(t-origin)/1e9 for t,y in points];ys=[y for t,y in points]
    if any(b<=a for a,b in zip(xs,xs[1:])):raise ValueError('nonmonotonic memory timestamps')
    n=len(xs);mx=math.fsum(xs)/n;my=math.fsum(ys)/n
    xx=math.fsum((x-mx)**2 for x in xs)
    slope=math.fsum((x-mx)*(y-my) for x,y in zip(xs,ys))/xx
    residual=[y-(my+slope*(x-mx)) for x,y in zip(xs,ys)]
    sse=math.fsum(r*r for r in residual)
    return dict(status='descriptive_fit',n=n,slope_bytes_per_second=slope,standard_error=math.sqrt(sse/(n-2)/xx),
        uncertainty_scope='OLS_standard_error_conditional_iid_not_confidence_or_leak_bound',
        span_seconds=xs[-1]-xs[0],min_bytes=min(ys),max_bytes=max(ys),delta_bytes=ys[-1]-ys[0],
        confidence_interval=None,confidence_status='unsupported_correlated_native_samples')


def analyze_memory_trend(stability_result, *, warmup_seconds=0., plateau_range_bytes=1024**2,slope_tolerance_bytes_per_second=1024.):
    if type(stability_result) is not RuntimeResult or stability_result.to_dict()['feature_id']!='P31':raise TypeError('P31 RuntimeResult required')
    for v in (warmup_seconds,plateau_range_bytes,slope_tolerance_bytes_per_second):_number(v)
    return _result('P32',dict(stability=stability_result.to_dict(),warmup_seconds=warmup_seconds,
        plateau_range_bytes=plateau_range_bytes,slope_tolerance_bytes_per_second=slope_tolerance_bytes_per_second))


def _analyze_trend(e):
    source=RuntimeResult(e['stability']).to_dict()
    if source['feature_id']!='P31':raise ValueError('P32 requires P31 source')
    for k in ('warmup_seconds','plateau_range_bytes','slope_tolerance_bytes_per_second'):_number(e[k])
    raw=source['evidence'];terminal=raw['terminal']
    if terminal is None:return dict(status='insufficient_samples',reason='no_workload_window',leak_proven=False)
    cutoff=terminal['work_start_ns']+e['warmup_seconds']*1e9
    points=[];allocated=[];reserved=[];snapshots=[];halfwidth=[];excluded=0
    for row in raw['rows']:
        memory=row['rss']
        if row['phase']=='idle' or memory is None or memory['query_start_ns']<cutoff:
            excluded+=1;continue
        points.append((memory['timestamp_ns'],memory['rss_bytes']))
        halfwidth.append((memory['query_end_ns']-memory['query_start_ns'])/2e9)
        if row['allocator'] is not None:
            allocated.append((memory['timestamp_ns'],row['allocator']['allocated_bytes']))
            reserved.append((memory['timestamp_ns'],row['allocator']['reserved_bytes']))
        if row['retention'] is not None:
            snapshot=row['retention'];_retention(snapshot)
            if snapshot['timestamp_ns']>row['timestamp_ns']:raise ValueError('retention snapshot outside observation')
            snapshots.append(snapshot)
    rss=_fit(points);alloc=_fit(allocated);cache=_fit(reserved);classification='insufficient_samples'
    retained_growth=False;released=[]
    if len(snapshots)>=2:
        first,last=snapshots[0],snapshots[-1]
        retained_growth=last['logical_live_bytes']>first['logical_live_bytes']
        released=sorted(_retention(first)-_retention(last))
    if rss['status']=='descriptive_fit':
        tolerance=e['slope_tolerance_bytes_per_second'];plateau=e['plateau_range_bytes']
        if retained_growth:classification='tracked_live_reference_growth_observed'
        elif (alloc['status']=='descriptive_fit' and cache['status']=='descriptive_fit'
                and alloc['max_bytes']-alloc['min_bytes']<=plateau and cache['delta_bytes']>plateau):
            classification='allocator_cache_growth_observed'
        elif abs(rss['slope_bytes_per_second'])<=tolerance and rss['max_bytes']-rss['min_bytes']<=plateau:
            classification='sampled_plateau'
        else:classification='rss_change_unattributed'
    return dict(status='derived' if rss['status']=='descriptive_fit' else 'insufficient_samples',classification=classification,
        leak_proven=False,leak_status='unsupported_no_causal_ownership_leak_proof',rss=rss,allocated=alloc,reserved=cache,
        warmup_seconds=e['warmup_seconds'],excluded_samples=excluded,source_complete=source['analysis']['status']=='completed',
        retention=dict(snapshot_count=len(snapshots),live_reference_growth=retained_growth,released_object_ids=released,
            logical_byte_scope='observed_distinct_tensor_objects_may_share_storage'),
        timestamp_max_half_width_seconds=max(halfwidth) if halfwidth else None,
        attribution='process_RSS_not_model_only; allocator_reserved_not_physical_VRAM; retained_objects_not_causal_RSS_proof')
