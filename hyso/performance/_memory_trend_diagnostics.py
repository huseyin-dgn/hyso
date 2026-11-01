"""P32 v2: independent memory domains over validated, already-recorded P31 data."""
import math

from .runtime_result import RuntimeResult,_int,_number
from .memory_trend import _fit,_retention

SCOPES=dict(rss='current_worker_process_RSS_not_model_only',
    private_commit='current_worker_process_private_commit_not_RSS_or_job_total',
    allocated='owned_process_device_PyTorch_allocated_not_physical_VRAM',
    reserved='owned_process_device_PyTorch_reserved_not_driver_or_physical_VRAM')


def _point(row,kind,header):
    node=row.get('rss' if kind=='rss' else 'process_commit' if kind=='private_commit' else 'allocator')
    if node is None:return None,'source_did_not_record_this_memory_domain'
    if kind in ('allocated','reserved') and header['workload']['device_id']=='cpu':
        raise ValueError('CPU-only source cannot contain CUDA allocator observations')
    if node.get('clock_id',header['clock_id'])!=header['clock_id']:raise ValueError('memory clock domain mismatch')
    if node.get('pid',header['pid'])!=header['pid']:raise ValueError('memory process ownership mismatch')
    if node.get('unit','byte')!='byte':raise ValueError('memory bytes required; implicit unit conversion is unsupported')
    if kind=='rss' and node.get('scope',SCOPES[kind]) not in (SCOPES[kind],'current_process_rss'):
        raise ValueError('RSS scope mismatch')
    if kind=='private_commit':
        if node.get('scope',SCOPES[kind])!=SCOPES[kind]:raise ValueError('private commit scope mismatch')
        if node['status']=='unsupported':
            if node.get('value') is not None or not node.get('reason'):raise ValueError('unavailable commit needs null and reason')
            _int(node['query_start_ns']);_int(node['query_end_ns'],node['query_start_ns'])
            if node['query_end_ns']>row['timestamp_ns']:raise ValueError('unavailable commit query after observation')
            return None,node['reason']
        if node['status']!='measured':raise ValueError('unknown private commit status')
        value=node['value']
    else:
        value=node['rss_bytes'] if kind=='rss' else node[kind+'_bytes']
        if kind in ('allocated','reserved') and node['device_id']!=header['workload']['device_id']:
            raise ValueError('allocator device ownership mismatch')
        if kind in ('allocated','reserved') and node.get('scope','process_device_allocator')!='process_device_allocator':
            raise ValueError('allocator series cannot represent physical driver memory')
    _int(value)
    if 'query_start_ns' in node:
        start,end=node['query_start_ns'],node['query_end_ns'];_int(start);_int(end,start)
        stamp=node.get('timestamp_ns',(start+end)//2);_int(stamp,start,end)
        if end>row['timestamp_ns']:raise ValueError('memory query after observation')
        timing='native_query_bracket'
    else:
        # Old allocator rows have no individual query bracket. Associate them
        # with their journal observation, never with an unrelated RSS query.
        if kind not in ('allocated','reserved'):raise ValueError('CPU memory query bracket required')
        stamp=start=end=row['timestamp_ns'];timing='legacy_journal_timestamp_association_not_query_time'
    return dict(index=row['index'],timestamp_ns=stamp,query_start_ns=start,query_end_ns=end,
        value=value,phase=row['phase'],timing_scope=timing),None


def _series(points,missing,window,cutoff,e,scope,clock,nominal_ns,not_requested=False):
    _fit([(p['timestamp_ns'],p['value']) for p in points])  # Validate even excluded or short series.
    contained=[];outside=[]
    for point in points:
        (contained if window is not None and point['phase']!='idle' and
            window[0]<=point['query_start_ns']<=point['query_end_ns']<=window[1] else outside).append(point)
    selected=[p for p in contained if cutoff is not None and p['query_start_ns']>=cutoff]
    def describe(items):
        fit=_fit([(p['timestamp_ns'],p['value']) for p in items]);values=[p['value'] for p in items]
        gaps=[dict(before_index=a['index'],after_index=b['index'],gap_ns=b['timestamp_ns']-a['timestamp_ns'])
              for a,b in zip(items,items[1:]) if b['timestamp_ns']-a['timestamp_ns']>2*nominal_ns]
        plateau=(fit['status']=='descriptive_fit' and abs(fit['slope_bytes_per_second'])<=e['slope_tolerance_bytes_per_second']
                 and max(values)-min(values)<=e['plateau_range_bytes'])
        return dict(fit,classification='sampled_plateau' if plateau else 'observed_change_unattributed'
                if fit['status']=='descriptive_fit' else 'insufficient_samples',
            baseline_bytes=values[0] if values else None,end_bytes=values[-1] if values else None,
            observed_delta_bytes=values[-1]-values[0] if values else None,
            sampled_peak_bytes=max(values) if values else None,sampled_min_bytes=min(values) if values else None,
            observed_span_seconds=(items[-1]['timestamp_ns']-items[0]['timestamp_ns'])/1e9 if len(items)>1 else None,
            start_ns=items[0]['timestamp_ns'] if items else None,end_ns=items[-1]['timestamp_ns'] if items else None,
            gaps=gaps,gap_count=len(gaps),timeline=items,
            adjacent_increase_bytes=sum(max(0,b-a) for a,b in zip(values,values[1:])) if values else None,
            adjacent_decrease_bytes=sum(max(0,a-b) for a,b in zip(values,values[1:])) if values else None)
    result=describe(selected);full=describe(contained)
    # Boundary readings remain evidence, but cannot leak into an exact work-window fit.
    boundaries=[p for p in outside if window is not None and p['phase']!='idle']
    result.update(unit='byte',scope=scope,clock_id=clock,
        status='unsupported' if not_requested else result['status'],
        reason='CUDA_allocator_not_requested_for_CPU_workload' if not_requested else
            None if result['n']>=3 else 'fewer_than_3_post_warmup_observations',
        missing_samples=[] if not_requested else missing,missing_sample_count=0 if not_requested else len(missing),full_work_window=full,
        recorded_non_idle_observations=sum(p['phase']!='idle' for p in points),
        excluded_idle_observations=sum(p['phase']=='idle' for p in points),
        excluded_warmup_samples=len(contained)-len(selected),excluded_outside_work_window=len(outside),
        boundary_observations=boundaries,peak_scope='sampled_values_only_not_absolute_or_allocator_reset_peak',
        timestamp_max_half_width_seconds=max((p['query_end_ns']-p['query_start_ns'])/2e9 for p in selected) if selected else None)
    early=[p for p in contained if cutoff is not None and p['query_end_ns']<cutoff]
    result['warmup']=dict(early_observations=early,
        before_after_delta_bytes=selected[0]['value']-early[0]['value'] if early and selected else None,
        transient_growth_then_plateau=(bool(early) and result['classification']=='sampled_plateau'
            and max(p['value'] for p in early+selected)-min(p['value'] for p in early+selected)>e['plateau_range_bytes']),
        interpretation='observed_pre_post_warmup_difference_not_causal_allocator_or_leak_attribution')
    return result


def _witnesses(rows,header,window,cutoff,warmup):
    all_snapshots=[];known={};previous_ns=None;labels={}
    for row in rows:
        snapshot=row.get('retention')
        if snapshot is None:continue
        _retention(snapshot)
        _int(snapshot['alive_objects']);_int(snapshot['logical_live_bytes'])
        stamp=snapshot['timestamp_ns']
        if previous_ns is not None and stamp<=previous_ns:raise ValueError('duplicate/reordered retention observations')
        if stamp>row['timestamp_ns'] or snapshot.get('clock_id',header['clock_id'])!=header['clock_id']:
            raise ValueError('retention clock association mismatch')
        if snapshot.get('scope','explicit_weakrefs_distinct_tensor_objects_not_storage_or_heap_census')!='explicit_weakrefs_distinct_tensor_objects_not_storage_or_heap_census':
            raise ValueError('retention logical object scope mismatch')
        objects={r['object_id']:r for r in snapshot['objects']}
        if len({r['label'] for r in objects.values()})!=len(objects):raise ValueError('duplicate retention labels')
        if not known.keys()<=objects.keys():raise ValueError('tracked object history disappeared')
        for token,r in objects.items():
            if token in known and (known[token]['label']!=r['label'] or not known[token]['alive'] and r['alive']):
                raise ValueError('retention ID relabelled or dead weakref resurrected')
            if r['label'] in labels and labels[r['label']]!=token:raise ValueError('retention label rebound to another object')
            labels[r['label']]=token
        known=objects;previous_ns=stamp
        # The initial zero-counter boundary is a real pre-work snapshot. Keep it
        # explicitly associated with the start when no warmup is requested.
        boundary=(window is not None and warmup==0 and row['phase']=='work' and row['steps']==0
                  and row['timestamp_ns']==window[0] and stamp<=window[0])
        if window is not None and row['phase']!='idle' and (boundary or
                cutoff is not None and cutoff<=stamp<=window[1]):
            all_snapshots.append(dict(snapshot,boundary_association='initial_zero_counter_observation' if boundary else None))
    snapshots=all_snapshots;released=set()
    for a,b in zip(snapshots,snapshots[1:]):released.update(_retention(a)-_retention(b))
    first,last=(snapshots[0],snapshots[-1]) if snapshots else (None,None)
    return dict(snapshot_count=len(snapshots),live_reference_growth=last['logical_live_bytes']>first['logical_live_bytes'] if len(snapshots)>=2 else None,
        live_object_count_growth=last['alive_objects']>first['alive_objects'] if len(snapshots)>=2 else None,
        released_object_ids=sorted(released),timeline=snapshots,
        initial_alive_objects=first['alive_objects'] if first else None,final_alive_objects=last['alive_objects'] if last else None,
        initial_logical_bytes=first['logical_live_bytes'] if first else None,final_logical_bytes=last['logical_live_bytes'] if last else None,
        reason=None if len(snapshots)>=2 else 'fewer_than_2_post_warmup_weakref_snapshots',
        logical_byte_scope='observed_distinct_tensor_objects_may_share_storage',
        causal_ownership='not_proven_weakrefs_do_not_identify_owning_referrers_or_a_leak',
        physical_storage_released=None,physical_storage_reason='dead_Tensor_object_does_not_prove_allocator_or_OS_release')


def _analyze(e):
    source=RuntimeResult(e['stability']).to_dict()
    if source['feature_id']!='P31':raise ValueError('P32 requires P31 source')
    for key in ('warmup_seconds','plateau_range_bytes','slope_tolerance_bytes_per_second'):_number(e[key])
    raw=source['evidence'];header=raw['header'];rows=raw['rows'];terminal=raw['terminal']
    window=None;origin='unavailable'
    if terminal is not None:
        window=(terminal['work_start_ns'],terminal['work_end_ns']);origin='exact_terminal_work_window'
    elif any(r['phase']=='work' for r in rows):
        window=(next(r['timestamp_ns'] for r in rows if r['phase']=='work'),rows[-1]['timestamp_ns'])
        origin='committed_work_observation_prefix_not_exact_full_run'
    duration=(window[1]-window[0])/1e9 if window is not None else None
    cutoff=window[0]+math.ceil(e['warmup_seconds']*1e9) if window is not None and e['warmup_seconds']<=duration else None
    nominal_ns=int(raw['request']['protocol']['interval_seconds']*1e9)
    metrics={};clock=header['clock_id'] if header else None;cpu=header is not None and header['workload']['device_id']=='cpu'
    for kind,scope in SCOPES.items():
        points=[];missing=[]
        for row in rows:
            point,reason=_point(row,kind,header)
            if point is not None:points.append(point)
            elif row['phase']!='idle':missing.append(dict(index=row['index'],reason=reason))
        metrics[kind]=_series(points,missing,window,cutoff,e,scope,clock,nominal_ns,cpu and kind in ('allocated','reserved'))
    retention=_witnesses(rows,header,window,cutoff,e['warmup_seconds']) if header else dict(snapshot_count=0,
        live_reference_growth=None,released_object_ids=[],reason='source_has_no_worker_header',
        logical_byte_scope='observed_distinct_tensor_objects_may_share_storage')
    usable=[m for m in metrics.values() if m['status']=='descriptive_fit']
    classification='insufficient_samples'
    if retention['live_reference_growth']:classification='tracked_live_reference_growth_observed'
    elif (metrics['allocated']['classification']=='sampled_plateau' and metrics['reserved']['n']>=3
            and metrics['reserved']['observed_delta_bytes']>e['plateau_range_bytes']):classification='allocator_cache_growth_observed'
    elif usable:
        if all(m['classification']=='sampled_plateau' for m in usable):
            classification='warmup_growth_then_sampled_plateau' if any(m['warmup']['transient_growth_then_plateau'] for m in usable) else 'sampled_plateau'
        elif metrics['rss']['classification']=='observed_change_unattributed':classification='rss_change_unattributed'
        elif metrics['private_commit']['classification']=='observed_change_unattributed':classification='private_commit_change_unattributed'
        else:classification='cuda_allocator_change_unattributed'
    return dict(status='derived' if usable or retention['snapshot_count']>=2 else 'insufficient_samples',
        classification=classification,leak_proven=False,leak_status='unsupported_no_causal_ownership_repeat_or_leak_proof',
        **metrics,retention=retention,warmup_seconds=e['warmup_seconds'],
        excluded_samples=metrics['rss']['excluded_warmup_samples']+metrics['rss']['excluded_outside_work_window']+metrics['rss']['missing_sample_count'],
        source_complete=source['analysis']['status']=='completed',source_status=source['analysis']['status'],
        source_reason=source['analysis'].get('reason'),source_result_id=source['result_id'],source_sha256=source['source_sha256'],
        measurement_window=dict(start_ns=window[0] if window else None,end_ns=window[1] if window else None,
            seconds=duration,warmup_cutoff_ns=cutoff,origin=origin,clock_id=clock,
            warmup_reason='warmup_exceeds_available_window_or_window_missing' if cutoff is None else None),
        source_gaps=source['analysis'].get('gaps',[]),source_coverage=source['analysis'].get('coverage','none'),
        timestamp_max_half_width_seconds=metrics['rss']['timestamp_max_half_width_seconds'],
        allocator_peak=dict(allocated_bytes=None,reserved_bytes=None,status='unsupported',
            reason='source_has_no_owned_window_peak_reset_domain_evidence;sampled_maxima_are_separate',
            reset_performed_by_P32=False,source_reset_scope='not_recorded_in_P31_series'),
        attribution='process_RSS_and_private_commit_not_model_only; allocator_reserved_not_physical_VRAM; retained_objects_not_causal_RSS_proof',
        limitations=['Each memory domain uses its own query clock or an explicitly labelled legacy row association.',
            'Fits use only contained post-warmup observations; boundary values are reported separately.',
            'No interpolation across missing observations; OLS is descriptive and spans only recorded timestamps.',
            'Conditional iid OLS standard error is not a confidence interval for correlated native samples.',
            'Sampled maxima can miss transients; cache growth and weakref object lifetimes do not prove a leak.',
            'Short runs cannot establish long-term leak absence, causality, physical storage release or stable future memory.'])
