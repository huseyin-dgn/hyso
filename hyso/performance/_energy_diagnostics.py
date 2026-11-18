"""P33 v2 reporting over existing P31 power evidence; never executes a query."""
import math

from .runtime_result import RuntimeResult,_int,_number
from .energy import _validate_power_sample,_integral


def _events(rows,device,clock):
    events=[];identity={};last=-1
    for row in rows:
        sample=row['gpu'];status=sample.get('status');valid=False;reason=None
        if sample.get('device_id',device)!=device or sample.get('clock_id',clock)!=clock:
            raise ValueError('power source device/clock mismatch')
        a=sample.get('query_start_ns',row['timestamp_ns']);b=sample.get('query_end_ns',row['timestamp_ns'])
        _int(a);_int(b,a)
        if b>row['timestamp_ns']:raise ValueError('power query after source observation')
        stamp=sample.get('timestamp_ns',(a+b)//2);_int(stamp,a,b)
        if stamp<=last:raise ValueError('duplicate/nonmonotonic power event timestamps')
        last=stamp
        if sample.get('backend')=='nvidia-smi.power.draw.instant':
            _validate_power_sample(sample,device,clock,require_power=status=='measured')
            if status=='measured':
                if sample.get('reason') is not None:raise ValueError('measured power has unavailable reason')
                valid=True
            elif sample.get('watts') is not None:raise ValueError('unavailable power must be null')
            for key in ('pci_bus_id','device_name','sensor_id'):
                value=sample.get(key)
                if value is not None:
                    if key in identity and identity[key]!=value:raise ValueError('incompatible physical power sensor identity')
                    identity[key]=value
        elif sample.get('backend') is not None:
            # No cumulative-counter adapter is supported here. Do not infer a
            # delta, unwrap/reset correction, unit scale or a power reading.
            for key in ('counter_value','joules','energy_millijoules'):
                if sample.get(key) is not None:_number(sample[key])
            reason='unsupported_sensor_or_energy_counter_backend_no_wrap_reset_conversion'
        if not valid:reason=reason or sample.get('reason') or 'power_not_measured'
        events.append(dict(index=row['index'],phase=row['phase'],timestamp_ns=stamp,query_start_ns=a,query_end_ns=b,
            watts=sample['watts'] if valid else None,valid=valid,reason=reason,
            legacy_declared_accuracy_watts=sample.get('sensor_accuracy_watts')))
    return events,dict(backend='nvidia-smi.power.draw.instant',device_id=device,scope='physical_board',unit='W',clock_id=clock,**identity)


def _coverage(events,start,end,gap):
    _int(start);_int(end,start+1)
    pieces=[];holes=[];cursor=start
    for a,b in zip(events,events[1:]):
        left=max(start,a['timestamp_ns']);right=min(end,b['timestamp_ns'])
        if right<=left:continue
        if left>cursor:holes.append(dict(start_ns=cursor,end_ns=left,reason='unbracketed_window'))
        if not a['valid'] or not b['valid']:reason='missing_power_samples'
        elif b['timestamp_ns']-a['timestamp_ns']>gap:reason='sensor_sample_gap'
        else:reason=None
        if reason:holes.append(dict(start_ns=left,end_ns=right,reason=reason))
        else:
            energy,problem=_integral([a,b],left,right,gap)
            if problem:raise ValueError('inconsistent power integration segment')
            pieces.append(dict(start_ns=left,end_ns=right,joules=energy,
                left_source_index=a['index'],right_source_index=b['index']))
        cursor=right
    if cursor<end:holes.append(dict(start_ns=cursor,end_ns=end,reason='unbracketed_window'))
    covered=sum(p['end_ns']-p['start_ns'] for p in pieces);duration=end-start
    if covered+sum(h['end_ns']-h['start_ns'] for h in holes)!=duration:
        raise ValueError('power coverage accounting mismatch')
    return dict(start_ns=start,end_ns=end,requested_seconds=duration/1e9,covered_seconds=covered/1e9,
        full=covered==duration,joules=math.fsum(p['joules'] for p in pieces) if pieces else None,
        covered_segments=pieces,uncovered_intervals=holes,
        reason=None if covered==duration else holes[0]['reason'] if holes else 'insufficient_samples')


def _window(raw,source_analysis):
    rows=raw['rows'];terminal=raw['terminal'];identity=raw['header']['workload']
    batch=identity['input_shape'][0];_int(batch,1)
    old_steps=old_images=0;last_completion=None
    for row in rows:
        steps,images=row['steps'],row['completed']
        if images!=steps*batch:raise ValueError('successful inference/image denominator mismatch')
        marker=row['last_completion_ns']
        if steps>old_steps:
            if marker is None or marker<= (last_completion if last_completion is not None else -1):
                raise ValueError('successful work lacks a new completion boundary')
        elif marker!=last_completion:raise ValueError('completion changed without completed work')
        old_steps,old_images,last_completion=steps,images,marker
    if not old_steps:return None,'zero_completed_work'
    if source_analysis['status']=='completed':
        return dict(start_ns=terminal['work_start_ns'],end_ns=terminal['work_end_ns'],
            scope='exact_requested_work_window',completed_forwards=old_steps,completed_images=old_images),None
    first=next((r for r in rows if r['phase']=='work'),None)
    if (first is None or first['steps'] or first['completed'] or first['last_completion_ns'] is not None
            or last_completion is None or last_completion<=first['timestamp_ns']):
        return None,'no_verified_zero_counter_to_completion_prefix'
    return dict(start_ns=first['timestamp_ns'],end_ns=last_completion,
        scope='verified_committed_zero_counter_observation_to_last_successful_completion_prefix',
        completed_forwards=old_steps,completed_images=old_images),None


def _baseline(events,window,e):
    idle=[p for p in events if p['phase']=='idle']
    empty=dict(watts=None,status='unsupported',reason=None,sample_count=len(idle),window=None,
        method='time_weighted_idle_piecewise_linear_trapezoid',attribution='inactive_board_background_uncontrolled_not_model_exclusive')
    if any(p['query_end_ns']>window['start_ns'] for p in idle):raise ValueError('idle baseline overlaps active inference window')
    if len(idle)<2:return dict(empty,reason='insufficient_idle_baseline')
    if (window['start_ns']-idle[-1]['timestamp_ns'])/1e9>e['idle_max_age_seconds']:
        return dict(empty,reason='idle_baseline_stale')
    coverage=_coverage(idle,idle[0]['timestamp_ns'],idle[-1]['timestamp_ns'],e['max_gap_seconds']*1e9)
    if not coverage['full']:return dict(empty,reason='idle_'+coverage['reason'],window=coverage)
    return dict(empty,status='derived',reason=None,watts=coverage['joules']/coverage['covered_seconds'],window=coverage)


def _analyze(e):
    source=RuntimeResult(e['stability']).to_dict()
    if source['feature_id']!='P31':raise ValueError('P33 requires P31 completion/clock evidence')
    for key in ('max_gap_seconds','idle_max_age_seconds'):_number(e[key],.001)
    raw=source['evidence'];a=source['analysis'];h=raw['header'];identity=h['workload'] if h else {}
    device=identity.get('device_id');clock=h['clock_id'] if h else None
    counts=dict(completed_forwards=a['steps'],completed_images=a['completed'])
    window=None;reason=None;events=[];sensor=None;coverage=None
    idle=dict(status='unsupported',reason='no_verified_active_energy_window',watts=None,sample_count=0,window=None)
    gross=net=per_image=per_call=net_per_image=None
    eligible=(h is not None and identity.get('kind')=='model_inference' and identity.get('unit')=='image'
        and type(device) is str and device.startswith('GPU-'))
    if not eligible:reason='no_matching_physical_GPU_inference_source;CPU_RAPL_and_external_counter_adapters_unsupported'
    else:
        events,sensor=_events(raw['rows'],device,clock);window,reason=_window(raw,a)
        if window is not None:
            coverage=_coverage(events,window['start_ns'],window['end_ns'],e['max_gap_seconds']*1e9)
            idle=_baseline(events,window,e)
            if coverage['full']:
                gross=coverage['joules'];per_image=gross/window['completed_images'];per_call=gross/window['completed_forwards']
                if idle['watts'] is not None:
                    net=gross-idle['watts']*coverage['requested_seconds'];net_per_image=net/window['completed_images']
            else:reason=coverage['reason']
    measured=[p for p in events if p['valid']]
    if not measured and eligible:
        reason='unsupported_sensor_or_energy_counter_backend' if any('counter_backend' in (p['reason'] or '') for p in events) else 'no_supported_power_samples'
        sensor=None
    status=('derived_from_measured_power_samples' if a['status']=='completed' else 'derived_verified_prefix_from_measured_power_samples')
    if gross is None:status='partial_power_coverage' if coverage and coverage['joules'] is not None else 'unsupported'
    duration=coverage['requested_seconds'] if coverage else None
    raw_mean=math.fsum(p['watts'] for p in measured)/len(measured) if measured else None
    declared=sorted({p['legacy_declared_accuracy_watts'] for p in measured if p['legacy_declared_accuracy_watts'] is not None})
    uncertainty=dict(sensor_accuracy_watts=None,sensor_accuracy_reason='no_device_specific_calibration_or_accuracy_reported_by_query',
        legacy_unverified_declared_accuracy_watts=declared,total_energy_error_bound_joules=None,confidence_interval=None,
        query_max_half_width_seconds=max((p['query_end_ns']-p['query_start_ns'])/2e9 for p in measured) if measured else None,
        sampling='piecewise_linear_assumption;refresh_latency_and_between_sample_variation_unknown;no_interpolation_across_holes',
        overhead='work_window_includes_source_health_queries_validation_journal_and_scheduling;not_calibrated')
    trace_window=dict(start_ns=measured[0]['timestamp_ns'],end_ns=measured[-1]['timestamp_ns'],scope='all_recorded_power_samples_including_idle_and_boundary') if measured else None
    context=dict(sensor=sensor,device_id=device,domain='physical_GPU_board_not_CPU_plus_GPU_system_or_model_exclusive' if eligible else 'unsupported_power_domain_no_matching_sensor',
        clock_id=clock,active_window=window,source_result_id=source['result_id'],source_sha256=source['source_sha256'],
        raw_power_reference='evidence.stability.evidence.rows[*].gpu',raw_completed_work_reference='evidence.stability.evidence.rows[*].steps/completed/last_completion_ns',
        source_complete=a['status']=='completed',source_status=a['status'],source_reason=a.get('reason'))
    default_window=object()
    def metric(name,value,unit,kind,method,why=None,metric_window=default_window,uses_count=False):
        return dict(name=name,value=value,unit=unit,kind=kind,status='supported' if value is not None else 'unsupported',
            reason=None if value is not None else why or reason or 'measurement_unavailable',sensor_reference='measurement_context.sensor',
            device_id=device,domain=context['domain'],clock_id=clock,window=window if metric_window is default_window else metric_window,
            raw_evidence_reference=context['raw_power_reference'],completed_work_reference=dict(counts,used_as_denominator=uses_count and value is not None),
            method=method,uncertainty_reference='uncertainty_details')
    measurements={
        'sample_min_power':metric('sample_min_power',min(p['watts'] for p in measured) if measured else None,'W','derived_summary_of_measured_samples','minimum_observed_sample_not_continuous_extremum',metric_window=trace_window),
        'sample_max_power':metric('sample_max_power',max(p['watts'] for p in measured) if measured else None,'W','derived_summary_of_measured_samples','maximum_observed_sample_not_continuous_extremum',metric_window=trace_window),
        'sample_mean_power':metric('sample_mean_power',raw_mean,'W','derived_summary_of_measured_samples','arithmetic_sample_mean_not_time_weighted',metric_window=trace_window),
        'active_time_weighted_mean_power':metric('active_time_weighted_mean_power',gross/duration if gross is not None else None,'W','derived','gross_J/fully_covered_active_seconds'),
        'gross_energy':metric('gross_energy',gross,'J','derived','clipped_piecewise_linear_power_trapezoid'),
        'covered_energy':metric('covered_energy',coverage['joules'] if coverage else None,'J','derived','sum_of_covered_segments_only;not_full_window_when_gaps_exist'),
        'idle_power':metric('idle_power',idle['watts'],'W','derived',idle.get('method','time_weighted_idle_trapezoid'),idle['reason'],metric_window=idle.get('window')),
        'net_above_idle_energy':metric('net_above_idle_energy',net,'J','derived','gross_J-idle_W*active_s;negative_values_preserved',idle['reason']),
        'energy_per_forward':metric('energy_per_forward',per_call,'J/forward','derived','gross_J/verified_successful_forward_calls',uses_count=True),
        'energy_per_image':metric('energy_per_image',per_image,'J/image','derived','gross_J/verified_completed_images',uses_count=True),
        'net_energy_per_image':metric('net_energy_per_image',net_per_image,'J/image','derived','net_J/verified_completed_images',idle['reason'],uses_count=True),
        'direct_counter_energy':metric('direct_counter_energy',None,'J','unsupported','not_computed','no_supported_cumulative_counter_adapter;wrap_reset_units_not_inferred')}
    intervals=[b['timestamp_ns']-a['timestamp_ns'] for a,b in zip(events,events[1:])]
    return dict(status=status,reason=reason,gross_joules=gross,idle_mean_watts=idle['watts'],net_above_idle_joules=net,
        joules_per_completed_image=per_image,joules_per_forward=per_call,net_joules_per_completed_image=net_per_image,
        completed_images=counts['completed_images'],completed_forwards=counts['completed_forwards'],duration_seconds=duration,
        device_id=device,sample_count=len(measured),idle_sample_count=idle['sample_count'],
        max_query_half_width_seconds=uncertainty['query_max_half_width_seconds'],nominal_sensor_spec_energy_bound_joules=None,
        nominal_sensor_spec_bound_reason='legacy_5W_constant_is_not_a_verified_device_specific_sensor_bound',
        uncertainty='no_total_error_bound_or_confidence_interval;unverified_accuracy_refresh_latency_and_between_sample_variation',
        integration='host_associated_instant_power_piecewise_linear_trapezoid_not_continuous_energy_counter',
        idle_scope='workload_inactive_physical_board_background_uncontrolled',
        attribution='whole_physical_board_not_model_exclusive_or_CPU_plus_GPU_system;negative_idle_subtraction_preserved',
        measurements=measurements,measurement_context=context,coverage=coverage,idle_baseline=idle,raw_power_observations=events,
        sampling=dict(requested_interval_seconds=raw['request']['protocol']['interval_seconds'],observed_intervals_ns=intervals,
            min_interval_seconds=min(intervals)/1e9 if intervals else None,max_interval_seconds=max(intervals)/1e9 if intervals else None,
            mean_interval_seconds=math.fsum(intervals)/len(intervals)/1e9 if intervals else None,
            missing_sample_indices=[p['index'] for p in events if not p['valid']],
            max_gap_seconds=e['max_gap_seconds'],sensor_refresh_period_seconds=None,sensor_refresh_reason='not_recorded_by_query'),
        uncertainty_details=uncertainty,source_complete=context['source_complete'],
        limitations=['Measured W and derived J are separate; no TDP/utilization/nominal-power estimate is used.',
            'Incomplete energy coverage has no per-work denominator; covered Joules are labelled separately.',
            'An aborted-source ratio, when supported, covers only a verified committed zero-counter-to-completion prefix.',
            'No idle baseline means net dynamic energy is unsupported, while covered gross energy can remain supported.',
            'Baseline subtraction is descriptive and does not establish model-only causality or energy savings.',
            'No CPU/RAPL, external meter or cumulative energy-counter unit/wrap/reset adapter is implemented.'])
