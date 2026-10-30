"""P31 descriptive diagnostics over verified journal samples; no P32 leak model."""
import math

from .runtime_result import _int, _number


def _validate_extensions(rows, terminal, header):
    extended=any('step_timing' in r for r in rows)
    attempted=0
    kinds=(None,'oom','device_error','workload_exception','measurement_error')
    for row in rows:
        alloc=row['allocator']
        if alloc is not None and 'query_start_ns' in alloc:
            _int(alloc['query_start_ns']);_int(alloc['query_end_ns'],alloc['query_start_ns'])
            if alloc['query_end_ns']>row['timestamp_ns']:raise ValueError('allocator observation after journal sample')
        if extended:
            count=row['steps'];timing=row['step_timing']
            _int(row['attempted_steps'],max(attempted,count));attempted=row['attempted_steps']
            if attempted-count>1 or row['failure_kind'] not in kinds:raise ValueError('invalid step/failure accounting')
            if ((row['failure'] is None)!=(row['failure_kind'] is None) or
                    (row['phase'] in ('work','terminal','idle') and attempted!=count)):
                raise ValueError('inconsistent successful/failing step receipt')
            if timing['count']!=count or type(timing['count']) is not int:raise ValueError('step timing count mismatch')
            _number(timing['m2_ns2'])
            if count:
                _int(timing['min_ns']);_int(timing['max_ns'],timing['min_ns']);_number(timing['mean_ns'])
                if not (timing['min_ns']<=timing['mean_ns']<=timing['max_ns'] and
                        math.isclose(timing['mean_ns']*count,row['work_ns'],rel_tol=1e-9,abs_tol=.5)):
                    raise ValueError('inconsistent cumulative step timing')
                if timing['m2_ns2']>count*(timing['max_ns']-timing['min_ns'])**2:
                    raise ValueError('step variance exceeds observed range')
            elif row['work_ns'] or any(timing[k] is not None for k in ('mean_ns','min_ns','max_ns')) or timing['m2_ns2']:
                raise ValueError('zero steps cannot have measured timing')
        gpu=row.get('gpu_utilization')
        if gpu is None:continue  # Legacy journal.
        if gpu['status']=='not_requested':
            if header['workload']['device_id']!='cpu':raise ValueError('missing requested GPU observation')
            continue
        _int(gpu['query_start_ns']);_int(gpu['query_end_ns'],gpu['query_start_ns'])
        if gpu['query_end_ns']>row['timestamp_ns']:raise ValueError('GPU observation after journal sample')
        if gpu['status']=='measured':
            values=gpu['values']
            if values['uuid']!=header['workload']['device_id']:raise ValueError('GPU utilization device mismatch')
            for name in ('gpu_busy_percent','memory_controller_busy_percent'):
                if values[name] is not None:
                    _number(values[name])
                    if values[name]>100:raise ValueError('GPU percentage exceeds 100')
            for name in ('memory_used_bytes','memory_free_bytes','memory_total_bytes'):
                if values[name] is not None:_number(values[name])
            from ._telemetry import _validate_gpu_memory
            _validate_gpu_memory(values)
        elif gpu['status'] not in ('unsupported','unavailable','timed_out'):
            raise ValueError('unknown GPU utilization status')
    if extended and terminal is not None:
        if terminal['attempted_steps']!=attempted or terminal['failure_kind'] not in kinds:
            raise ValueError('terminal step/failure accounting mismatch')
        if (terminal['status']=='completed')!=(terminal['failure_kind'] is None):
            raise ValueError('terminal success/failure category mismatch')


def _changes(points):
    values=[p['value'] for p in points]
    return dict(status='observed' if values else 'unsupported',samples=len(values),timeline=points,
        baseline=values[0] if values else None,end=values[-1] if values else None,
        delta=values[-1]-values[0] if values else None,
        sampled_min=min(values) if values else None,sampled_max=max(values) if values else None)


def _diagnostics(evidence, analysis):
    """No queries, interpolation, causal diagnosis or duration extrapolation."""
    rows=evidence['rows'];header=evidence['header'];terminal=evidence['terminal'];controller=evidence['controller']
    if header is None:
        return dict(core_status='not_executed',reason=analysis['reason'],
                    failure_kind=evidence.get('worker_failure',{}).get('kind'),platform_support='owned_worker_windows_only')
    interval_ns=int(header['protocol']['interval_seconds']*1e9)
    performance=[];eligible=[];thermal_pairs=[]
    for a,b in zip(rows,rows[1:]):
        span=b['timestamp_ns']-a['timestamp_ns'];steps=b['steps']-a['steps'];completed=b['completed']-a['completed']
        work=b['work_ns']-a['work_ns']
        in_work=a['phase']=='work' and b['phase'] in ('work','terminal','failure')
        item=dict(start_ns=a['timestamp_ns'],end_ns=b['timestamp_ns'],seconds=span/1e9,
            phase=b['phase'],completed=completed,steps=steps,work_ns=work,
            completed_per_second=completed*1e9/span,mean_step_ms=work/steps/1e6 if steps else None,
            gap=span>2*interval_ns,work_interval=in_work)
        performance.append(item)
        if in_work and not item['gap'] and steps:
            eligible.append(item)
            gpu=b['gpu']
            if (gpu.get('temperature_c') is not None and
                    a['timestamp_ns']<=gpu['query_start_ns']<=gpu['query_end_ns']<=b['timestamp_ns']):
                thermal_pairs.append(dict(temperature_c=gpu['temperature_c'],completed_per_second=item['completed_per_second'],
                    query_start_ns=gpu['query_start_ns'],query_end_ns=gpu['query_end_ns'],interval_end_ns=b['timestamp_ns']))
    drift=dict(status='insufficient_samples',reason='requires_6_non_gap_work_intervals',
        throughput_change_fraction=None,mean_step_change_fraction=None,indicator=None)
    if len(eligible)>=6:
        split=len(eligible)//2
        def window(items):
            seconds=sum(p['seconds'] for p in items);steps=sum(p['steps'] for p in items)
            return dict(intervals=len(items),observed_seconds=seconds,
                completed_per_second=sum(p['completed'] for p in items)/seconds,
                mean_step_ms=sum(p['work_ns'] for p in items)/steps/1e6)
        early,late=window(eligible[:split]),window(eligible[split:])
        rate=late['completed_per_second']/early['completed_per_second']-1
        step=late['mean_step_ms']/early['mean_step_ms']-1 if early['mean_step_ms'] else None
        drift=dict(status='descriptive_only',reason='no_significance_or_causal_claim',early=early,late=late,
            throughput_change_fraction=rate,mean_step_change_fraction=step,
            indicator='observed_slowdown' if rate<0 and step is not None and step>0 else 'mixed_or_no_slowdown')
    timing=rows[-1].get('step_timing') if rows else None
    fluctuation=dict(status='unsupported',reason='legacy_journal_has_no_step_moments',step_count=None,
        mean_ms=None,stddev_ms=None,coefficient_of_variation=None,min_ms=None,max_ms=None,
        p95_ms=None,p99_ms=None,scope='successful_workload_step_calls_including_their_sync_and_validation')
    if timing is not None:
        count=timing['count'];mean=timing['mean_ns'];sd=math.sqrt(timing['m2_ns2']/count) if count else None
        fluctuation.update(status='measured' if count else 'not_executed',reason='quantiles_not_recorded',step_count=count,
            mean_ms=mean/1e6 if count else None,stddev_ms=sd/1e6 if count else None,
            coefficient_of_variation=sd/mean if mean else None,
            min_ms=timing['min_ns']/1e6 if count else None,max_ms=timing['max_ns']/1e6 if count else None)
    memory={name:_changes([dict(timestamp_ns=r['rss']['timestamp_ns'],value=r['rss']['rss_bytes'])
                          for r in rows if r['rss'] is not None]) for name in ('rss_bytes',)}
    for name in ('allocated_bytes','reserved_bytes'):
        memory[name]=_changes([dict(timestamp_ns=(r['allocator']['query_start_ns']+r['allocator']['query_end_ns'])//2
                                   if 'query_start_ns' in r['allocator'] else r['timestamp_ns'],value=r['allocator'][name],
                                   time_scope='host_query_bracket' if 'query_start_ns' in r['allocator'] else 'legacy_parent_sample_association')
                               for r in rows if r['allocator'] is not None])
    gpu_timeline=[dict(index=r['index'],observation=r['gpu_utilization']) for r in rows if 'gpu_utilization' in r]
    temperatures=[dict(index=r['index'],query_start_ns=r['gpu']['query_start_ns'],query_end_ns=r['gpu']['query_end_ns'],
                       temperature_c=r['gpu']['temperature_c'],thermal_slowdown=r['gpu'].get('thermal_slowdown'))
                  for r in rows if r['gpu'].get('temperature_c') is not None]
    thermal=dict(status='insufficient_samples',paired_samples=thermal_pairs,pearson_r=None,
                 interpretation='association_only_not_throttling_or_cause')
    if len(thermal_pairs)>=5:
        xs=[p['temperature_c'] for p in thermal_pairs];ys=[p['completed_per_second'] for p in thermal_pairs]
        xbar=sum(xs)/len(xs);ybar=sum(ys)/len(ys)
        xx=sum((x-xbar)**2 for x in xs);yy=sum((y-ybar)**2 for y in ys)
        if xx and yy:
            thermal.update(status='descriptive_only',pearson_r=sum((x-xbar)*(y-ybar) for x,y in zip(xs,ys))/math.sqrt(xx*yy))
        else:thermal['status']='insufficient_variation'
    attempted=rows[-1].get('attempted_steps') if rows else None
    failure_kind=(terminal.get('failure_kind') if terminal else None) or evidence.get('worker_failure',{}).get('kind')
    if controller['reason']:interruption=controller['reason']
    elif controller['exit_code']!=0:
        interruption='abnormal_native_exit' if controller['exit_code']<0 or controller['exit_code']&0xffffffff>=0x80000000 else 'worker_exit_without_terminal'
    else:interruption=None
    result=dict(core_status='complete' if analysis['status']=='completed' else 'partial',advanced_status='partial',
        observed_work_seconds=analysis['duration_seconds'],requested_work_seconds=analysis['requested_duration_seconds'],
        verified_sample_span_seconds=(rows[-1]['timestamp_ns']-rows[0]['timestamp_ns'])/1e9 if len(rows)>1 else None,
        controller_elapsed_seconds=controller['elapsed_seconds'],completed=analysis['completed'],successful_steps=analysis['steps'],
        attempted_steps=attempted,observed_failed_steps=attempted-analysis['steps'] if attempted is not None else None,
        failure_kind=failure_kind,interruption=interruption,performance_timeline=performance,step_fluctuation=fluctuation,
        drift=drift,memory=memory,cpu_utilization_timeline=analysis.get('cpu_timeline',[]),gpu_utilization_timeline=gpu_timeline,
        thermal_observations=temperatures,thermal_association=thermal,
        configured_memory_limits=controller.get('memory_limits'),
        configured_process_limit=controller.get('process_limit'),
        sample_coverage=dict(requested_interval_seconds=header['protocol']['interval_seconds'],samples=len(rows),
            gaps=analysis['gaps'],missing_rss_samples=sum(r['rss'] is None for r in rows),
            missing_cpu_samples=sum(r['cpu'] is None or r['cpu'].get('status')!='measured' for r in rows),
            allocator_status='not_executed_cpu' if header['workload']['device_id']=='cpu' else 'boundary_samples',
            missing_allocator_samples=sum(r['allocator'] is None for r in rows) if header['workload']['device_id']!='cpu' else None,
            gpu_utilization_status='not_requested_cpu' if header['workload']['device_id']=='cpu' else 'optional_physical_device_sensor',
            missing_gpu_utilization_samples=sum(r.get('gpu_utilization',{}).get('status')!='measured' or
                r['gpu_utilization']['values']['gpu_busy_percent'] is None for r in rows)
                if header['workload']['device_id']!='cpu' else None),
        resource_policy='advisory_unless_explicit_limits_requested',platform_support='owned_worker_windows_only; verified_artifact_analysis_portable',
        limitations=['Only committed journal samples count; in-flight work at interruption is unknown.',
            'Duration covers the observed window only; no hours-long stability extrapolation.',
            'Wall throughput includes health queries, validation, serialization and scheduling; total overhead uncalibrated.',
            'Step timings exclude between-step health/journal work, but include workload-owned synchronization and validation.',
            'GPU utilization and temperatures describe the physical device including other applications; CLI sensors are optional.',
            'Boundary allocator/RSS samples can miss transients; memory deltas do not establish a leak or slowdown cause.',
            'Drift excludes gap/idle intervals without filling missing data; population step dispersion is not a latency quantile.',
            'Thermal pairs use host query brackets; sensor refresh latency is unknown; association is not causation.'])
    if header.get('stability_version')==2:
        from ._stability_windows import _window_diagnostics
        result.update(_window_diagnostics(evidence,analysis))
    return result
