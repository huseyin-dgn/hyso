"""P31 exact completed-step windows and process private commit; no leak model."""
from functools import lru_cache
import math
import os
import time

from .runtime_result import _int, _number


@lru_cache(maxsize=1)
def _commit_backend():
    import ctypes as c
    from ctypes import wintypes as w
    from .memory import _windows_backend
    _,_,Base=_windows_backend()
    class Counters(Base):
        _fields_=[('PrivateUsage',c.c_size_t)]
    kernel=c.WinDLL('kernel32',use_last_error=True);api=c.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.argtypes=[];kernel.GetCurrentProcess.restype=w.HANDLE
    api.GetProcessMemoryInfo.argtypes=[w.HANDLE,c.POINTER(Counters),w.DWORD]
    api.GetProcessMemoryInfo.restype=w.BOOL
    return kernel.GetCurrentProcess,api.GetProcessMemoryInfo,Counters


def _observe_commit():
    a=time.perf_counter_ns();value=None;reason=None
    try:
        if os.name!='nt':raise NotImplementedError('private commit adapter is Windows-only')
        import ctypes as c
        current,read,Counters=_commit_backend();data=Counters();data.cb=c.sizeof(data)
        if not read(current(),c.byref(data),c.sizeof(data)):raise OSError(c.get_last_error(),'GetProcessMemoryInfo')
        value=int(data.PrivateUsage)
    except (OSError,NotImplementedError) as error:reason=str(error)
    return dict(value=value,unit='byte',status='measured' if value is not None else 'unsupported',reason=reason,
        query_start_ns=a,query_end_ns=time.perf_counter_ns(),backend='Windows.PROCESS_MEMORY_COUNTERS_EX.PrivateUsage',
        scope='current_worker_process_private_commit_not_RSS_or_job_total')


def _quantiles(durations):
    ordered=sorted(durations);n=len(ordered)
    def q(f):
        pos=(n-1)*f;lo=int(pos);hi=min(lo+1,n-1)
        return (ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo))/1e6
    return dict(count=n,mean_ms=sum(ordered)/n/1e6 if n else None,p50_ms=q(.5) if n else None,
        p95_ms=q(.95) if n else None,p99_ms=q(.99) if n>=100 else None,
        p99_reason=None if n>=100 else 'requires_at_least_100_completed_step_samples',
        status='measured' if n else 'unavailable',reason=None if n else 'no_completed_steps',
        method='exact_empirical_linear_interpolation_at_(n-1)*q',
        scope='successful_step_host_brackets_including_sync_and_validation_no_population_tail_claim')


def _validate_windows(e):
    header=e['header'];rows=e['rows'];terminal=e['terminal']
    if header['clock_id']!=e['request']['token']+':perf_counter_ns':raise ValueError('P31 host clock association mismatch')
    previous_steps=previous_images=previous_work=0;previous_stamp=0;last_completion=None;phase='idle'
    for row in rows:
        _int(row['index'])
        if row.get('clock_id',header['clock_id'])!=header['clock_id']:raise ValueError('mixed P31 clock domains')
        if phase in ('terminal','failure') or (phase=='work' and row['phase']=='idle'):
            raise ValueError('invalid P31 phase order')
        phase=row['phase'];records=row['window_steps']
        if type(records) is not list:raise ValueError('exact step list required')
        end=previous_stamp;images=work=0
        for record in records:
            if type(record) is not list or len(record)!=3:raise ValueError('invalid completed-step receipt')
            a,b,count=record;_int(a,end);_int(b,a);_int(count,1)
            if b>row['timestamp_ns']:raise ValueError('step completion after observation')
            if header['workload']['kind']=='model_inference' and count!=header['workload']['input_shape'][0]:
                raise ValueError('completed step batch mismatch')
            end=b;images+=count;work+=b-a
        if (len(records)!=row['steps']-previous_steps or images!=row['completed']-previous_images
                or work!=row['work_ns']-previous_work):raise ValueError('exact step/cumulative ledger mismatch')
        expected_completion=end if records else last_completion
        if row['last_completion_ns']!=expected_completion:raise ValueError('last completion differs from actual step evidence')
        failed=row['failed_step']
        if failed is not None:
            _int(failed['start_ns'],end);_int(failed['end_ns'],failed['start_ns'])
            if (failed['end_ns']>row['timestamp_ns'] or failed['kind']!=row['failure_kind']
                    or row['attempted_steps']!=row['steps']+1 or row['phase']!='failure'):
                raise ValueError('invalid failed-step bracket')
        elif row['attempted_steps']!=row['steps']:raise ValueError('attempted failure without failure bracket')
        commit=row['process_commit'];_int(commit['query_start_ns']);_int(commit['query_end_ns'],commit['query_start_ns'])
        if commit['query_end_ns']>row['timestamp_ns'] or commit['unit']!='byte':raise ValueError('private commit clock/unit mismatch')
        if commit['status']=='measured':
            _int(commit['value'])
            if commit['reason'] is not None:raise ValueError('measured commit with unavailable reason')
        elif commit['status']=='unsupported':
            if commit['value'] is not None or not commit['reason']:raise ValueError('missing private commit needs null/reason')
        else:raise ValueError('unknown private commit status')
        previous_stamp=row['timestamp_ns'];previous_steps=row['steps'];previous_images=row['completed'];previous_work=row['work_ns']
        last_completion=expected_completion
    if terminal is not None:
        _int(terminal['close_start_ns'],terminal['work_end_ns']);_int(terminal['close_end_ns'],terminal['close_start_ns'])
        if terminal['status'] not in ('completed','failed') or type(terminal['closed']) is not bool:
            raise ValueError('invalid P31 terminal disposition')
        if rows and terminal['work_end_ns']>rows[-1]['timestamp_ns']:raise ValueError('work end after final observation')
        if terminal['status']=='completed':
            if not rows or rows[-1]['phase']!='terminal' or terminal['reason'] is not None or terminal['work_end_ns']!=last_completion:
                raise ValueError('success lacks exact final completion boundary')
            first_work=next((r for r in rows if r['phase']=='work'),None)
            if first_work is None or terminal['work_start_ns']!=first_work['timestamp_ns']:
                raise ValueError('work start differs from initial observation boundary')


def _window_diagnostics(e,analysis):
    from ._stability_diagnostics import _changes
    rows=e['rows'];all_steps=[];timeline=[];events=[];previous_stamp=None
    for row in rows:
        steps=row['window_steps'];all_steps.extend(steps)
        durations=[b-a for a,b,_ in steps]
        span=(row['timestamp_ns']-previous_stamp)/1e9 if previous_stamp is not None else None
        timeline.append(dict(index=row['index'],start_ns=previous_stamp,end_ns=row['timestamp_ns'],phase=row['phase'],
            completed_inference_calls=len(steps),completed_images=sum(r[2] for r in steps),latency=_quantiles(durations),
            completion_window_images_per_second=sum(r[2] for r in steps)/span if span and span>0 else None,
            seconds=span,gap=bool(span and span>2*e['header']['protocol']['interval_seconds'])))
        if row['failure'] is not None:
            events.append(dict(event='failure',timestamp_ns=row['timestamp_ns'],kind=row['failure_kind'],
                reason=row['failure'],failed_step=row['failed_step']))
        previous_stamp=row['timestamp_ns']
    terminal=e['terminal'];controller=e['controller'];all_latency=_quantiles([b-a for a,b,_ in all_steps])
    completed_span=(all_steps[-1][1]-all_steps[0][0])/1e9 if all_steps else None
    commit=[dict(timestamp_ns=(r['process_commit']['query_start_ns']+r['process_commit']['query_end_ns'])//2,
                 value=r['process_commit']['value']) for r in rows if r['process_commit']['status']=='measured']
    if terminal is not None:
        events.append(dict(event='worker_close_and_terminal',timestamp_ns=terminal['close_end_ns'],
            closed=terminal['closed'],status=terminal['status'],reason=terminal['reason'],failure_kind=terminal['failure_kind'],
            close_start_ns=terminal['close_start_ns'],time_scope='actual_close_host_bracket'))
    events.append(dict(event='controller_cleanup',clock_id=e['request']['token']+':controller_perf_counter_ns',
        timestamp_ns=controller.get('lifecycle',{}).get('cleanup_observed_ns'),reason=controller['reason'],exit_code=controller['exit_code'],
        cleanup=controller['cleanup'],time_scope='controller_domain_cleanup_observation'))
    return dict(latency=all_latency,latency_throughput_windows=timeline,
        completion_window=dict(start_ns=all_steps[0][0] if all_steps else None,end_ns=all_steps[-1][1] if all_steps else None,
            seconds=completed_span,completed_calls=len(all_steps),completed_images=analysis['completed'],
            images_per_second=analysis['completed']/completed_span if completed_span and completed_span>0 else None,
            status='complete_run' if analysis['status']=='completed' else 'committed_prefix_only'),
        private_commit=dict(_changes(commit),reason=None if commit else 'no_native_private_commit_samples'),resource_monitor=controller.get('resource_monitor'),
        resource_policy='stop_owned_job_on_Windows_low_memory_notification_no_implicit_RAM_VRAM_quota',
        events=events,recovery=dict(execution_resume=False,automatic_retry=False,
            status='unsupported',reason='reopen_validates_evidence_only_no_execution_recovery'),
        continuity=dict(requested_duration_satisfied=analysis['status']=='completed',
            no_sampling_gaps_detected=not analysis['gaps'],single_worker_pid=e['header']['pid'],
            no_failures_observed=not any(r['failure'] is not None for r in rows) and (terminal is None or terminal['reason'] is None),
            scope='one_worker_observed_host_work_window_includes_between_step_monitoring_not_continuous_kernel_activity'),
        missing_measurements=dict(process_commit_samples=sum(r['process_commit']['value'] is None for r in rows),
            gpu_temperature=dict(value=None,reason='GPU_not_requested_for_CPU_workload') if e['header']['workload']['device_id']=='cpu' else None,
            gpu_power=dict(value=None,reason='GPU_not_requested_for_CPU_workload') if e['header']['workload']['device_id']=='cpu' else None),
        clock_domains=dict(worker=e['header']['clock_id'],controller=e['request']['token']+':controller_perf_counter_ns',
            cross_domain_duration='not_computed'))
