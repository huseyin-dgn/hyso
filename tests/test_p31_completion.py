"""P31 exact-window arithmetic and bounded CPU regressions, never long-run proof."""
import copy
from pathlib import Path

import pytest

from hyso.performance import RuntimeResult,StabilityProtocol,WorkloadRecipe,measure_stability
from hyso.performance.runtime_result import _result
from hyso.performance.stability import _analyze_stability
from hyso.performance._stability_windows import _quantiles
from test_stability_performance import evidence,seal


@pytest.fixture
def exact():
    e=evidence();h=e['header'];h['stability_version']=2;h['clock_id']=h['run_id']+':perf_counter_ns'
    for p in (e['request']['protocol'],h['protocol']):p['duration_seconds']=7.5
    previous_steps=0
    for row in e['rows']:
        row['gpu']['clock_id']=h['clock_id'];n=row['steps']-previous_steps
        duration=50_000_000 if row['index']<=4 else 150_000_000
        start=row['timestamp_ns']-1000-n*duration
        records=[[start+j*duration,start+(j+1)*duration,1] for j in range(n)]
        row.update(window_steps=records,failed_step=None,process_commit=dict(value=12345,unit='byte',status='measured',
            reason=None,query_start_ns=row['timestamp_ns']-100,query_end_ns=row['timestamp_ns']-50))
        row['last_completion_ns']=records[-1][1] if records else None
        previous_steps=row['steps']
    e['terminal'].update(work_end_ns=e['rows'][-1]['last_completion_ns'],close_start_ns=9*10**9+1,close_end_ns=9*10**9+2)
    return seal(e)


def test_exact_latency_windows_counts_and_completion_window(exact):
    r=_result('P31',exact);d=r.stability_diagnostics()
    assert r.to_dict()['analysis']['status']=='completed'
    assert d['latency']['count']==60 and d['latency']['p50_ms']==50 and d['latency']['p95_ms']==150
    assert d['latency']['p99_ms'] is None and d['latency']['p99_reason']
    assert sum(w['completed_inference_calls'] for w in d['latency_throughput_windows'])==60
    assert d['latency_throughput_windows'][1]['latency']['mean_ms']==50
    assert d['latency_throughput_windows'][5]['latency']['mean_ms']==150
    assert d['completion_window']['completed_calls']==d['completion_window']['completed_images']==60
    assert d['completion_window']['seconds']==pytest.approx(7.5)
    assert d['private_commit']['baseline']==d['private_commit']['end']==12345
    assert d['recovery']['automatic_retry'] is False and d['recovery']['execution_resume'] is False


def test_independent_empirical_p99_and_null_tail_policy():
    d=_quantiles([i*10**6 for i in range(100)])
    assert d['mean_ms']==d['p50_ms']==49.5
    assert d['p95_ms']==pytest.approx(94.05) and d['p99_ms']==pytest.approx(98.01)
    assert _quantiles([1]*99)['p99_ms'] is None
    assert _quantiles([])['mean_ms'] is None


@pytest.mark.parametrize('damage',['missing','duplicate','reverse','outside','count','last_completion','clock',
    'commit_unit','commit_time','commit_missing','phase','work_end','close_order'])
def test_rehashed_windows_reject_contradictory_evidence(exact,damage):
    row=exact['rows'][1]
    if damage=='missing':row['window_steps'].pop()
    elif damage=='duplicate':row['window_steps'].append(row['window_steps'][0])
    elif damage=='reverse':row['window_steps'][1][0]=0
    elif damage=='outside':row['window_steps'][-1][1]=row['timestamp_ns']+1
    elif damage=='count':row['window_steps'][0][2]=2
    elif damage=='last_completion':row['last_completion_ns']+=1
    elif damage=='clock':row['clock_id']='other'
    elif damage=='commit_unit':row['process_commit']['unit']='GiB'
    elif damage=='commit_time':row['process_commit']['query_end_ns']=row['timestamp_ns']+1
    elif damage=='commit_missing':row['process_commit'].update(status='unsupported',value=None,reason=None)
    elif damage=='phase':exact['rows'][2]['phase']='terminal'
    elif damage=='work_end':exact['terminal']['work_end_ns']+=1
    elif damage=='close_order':exact['terminal']['close_end_ns']=0
    with pytest.raises(ValueError):_analyze_stability(seal(exact))


def test_partial_windows_do_not_become_completed_and_new_csv_reopens(exact,tmp_path):
    exact['terminal']=None;exact['incomplete_tail']=True;exact['controller']['reason']='heartbeat_timeout'
    r=_result('P31',exact);a=r.to_dict()['analysis'];d=r.stability_diagnostics()
    assert a['status']=='aborted' and a['completed_per_second'] is None
    assert d['completion_window']['status']=='committed_prefix_only'
    assert d['interruption']=='heartbeat_timeout' and not d['continuity']['requested_duration_satisfied']
    r.to_json(tmp_path/'r.json');r.to_csv(tmp_path/'r.csv')
    assert RuntimeResult.from_json(tmp_path/'r.json').to_dict()==r.to_dict()
    assert RuntimeResult.from_csv(tmp_path/'r.csv').to_dict()==r.to_dict()


def test_unavailable_private_commit_is_null_with_reason(exact):
    for row in exact['rows']:row['process_commit'].update(status='unsupported',value=None,reason='injected_backend_unavailable')
    d=_result('P31',seal(exact)).stability_diagnostics()
    assert d['private_commit']['delta'] is None and d['private_commit']['reason']


def test_native_cpu_exact_windows_private_commit_and_lossless_exports(tmp_path):
    recipe=WorkloadRecipe('runtime_support:make_cpu',dict(delay=.002),search_path=str(Path(__file__).parent.resolve()))
    r=measure_stability(recipe,tmp_path/'journal',protocol=StabilityProtocol(duration_seconds=.25,interval_seconds=.02,chunk_rows=1))
    a=r.to_dict()['analysis'];d=r.stability_diagnostics();c=r.to_dict()['evidence']['controller']
    assert a['status']=='completed' and a['duration_seconds']>=.25
    assert d['latency']['count']==a['steps'] and d['completion_window']['completed_images']==a['completed']
    assert d['private_commit']['samples']>0 and c['resource_monitor']['samples']>0
    assert c['memory_limits']==dict(process_memory_bytes=None,job_memory_bytes=None)
    assert c['cleanup']['active_processes']==0 and d['missing_measurements']['gpu_power']['value'] is None
    for kind in ('json','csv'):
        path=tmp_path/('r.'+kind);getattr(r,'to_'+kind)(path)
        assert getattr(RuntimeResult,'from_'+kind)(path).to_dict()==r.to_dict()


def test_os_pressure_ends_only_owned_job_without_allocation_stress(monkeypatch,tmp_path):
    from hyso.performance._stream_resources import _StreamResources
    monkeypatch.setattr(_StreamResources,'sample',lambda self,job:True)
    recipe=WorkloadRecipe('runtime_support:make_cpu',search_path=str(Path(__file__).parent.resolve()))
    r=measure_stability(recipe,tmp_path/'journal');c=r.to_dict()['evidence']['controller']
    assert c['reason']=='os_low_memory' and c['cleanup']['active_processes']==0
    assert r.to_dict()['analysis']['status']=='aborted'
    r.to_csv(tmp_path/'aborted.csv')
    assert RuntimeResult.from_csv(tmp_path/'aborted.csv').to_dict()==r.to_dict()
