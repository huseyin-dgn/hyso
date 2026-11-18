"""P31 arithmetic, bounded recording, opt-in resources and failure receipts."""
from dataclasses import asdict
import copy
import csv
import math
import os
from pathlib import Path

import pytest
import torch

from hyso.performance import StabilityProtocol,WorkloadRecipe,InferenceWorkload,measure_stability,RuntimeResult
from hyso.performance.stability import _analyze_stability,_failure_kind
from hyso.performance.runtime_result import _hash,_result


def seal(e):
    previous=_hash(e['header']);chunks=[]
    for i in range(0,len(e['rows']),2):
        rows=e['rows'][i:i+2];index=len(chunks)
        previous=_hash(dict(schema_version=1,run_id=e['header']['run_id'],index=index,
                            previous_sha256=previous,rows=rows))
        chunks.append(dict(name=f'chunk-{index:06d}.json',sha256=previous,rows=len(rows)))
    e['chunks']=chunks
    if e['terminal'] is not None:
        e['terminal'].update(chunks=chunks,last_chunk_sha256=previous,rows=len(e['rows']))
    return e


def evidence():
    # Synthetic receipts for exact arithmetic only; this is not native runtime evidence.
    token='synthetic-arithmetic';gpu='GPU-11111111-1111-1111-1111-111111111111'
    protocol=asdict(StabilityProtocol(duration_seconds=8,interval_seconds=1,chunk_rows=2))
    recipe=asdict(WorkloadRecipe('runtime_support:make_cpu',gpu_uuid=gpu))
    header=dict(run_id=token,pid=123,clock_id=token+':clock',protocol=protocol,
        source=dict(factory=recipe['factory'],configuration={}),
        workload=dict(kind='synthetic_steps',device_id=gpu,unit='operation'),runtime=dict(logical_cores=4))
    rows=[];steps=work=0;mean=m2=0.;durations=[]
    for i in range(9):
        if i:
            for _ in range(10 if i<=4 else 5):
                duration=50000000 if i<=4 else 150000000
                steps+=1;work+=duration;delta=duration-mean;mean+=delta/steps;m2+=delta*(duration-mean)
                durations.append(duration)
        stamp=(i+1)*10**9
        power=dict(status='measured',reason=None,device_id=gpu,clock_id=header['clock_id'],
            query_start_ns=stamp-10,query_end_ns=stamp-2,timestamp_ns=stamp-6,watts=10.,unit='W',
            temperature_c=50.+i,thermal_slowdown='Not Active',backend='nvidia-smi.power.draw.instant',
            scope='physical_board',sensor_accuracy_watts=5.)
        rows.append(dict(index=i,timestamp_ns=stamp,phase='terminal' if i==8 else 'work',
            completed=steps,steps=steps,work_ns=work,last_completion_ns=stamp-1 if steps else None,
            rss=None,cpu=None,allocator=None,gpu=power,retention=None,failure=None,
            attempted_steps=steps,failure_kind=None,
            step_timing=dict(count=steps,mean_ns=mean if steps else None,m2_ns2=m2,
                             min_ns=min(durations) if durations else None,max_ns=max(durations) if durations else None),
            gpu_utilization=dict(status='unsupported',values=None,query_start_ns=stamp-20,query_end_ns=stamp-11)))
    terminal=dict(status='completed',reason=None,closed=True,work_start_ns=10**9,work_end_ns=9*10**9,
        completed=steps,steps=steps,attempted_steps=steps,failure_kind=None)
    return seal(dict(request=dict(token=token,recipe=recipe,protocol=protocol),header=header,rows=rows,
        terminal=terminal,controller=dict(token=token,launcher_pid=123,exit_code=0,reason=None,
            cleanup=dict(active_processes=0),kill_on_close=True,elapsed_seconds=9.),incomplete_tail=False))


def test_exact_drift_dispersion_thermal_association_without_cause():
    result=_result('P31',evidence());d=result.stability_diagnostics()
    assert d['core_status']=='complete' and d['advanced_status']=='partial'
    assert d['observed_work_seconds']==8 and d['completed']==d['successful_steps']==60
    assert d['drift']['throughput_change_fraction']==-.5
    assert d['drift']['mean_step_change_fraction']==pytest.approx(2.)
    assert d['drift']['indicator']=='observed_slowdown'
    assert d['step_fluctuation']['mean_ms']==pytest.approx(5000/60)
    assert d['step_fluctuation']['stddev_ms']==pytest.approx(math.sqrt(20000/9))
    assert d['step_fluctuation']['min_ms']==50 and d['step_fluctuation']['max_ms']==150
    assert d['step_fluctuation']['p99_ms'] is None
    assert d['thermal_association']['pearson_r']<0
    assert d['thermal_association']['interpretation']=='association_only_not_throttling_or_cause'
    assert d['observed_failed_steps']==0
    assert d['memory']['rss_bytes']['delta'] is None


def test_successful_sensor_query_with_na_utilization_is_still_missing():
    e=evidence();e['rows'][-1]['gpu_utilization'].update(status='measured',values=dict(
        uuid=e['header']['workload']['device_id'],gpu_busy_percent=None,memory_controller_busy_percent=None,
        memory_used_bytes=None,memory_free_bytes=None,memory_total_bytes=None))
    result=_result('P31',seal(e))
    assert result.stability_diagnostics()['sample_coverage']['missing_gpu_utilization_samples']==9


@pytest.mark.parametrize('damage',['mean','variance','count','attempted','gpu_uuid','gpu_percent',
                                  'terminal_failure','row_failure','inflight_success'])
def test_rehashed_new_samples_cannot_invent_measurements(damage):
    e=evidence();row=e['rows'][-1]
    if damage=='mean':row['step_timing']['mean_ns']*=2
    elif damage=='variance':row['step_timing']['m2_ns2']=-1
    elif damage=='count':row['step_timing']['count']+=1
    elif damage=='attempted':row['attempted_steps']=row['steps']+2
    elif damage=='terminal_failure':e['terminal']['failure_kind']='oom'
    elif damage=='row_failure':row['failure_kind']='device_error'
    elif damage=='inflight_success':row['attempted_steps']+=1;e['terminal']['attempted_steps']+=1
    else:
        row['gpu_utilization'].update(status='measured',values=dict(uuid=e['header']['workload']['device_id'],
            gpu_busy_percent=10.,memory_controller_busy_percent=5.,memory_used_bytes=10.,memory_free_bytes=90.,memory_total_bytes=100.))
        if damage=='gpu_uuid':row['gpu_utilization']['values']['uuid']='foreign'
        else:row['gpu_utilization']['values']['gpu_busy_percent']=101.
    with pytest.raises(ValueError):_analyze_stability(seal(e))


def test_gaps_and_interrupted_prefix_never_interpolate_or_extrapolate():
    e=evidence()
    for row in e['rows'][4:]:
        row['timestamp_ns']+=3*10**9
        if row['last_completion_ns'] is not None:row['last_completion_ns']+=3*10**9
    e['terminal']['work_end_ns']+=3*10**9
    d=_result('P31',seal(e)).stability_diagnostics()
    assert len(d['performance_timeline'])==8
    assert sum(r['gap'] for r in d['performance_timeline'])==1
    assert d['drift']['early']['observed_seconds']+d['drift']['late']['observed_seconds']==7
    e['terminal']=None;e['incomplete_tail']=True;e['controller']['reason']='heartbeat_timeout'
    result=_result('P31',seal(e));d=result.stability_diagnostics()
    assert d['core_status']=='partial' and d['observed_work_seconds'] is None
    assert d['interruption']=='heartbeat_timeout' and d['completed']==60


def test_legacy_json_csv_contract_and_missing_moments(tmp_path):
    e=evidence()
    for config in (e['request']['protocol'],e['header']['protocol']):
        config.pop('process_memory_bytes',None);config.pop('job_memory_bytes',None)
    for row in e['rows']:
        for field in ('step_timing','attempted_steps','failure_kind','gpu_utilization'):row.pop(field)
    e['terminal'].pop('attempted_steps');e['terminal'].pop('failure_kind')
    result=_result('P31',seal(e));original=result.to_dict()
    result.to_json(tmp_path/'legacy.json');result.to_csv(tmp_path/'legacy.csv')
    assert RuntimeResult.from_json(tmp_path/'legacy.json').to_dict()==original
    with (tmp_path/'legacy.csv').open(newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream)
        assert reader.fieldnames==['feature_id','result_id','source_sha256','field','value']
        assert any(r['field']=='analysis.completed' and r['value']=='60' for r in reader)
    d=result.stability_diagnostics()
    assert d['step_fluctuation']['status']=='unsupported' and d['attempted_steps'] is None
    assert result.to_dict()==original


def test_default_resources_and_cpu_integrity_without_cuda(monkeypatch):
    protocol=StabilityProtocol()
    assert protocol.process_memory_bytes is protocol.job_memory_bytes is protocol.max_processes is None
    def forbidden(*a,**kw):raise AssertionError('CPU helper must not initialize or limit CUDA')
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',forbidden)
    model=torch.nn.Linear(3,2);before=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
    work=InferenceWorkload(model,torch.ones(2,3));assert work.step()==2;work.close()
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    with pytest.raises(ValueError):InferenceWorkload(model,torch.ones(2,3),cuda_memory_fraction=.1)


@pytest.mark.parametrize('fraction',[None,.2])
def test_cuda_fraction_is_set_only_on_explicit_request_before_model_copy(fraction,monkeypatch):
    import copy
    gpu='GPU-11111111-1111-1111-1111-111111111111';calls=[]
    model=torch.nn.Linear(3,2);inputs=torch.ones(2,3)
    monkeypatch.setenv('HYSO_RUNTIME_ADMITTED_PID',str(os.getpid()))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',gpu)
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda value,device:calls.append(value))
    class AtCopy(Exception):pass
    def stop(value):raise AtCopy()
    monkeypatch.setattr(copy,'deepcopy',stop)
    with pytest.raises(AtCopy):InferenceWorkload(model,inputs,device='cuda:0',gpu_uuid=gpu,cuda_memory_fraction=fraction)
    assert calls==([] if fraction is None else [fraction])


@pytest.mark.parametrize('error,kind',[(MemoryError('bounded injected exception'),'oom'),
    (torch.cuda.OutOfMemoryError('injected without CUDA allocation'),'oom'),
    (RuntimeError('CUDA error: device-side assert triggered'),'device_error'),
    (RuntimeError('unrelated failure'),'workload_exception')])
def test_failure_categories_do_not_use_occupancy(error,kind):
    assert _failure_kind(error)==kind


@pytest.mark.parametrize('mode,kind',[('oom','oom'),('device_error','device_error'),('setup_oom','oom')])
def test_real_owned_exception_receipts_without_allocation_stress(mode,kind,tmp_path):
    recipe=WorkloadRecipe('runtime_support:make_cpu',dict(mode=mode),search_path=str(Path(__file__).parent.resolve()))
    result=measure_stability(recipe,tmp_path/'journal',protocol=StabilityProtocol(duration_seconds=.1,
        interval_seconds=.02,chunk_rows=1,process_memory_bytes=4*1024**3,job_memory_bytes=6*1024**3,max_processes=4))
    d=result.stability_diagnostics()
    assert result.to_dict()['analysis']['status']=='aborted' and d['failure_kind']==kind
    if mode!='setup_oom':assert d['attempted_steps']==1 and d['observed_failed_steps']==1
    assert result.to_dict()['evidence']['controller']['cleanup']['active_processes']==0


def test_real_cpu_slowdown_is_descriptive_with_bounded_moments(tmp_path):
    recipe=WorkloadRecipe('runtime_support:make_cpu',dict(mode='slowdown',delay=.001),
        search_path=str(Path(__file__).parent.resolve()))
    result=measure_stability(recipe,tmp_path/'journal',protocol=StabilityProtocol(duration_seconds=1.,
        interval_seconds=.03,chunk_rows=2,process_memory_bytes=4*1024**3,job_memory_bytes=6*1024**3,max_processes=4))
    d=result.stability_diagnostics()
    assert d['core_status']=='complete'
    assert d['drift']['status']=='descriptive_only' and d['drift']['indicator']=='observed_slowdown'
    assert d['step_fluctuation']['max_ms']>d['step_fluctuation']['min_ms']
    assert d['observed_work_seconds']>=1 and d['advanced_status']=='partial'
    assert len(result.to_dict()['evidence']['rows'])<=10000


def test_controller_interrupt_preserves_cleanup_and_verified_prefix(tmp_path,monkeypatch):
    from hyso.performance import stability,reopen_stability
    root=tmp_path/'journal';native_sleep=stability.time.sleep;fired=[]
    def interrupt_after_commit(seconds):
        if not fired and (root/'chunk-000001.json').exists():
            fired.append(True)
            raise KeyboardInterrupt('controlled parent interruption')
        native_sleep(seconds)
    monkeypatch.setattr(stability.time,'sleep',interrupt_after_commit)
    recipe=WorkloadRecipe('runtime_support:make_cpu',search_path=str(Path(__file__).parent.resolve()))
    with pytest.raises(KeyboardInterrupt):
        measure_stability(recipe,root,protocol=StabilityProtocol(duration_seconds=2,interval_seconds=.02,chunk_rows=1,
            process_memory_bytes=4*1024**3,job_memory_bytes=6*1024**3,max_processes=4))
    result=reopen_stability(root);data=result.to_dict();d=result.stability_diagnostics()
    assert fired and d['core_status']=='partial' and d['interruption']=='controller_interrupted'
    assert data['evidence']['controller']['cleanup']['active_processes']==0
    assert data['analysis']['coverage']=='committed_prefix_only' and len(data['evidence']['chunks'])>=2


def test_unlimited_job_retains_only_ownership_flag():
    import ctypes as c
    from hyso.performance._training_process import _Job
    job=_Job(None,None,None)
    try:
        limits=job.Extended()
        assert job.api.QueryInformationJobObject(job.handle,9,c.byref(limits),c.sizeof(limits),None)
        assert limits.basic.flags&0x2000
        assert not limits.basic.flags&(0x100|0x200|0x8)
    finally:job.close()


def test_launch_error_preserves_original_exception_and_closes_job(tmp_path,monkeypatch):
    from hyso.performance import stability,_training_process
    calls=[]
    class Job:
        def __init__(self,*limits):pass
        def terminate(self):calls.append('terminate')
        def observation(self):return dict(active_processes=0)
        def close(self):calls.append('close')
    def refused(*a,**kw):raise OSError('controlled launch refusal')
    monkeypatch.setattr(_training_process,'_Job',Job)
    monkeypatch.setattr(stability.subprocess,'Popen',refused)
    with pytest.raises(OSError,match='controlled launch refusal'):
        stability._run_owned(tmp_path/'journal','stability',dict(recipe=asdict(WorkloadRecipe('runtime_support:make_cpu')),
            protocol=asdict(StabilityProtocol())),startup_timeout=1,heartbeat_timeout=1,max_runtime=2,max_bytes=65536)
    assert calls==['terminate','close']
