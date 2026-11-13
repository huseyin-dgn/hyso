"""P34 CPU/constructed contracts; simulated GPU copies are not native proof."""
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from hyso.performance import RuntimeResult
from hyso.performance.devices import _analyze_devices
from hyso.performance.runtime_result import _result
from hyso.performance._device_diagnostics import _capture_origin,_state_hash,_checkpoint_unchanged
from test_device_comparison import pair,cpu,rebind,GPU


def result(e):return _result('P34',e)


def test_pairwise_direction_math_and_distinct_denominators(pair):
    r=pair['runs'][1]['report']
    # Explicitly constructed interval duration scaling, not a real GPU result.
    window=r['measurement_window'];anchor=window['start_ns']
    for span in r['sample_intervals']:
        span['start_ns']=anchor+(span['start_ns']-anchor)*2
        span['end_ns']=anchor+(span['end_ns']-anchor)*2
    window['end_ns']=anchor+(window['end_ns']-anchor)*2
    r['latency_ns']=[x*2 for x in r['latency_ns']];r['window_ns']*=2
    value=result(pair);d=value.device_diagnostics();p=d['comparisons'][0]
    assert p['status']=='comparable' and p['latency_speedup']['value']==pytest.approx(.5)
    assert p['throughput_ratio']['value']==pytest.approx(.5)
    assert p['latency_difference_percent']['value']==pytest.approx(100)
    reverse=value.device_diagnostics(baseline_device=GPU)['comparisons'][0]
    assert reverse['latency_speedup']['value']==pytest.approx(2)
    assert reverse['latency_difference_percent']['value']==pytest.approx(-50)
    assert not p['trained_model_comparison']['comparable'] and d['ranking'] is None
    row=d['devices'][0];m=row['metrics']
    assert m['images_per_second']['value']==pytest.approx(row['completed_images']/row['duration_seconds'])
    assert m['completed_forward_throughput']['value']==pytest.approx(row['completed_forwards']/row['duration_seconds'])
    assert m['images_per_second']['value']!=pytest.approx(row['contract']['batch']*1000/m['mean_batch_latency']['value'])


@pytest.mark.parametrize('key',['model_id','loaded_state_sha256','preprocessing_id','preprocessing_sha256'])
def test_origin_mismatch_has_no_speedup(pair,key):
    pair['runs'][1]['report']['identity']['origin'][key]='b'*64 if key.endswith('sha256') else 'different'
    data=result(pair).to_dict()['analysis'];d=data['diagnostics'];p=d['comparisons'][0]
    assert not data['comparable'] and p['status']=='not_comparable'
    assert p['latency_speedup']['value'] is None and p['latency_speedup']['reason']
    assert key in p['differences']


def attest_constructed_origin(pair):
    for run in pair['runs']:
        origin=run['report']['identity']['origin']
        origin.update(kind='trained_checkpoint',checkpoint=dict(path='C:/CONSTRUCTED_ONLY/checkpoint.pth',sha256='a'*64),checkpoint_reason=None)


def test_matching_and_different_checkpoint(pair):
    attest_constructed_origin(pair)
    assert result(pair).device_diagnostics()['comparisons'][0]['trained_model_comparison']['comparable']
    pair['runs'][1]['report']['identity']['origin']['checkpoint']['sha256']='b'*64
    d=result(pair).device_diagnostics()['comparisons'][0]
    assert d['status']=='not_comparable' and 'checkpoint_sha256' in d['differences']


@pytest.mark.parametrize('key',['checkpoint','loaded_state_sha256','preprocessing_sha256'])
def test_missing_trained_provenance_rejected(pair,key):
    attest_constructed_origin(pair)
    pair['runs'][1]['report']['identity']['origin'][key]=None
    with pytest.raises(ValueError):result(pair)


@pytest.mark.parametrize('case',['clock','instance','window','order','duplicate_index','sample_duration','missing_intervals','missing_origin'])
def test_rehashed_timestamp_or_origin_corruption(pair,case):
    r=pair['runs'][1]['report'];w=r['measurement_window'];spans=r['sample_intervals']
    if case=='clock':w['clock']='cuda_event'
    elif case=='instance':w['clock_instance_id']='other_process'
    elif case=='window':w['end_ns']+=1
    elif case=='order':spans.reverse()
    elif case=='duplicate_index':spans[1]['index']=0
    elif case=='sample_duration':spans[0]['end_ns']+=1
    elif case=='missing_intervals':r['sample_intervals']=None
    else:del r['identity']['origin']
    with pytest.raises(ValueError):result(pair)


@pytest.mark.parametrize('kind',['timeout','exception','crash','partial','missing_report','cleanup'])
def test_failed_work_has_no_performance_or_retry(pair,kind):
    run=pair['runs'][1]
    if kind=='cleanup':
        run['controller']['cleanup']['active_processes']=1
        with pytest.raises(ValueError):result(pair)
        return
    if kind=='timeout':run['controller']['reason']='heartbeat_timeout'
    elif kind=='crash':run['controller']['exit_code']=0xc0000005
    elif kind=='exception':run['report'].update(status='failed',reason='controlled exception')
    elif kind=='partial':run['report'].update(status='partial',reason='incomplete')
    else:run['report']=None
    d=result(pair).device_diagnostics();p=d['comparisons'][0]
    assert p['status']=='not_comparable' and p['throughput_ratio']['value'] is None
    assert d['devices'][1]['metrics'] is None


def test_tail_policy_no_small_sample_p95_p99(pair):
    row=result(pair).device_diagnostics()['devices'][0]
    assert row['metrics']['p50']['value']==pytest.approx(np.median(pair['runs'][0]['report']['latency_ns'])/1e6)
    for key in ('p95','p99'):
        assert row['metrics'][key]['value'] is None and row['metrics'][key]['reason']


def test_large_constructed_tail_math(pair):
    pair['protocol']['samples']=1000
    for run in pair['runs']:
        run['protocol']['samples']=1000;r=run['report'];i=r['identity'];i['measured_samples']=1000
        r['latency_ns']=list(range(1,1001));r['window_ns']=1_000_000;r['completed']=1000*i['batch']
        w=r['measurement_window'];w.update(start_ns=0,end_ns=1_000_000);cursor=0;spans=[]
        for n,latency in enumerate(r['latency_ns']):
            spans.append(dict(index=n,start_ns=cursor,end_ns=cursor+latency));cursor+=latency
        r['sample_intervals']=spans;rebind(run)
    row=result(pair).device_diagnostics()['devices'][0]
    for key,q in (('p95',.95),('p99',.99)):
        assert row['metrics'][key]['value']==pytest.approx(np.quantile(np.arange(1,1001),q)/1e6)


def test_memory_missing_domain_and_sensor(pair):
    for run in pair['runs']:run['report']['memory']['logical_model_bytes']=None
    d=result(pair).device_diagnostics();p=d['comparisons'][0]
    assert p['status']=='comparable' and not p['memory']['logical_model_bytes']['comparable']
    assert not p['memory']['cuda_peak_allocated_bytes']['comparable']
    assert p['memory']['cuda_peak_allocated_bytes']['baseline_value'] is None
    assert all(row['missing_sensors']['energy']['value'] is None for row in d['devices'])


def test_lossless_and_hash_integrity(pair,tmp_path):
    value=result(pair)
    for suffix in ('json','csv'):
        p=tmp_path/('result.'+suffix);getattr(value,'to_'+suffix)(p)
        reopened=getattr(RuntimeResult,'from_'+suffix)(p)
        assert reopened.to_dict()==value.to_dict() and reopened.device_diagnostics()==value.device_diagnostics()
        with pytest.raises(FileExistsError):getattr(value,'to_'+suffix)(p)
    bad=value.to_dict();bad['evidence']['runs'][0]['report']['window_ns']+=1
    with pytest.raises(ValueError,match='hash'):RuntimeResult(bad)


def test_offline_replay_preserves_rng_state_and_never_launches(pair,monkeypatch):
    from hyso.performance import devices,stability,_telemetry
    def forbidden(*a,**kw):raise AssertionError('offline replay launched native operation')
    value=result(pair)
    model=torch.nn.Linear(2,2);state=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
    py=random.getstate();nrng=np.random.get_state()
    monkeypatch.setattr(devices,'_launch',forbidden);monkeypatch.setattr(stability,'_run_owned',forbidden)
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden);monkeypatch.setattr(_telemetry,'gpu_observe',forbidden)
    assert value.device_diagnostics()['comparable']
    assert torch.equal(torch.get_rng_state(),rng) and py==random.getstate() and np.array_equal(nrng[1],np.random.get_state()[1])
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())


def test_factory_checkpoint_and_expected_state_guard(tmp_path):
    model=torch.nn.Linear(2,2);p=tmp_path/'constructed.txt';p.write_bytes(b'not a trained model;hash guard only')
    source=dict(module_sha256='a'*64,configuration={})
    declared=dict(kind='trained_checkpoint',model_id='CONSTRUCTED_GUARD_ONLY',checkpoint_path=str(p),
        checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),preprocessing_id='identity',preprocessing_sha256='b'*64,
        expected_loaded_state_sha256=_state_hash(model))
    origin=_capture_origin(declared,model,source);_checkpoint_unchanged(origin)
    declared['expected_loaded_state_sha256']='f'*64
    with pytest.raises(ValueError,match='loaded state'):_capture_origin(declared,model,source)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint'):_checkpoint_unchanged(origin)


def test_legacy_replay_no_retroactive_timestamp_claim():
    fixtures=Path(__file__).parent/'fixtures'
    r=RuntimeResult.from_json(fixtures/'device-comparison-v1.json')
    assert r.to_dict()==RuntimeResult.from_csv(fixtures/'device-comparison-v1.csv').to_dict()
    d=r.device_diagnostics()
    assert not d['comparable'] and d['devices'][0]['timing']['value'] is None
    with pytest.raises(ValueError):r.device_diagnostics(baseline_device=GPU)


@pytest.mark.parametrize('mode',['normal','wrong_state','wrong_input_device'])
def test_optional_origin_in_real_owned_cpu_worker(tmp_path,mode):
    from hyso.performance import compare_devices,WorkloadRecipe,DeviceProtocol
    (tmp_path/'p34_origin_fixture.py').write_text('''
def build(mode):
    import torch
    from hyso.performance._device_diagnostics import _state_hash
    model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(4,2))
    inputs=torch.ones(2,1,2,2,device='meta' if mode=='wrong_input_device' else 'cpu')
    origin=dict(kind='controlled_fixture',model_id='P34_UNTRAINED_REGRESSION_ONLY',
        preprocessing_id='constructed_ones_NCHW',preprocessing_sha256='c'*64,
        expected_loaded_state_sha256='f'*64 if mode=='wrong_state' else _state_hash(model))
    return model,inputs,origin
''',encoding='utf-8')
    r=compare_devices(WorkloadRecipe('p34_origin_fixture:build',dict(mode=mode),search_path=str(tmp_path)),
        ['cpu'],tmp_path/'workers',protocol=DeviceProtocol(warmup=1,samples=3))
    data=r.to_dict();run=data['evidence']['runs'][0]
    assert run['controller']['cleanup']['active_processes']==0
    from p34_runtime_acceptance import assert_pid_absent
    assert_pid_absent(run['controller']['launcher_pid'])
    for suffix in ('json','csv'):
        path=tmp_path/('result.'+suffix);getattr(r,'to_'+suffix)(path)
        assert getattr(RuntimeResult,'from_'+suffix)(path).to_dict()==data
    if mode=='normal':
        row=r.device_diagnostics()['devices'][0]
        assert row['origin']['kind']=='controlled_fixture' and row['sample_count']==3
        assert row['timing']['value'] and not row['trained_checkpoint_attested']
    else:
        assert data['analysis']['devices'][0]['status']=='failed'
        assert data['analysis']['devices'][0]['latency']['value'] is None


def test_modified_batch_source_and_precision_scopes(pair):
    run=pair['runs'][1];i=run['report']['identity']
    i['batch']*=2;i['shape'][0]*=2;run['report']['completed']*=2
    p=result(pair).device_diagnostics()['comparisons'][0]
    assert p['status']=='not_comparable' and 'batch' in p['differences']
    assert p['throughput_ratio']['value'] is None
    i['factory_sha256']=None
    with pytest.raises(ValueError):result(pair)


def test_unsupported_precision_rejected_before_worker(tmp_path):
    from hyso.performance import DeviceProtocol
    with pytest.raises(ValueError,match='FP32/FP64'):DeviceProtocol(precision='float16')


def test_independent_loaded_state_mutation_marker_rejected(pair):
    pair['runs'][1]['report']['loaded_state_unchanged']=False
    with pytest.raises(ValueError,match='mutated'):result(pair)


def test_factory_entrypoint_mismatch_never_overrides_base_rejection(pair):
    pair['runs'][1]['recipe']['factory']='deployment_support:another_factory';rebind(pair['runs'][1])
    d=result(pair).device_diagnostics()['comparisons'][0]
    assert d['status']=='not_comparable' and 'factory_entrypoint' in d['differences']
    assert d['latency_speedup']['value'] is None
