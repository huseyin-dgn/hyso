"""Actual CPU execution plus hostile evidence/protocol validation; CUDA is opt-in."""
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import pytest
import torch

from hyso.performance import (WorkloadRecipe,RuntimeResult,DeviceProtocol,compare_devices,
    ProviderProtocol,profile_execution_providers)
from hyso.performance.devices import _analyze_devices,_model_hash
from hyso.performance.providers import _analyze_providers
from hyso.performance.runtime_result import _hash

HERE=str(Path(__file__).resolve().parent)
ORT=os.environ.get('HYSO_OPTIONAL_ORT_PATH')


def recipe(factory='torch_model',mode='normal'):
    return WorkloadRecipe('deployment_support:'+factory,dict(mode=mode),search_path=HERE)


def save(result,name):
    directory=os.environ.get('HYSO_DEPLOYMENT_EVIDENCE')
    if directory:
        result.to_json(Path(directory)/(name+'.json'));result.to_csv(Path(directory)/(name+'.csv'))
    return result


@pytest.fixture(scope='module')
def cpu(tmp_path_factory):
    return save(compare_devices(recipe(),['cpu'],tmp_path_factory.mktemp('compare')/'run',
        protocol=DeviceProtocol(warmup=1,samples=5)),'p34-cpu')


def test_cpu_real_and_independent_arithmetic(cpu):
    d=cpu.to_dict();report=d['evidence']['runs'][0]['report'];row=d['analysis']['devices'][0]
    assert row['status']=='measured',d
    assert d['analysis']['status']=='limited' and not d['analysis']['comparable']
    assert row['latency']['value']['mean_ms']==pytest.approx(np.mean(report['latency_ns'])/1e6)
    assert row['throughput']['value']==pytest.approx(10e9/report['window_ns'])
    assert row['energy']['value'] is None and row['quality']['value'] is None
    assert row['memory']['value']['rss_after_bytes']>0
    assert d['evidence']['runs'][0]['controller']['cleanup']['active_processes']==0


@pytest.mark.parametrize('values',[dict(samples=2),dict(precision='float16'),dict(warmup=-1),
    dict(samples=True),dict(step_timeout_seconds=float('inf')),dict(max_runtime_seconds=1000)])
def test_device_protocol_rejects(values):
    with pytest.raises(ValueError):DeviceProtocol(**values)


@pytest.mark.parametrize('devices',[[],['cpu','cpu'],['cuda:99'],['mps']])
def test_explicit_device_admission(devices,tmp_path):
    with pytest.raises(ValueError):compare_devices(recipe(),devices,tmp_path/'run')
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('mode',['hang','oom','mutate'])
def test_failure_timeout_mutation_and_cleanup(mode,tmp_path):
    start=time.monotonic()
    if mode=='mutate':
        with pytest.raises(ValueError,match='mutated'):
            compare_devices(recipe(mode=mode),['cpu'],tmp_path/'run',protocol=DeviceProtocol(warmup=0,samples=3))
    else:
        result=compare_devices(recipe(mode=mode),['cpu'],tmp_path/'run',protocol=DeviceProtocol(
            warmup=0,samples=3,step_timeout_seconds=6))
        d=result.to_dict();assert d['analysis']['devices'][0]['status']=='failed'
        assert d['analysis']['devices'][0]['throughput']['value'] is None
        assert d['evidence']['runs'][0]['controller']['cleanup']['active_processes']==0
        if mode=='hang':assert d['evidence']['runs'][0]['controller']['reason']=='heartbeat_timeout'
    assert time.monotonic()-start<50


def test_missing_real_device(tmp_path):
    result=compare_devices(recipe(),['GPU-00000000-0000-0000-0000-000000000000'],tmp_path/'run',
                           protocol=DeviceProtocol(warmup=0,samples=3))
    assert result.to_dict()['analysis']['devices'][0]['status']=='failed'


def test_no_clobber_and_parent_state(cpu,tmp_path):
    import deployment_support
    model,inputs=deployment_support.torch_model();optimizer=torch.optim.Adam(model.parameters())
    before=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone();py=random.getstate();npstate=np.random.get_state()
    opt=copy.deepcopy(optimizer.state_dict());original=inputs.clone()
    compare_devices(recipe(),['cpu'],tmp_path/'actual',protocol=DeviceProtocol(warmup=0,samples=3))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert torch.equal(rng,torch.get_rng_state()) and py==random.getstate()
    assert np.array_equal(npstate[1],np.random.get_state()[1])
    assert optimizer.state_dict()==opt and torch.equal(original,inputs)
    with pytest.raises(FileExistsError):compare_devices(recipe(),['cpu'],tmp_path/'actual')


def test_device_corruption_identity_and_samples(cpu):
    data=cpu.to_dict()
    for key,value in [('completed',1),('model_unchanged',False),('window_ns',1)]:
        bad=copy.deepcopy(data['evidence']);bad['runs'][0]['report'][key]=value
        with pytest.raises(ValueError):_analyze_devices(bad)
    bad=copy.deepcopy(data);bad['analysis']['devices'][0]['throughput']['value']=1
    with pytest.raises(ValueError):RuntimeResult(bad)


def test_comparison_rejects_model_input_backend_or_precision_identity(cpu):
    # Reuse a real run as a validation fixture, never as a second native result.
    evidence=cpu.to_dict()['evidence'];evidence['devices']=['cpu','GPU-00000000-0000-0000-0000-000000000000']
    second=copy.deepcopy(evidence['runs'][0]);second['device']=evidence['devices'][1]
    second['token']=str(uuid.uuid4());second['controller']['token']=second['token'];second['report']['token']=second['token']
    second['recipe']['gpu_uuid']=second['device']
    second['report']['identity']['device_id']=second['device']
    second['report']['request_sha256']=_hash({k:second[k] for k in ('recipe','protocol','device')})
    evidence['runs'].append(second)
    for key in ('model_sha256','input_sha256'):
        bad=copy.deepcopy(evidence);bad['runs'][1]['report']['identity'][key]='f'*64
        assert not _analyze_devices(bad)['comparable']
    for key in ('backend','precision'):
        bad=copy.deepcopy(evidence);bad['runs'][1]['report']['identity'][key]='unsupported'
        with pytest.raises(ValueError):_analyze_devices(bad)


@pytest.fixture(scope='module')
def provider_cpu(tmp_path_factory):
    if importlib.util.find_spec('onnx') is None:pytest.skip('optional onnx not installed')
    if not ORT and importlib.util.find_spec('onnxruntime') is None:pytest.skip('optional ORT not installed')
    return save(profile_execution_providers(recipe('onnx_model'),tmp_path_factory.mktemp('provider')/'run',
        providers=['AzureExecutionProvider','CPUExecutionProvider'],expected={'*':'CPUExecutionProvider'},
        runtime_search_path=ORT),'p35-cpu')


def test_actual_provider_events_not_provider_list(provider_cpu):
    d=provider_cpu.to_dict();a=d['analysis'];report=d['evidence']['report']
    assert a['status']=='expected_execution_observed',d
    assert a['fallback_detected'] is False
    assert 'AzureExecutionProvider' in report['session_providers']
    assert 'AzureExecutionProvider' in a['unused_session_providers']
    assert a['events'] and all(x['provider']=='CPUExecutionProvider' for x in a['events'])
    assert all(x['source_node']=='relu' and x['mapping']=='mapped' for x in a['events'])


def test_real_unexpected_cpu_execution(tmp_path,provider_cpu):
    r=profile_execution_providers(recipe('onnx_model'),tmp_path/'run',
        providers=['AzureExecutionProvider','CPUExecutionProvider'],expected={'relu':'AzureExecutionProvider'},
        runtime_search_path=ORT)
    save(r,'p35-unexpected-cpu')
    a=r.to_dict()['analysis'];assert a['fallback_detected'] is True and a['unexpected_executions']
    assert all(x['provider']=='CPUExecutionProvider' for x in a['unexpected_executions'])


def rebind(e):
    r=e['report'];r['trace_sha256']=_hash(r['trace'])
    r['profile_binding']=_hash(dict(session_id=e['token'],model_sha256=r['model_sha256'],
        trace_sha256=r['trace_sha256'],input_sha256=r['input_sha256'],providers=r['session_providers']))


@pytest.mark.parametrize('kind',['missing','renamed','op_mismatch'])
def test_missing_or_unmapped_execution_unknown(provider_cpu,kind):
    e=provider_cpu.to_dict()['evidence']
    if kind=='missing':e['report']['trace']=[]
    else:
        for x in e['report']['trace']:
            if x.get('cat')=='Node':
                if kind=='renamed':x['name']='fused_kernel_time'
                else:x['args']['op_name']='NotRelu'
    rebind(e);a=_analyze_providers(e)
    assert a['fallback_detected'] is None and a['status']=='unknown'


@pytest.mark.parametrize('key',['token','model_sha256','graph_sha256','trace_sha256','session_id','profile_binding'])
def test_provider_identity_and_trace_corruption(provider_cpu,key):
    e=provider_cpu.to_dict()['evidence']
    if key=='token':e['token']='wrong'
    else:e['report'][key]='f'*64
    with pytest.raises(ValueError):_analyze_providers(e)


def test_wrong_process_trace_rejected_even_after_rehash(provider_cpu):
    e=provider_cpu.to_dict()['evidence']
    for event in e['report']['trace']:
        if event.get('cat')=='Node':event['pid']=-1
    rebind(e)
    with pytest.raises(ValueError,match='another worker'):_analyze_providers(e)


def test_optimization_conservatively_unknown(tmp_path,provider_cpu):
    r=profile_execution_providers(recipe('onnx_model'),tmp_path/'run',providers=['CPUExecutionProvider'],
        expected={'*':'CPUExecutionProvider'},optimization='all',runtime_search_path=ORT)
    assert r.to_dict()['analysis']['fallback_detected'] is None


def test_provider_timeout_cleanup(tmp_path,provider_cpu):
    r=profile_execution_providers(recipe('onnx_model','hang'),tmp_path/'run',providers=['CPUExecutionProvider'],
        expected={'*':'CPUExecutionProvider'},runtime_search_path=ORT,
        protocol=ProviderProtocol(step_timeout_seconds=6))
    d=r.to_dict();assert d['analysis']['status']=='unavailable'
    assert d['evidence']['controller']['reason']=='heartbeat_timeout'
    assert d['evidence']['controller']['cleanup']['active_processes']==0


def test_unsupported_provider_and_no_clobber(tmp_path,provider_cpu):
    with pytest.raises(ValueError,match='unsupported execution provider'):
        profile_execution_providers(recipe('onnx_model'),tmp_path/'unsupported',
            providers=['UnknownExecutionProvider'],expected={'*':'CPUExecutionProvider'})
    with pytest.raises(ValueError):profile_execution_providers(recipe('onnx_model'),tmp_path/'bad',
        providers=['CPUExecutionProvider'],expected={'*':'CPUExecutionProvider'},optimization='unsupported')
    directory=tmp_path/'run';directory.mkdir()
    with pytest.raises(FileExistsError):profile_execution_providers(recipe('onnx_model'),directory,
        providers=['CPUExecutionProvider'],expected={'*':'CPUExecutionProvider'},runtime_search_path=ORT)


@pytest.mark.parametrize('fixture',['cpu','provider_cpu'])
def test_lossless_json_csv_and_tamper(request,fixture,tmp_path):
    result=request.getfixturevalue(fixture)
    j=tmp_path/'result.json';c=tmp_path/'result.csv'
    result.to_json(j);result.to_csv(c)
    assert RuntimeResult.from_json(j).to_dict()==result.to_dict()
    assert RuntimeResult.from_csv(c).to_dict()==result.to_dict()
    with pytest.raises(FileExistsError):result.to_json(j)
    with pytest.raises(FileExistsError):result.to_csv(c)
    rows=list(csv.DictReader(c.open(encoding='utf-8',newline='')));rows[0]['source_sha256']='f'*64
    with c.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError):RuntimeResult.from_csv(c)


def test_provider_parent_state_and_execution_ledger(provider_cpu,tmp_path):
    import deployment_support
    model,inputs=deployment_support.torch_model();before=copy.deepcopy(model.state_dict())
    optimizer=torch.optim.Adam(model.parameters());opt=copy.deepcopy(optimizer.state_dict())
    rng=torch.get_rng_state().clone();py=random.getstate();npstate=np.random.get_state()
    profile_execution_providers(recipe('onnx_model'),tmp_path/'parent',providers=['CPUExecutionProvider'],
        expected={'*':'CPUExecutionProvider'},runtime_search_path=ORT)
    assert torch.equal(rng,torch.get_rng_state()) and py==random.getstate()
    assert np.array_equal(npstate[1],np.random.get_state()[1]) and optimizer.state_dict()==opt
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    for key,value in [('completed_runs',0),('runtime',{'name':'unsupported'})]:
        e=provider_cpu.to_dict()['evidence'];e['report'][key]=value
        with pytest.raises(ValueError):_analyze_providers(e)


@pytest.mark.skipif(not os.environ.get('HYSO_ORT_CUDA_AVAILABLE'),reason='real ORT CUDA provider unavailable')
def test_actual_ort_cuda(tmp_path):
    selected=WorkloadRecipe('deployment_support:onnx_model',search_path=HERE,
        gpu_uuid=os.environ['HYSO_NATIVE_GPU_UUID'])
    result=profile_execution_providers(selected,tmp_path/'run',
        providers=['CUDAExecutionProvider','CPUExecutionProvider'],expected={'*':'CUDAExecutionProvider'},runtime_search_path=ORT)
    a=result.to_dict()['analysis']
    assert a['events'] and any(x['provider']=='CUDAExecutionProvider' for x in a['events'])


@pytest.mark.skipif(not os.environ.get('HYSO_NATIVE_GPU_UUID'),reason='explicit owned native GPU acceptance only')
def test_actual_cpu_cuda_comparison(tmp_path):
    gpu=os.environ['HYSO_NATIVE_GPU_UUID']
    r=save(compare_devices(recipe(),['cpu',gpu],tmp_path/'run',protocol=DeviceProtocol(samples=10)),
           'p34-cpu-cuda')
    a=r.to_dict()['analysis'];assert a['comparable'],r.to_dict()
    assert all(x['status']=='measured' for x in a['devices'])
    assert a['devices'][1]['memory']['value']['cuda_peak_allocated_bytes']>0


@pytest.mark.skipif(not os.environ.get('HYSO_NATIVE_GPU_UUID'),reason='explicit native GPU OOM acceptance only')
def test_native_budget_oom_then_recovery(tmp_path):
    gpu=os.environ['HYSO_NATIVE_GPU_UUID']
    r=compare_devices(recipe(mode='cuda_oom'),[gpu],tmp_path/'oom',protocol=DeviceProtocol(samples=3,warmup=0))
    a=r.to_dict()['analysis']['devices'][0];assert a['status']=='failed' and 'OutOfMemory' in a['reason']
    r=compare_devices(recipe(),[gpu],tmp_path/'recovery',protocol=DeviceProtocol(samples=3,warmup=0))
    assert r.to_dict()['analysis']['devices'][0]['status']=='measured'


def test_actual_fp64_cpu_and_optional_cuda(tmp_path):
    devices=['cpu']
    if os.environ.get('HYSO_NATIVE_GPU_UUID'):devices.append(os.environ['HYSO_NATIVE_GPU_UUID'])
    r=compare_devices(recipe(),devices,tmp_path/'run',protocol=DeviceProtocol(samples=3,warmup=1,precision='float64'))
    assert all(x['status']=='measured' and x['identity']['precision']=='float64'
               for x in r.to_dict()['analysis']['devices'])
    assert r.to_dict()['analysis']['comparable']==(len(devices)==2)
