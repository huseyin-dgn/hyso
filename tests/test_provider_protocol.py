"""P35 protocol doubles and real missing-dependency cleanup, never native ORT PASS."""
from dataclasses import asdict
import builtins
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

import numpy as np
import pytest
import torch

from hyso.performance import ProviderProtocol,WorkloadRecipe,profile_execution_providers,RuntimeResult
from hyso.performance import providers as module
from hyso.performance.runtime_result import _hash,_result

HERE=str(Path(__file__).parent.resolve())


@pytest.fixture(autouse=True)
def optional_graph_validation_dependency(request):
    pure={'test_base_import_and_missing_dependency_do_not_initialize_cuda',
          'test_public_api_accepts_defaults_without_runtime_and_cleans_owned_tree',
          'test_default_quotas_unset_and_invalid_device_limit_options'}
    if request.node.originalname not in pure:pytest.importorskip('onnx')


def request(**kwargs):
    data=dict(token=str(uuid.uuid4()),recipe=asdict(WorkloadRecipe('deployment_support:onnx_model',search_path=HERE)),
        protocol=asdict(ProviderProtocol()),providers=None,expected={},optimization='all',runtime_search_path=None,provider_options={})
    data.update(kwargs)
    return data


class FakeORT:
    """Explicit protocol double: NumPy output and synthetic host events, not ORT."""
    __version__='synthetic-test-double-not-native'
    GraphOptimizationLevel=SimpleNamespace(ORT_DISABLE_ALL=0,ORT_ENABLE_ALL=99)
    ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0)
    SessionOptions=SimpleNamespace
    def __init__(self,available=('CPUExecutionProvider','AzureExecutionProvider'),failure=None,enabled=None):
        self.available=list(available);self.failure=failure;self.enabled=enabled
        self.sessions=[];self.runs=0;self.kwargs=None;self.options=None
    def get_available_providers(self):return list(self.available)
    def get_build_info(self):return 'synthetic protocol fixture, no runtime installed'
    def InferenceSession(self,raw,*,sess_options,providers,**kwargs):
        self.sessions.append(providers);self.kwargs=kwargs;self.options=sess_options
        if self.failure=='constructor_oom':raise RuntimeError('Failed to allocate memory for requested buffer')
        owner=self
        class Session:
            disabled=False
            def disable_fallback(self):self.disabled=True
            def get_providers(self):
                return owner.enabled if owner.enabled is not None else [p[0] if isinstance(p,tuple) else p for p in providers]
            def get_provider_options(self):return {p:{} for p in self.get_providers()}
            def run(self,names,feed):
                assert self.disabled
                owner.runs+=1
                if owner.failure=='run_oom':raise RuntimeError('CUDA out of memory')
                if owner.failure=='cuda_error':raise RuntimeError('CUDA error: invalid device ordinal')
                if owner.failure=='ort_error':raise RuntimeError('ORT execution failed')
                return [np.maximum(feed['X'],0)]
            def end_profiling(self):
                path=Path(sess_options.profile_file_prefix+'.json')
                events=[dict(cat='Node',name='relu_kernel_time',pid=os.getpid(),dur=4.5,ts=100.+i*10,
                    args=dict(provider=self.get_providers()[-1],op_name='Relu')) for i in range(owner.runs)]
                path.write_text(json.dumps(events),encoding='utf-8')
                Path(sess_options.optimized_model_filepath).write_bytes(raw)
                return str(path)
        return Session()


def simulated(monkeypatch,tmp_path,*,fake=None,**kwargs):
    fake=FakeORT() if fake is None else fake
    monkeypatch.setitem(sys.modules,'onnxruntime',fake)
    req=request(**kwargs)
    report=module._worker_providers(tmp_path,req)
    report.update(status='completed',reason=None,token=req['token'],request_sha256=_hash(
        {k:req[k] for k in ('recipe','protocol','providers','expected','optimization','runtime_search_path','provider_options')}))
    evidence={k:req[k] for k in ('token','recipe','protocol','providers','expected','optimization','runtime_search_path','provider_options')}
    evidence.update(report=report,controller=dict(token=req['token'],launcher_pid=os.getpid(),exit_code=0,reason=None,
        cleanup=dict(active_processes=0),kill_on_close=True))
    return _result('P35',evidence),fake


def rebind(e):
    r=e['report'];r['trace_sha256']=_hash(r['trace'])
    r['profile_binding']=_hash(dict(session_id=e['token'],model_sha256=r['model_sha256'],trace_sha256=r['trace_sha256'],
        input_sha256=r['input_sha256'],providers=r['session_providers']))


def test_default_once_automatic_observation_without_expectations(monkeypatch,tmp_path):
    result,fake=simulated(monkeypatch,tmp_path)
    d=result.provider_diagnostics()
    assert fake.runs==1 and len(fake.sessions)==1 and fake.kwargs['enable_fallback'] is False
    assert ProviderProtocol().warmup==0 and ProviderProtocol().runs==1
    assert d['core_status']=='observed' and d['requested_providers']==['CPUExecutionProvider']
    assert d['available_providers']==['CPUExecutionProvider','AzureExecutionProvider']
    assert d['observed_execution_providers']==['CPUExecutionProvider']
    assert d['expectation_mismatch']['status']=='not_requested'
    assert d['execution_fallback']['detected'] is None
    assert d['source_node_mapping']['status']=='unknown'  # Normal optimization remains on.
    assert d['operator_timings'][0]['op_type']=='Relu'
    assert d['operator_timings'][0]['mean_host_node_us']==4.5


def test_unavailable_optional_build_metadata_does_not_erase_observed_execution(monkeypatch,tmp_path):
    monkeypatch.setattr(FakeORT,'get_build_info',None)
    result,_=simulated(monkeypatch,tmp_path)
    d=result.provider_diagnostics()
    assert d['runtime']['build'] is None and d['core_status']=='observed'


@pytest.mark.parametrize('selected,executed',[(None,'CPUExecutionProvider'),(['CUDAExecutionProvider'],'CUDAExecutionProvider')])
def test_default_cpu_and_explicit_cuda_keep_available_configured_executed_distinct(selected,executed,monkeypatch,tmp_path):
    # Protocol double only: this never establishes native CUDA execution.
    available=['CUDAExecutionProvider','CPUExecutionProvider']
    result,_=simulated(monkeypatch,tmp_path,fake=FakeORT(available=available),providers=selected)
    d=result.provider_diagnostics()
    assert d['available_providers']==available
    assert d['requested_providers']==d['enabled_session_providers']==[executed]
    assert d['observed_execution_providers']==[executed]
    assert d['configuration_fallback']['status']=='requested_providers_present'
    assert d['execution_fallback']['detected'] is None


def test_explicit_expectation_mismatch_is_not_execution_fallback(monkeypatch,tmp_path):
    result,_=simulated(monkeypatch,tmp_path,providers=['AzureExecutionProvider','CPUExecutionProvider'],
        expected={'relu':'AzureExecutionProvider'},optimization='disabled')
    assert result.to_dict()['analysis']['fallback_detected'] is True  # Preserved legacy semantics.
    d=result.provider_diagnostics()
    assert d['expectation_mismatch']['status']=='observed'
    assert d['source_node_mapping']['status']=='observed'
    assert d['execution_fallback']['status']=='unknown' and d['execution_fallback']['detected'] is None


@pytest.mark.parametrize('provider',['CUDAExecutionProvider','DmlExecutionProvider','TensorrtExecutionProvider','OpenVINOExecutionProvider'])
def test_available_provider_options_without_uuid_or_default_gpu_quota(provider,monkeypatch,tmp_path):
    fake=FakeORT(available=[provider,'CPUExecutionProvider'])
    result,fake=simulated(monkeypatch,tmp_path,fake=fake,providers=[provider,'CPUExecutionProvider'])
    assert fake.sessions==[[provider,'CPUExecutionProvider']]
    assert result.to_dict()['evidence']['recipe']['gpu_uuid'] is None
    assert result.to_dict()['evidence']['provider_options']=={}
    if provider=='DmlExecutionProvider':assert not fake.options.enable_mem_pattern and fake.options.execution_mode==0


def test_explicit_gpu_limit_and_device_are_forwarded(monkeypatch,tmp_path):
    options={'CUDAExecutionProvider':{'device_id':1,'gpu_mem_limit':128*1024**2}}
    result,fake=simulated(monkeypatch,tmp_path,fake=FakeORT(available=['CUDAExecutionProvider','CPUExecutionProvider']),
        providers=['CUDAExecutionProvider','CPUExecutionProvider'],provider_options=options)
    assert fake.sessions[0][0]==('CUDAExecutionProvider',options['CUDAExecutionProvider'])
    assert result.provider_diagnostics()['requested_provider_options']==options


@pytest.mark.parametrize('failure,kind',[('constructor_oom','oom'),('run_oom','oom'),('cuda_error','cuda_error'),('ort_error','ort_error')])
def test_actual_exception_path_never_retries_or_swallows(failure,kind,monkeypatch,tmp_path):
    fake=FakeORT(failure=failure);monkeypatch.setitem(sys.modules,'onnxruntime',fake)
    with pytest.raises(module._ProviderFailure) as captured:module._worker_providers(tmp_path,request())
    assert captured.value.details['failure_kind']==kind
    assert len(fake.sessions)==1 and fake.runs==(0 if failure=='constructor_oom' else 1)


def test_unsupported_provider_is_detected_before_session_or_execution(monkeypatch,tmp_path):
    fake=FakeORT();monkeypatch.setitem(sys.modules,'onnxruntime',fake)
    with pytest.raises(module._ProviderFailure) as captured:
        module._worker_providers(tmp_path,request(providers=['CUDAExecutionProvider']))
    assert captured.value.details['failure_kind']=='unsupported_provider'
    assert captured.value.details['available_providers']==fake.available
    assert not fake.sessions and not fake.runs


def test_session_configuration_loss_does_not_execute_cpu(monkeypatch,tmp_path):
    fake=FakeORT(available=['CUDAExecutionProvider','CPUExecutionProvider'],enabled=['CPUExecutionProvider'])
    monkeypatch.setitem(sys.modules,'onnxruntime',fake)
    with pytest.raises(module._ProviderFailure) as captured:
        module._worker_providers(tmp_path,request(providers=['CUDAExecutionProvider','CPUExecutionProvider']))
    assert captured.value.details['phase']=='session_configuration_difference' and fake.runs==0


def test_configuration_fallback_receipt_never_claims_execution_fallback(monkeypatch,tmp_path):
    from hyso.performance.devices import _worker_dispatch
    fake=FakeORT(available=['CUDAExecutionProvider','CPUExecutionProvider'],enabled=['CPUExecutionProvider'])
    monkeypatch.setitem(sys.modules,'onnxruntime',fake)
    req=request(providers=['CUDAExecutionProvider','CPUExecutionProvider']);req['mode']='providers'
    _worker_dispatch(tmp_path,req)
    report=json.loads((tmp_path/'report.json').read_text())
    e={k:req[k] for k in ('token','recipe','protocol','providers','expected','optimization','runtime_search_path','provider_options')}
    e.update(report=report,controller=dict(token=req['token'],launcher_pid=os.getpid(),exit_code=0,reason=None,
        cleanup=dict(active_processes=0),kill_on_close=True))
    d=_result('P35',e).provider_diagnostics()
    assert d['configuration_fallback']['status']=='configuration_difference'
    assert d['configuration_fallback']['missing_requested']==['CUDAExecutionProvider']
    assert d['execution_fallback']['detected'] is None and not d['observed_execution_providers']
    assert d['completed_runs']==0 and fake.runs==0


@pytest.mark.parametrize('damage',['missing_trace','missing_provider','missing_duration','renamed','fused_op','foreign_pid','nan_duration','options'])
def test_missing_ambiguous_and_corrupted_evidence(damage,monkeypatch,tmp_path):
    result,_=simulated(monkeypatch,tmp_path,optimization='disabled')
    e=result.to_dict()['evidence'];node=e['report']['trace'][0]
    if damage=='missing_trace':e['report']['trace']=[]
    elif damage=='missing_provider':node['args'].pop('provider')
    elif damage=='missing_duration':node.pop('dur')
    elif damage=='renamed':node['name']='fused_kernel_time'
    elif damage=='fused_op':node['args']['op_name']='FusedRelu'
    elif damage=='foreign_pid':node['pid']+=1
    elif damage=='nan_duration':node['dur']=float('nan')
    else:e['report']['requested_provider_options']={'CPUExecutionProvider':{'extra':1}}
    if damage=='nan_duration':
        with pytest.raises(ValueError):_hash(e['report']['trace'])
        return
    rebind(e)
    if damage in ('foreign_pid','options'):
        with pytest.raises(ValueError):module._analyze_providers(e)
        return
    output=_result('P35',e).provider_diagnostics()
    assert output['execution_fallback']['detected'] is None
    if damage in ('renamed','fused_op'):assert output['source_node_mapping']['status']=='unknown'
    if damage in ('missing_trace','missing_provider'):assert not output['observed_execution_providers']
    if damage=='missing_duration':assert output['operator_timings'][0]['total_host_node_us'] is None


def test_legacy_and_modern_json_csv_readers_preserve_payload(monkeypatch,tmp_path):
    result,_=simulated(monkeypatch,tmp_path,providers=['CPUExecutionProvider'],optimization='disabled',
        expected={'*':'CPUExecutionProvider'},protocol=asdict(ProviderProtocol(warmup=1,runs=2)))
    for label,e in [('modern',result.to_dict()['evidence']),('legacy',copy.deepcopy(result.to_dict()['evidence']))]:
        if label=='legacy':
            for key in ('observation_schema','requested_provider_options','device_selection'):e['report'].pop(key)
            e.pop('provider_options')
            for key in ('process_memory_bytes','job_memory_bytes','max_processes'):e['protocol'].pop(key)
            e['report']['request_sha256']=_hash({k:e[k] for k in
                ('recipe','protocol','providers','expected','optimization','runtime_search_path')})
        reopened=_result('P35',e);payload=reopened.to_dict()
        j=tmp_path/(label+'.json');c=tmp_path/(label+'.csv')
        reopened.to_json(j);reopened.to_csv(c)
        assert RuntimeResult.from_json(j).to_dict()==payload
        assert RuntimeResult.from_csv(c).to_dict()==payload
        assert reopened.to_dict()['evidence']['report']['completed_runs']==3
        assert reopened.provider_diagnostics()['execution_fallback']['detected'] is None


def test_base_import_and_missing_dependency_do_not_initialize_cuda(monkeypatch,tmp_path):
    original=builtins.__import__
    def missing(name,*args,**kwargs):
        if name=='onnxruntime':raise ModuleNotFoundError("controlled missing onnxruntime",name='onnxruntime')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',missing)
    def forbidden(*a,**kw):raise AssertionError('CPU protocol must not initialize CUDA')
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    with pytest.raises(module._ProviderFailure) as captured:module._worker_providers(tmp_path,request())
    assert captured.value.details['failure_kind']=='missing_dependency'
    assert captured.value.details['completed_runs']==0


def test_public_api_accepts_defaults_without_runtime_and_cleans_owned_tree(tmp_path):
    # Native process/cleanup evidence only. This does not assert native ORT success.
    import importlib.util
    if importlib.util.find_spec('onnxruntime') is not None:pytest.skip('missing-runtime contract requires absent optional package')
    result=profile_execution_providers(WorkloadRecipe('deployment_support:onnx_model',search_path=HERE),tmp_path/'run',
        protocol=ProviderProtocol(process_memory_bytes=4*1024**3,job_memory_bytes=6*1024**3,max_processes=4))
    assert result.to_dict()['analysis']['failure_kind']=='missing_dependency'
    assert result.provider_diagnostics()['core_status']=='unavailable'
    assert result.to_dict()['evidence']['controller']['cleanup']['active_processes']==0


def test_default_quotas_unset_and_invalid_device_limit_options():
    p=ProviderProtocol()
    assert p.process_memory_bytes is p.job_memory_bytes is p.max_processes is None
    for options in ({'CUDAExecutionProvider':{'gpu_mem_limit':0}}, {'CUDAExecutionProvider':{'device_id':-1}}):
        with pytest.raises(ValueError):module._options(options,['CUDAExecutionProvider'])
    with pytest.raises(ValueError):module._options({'CUDAExecutionProvider':{'device_id':1}},['CUDAExecutionProvider'],'GPU-explicit')
