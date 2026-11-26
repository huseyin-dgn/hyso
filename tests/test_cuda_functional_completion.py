"""CPU/offline guards for explicit CUDA APIs; mocks are never GPU acceptance."""
import copy
from types import SimpleNamespace
import pytest
import torch
from hyso import quality
from hyso.performance.dataloader import _LoaderMixin,_validate_transfers,DataLoaderAnalysisProtocol
from hyso.performance import FrameInferenceProcessor,StreamProtocol


def test_quality_devices_never_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    for operation in (
        lambda:quality.evaluate_robustness(None,None,None,corruption_id='x',severity=.1,seed=1,device='cuda:0'),
        lambda:quality.measure_time_to_quality(None,None,seeds=[1,2],target=1.,device='cuda:0'),
        lambda:quality._quality_device('cuda:0')):
        with pytest.raises(NotImplementedError,match='no CPU fallback'):operation()
    with pytest.raises(NotImplementedError):quality._quality_device('meta')
    assert quality._quality_device('cpu')=='cpu'


def test_peak_reset_consent_precedes_cuda_initialization(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('no device query before consent')
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    with pytest.raises(ValueError,match='allow_cuda_peak_reset'):
        quality.compare_performance_quality(None,None,None,device='cuda:0',memory_metric='cuda_allocated_peak_bytes')


def test_p27_explicit_cpu_matches_default_and_preserves_state():
    from test_quality_experiments import classifier,cohort,flip
    from trainer_replay_support import equal
    from hyso.checkpoint import _rng_state
    m=classifier();m.train();before=copy.deepcopy(m.state_dict());rng=_rng_state()
    for p in m.parameters():p.grad=torch.ones_like(p)
    grads=[p.grad.clone() for p in m.parameters()]
    a=quality.evaluate_robustness(m,cohort(),flip,corruption_id='flip',severity=.1,seed=1)
    b=quality.evaluate_robustness(m,cohort(),flip,corruption_id='flip',severity=.1,seed=1,device='cpu')
    assert a.to_dict()['analysis']==b.to_dict()['analysis']
    equal(before,m.state_dict());equal(rng,_rng_state());assert m.training
    for p,g in zip(m.parameters(),grads):assert torch.equal(p.grad,g)


def test_p24_cpu_censored_target_has_no_invented_time():
    from tools.cuda_acceptance import workloads as w
    from tools.cuda_acceptance.cases import cohort
    def factory(seed):return w.training(seed,device='cpu',frozen=True)
    d=quality.measure_time_to_quality(factory,cohort(),seeds=[1,2],target=1.,epochs=1,device='cpu').to_dict()
    assert d['analysis']['censored']==2 and d['analysis']['reached_mean_seconds'] is None


def test_p29_legacy_cpu_and_gpu_provenance_rejection():
    from test_p29_batched import fixture
    m=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(2,2))
    d=quality.compare_performance_quality(m,m,fixture(3),iterations=2,batch_size=2,device='cpu').to_dict()
    assert d['sources']['comparison_version']==3
    bad=copy.deepcopy(d);bad['sources']['comparison_version']=4
    bad['sources']['policy'].update(device='cuda:0',allow_cuda_peak_reset=False)
    bad['source_sha256']=quality._hash(bad['sources'])
    with pytest.raises(ValueError):quality.QualityExperimentResult(bad)


class _Event:
    def __init__(self,**kw):pass
    def record(self,stream):pass
    def synchronize(self):pass
    def query(self):return True
    def elapsed_time(self,other):return .25


def test_p21_mock_transfer_records_only_actual_copy(monkeypatch):
    monkeypatch.setattr(torch.cuda,'Event',_Event)
    owner=_LoaderMixin();owner.init_loader(DataLoaderAnalysisProtocol())
    owner.trainer=SimpleNamespace(device=torch.device('cuda:0'))
    owner.stream=SimpleNamespace(cuda_stream=17);owner._check_stream=lambda:None
    owner.active={'microbatch':object()};owner.waiting={'training':dict(epoch_id=1,phase='training',batch_index=1)}
    tensor=SimpleNamespace(device=torch.device('cpu'),to=lambda d:'copied',numel=lambda:4,element_size=lambda:4,is_pinned=lambda:True)
    assert owner.transfer_tensor(tensor,'inputs')=='copied'
    row=owner.transfer_rows[0]
    assert row['bytes']==16 and row['completed'] and row['cuda_event_ms']==.25 and row['pinned']
    tensor.device=torch.device('cuda:0')
    owner.transfer_tensor(tensor,'inputs');assert len(owner.transfer_rows)==1


def test_p21_mock_transfer_rejects_incomplete_event(monkeypatch):
    class Incomplete(_Event):
        def query(self):return False
    monkeypatch.setattr(torch.cuda,'Event',Incomplete)
    owner=_LoaderMixin();owner.init_loader(DataLoaderAnalysisProtocol())
    owner.trainer=SimpleNamespace(device=torch.device('cuda:0'));owner.stream=None;owner._check_stream=lambda:None
    tensor=SimpleNamespace(device=torch.device('cpu'),to=lambda d:None)
    with pytest.raises(RuntimeError,match='completion'):owner.transfer_tensor(tensor,'inputs')
    assert not owner.transfer_rows


def test_p21_transfer_import_checks_identity_clocks_counts():
    row=dict(epoch_id=1,phase='training',batch_index=1,role='inputs',start_ns=2,end_ns=4,host_ms=.000002,
        cuda_event_ms=.5,bytes=4,source_device='cpu',device='cuda:0',pinned=True,stream_id=1,completed=True)
    d=dict(value=.5,unit='ms',status='measured',reason=None,host_scope='blocking_Tensor.to_call_only',
        device_scope='inclusive_selected_stream_event_interval',excludes='loader_wait_input_validation_and_model_compute; not_DMA_engine_exclusive_time',observations=[row])
    source=dict(epochs=[dict(epoch_id=1,end_ns=5)],provenance=dict(runtime=dict(device='cuda:0')))
    batches=[dict(kind='batch',epoch_id=1,phase='training',batch_index=1,delivery_ns=1)]
    _validate_transfers(d,source,batches,2)
    for field,value in [('completed',False),('cuda_event_ms',float('nan')),('host_ms',3.),('device','cpu'),('role','invented')]:
        bad=copy.deepcopy(d);bad['observations'][0][field]=value
        with pytest.raises(ValueError):_validate_transfers(bad,source,batches,2)
    bad=copy.deepcopy(d);bad['observations']*=2
    with pytest.raises(ValueError):_validate_transfers(bad,source,batches,2)


def test_p30_gpu_identity_guard_runs_before_cuda(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    def forbidden(*a,**k):raise AssertionError('unadmitted CUDA query')
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    with pytest.raises(ValueError,match='UUID-visible'):FrameInferenceProcessor(torch.nn.Identity(),device='cuda:0',gpu_uuid='GPU-fake',cuda_memory_fraction=.1)
    with pytest.raises(ValueError):FrameInferenceProcessor(torch.nn.Identity(),device='cpu',gpu_uuid='GPU-fake')


def test_p30_cpu_processor_retains_two_completed_stages():
    import numpy as np
    m=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(3,2));before=copy.deepcopy(m.state_dict())
    p=FrameInferenceProcessor(m,size=(8,8),device='cpu')
    try:
        stages=p.process(np.zeros((8,8,3),dtype=np.uint8))
        assert [s['name'] for s in stages]==['preprocess','inference'] and p.identity['device_id']=='cpu'
        assert 'processing_version' not in p.identity
    finally:p.close()
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())


def test_p30_gpu_mock_requires_completion_before_result(monkeypatch):
    import numpy as np
    p=FrameInferenceProcessor.__new__(FrameInferenceProcessor)
    p.torch=torch;p.device=torch.device('cuda:0');p.size=(8,8);p.model=lambda x:x
    p.output_check=lambda *a,**kw:None;p.guard=SimpleNamespace(check=lambda **k:None)
    monkeypatch.setattr(torch.cuda,'current_stream',lambda d:SimpleNamespace(cuda_stream=1));monkeypatch.setattr(torch.cuda,'Event',_Event)
    original=torch.Tensor.to
    def to(t,device,*a,**kw):return t if isinstance(device,torch.device) and device.type=='cuda' else original(t,device,*a,**kw)
    monkeypatch.setattr(torch.Tensor,'to',to)
    stages=p.process(np.zeros((8,8,3),dtype=np.uint8))
    assert [s['name'] for s in stages]==['preprocess','transfer','inference']
    assert stages[-1]['completed'] is True and stages[-1]['cuda_event_ms']==.25
    class Incomplete(_Event):
        def query(self):return False
    monkeypatch.setattr(torch.cuda,'Event',Incomplete)
    with pytest.raises(RuntimeError,match='completion'):p.process(np.zeros((8,8,3),dtype=np.uint8))


def test_stream_resource_caps_are_explicit():
    assert StreamProtocol().process_memory_bytes is None
    with pytest.raises(ValueError):StreamProtocol(process_memory_bytes=20,job_memory_bytes=10)
    with pytest.raises(ValueError):StreamProtocol(max_processes=0)


def test_gpu_evidence_never_accepts_missing_owned_receipts(tmp_path):
    from tools.cuda_acceptance.validation import validate_gpu_attempt
    values,critical=validate_gpu_attempt(tmp_path,{},dict(schema_version=3,source_integrity=True,cleanup_verified=False))
    assert values=={} and 'OWNED_CLEANUP' in critical
    values,critical=validate_gpu_attempt(tmp_path,{},dict(schema_version=2))
    assert values=={} and 'SCHEMA' in critical


def test_cuda_prepare_and_cpu_runner_are_import_isolated():
    import ast
    from pathlib import Path
    repo=Path(__file__).resolve().parents[1]
    tree=ast.parse((repo/'tools/cuda_acceptance/runner.py').read_text(encoding='utf-8'))
    imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert not any('torch' in ast.unparse(n) or 'cases' in ast.unparse(n) for n in imports)
    cpu=(repo/'tools/final_acceptance/campaign_runner.py').read_text(encoding='utf-8')
    assert 'CPU_ONLY_RUNTIME_REQUIRED' in cpu and 'CPU campaign attempted real CUDA initialization' in cpu


def test_cuda_cpu_selection_keeps_runtime_allowlist_and_b4(monkeypatch):
    from tools.cuda_acceptance import runner
    selected='tests/test_runtime_analysis.py::test_cpu_model_state_modes_rng_preserved'
    excluded='tests/test_runtime_analysis.py::test_native_allocator_cache_growth_is_not_leak'
    b4='tests/test_isolated_training_memory.py::test_cuda_activation_crash_gate_precedes_allocator_and_hook_setup'
    inventory={n:dict(path=n.split('::')[0]) for n in (selected,excluded,b4)}
    monkeypatch.setattr(runner,'read_json',lambda *a,**k:dict(inventory=inventory))
    monkeypatch.setattr(runner,'file_hash',lambda *a:'offline-test-hash')
    result=runner.cpu_selection()
    assert selected in result and b4 in result and excluded not in result
    assert 'tests/test_runtime_analysis.py' not in runner.CPU_FILES


def test_cuda_resume_never_retries_terminal_cases_and_preserves_pass():
    from tools.cuda_acceptance.runner import pending_cases,retain_results
    rows={n:dict(status=s) for n,s in [('ok','PASS'),('bad','FAIL'),('held','BLOCKED')]}
    assert pending_cases(['ok','bad','held','new'],rows,set())==['new']
    assert pending_cases(['ok','bad','held','new'],rows,{'bad'})==['bad']
    cp=dict(results=copy.deepcopy(rows),history=[])
    retain_results(cp,'results',{'bad':dict(status='PASS')})
    assert cp['results']['ok']==rows['ok'] and cp['history'][0]['previous']==rows['bad']
    with pytest.raises(RuntimeError,match='PASS_OVERWRITE'):
        retain_results(cp,'results',{'ok':dict(status='FAIL')})


def test_cuda_parent_binding_direct_interpreter_and_venv_redirector():
    from tools.cuda_acceptance.identity import parent_binding
    for launcher,interpreter,parent in [(7,7,3),(7,8,7)]:
        ids=dict(launcher=dict(pid=launcher),interpreter=dict(pid=interpreter),parent_pid=parent)
        ready=dict(pid=interpreter,parent_pid=parent);binding=dict(identity=ids['launcher'])
        assert parent_binding(ids,ready,binding)
        assert not parent_binding(ids,dict(ready,parent_pid=100),binding)
    ids=dict(launcher=dict(pid=7),interpreter=dict(pid=8),parent_pid=9)
    assert not parent_binding(ids,dict(pid=8,parent_pid=9),dict(identity=ids['launcher']))


def test_cuda_uuid_canonicalization_is_physical_not_device_label():
    from uuid import UUID
    from tools.cuda_acceptance.identity import gpu_uuid
    value='9eec2f80-4c4e-a5cd-7091-03f0d013cf11'
    for display in (value,value.upper(),'GPU-'+value,UUID(value),UUID(value).bytes):
        assert gpu_uuid(display)=='GPU-'+value
    for invalid in ('cuda:0','0','RTX 3060',None):
        with pytest.raises((ValueError,TypeError,AttributeError)):gpu_uuid(invalid)


def test_cuda_all_feature_pass_gates_reject_device_label_only():
    from tools.cuda_acceptance.runner import FEATURES
    from tools.cuda_acceptance.feature_evidence import verify_feature
    for feature in (f for group in FEATURES.values() for f in group):
        with pytest.raises((KeyError,ValueError,TypeError)):
            verify_feature(feature,dict(device='cuda:0'),{})


def test_cuda_nested_resource_and_cleanup_failures_stop_scope():
    from tools.cuda_acceptance.feature_evidence import verify_nested_safety,SafetyViolation
    for payload in (dict(controller=dict(cleanup=dict(active_processes=1))),
                    dict(source=dict(terminal=dict(failure_kind='oom'))),
                    dict(worker=dict(monitor_outcome='resource_limit'))):
        with pytest.raises(SafetyViolation):verify_nested_safety(payload)
    verify_nested_safety(dict(controller=dict(cleanup=dict(active_processes=0),monitor_reason=None)))
