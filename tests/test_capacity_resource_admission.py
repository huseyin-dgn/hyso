"""P17 payload gates plus one tiny CPU worker; no large allocation or CUDA."""
import copy
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyso.performance import BatchSearchProtocol, ColdStartRecipe, capacity, cold
from hyso.performance import _batch_worker as worker


def test_known_live_copies_reject_before_any_launcher(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError('rejected candidate must not dispatch or start a process')
    monkeypatch.setattr(capacity.subprocess, 'Popen', forbidden)
    token = cold._owned_worker_launcher.set(forbidden)
    try:
        # One input fits both caps, but three unavoidable live host payloads do not.
        recipe = ColdStartRecipe('torch.nn:Identity', input_shape=(1, 1, 8192, 8192))
        plan = BatchSearchProtocol(candidates=(1, 2), maximum_batch=2,
            cpu_memory_limit_bytes=512*1024**2, input_limit_bytes=1024**3, timeout_seconds=30)
        data = capacity.find_max_batch_size(recipe, protocol=plan).to_dict()
    finally:
        cold._owned_worker_launcher.reset(token)
    rows = data['raw_evidence']['candidates']
    assert [r['status'] for r in rows] == ['resource_limit', 'not_run']
    assert all(r['classification'] == 'unknown' for r in rows)
    assert rows[0]['worker_pid'] is rows[0]['launcher_pid'] is None
    evidence = rows[0]['memory']['admission']
    assert evidence['input_bytes'] == 256*1024**2
    assert evidence['host_private_payload_lower_bound_bytes'] == 768*1024**2
    assert not evidence['fit_guaranteed'] and evidence['total_peak_bytes'] is None
    assert data['measurements']['maximum_observed_feasible_batch'] is None
    assert data['measurements']['maximum_proven_within_range'] is None
    result = capacity.CapacityResult(data)
    result.to_json(tmp_path/'result.json'); result.to_csv(tmp_path/'result.csv')
    assert capacity.CapacityResult(json.loads((tmp_path/'result.json').read_text())).to_dict() == data


def test_input_budget_is_checked_independently():
    data = worker._admission((1, 1, 16, 16), 'fp32', 'cpu', 1000, 1024**3)
    assert data['input_bytes'] == 1024 and 'input byte budget' in data['reason']


def test_resource_rejection_preserves_previous_success(monkeypatch):
    point=json.loads((Path(__file__).parent/'fixtures/advisor-native-v1.json').read_text(encoding='utf-8'))['payloads']['P08']['raw_evidence']['points'][0]['result']
    recipe=ColdStartRecipe('hyso.models.modular:ModularCNN',input_shape=(1,1,28,28))
    plan=BatchSearchProtocol(candidates=(1,1048576,3),maximum_batch=1048576,
        cpu_memory_limit_bytes=1024**3,input_limit_bytes=4*1024**3,timeout_seconds=30,
        warmup_iterations=2,measure_iterations=10,repeats=2)
    original=capacity._candidate
    def candidate(config, settings, batch, root):
        if batch!=1:return original(config,settings,batch,root)
        row=capacity._row(batch,config)
        row.update(status='feasible',classification='feasible',actual_batch=1,
            forward_completed=True,latency_mean_ms=point['measurements']['latency_mean_ms']['value'],
            memory=dict(private_commit_limit_bytes=settings.cpu_memory_limit_bytes,input_limit_bytes=settings.input_limit_bytes),
            sweep=copy.deepcopy(point),worker_pid=12341,launcher_pid=12341,exit_code=0,
            worker_phase='runtime_prepared',runtime_ready=True,provenance=dict(device='cpu'))
        return row
    monkeypatch.setattr(capacity,'_candidate',candidate)
    def forbidden(*args,**kwargs):raise AssertionError('resource gate must reject before launch')
    monkeypatch.setattr(capacity.subprocess,'Popen',forbidden)
    data=capacity.find_max_batch_size(recipe,protocol=plan).to_dict()
    assert [r['status'] for r in data['raw_evidence']['candidates']]==['feasible','resource_limit','not_run']
    assert data['raw_evidence']['candidates'][0]['sweep']==point
    assert data['measurements']['prior_observed_feasible_batch']==1
    assert data['measurements']['maximum_observed_feasible_batch'] is None
    assert data['measurements']['maximum_proven_within_range'] is None


def test_model_snapshot_and_owned_payload_are_included():
    data = worker._admission((1, 1, 1, 1), 'fp64', 'cpu', 1000, 1000,
                             model_snapshot_bytes=600, largest_owned_tensor_bytes=400)
    assert data['host_private_payload_lower_bound_bytes'] == 1024
    assert data['decision'] == 'resource_limit'


def test_model_payload_deduplicates_objects_and_respects_shared_storage():
    storage=SimpleNamespace(nbytes=lambda:8)
    tensor=SimpleNamespace(numel=lambda:100,is_floating_point=lambda:True,
        element_size=lambda:4,untyped_storage=lambda:storage)
    model=SimpleNamespace(parameters=lambda:[tensor],buffers=lambda:[tensor])
    assert worker._model_payloads(model,'fp32')==(400,8)
    assert worker._model_payloads(model,'fp64')==(800,8)


def test_small_lower_bound_is_not_a_total_memory_guarantee():
    data = worker._admission((1, 1, 5, 7), 'fp32', 'cpu', 1024, 1024**3, 4, 4)
    assert data['decision'] == 'not_rejected_peak_unknown'
    assert data['workspace_peak_bytes'] is data['total_peak_bytes'] is None
    assert not data['fit_guaranteed']


def test_device_budget_is_separate_from_host_commit():
    data = worker._admission((1, 1, 5, 7), 'amp_fp16', 'cuda:0', 1024, 1024**3,
                             4, 4, allocator_limit=280)
    assert data['input_bytes'] == 140  # AMP master/input remains FP32.
    assert data['host_private_payload_lower_bound_bytes'] == 424
    assert data['device_payload_lower_bound_bytes'] == 284
    assert 'allocator budget' in data['reason']


@pytest.mark.parametrize('capture', [False, True])
def test_job_readback_and_known_inherited_ceiling(monkeypatch, capture):
    class Function:
        def __init__(self, call): self.call = call
        def __call__(self, *args): return self.call(*args)
    configured = {}
    def set_info(handle, kind, pointer, size):
        x = pointer._obj
        configured.update(flags=x.basic.flags, process=x.process_memory, job=x.job_memory)
        return 1
    def query(handle, kind, pointer, size, returned):
        x = pointer._obj
        if handle is None:
            x.basic.flags = 0x300; x.process_memory = 800; x.job_memory = 700
        else:
            x.basic.flags = configured['flags']
            x.process_memory = configured['process']; x.job_memory = configured['job']
        return 1
    api = SimpleNamespace(CreateJobObjectW=Function(lambda *a:101),
        SetInformationJobObject=Function(set_info), QueryInformationJobObject=Function(query),
        AssignProcessToJobObject=Function(lambda *a:1), GetCurrentProcess=Function(lambda:1),
        CloseHandle=Function(lambda *a:1))
    monkeypatch.setattr(worker,'os',SimpleNamespace(name='nt'))
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *a, **kw:api, raising=False)
    guard = worker._memory_guard(1000, capture=capture)
    assert len(guard) == (3 if capture else 2)  # Shared legacy users unchanged.
    if capture:
        assert guard[2]['known_commit_ceiling_bytes'] == 700
        assert guard[2]['own_limit_bytes'] == 1000
        assert 'unknown' in guard[2]['chain_scope']
        def corrupt_readback(handle, kind, pointer, size, returned):
            status=query(handle,kind,pointer,size,returned)
            if handle is not None:pointer._obj.basic.flags ^= 0x10
            return status
        api.QueryInformationJobObject=Function(corrupt_readback)
        with pytest.raises(RuntimeError,match='readback mismatch'):
            worker._memory_guard(1000,capture=True)


def test_stricter_job_gate_rejects_before_factory(monkeypatch, tmp_path):
    recipe = ColdStartRecipe('never_import:factory', input_shape=(1, 1, 8, 8)).to_dict()
    request = dict(recipe=recipe, batch=1, protocol=dict(cpu_memory_limit_bytes=1024**3,
        input_limit_bytes=1024**2))
    req=tmp_path/'request.json'; out=tmp_path/'output.json'; ready=tmp_path/'ready.json'
    req.write_text(json.dumps(request))
    closed=[]
    fake_api=SimpleNamespace(CloseHandle=lambda h:closed.append(h))
    monkeypatch.setattr(worker, '_memory_guard', lambda *a, **kw:(fake_api, 101,
        dict(known_commit_ceiling_bytes=700)))
    monkeypatch.setattr(worker.sys, 'argv', ['worker', str(req), str(out), str(ready), str(tmp_path)])
    worker.main()
    result=json.loads(out.read_text())
    assert result['status']=='resource_limit' and result['sweep'] is None
    assert 'lower bound' in result['reason'] and not result['forward_completed']
    assert closed==[101]


def test_private_commit_uses_private_usage_not_working_set(monkeypatch):
    class Function:
        def __init__(self, call):self.call=call
        def __call__(self,*args):return self.call(*args)
    def memory(handle,pointer,size):
        pointer._obj.private=1234;pointer._obj.ws=9876
        return 1
    kernel=SimpleNamespace(GetCurrentProcess=Function(lambda:1))
    api=SimpleNamespace(GetProcessMemoryInfo=Function(memory))
    monkeypatch.setattr(ctypes,'WinDLL',lambda name,**kw:kernel if name=='kernel32' else api,raising=False)
    assert worker._private_commit()==1234


def test_small_native_fixture_records_unknown_peak_admission():
    # Existing safe fixture/protocol, batch1 only; no forced allocation failure.
    from test_performance_max_batch import recipe,plan
    result=capacity.find_max_batch_size(recipe(reject=0),protocol=plan((1,))).to_dict()
    row=result['raw_evidence']['candidates'][0]
    assert row['status']=='feasible' and row['actual_batch']==1
    memory=row['memory'];evidence=memory['admission']
    assert evidence['host_private_payload_lower_bound_bytes']==3*140+4+4
    assert evidence['observed_current_private_commit_bytes']>0
    assert evidence['job_limits']['own_limit_bytes']==memory['private_commit_limit_bytes']
    assert evidence['decision']=='not_rejected_peak_unknown' and not evidence['fit_guaranteed']
    assert evidence['workspace_peak_bytes'] is evidence['total_peak_bytes'] is None
    assert memory['rss_before']['rss_bytes']>0 and memory['rss_after']['rss_bytes']>0
