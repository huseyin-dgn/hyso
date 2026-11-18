"""Native process observations, artifact accounting, failure/ownership invariants."""
from fixture_support import example_fixture
from dataclasses import FrozenInstanceError
import csv
import gc
import hashlib
import json
import os
import pickle
from pathlib import Path
import random
import re
import threading
import time
import weakref

import numpy as np
import pytest
import torch
from torch import nn

from hyso.benchmark import BenchmarkRunner
from hyso.performance import MemoryProtocol, MemoryResult, FootprintResult, measure_memory, measure_footprint
from hyso.performance import memory as engine
from test_benchmark_state_safety import prepared, state, equal


def sampler_threads():
    return [t for t in threading.enumerate() if t.name=='hyso-rss-sampler']


@pytest.fixture
def snapshot():
    return measure_memory()


@pytest.fixture
def footprint():
    return measure_footprint(nn.Linear(2,3))


def test_native_os_read_and_lifetime_scope(snapshot):
    data=snapshot.to_dict(); sample=data['raw_evidence']['observations'][0]
    assert sample['rss_bytes']>0 and sample['lifetime_hwm_bytes']>=sample['rss_bytes']
    assert data['provenance']['environment']['pid']==os.getpid()
    assert data['measurements']['sampled_peak_rss_bytes']['value']==sample['rss_bytes']
    assert data['protocol']['lifetime_peak_reset'] is False
    assert data['actual']['observations']==1 and data['raw_evidence']['gaps_ns']==[]
    assert sample['backend'].startswith('windows.' if os.name=='nt' else 'linux.')


def test_real_tensor_touch_release_sampled_window(monkeypatch):
    observed=threading.Event(); held=[]; refs=[]; phase=[]
    original=engine._observe
    def observe():
        row=original()
        if held: phase.append(row); observed.set()
        return row
    monkeypatch.setattr(engine,'_observe',observe)
    def workload():
        tensor=torch.empty(8*1024*1024,dtype=torch.float32)
        tensor.fill_(1.25)
        held.append(tensor); refs.append(weakref.ref(tensor))
        assert observed.wait(10), 'sampler failed to observe touched native allocation'
        held.clear()
    data=measure_memory(workload,protocol=MemoryProtocol(interval_seconds=.002)).to_dict()
    rows=data['raw_evidence']['observations']
    assert phase and all(ref() is None for ref in refs)
    assert data['measurements']['sampled_peak_rss_bytes']['value']==max(r['rss_bytes'] for r in rows)
    assert rows[-1]['lifetime_hwm_bytes']>=max(r['rss_bytes'] for r in rows)
    assert not sampler_threads()


def test_gap_and_query_overhead_evidence():
    data=measure_memory(lambda:time.sleep(.025),protocol=MemoryProtocol(interval_seconds=.002)).to_dict()
    rows=data['raw_evidence']['observations']
    assert len(rows)>=2
    assert data['raw_evidence']['gaps_ns']==[b['timestamp_ns']-a['timestamp_ns'] for a,b in zip(rows,rows[1:])]
    assert all(r['query_end_ns']>=r['query_start_ns'] for r in rows)
    assert data['actual']['duration_seconds']==(data['actual']['end_ns']-data['actual']['start_ns'])/1e9


@pytest.mark.parametrize('text', ['VmRSS: 2 kB\nVmHWM: 3 kB\n', 'Name: test\nVmHWM:\t3 kB\nVmRSS:\t2 kB\n'])
def test_linux_parser_units_and_order(text):
    assert engine._linux_status(text)==(2048,3072)


@pytest.mark.parametrize('text',['','VmRSS: 2 kB','VmRSS: -1 kB\nVmHWM: 3 kB',
    'VmRSS: 2 MB\nVmHWM: 3 kB','VmRSS: 4 kB\nVmHWM: 3 kB',
    'VmRSS: 2 kB\nVmRSS: 2 kB\nVmHWM: 3 kB'])
def test_invalid_linux_parser_is_not_native_pass(text):
    with pytest.raises(engine._BackendUnavailable): engine._linux_status(text)


@pytest.mark.parametrize('system',['Darwin','Unknown'])
def test_unsupported_os_null_no_workload(monkeypatch,system):
    monkeypatch.setattr(engine.platform,'system',lambda:system)
    called=[]
    data=measure_memory(lambda:called.append(True)).to_dict()
    assert not called and data['actual']['observations']==0
    assert all(m['value'] is None and m['status']=='unsupported' and m['reason'] for m in data['measurements'].values())


def test_backend_permission_error(monkeypatch):
    monkeypatch.setattr(engine.platform,'system',lambda:'Windows')
    def fail(): raise PermissionError('native access denied')
    monkeypatch.setattr(engine,'_windows_rss',fail)
    assert 'access denied' in measure_memory().to_dict()['measurements']['current_rss_bytes']['reason']


@pytest.mark.parametrize('options',[dict(interval_seconds=0),dict(interval_seconds=-1),dict(interval_seconds=True),
    dict(interval_seconds=float('nan')),dict(interval_seconds=float('inf')),dict(interval_seconds=1e-20),
    dict(max_samples=1),dict(max_samples=True),dict(max_samples=2.5)])
def test_invalid_memory_protocol(options):
    with pytest.raises(ValueError): MemoryProtocol(**options)


def test_invalid_workload_protocol_and_async():
    async def asynchronous(): return 1
    for workload in ([],42,asynchronous):
        with pytest.raises(TypeError): measure_memory(workload)
    with pytest.raises(TypeError): measure_memory(protocol={})
    with pytest.raises(TypeError): measure_memory(lambda:(x for x in range(1)))


def test_workload_failure_stops_thread_restores_rng_and_releases_lock():
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    def fail():
        random.random(); np.random.random(); torch.rand(2)
        time.sleep(.005)
        raise LookupError('workload failed')
    with pytest.raises(LookupError): measure_memory(fail,protocol=MemoryProtocol(interval_seconds=.001))
    assert not sampler_threads() and not engine._RUN_LOCK.locked()
    after=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    equal(after,before)


def test_sampler_failure_and_budget_exhaustion(monkeypatch):
    original=engine._observe; calls=[]
    def observe():
        calls.append(True)
        if len(calls)>1: raise engine._BackendUnavailable('backend lost')
        return original()
    monkeypatch.setattr(engine,'_observe',observe)
    with pytest.raises(RuntimeError,match='sampler failed'):
        measure_memory(lambda:time.sleep(.02),protocol=MemoryProtocol(interval_seconds=.001))
    assert not sampler_threads() and not engine._RUN_LOCK.locked()
    monkeypatch.setattr(engine,'_observe',original)
    with pytest.raises(RuntimeError,match='sampler failed'):
        measure_memory(lambda:time.sleep(.02),protocol=MemoryProtocol(interval_seconds=.001,max_samples=2))
    assert not sampler_threads() and not engine._RUN_LOCK.locked()


def test_thread_start_failure_cleanup(monkeypatch):
    def fail(*args): raise RuntimeError('thread creation failed')
    monkeypatch.setattr(engine.threading.Thread,'start',fail)
    with pytest.raises(RuntimeError,match='thread creation failed'): measure_memory(lambda:None)
    assert not engine._RUN_LOCK.locked()


def test_no_automatic_gpu_calls_and_no_window_writes(monkeypatch):
    import builtins
    import socket
    import sqlite3
    from unittest.mock import patch
    def forbidden(*args,**kwargs): raise AssertionError('forbidden instrumentation side effect')
    for name in ('synchronize','reset_peak_memory_stats','empty_cache','get_rng_state','set_rng_state'):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    def workload():
        with patch.object(builtins,'open',forbidden),patch.object(socket,'socket',forbidden), \
                patch.object(sqlite3,'connect',forbidden),patch.object(engine.hashlib,'sha256',forbidden):
            time.sleep(.01)
    measure_memory(workload,protocol=MemoryProtocol(interval_seconds=.002))
    measure_footprint(nn.Linear(2,2))


def test_logical_reuses_benchmark_unchanged_accounting():
    model=prepared(); before=state(model)
    expected=BenchmarkRunner(model,input_shape=(2,1,5,7),measure_iterations=1,repeats=1,warmup_iterations=0).run().to_dict()['parameters']
    data=measure_footprint(model).to_dict()
    assert all(data['raw_evidence']['logical'][key]==value for key,value in expected.items())
    equal(state(model),before)
    assert data['measurements']['physical_memory_bytes']==dict(value=None,unit='byte',status='unsupported',reason='physical_memory_not_measured')


class Aliases(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.arange(6,dtype=torch.float32).reshape(2,3))
        self.tied=self.weight
        self.view=nn.Parameter(self.weight.detach().view(3,2))
        self.register_buffer('cross',self.weight)
        self.register_buffer('integer',torch.arange(3,dtype=torch.int64))
        self.register_buffer('boolean',torch.tensor([True,False]))
        self.register_buffer('half_buffer',torch.ones(2,dtype=torch.float16))
        self.double_weight=nn.Parameter(torch.ones(2,dtype=torch.float64),requires_grad=False)


def test_tied_views_cross_category_dtypes_and_buffers(tmp_path):
    model=Aliases(); before=state(model)
    data=measure_footprint(model,serialize_to=tmp_path/'aliases.pt',verify_roundtrip=True).to_dict()
    logical=data['raw_evidence']['logical']
    assert logical['total_parameters']==14 and logical['parameter_bytes']==64
    assert logical['buffer_bytes']==24+24+2+4
    assert logical['parameter_buffer_bytes']==64+24+2+4
    assert logical['unique_parameter_tensors']==3 and logical['unique_buffer_tensors']==4
    assert logical['cross_category_object_alias'] and logical['distinct_objects_share_storage']
    assert data['raw_evidence']['artifact']['format_validation']=='roundtrip_verified'
    equal(state(model),before)


def test_nonpersistent_buffers_and_shared_module_serialization(tmp_path):
    class Shared(nn.Module):
        def __init__(self):
            super().__init__(); self.a=nn.Linear(2,2); self.b=self.a
            self.register_buffer('transient',torch.ones(7),persistent=False)
    model=Shared(); destination=tmp_path/'shared.pt'
    data=measure_footprint(model,serialize_to=destination,verify_roundtrip=True).to_dict()
    loaded=torch.load(destination,weights_only=True)
    assert set(loaded)=={'a.weight','a.bias','b.weight','b.bias'}
    assert data['raw_evidence']['logical']['buffer_bytes']==28


def test_true_stat_streamed_sha_and_no_unsafe_load(tmp_path,monkeypatch):
    path=tmp_path/'opaque.onnx'; path.write_bytes(b'not an ONNX graph, just caller owned bytes'*100)
    def forbidden(*args,**kwargs): raise AssertionError('never deserialize caller artifact')
    monkeypatch.setattr(torch,'load',forbidden)
    data=measure_footprint(nn.Linear(2,3),artifact_path=path,artifact_format='onnx',
                           expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()).to_dict()
    artifact=data['raw_evidence']['artifact']
    assert artifact['size_bytes']==path.stat().st_size and artifact['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert artifact['format_validation']=='caller_declared_not_deserialized' and artifact['model_match']=='not_verified'
    assert data['measurements']['parameter_bytes']['value']!=artifact['size_bytes']


def test_explicit_serialization_uses_weights_only_and_preserves_state(tmp_path,monkeypatch):
    calls=[]; original=torch.load
    def load(path,**kwargs):
        calls.append(kwargs)
        assert kwargs==dict(map_location='cpu',weights_only=True)
        return original(path,**kwargs)
    monkeypatch.setattr(torch,'load',load)
    model=prepared(); before=state(model); path=tmp_path/'state.pt'
    data=measure_footprint(model,serialize_to=path,verify_roundtrip=True).to_dict()
    equal(state(model),before)
    assert len(calls)==1 and data['raw_evidence']['artifact']['size_bytes']==path.stat().st_size
    assert not list(tmp_path.glob('.hyso-footprint-*'))
    original_bytes=path.read_bytes()
    with pytest.raises(FileExistsError): measure_footprint(model,serialize_to=path)
    assert path.read_bytes()==original_bytes
    measure_footprint(model,serialize_to=path,overwrite=True,verify_roundtrip=True)


@pytest.mark.parametrize('options',[dict(artifact_path='x',serialize_to='y'),dict(verify_roundtrip=True),
    dict(artifact_path='x'),dict(artifact_path='x',artifact_format=''),dict(artifact_path='x',artifact_format=3),
    dict(serialize_to='x',artifact_format='onnx'),dict(expected_sha256='a'*64),
    dict(artifact_path='x',artifact_format='x',expected_sha256='bad'),dict(overwrite=True),
    dict(verify_roundtrip=1),dict(overwrite=1),dict(artifact_format='x')])
def test_invalid_artifact_workflow(options):
    with pytest.raises((ValueError,TypeError)): measure_footprint(nn.Linear(1,1),**options)


@pytest.mark.parametrize('case',['empty','missing','directory','hash'])
def test_invalid_or_corrupt_artifact(tmp_path,case):
    path=tmp_path/'artifact'
    if case=='empty': path.write_bytes(b'')
    elif case=='directory': path.mkdir()
    elif case=='hash': path.write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        measure_footprint(nn.Linear(1,1),artifact_path=path,artifact_format='caller_state_dict',
                          expected_sha256='0'*64 if case=='hash' else None)


@pytest.mark.parametrize('failure',['save','roundtrip','publish'])
def test_serialization_failure_cleanup_state_and_ownership(tmp_path,monkeypatch,failure):
    model=prepared(); before=state(model); path=tmp_path/'state.pt'
    def fail(*args,**kwargs): raise OSError('injected file error')
    if failure=='save': monkeypatch.setattr(torch,'save',fail)
    elif failure=='roundtrip': monkeypatch.setattr(torch,'load',fail)
    else: monkeypatch.setattr(engine.os,'link',fail)
    with pytest.raises(OSError): measure_footprint(model,serialize_to=path,verify_roundtrip=True)
    assert not path.exists() and not list(tmp_path.glob('.hyso-footprint-*'))
    equal(state(model),before)
    assert not engine._RUN_LOCK.locked()


def test_corrupt_own_serialization_never_published(tmp_path,monkeypatch):
    def corrupt(state,handle): handle.write(b'invalid serialized state')
    monkeypatch.setattr(torch,'save',corrupt)
    with pytest.raises((pickle.UnpicklingError,RuntimeError,EOFError,ValueError)):
        measure_footprint(nn.Linear(1,1),serialize_to=tmp_path/'bad.pt',verify_roundtrip=True)
    assert not (tmp_path/'bad.pt').exists() and not list(tmp_path.glob('.hyso-footprint-*'))


@pytest.mark.parametrize('case',['meta','sparse','hook','custom_state','cycle'])
def test_unsupported_model_profiles(case):
    model=nn.Linear(2,2)
    if case=='meta': model=model.to('meta')
    elif case=='sparse': model.register_buffer('sparse',torch.eye(2).to_sparse())
    elif case=='hook': model.register_forward_hook(lambda *args:None)
    elif case=='custom_state':
        class Custom(nn.Linear):
            def state_dict(self,*args,**kwargs): return super().state_dict(*args,**kwargs)
        model=Custom(2,2)
    else: model.add_module('cycle',model)
    with pytest.raises(ValueError): measure_footprint(model)


@pytest.mark.parametrize('kind',['memory','footprint'])
def test_frozen_plain_result_exports_and_atomic_failure(kind,snapshot,footprint,tmp_path,monkeypatch):
    result=snapshot if kind=='memory' else footprint
    before=result.to_dict(); copy=result.to_dict(); copy['diagnostics']['warnings'].append('changed')
    assert result.to_dict()==before
    with pytest.raises(FrozenInstanceError): result._payload='changed'
    path=tmp_path/'result.json'; csv_path=tmp_path/'result.csv'
    result.to_json(path); result.to_csv(csv_path)
    assert json.loads(path.read_text(encoding='utf-8'))==before
    with csv_path.open(encoding='utf-8',newline='') as handle: rows=list(csv.DictReader(handle))
    assert rows
    with pytest.raises(FileExistsError): result.to_json(path)
    def fail(*args): raise OSError('atomic replace failed')
    monkeypatch.setattr(engine.os,'replace',fail)
    with pytest.raises(OSError): result.to_json(path,overwrite=True)
    assert json.loads(path.read_text(encoding='utf-8'))==before and not list(tmp_path.glob('.hyso-export-*'))


@pytest.mark.parametrize('field',['rss','hwm','count','gap','clock','reset','unit','nan','version'])
def test_reject_forged_memory(snapshot,field):
    data=snapshot.to_dict(); row=data['raw_evidence']['observations'][0]
    if field=='rss': row['rss_bytes']=-1
    elif field=='hwm': row['lifetime_hwm_bytes']=0
    elif field=='count': data['actual']['observations']=2
    elif field=='gap': data['raw_evidence']['gaps_ns']=[1]
    elif field=='clock': row['timestamp_ns']=data['actual']['end_ns']+1
    elif field=='reset': data['protocol']['lifetime_peak_reset']=True
    elif field=='unit': data['measurements']['current_rss_bytes']['unit']='MiB'
    elif field=='nan': data['measurements']['current_rss_bytes']['value']=float('nan')
    else: data['schema_version']=2
    with pytest.raises(ValueError): MemoryResult(data)


@pytest.mark.parametrize('field',['logical','artifact','hash','physical','count','nan'])
def test_reject_forged_footprint(footprint,field):
    data=footprint.to_dict()
    if field=='logical': data['raw_evidence']['logical']['parameter_bytes']+=1
    elif field=='artifact': data['raw_evidence']['artifact']['size_bytes']=10
    elif field=='hash': data['raw_evidence']['artifact']['sha256']='a'*64
    elif field=='physical': data['measurements']['physical_memory_bytes']['value']=0
    elif field=='count': data['actual']['tensor_entries']+=1
    else: data['measurements']['buffer_bytes']['value']=float('inf')
    with pytest.raises(ValueError): FootprintResult(data)


def test_cpu_tensor_model_lifetimes_do_not_need_gc():
    def run():
        model=prepared(); refs=[weakref.ref(model),*[weakref.ref(p) for p in model.parameters()]]
        measure_footprint(model)
        return refs
    enabled=gc.isenabled(); gc.disable()
    try: assert all(ref() is None for ref in run())
    finally:
        gc.collect()
        if enabled: gc.enable()


def test_reentrant_calls_fail_without_deadlock():
    engine._RUN_LOCK.acquire()
    try:
        with pytest.raises(RuntimeError): measure_memory()
        with pytest.raises(RuntimeError): measure_footprint(nn.Linear(1,1))
    finally: engine._RUN_LOCK.release()


def test_executable_memory_guide_and_experiment_reopen(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    guide=example_fixture('docs/guides/performance-memory.md')
    namespace={}
    for block in re.findall(r'```python\n(.*?)```',guide.read_text(encoding='utf-8'),re.S):
        exec(compile(block,str(guide),'exec'),namespace)
    from hyso.experiments import ExperimentManager
    manager=ExperimentManager(namespace['root'])
    run=manager.get_run(namespace['run_id'])
    assert run.to_dict()['status']=='completed' and len(run.read_metrics())==3
    for identity in namespace['artifact_ids']: assert manager.verify_artifact(identity)['status']=='verified'
    assert namespace['artifact_info']['sha256']==hashlib.sha256(namespace['model_path'].read_bytes()).hexdigest()
    manager.close()


def test_changed_artifact_during_stream_hash(tmp_path,monkeypatch):
    path=tmp_path/'changing.bin'; path.write_bytes(b'initial bytes')
    original=Path.open
    class Reader:
        def __init__(self): self.handle=original(path,'rb'); self.changed=False
        def __enter__(self): return self
        def __exit__(self,*args): self.handle.close()
        def fileno(self): return self.handle.fileno()
        def read(self,size):
            chunk=self.handle.read(size)
            if not self.changed:
                self.changed=True
                with original(path,'ab') as other: other.write(b'concurrent modification')
            return chunk
    def opened(self,*args,**kwargs):
        return Reader() if self==path and args==('rb',) else original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'open',opened)
    with pytest.raises(ValueError,match='changed during'):
        measure_footprint(nn.Linear(1,1),artifact_path=path,artifact_format='opaque')


def test_unsupported_csv_reason_formula_safety(tmp_path,monkeypatch):
    monkeypatch.setattr(engine.platform,'system',lambda:'Unknown')
    data=measure_memory().to_dict()
    for metric in data['measurements'].values(): metric['reason']='=1+2'
    result=MemoryResult(data); destination=tmp_path/'unsupported.csv'; result.to_csv(destination)
    with destination.open(encoding='utf-8',newline='') as handle: rows=list(csv.DictReader(handle))
    assert all(row['reason']=="'=1+2" for row in rows)


@pytest.mark.parametrize('field',['verified','ownership','actual_bool'])
def test_artifact_cannot_claim_unverified_format_model_match(footprint,field):
    data=footprint.to_dict()
    if field=='verified': data['protocol']['verify_roundtrip']=1
    elif field=='ownership': data['raw_evidence']['artifact']['ownership']='manager'
    else: data['actual']['artifacts']=False
    with pytest.raises(ValueError): FootprintResult(data)
