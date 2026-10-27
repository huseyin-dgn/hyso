"""P01/P02 arithmetic, completion evidence, lifetime and manual tracking."""
from fixture_support import example_fixture
from dataclasses import FrozenInstanceError
import csv
import gc
import json
import random
import re
from pathlib import Path
import weakref

import numpy as np
import pytest
import torch
from torch import nn

from hyso.benchmark import BenchmarkRunner, BenchmarkResult
from hyso.benchmark.result import _statistics, _throughput
from hyso.benchmark.runner import StateMutationError
from hyso.performance import (TailPolicy, ThroughputProtocol, LatencyAnalysis, ThroughputResult,
                              analyze_latency, measure_throughput)
from hyso.performance import inference as engine
from test_benchmark_state_safety import prepared, state, equal


def source(rows):
    data = BenchmarkRunner(nn.Identity(), input_shape=(1, 1, 2, 2), warmup_iterations=0,
                           measure_iterations=1, repeats=1).run().to_dict()
    data["configuration"].update(repeats=len(rows), measure_iterations=len(rows[0]))
    data["timing"]["latency_samples_ms"] = rows
    flat = sum(rows, [])
    data["stats"] = dict(per_repeat=[_statistics(row) for row in rows], pooled=_statistics(flat),
                         quantile_method="linear_(n-1)*q", std_ddof=1)
    data["throughput"].update(images_per_sec=_throughput(flat, 1),
                             per_repeat_images_per_sec=[_throughput(row, 1) for row in rows])
    return BenchmarkResult(data)


@pytest.fixture
def tiny():
    return measure_throughput(nn.Identity(), torch.ones(2, 1, 3, 5),
                              protocol=ThroughputProtocol(batch_count=3, warmup_iterations=0, repeats=1))


def test_known_array_p99_provenance_and_no_execution(monkeypatch):
    benchmark = source([list(map(float, range(1, 501))), list(map(float, range(501, 1001)))])
    before = benchmark.to_dict()
    monkeypatch.setattr(BenchmarkRunner, "run", lambda *a: pytest.fail("analysis must not execute"))
    result = analyze_latency(benchmark, quantiles=(.5, .9, .95, .99))
    data = result.to_dict()
    assert [m["value"] for m in data["measurements"]] == pytest.approx([500.5, 900.1, 950.05, 990.01])
    assert data["actual"] == dict(samples=1000, repeats=2, samples_per_repeat=[500, 500])
    assert data["identity"]["source_result_id"] == before["identity"]["run_id"]
    assert data["raw_evidence"]["latency_samples_ms"] == before["timing"]["latency_samples_ms"]
    assert data["measurements"][-1]["effective_tail_count"] == 10
    assert benchmark.to_dict() == before
    assert data["provenance"]["environment"] == before["environment"]


@pytest.mark.parametrize("n", [1, 10, 999])
def test_insufficient_tail_never_invents_p99(n):
    result = analyze_latency(source([list(map(float, range(1, n+1)))])).to_dict()
    metric = result["measurements"][0]
    assert metric["value"] is None and metric["status"] == "insufficient_samples"
    assert metric["reason"] == "tail_policy" and metric["unit"] == "ms"


@pytest.mark.parametrize("samples,q,expected", [([1., 3.], .5, 2.), ([1., 3.], .99, 2.98),
                                                ([2., 2., 2.], .95, 2.), ([1.], .5, 1.)])
def test_explicit_exploratory_policy(samples, q, expected):
    data = analyze_latency(source([samples]), quantiles=[q], tail_policy=TailPolicy(1, 1))
    # Min-tail=1 cannot license a .99 quantile from two points. An interior
    # quantile with enough tail is derived; insufficient status remains explicit.
    metric = data.to_dict()["measurements"][0]
    if (1-q)*len(samples) >= 1:
        assert metric["value"] == pytest.approx(expected)
    else:
        assert metric["status"] == "insufficient_samples"


@pytest.mark.parametrize("qs", [(), [0], [1], [-.1], [float('nan')], [True], [.99, .99], '.99'])
def test_invalid_quantiles(qs):
    with pytest.raises((TypeError, ValueError)):
        analyze_latency(source([[1.]]), quantiles=qs)


@pytest.mark.parametrize("policy", [dict(min_samples=0), dict(min_samples=True), dict(min_tail_samples=-1),
                                   dict(min_tail_samples=1.5)])
def test_invalid_tail_policy(policy):
    with pytest.raises(ValueError): TailPolicy(**policy)


def test_missing_raw_samples_and_bad_source():
    benchmark = source([[1.]])
    with pytest.raises(TypeError): analyze_latency(benchmark.to_dict())
    with pytest.raises(TypeError): analyze_latency(benchmark, tail_policy={})
    class Corrupt(BenchmarkResult):
        def to_dict(self):
            data = benchmark.to_dict()
            data["timing"]["latency_samples_ms"] = []
            return data
    with pytest.raises(ValueError): analyze_latency(Corrupt(benchmark.to_dict()))


@pytest.mark.parametrize("options", [dict(batch_count=0), dict(batch_count=-1), dict(batch_count=True),
    dict(batch_count=1.5), dict(batch_count=None), dict(duration_seconds=.1),
    dict(batch_count=None, duration_seconds=0), dict(batch_count=None, duration_seconds=-1),
    dict(batch_count=None, duration_seconds=float('inf')), dict(batch_count=None, duration_seconds=float('nan')),
    dict(batch_count=None, duration_seconds=True), dict(batch_count=None, duration_seconds=1e-20),
    dict(warmup_iterations=-1), dict(repeats=0), dict(repeats=True), dict(queue_depth=1),
    dict(queue_depth=True), dict(concurrency=2), dict(concurrency=True), dict(device='mps'),
    dict(device='cuda:-1'), dict(device='cuda:01'), dict(device='cpu:0'), dict(precision='amp_fp16'),
    dict(device='cuda', precision='fp64'), dict(precision='bf16')])
def test_invalid_protocol(options):
    with pytest.raises((ValueError, TypeError)): ThroughputProtocol(**options)


@pytest.mark.parametrize("batch", [1, 2, 7])
@pytest.mark.parametrize("dtype,precision", [(torch.float32, 'fp32'), (torch.float64, 'fp64')])
def test_actual_cpu_completion_cardinality_warmup_and_no_allocator(batch, dtype, precision, monkeypatch):
    calls = []
    class Count(nn.Module):
        def forward(self, x):
            calls.append(x.shape[0])
            return x * 2
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *a: pytest.fail('no reset'))
    monkeypatch.setattr(torch.Tensor, 'backward', lambda *a: pytest.fail('no backward'))
    result = measure_throughput(Count(), torch.ones(batch, 1, 3, 5, dtype=dtype), protocol=ThroughputProtocol(
        precision=precision, batch_count=5, repeats=2, warmup_iterations=3))
    data = result.to_dict()
    assert len(calls) == 1 + 2*(3+5)
    assert data['actual'] == dict(repeats=2, completed_batches=10, completed_images=10*batch, pending_batches=0)
    for row in data['raw_evidence']['windows']:
        assert row['completed_id_range'] == [0, 4] and row['completion_groups'] == 5
        assert row['terminal_drains'] == row['backpressure_syncs'] == 0
        assert row['images_per_second'] == 5*batch/row['elapsed_wall_seconds']
    assert data['measurements']['value'] == 10*batch/sum(r['elapsed_wall_seconds'] for r in data['raw_evidence']['windows'])


def test_exact_clock_and_tail_inclusion(monkeypatch):
    # Three clock readings: t0, admission stop, after terminal drain.
    clock = iter([100, 400, 600])
    monkeypatch.setattr(engine.time, 'perf_counter_ns', lambda: next(clock))
    data = measure_throughput(nn.Identity(), torch.ones(2, 1, 2, 2), protocol=ThroughputProtocol(
        batch_count=3, repeats=1, warmup_iterations=0)).to_dict()
    row = data['raw_evidence']['windows'][0]
    assert row['elapsed_wall_seconds'] == 500/1e9 and row['tail_seconds'] == 200/1e9
    assert row['images_per_second'] == 6/(500/1e9)


def test_duration_admission_boundary_deterministic(monkeypatch):
    clock = iter([100, 100, 200, 1200, 1300, 1400])
    monkeypatch.setattr(engine.time, 'perf_counter_ns', lambda: next(clock))
    data = measure_throughput(nn.Identity(), torch.ones(3, 1, 2, 2), protocol=ThroughputProtocol(
        batch_count=None, duration_seconds=1e-6, repeats=1, warmup_iterations=0)).to_dict()
    row = data['raw_evidence']['windows'][0]
    assert row['completed_batches'] == 2 and row['completed_images'] == 6
    assert row['elapsed_wall_seconds'] == 1300/1e9


def test_real_duration_window():
    result = measure_throughput(nn.Identity(), torch.ones(1, 1, 2, 2), protocol=ThroughputProtocol(
        batch_count=None, duration_seconds=.005, warmup_iterations=1, repeats=1)).to_dict()
    assert result['actual']['completed_batches'] > 0
    row = result['raw_evidence']['windows'][0]
    assert row['admission_end_ns'] - row['start_ns'] >= 5_000_000


@pytest.mark.parametrize("clock", [[100, 100, 100], [100, 90, 110], [100, 200, 150], [float('nan')]])
def test_invalid_clock_never_returns_success(clock, monkeypatch):
    values = iter(clock)
    monkeypatch.setattr(engine.time, 'perf_counter_ns', lambda: next(values))
    with pytest.raises((ValueError, FloatingPointError)):
        measure_throughput(nn.Identity(), torch.ones(1,1,2,2), protocol=ThroughputProtocol(
            batch_count=1, repeats=1, warmup_iterations=0))


def test_no_completed_duration_work(monkeypatch):
    values = iter([100, 10000, 10001, 10002])
    monkeypatch.setattr(engine.time, 'perf_counter_ns', lambda: next(values))
    with pytest.raises(RuntimeError, match='no completed'):
        measure_throughput(nn.Identity(), torch.ones(1,1,2,2), protocol=ThroughputProtocol(
            batch_count=None, duration_seconds=1e-9, repeats=1, warmup_iterations=0))


@pytest.mark.parametrize('ids', [[0,0], [-1], [1], [], [0,2]])
def test_duplicate_negative_inconsistent_completion_rejected(ids):
    ledger = engine._CompletionLedger(2)
    with pytest.raises(ValueError): ledger.commit(ids)


def test_committing_same_completion_twice_rejected():
    ledger = engine._CompletionLedger(2)
    ledger.commit([0,1])
    with pytest.raises(ValueError): ledger.commit([1,2])


def test_cpu_state_rng_modes_gradient_identity_and_no_transfer():
    model = prepared()
    x = torch.randn(2,1,5,7)
    xcopy = x.clone()
    before = state(model)
    measure_throughput(model, x, protocol=ThroughputProtocol(batch_count=3, repeats=2, warmup_iterations=1))
    equal(state(model), before)
    assert torch.equal(x, xcopy)


@pytest.mark.parametrize('when', [1, 2, 4])
def test_exception_cleanup_at_preflight_warmup_measurement(when):
    calls = []
    class Fail(nn.Sequential):
        def forward(self, x):
            calls.append(True)
            random.random(); np.random.random(); torch.rand(1)
            if len(calls) == when: raise LookupError('injected workload error')
            return super().forward(x)
    model = Fail(*list(prepared().children()))
    before = state(model)
    with pytest.raises(LookupError):
        measure_throughput(model, torch.ones(2,1,5,7), protocol=ThroughputProtocol(
            batch_count=3, warmup_iterations=1, repeats=1))
    equal(state(model), before)
    assert not engine._RUN_LOCK.locked()


@pytest.mark.parametrize('mutation', ['parameter','buffer','gradient','input','mode','data_write'])
def test_caller_mutation_rejected_and_values_restored(mutation):
    class Mutate(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.weight.grad = torch.full_like(self.weight,.5)
            self.register_buffer('counter',torch.ones(1))
        def forward(self,x):
            if mutation == 'parameter': self.weight.add_(1)
            elif mutation == 'buffer': self.counter.add_(1)
            elif mutation == 'gradient': self.weight.grad.add_(1)
            elif mutation == 'input': x.add_(1)
            elif mutation == 'mode': self.training=True
            else: self.weight.data.add_(1)
            return x*self.weight
    model = Mutate()
    weight, grad = model.weight, model.weight.grad
    with pytest.raises(StateMutationError):
        measure_throughput(model, torch.ones(1,1,2,2), protocol=ThroughputProtocol(batch_count=1))
    assert model.weight is weight and model.weight.grad is grad
    assert torch.equal(weight,torch.ones(1)) and torch.equal(grad,torch.full_like(grad,.5))
    assert torch.equal(model.counter,torch.ones(1)) and model.training


@pytest.mark.parametrize('value', [None, [], lambda: torch.ones(1), {'x':1}])
def test_invalid_input_source(value):
    with pytest.raises(TypeError): measure_throughput(nn.Identity(), value)


def test_invalid_protocol_type_and_cuda_no_fallback(monkeypatch):
    with pytest.raises(TypeError): measure_throughput(nn.Identity(),torch.ones(1,1,2,2),protocol={})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(ValueError,match='CUDA is unavailable'):
        measure_throughput(nn.Identity(),torch.ones(1,1,2,2),protocol=ThroughputProtocol(device='cuda'))
    assert not engine._RUN_LOCK.locked()


@pytest.mark.parametrize('case', ['nonfinite','shape','exception'])
def test_measured_forward_failure(case):
    calls=[]
    class Bad(nn.Module):
        def forward(self,x):
            calls.append(True)
            if len(calls)>1:
                if case=='exception': raise RuntimeError('forward failed')
                if case=='shape': return x[:,:,:1,:]
                return x*float('inf')
            return x.clone()
    with pytest.raises((ValueError,RuntimeError,FloatingPointError)):
        measure_throughput(Bad(),torch.ones(1,1,2,2),protocol=ThroughputProtocol(
            batch_count=2,repeats=1,warmup_iterations=0))


@pytest.mark.parametrize('rejected',[False,True])
def test_no_gc_required_for_models_tensors_outputs(rejected):
    def exercise():
        model=prepared()
        refs=[weakref.ref(model),*[weakref.ref(p) for p in model.parameters()]]
        if rejected: model[-1].train=lambda mode=True:None
        try:
            measure_throughput(model,torch.ones(2,1,5,7),protocol=ThroughputProtocol(batch_count=2,repeats=1))
        except ValueError:
            assert rejected
        return refs
    enabled=gc.isenabled(); gc.disable()
    try: assert all(ref() is None for ref in exercise())
    finally:
        gc.collect()
        if enabled: gc.enable()


@pytest.mark.parametrize('kind',['latency','throughput'])
def test_result_immutable_defensive_exports_atomic_csv(kind,tiny,tmp_path,monkeypatch):
    result=analyze_latency(source([[1.,2.]])) if kind=='latency' else tiny
    data=result.to_dict()
    copy=result.to_dict(); copy['diagnostics']['warnings'].append('changed')
    assert result.to_dict()==data
    with pytest.raises(FrozenInstanceError): result._payload='bad'
    with pytest.raises((FrozenInstanceError,AttributeError,TypeError)): result.arbitrary='bad'
    path=tmp_path/'result.json'; summary=tmp_path/'result.csv'
    result.to_json(path); result.to_csv(summary)
    assert json.loads(path.read_text(encoding='utf-8'))==data
    with summary.open(encoding='utf-8',newline='') as handle: rows=list(csv.DictReader(handle))
    assert len(rows)==1 and rows[0]['unit']==('ms' if kind=='latency' else 'images/s')
    with pytest.raises(FileExistsError): result.to_json(path)
    result.to_json(path,overwrite=True)
    before=path.read_bytes()
    import hyso.benchmark.result as exports
    def fail(*args): raise OSError('publication failed')
    monkeypatch.setattr(exports.os,'replace',fail)
    with pytest.raises(OSError): result.to_json(path,overwrite=True)
    assert path.read_bytes()==before and not list(tmp_path.glob('.hyso-export-*'))


def test_csv_formula_safety(tmp_path):
    benchmark=source([[1.,2.]]).to_dict()
    # Legacy source identity accepts caller labels; the analysis preserves it
    # faithfully in JSON and protects spreadsheet users at its CSV boundary.
    benchmark['identity']['run_id']='  =1+2'
    result=analyze_latency(BenchmarkResult(benchmark))
    destination=tmp_path/'safe.csv'
    result.to_csv(destination)
    with destination.open(encoding='utf-8',newline='') as handle: rows=list(csv.DictReader(handle))
    assert rows[0]['source_result_id']=="'  =1+2"


@pytest.mark.parametrize('case',['version','nan','negative','count','pending','duplicate','tail','rate','no_work','unit','requested'])
def test_result_rejects_forged_throughput_evidence(tiny,case):
    data=tiny.to_dict(); row=data['raw_evidence']['windows'][0]
    if case=='version': data['schema_version']=2
    elif case=='nan': data['measurements']['value']=float('nan')
    elif case=='negative': row['completed_batches']=-1
    elif case=='count': row['completed_images']+=1
    elif case=='pending': row['pending_batches']=1
    elif case=='duplicate': row['completed_id_range']=[0,0]
    elif case=='tail': row['admission_end_ns']=row['end_ns']+1
    elif case=='rate': row['images_per_second']+=1
    elif case=='no_work': row['completed_batches']=row['admitted_batches']=0
    elif case=='unit': data['measurements']['unit']='FPS'
    else: data['requested']['batch_count']=4
    with pytest.raises(ValueError): ThroughputResult(data)


@pytest.mark.parametrize('case',['quantile','count','raw','policy','status','identity'])
def test_result_rejects_forged_latency(case):
    data=analyze_latency(source([[1.,2.]])).to_dict()
    if case=='quantile': data['measurements'][0]['value']=3
    elif case=='count': data['actual']['samples']=3
    elif case=='raw': data['raw_evidence']['latency_samples_ms']=[]
    elif case=='policy': data['protocol']['tail_policy']['min_samples']=0
    elif case=='status': data['measurements'][0]['status']='derived'
    else: data['identity']['source_result_id']=None
    with pytest.raises(ValueError): LatencyAnalysis(data)


def test_reentrant_measurement_fails_cleanly():
    engine._RUN_LOCK.acquire()
    try:
        with pytest.raises(RuntimeError,match='concurrent'):
            measure_throughput(nn.Identity(),torch.ones(1,1,2,2))
    finally: engine._RUN_LOCK.release()


def test_executable_guide_artifact_reopen(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    guide=example_fixture('docs/guides/performance-inference.md')
    namespace={}
    for block in re.findall(r'```python\n(.*?)```',guide.read_text(encoding='utf-8'),re.S):
        exec(compile(block,str(guide),'exec'),namespace)
    from hyso.experiments import ExperimentManager
    manager=ExperimentManager(namespace['root'])
    run=manager.get_run(namespace['run_id'])
    assert run.to_dict()['status']=='completed'
    assert len(run.read_metrics())==2
    assert manager.verify_artifact(namespace['artifact_id'])['status']=='verified'
    assert manager.get_artifact(namespace['artifact_id'])['expected_size_bytes']>0
    artifact=manager.get_artifact(namespace['artifact_id'])
    stored=Path(namespace['root'])/artifact['managed_relative_path']
    assert json.loads(stored.read_text(encoding='utf-8'))==namespace['throughput'].to_dict()
    manager.close()


@pytest.mark.parametrize('case',['device','clock','sync','bool_count','missing_field','terminal','groups'])
def test_result_rejects_invalid_runtime_protocol_evidence(tiny,case):
    data=tiny.to_dict()
    if case=='device': data['provenance']['environment']['device']='cuda:0'
    elif case=='clock': data['protocol']['clock']='cuda_events'
    elif case=='sync': data['protocol']['synchronization']='none'
    elif case=='bool_count': data['actual']['pending_batches']=False
    elif case=='missing_field': del data['provenance']['environment']
    elif case=='terminal': data['raw_evidence']['windows'][0]['terminal_drains']=1
    else: data['raw_evidence']['windows'][0]['completion_groups']=1
    with pytest.raises(ValueError): ThroughputResult(data)


def test_window_has_no_file_database_network_or_hash_work(monkeypatch):
    import builtins
    import socket
    import sqlite3
    from unittest.mock import patch
    original=engine._window
    def forbidden(*args,**kwargs): raise AssertionError('I/O or hashing inside measurement window')
    def guarded(*args,**kwargs):
        with patch.object(builtins,'open',forbidden), patch.object(sqlite3,'connect',forbidden), \
                patch.object(socket,'socket',forbidden), patch.object(engine.hashlib,'sha256',forbidden):
            return original(*args,**kwargs)
    monkeypatch.setattr(engine,'_window',guarded)
    measure_throughput(nn.Identity(),torch.ones(1,1,2,2),protocol=ThroughputProtocol(
        batch_count=3,repeats=1,warmup_iterations=0))


@pytest.mark.parametrize('bad_input',[torch.ones(1,2),torch.ones(1,1,2,2,dtype=torch.int64),
    torch.ones(1,1,2,2,requires_grad=True),torch.ones(1,1,2,3).transpose(2,3),
    torch.full((1,1,2,2),float('nan'))])
def test_unsupported_resident_input_profiles(bad_input):
    with pytest.raises((ValueError,TypeError)):
        measure_throughput(nn.Identity(),bad_input,protocol=ThroughputProtocol(batch_count=1))
