"""Standalone real CUDA P01/P02 gate; no unavailable-device skip or fallback.

python -B tests/cuda_performance_validation.py --output <evidence.json>
Optional --installed-target selects the final wheel's isolated install.
"""
import argparse
import gc
import importlib
import json
from pathlib import Path
import random
import sys
import traceback
from unittest.mock import patch
import weakref

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--installed-target', type=Path)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(args.installed_target.resolve() if args.installed_target else ROOT))
import numpy as np
import torch
from torch import nn
from hyso.performance import ThroughputProtocol, measure_throughput, analyze_latency
from hyso.performance import inference as engine
from hyso.benchmark import BenchmarkRunner
from hyso.benchmark.runner import StateMutationError

if not torch.cuda.is_available():
    raise RuntimeError('Native CUDA required; unavailable is failure')
torch.set_num_threads(1)
RESULTS = []


def network():
    return nn.Sequential(nn.Conv2d(1,3,3,padding=1),nn.BatchNorm2d(3),nn.Dropout(.5),
                         nn.AdaptiveAvgPool2d(1)).to('cuda:0')


def snapshot(model):
    return dict(tensors=[(t,t.data_ptr(),t._version,t.cpu().clone()) for t in
                         list(model.parameters())+list(model.buffers())],
        grads=[(p,p.grad,None if p.grad is None else p.grad.cpu().clone()) for p in model.parameters()],
        modes=[(m,m.training) for m in model.modules()], python=random.getstate(),
        numpy=np.random.get_state(), cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(0))


def assert_snapshot(before):
    for t,pointer,version,value in before['tensors']:
        assert t.data_ptr()==pointer and t._version==version and torch.equal(t.cpu(),value)
    for p,grad,value in before['grads']:
        assert p.grad is grad
        if grad is not None: assert torch.equal(grad.cpu(),value)
    assert all(m.training==mode for m,mode in before['modes'])
    assert random.getstate()==before['python']
    now=np.random.get_state()
    assert now[0]==before['numpy'][0] and np.array_equal(now[1],before['numpy'][1]) and now[2:]==before['numpy'][2:]
    assert torch.equal(torch.get_rng_state(),before['cpu'])
    assert torch.equal(torch.cuda.get_rng_state(0),before['cuda'])


def run_case(precision='fp32', batch=2, nondefault=False, duration=False, outer=False):
    model=network()
    for i,m in enumerate(model.modules()): m.training=bool(i%2)
    for p in model.parameters(): p.grad=torch.full_like(p,.25)
    inputs=torch.ones(batch,1,9,11,device='cuda:0')
    copy=inputs.cpu().clone()
    stream=torch.cuda.Stream() if nondefault else torch.cuda.current_stream(0)
    torch.cuda.synchronize(0)
    before=snapshot(model)
    previous=torch.cuda.current_stream(0)
    device_before=torch.cuda.current_device()
    waits=[]
    original=torch.cuda.Event.synchronize
    def wait(event):
        waits.append(event.cuda_event)
        return original(event)
    with torch.cuda.stream(stream), patch.object(torch.cuda.Event,'synchronize',wait), \
            patch.object(torch.cuda,'synchronize',wraps=torch.cuda.synchronize) as device_drains, \
            patch.object(torch.cuda,'reset_peak_memory_stats',side_effect=AssertionError('no peak reset')):
        plan=ThroughputProtocol(device='cuda:0',precision=precision,warmup_iterations=2,repeats=2,queue_depth=8,
            batch_count=None if duration else 33,duration_seconds=.02 if duration else None)
        with torch.inference_mode(outer):
            result=measure_throughput(model,inputs,protocol=plan)
        assert torch.cuda.current_stream(0)==stream
        assert device_drains.call_count==plan.repeats  # warmup drain only, never per forward
    assert torch.cuda.current_stream(0)==previous and torch.cuda.current_device()==device_before
    assert_snapshot(before)
    assert torch.equal(inputs.cpu(),copy)
    data=result.to_dict()
    assert data['provenance']['environment']['stream_id']==stream.cuda_stream
    for row in data['raw_evidence']['windows']:
        if not duration: assert row['completed_batches']==33
        assert row['pending_batches']==0 and row['max_pending_batches']<=8
        assert row['completed_images']==row['completed_batches']*batch
        assert row['completed_id_range']==[0,row['completed_batches']-1]
        assert row['terminal_drains']==1 and row['tail_seconds']>=0
        assert row['images_per_second']==row['completed_images']/row['elapsed_wall_seconds']
        assert row['backpressure_syncs']<=row['completed_batches']//8
    if not duration:
        assert len(waits)<=2*(2+33//8) and len(waits)<33
    return dict(actual=data['actual'],windows=data['raw_evidence']['windows'],event_wait_count=len(waits),
                device_drain_count=device_drains.call_count,
                protocol=data['protocol'],provenance=data['provenance'])


def lifetime():
    event_refs=[]
    max_events=[0]
    event_type=torch.cuda.Event
    def new_event():
        event=event_type()
        event_refs.append(weakref.ref(event))
        max_events[0]=max(max_events[0],sum(ref() is not None for ref in event_refs))
        return event
    def exercise():
        refs=[]
        max_live=[0]
        class Track(nn.Module):
            def __init__(self):
                super().__init__(); self.weight=nn.Parameter(torch.ones(1,device='cuda:0'))
            def forward(self,x):
                out=x*self.weight
                refs.append(weakref.ref(out))
                max_live[0]=max(max_live[0],sum(ref() is not None for ref in refs))
                return out
        model=Track(); model_ref=weakref.ref(model); weight_ref=weakref.ref(model.weight)
        with patch.object(torch.cuda,'Event',side_effect=new_event):
            result=measure_throughput(model,torch.ones(2,1,9,11,device='cuda:0'),protocol=ThroughputProtocol(
                device='cuda:0',batch_count=35,queue_depth=4,repeats=2,warmup_iterations=1))
        assert max_live[0]<=4
        assert all(ref() is None for ref in refs)
        assert len(event_refs)==10 and max_events[0]<=5 and all(ref() is None for ref in event_refs)
        return model_ref,weight_ref,result,max_live[0]
    enabled=gc.isenabled(); gc.disable()
    try:
        model_ref,weight_ref,result,live=exercise()
        assert model_ref() is None and weight_ref() is None
        return dict(max_live_outputs=live,max_live_events=max_events[0],actual=result.to_dict()['actual'])
    finally:
        gc.collect()
        if enabled: gc.enable()


def failure(when):
    calls=[]
    class Fail(nn.Sequential):
        def forward(self,x):
            calls.append(True); random.random(); np.random.random(); torch.rand(1,device='cuda:0')
            if len(calls)==when: raise LookupError('native forward exception')
            return super().forward(x)
    model=Fail(*list(network().children())); before=snapshot(model)
    try:
        measure_throughput(model,torch.ones(2,1,9,11,device='cuda:0'),protocol=ThroughputProtocol(
            device='cuda:0',batch_count=9,warmup_iterations=1,repeats=1,queue_depth=4))
    except LookupError: pass
    else: raise AssertionError('expected failure')
    assert_snapshot(before)
    assert not engine._RUN_LOCK.locked()
    return dict(calls=len(calls),error_cleanup=True)


def injected_terminal_failure(incomplete=False):
    original=torch.cuda.Event
    created=[]
    class Event:
        def __init__(self):
            self.inner=original(); self.waits=0; self.terminal=len(created)==4
            created.append(self)
        def record(self,stream): return self.inner.record(stream)
        def synchronize(self):
            self.waits+=1
            self.inner.synchronize()
            if self.terminal and self.waits==2 and not incomplete:
                raise RuntimeError('injected asynchronous terminal failure')
        def query(self):
            if self.terminal and self.waits==2 and incomplete: return False
            return self.inner.query()
    model=network(); before=snapshot(model)
    with patch.object(torch.cuda,'Event',Event):
        try:
            measure_throughput(model,torch.ones(2,1,9,11,device='cuda:0'),protocol=ThroughputProtocol(
                device='cuda:0',batch_count=5,warmup_iterations=0,repeats=1,queue_depth=4))
        except RuntimeError as error:
            assert 'incomplete' in str(error) if incomplete else 'injected asynchronous' in str(error)
        else: raise AssertionError('expected terminal failure')
    assert_snapshot(before)
    assert not engine._RUN_LOCK.locked()
    return dict(real_cuda_workload=True,fault_injection='incomplete_query' if incomplete else 'terminal_drain_exception')


def cuda_p99():
    result=BenchmarkRunner(network(),device='cuda:0',input_shape=(1,1,5,7),warmup_iterations=1,
                           measure_iterations=500,repeats=2).run()
    before=result.to_dict()
    with patch.object(BenchmarkRunner,'run',side_effect=AssertionError('no rerun')):
        analysis=analyze_latency(result).to_dict()
    assert analysis['measurements'][0]['status']=='derived'
    assert analysis['actual']['samples']==1000 and result.to_dict()==before
    return dict(source_result_id=analysis['identity']['source_result_id'],metric=analysis['measurements'][0])


def mutation():
    class Mutate(nn.Module):
        def __init__(self):
            super().__init__(); self.weight=nn.Parameter(torch.ones(1,device='cuda:0'))
        def forward(self,x): self.weight.add_(1); return x*self.weight
    model=Mutate(); weight=model.weight
    try: measure_throughput(model,torch.ones(1,1,2,2,device='cuda:0'),protocol=ThroughputProtocol(device='cuda:0'))
    except StateMutationError: pass
    else: raise AssertionError('expected mutation failure')
    assert model.weight is weight and torch.equal(weight,torch.ones_like(weight)) and model.training
    return dict(rollback=True)


cases=[]
for precision in ('fp32','amp_fp16'):
    for nondefault in (False,True):
        cases.append((precision+'_'+('nondefault' if nondefault else 'default'),
                      lambda p=precision,s=nondefault:run_case(p,3,s)))
    cases.append((precision+'_duration',lambda p=precision:run_case(p,1,True,True)))
cases += [('outer_inference',lambda:run_case(outer=True)),('bounded_lifetime',lifetime),
          ('preflight_exception',lambda:failure(1)),('warmup_exception',lambda:failure(2)),
          ('measured_exception',lambda:failure(5)),('terminal_async_fault_injection',injected_terminal_failure),
          ('incomplete_tail_fault_injection',lambda:injected_terminal_failure(True)),
          ('cuda_raw_p99',cuda_p99),('mutation_rollback',mutation)]
for name,operation in cases:
    try:
        evidence=operation()
        RESULTS.append(dict(id=name,status='PASS',evidence=evidence))
    except BaseException:
        RESULTS.append(dict(id=name,status='FAIL',traceback=traceback.format_exc()))
    print(name,RESULTS[-1]['status'],flush=True)
report=dict(python=sys.version,torch=str(torch.__version__),cuda_build=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),
    selected_device=torch.cuda.current_device(),cases=RESULTS,
    imports={name:str(Path(importlib.import_module(name).__file__).resolve()) for name in
             ('hyso.performance','hyso.performance.inference','hyso.performance.result')})
if args.installed_target:
    assert all(Path(path).is_relative_to(args.installed_target.resolve()) for path in report['imports'].values())
args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
raise SystemExit(int(any(case['status']!='PASS' for case in RESULTS)))
