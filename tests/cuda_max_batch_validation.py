"""Bounded P17 real CUDA validation. Never retries an unexpected worker failure."""
import argparse
import json
from pathlib import Path
import random
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--installed-target',type=Path)
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(args.installed_target.resolve() if args.installed_target else ROOT))
import numpy as np
import torch
from hyso.performance import ColdStartRecipe,BatchSearchProtocol,find_max_batch_size

assert torch.cuda.is_available(),'real CUDA required; no skip/fallback'
torch.set_num_threads(1)
torch.cuda.set_device(0)
torch.cuda.init()
parent_model=torch.nn.Conv2d(1,1,1).cuda().train()
parent_model.weight.grad=torch.ones_like(parent_model.weight)
state={k:v.clone() for k,v in parent_model.state_dict().items()}
grad=parent_model.weight.grad;grad_value=grad.clone()
stream=torch.cuda.Stream()
records=[]
environment=dict(torch=str(torch.__version__),cuda_build=torch.version.cuda,python=sys.version,
    gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
    total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
    package_source=sys.modules['hyso.performance'].__file__)


def save():
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(environment=environment,cases=records),indent=2,allow_nan=False),encoding='utf-8')


def assert_dead(pid):
    if pid is None: return
    import ctypes as c
    from ctypes import wintypes as w
    api=c.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];api.OpenProcess.restype=w.HANDLE
    api.GetExitCodeProcess.argtypes=[w.HANDLE,c.POINTER(w.DWORD)];api.GetExitCodeProcess.restype=w.BOOL
    api.CloseHandle.argtypes=[w.HANDLE]
    handle=api.OpenProcess(0x1000,False,pid)
    if not handle: return
    try:
        code=w.DWORD();assert api.GetExitCodeProcess(handle,c.byref(code)) and code.value!=259
    finally: api.CloseHandle(handle)


def exercise(name,precision='fp32',pressure=False,timeout=False):
    recipe=ColdStartRecipe('torch.nn:Conv2d',device='cuda:0',precision=precision,
        input_shape=(1,16,128,128) if pressure else (1,1,7,9),
        model_kwargs=dict(in_channels=16 if pressure else 1,out_channels=16 if pressure else 2,
                          kernel_size=1,bias=False))
    if timeout:
        recipe=ColdStartRecipe('batch_factories:Selective',device='cuda:0',
            input_shape=(1,1,7,9),factory_search_path=ROOT/'tests',
            model_kwargs=dict(reject=1,failure='timeout'))
    plan=BatchSearchProtocol(candidates=(1,2,4,8,32,16) if pressure else (1,2,4),
        maximum_batch=32 if pressure else 4,cpu_memory_limit_bytes=3*1024**3,
        input_limit_bytes=64*1024**2,timeout_seconds=15 if timeout else 45,
        warmup_iterations=1,measure_iterations=3,repeats=2,cuda_memory_fraction=.02 if pressure else .1)
    py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
    allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved();peak=torch.cuda.max_memory_allocated()
    previous=torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        result=find_max_batch_size(recipe,protocol=plan).to_dict()
        assert torch.cuda.current_stream()==stream and torch.cuda.current_device()==0
    records.append(dict(name=name,result=result));save()
    rows=result['raw_evidence']['candidates']
    # The current run must stop, even if earlier points succeeded.
    if any(r['status']=='crashed' for r in rows):
        raise RuntimeError('NATIVE WORKER CRASH: stop B5 and preserve exact exit code')
    assert torch.cuda.current_stream()==previous
    assert random.getstate()==py and np.array_equal(np.random.get_state()[1],npstate[1])
    assert torch.equal(torch.get_rng_state(),cpu) and torch.equal(torch.cuda.get_rng_state(),gpu)
    assert torch.cuda.memory_allocated()==allocated and torch.cuda.memory_reserved()==reserved
    assert torch.cuda.max_memory_allocated()==peak
    assert parent_model.training and parent_model.weight.grad is grad and torch.equal(grad,grad_value)
    assert all(torch.equal(v,parent_model.state_dict()[k]) for k,v in state.items())
    assert all(r['process_terminated'] for r in rows)
    for row in rows:
        assert_dead(row['worker_pid']);assert_dead(row['launcher_pid'])
    if timeout:
        assert [r['status'] for r in rows]==['timed_out','not_run','not_run']
        assert rows[0]['runtime_ready'] and rows[0]['worker_phase']=='runtime_prepared'
        assert result['measurements']['status']=='aborted'
        return
    assert result['measurements']['status']=='complete',rows
    assert len({r['worker_pid'] for r in rows})==len(rows)
    assert all(r['exit_code']==0 for r in rows)
    if pressure:
        assert any(r['status']=='feasible' for r in rows)
        assert any(r['status']=='infeasible_oom' for r in rows)
        assert all(r['classification'] in ('feasible','infeasible') for r in rows)
        assert result['measurements']['maximum_proven_within_range'] is None
        assert rows[-2]['status']=='infeasible_oom' and rows[-1]['status']=='feasible'
    else:
        assert all(r['status']=='feasible' for r in rows)
        assert result['measurements']['maximum_observed_feasible_batch']==4
    for row in rows:
        if row['status']!='feasible': continue
        assert row['actual_batch']==row['batch'] and row['forward_completed']
        point=row['sweep'];assert point['provenance']['input']['device']=='cuda:0'
        assert point['provenance']['precision']['output_dtype']==('torch.float16' if precision=='amp_fp16' else 'torch.float32')
        assert point['raw_evidence']['benchmark']['timing']['timing_method']=='cuda_event_elapsed'
        assert row['memory']['cuda_allocator_probe']['scope']=='separate_single_forward'
        assert row['memory']['cuda_allocator_probe']['peak_allocated_bytes']>0
        assert row['provenance']['gpu']==environment['gpu']
        if args.installed_target:
            assert Path(row['provenance']['package_source']).resolve().is_relative_to(args.installed_target.resolve())


for name,options in [('fp32',{}),('amp_fp16',dict(precision='amp_fp16')),
                     ('allocator_budget_oom',dict(pressure=True)),('owned_timeout',dict(timeout=True)),
                     ('fresh_after_oom_and_timeout',{})]:
    exercise(name,**options)
print(json.dumps(dict(status='PASS',cases=len(records),environment=environment)))
