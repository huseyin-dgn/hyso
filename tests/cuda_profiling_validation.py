"""Real, bounded GPU forwards; missing kernel activity is evidence, never a skip."""
import argparse
import json
from pathlib import Path
import random
import sys

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--installed-target',type=Path);args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(args.installed_target.resolve() if args.installed_target else ROOT))
import numpy as np
import torch
from hyso.performance import (ColdStartRecipe,ProfilingProtocol,ProfilingError,
    profile_operators,profile_layers,analyze_flops)
assert torch.cuda.is_available(),'native CUDA required'
torch.set_num_threads(1);torch.cuda.set_device(0)
model=torch.nn.Conv2d(1,1,1).cuda().train();model.weight.grad=torch.ones_like(model.weight)
hook=model.register_forward_hook(lambda m,a,o:None);hooks=dict(model._forward_hooks)
weights={k:v.clone() for k,v in model.state_dict().items()};grad=model.weight.grad;gradient=grad.clone()
stream=torch.cuda.Stream();records=[]
environment=dict(torch=str(torch.__version__),cuda_build=torch.version.cuda,gpu=torch.cuda.get_device_name(),
    supported_activities=[str(a) for a in torch.profiler.supported_activities()],python=sys.version)


def save():
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(environment=environment,cases=records),indent=2,allow_nan=False),encoding='utf-8')


def dead(pid):
    if pid is None:return
    import ctypes as c
    from ctypes import wintypes as w
    api=c.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];api.OpenProcess.restype=w.HANDLE
    api.GetExitCodeProcess.argtypes=[w.HANDLE,c.POINTER(w.DWORD)];api.GetExitCodeProcess.restype=w.BOOL
    api.CloseHandle.argtypes=[w.HANDLE]
    handle=api.OpenProcess(0x1000,False,pid)
    if handle:
        try:
            code=w.DWORD();assert api.GetExitCodeProcess(handle,c.byref(code)) and code.value!=259
        finally:api.CloseHandle(handle)


def capture(name,precision='fp32',factory='profiling_factories:Shared',shape=(2,2,5,7),kwargs=None,fail=None):
    recipe=ColdStartRecipe(factory,factory_search_path=ROOT/'tests',model_kwargs=kwargs,
        input_shape=shape,precision=precision,device='cuda:0')
    plan=ProfilingProtocol(warmup_iterations=0 if fail=='timed_out' else 1,measure_iterations=2,
        timeout_seconds=15 if fail=='timed_out' else 45,cuda_memory_fraction=.1)
    py=random.getstate();ns=np.random.get_state();cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
    allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved();peak=torch.cuda.max_memory_allocated()
    previous=torch.cuda.current_stream();trace=args.output.with_name(args.output.stem+'-'+name+'-trace.json')
    try:
        with torch.cuda.stream(stream):
            try:
                result=profile_operators(recipe,protocol=plan,trace_path=trace)
            except ProfilingError as error:
                records.append(dict(id=name,status='OBSERVED_ERROR',diagnostics=error.diagnostics));save()
                if error.diagnostics['status']=='crashed':raise RuntimeError('unexpected native worker crash; stop B6') from error
                assert fail and error.diagnostics['status']==fail,error.diagnostics
                assert error.diagnostics['process_terminated']
                dead(error.diagnostics['worker_pid']);dead(error.diagnostics['launcher_pid'])
                if fail=='failed':assert all(error.diagnostics['cleanup'].values())
                if fail=='timed_out':assert error.diagnostics['worker_phase']=='profiler_active'
                records[-1]['status']='PASS';save();return
            assert fail is None
            data=result.to_dict();counts=analyze_flops(result).to_dict();layers=profile_layers(result).to_dict()
            records.append(dict(id=name,status='CAPTURED',operators=data,counts=counts,layers=layers));save()
            assert data['raw_evidence']['output']['device']=='cuda:0'
            assert data['raw_evidence']['output']['dtype']==('torch.float16' if precision=='amp_fp16' else 'torch.float32')
            assert data['measurements']['operators'] and data['raw_evidence']['events']
            assert all(row['cuda_activity_us'] is None for row in layers['raw_evidence']['invocations'])
            raw=json.loads(trace.read_text(encoding='utf-8'));actual=[e for e in raw['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X']
            activity=data['measurements']['cuda_activity']
            if actual:
                assert len(actual)==activity['kernel_count']
                assert data['measurements']['cuda_kernels'] and activity['status'] in ('measured','partial')
            else:
                assert activity['status']=='unsupported' and activity['total_kernel_us'] is None
                assert activity['kernel_count'] is None and data['measurements']['cuda_kernels']==[]
            if factory.endswith(':Shared'):assert counts['measurements']['total_flops_per_batch']==1820
            if factory.endswith(':ModularCNN'):assert counts['measurements']['status']=='partial'
            assert all(data['raw_evidence']['safety'].values())
            dead(data['raw_evidence']['worker']['worker_pid']);dead(data['raw_evidence']['worker']['launcher_pid'])
            if args.installed_target:
                assert Path(data['provenance']['environment']['package_source']).resolve().is_relative_to(args.installed_target.resolve())
            assert torch.cuda.current_stream()==stream
    finally:
        assert torch.cuda.current_stream()==previous and torch.cuda.current_device()==0
        assert random.getstate()==py and np.array_equal(np.random.get_state()[1],ns[1])
        assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
        assert torch.cuda.memory_allocated()==allocated and torch.cuda.memory_reserved()==reserved and torch.cuda.max_memory_allocated()==peak
        assert model.training and model.weight.grad is grad and torch.equal(grad,gradient) and dict(model._forward_hooks)==hooks
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in weights.items())
    records[-1]['status']='PASS';save()


capture('fp32')
capture('amp_fp16',precision='amp_fp16')
capture('modular_eca',factory='hyso.models.modular:ModularCNN',shape=(1,1,5,7),
    kwargs=dict(input_channels=1,num_classes=2,widths=[3],repeats=[1],strides=[1],attention='eca'))
capture('forward_exception_cleanup',kwargs=dict(fail=True),fail='failed')
capture('active_profiler_timeout',factory='batch_factories:Selective',shape=(1,1,5,7),
    kwargs=dict(reject=1,failure='timeout'),fail='timed_out')
capture('fresh_after_failure')
hook.remove()
print(json.dumps(dict(status='PASS',cases=len(records),environment=environment)))
