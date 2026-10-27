"""Real, bounded GPU forwards; missing kernel activity is evidence, never a skip."""
import argparse
import ctypes as c
from ctypes import wintypes as w
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--installed-target',type=Path);parser.add_argument('--admit',type=Path,required=True)
parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
deadline=time.monotonic()+10
while not args.admit.exists():
    if time.monotonic()>deadline:raise RuntimeError('bounded Job admission missing; no Torch imported')
    time.sleep(.01)
assert args.admit.read_text()=='admitted','invalid Job gate'
assert 'torch' not in sys.modules,'native admission must precede Torch import'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


process=load(ROOT/'hyso/performance/_training_process.py','profile_job_layout')


def job_observation():
    probe=process._Job(4*1024**3,12*1024**3,8)
    try:
        info=probe.Extended();accounting=probe.Accounting()
        api=probe.api
        api.GetCurrentProcess.restype=w.HANDLE
        api.IsProcessInJob.argtypes=[w.HANDLE,w.HANDLE,c.POINTER(w.BOOL)];api.IsProcessInJob.restype=w.BOOL
        member=w.BOOL()
        assert api.IsProcessInJob(api.GetCurrentProcess(),None,c.byref(member)) and member.value
        assert api.QueryInformationJobObject(None,9,c.byref(info),c.sizeof(info),None)
        assert api.QueryInformationJobObject(None,1,c.byref(accounting),c.sizeof(accounting),None)
        class Pids(c.Structure):
            _fields_=[('assigned',w.DWORD),('listed',w.DWORD),('pids',c.c_size_t*8)]
        pids=Pids()
        assert api.QueryInformationJobObject(None,3,c.byref(pids),c.sizeof(pids),None)
        assert pids.assigned==pids.listed and pids.listed<=8
        assert info.basic.flags & 0x2308 == 0x2308
        assert (info.process_memory,info.job_memory,info.basic.processes)==(4*1024**3,12*1024**3,8)
        return dict(flags=info.basic.flags,process_bytes=info.process_memory,job_bytes=info.job_memory,
            process_limit=info.basic.processes,active_processes=accounting.active,total_processes=accounting.total,
            peak_job_bytes=info.peak_job,pids=list(pids.pids[:pids.listed]))
    finally:probe.close()


admission=load(ROOT/'tools/forensics/native_admission.py','profile_preflight')
preflight=admission.collect();preflight['job']=job_observation()
preflight['job_before_console_detach']=preflight['job']
# Windows may attach a private conhost to this headless query process. Detach
# only our console (never terminate a PID), then require the exact single PID.
console=c.WinDLL('kernel32',use_last_error=True)
console.FreeConsole.argtypes=[];console.FreeConsole.restype=w.BOOL
preflight['own_console_detached']=bool(console.FreeConsole())
# Query helpers are owned descendants too. Wait boundedly for their kernel
# process/Job accounting teardown; never admit an extra surviving PID.
settle=time.monotonic()+2
while (preflight['job']['active_processes']!=1 or preflight['job']['pids']!=[os.getpid()]) and time.monotonic()<settle:
    time.sleep(.02);preflight['job']=job_observation()
preflight['warnings']=['GPU process occupancy/utilization is telemetry, not an admission blocker']
preflight['torch_imported_before_admission']='torch' in sys.modules
preflight['blockers']=[]
if not all(s['uuid']==admission.policy.GPU_UUID for s in preflight['gpu_samples']):
    preflight['blockers'].append('device_identity_mismatch')
preflight['resource_policy']='RAM/VRAM/disk/PIDs/utilization are advisory; only identity/Job ownership are admission gates'
preflight['warnings'].append('Resource contention can perturb timings; occupancy alone does not establish a performance cause')
if preflight['job']['active_processes']!=1 or preflight['job']['pids']!=[os.getpid()]:preflight['blockers'].append('unexpected_owned_process')
preflight['policy_scope']='P13/P14 bounded profiler; GPU PID/utilization warnings per explicit task; P23 policy unchanged'
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.with_suffix('.preflight.json').write_text(json.dumps(preflight,indent=2),encoding='utf-8')
assert not preflight['blockers'],'native preflight blocked before Torch import'
if args.preflight_only:
    print(json.dumps(dict(status='PREFLIGHT_PASS',cuda_workload_started=False,job=preflight['job'])))
    raise SystemExit(0)
source_files=[ROOT/p for p in ('hyso/performance/profiling.py','hyso/performance/_profile_worker.py',
    'hyso/performance/_profile_evidence.py','hyso/performance/_operation_models.py','tests/profiling_factories.py',
    'tests/cuda_profiling_validation.py','tests/p21_native_supervisor.py')]
source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
sys.path.insert(0,str(args.installed_target.resolve() if args.installed_target else ROOT))
import numpy as np
import torch
from hyso.performance import (ColdStartRecipe,ProfilingProtocol,ProfilingError,
    profile_operators,profile_layers,analyze_flops)
assert torch.cuda.is_available(),'native CUDA required'
assert torch.__version__=='2.5.1+cu121','use the unchanged verified native runtime'
torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.1,0)
model=torch.nn.Conv2d(1,1,1).cuda().train();model.weight.grad=torch.ones_like(model.weight)
hook=model.register_forward_hook(lambda m,a,o:None);hooks=dict(model._forward_hooks)
weights={k:v.clone() for k,v in model.state_dict().items()};grad=model.weight.grad;gradient=grad.clone()
stream=torch.cuda.Stream();records=[]
environment=dict(torch=str(torch.__version__),cuda_build=torch.version.cuda,gpu=torch.cuda.get_device_name(),
    supported_activities=[str(a) for a in torch.profiler.supported_activities()],python=sys.version,source_hashes=source_hashes)


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


def capture(name,precision='fp32',factory='profiling_factories:tiny_cuda',shape=(1,2,3,3),kwargs=None,fail=None):
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
                if error.diagnostics['status']=='crashed':raise RuntimeError('native worker crash; stop P13/P14 series without retry') from error
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
            assert data['raw_evidence']['output']['dtype']==('torch.float64' if precision=='fp64' else 'torch.float16' if precision=='amp_fp16' else 'torch.float32')
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
            if factory.endswith(':tiny_cuda'):assert counts['measurements']['total_flops_per_batch']==2*(18*2+3*18)+18+3
            if factory.endswith(':ModularCNN'):assert counts['measurements']['status']=='partial'
            assert all(data['raw_evidence']['safety'].values())
            dead(data['raw_evidence']['worker']['worker_pid']);dead(data['raw_evidence']['worker']['launcher_pid'])
            records[-1]['job_after_capture']=job_observation()
            assert records[-1]['job_after_capture']['active_processes']==1
            for p in source_files:assert hashlib.sha256(p.read_bytes()).hexdigest()==source_hashes[str(p)]
            from hyso.performance import OperatorProfileResult,LayerProfileResult
            assert OperatorProfileResult(data).to_dict()==data and LayerProfileResult(layers).to_dict()==layers
            result.to_json(args.output.with_name(args.output.stem+'-'+name+'-operators.json'))
            result.to_csv(args.output.with_name(args.output.stem+'-'+name+'-operators.csv'))
            profile_layers(result).to_json(args.output.with_name(args.output.stem+'-'+name+'-layers.json'))
            profile_layers(result).to_csv(args.output.with_name(args.output.stem+'-'+name+'-layers.csv'))
            attribution=data['measurements']['cuda_attribution']
            records[-1]['correlation_summary']={k:attribution[k] for k in ('status','mapped_kernel_count','unmapped_kernel_count')}
            records[-1]['activity_usable']=activity['status']=='measured'
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
    records[-1]['status']='PASS' if records[-1]['activity_usable'] else 'UNSUPPORTED';save()
    return records[-1]['activity_usable']


try:
    if capture('tiny_fp32'):
        if capture('shared_nested_fp32',factory='profiling_factories:Shared',shape=(2,2,5,7)):
            capture('tiny_fp64',precision='fp64')
finally:
    hook.remove();save()
print(json.dumps(dict(status='PASS' if records and all(r['status']=='PASS' for r in records) else 'UNSUPPORTED',
    cases=len(records),environment=environment)))
