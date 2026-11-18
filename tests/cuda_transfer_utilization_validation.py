"""Bounded installed-package P15 copies and P16 idle/load sensor evidence."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
import time

parser=argparse.ArgumentParser();parser.add_argument('--installed-target',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
target=args.installed_target.resolve();sys.path.insert(0,str(target))
import numpy as np
import torch
from hyso.performance import (TransferProtocol,TransferError,measure_transfers,UtilizationProtocol,measure_utilization)
from hyso.performance.cold import _terminate
from hyso.experiments import ExperimentManager
assert torch.cuda.is_available()
torch.set_num_threads(1);torch.cuda.set_device(0)
model=torch.nn.Conv2d(1,1,1).cuda().train();model.weight.grad=torch.ones_like(model.weight)
hook=model.register_forward_hook(lambda *a:None);hooks=dict(model._forward_hooks)
grad=model.weight.grad;weights={k:v.clone() for k,v in model.state_dict().items()};g=grad.clone()
stream=torch.cuda.Stream();cases=[]
environment=dict(torch=str(torch.__version__),cuda_build=torch.version.cuda,gpu=torch.cuda.get_device_name(),
    supported_activities=[str(a) for a in torch.profiler.supported_activities()],
    nocupti='LIBKINETO_NOCUPTI' in torch.__config__.show(),target=str(target))


def save():
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(environment=environment,cases=cases),indent=2,allow_nan=False),encoding='utf-8')


def dead(pid):
    import ctypes as c
    from ctypes import wintypes as w
    api=c.WinDLL('kernel32',use_last_error=True);api.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];api.OpenProcess.restype=w.HANDLE
    api.GetExitCodeProcess.argtypes=[w.HANDLE,c.POINTER(w.DWORD)];api.GetExitCodeProcess.restype=w.BOOL
    api.CloseHandle.argtypes=[w.HANDLE]
    handle=api.OpenProcess(0x1000,False,pid)
    if handle:
        try:
            code=w.DWORD();assert api.GetExitCodeProcess(handle,c.byref(code)) and code.value!=259
        finally:api.CloseHandle(handle)


def preserved(call):
    py=random.getstate();ns=np.random.get_state();cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
    allocator=(torch.cuda.memory_allocated(),torch.cuda.memory_reserved(),torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())
    previous=torch.cuda.current_stream()
    try:
        with torch.cuda.stream(stream):
            result=call();assert torch.cuda.current_stream()==stream
    except TransferError as error:
        cases.append(dict(status='ERROR',diagnostics=error.diagnostics));save()
        if error.diagnostics['status']=='crashed':raise RuntimeError('B4 native stop: unexpected crash') from error
        raise
    finally:
        assert torch.cuda.current_stream()==previous and torch.cuda.current_device()==0
        assert random.getstate()==py and np.array_equal(np.random.get_state()[1],ns[1])
        assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
        assert allocator==(torch.cuda.memory_allocated(),torch.cuda.memory_reserved(),torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())
        assert model.training and model.weight.grad is grad and torch.equal(grad,g) and dict(model._forward_hooks)==hooks
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in weights.items())
    return result


results=[]
for pinned in (False,True):
    for nonblocking in (False,True):
        result=preserved(lambda:measure_transfers(protocol=TransferProtocol(pinned=pinned,non_blocking=nonblocking,
            warmup_iterations=1,measure_iterations=3,repeats=2)))
        d=result.to_dict();receipt=d['raw_evidence']['worker']
        assert receipt['process_terminated'];dead(receipt['worker_pid']);dead(receipt['launcher_pid'])
        assert d['actual']['completed_copies']==12 and d['actual']['warmup_copies']==4
        assert Path(d['provenance']['environment']['package_source']).resolve().is_relative_to(target)
        assert all(r['transfer_bytes']==1048576 and r['terminal_complete'] and r['equal'] and r['host_pinned']==pinned
                   for r in d['raw_evidence']['copies'])
        assert all(m['completed_bytes_per_second']>0 for m in d['measurements']['directions'])
        assert d['measurements']['overlap']['value'] is None and d['measurements']['asynchronous_execution']['value'] is None
        cases.append(dict(id='copy_pinned_'+str(pinned)+'_nonblocking_'+str(nonblocking),status='PASS',result=d));results.append(result);save()

result=preserved(lambda:measure_transfers(protocol=TransferProtocol(shape=(131072,),dtype='uint8',pinned=True,
    non_blocking=True,warmup_iterations=1,measure_iterations=2,repeats=1)))
assert result.to_dict()['raw_evidence']['tensor']['transfer_bytes']==131072
dead(result.to_dict()['raw_evidence']['worker']['worker_pid'])
cases.append(dict(id='uint8_exact_bytes',status='PASS',result=result.to_dict()));save()

idle=preserved(lambda:measure_utilization(protocol=UtilizationProtocol(duration_seconds=1.2,interval_seconds=.15,gpu=True,workload_label='idle_no_owned_GPU_work')))
assert idle.to_dict()['measurements']['gpu']['fields']['gpu_busy_percent']['status']=='measured'
cases.append(dict(id='idle_gpu_sensor',status='PASS',result=idle.to_dict()));results.append(idle);save()

with tempfile.TemporaryDirectory(prefix='hyso-b7-load-') as directory:
    directory=Path(directory);ready=directory/'ready';finished=directory/'finished.json';script=directory/'load.py'
    contents="""import importlib.util,json,sys,time
from pathlib import Path
target,ready,finished=sys.argv[1:]
spec=importlib.util.spec_from_file_location('guard',Path(target)/'hyso/performance/_batch_worker.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);guard=module._memory_guard(3*1024**3)
import torch
torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.1)
a=torch.ones(256,256,device='cuda');stream=torch.cuda.Stream()
torch.cuda.synchronize()
start=time.perf_counter_ns();Path(ready).write_text(str(start));n=0
with torch.inference_mode(),torch.cuda.stream(stream):
 while time.perf_counter_ns()-start<3*10**9:
  b=a@a;stream.synchronize();n+=1;time.sleep(.002)
Path(finished).write_text(json.dumps(dict(start_ns=start,end_ns=time.perf_counter_ns(),iterations=n)))
"""
    script.write_text(contents,encoding='utf-8')
    with tempfile.TemporaryFile() as log:
        process=subprocess.Popen([sys._base_executable,'-I','-B',str(script),str(target),str(ready),str(finished)],stdout=log,stderr=log)
        try:
            deadline=time.perf_counter()+30
            while not ready.exists() and process.poll() is None and time.perf_counter()<deadline:time.sleep(.02)
            assert ready.exists(),'bounded load did not become ready'
            load=preserved(lambda:measure_utilization(protocol=UtilizationProtocol(duration_seconds=2.,interval_seconds=.15,gpu=True,
                workload_label='separate_owned_3s_256x256_matmul_child_nonexclusive')))
            process.wait(timeout=10)
            assert process.returncode==0,'native load process failed; stop'
            workload=json.loads(finished.read_text());assert workload['iterations']>0
            d=load.to_dict();assert d['measurements']['gpu']['fields']['gpu_busy_percent']['status']=='measured'
            assert workload['start_ns']<d['actual']['end_ns'] and workload['end_ns']>d['actual']['start_ns']
            cases.append(dict(id='controlled_gpu_load_sensor',status='PASS',result=d,workload=workload));results.append(load);save()
        finally:
            if process.poll() is None:_terminate(process)
            process.wait(timeout=10);dead(process.pid)

stop=threading.Event()
def busy_cpu():
    while not stop.is_set():sum(i*i for i in range(1000))
thread=threading.Thread(target=busy_cpu);thread.start()
try:
    cpu=preserved(lambda:measure_utilization(protocol=UtilizationProtocol(duration_seconds=.4,interval_seconds=.03,
        workload_label='same_process_CPU_thread')))
finally:stop.set();thread.join(timeout=2)
assert not thread.is_alive() and cpu.to_dict()['measurements']['cpu']['single_core_percent']>0
cases.append(dict(id='native_CPU_load',status='PASS',result=cpu.to_dict()));results.append(cpu);save()

# All persistence is outside transfer/sampling windows.
store=args.output.with_suffix('.store');m=ExperimentManager(store);e=m.create_experiment('B7');m.close()
ExperimentManager.migrate_store(store);m=ExperimentManager(store);run=m.create_run(e['id'],'benchmark',config={'feature':'P15/P16'})
run.start();artifacts=[]
for i,result in enumerate(results):
    path=args.output.with_name(args.output.stem+'-result-'+str(i)+'.json');result.to_json(path);result.to_csv(path.with_suffix('.csv'))
    artifacts.append(run.add_artifact(path,kind='performance_result',ownership='copy'))
    run.log_metric(name='B7.observation',value=i,unit='count',step=i,step_kind='observation',source='performance.transfer_utilization')
rid=run.id;run.finish();m.close();m=ExperimentManager(store)
assert len(m.get_run(rid).read_metrics())==len(results)
assert all(m.verify_artifact(a['id'])['status']=='verified' for a in artifacts);m.close()
cases.append(dict(id='experiment_reopen_sha',status='PASS',artifacts=len(artifacts)));save();hook.remove()
print(json.dumps(dict(status='PASS',cases=len(cases),environment=environment)))
