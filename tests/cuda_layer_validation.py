"""P13 CUDA Events acceptance under the unchanged bounded native supervisor."""
import argparse
import ctypes as c
from ctypes import wintypes as w
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--admit',type=Path,required=True)
parser.add_argument('--case',choices=('all','remaining'),default='all');args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
# Reuse the existing readback/PID/resource preflight in this very Job/process.
# Its preflight-only entrypoint exits before importing Torch or running P14.
path=ROOT/'tests/cuda_profiling_validation.py'
spec=importlib.util.spec_from_file_location('p13_native_preflight',path)
preflight=importlib.util.module_from_spec(spec);argv=sys.argv
try:
    sys.argv=[str(path),'--output',str(args.output.with_suffix('.bootstrap.json')),
              '--admit',str(args.admit),'--preflight-only']
    try:spec.loader.exec_module(preflight)
    except SystemExit as exit:
        if exit.code!=0:raise
finally:sys.argv=argv
assert 'torch' not in sys.modules
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from hyso.performance import (ColdStartRecipe,ProfilingProtocol,ProfilingError,
    OperatorProfileResult,LayerProfileResult,profile_operators,profile_layers,analyze_flops)

assert torch.__version__=='2.5.1+cu121' and torch.cuda.is_available()
torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.1,0)
model=torch.nn.Conv2d(1,1,1).cuda().train();model.weight.grad=torch.ones_like(model.weight)
optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9);optimizer.step()
momentum=optimizer.state[model.weight]['momentum_buffer'];momentum_copy=momentum.clone()
grad=model.weight.grad;gradient=grad.clone();weights={k:v.clone() for k,v in model.state_dict().items()}
hook=model.register_forward_hook(lambda m,a,o:None);hooks=dict(model._forward_hooks)
stream=torch.cuda.Stream();records=[]
files=[ROOT/p for p in ('hyso/performance/profiling.py','hyso/performance/_profile_worker.py',
    'hyso/performance/_profile_evidence.py','hyso/performance/_operation_models.py',
    'tests/profiling_factories.py','tests/cuda_layer_validation.py','tests/cuda_profiling_validation.py','tests/p21_native_supervisor.py')]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
environment=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),source_hashes=hashes)


def save():
    args.output.write_text(json.dumps(dict(environment=environment,cases=records),indent=2,allow_nan=False),encoding='utf-8')


def dead(pid):
    api=c.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];api.OpenProcess.restype=w.HANDLE
    api.GetExitCodeProcess.argtypes=[w.HANDLE,c.POINTER(w.DWORD)];api.GetExitCodeProcess.restype=w.BOOL
    api.CloseHandle.argtypes=[w.HANDLE]
    handle=api.OpenProcess(0x1000,False,pid)
    if handle:
        try:
            code=w.DWORD();assert api.GetExitCodeProcess(handle,c.byref(code)) and code.value!=259
        finally:api.CloseHandle(handle)


def capture(name,factory='tiny_cuda',shape=(1,2,3,3),precision='fp32',kwargs=None,unsupported=False,failure=False):
    recipe=ColdStartRecipe('profiling_factories:'+factory,factory_search_path=ROOT/'tests',
        input_shape=shape,device='cuda:0',precision=precision,model_kwargs=kwargs)
    protocol=ProfilingProtocol(warmup_iterations=1,measure_iterations=2,timeout_seconds=45,
        cpu_memory_limit_bytes=3*1024**3,input_limit_bytes=32*1024**2,cuda_memory_fraction=.1)
    py=random.getstate();ns=np.random.get_state();cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
    allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved();peak=torch.cuda.max_memory_allocated()
    previous=torch.cuda.current_stream();set_stream=torch.cuda.set_stream
    record=dict(id=name,status='STARTED');records.append(record);save()
    try:
        with torch.cuda.stream(stream):
            try:
                source=profile_operators(recipe,protocol=protocol)
            except ProfilingError as error:
                record.update(status='OBSERVED_ERROR',diagnostics=error.diagnostics);save()
                assert failure and error.diagnostics['status']=='failed',error.diagnostics
                assert error.diagnostics['exit_code']==0 and all(error.diagnostics['cleanup'].values())
                assert error.diagnostics['process_terminated']
                dead(error.diagnostics['worker_pid']);dead(error.diagnostics['launcher_pid'])
            else:
                assert not failure
                layers=profile_layers(source);data=source.to_dict();layer=layers.to_dict()
                record.update(source=data,layers=layer);save()
                assert data['raw_evidence']['output']['device']=='cuda:0'
                assert data['raw_evidence']['output']['dtype']==('torch.float64' if precision=='fp64' else 'torch.float32')
                raw=data['raw_evidence'];cfg=raw['cuda_event_timing'];rows=layer['raw_evidence']['invocations']
                assert cfg['requested'] and cfg['synchronized'] and cfg['device']=='cuda:0'
                assert len(cfg['pairs'])==len(rows)
                if unsupported:
                    assert cfg['scope_reason']=='multiple_or_changed_CUDA_streams'
                    assert layer['measurements']['cuda_event_timing']['status']=='unsupported'
                    assert all(r['cuda_elapsed_us'] is None and r['cuda_elapsed_status']=='unsupported' for r in rows)
                    assert layer['measurements']['status']=='partial'
                else:
                    assert cfg['scope_reason'] is None and layer['measurements']['status']=='measured'
                    assert layer['measurements']['cuda_event_timing']['status']=='measured'
                    for r,pair in zip(rows,cfg['pairs']):
                        assert r['invocation_id']==pair['invocation_id'] and r['measurement_index']==pair['measurement_index']
                        assert pair['start_event_index']==2*r['invocation_id'] and pair['end_event_index']==2*r['invocation_id']+1
                        assert pair['start_stream_id']==pair['end_stream_id']==cfg['stream_id']
                        assert all(pair[k] is True for k in ('start_recorded','end_recorded','start_complete','end_complete'))
                        assert pair['elapsed_ms']>0 and np.isfinite(pair['elapsed_ms'])
                        assert r['cuda_elapsed_us']==pair['elapsed_ms']*1000
                assert all(r['cuda_activity_us'] is None and r['cuda_exclusive_elapsed_us'] is None for r in rows)
                if factory=='Shared':
                    proj=next(m for m in layer['measurements']['modules'] if m['path']=='proj')
                    assert proj['invocation_count']==6 and set(proj['aliases'])=={'proj','alias','branch.1'}
                    assert len(rows)==12 and analyze_flops(source).to_dict()['measurements']['total_flops_per_batch']==1820
                    assert np.isclose(proj['cuda_elapsed_us'],sum(r['cuda_elapsed_us'] for r in rows if r['module_id']==proj['module_id']),rtol=1e-12,atol=0)
                if factory=='tiny_cuda':assert analyze_flops(source).to_dict()['measurements']['total_flops_per_batch']==201
                assert all(raw['safety'].values())
                assert OperatorProfileResult(data).to_dict()==data and LayerProfileResult(layer).to_dict()==layer
                for kind,result in (('source',source),('layers',layers)):
                    path=args.output.with_name(args.output.stem+'-'+name+'-'+kind+'.json')
                    result.to_json(path);result.to_csv(path.with_suffix('.csv'))
                    assert type(result)(json.loads(path.read_text())).to_dict()==result.to_dict()
                dead(raw['worker']['worker_pid']);dead(raw['worker']['launcher_pid'])
            assert torch.cuda.current_stream()==stream
    finally:
        assert torch.cuda.current_stream()==previous and torch.cuda.set_stream is set_stream
        assert torch.cuda.current_device()==0 and torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
        assert random.getstate()==py and np.array_equal(ns[1],np.random.get_state()[1]) and ns[2:]==np.random.get_state()[2:]
        assert torch.cuda.memory_allocated()==allocated and torch.cuda.memory_reserved()==reserved and torch.cuda.max_memory_allocated()==peak
        assert model.training and model.weight.grad is grad and torch.equal(grad,gradient) and dict(model._forward_hooks)==hooks
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in weights.items())
        assert optimizer.state[model.weight]['momentum_buffer'] is momentum and torch.equal(momentum,momentum_copy)
        assert optimizer.param_groups[0]['lr']==.01
    record.update(job=preflight.job_observation(),parent_state_preserved=True)
    assert record['job']['active_processes']==1
    for p in files:assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)]
    record['status']='PASS';save()


try:
    if args.case=='all':
        capture('tiny_fp32')
        capture('shared_fp32',factory='Shared',shape=(2,2,5,7),kwargs={'unused':True})
    capture('other_stream',factory='OtherStream',unsupported=True)
    capture('forward_failure_cleanup',factory='Shared',shape=(2,2,5,7),kwargs={'fail':True},failure=True)
finally:hook.remove();save()
print(json.dumps(dict(status='PASS',cases=len(records),gpu=environment['gpu'])))
