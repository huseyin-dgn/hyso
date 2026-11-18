"""Real-device B4 gate: separate process, no fallback/skip or numeric speed target."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import traceback

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--installed-target',type=Path)
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(args.installed_target.resolve() if args.installed_target else ROOT))
import numpy as np
import torch
from hyso.performance import SweepProtocol, run_sweep
from hyso.benchmark.runner import StateMutationError
import hyso.performance.sweeps as module

assert torch.cuda.is_available(), 'Native CUDA required'
torch.set_num_threads(1)


def model():
    return torch.nn.Sequential(torch.nn.Conv2d(1,3,3,padding=1),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten())


def inputs(shape,dtype,device):
    random.random();np.random.rand();torch.rand(1);torch.rand(1,device=device)
    return torch.randn(shape,dtype=dtype,device=device)


def exercise(axis,precision,nondefault=False):
    source=model(); source[0].eval();source[0].weight.grad=torch.ones_like(source[0].weight)
    saved=copy.deepcopy(source.state_dict());modes=[m.training for m in source.modules()]
    grad=source[0].weight.grad; grad_values=grad.clone()
    previous=torch.cuda.current_stream();stream=torch.cuda.Stream() if nondefault else previous
    py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
    peak=torch.cuda.max_memory_allocated()
    with torch.cuda.stream(stream):
        result=run_sweep(source,inputs,axis=axis,points=[1,2,4,8] if axis=='batch' else [(5,7),(8,8),(9,13)],
            protocol=SweepProtocol(device='cuda:0',precision=precision,warmup_iterations=2,
                measure_iterations=4,repeats=2,sustained_batches=5))
        assert torch.cuda.current_stream()==stream
    assert torch.cuda.current_stream()==previous and torch.cuda.current_device()==0
    assert all(torch.equal(v,source.state_dict()[k]) for k,v in saved.items())
    assert modes==[m.training for m in source.modules()] and source[0].weight.grad is grad and torch.equal(grad,grad_values)
    assert random.getstate()==py and np.array_equal(np.random.get_state()[1],npstate[1])
    assert torch.equal(torch.get_rng_state(),cpu) and torch.equal(torch.cuda.get_rng_state(),gpu)
    assert torch.cuda.max_memory_allocated()>=peak
    data=result.to_dict();assert data['actual']['measured']==(4 if axis=='batch' else 3)
    for point in data['raw_evidence']['points']:
        r=point['result'];assert r['actual']['samples']==8
        assert r['provenance']['input']['device']=='cuda:0'
        assert r['provenance']['precision']['output_dtype']==('torch.float16' if precision=='amp_fp16' else 'torch.bfloat16' if precision=='bf16' else 'torch.float32')
        assert r['measurements']['latency_mean_ms']['value']>0
        raw=r['raw_evidence']
        if raw['benchmark'] is not None:
            assert raw['benchmark']['timing']['timing_method']=='cuda_event_elapsed'
            assert raw['throughput']['actual']['completed_images']==10*point['shape'][0]
        else:
            assert raw['special']['clock']=='cuda_event_elapsed'
            assert all(w['terminal_drains']==1 and w['pending_batches']==0 for w in raw['special']['windows'])
    return data


def precision_availability():
    source=model()
    data=run_sweep(source,inputs,axis='precision',points=['bf16','int8_dynamic','fp64'],
        protocol=SweepProtocol(device='cuda:0',measure_iterations=2,warmup_iterations=1)).to_dict()
    rows=data['raw_evidence']['points']
    available=torch.cuda.get_device_capability()[0]>=8 and torch.cuda.is_bf16_supported()
    assert rows[0]['status']==('measured' if available else 'unsupported')
    assert rows[1]['status']==rows[2]['status']=='unsupported'
    return dict(bf16_hardware_available=available,result=data)


def real_oom():
    def source(s,d,v):
        if s[0]==2: return torch.empty((1<<40,),dtype=d,device=v)
        return inputs(s,d,v)
    data=run_sweep(model(),source,axis='batch',points=[1,2,4],protocol=SweepProtocol(device='cuda:0',measure_iterations=2)).to_dict()
    rows=data['raw_evidence']['points'];assert [r['status'] for r in rows]==['measured','failed','not_run']
    assert rows[1]['reason'].startswith('out_of_memory:')
    torch.cuda.synchronize();return data


def shape_failure():
    m=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(35,2))
    data=run_sweep(m,inputs,axis='resolution',points=[(4,4),(5,7)],protocol=SweepProtocol(device='cuda:0',measure_iterations=2)).to_dict()
    assert [r['status'] for r in data['raw_evidence']['points']]==['failed','measured'];return data


def stream_failure():
    previous=torch.cuda.current_stream();other=torch.cuda.Stream()
    def bad(s,d,v):torch.cuda.set_stream(other);return inputs(s,d,v)
    try:run_sweep(model(),bad,axis='batch',points=[1],protocol=SweepProtocol(device='cuda:0'))
    except StateMutationError:pass
    else:raise AssertionError('stream mutation accepted')
    assert torch.cuda.current_stream()==previous
    return dict(restored=True)


cases=[]
for precision in ('fp32','amp_fp16'):
    for axis in ('batch','resolution'):
        cases.append((axis+'_'+precision,lambda a=axis,p=precision:exercise(a,p)))
cases += [('nondefault_fp32',lambda:exercise('batch','fp32',True)),
          ('nondefault_amp_fp16',lambda:exercise('batch','amp_fp16',True)),
          ('bf16_capability',precision_availability),
          ('bf16_nondefault',lambda:exercise('resolution','bf16',True) if torch.cuda.is_bf16_supported() else precision_availability()),
          ('actual_cuda_oom_stop',real_oom),('invalid_shape_continuation',shape_failure),('stream_failure_cleanup',stream_failure)]
results=[]
for name,operation in cases:
    try:results.append(dict(id=name,status='PASS',evidence=operation()))
    except BaseException:results.append(dict(id=name,status='FAIL',traceback=traceback.format_exc()))
    print(name,results[-1]['status'],flush=True)
report=dict(python=sys.version,torch=str(torch.__version__),cuda_build=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
    source=module.__file__,cases=results)
if args.installed_target:assert Path(module.__file__).resolve().is_relative_to(args.installed_target.resolve())
args.output.write_text(json.dumps(report,allow_nan=False,indent=2),encoding='utf-8')
raise SystemExit(int(any(r['status']!='PASS' for r in results)))
