"""Standalone P03/P04/P05 native CUDA gate. Missing hardware is failure."""
import argparse
import json
import random
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import traceback

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--installed-target',type=Path)
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(args.installed_target.resolve() if args.installed_target else ROOT))
import torch
import numpy as np
from hyso.performance import (PipelineSample,PipelineStage,PipelineProtocol,measure_pipeline,analyze_stages,
    ColdStartRecipe,ColdStartProtocol,measure_cold_start,measure_footprint)
from hyso.benchmark.runner import StateMutationError

if not torch.cuda.is_available(): raise RuntimeError('Native CUDA required; no skip or fallback')
torch.set_num_threads(1)


def check_pipeline(precision,nondefault):
    model=torch.nn.Conv2d(1,2,3,padding=1).to('cuda:0')
    for p in model.parameters(): p.grad=torch.full_like(p,.25)
    values=[(p,p.data_ptr(),p._version,p.cpu().clone(),p.grad,p.grad.cpu().clone()) for p in model.parameters()]
    modes=[(m,m.training) for m in model.modules()]
    before_cpu=torch.get_rng_state(); before_cuda=torch.cuda.get_rng_state(0)
    before_python=random.getstate(); before_numpy=np.random.get_state()
    previous=torch.cuda.current_stream(0); stream=torch.cuda.Stream() if nondefault else previous
    torch.cuda.synchronize(0)
    counter=iter(range(20)); delivered=[]; record_streams=[]; complete=[]
    def source():
        index=next(counter); return PipelineSample(str(index),torch.full((2,1,7,9),float(index)))
    def transfer(frame): return PipelineSample(frame.sample_id,frame.payload.to('cuda:0',non_blocking=True))
    def post(frame):
        random.random(); np.random.random(); torch.rand(1); torch.rand(1,device='cuda:0')
        return PipelineSample(frame.sample_id,frame.payload.mean((2,3),keepdim=True))
    def deliver(frame):
        # Terminal completion must already be reached before delivery starts.
        complete.append(torch.cuda.current_stream().query()); delivered.append(frame.sample_id)
        return PipelineSample(frame.sample_id,frame.payload.cpu())
    original=torch.cuda.Event.record
    def record(event,selected=None):
        record_streams.append(selected.cuda_stream if selected is not None else torch.cuda.current_stream().cuda_stream)
        return original(event,selected)
    with torch.cuda.stream(stream),patch.object(torch.cuda.Event,'record',record), \
            patch.object(torch.cuda,'reset_peak_memory_stats',side_effect=AssertionError('no allocator reset')):
        result=measure_pipeline(model,source,transfer=PipelineStage('transfer',transfer,device='cuda'),
            postprocess=PipelineStage('postprocess',post,device='cuda'),deliver=PipelineStage('deliver',deliver),
            protocol=PipelineProtocol(device='cuda:0',precision=precision,samples=3,repeats=2,warmup_samples=1,queue_depth=2))
        assert torch.cuda.current_stream(0)==stream
    assert torch.cuda.current_stream(0)==previous
    assert torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_cuda,torch.cuda.get_rng_state(0))
    after_numpy=np.random.get_state()
    assert random.getstate()==before_python
    assert before_numpy[0]==after_numpy[0] and np.array_equal(before_numpy[1],after_numpy[1]) and before_numpy[2:]==after_numpy[2:]
    for p,pointer,version,value,gradient,grad_value in values:
        assert p.data_ptr()==pointer and p._version==version and torch.equal(p.cpu(),value)
        assert p.grad is gradient and torch.equal(gradient.cpu(),grad_value)
    assert all(m.training==flag for m,flag in modes)
    assert all(complete) and len(delivered)==8 and set(record_streams)=={stream.cuda_stream}
    data=result.to_dict(); stages=analyze_stages(result).to_dict()
    assert data['actual']['completed']==6 and data['actual']['forward_count']==8 and data['actual']['pending']==0
    for row in data['raw_evidence']['samples']:
        assert row['postprocess_completion_ns']<=row['delivery_completion_ns']
        for stage in row['stages']:
            if stage['device']=='cuda':
                assert stage['device_event_ms']>0 and stage['sync_policy']=='end_event_sync'
                assert stage['stream_id']==stream.cuda_stream
    assert stages['measurements']['inference']['device_mean_ms']>0
    return dict(actual=data['actual'],samples=data['raw_evidence']['samples'],measurements=stages['measurements'])


def cold_case(precision):
    cpu=torch.get_rng_state(); cuda=torch.cuda.get_rng_state(0); stream=torch.cuda.current_stream(0)
    with tempfile.TemporaryDirectory(prefix='hyso-native-cold-') as root:
        model=torch.nn.Conv2d(1,2,3,padding=1)
        path=Path(root)/'state.pt'
        info=measure_footprint(model,serialize_to=path,verify_roundtrip=True).to_dict()['raw_evidence']['artifact']
        # Model construction consumed CPU RNG outside the P05 boundary.
        cpu=torch.get_rng_state()
        data=measure_cold_start(ColdStartRecipe('torch.nn:Conv2d',model_kwargs=dict(in_channels=1,out_channels=2,kernel_size=3,padding=1),
            device='cuda:0',precision=precision,artifact_path=path,artifact_sha256=info['sha256']),
            protocol=ColdStartProtocol(trials=1,warm_samples=2,warmup_iterations=1)).to_dict()
    assert data['actual']['measured']==1,data
    row=data['raw_evidence']['trials'][0]
    assert not row['provenance']['cuda_initialized_after_import'] and not row['provenance']['cuda_initialized_after_construction']
    assert row['provenance']['cuda_initialized'] and row['provenance']['device']=='cuda:0'
    assert row['provenance']['forward_count']==4 and row['phases']['runtime_prepare']['status']=='measured'
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state(0))
    assert torch.cuda.current_stream(0)==stream and row['process_terminated']
    return data['raw_evidence']['trials'][0]


def wrong_stream():
    model=torch.nn.Identity().cuda(); other=torch.cuda.Stream(); previous=torch.cuda.current_stream()
    counter=iter(range(10))
    def source(): return PipelineSample(str(next(counter)),torch.ones(1,1,3,3))
    def bad(frame): torch.cuda.set_stream(other); return PipelineSample(frame.sample_id,frame.payload.cuda())
    try:
        measure_pipeline(model,source,transfer=PipelineStage('transfer',bad,device='cuda'),protocol=PipelineProtocol(device='cuda:0'))
    except StateMutationError: pass
    else: raise AssertionError('expected stream-ownership failure')
    assert torch.cuda.current_stream()==previous and model.training
    return dict(stream_cleanup=True)


def unsupported_device():
    data=measure_cold_start(ColdStartRecipe('torch.nn:Identity',device='cuda:999'),protocol=ColdStartProtocol(trials=1)).to_dict()
    assert data['actual']['unsupported']==1 and data['actual']['measured']==0
    return data['raw_evidence']['trials'][0]


cases=[]
for precision in ('fp32','amp_fp16'):
    for nondefault in (False,True):
        cases.append((precision+('_nondefault' if nondefault else '_default'),lambda p=precision,s=nondefault:check_pipeline(p,s)))
    cases.append(('cold_'+precision,lambda p=precision:cold_case(p)))
cases += [('stream_failure_cleanup',wrong_stream),('cold_unsupported_device',unsupported_device)]
results=[]
for name,operation in cases:
    try: results.append(dict(id=name,status='PASS',evidence=operation()))
    except BaseException: results.append(dict(id=name,status='FAIL',traceback=traceback.format_exc()))
    print(name,results[-1]['status'],flush=True)
import hyso.performance.pipeline as module
report=dict(python=sys.version,torch=str(torch.__version__),cuda_build=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
            source=module.__file__,cases=results)
if args.installed_target: assert Path(module.__file__).resolve().is_relative_to(args.installed_target.resolve())
args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
raise SystemExit(int(any(row['status']!='PASS' for row in results)))
