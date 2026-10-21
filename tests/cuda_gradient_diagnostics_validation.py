"""P25 real CUDA/scaler evidence, run only under the bounded Job supervisor."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import weakref
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,os.environ.get('HYSO_TEST_PACKAGE_ROOT',str(Path(__file__).resolve().parents[1])))
import torch
from torch.utils.data import DataLoader,TensorDataset
from hyso.training import Trainer
from hyso.performance import (measure_gradient_diagnostics,TrainingTimingProtocol,GradientDiagnosticsProtocol,
    TrainingPerformanceResult,analyze_training_throughput)
from hyso.performance import gradients as module
from trainer_replay_support import seed,snapshot,equal
import cuda_training_timing_validation as baseline


def paired(opt_type,amp,nondefault,overflow,*,nan=False,growth=False):
    stream=torch.cuda.Stream() if nondefault else torch.cuda.default_stream()
    states=[]
    for observed in (False,True):
        seed(64)
        with torch.cuda.stream(stream):
            model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.2),torch.nn.Linear(4,2)).cuda()
            opt=opt_type(model.parameters(),lr=.01)
            scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,.02,total_steps=4,cycle_momentum=False)
            t=Trainer(model,optimizer=opt,loss_fn=torch.nn.CrossEntropyLoss(),device='cuda',amp=amp,
                      accum_steps=2,scheduler=scheduler,ema_decay=.8,grad_clip=.1)
            if amp:
                scaler=t.scaler.state_dict();scaler.update(scale=128.,growth_interval=1 if growth else 1000)
                t.scaler.load_state_dict(scaler)
            calls=[]
            def poison(g):
                calls.append(True)
                return torch.full_like(g,float('nan') if nan else float('inf')) if len(calls)==1 else g
            handle=model[0].weight.register_hook(poison) if overflow else None
            dataset=TensorDataset(torch.linspace(-1,1,15).reshape(5,3),torch.tensor([0,1,0,1,0]))
            train=DataLoader(dataset,batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(83))
            try:
                if observed:
                    result=measure_gradient_diagnostics(t,train,train,epochs=2,
                        timing_protocol=TrainingTimingProtocol(event_buffer=2),protocol=GradientDiagnosticsProtocol(chunk_elements=3))
                else:t.fit(train,train,epochs=2)
            finally:
                if handle:handle.remove()
            stream.synchronize();states.append((snapshot(t),train.generator.get_state()))
            assert not hasattr(t,'_stage_recorder') and not opt._optimizer_step_pre_hooks and not opt._optimizer_step_post_hooks
    equal(*states)
    d=result.to_dict();rows=d['observations'];p19=d['source_p19']
    p22=analyze_training_throughput(TrainingPerformanceResult(p19)).to_dict()
    assert d['status']=='measured' and d['summary']['applied']==4-int(overflow)
    assert d['summary']['skipped']==int(overflow) and p19['counters']['after']['ema_updates']==4-int(overflow)
    assert p22['segment']['completed']['applied']==4-int(overflow)
    assert d['instrumentation']['stream_id']==int(stream.cuda_stream) and d['instrumentation']['cuda_host_scalar_sync']
    assert [r['samples'] for r in rows]==[4,1,4,1]
    assert all(r['optimizer_executed']==r['applied'] for r in rows)
    if overflow:
        first=rows[0]
        assert first['skipped'] and first['preclip']['summary']['finite'] is False
        assert first['preclip']['summary']['global_l2']['value'] is None
        assert first['preclip']['summary']['nan_tensor_count' if nan else 'inf_tensor_count']==1
        assert first['postclip'] is None and first['clipping']['status']=='not_executed'
        assert first['scale_after']<first['scale_before'] and first['overflow_evidence']=='native_skip_with_nonfinite_gradient'
        assert d['summary']['recovered_after_skip_attempt_ids']==[2] and all(r['applied'] for r in rows[1:])
    if growth:assert all(r['scale_after']==r['scale_before']*2 and r['scale_change']=='increased' for r in rows)
    for row in rows[int(overflow):]:
        assert row['preclip']['summary']['finite'] and row['postclip']['summary']['global_l2']['value']<=.100001
    return dict(id=f'{opt_type.__name__}-amp{amp}-stream{nondefault}-overflow{overflow}-nan{nan}-growth{growth}',p25=d,exact_state_rng=True)


def reductions():
    values=[]
    for dtype in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
        d=module._tensor_gradient(torch.tensor([3.,4.],device='cuda',dtype=dtype),1)
        assert d['global_l2']==5 and d['scalar_reads']==8;values.append(d)
    large=module._tensor_gradient(torch.tensor([1e200,1e200],device='cuda',dtype=torch.float64),1)
    assert abs(large['global_l2']/1e200-2**.5)<1e-14
    sparse=torch.sparse_coo_tensor(torch.tensor([[0,0]],device='cuda'),torch.tensor([3.,4.],device='cuda'),(3,))
    def forbidden(*a,**kw):raise AssertionError('implicit sparse materialization')
    with patch.object(torch.Tensor,'coalesce',forbidden),patch.object(torch.Tensor,'to_dense',forbidden):
        unsupported=module._tensor_gradient(sparse,1)
    assert unsupported['status']=='unsupported' and unsupported['global_l2'] is None
    return dict(id='native_bounded_fp64_reduction_and_sparse_coverage',values=values,large=large,sparse=unsupported)


def lifetime():
    enabled=gc.isenabled();gc.disable()
    try:
        model=torch.nn.Linear(3,2).cuda();ref=weakref.ref(model);tensor=weakref.ref(model.weight)
        opt=torch.optim.AdamW(model.parameters(),lr=.01)
        t=Trainer(model,optimizer=opt,device='cuda',loss_fn=torch.nn.CrossEntropyLoss())
        train=DataLoader(TensorDataset(torch.ones(3,3),torch.tensor([0,1,0])),batch_size=2)
        r=measure_gradient_diagnostics(t,train,epochs=1)
        del t,model,opt
        assert ref() is None and tensor() is None and r.to_dict()['status']=='measured'
    finally:
        if enabled:gc.enable()
    return dict(id='gc_disabled_result_retains_no_model_tensor')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--admit');a=p.parse_args()
    if a.admit:
        deadline=time.monotonic()+15
        while not Path(a.admit).exists() and time.monotonic()<deadline:time.sleep(.02)
        assert Path(a.admit).exists()
    assert torch.cuda.is_available()
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.15);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[];suffix=[]
    for opt in (torch.optim.SGD,torch.optim.Adam,torch.optim.AdamW):
        for amp in (False,True):
            for stream in (False,True):
                for overflow in ((False,True) if amp else (False,)):
                    rows.append(paired(opt,amp,stream,overflow))
    def measured(*args,**kwargs):
        kwargs['timing_protocol']=TrainingTimingProtocol(event_buffer=2)
        result=measure_gradient_diagnostics(*args,**kwargs);suffix.append(result.to_dict())
        return TrainingPerformanceResult(result.to_dict()['source_p19'])
    with patch.object(baseline,'measure_training',measured):
        for amp in (False,True):rows.append(dict(id=baseline.checkpoint_case(amp)['id']))
    model=torch.nn.Linear(3,2).cuda()
    t=Trainer(model,optimizer=torch.optim.SGD(model.parameters(),lr=.01),device='cuda',loss_fn=torch.nn.CrossEntropyLoss())
    native=module._tensor_gradient
    def fail(*args,**kwargs):raise RuntimeError('injected diagnostic query failure')
    try:
        with patch.object(module,'_tensor_gradient',fail):
            measure_gradient_diagnostics(t,[(torch.ones(2,3),torch.tensor([0,1]))],epochs=1)
    except RuntimeError as error:assert 'injected diagnostic query failure' in str(error)
    else:raise AssertionError('expected diagnostic failure')
    assert not hasattr(t,'_timing_recorder') and not t.optimizer._optimizer_step_pre_hooks and not t.optimizer._optimizer_step_post_hooks
    assert all(p.grad is None for p in t.model.parameters())
    rows.append(dict(id='diagnostic_failure_hook_gradient_cleanup'))
    rows.append(paired(torch.optim.AdamW,True,False,False,growth=True))
    rows.append(paired(torch.optim.AdamW,True,False,True,nan=True))
    rows.append(reductions());rows.append(lifetime())
    assert len(rows)==25 and len({r['id'] for r in rows})==25
    if os.environ.get('HYSO_TEST_PACKAGE_ROOT'):
        assert Path(module.__file__).resolve().is_relative_to(Path(os.environ['HYSO_TEST_PACKAGE_ROOT']).resolve())
    Path(a.output).write_text(json.dumps(dict(status='passed',scenarios=rows,resume=suffix,package_source=module.__file__,
        process_pid=os.getpid(),device=torch.cuda.get_device_name(),torch=str(torch.__version__),allocator_fraction=.15),indent=2),encoding='utf-8')
    print('25 P25 native CUDA scenarios passed')


if __name__=='__main__':main()
