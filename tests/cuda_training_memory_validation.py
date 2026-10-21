"""Test-only bounded native CUDA P23 evidence. No production isolation claim."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import weakref
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,os.environ.get('HYSO_TEST_PACKAGE_ROOT',str(Path(__file__).resolve().parents[1])))
import torch
from torch.utils.data import DataLoader,TensorDataset
from hyso.training import Trainer
from hyso.performance import measure_training_memory,TrainingMemoryProtocol,TrainingTimingProtocol
from hyso.performance import training_memory as module
from trainer_replay_support import seed,snapshot,equal
import cuda_training_timing_validation as baseline


def paired(opt_type,amp,nondefault,overflow):
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
                scaler=t.scaler.state_dict();scaler.update(scale=128.,growth_interval=1000);t.scaler.load_state_dict(scaler)
            calls=[]
            def poison(g):
                calls.append(True)
                return torch.full_like(g,float('inf')) if len(calls)==1 else g
            handle=model[0].weight.register_hook(poison) if overflow else None
            dataset=TensorDataset(torch.linspace(-1,1,15).reshape(5,3),torch.tensor([0,1,0,1,0]))
            train=DataLoader(dataset,batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(83))
            before=module._cuda(torch.device('cuda:0'))
            def forbidden(*a,**kw):raise AssertionError('reset/cache mutation')
            try:
                with patch.object(torch.cuda,'reset_peak_memory_stats',forbidden),patch.object(torch.cuda,'empty_cache',forbidden):
                    if observed:
                        result=measure_training_memory(t,train,train,epochs=2,protocol=TrainingMemoryProtocol(device='cuda:0'),
                                                       timing_protocol=TrainingTimingProtocol(event_buffer=2))
                    else:t.fit(train,train,epochs=2)
            finally:
                if handle:handle.remove()
            stream.synchronize();states.append((snapshot(t),train.generator.get_state()))
            assert not hasattr(t,'_stage_recorder') and not opt._optimizer_step_pre_hooks and not opt._optimizer_step_post_hooks
    equal(*states)
    d=result.to_dict();p19=d['source_p19'];rows=d['observations'];cuda=d['cuda']
    assert p19['counters']['after']['applied']==4-int(overflow)
    assert d['instrumentation']['stream_id']==int(stream.cuda_stream)
    assert cuda['baseline']==before and cuda['reset_performed'] is False and cuda['window_peak']['value'] is None
    assert all(r['cuda']['allocated_bytes']<=r['cuda']['reserved_bytes'] for r in rows)
    attempts=[r for r in rows if r['boundary']=='post_attempt']
    assert sum(r['fields']['skipped'] for r in attempts)==int(overflow)
    if opt_type is not torch.optim.SGD:
        assert rows[0]['logical']['summary']['optimizer']['logical_bytes']==0
        if overflow:assert attempts[0]['logical']['summary']['optimizer']['logical_bytes']==0
        assert attempts[-1]['logical']['summary']['optimizer']['logical_bytes']>0
    return dict(id=f'{opt_type.__name__}-amp{amp}-stream{nondefault}-overflow{overflow}',p23=d,exact_state_rng=True)


def lifetime():
    observations=[];enabled=gc.isenabled();gc.disable()
    try:
        for index in range(2):
            model=torch.nn.Linear(3,2).cuda();ref=weakref.ref(model);tensor=weakref.ref(model.weight)
            opt=torch.optim.AdamW(model.parameters(),lr=.01)
            t=Trainer(model,optimizer=opt,device='cuda',loss_fn=torch.nn.CrossEntropyLoss())
            train=DataLoader(TensorDataset(torch.ones(5,3),torch.tensor([0,1,0,1,0])),batch_size=2)
            result=measure_training_memory(t,train,epochs=1,protocol=TrainingMemoryProtocol(device='cuda:0'))
            retained=[model] if index==0 else []
            del t,model,opt
            if retained:assert ref() is not None and tensor() is not None
            retained.clear()
            assert ref() is None and tensor() is None
            torch.cuda.synchronize()
            observations.append(dict(run=index,counters=module._cuda(torch.device('cuda:0')),result_retained=True))
            assert result.to_dict()['status']=='partial'
    finally:
        if enabled:gc.enable()
    return dict(id='gc_disabled_weakref_positive_control_two_run_observation',observations=observations,leak_verdict=None)


def scheduled_gc_reset_diagnostic():
    # Test-only explicit reset in an owned fresh test process, never production.
    gc.collect();torch.cuda.synchronize();before=module._cuda(torch.device('cuda:0'))
    torch.cuda.reset_peak_memory_stats();after=module._cuda(torch.device('cuda:0'))
    assert after['lifetime_peak_allocated_bytes']==after['allocated_bytes']
    assert after['lifetime_peak_reserved_bytes']==after['reserved_bytes']
    measured=paired(torch.optim.AdamW,False,False,False)
    return dict(id='test_only_scheduled_gc_baseline_reset_diagnostic',previous=before,after_reset=after,
                explicit_test_only_reset=True,production_reset=False,measured=measured)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--admit');parser.add_argument('--case',default='matrix');args=parser.parse_args()
    if args.admit:
        deadline=time.monotonic()+15
        while not Path(args.admit).exists() and time.monotonic()<deadline:time.sleep(.02)
        assert Path(args.admit).exists()
    assert torch.cuda.is_available()
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.15);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[];suffix=[]
    if args.case=='timeout':
        class Slow(torch.nn.Linear):
            def forward(self,x):
                Path(args.output).with_suffix('.ready').write_text(str(os.getpid()))
                time.sleep(120)
                return super().forward(x)
        model=Slow(3,2).cuda()
        t=Trainer(model,optimizer=torch.optim.SGD(model.parameters(),lr=.01),device='cuda',loss_fn=torch.nn.CrossEntropyLoss())
        train=DataLoader(TensorDataset(torch.ones(2,3),torch.tensor([0,1])),batch_size=2)
        measure_training_memory(t,train,epochs=1,protocol=TrainingMemoryProtocol(device='cuda:0'))
        raise AssertionError('supervisor must terminate this test')
    elif args.case=='oom':
        # A typed, controlled allocator-fraction OOM, then this worker exits.
        total=torch.cuda.get_device_properties(0).total_memory
        class OverBudget(torch.nn.Linear):
            def forward(self,x):
                torch.empty(int(total*.20),dtype=torch.uint8,device='cuda')
                return super().forward(x)
        model=OverBudget(3,2).cuda()
        t=Trainer(model,optimizer=torch.optim.SGD(model.parameters(),lr=.01),device='cuda',loss_fn=torch.nn.CrossEntropyLoss())
        train=DataLoader(TensorDataset(torch.ones(2,3),torch.tensor([0,1])),batch_size=2)
        try:measure_training_memory(t,train,epochs=1,protocol=TrainingMemoryProtocol(device='cuda:0'))
        except torch.cuda.OutOfMemoryError:
            import threading
            assert not module._RUN_LOCK.locked() and not hasattr(t,'_timing_recorder') and not hasattr(t,'_stage_recorder')
            assert not [thread for thread in threading.enumerate() if thread.name=='hyso-training-rss']
            rows.append(dict(id='controlled_fraction_training_oom',typed_oom=True,no_retry=True,recorder_cleanup=True))
        else:raise AssertionError('fraction guard did not reject over-budget allocation')
    elif args.case=='child':
        rows.append(paired(torch.optim.AdamW,False,False,False))
    elif args.case=='parent':
        owner=torch.ones(1024,device='cuda');torch.cuda.synchronize();before=dict(torch.cuda.memory_stats());rng=torch.cuda.get_rng_state().clone()
        child=Path(args.output).with_name(Path(args.output).stem+'-child.json')
        command=[sys.executable,str(Path(__file__).with_name('p21_native_supervisor.py')),
            '--script',str(Path(__file__).resolve()),'--case','child','--output',str(child),'--timeout','60','--cuda']
        if os.environ.get('HYSO_TEST_PACKAGE_ROOT'):command+=['--package-root',os.environ['HYSO_TEST_PACKAGE_ROOT']]
        run=subprocess.run(command,capture_output=True,text=True,timeout=75)
        assert run.returncode==0,run.stdout+run.stderr
        assert before==dict(torch.cuda.memory_stats()) and torch.equal(rng,torch.cuda.get_rng_state())
        assert torch.equal(owner,torch.ones_like(owner))
        rows.append(dict(id='test_only_fresh_process_parent_preservation',child=str(child),parent_allocator_and_rng_equal=True))
    else:
        for opt in (torch.optim.SGD,torch.optim.Adam,torch.optim.AdamW):
            for amp in (False,True):
                for stream in (False,True):
                    for overflow in ((False,True) if amp else (False,)):
                        rows.append(paired(opt,amp,stream,overflow))
        def measured(*a,**kw):
            kw['timing_protocol']=TrainingTimingProtocol(event_buffer=2)
            kw['protocol']=TrainingMemoryProtocol(device='cuda:0')
            result=measure_training_memory(*a,**kw);suffix.append(result.to_dict())
            return module.TrainingPerformanceResult(result.to_dict()['source_p19'])
        with patch.object(baseline,'measure_training',measured):
            for amp in (False,True):rows.append(dict(id=baseline.checkpoint_case(amp)['id']))
        model=torch.nn.Linear(3,2).cuda()
        t=Trainer(model,optimizer=torch.optim.SGD(model.parameters(),lr=.01),device='cuda',loss_fn=torch.nn.CrossEntropyLoss())
        def fail(*a):raise RuntimeError('injected training failure')
        handle=t.model.register_forward_pre_hook(fail)
        try:
            try:measure_training_memory(t,DataLoader(TensorDataset(torch.ones(2,3),torch.tensor([0,1])),batch_size=2),epochs=1,protocol=TrainingMemoryProtocol(device='cuda:0'))
            except RuntimeError as e:assert 'injected training failure' in str(e)
            else:raise AssertionError('expected Python training failure')
        finally:handle.remove()
        assert not module._RUN_LOCK.locked() and not hasattr(t,'_timing_recorder')
        rows.append(dict(id='python_training_failure_cleanup'))
        rows.append(lifetime())
        rows.append(scheduled_gc_reset_diagnostic())
    if os.environ.get('HYSO_TEST_PACKAGE_ROOT'):
        assert Path(module.__file__).resolve().is_relative_to(Path(os.environ['HYSO_TEST_PACKAGE_ROOT']).resolve())
    Path(args.output).write_text(json.dumps(dict(status='passed',scenarios=rows,resume=suffix,package_source=module.__file__,
        process_pid=os.getpid(),device=torch.cuda.get_device_name(0),torch=str(torch.__version__),allocator_fraction=.15),indent=2),encoding='utf-8')
    print(len(rows),'P23 CUDA cases passed')


if __name__=='__main__':main()
