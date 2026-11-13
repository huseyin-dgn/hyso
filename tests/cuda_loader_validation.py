"""Native deterministic P21 DataLoader / selected-stream AMP conformance."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,os.environ.get('HYSO_TEST_PACKAGE_ROOT',str(Path(__file__).resolve().parents[1])))
import torch
from torch.utils.data import DataLoader,TensorDataset
from hyso.training import Trainer
from hyso.performance import measure_training_with_loader,TrainingTimingProtocol
import hyso.performance.dataloader as module
from trainer_replay_support import seed,snapshot,equal
import cuda_training_timing_validation as baseline


def paired(opt_type,amp,nondefault,overflow,pinned):
    stream=torch.cuda.Stream() if nondefault else torch.cuda.default_stream()
    states=[]
    for observed in (False,True):
        seed(64)
        with torch.cuda.stream(stream):
            model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.2),torch.nn.Linear(4,2)).cuda()
            opt=opt_type(model.parameters(),lr=.01)
            scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,.02,total_steps=4,cycle_momentum=False)
            t=Trainer(model,optimizer=opt,loss_fn=torch.nn.CrossEntropyLoss(),device='cuda',amp=amp,accum_steps=2,scheduler=scheduler,ema_decay=.8,grad_clip=.1)
            if amp:
                scaler=t.scaler.state_dict();scaler.update(scale=128.,growth_interval=1000);t.scaler.load_state_dict(scaler)
            calls=[]
            def poison(g):
                calls.append(True)
                return torch.full_like(g,float('inf')) if len(calls)==1 else g
            handle=model[0].weight.register_hook(poison) if overflow else None
            dataset=TensorDataset(torch.linspace(-1,1,15).reshape(5,3),torch.tensor([0,1,0,1,0]))
            train=DataLoader(dataset,batch_size=2,shuffle=True,pin_memory=pinned,generator=torch.Generator().manual_seed(83))
            try:
                if observed:result=measure_training_with_loader(t,train,train,epochs=2,protocol=TrainingTimingProtocol(event_buffer=2))
                else:t.fit(train,train,epochs=2)
            finally:
                if handle:handle.remove()
            stream.synchronize();states.append((snapshot(t),train.generator.get_state()))
            assert not hasattr(t,'_loader_recorder') and not opt._optimizer_step_pre_hooks and not opt._optimizer_step_post_hooks
    equal(*states)
    d=result.loader.to_dict();p19=result.timing.to_dict();p22=result.throughput.to_dict()
    assert d['h2d_transfer']['value'] is None and d['h2d_transfer']['status']=='unsupported'
    assert p22['segment']['completed']['samples']==10 and p22['segment']['completed']['applied']==4-int(overflow)
    assert p19['instrumentation']['stream_id']==int(stream.cuda_stream)
    assert all(x['samples']==5 for x in d['summaries'])
    assert all(r['delivery_ns']<=r['end_ns'] for r in d['observations'] if r['kind']=='batch')
    return dict(id=f'{opt_type.__name__}-amp{amp}-stream{nondefault}-overflow{overflow}-pinned{pinned}',p21=d,exact_state_rng=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--admit');args=p.parse_args()
    if args.admit:
        deadline=time.monotonic()+15
        while not Path(args.admit).exists() and time.monotonic()<deadline:time.sleep(.02)
        assert Path(args.admit).exists()
    assert torch.cuda.is_available()
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.15);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[]
    for opt in (torch.optim.SGD,torch.optim.Adam,torch.optim.AdamW):
        for amp in (False,True):
            for stream in (False,True):
                for overflow in ((False,True) if amp else (False,)):
                    rows.append(paired(opt,amp,stream,overflow,pinned=stream))
    suffix=[]
    def measured(*a,**kw):
        kw['protocol']=TrainingTimingProtocol(event_buffer=2)
        result=measure_training_with_loader(*a,**kw);suffix.append(result.loader.to_dict());return result.timing
    with patch.object(baseline,'measure_training',measured):
        for amp in (False,True):rows.append(dict(id=baseline.checkpoint_case(amp)['id']))
    if os.environ.get('HYSO_TEST_PACKAGE_ROOT'):assert Path(module.__file__).resolve().is_relative_to(Path(os.environ['HYSO_TEST_PACKAGE_ROOT']).resolve())
    Path(args.output).write_text(json.dumps(dict(status='passed',scenarios=rows,resume=suffix,package_source=module.__file__,process_pid=os.getpid()),indent=2),encoding='utf-8')
    print(len(rows),'P21 CUDA cases passed')


if __name__=='__main__':main()
