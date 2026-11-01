"""Standalone Windows spawn/ImageFolder P21 validation; no worker trace claims."""
import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import tempfile
import time

sys.path.insert(0,os.environ.get('HYSO_TEST_PACKAGE_ROOT',str(Path(__file__).resolve().parents[1])))
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset,Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from PIL import Image
from hyso.training import Trainer
from hyso.callbacks.callbacks import EarlyStopping
from hyso.performance import measure_training_with_loader
from trainer_replay_support import seed,equal,snapshot


class RandomTransform:
    def __call__(self,image):
        value=T.RandomHorizontalFlip()(image)
        return T.ToTensor()(value)+random.random()*.001+float(np.random.rand())*.001+torch.rand(())*.001


class BadDataset(Dataset):
    def __len__(self):return 5
    def __getitem__(self,index):
        if index>=2:raise RuntimeError('intentional worker dataset exception')
        return torch.ones(3),0


class SlowDataset(Dataset):
    def __len__(self):return 5
    def __getitem__(self,index):time.sleep(1);return torch.ones(3),0


class CrashDataset(Dataset):
    def __len__(self):return 5
    def __getitem__(self,index):
        if index>=2:os._exit(7)
        return torch.ones(3),0


def images(root):
    for i in range(5):
        directory=root/str(i%2);directory.mkdir(parents=True,exist_ok=True)
        data=np.arange(4*4*3,dtype=np.uint8).reshape(4,4,3)+i*3
        Image.fromarray(data).save(directory/f'{i}.png')


def make_loader(case,root):
    if case['image']:
        data=ImageFolder(root,transform=RandomTransform() if case['random'] else T.ToTensor())
    else:
        data=TensorDataset(torch.arange(15).reshape(5,3).float()/20,torch.tensor([0,1,0,1,0]))
    workers=case['workers']
    batching=(dict(batch_sampler=[[4],[1,3,0],[2]]) if case.get('sampler') else
              dict(batch_size=2,shuffle=case['shuffle']))
    return DataLoader(data,**batching,num_workers=workers,
        generator=torch.Generator().manual_seed(821),pin_memory=False,
        persistent_workers=case['persistent'],**(dict(multiprocessing_context='spawn',prefetch_factor=case.get('prefetch',2),timeout=30) if workers else {}))


def worker_cleanup():
    # Drop observer/loader references, then require native DataLoader ownership
    # to finish workers. No manual shutdown/terminate in the passing fixture.
    gc.collect()
    deadline=time.monotonic()+8
    while mp.active_children() and time.monotonic()<deadline:time.sleep(.05);gc.collect()
    assert not mp.active_children(),[(p.pid,p.is_alive()) for p in mp.active_children()]


def paired(case,root):
    states=[]; workers_seen=[]; result=None
    for observed in (False,True):
        seed(283);train=make_loader(case,root)
        components=('dataset','sampler','batch_sampler','collate_fn','generator','worker_init_fn')
        identities=tuple(id(getattr(train,k)) for k in components)
        epoch_pids=[]
        def callback(event,logs):
            if event=='epoch_end' and case['persistent']:
                epoch_pids.append([p.pid for p in train._iterator._workers])
        features=48 if case['image'] else 3
        model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(features,4),torch.nn.Dropout(.2),torch.nn.Linear(4,2))
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,gamma=.9)
        early=EarlyStopping(patience=1,delta=1e6,verbose=False,restore_best_weights=True) if case.get('early') else None
        trainer=Trainer(model,optimizer=optimizer,loss_fn=torch.nn.CrossEntropyLoss(),accum_steps=2,
                        scheduler=scheduler,ema_decay=.8,early_stopping=early,callbacks=(callback,))
        # Validation uses the original loader too; worker/prefetch behavior stays
        # caller-configured. It intentionally consumes another native iterator.
        val=train if case.get('validation') or early else None
        content=[];handle=model.register_forward_pre_hook(lambda m,args:content.append(args[0].detach().clone()))
        workers_before=set(p.pid for p in mp.active_children())
        try:
            if observed:result=measure_training_with_loader(trainer,train,val,epochs=3 if early else 2)
            else:trainer.fit(train,val,epochs=3 if early else 2)
        finally:handle.remove()
        states.append((snapshot(trainer),train.generator.get_state().clone(),content))
        assert tuple(id(getattr(train,k)) for k in components)==identities
        if case['persistent']:
            assert len(epoch_pids)>=2 and all(pids==epoch_pids[0] for pids in epoch_pids)
        assert all(not hasattr(trainer,k) for k in ('_loader_recorder','_timing_recorder','_stage_recorder'))
        if case['persistent']:
            live=set(p.pid for p in mp.active_children())-workers_before
            assert len(live)==2 and all(p.is_alive() for p in mp.active_children())
            workers_seen.append(sorted(live))
        del train,val
        worker_cleanup()
    equal(*states)
    loader_data=result.loader.to_dict()
    assert loader_data['provenance']['loaders']['training']['num_workers']==case['workers']
    assert all(s['batches']==3 and s['samples']==5 for s in loader_data['summaries'])
    if case.get('sampler'):
        assert [r['samples'] for r in loader_data['observations'] if r['kind']=='batch']==[1,3,1]*2
    assert result.loader.core_metrics()['status']=='complete'
    assert not torch.cuda.is_initialized()
    return dict(case=case,exact_state_rng_history_batch_content=True,persistent_worker_pids=workers_seen,
        all_workers_exited=True,p21=loader_data,
        content_sha256=hashlib.sha256(b''.join(t.contiguous().numpy().tobytes() for t in states[0][2])).hexdigest())


def failure_case(kind):
    seed(93)
    dataset=SlowDataset() if kind=='timeout' else CrashDataset() if kind=='crash' else BadDataset()
    loader=DataLoader(dataset,batch_size=2,num_workers=2,multiprocessing_context='spawn',prefetch_factor=2,
                      timeout=.1 if kind=='timeout' else 30,persistent_workers=kind=='exception_persistent')
    model=torch.nn.Linear(3,2);t=Trainer(model,optimizer=torch.optim.SGD(model.parameters(),lr=.01),loss_fn=torch.nn.CrossEntropyLoss())
    try:measure_training_with_loader(t,loader,epochs=1)
    except RuntimeError as error:
        expected=('timed out' if kind=='timeout' else 'exited unexpectedly' if kind=='crash'
                  else 'intentional worker dataset exception')
        assert expected in str(error),str(error)
    else:raise AssertionError('expected worker failure')
    assert all(not hasattr(t,k) for k in ('_loader_recorder','_timing_recorder','_stage_recorder'))
    if kind=='exception_persistent':
        assert loader._iterator is not None and all(w.is_alive() for w in loader._iterator._workers)
    del loader
    worker_cleanup()
    return dict(case=kind,expected_failure=True,all_workers_exited=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output',required=True);p.add_argument('--admit')
    args=p.parse_args()
    if args.admit:
        deadline=time.monotonic()+15
        while not Path(args.admit).exists() and time.monotonic()<deadline:time.sleep(.02)
        assert Path(args.admit).exists(),'supervisor job admission missing'
    torch.set_num_threads(1)
    cases={
        'tensor_spawn':dict(image=False,random=False,workers=2,shuffle=True,persistent=False),
        'image_deterministic':dict(image=True,random=False,workers=0,shuffle=False,persistent=False,validation=True),
        'image_random_cpu':dict(image=True,random=True,workers=0,shuffle=True,persistent=False),
        'image_spawn_random':dict(image=True,random=True,workers=2,shuffle=True,persistent=True,validation=True),
        'early_spawn':dict(image=False,random=False,workers=2,shuffle=True,persistent=True,early=True),
        'sampler_spawn':dict(image=False,random=False,workers=2,shuffle=False,persistent=False,sampler=True,prefetch=1),
        'prefetch_persistent':dict(image=False,random=False,workers=2,shuffle=True,persistent=True,prefetch=3),
    }
    with tempfile.TemporaryDirectory(prefix='hyso-p21-images-') as temp:
        root=Path(temp);images(root)
        result=failure_case(args.case) if args.case in ('timeout','exception','crash','exception_persistent') else paired(cases[args.case],root)
    import hyso.performance.dataloader as module
    result['package_source']=module.__file__
    result['process_pid']=os.getpid()
    if os.environ.get('HYSO_TEST_PACKAGE_ROOT'):assert Path(module.__file__).resolve().is_relative_to(Path(os.environ['HYSO_TEST_PACKAGE_ROOT']).resolve())
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(args.case,'PASS')


if __name__=='__main__':main()
