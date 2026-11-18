"""Trusted, local, construction-only fixtures for isolated training tests."""
import atexit
import ctypes as c
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import weakref

import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from hyso.training import Trainer


def state_digest(trainer):
    from trainer_replay_support import snapshot
    state=snapshot(trainer);h=hashlib.sha256()
    def visit(value):
        if isinstance(value,torch.Tensor):
            h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value,np.ndarray):h.update(str(value.dtype).encode());h.update(value.tobytes())
        elif isinstance(value,dict):
            for key,item in value.items():h.update(str(key).encode());visit(item)
        elif isinstance(value,(list,tuple)):
            h.update(str(type(value)).encode())
            for item in value:visit(item)
        else:h.update(repr(value).encode())
    try:visit(state);return h.hexdigest()
    finally:visit=None;state=None


def tiny(*,device,optimizer="sgd",failure=None,state_path=None,lifetime_path=None,nondefault=False,amp=False,overflow=False,modular=False,tied=False,early=False,interrupt_after_epoch=None,checkpoint=None,resume=None):
    if failure=="constructor":raise ValueError("controlled constructor failure")
    if failure=="memory":
        from hyso.performance._training_process import ResourceLimitError
        kernel=c.WinDLL("kernel32",use_last_error=True)
        kernel.VirtualAlloc.argtypes=[c.c_void_p,c.c_size_t,c.c_ulong,c.c_ulong];kernel.VirtualAlloc.restype=c.c_void_p
        kernel.VirtualFree.argtypes=[c.c_void_p,c.c_size_t,c.c_ulong];kernel.VirtualFree.restype=c.c_int
        pointer=kernel.VirtualAlloc(None,8*1024**3,0x3000,4)
        if pointer:
            kernel.VirtualFree(pointer,0,0x8000);raise RuntimeError("Job did not deny over-budget private commitment")
        raise ResourceLimitError("actual VirtualAlloc commitment denied by owned Job: "+str(c.get_last_error()))
    if failure=="logs":os.write(1,b"x"*200000)
    if failure=="tree":
        code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-I','-S','-c','import time;time.sleep(120)']); print('grandchild='+str(p.pid),flush=True); time.sleep(120)"
        child=subprocess.Popen([sys.executable,"-I","-S","-c",code])
        print("child="+str(child.pid),flush=True)
    if modular:
        from trainer_replay_support import build,CRITICAL
        t,train,val=build(dict(CRITICAL,optimizer=optimizer,scheduler='step',early='terminal' if early else False),device=device,amp=amp,overflow=overflow)
    else:
        model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.2),torch.nn.Linear(4,2)).to(device)
        if tied:
            shared=torch.nn.Linear(3,3)
            model=torch.nn.Sequential(shared,torch.nn.Sigmoid(),shared,torch.nn.Linear(3,2)).to(device)
        opt={"sgd":torch.optim.SGD,"adam":torch.optim.Adam,"adamw":torch.optim.AdamW}[optimizer](model.parameters(),lr=.01)
        sched=torch.optim.lr_scheduler.OneCycleLR(opt,.02,total_steps=4,cycle_momentum=False)
        t=Trainer(model,optimizer=opt,loss_fn=torch.nn.CrossEntropyLoss(),device=device,amp=amp,accum_steps=2,ema_decay=.8,scheduler=sched,grad_clip=.1)
        if amp:
            scale=t.scaler.state_dict();scale.update(scale=128.,growth_interval=1000);t.scaler.load_state_dict(scale)
        data=TensorDataset(torch.linspace(-1,1,15).reshape(5,3),torch.tensor([0,1,0,1,0]))
        train=DataLoader(data,batch_size=2,shuffle=True);val=DataLoader(data,batch_size=2)
        if overflow:
            count=[]
            def poison(g):
                count.append(True);return torch.full_like(g,float("inf")) if len(count)==1 else g
            t.model[0].weight.register_hook(poison)
    if failure in ("forward","timeout","oom"):
        def forward_hook(*args):
            if failure=="forward":raise RuntimeError("controlled training failure")
            if failure=="timeout":
                print("training_forward_started",flush=True);time.sleep(120)
            else:
                total=torch.cuda.get_device_properties(device).total_memory
                torch.empty(int(total*.30),device=device,dtype=torch.uint8)
        t.model.register_forward_pre_hook(forward_hook)
    if failure=='after_forward':
        def fail_after(*args):raise RuntimeError('controlled post-forward failure with saved payloads')
        t.model.register_forward_hook(fail_after)
    if failure=='backward':
        def fail_backward(gradient):raise RuntimeError('controlled backward failure with saved payloads')
        next(t.model.parameters()).register_hook(fail_backward)
    if lifetime_path:
        gc.disable();model_ref=weakref.ref(t.model);tensor_ref=weakref.ref(next(t.model.parameters()))
        def final_lifetime():
            Path(lifetime_path).write_text(json.dumps(dict(model_released=model_ref() is None,tensor_released=tensor_ref() is None)))
        atexit.register(final_lifetime)
    if state_path or interrupt_after_epoch:
        owner=weakref.ref(t)
        def fit(*args,**kwargs):
            actual=owner()
            if interrupt_after_epoch:
                from unittest.mock import patch
                from hyso.training import resume as checkpoint_module
                native=checkpoint_module.save_last
                def cut(*a,**kw):
                    native(*a,**kw)
                    if len(a[0].history['train_loss'])==interrupt_after_epoch:raise RuntimeError('controlled completed-epoch interruption')
                with patch.object(checkpoint_module,'save_last',cut):result=Trainer.fit(actual,*args,**kwargs)
            else:result=Trainer.fit(actual,*args,**kwargs)
            if state_path:Path(state_path).write_text(json.dumps(dict(state_sha256=state_digest(actual),fits=1)),encoding="utf-8")
            return result
        t.fit=fit
    stream=torch.cuda.Stream(device=device) if nondefault else None
    return dict(trainer=t,train_loader=train,val_loader=val,stream=stream)
