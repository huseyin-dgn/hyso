"""Trusted actual-model fixtures for owned Windows runtime workers."""
import os
import time
import torch
from torch import nn
from hyso.performance import InferenceWorkload, FrameInferenceProcessor, RetentionTracker


def make_cpu(mode='normal', delay=.005):
    if mode=='setup_oom':raise MemoryError('controlled constructor OOM receipt')
    model=nn.Sequential(nn.Flatten(),nn.Linear(12,2))
    base=InferenceWorkload(model,torch.ones(2,3,2,2))
    class Work:
        identity=base.identity
        def __init__(self):
            self.calls=0;self.held=[];self.retention_tracker=RetentionTracker()
            if mode=='release':
                self.held=[torch.ones(262144)];self.retention_tracker.watch(self.held[0],label='release-after-first-step')
        def step(self):
            self.calls+=1
            if mode=='hang':time.sleep(60)
            if mode=='crash':os._exit(17)
            if mode=='exception':raise RuntimeError('intentional real step failure')
            if mode=='oom':raise torch.cuda.OutOfMemoryError('controlled OOM exception without allocation stress')
            if mode=='device_error':raise RuntimeError('CUDA error: controlled device error receipt')
            if mode=='retain':
                tensor=torch.ones(32768);self.retention_tracker.watch(tensor,label='allocation-'+str(self.calls));self.held.append(tensor)
            if mode=='release':self.held.clear()
            time.sleep(delay+min(self.calls//20,5)*.002 if mode=='slowdown' else delay)
            return base.step()
        def close(self):
            if mode=='close_hang':time.sleep(60)
            if mode=='close_exception':raise RuntimeError('intentional close failure')
            base.close();self.held.clear()
    return Work()


def make_frames(delay=.03,mode='normal'):
    model=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,2))
    base=FrameInferenceProcessor(model)
    class Processor:
        identity=base.identity
        def process(self,frame):
            if mode=='exception':raise RuntimeError('intentional frame processing failure')
            if mode=='hang':time.sleep(60)
            a=time.perf_counter_ns();time.sleep(delay);b=time.perf_counter_ns()
            return [dict(name='controlled_service_wait',start_ns=a,end_ns=b)]+base.process(frame)
        def close(self):base.close()
    return Processor()


def make_gpu(gpu_uuid,cuda_memory_fraction=None):
    model=nn.Sequential(nn.Conv2d(3,4,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(4,2))
    return InferenceWorkload(model,torch.ones(2,3,16,16),device='cuda:0',gpu_uuid=gpu_uuid,cuda_memory_fraction=cuda_memory_fraction)


def make_gpu_cache(gpu_uuid):
    base=InferenceWorkload(nn.Sequential(nn.Flatten(),nn.Linear(12,2)),torch.ones(2,3,2,2),device='cuda:0',gpu_uuid=gpu_uuid,
                           cuda_memory_fraction=.1)  # Explicit test-only cache probe guard.
    base.step()  # Settle lazy GEMM workspace before the observed cache probe.
    class Cache:
        identity=dict(base.identity,setup_warmup_forwards=1,controlled_temporary_bytes=[2*1024**2,8*1024**2,16*1024**2])
        def __init__(self):self.calls=0
        def step(self):
            self.calls+=1
            size=self.identity['controlled_temporary_bytes'][min(self.calls-1,2)]
            temporary=torch.empty(size//4,device='cuda:0').zero_()
            torch.cuda.synchronize(0);del temporary
            return base.step()
        def close(self):base.close()
    return Cache()
