"""Pinned toy factories. Imported only inside explicitly admitted workers."""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def model(channels=1):
    return nn.Sequential(nn.Conv2d(channels,4,3,padding=1),nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(4,2))


def data():
    x=torch.arange(8*16*16,dtype=torch.float32).reshape(8,1,16,16)/2048
    y=torch.arange(8,dtype=torch.long)%2
    return x,y


def training(seed=17,device='cuda:0',amp=False,frozen=False):
    from hyso.training import Trainer
    torch.manual_seed(seed);m=model()
    if frozen:
        with torch.no_grad():m[-1].weight.zero_();m[-1].bias.copy_(torch.tensor([20.,-20.]))
    m=m.to(device)
    t=Trainer(m,optimizer=torch.optim.SGD(m.parameters(),lr=0.001),
        loss_fn=nn.CrossEntropyLoss(),device=device,amp=amp)
    x,y=data();return t,DataLoader(TensorDataset(x,y),batch_size=4,shuffle=False)


def time_target_factory(seed):return training(seed,frozen=True)


def frame_processor(gpu_uuid):
    from hyso.performance import FrameInferenceProcessor
    return FrameInferenceProcessor(model(3),size=(16,16),device='cuda:0',
        gpu_uuid=gpu_uuid,cuda_memory_fraction=0.15)


def stability_workload(gpu_uuid):
    from hyso.performance import InferenceWorkload
    import time
    class BoundedTrend:
        def __init__(self):
            self.base=InferenceWorkload(model(),data()[0][:1],device='cuda:0',gpu_uuid=gpu_uuid,cuda_memory_fraction=0.15)
            self.identity=dict(self.base.identity,extra_allocation_ceiling_bytes=512*1024)
            self.extra=None;self.index=0
        def step(self):
            # At most 512 KiB live extras. Observe release/cache reuse without
            # peak reset, empty_cache, deliberate OOM, or allocation stress.
            self.extra=None
            size=(0,65536,131072,0,65536,0)[self.index%6];self.index+=1
            if size:self.extra=torch.ones(size,device='cuda:0')
            count=self.base.step();time.sleep(.11);return count
        def close(self):self.extra=None;self.base.close()
    return BoundedTrend()


def device_pair():return model(),data()[0][:1]


def onnx_model(identity_path=None):
    import io
    import numpy as np
    if identity_path is not None:
        from .native_identity import register_ort_identity
        register_ort_identity(identity_path)
    buf=io.BytesIO();m=model().eval();x=data()[0][:1]
    torch.onnx.export(m,x,buf,input_names=['image'],output_names=['logits'],opset_version=17)
    return buf.getvalue(),{'image':np.asarray(x.numpy())}


def perturb(x,generator,severity):
    return x+severity*torch.rand(x.shape,generator=generator,dtype=x.dtype)/100
