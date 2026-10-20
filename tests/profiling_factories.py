"""Small importable profiling models; no downloads or external state."""
import torch
from torch import nn


def spatial(channels=2,outputs=4,groups=1,bias=True):
    return nn.Conv2d(channels,outputs,3,padding=1,groups=groups,bias=bias)


def conv1():
    return nn.Sequential(nn.Flatten(2),nn.Conv1d(2,4,3,padding=1,groups=2,bias=False))


class Shared(nn.Module):
    def __init__(self,fail=False,mutate=False):
        super().__init__();self.proj=nn.Conv2d(2,2,1,bias=False)
        self.alias=self.proj;self.branch=nn.Sequential(nn.ReLU(),self.proj)
        self.fail=fail;self.mutate=mutate;self.calls=0
        self.register_buffer('counter',torch.zeros(1))
        self.proj.weight.grad=torch.ones_like(self.proj.weight)
        self.proj.register_forward_hook(lambda m,a,o: None)
    def forward(self,x):
        self.calls+=1
        x=self.proj(x);x=self.alias(x);x=self.branch(x)
        if self.mutate and self.calls>1:self.counter.add_(1)
        if self.fail and self.calls>1:raise RuntimeError('controlled forward failure')
        return x+x


class Matrix(nn.Module):
    def __init__(self,kind='linear'):
        super().__init__();self.kind=kind
        self.register_buffer('w',torch.ones(35,3))
    def forward(self,x):
        a=x.flatten(1)
        if self.kind=='linear': return torch.nn.functional.linear(a,self.w.t(),torch.ones(3,device=x.device))
        if self.kind=='mm': return torch.mm(a,self.w)
        if self.kind=='matmul': return a@self.w
        if self.kind=='bmm':
            a=x.reshape(x.shape[0],5,7)
            return torch.bmm(a,a.transpose(1,2))
        if self.kind=='addmm': return torch.addmm(torch.ones(3,device=x.device),a,self.w)
        return torch.sin(a)


def pooled():
    return nn.Sequential(nn.ReLU(),nn.AdaptiveAvgPool2d(1))


class Norm(nn.Module):
    def __init__(self):
        super().__init__();self.bn=nn.BatchNorm2d(2);self.activation=nn.SiLU()
    def forward(self,x):return self.activation(self.bn(x))


def fixed_pool():return nn.MaxPool2d(2)


def adaptive_pool():return nn.AdaptiveAvgPool2d((2,3))


class Custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.library=torch.library.Library('hyso_profile_custom','DEF')
        self.library.define('linear(Tensor input, Tensor weight) -> Tensor')
        self.library.impl('linear',lambda input,weight:input@weight,'CPU')
        self.register_buffer('weight',torch.ones(35,3))
    def forward(self,x):return torch.ops.hyso_profile_custom.linear(x.flatten(1),self.weight)
