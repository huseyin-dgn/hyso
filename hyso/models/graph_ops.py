"""Original HYSo fusion primitives, strict equal-shape arithmetic, root MIT."""
from __future__ import annotations
import math
import torch
from torch import nn

NAMES = frozenset(('add','concat','multiply','weighted_sum'))


def parameters(name,params):
    from .registry import ModelConfigError,_plain
    p=_plain(params)
    fields={'axis'} if name=='concat' else {'weights','learnable'} if name=='weighted_sum' else set()
    if type(p) is not dict or set(p)-fields:
        raise ModelConfigError('unknown fusion parameters',expected=sorted(fields),actual=p)
    if name=='concat':
        axis=p.get('axis',1)
        if type(axis) is not int or not 1<=axis<=3:
            raise ModelConfigError('concat axis must be a non-batch axis in 1..3',expected=[1,2,3],actual=axis)
        return dict(axis=axis)
    if name=='weighted_sum':
        if not weights_value(p.get('weights'))[0] or type(p.get('learnable',False)) is not bool:
            raise ModelConfigError('weighted_sum requires 2..8 finite weights in [-1e6,1e6] and bool learnable',actual=p)
        return dict(weights=p['weights'],learnable=p.get('learnable',False))
    return {}


def weights_value(value):
    valid = type(value) is list and 2<=len(value)<=8 and all(
        type(v) in (int,float) and abs(v)<=1e6 and math.isfinite(v) for v in value)
    return valid,value


def fusion_shape(name,params,shapes,error):
    if not 2<=len(shapes)<=8:
        raise error('fusion requires 2..8 ordered inputs',code='GRAPH_STRUCTURE_ERROR',expected='2..8 inputs',actual=len(shapes))
    first=tuple(shapes[0])
    if any(len(s)!=len(first) for s in shapes):
        raise error('fusion ranks differ',code='GRAPH_SHAPE_ERROR',expected=list(first),actual=shapes)
    if name=='concat':
        axis=params['axis']
        if not 1<=axis<=len(first):
            raise error('concat axis must address a non-batch axis',code='GRAPH_SHAPE_ERROR',expected=list(range(1,len(first)+1)),actual=axis)
        index=axis-1
        if any(any(a!=b for i,(a,b) in enumerate(zip(first,s)) if i!=index) for s in shapes):
            raise error('concat shapes differ outside the selected axis',code='GRAPH_SHAPE_ERROR',expected=list(first),actual=shapes)
        output=list(first);output[index]=sum(s[index] for s in shapes)
    else:
        if any(tuple(s)!=first for s in shapes):
            raise error('fusion forbids broadcasting and requires identical shapes',code='GRAPH_SHAPE_ERROR',
                        expected=list(first),actual=shapes,
                        suggestion='Kanal için explicit 1x1 conv, spatial boyut için stride/resize uygulayın veya uygun concat seçin.')
        if name=='weighted_sum' and len(params['weights'])!=len(shapes):
            raise error('weight count must match ordered input count',code='GRAPH_SHAPE_ERROR',expected=len(shapes),actual=len(params['weights']))
        output=list(first)
    if min(output)<1 or max(output)>8192:
        raise error('fusion output dimension exceeds 1..8192',code='GRAPH_LIMIT_ERROR',expected='dimensions 1..8192',actual=output)
    return tuple(output)


class Fusion(nn.Module):
    name='add'
    def __init__(self,**params):
        p=parameters(self.name,params)
        super().__init__()
        self.config=p
        if self.name=='weighted_sum':
            weights=torch.tensor(p['weights'],dtype=torch.float32)
            if p['learnable']:self.weights=nn.Parameter(weights)
            else:self.register_buffer('weights',weights)

    def forward(self,*inputs):
        from .registry import ModelConfigError
        if any(not isinstance(x,torch.Tensor) or x.ndim not in (2,4) or min(x.shape)<1 or not x.is_floating_point() for x in inputs):
            raise ModelConfigError('fusion requires nonempty floating NC/NCHW',code='GRAPH_SHAPE_ERROR')
        fusion_shape(self.name,self.config,[tuple(x.shape[1:]) for x in inputs],ModelConfigError)
        if any(x.shape[0]!=inputs[0].shape[0] or x.device!=inputs[0].device or x.dtype!=inputs[0].dtype for x in inputs):
            raise ModelConfigError('fusion batch/device/dtype mismatch',code='GRAPH_SHAPE_ERROR')
        if self.name=='concat':return torch.cat(inputs,dim=self.config['axis'])
        result=inputs[0]*self.weights[0] if self.name=='weighted_sum' else inputs[0]
        for index,value in enumerate(inputs[1:],1):
            if self.name=='multiply':result=result*value
            elif self.name=='weighted_sum':result=result+self.weights[index]*value
            else:result=result+value
        return result


class Concat(Fusion):
    name='concat'


class Multiply(Fusion):
    name='multiply'


class WeightedSum(Fusion):
    name='weighted_sum'


CONSTRUCTORS={'add':Fusion,'concat':Concat,'multiply':Multiply,'weighted_sum':WeightedSum}


def metadata(name):
    return dict(name=name,category='graph_fusion',input_count='2..8',layouts=['NC','NCHW'],
                description='Explicit fusion without broadcasting, implicit projection or resize.',
                parameters={'axis':dict(default=1,minimum=1,maximum=3)} if name=='concat' else
                           {'weights':dict(min_items=2,max_items=8,minimum=-1e6,maximum=1e6),
                            'learnable':dict(default=False)} if name=='weighted_sum' else {},
                train_eval='Identical arithmetic in train and eval.',
                corrections='Match ranks and shapes, use a non-batch concat axis and one weight per ordered input.')
