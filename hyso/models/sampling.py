"""Original HYSo sampling, restricted tensor utilities and prediction heads.

Public torch operators, root MIT license. Geometry never allocates tensors.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .components import _parameters, _input

SAMPLING = frozenset(('resize','pixel_shuffle','pixel_unshuffle'))
UTILITIES = frozenset(('flatten','permute','reshape'))
HEADS = frozenset(('classification_head','regression_head','embedding_head'))
NEW_NAMES = SAMPLING | UTILITIES | HEADS


def scalar(kind, value):
    if kind == 'scale_pair':
        values = [value,value] if type(value) in (int,float) else value
        valid = value is None or type(values) is list and len(values)==2 and all(
            type(v) in (int,float) and 0 < v <= 8192 and math.isfinite(v) for v in values)
        return valid, None if value is None else values
    if kind == 'optional_bool':
        return value is None or type(value) is bool, value
    if kind == 'chw':
        return type(value) is list and len(value)==3 and all(type(v) is int and 1<=v<=8192 for v in value),value
    if kind == 'axes':
        return type(value) is list and value in ([0,1,2,3],[0,1,3,2]),value
    return False,value


def validate(name,p,error):
    if name=='resize':
        if (p['output_size'] is None)==(p['scale_factor'] is None):
            raise error('resize requires exactly one of output_size or scale_factor',
                        expected='one explicit spatial target',actual=p)
        if p['mode'] in ('nearest','area') and p['align_corners'] is not None:
            raise error('align_corners must be None for nearest/area',expected=None,actual=p['align_corners'])


def shape(name,p,source,error):
    if len(source)!=3:
        raise error(f'{name} requires rank-4 NCHW input',code='INVALID_BLOCK_ORDER',
                    expected=['N','C','H','W'],actual=['N',*source])
    c,h,w=source
    if name=='resize':
        spatial=p['output_size'] or [math.floor(v*s) for v,s in zip((h,w),p['scale_factor'])]
        output=(c,*spatial)
    elif name=='pixel_shuffle':
        factor=p['upscale_factor']
        if c%(factor*factor):
            raise error('PixelShuffle C must be divisible by upscale_factor squared',code='CHANNEL_MISMATCH',
                        expected=f'C divisible by {factor*factor}',actual=c)
        output=(c//(factor*factor),h*factor,w*factor)
    elif name=='pixel_unshuffle':
        factor=p['downscale_factor']
        if h%factor or w%factor:
            raise error('PixelUnshuffle H and W must be divisible by downscale_factor',code='SPATIAL_DIMENSION_ERROR',
                        expected=f'H/W divisible by {factor}',actual=[h,w])
        output=(c*factor*factor,h//factor,w//factor)
    elif name=='flatten':
        output=(c*h*w,)
    elif name=='permute':
        output=tuple(source[index-1] for index in p['dims'][1:])
    else:
        output=tuple(p['shape'])
        if math.prod(output)!=c*h*w:
            raise error('reshape must preserve per-sample element count',code='INVALID_INPUT_SHAPE',
                        expected=c*h*w,actual=math.prod(output))
    if min(output)<1 or max(output)>8192:
        raise error(f'{name} produces a dimension outside 1..8192',code='SPATIAL_DIMENSION_ERROR',
                    expected='output dimensions in 1..8192',actual=list(output))
    return output


class Resize(nn.Module):
    """NCHW interpolation with exactly one target and explicit floor scale geometry."""
    def __init__(self,output_size=None,scale_factor=None,mode='nearest',align_corners=None):
        p=_parameters('resize',output_size=output_size,scale_factor=scale_factor,mode=mode,align_corners=align_corners)
        super().__init__()
        self.config=p

    def forward(self,x):
        from .registry import ModelConfigError
        _input(x,x.shape[1] if isinstance(x,torch.Tensor) and x.ndim==4 else None)
        shape('resize',self.config,tuple(x.shape[1:]),ModelConfigError)
        p=self.config
        return F.interpolate(x,size=p['output_size'],scale_factor=p['scale_factor'],
                             mode=p['mode'],align_corners=p['align_corners'])


class Permute(nn.Module):
    """Keep N/C axes fixed and optionally exchange spatial H/W axes."""
    def __init__(self,dims=(0,1,3,2)):
        p=_parameters('permute',dims=dims)
        super().__init__()
        self.dims=tuple(p['dims'])

    def forward(self,x):
        _input(x,x.shape[1] if isinstance(x,torch.Tensor) and x.ndim==4 else None)
        return x.permute(self.dims)


class Reshape(nn.Module):
    """Reinterpret each sample as explicit positive CHW, preserving batch and elements."""
    def __init__(self,shape):
        p=_parameters('reshape',shape=shape)
        super().__init__()
        self.shape=tuple(p['shape'])

    def forward(self,x):
        from .registry import ModelConfigError
        _input(x,x.shape[1] if isinstance(x,torch.Tensor) and x.ndim==4 else None)
        shape('reshape',dict(shape=self.shape),tuple(x.shape[1:]),ModelConfigError)
        return x.reshape(x.shape[0],*self.shape)


class Flatten(nn.Flatten):
    """Flatten CHW to one feature axis, preserving symbolic N."""
    def __init__(self):
        super().__init__(start_dim=1)

    def forward(self,x):
        _input(x,x.shape[1] if isinstance(x,torch.Tensor) and x.ndim==4 else None)
        return super().forward(x)


def head_geometry(name,p,source,error):
    if len(source)==3:
        features=math.prod(source) if p['pooling']=='none' else source[0]
    elif len(source)==1 and p['pooling']=='none':
        features=source[0]
    else:
        raise error('head pooling requires NCHW, vector features require pooling=none',
                    code='INVALID_OUTPUT_HEAD',expected='NCHW or vector with pooling=none',actual=['N',*source])
    if p['in_features']!=features:
        raise error('prediction head in_features mismatch',code='INVALID_OUTPUT_HEAD',expected=features,actual=p['in_features'])
    outputs=p.get('num_classes',p.get('output_features',p.get('embedding_dim')))
    return dict(in_features=features,output_features=outputs,parameter_count=(features+1)*outputs)


class PredictionHead(nn.Module):
    registry_name='classification_head'
    def __init__(self,**params):
        p=_parameters(self.registry_name,**params)
        super().__init__()
        self.config=p
        outputs=p.get('num_classes',p.get('output_features',p.get('embedding_dim')))
        self.pool={'avg':nn.AdaptiveAvgPool2d,'max':nn.AdaptiveMaxPool2d}.get(p['pooling'],lambda size:nn.Identity())(1)
        self.dropout=nn.Dropout(p.get('dropout',0.))
        self.linear=nn.Linear(p['in_features'],outputs)

    def forward(self,x):
        from .registry import ModelConfigError
        if not isinstance(x,torch.Tensor) or x.ndim not in (2,4) or min(x.shape)<1 or not x.is_floating_point():
            raise ModelConfigError('prediction head requires nonempty floating NCHW or NF',code='INVALID_OUTPUT_HEAD',
                                   expected='NCHW or NF',actual=list(x.shape) if isinstance(x,torch.Tensor) else type(x).__name__)
        head_geometry(self.registry_name,self.config,tuple(x.shape[1:]),ModelConfigError)
        output=self.linear(self.dropout(torch.flatten(self.pool(x),1)))
        if self.config.get('normalize',False):
            epsilon=max(self.config['eps'],torch.finfo(output.dtype).tiny)
            output=F.normalize(output,p=2,dim=1,eps=epsilon)
        return output


class RegressionHead(PredictionHead):
    registry_name='regression_head'


class EmbeddingHead(PredictionHead):
    registry_name='embedding_head'


def specifications(p):
    common=(p('in_features'),p('pooling','choice','avg',('avg','max','none')))
    return (
        ('resize','sampling',Resize.__doc__,Resize,(p('output_size','optional_pair',None),p('scale_factor','scale_pair',None),
         p('mode','choice','nearest',('nearest','bilinear','bicubic','area')),p('align_corners','optional_bool',None))),
        ('pixel_shuffle','sampling','C/r² channels, H*r and W*r spatial dimensions.',nn.PixelShuffle,(p('upscale_factor'),)),
        ('pixel_unshuffle','sampling','C*r² channels, H/r and W/r spatial dimensions.',nn.PixelUnshuffle,(p('downscale_factor'),)),
        ('flatten','tensor_utility',Flatten.__doc__,Flatten,()),
        ('permute','tensor_utility',Permute.__doc__,Permute,(p('dims','axes',(0,1,3,2)),)),
        ('reshape','tensor_utility',Reshape.__doc__,Reshape,(p('shape','chw'),)),
        ('classification_head','prediction_head','Global pool or flat features, optional dropout, linear logits.',PredictionHead,
         (*common,p('num_classes'),p('dropout','probability',0.))),
        ('regression_head','prediction_head','Global pool or flat features, unrestricted linear regression output.',RegressionHead,
         (*common,p('output_features'))),
        ('embedding_head','prediction_head','Global pool or flat features, linear embedding, optional safe L2 normalization.',EmbeddingHead,
         (*common,p('embedding_dim'),p('normalize','bool',False),p('eps','positive_float',1e-12))),
    )


def metadata(name):
    return dict(train_eval='Sampling/utilities are deterministic in train/eval. Head dropout is train-only.',
                parameter_dependencies=['resize requires exactly one target','pixel factors require divisibility',
                                        'reshape preserves sample element count','heads are terminal, in_features is inferred by Builder'],
                shape_contract='Batch is preserved. NCHW utilities keep rank 4, flatten produces NF at schema2 terminal feature boundary.',
                cost_contract='Sampling/utilities have zero trainable parameters. Heads have (in_features+1)*output_features parameters.',
                common_errors=['Choose positive bounded dimensions and legal mode/align_corners.',
                               'Keep N/C axes fixed in permute. Use pooling=none after flatten.'])
