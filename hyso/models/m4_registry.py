# HYSo integration metadata for the sources listed in THIRD_PARTY_NOTICES.md.
# Full upstream licenses and notices are retained in THIRD_PARTY_NOTICES.md.
"""Closed Registry specifications and pure Builder geometry for M4 operators."""
from __future__ import annotations

import math
from torchvision.models._utils import _make_divisible
from .m4_blocks import (InvertedResidual, MBConv, FusedMBConv, GhostConv, GhostBottleneck,
                        ChannelShuffle, ShuffleUnit, DropBlock2D)
from .m4_reparam import RepVGGBlock, MobileOneBlock
from .mobile_architectures import MobileNetV2, MobileNetV3, EfficientNetV2, GhostNet, ShuffleNetV2
from .reparam_architectures import RepVGG, MobileOne
from .m4_config import M4_FAMILIES, validate_mobile, _conv, _se, _rep_block

M4_BLOCKS = ('inverted_residual', 'mbconv', 'fused_mbconv', 'ghost_conv', 'ghost_bottleneck',
             'channel_shuffle', 'shuffle_unit', 'repvgg_block', 'mobileone_block', 'dropblock2d')


def specifications(parameter):
    """Return constructor metadata without constructing tensors or modules."""
    p = parameter
    channels = (p('in_channels'),p('out_channels'),p('stride',default=1))
    expansion = (p('expand_ratio','positive_float',6.,description='Expansion ratio in [1,16].'),)
    mb_options = (p('kernel_size','odd_int',3),p('stochastic_depth','probability',0.))
    block_specs = (
        ('inverted_residual',InvertedResidual,channels+expansion),
        ('mbconv',MBConv,channels+expansion+mb_options),
        ('fused_mbconv',FusedMBConv,channels+(p('expand_ratio','positive_float',4.),)+mb_options),
        ('ghost_conv',GhostConv,channels+(p('kernel_size','odd_int',1),p('ratio',default=2),
                                        p('cheap_kernel','odd_int',3),p('relu','bool',True))),
        ('ghost_bottleneck',GhostBottleneck,channels+(p('hidden_channels'),p('kernel_size','odd_int',3),
                                                    p('se_ratio','probability_closed',0.))),
        ('channel_shuffle',ChannelShuffle,(p('channels'),p('groups',default=2))),
        ('shuffle_unit',ShuffleUnit,channels),
        ('repvgg_block',RepVGGBlock,channels+(p('groups',default=1),p('deploy','bool',False))),
        ('mobileone_block',MobileOneBlock,channels+(p('kernel_size','odd_int',3),p('groups',default=1),
                                                  p('num_conv_branches',default=1),p('use_se','bool',False),p('deploy','bool',False))),
        ('dropblock2d',DropBlock2D,(p('p','probability_closed',.1),p('block_size',default=3))),
    )
    result = []
    for name,constructor,params in block_specs:
        category = 'channel_mixing' if name in ('channel_shuffle',) else 'regularization' if name == 'dropblock2d' else 'mobile'
        result.append((name,category,constructor.__doc__,constructor,params))
    common = (p('input_channels',default=3),p('num_classes',default=1000))
    width = p('width_mult','positive_float',1.,description='Explicit width multiplier in [0.125,2].')
    depth = p('depth_mult','positive_float',1.,description='Explicit repeat multiplier in [0.1,1].')
    dropout = p('dropout','probability',.2)
    model_specs = (
        ('mobilenet_v2',MobileNetV2,common+(width,dropout)),
        ('mobilenet_v3',MobileNetV3,common+(p('variant','choice','large',('small','large')),width,dropout)),
        ('efficientnet_v2',EfficientNetV2,common+(p('variant','choice','s',('s','m','l')),width,depth,
                                               p('dropout','optional_probability',None),p('stochastic_depth','probability',.2))),
        ('ghostnet',GhostNet,common+(width,dropout)),
        ('shufflenet_v2',ShuffleNetV2,common+(p('variant','choice','x1_0',('x0_5','x1_0','x1_5','x2_0')),)),
        ('repvgg',RepVGG,common+(p('variant','choice','a0',('a0','a1')),width,depth,p('deploy','bool',False))),
        ('mobileone',MobileOne,common+(p('variant','choice','s0',('s0','s1','s2')),width,depth,p('deploy','bool',False))),
    )
    result.extend((name,'model',constructor.__doc__,constructor,params) for name,constructor,params in model_specs)
    return tuple(result)


def validate_block(name: str, p: dict) -> None:
    """Mirror scalar constructor restrictions while keeping preflight allocation-free."""
    if 'stride' in p and p['stride'] not in (1,2):
        raise ValueError('M4 stride must be 1 or 2')
    for key in ('kernel_size','cheap_kernel'):
        if key in p and p[key] not in (1,3,5,7):
            raise ValueError(f'{key} must be 1,3,5 or 7')
    if 'expand_ratio' in p and not 1 <= p['expand_ratio'] <= 16:
        raise ValueError('expand_ratio must be in [1,16]')
    if 'groups' in p:
        counts = (p['channels'],) if 'channels' in p else (p['in_channels'],p['out_channels'])
        if any(channels % p['groups'] for channels in counts):
            raise ValueError('groups must divide channel counts')
    if name == 'ghost_conv' and not 2 <= p['ratio'] <= 8:
        raise ValueError('Ghost ratio must be 2..8')
    if name == 'shuffle_unit' and (p['out_channels'] % 2 or (p['stride'] == 1 and p['in_channels'] != p['out_channels'])):
        raise ValueError('ShuffleUnit requires even output channels and equal channels for stride=1')
    if name == 'mobileone_block' and not 1 <= p['num_conv_branches'] <= 8:
        raise ValueError('num_conv_branches must be 1..8')


def block_cost(name: str, p: dict) -> int:
    """Count a conservative affine-parameter bound using the actual block geometry."""
    if name in ('channel_shuffle','dropblock2d'):
        return 0
    a,b = p['in_channels'],p['out_channels']
    if name in ('repvgg_block','mobileone_block'):
        total = _rep_block(a,b,p.get('kernel_size',3),p['stride'],p['groups'],p.get('num_conv_branches',1),p['deploy'])
        return total + (_se(b,max(1,b//16)) if p.get('use_se') else 0)
    if name in ('inverted_residual','mbconv','fused_mbconv'):
        hidden = round(a*p['expand_ratio']) if name == 'inverted_residual' else _make_divisible(a*p['expand_ratio'],8)
        kernel = p.get('kernel_size',3)
        if name == 'fused_mbconv':
            return _conv(a,hidden,kernel)+_conv(hidden,b) if hidden != a else _conv(a,b,kernel)
        has_expansion = p['expand_ratio'] != 1 if name == 'inverted_residual' else hidden != a
        total = (_conv(a,hidden) if has_expansion else 0) + _conv(hidden,hidden,kernel,hidden) + _conv(hidden,b)
        return total + (_se(hidden,max(1,a//4)) if name == 'mbconv' else 0)
    if name == 'ghost_conv':
        primary = math.ceil(b/p['ratio'])
        return _conv(a,primary,p['kernel_size'])+_conv(primary,primary*(p['ratio']-1),p['cheap_kernel'],primary)
    if name == 'ghost_bottleneck':
        hidden = p['hidden_channels']
        total = block_cost('ghost_conv',dict(in_channels=a,out_channels=hidden,ratio=2,kernel_size=1,cheap_kernel=3))
        total += block_cost('ghost_conv',dict(in_channels=hidden,out_channels=b,ratio=2,kernel_size=1,cheap_kernel=3))
        if p['stride'] == 2:
            total += _conv(hidden,hidden,p['kernel_size'],hidden)
        if p['se_ratio']:
            total += _se(hidden,_make_divisible(hidden*p['se_ratio'],4))
        if a != b or p['stride'] != 1:
            total += _conv(a,a,p['kernel_size'],a)+_conv(a,b)
        return total
    half = b//2
    if p['stride'] == 1:
        return 2*_conv(half,half)+_conv(half,half,3,half)
    return _conv(a,a,3,a)+2*_conv(a,half)+_conv(half,half,3,half)+_conv(half,half)


def block_shape(name: str, p: dict, shape: tuple[int,int,int]) -> tuple[int,int,int]:
    """Infer NCHW block output geometry without allocating or executing tensors."""
    channels,height,width = shape
    if name in ('channel_shuffle','dropblock2d'):
        return shape
    stride = p['stride']
    return p['out_channels'], (height+stride-1)//stride, (width+stride-1)//stride
