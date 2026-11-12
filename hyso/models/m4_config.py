# torchvision configuration metadata uses BSD-3-Clause.
# Ghost configuration adapted from Huawei Technologies Co., Ltd. (c) 2020, Apache-2.0.
# RepVGG configuration adapted from Copyright (c) 2020 DingXiaoH, MIT.
# MobileOne configuration adapted from Copyright (C) 2022 Apple Inc., Apple license.
# HYSo changes (2026) add scalar preflight and parameter accounting.
# Full upstream licenses and notices are retained in THIRD_PARTY_NOTICES.md.
"""Pure mobile-family layouts, validation and conservative parameter accounting."""
from __future__ import annotations

import math
from torchvision.models._utils import _make_divisible
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf
from torchvision.models.efficientnet import _efficientnet_conf, FusedMBConvConfig
from .m4_blocks import probability

M4_FAMILIES = ('mobilenet_v2', 'mobilenet_v3', 'efficientnet_v2', 'ghostnet',
               'shufflenet_v2', 'repvgg', 'mobileone')
MOBILE_V2_STAGES = ((1,16,1,1),(6,24,2,2),(6,32,3,2),(6,64,4,2),
                    (6,96,3,1),(6,160,3,2),(6,320,1,1))
# GhostNetV1 author configuration, flattened across the original stage groups.
GHOST_STAGES = ((3,16,16,0.,1),(3,48,24,0.,2),(3,72,24,0.,1),
                (5,72,40,.25,2),(5,120,40,.25,1),(3,240,80,0.,2),
                (3,200,80,0.,1),(3,184,80,0.,1),(3,184,80,0.,1),
                (3,480,112,.25,1),(3,672,112,.25,1),(5,672,160,.25,2),
                (5,960,160,0.,1),(5,960,160,.25,1),(5,960,160,0.,1),(5,960,160,.25,1))
SHUFFLE_WIDTHS = {'x0_5':(24,48,96,192,1024), 'x1_0':(24,116,232,464,1024),
                  'x1_5':(24,176,352,704,1024), 'x2_0':(24,244,488,976,2048)}
REP_VARIANTS = {'a0':(.75,.75,.75,2.5), 'a1':(1.,1.,1.,2.5)}
ONE_VARIANTS = {'s0':((.75,1.,1.,2.),4), 's1':((1.5,1.5,2.,2.5),1),
                's2':((1.5,2.,2.5,4.),1)}


def positive_scale(value: float, name: str, minimum: float = .125, maximum: float = 2.) -> None:
    """Validate a finite bounded width/depth multiplier without constructing models."""
    if type(value) not in (int, float) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be finite and in [{minimum},{maximum}]')


def efficient_settings(variant: str, width_mult: float, depth_mult: float):
    """Reuse fresh torchvision V2 config records, then apply explicit width/depth scaling."""
    settings, last = _efficientnet_conf('efficientnet_v2_' + variant)
    for config in settings:
        config.input_channels = _make_divisible(config.input_channels*width_mult, 8)
        config.out_channels = _make_divisible(config.out_channels*width_mult, 8)
        config.num_layers = max(1, math.ceil(config.num_layers*depth_mult))
    return settings, last




def rep_settings(family: str, variant: str, width_mult: float, depth_mult: float):
    """Return canonical RepVGG/MobileOne widths, repeats and branch multiplicity."""
    if family == 'repvgg':
        multipliers, repeats, branches = REP_VARIANTS[variant], (2,4,14,1), 1
    else:
        multipliers, branches = ONE_VARIANTS[variant]
        repeats = (2,8,10,1)
    widths = tuple(max(1,int(base*m*width_mult)) for base,m in zip((64,128,256,512),multipliers))
    depths = tuple(max(1,math.ceil(d*depth_mult)) for d in repeats)
    return min(64, widths[0]), widths, depths, branches


def _conv(a: int, b: int, kernel: int = 1, groups: int = 1, bn: bool = True, bias: bool = False) -> int:
    return a*b*kernel*kernel//groups + (2*b if bn else 0) + (b if bias else 0)


def _se(channels: int, hidden: int) -> int:
    return (channels+1)*hidden + (hidden+1)*channels


def _ghost(a: int, b: int) -> int:
    primary = math.ceil(b/2)
    return _conv(a,primary) + _conv(primary,primary,3,primary)


def _rep_block(a: int, b: int, kernel: int, stride: int, groups: int,
               branches: int, deploy: bool) -> int:
    if deploy:
        return _conv(a,b,kernel,groups,bn=False,bias=True)
    total = branches*_conv(a,b,kernel,groups)
    if kernel > 1:
        total += _conv(a,b,1,groups)
    if a == b and stride == 1:
        total += 2*a
    return total


def validate_mobile(name: str, params: dict) -> int:
    """Validate a closed family request and bound parameters without module allocation.

    Counting follows constructor geometry and includes BN affine parameters and
    classifier biases. The bound covers parameters, while activation memory and
    execution time depend on runtime inputs and protocols.
    """
    if name not in M4_FAMILIES:
        raise ValueError('unknown M4 family')
    for key in ('input_channels', 'num_classes'):
        if type(params[key]) is not int or not 1 <= params[key] <= 8192:
            raise ValueError(f'{key} must be an integer in 1..8192')
    width = params.get('width_mult',1.)
    depth = params.get('depth_mult',1.)
    positive_scale(width,'width_mult')
    positive_scale(depth,'depth_mult',.1,1.)
    if 'dropout' in params and not (name == 'efficientnet_v2' and params['dropout'] is None):
        probability(params['dropout'],'dropout')
    if name == 'efficientnet_v2' or params.get('stochastic_depth') is not None:
        probability(params['stochastic_depth'],'stochastic_depth')
    if 'deploy' in params and type(params['deploy']) is not bool:
        raise ValueError('deploy must be bool')
    variant = params.get('variant')
    variants = {'mobilenet_v3':('small','large'), 'efficientnet_v2':('s','m','l'),
                'shufflenet_v2':tuple(SHUFFLE_WIDTHS),
                'repvgg':tuple(REP_VARIANTS), 'mobileone':tuple(ONE_VARIANTS)}
    if name in variants and (type(variant) is not str or variant not in variants[name]):
        raise ValueError(f'{name} variant must be one of {variants[name]}')
    incoming, classes = params['input_channels'], params['num_classes']
    if name == 'mobilenet_v2':
        previous = _make_divisible(32*width,8)
        total = _conv(incoming,previous,3)
        for expansion,channels,repeats,stride in MOBILE_V2_STAGES:
            output = _make_divisible(channels*width,8)
            for _ in range(repeats):
                hidden = round(previous*expansion)
                total += (_conv(previous,hidden) if expansion != 1 else 0)
                total += _conv(hidden,hidden,3,hidden) + _conv(hidden,output)
                previous = output
        head = _make_divisible(1280*max(1.,width),8)
        total += _conv(previous,head) + (head+1)*classes
    elif name == 'mobilenet_v3':
        settings, head = _mobilenet_v3_conf('mobilenet_v3_'+variant,width_mult=width)
        total = _conv(incoming,settings[0].input_channels,3)
        for config in settings:
            a,b,e = config.input_channels,config.out_channels,config.expanded_channels
            total += (_conv(a,e) if a != e else 0) + _conv(e,e,config.kernel,e) + _conv(e,b)
            if config.use_se:
                total += _se(e,_make_divisible(e//4,8))
        last = settings[-1].out_channels
        total += _conv(last,6*last) + (6*last+1)*head + (head+1)*classes
    elif name == 'efficientnet_v2':
        settings, head = efficient_settings(variant,width,depth)
        total = _conv(incoming,settings[0].input_channels,3)
        for config in settings:
            for index in range(config.num_layers):
                a = config.input_channels if index == 0 else config.out_channels
                b = config.out_channels
                e = _make_divisible(a*config.expand_ratio,8)
                if isinstance(config,FusedMBConvConfig):
                    total += _conv(a,e,config.kernel) + _conv(e,b) if e != a else _conv(a,b,config.kernel)
                else:
                    total += (_conv(a,e) if a != e else 0) + _conv(e,e,config.kernel,e) + _se(e,max(1,a//4)) + _conv(e,b)
        total += _conv(settings[-1].out_channels,head) + (head+1)*classes
    elif name == 'ghostnet':
        previous = _make_divisible(16*width,4)
        total = _conv(incoming,previous,3)
        for kernel,expanded,channels,se,stride in GHOST_STAGES:
            hidden,output = _make_divisible(expanded*width,4),_make_divisible(channels*width,4)
            total += _ghost(previous,hidden) + _ghost(hidden,output)
            if stride == 2:
                total += _conv(hidden,hidden,kernel,hidden)
            if se:
                total += _se(hidden,_make_divisible(hidden*se,4))
            if stride != 1 or previous != output:
                total += _conv(previous,previous,kernel,previous) + _conv(previous,output)
            previous = output
        head = _make_divisible(960*width,4)
        total += _conv(previous,head) + _conv(head,1280,bn=False,bias=True) + (1280+1)*classes
    elif name == 'shufflenet_v2':
        stem,*stages,head = SHUFFLE_WIDTHS[variant]
        total = _conv(incoming,stem,3)
        previous = stem
        for output,repeats in zip(stages,(4,8,4)):
            half = output//2
            total += _conv(previous,previous,3,previous) + _conv(previous,half)
            total += _conv(previous,half) + _conv(half,half,3,half) + _conv(half,half)
            total += (repeats-1)*(2*_conv(half,half) + _conv(half,half,3,half))
            previous = output
        total += _conv(previous,head) + (head+1)*classes
    else:
        stem,widths,depths,branches = rep_settings(name,variant,width,depth)
        deploy = params['deploy']
        total = _rep_block(incoming,stem,3,2,1,1,deploy)
        previous = stem
        for channels,repeats in zip(widths,depths):
            for index in range(repeats):
                stride = 2 if index == 0 else 1
                if name == 'mobileone':
                    total += _rep_block(previous,previous,3,stride,previous,branches,deploy)
                    total += _rep_block(previous,channels,1,1,1,branches,deploy)
                else:
                    total += _rep_block(previous,channels,3,stride,1,1,deploy)
                previous = channels
        total += (previous+1)*classes
    if total > 200_000_000:
        raise ValueError('M4 model exceeds the 200-million parameter budget')
    return total
