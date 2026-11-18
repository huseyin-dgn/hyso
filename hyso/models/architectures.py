"""Closed, untrained classifier families. No pretrained downloads or implicit casts."""
from __future__ import annotations
import math
import torch
from torch import nn
from torchvision.models.resnet import ResNet as TVResNet, BasicBlock, Bottleneck
from torchvision.models.densenet import DenseNet as TVDenseNet
from .blocks import _positive_int
from .m3_residual import PreActivationBlock, WideResidualBlock, normalization

RESNET_LAYERS = {18:(2,2,2,2),34:(3,4,6,3),50:(3,4,6,3),101:(3,4,23,3),152:(3,8,36,3)}
DENSE_LAYERS = {121:(6,12,24,16),169:(6,12,32,32),201:(6,12,48,32)}


def validate_family(name: str, p: dict) -> int:
    """Pure structural validation and conservative parameter estimate, before allocation."""
    for key in ('input_channels','num_classes'): _positive_int(p[key],key)
    if max(p['input_channels'],p['num_classes']) > 8192: raise ValueError('channels/classes must be <=8192')
    if name not in ('resnet','preact_resnet','wide_resnet','resnext','densenet'): raise ValueError('unknown model family')
    if name == 'densenet':
        if type(p['depth']) is not int or p['depth'] not in DENSE_LAYERS: raise ValueError('DenseNet depth must be 121,169,201')
        g=p['growth_rate'];_positive_int(g,'growth_rate')
        if g>64: raise ValueError('growth_rate must be <=64')
        if type(p['dropout']) not in (int,float) or not math.isfinite(p['dropout']) or not 0<=p['dropout']<1: raise ValueError('dropout must be in [0,1)')
        c=2*g;cost=49*p['input_channels']*c+2*c
        for i,count in enumerate(DENSE_LAYERS[p['depth']]):
            for _ in range(count): cost+=2*c+c*4*g+8*g+36*g*g;c+=g
            if i<3:cost+=2*c+c*(c//2);c//=2
        cost+=2*c+(c+1)*p['num_classes']
    else:
        normalization(p['norm'])
        if name=='wide_resnet':
            d,k=p['depth'],p['widen_factor']
            if type(d) is not int or d<10 or d>100 or (d-4)%6: raise ValueError('WideResNet depth must satisfy 6n+4, 10..100')
            if type(k) is not int or not 1<=k<=16: raise ValueError('widen_factor must be 1..16')
            if type(p['dropout']) not in (int,float) or not math.isfinite(p['dropout']) or not 0<=p['dropout']<1: raise ValueError('dropout must be in [0,1)')
            widths=[16*k,32*k,64*k];counts=[(d-4)//6]*3;exp=1;previous=16
            cost=9*p['input_channels']*16
        else:
            if type(p['depth']) is not int or p['depth'] not in RESNET_LAYERS: raise ValueError('ResNet depth must be 18,34,50,101,152')
            if name=='resnext' and p['depth'] not in (50,101): raise ValueError('ResNeXt depth must be 50 or 101')
            exp=1 if p['depth']<50 else 4
            base=p.get('base_channels',64)
            if type(base) is not int or not 1<=base<=128: raise ValueError('base_channels must be 1..128')
            widths=[base,base*2,base*4,base*8];counts=RESNET_LAYERS[p['depth']];previous=base
            cost=49*p['input_channels']*base+2*base
        for width,count in zip(widths,counts):
            out=width*exp
            for _ in range(count):
                inner=width
                groups=p.get('groups',1);bwidth=p.get('width_per_group',64)
                if name=='resnext':
                    if type(groups) is not int or not 1<=groups<=64 or type(bwidth) is not int or not 1<=bwidth<=64: raise ValueError('groups and width_per_group must be 1..64')
                    inner=int(width*bwidth/64)*groups
                    if inner<1: raise ValueError('group bottleneck width is zero')
                cost+=(previous*inner+9*inner*inner//groups+inner*out if exp==4 else 9*previous*out+9*out*out)
                cost+=previous*out+8*(previous+out+inner) # conservative projection/normalization allowance
                previous=out
        cost+=(previous+1)*p['num_classes']
    if cost>200_000_000: raise ValueError('named model exceeds conservative 200-million parameter budget')
    return cost


class ResNet(TVResNet):
    """Torchvision ResNet 18/34 or v1.5 bottleneck 50/101/152; ImageNet stem.

    input_channels and num_classes change only input/output contracts. No weights.
    State keys match torchvision when dimensions/norm match. Not original v1
    stride placement for bottleneck models: stride is on 3x3 (v1.5).
    """
    def __init__(self, depth: int = 18, input_channels: int = 3, num_classes: int = 1000,
                 norm: str | None = 'batch') -> None:
        validate_family('resnet',dict(depth=depth,input_channels=input_channels,num_classes=num_classes,norm=norm))
        super().__init__(BasicBlock if depth<50 else Bottleneck,RESNET_LAYERS[depth],num_classes=num_classes,norm_layer=normalization(norm))
        if input_channels!=3:self.conv1=nn.Conv2d(input_channels,64,7,2,3,bias=False);nn.init.kaiming_normal_(self.conv1.weight,mode='fan_out',nonlinearity='relu')


class ResNeXt(TVResNet):
    """Torchvision grouped bottleneck topology; default ResNeXt-50 32x4d.

    For the common 101-32x8d variant explicitly select width_per_group=8.
    """
    def __init__(self, depth: int = 50, input_channels: int = 3, num_classes: int = 1000,
                 groups: int = 32, width_per_group: int = 4, norm: str | None = 'batch') -> None:
        validate_family('resnext',dict(depth=depth,input_channels=input_channels,num_classes=num_classes,groups=groups,width_per_group=width_per_group,norm=norm))
        super().__init__(Bottleneck,RESNET_LAYERS[depth],num_classes=num_classes,groups=groups,width_per_group=width_per_group,norm_layer=normalization(norm))
        if input_channels!=3:self.conv1=nn.Conv2d(input_channels,64,7,2,3,bias=False);nn.init.kaiming_normal_(self.conv1.weight,mode='fan_out',nonlinearity='relu')


class DenseNet(TVDenseNet):
    """Torchvision DenseNet-BC (bottleneck factor 4, transition compression 0.5).

    Depth layouts 121/169/201, default growth=32 and initial channels=64 preserve
    standard topology. A custom growth_rate is explicitly a width variant.
    """
    def __init__(self, depth: int = 121, input_channels: int = 3, num_classes: int = 1000,
                 growth_rate: int = 32, dropout: float = 0.) -> None:
        validate_family('densenet',dict(depth=depth,input_channels=input_channels,num_classes=num_classes,growth_rate=growth_rate,dropout=dropout))
        super().__init__(growth_rate,DENSE_LAYERS[depth],2*growth_rate,4,dropout,num_classes)
        if input_channels!=3:self.features.conv0=nn.Conv2d(input_channels,2*growth_rate,7,2,3,bias=False);nn.init.kaiming_normal_(self.features.conv0.weight)


class PreActResNet(nn.Module):
    """Four-stage full-preactivation ResNet with final BN/ReLU and global pool.

    ImageNet 7x7/stride2/maxpool stem; no stem BN/ReLU, block preactivation
    supplies it. base_channels=64 default; custom widths are explicit variants.
    Bottlenecks use stride on their 3x3, not a claimed exact paper checkpoint.
    """
    def __init__(self, depth: int = 18, input_channels: int = 3, num_classes: int = 1000,
                 base_channels: int = 64, norm: str | None = 'batch') -> None:
        super().__init__()
        validate_family('preact_resnet',dict(depth=depth,input_channels=input_channels,num_classes=num_classes,base_channels=base_channels,norm=norm))
        self.stem=nn.Sequential(nn.Conv2d(input_channels,base_channels,7,2,3,bias=False),nn.MaxPool2d(3,2,1))
        stages=[];previous=base_channels;exp=1 if depth<50 else 4
        for i,count in enumerate(RESNET_LAYERS[depth]):
            width=base_channels*2**i*exp;blocks=[]
            for j in range(count):
                blocks.append(PreActivationBlock(previous,width,2 if i and j==0 else 1,norm,exp==4));previous=width
            stages.append(nn.Sequential(*blocks))
        self.stages=nn.Sequential(*stages)
        self.final=nn.Sequential(normalization(norm)(previous),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.classifier=nn.Linear(previous,num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.final(self.stages(self.stem(x))))


class WideResNet(nn.Module):
    """CIFAR-style 6n+4 WideResNet, 3x3 stem, widths 16k/32k/64k.

    This is a basic-block WideResNet, not torchvision wide_resnet50_2 bottlenecks.
    Projection uses activated input; optional dropout sits between convolutions.
    """
    def __init__(self, depth: int = 28, input_channels: int = 3, num_classes: int = 10,
                 widen_factor: int = 10, dropout: float = 0., norm: str | None = 'batch') -> None:
        super().__init__()
        validate_family('wide_resnet',dict(depth=depth,input_channels=input_channels,num_classes=num_classes,widen_factor=widen_factor,dropout=dropout,norm=norm))
        self.stem=nn.Conv2d(input_channels,16,3,1,1,bias=False)
        blocks=[];previous=16
        for i,width in enumerate((16*widen_factor,32*widen_factor,64*widen_factor)):
            for j in range((depth-4)//6):
                blocks.append(WideResidualBlock(previous,width,2 if i and j==0 else 1,norm,dropout));previous=width
        self.blocks=nn.Sequential(*blocks)
        self.final=nn.Sequential(normalization(norm)(previous),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.classifier=nn.Linear(previous,num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.final(self.blocks(self.stem(x))))
