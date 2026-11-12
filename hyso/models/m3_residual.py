"""Canonical torchvision block adapters and full-preactivation residual blocks."""
import math
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck as TVBottleneck
from .blocks import _positive_int


def normalization(norm: str | None):
    """Resolve a closed normalization choice, without allocating tensors."""
    if norm == 'batch': return nn.BatchNorm2d
    if norm == 'group':
        def group(channels: int) -> nn.Module:
            return nn.GroupNorm(math.gcd(8, channels), channels)
        return group
    if norm is None: return lambda channels: nn.Identity()
    raise ValueError('norm must be batch, group or None')


def _settings(in_channels: int, out_channels: int, stride: int, norm):
    for k,v in [('in_channels',in_channels),('out_channels',out_channels),('stride',stride)]: _positive_int(v,k)
    if stride not in (1,2): raise ValueError('residual stride must be 1 or 2')
    return normalization(norm)


class BasicResidualBlock(BasicBlock):
    """ResNet v1 two 3x3/ReLU block, expansion=1, automatic projection."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, norm: str | None = 'batch') -> None:
        n = _settings(in_channels,out_channels,stride,norm)
        projection = None if in_channels==out_channels and stride==1 else nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,stride,bias=False),n(out_channels))
        super().__init__(in_channels,out_channels,stride,projection,norm_layer=n)


class Bottleneck(TVBottleneck):
    """ResNet v1.5 1x1/3x3/1x1, expansion=4; out_channels includes expansion."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 norm: str | None = 'batch', groups: int = 1, width_per_group: int = 64) -> None:
        n = _settings(in_channels,out_channels,stride,norm)
        _positive_int(groups,'groups'); _positive_int(width_per_group,'width_per_group')
        if out_channels % 4 or int(out_channels/4*(width_per_group/64))*groups < 1:
            raise ValueError('out_channels must be divisible by expansion=4; inner width must be positive')
        projection = None if in_channels==out_channels and stride==1 else nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,stride,bias=False),n(out_channels))
        super().__init__(in_channels,out_channels//4,stride,projection,groups,width_per_group,norm_layer=n)


class ResNeXtBottleneck(Bottleneck):
    """Grouped ResNeXt bottleneck; default cardinality 32, base width 4."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 norm: str | None = 'batch', groups: int = 32, width_per_group: int = 4) -> None:
        super().__init__(in_channels,out_channels,stride,norm,groups,width_per_group)


class PreActivationBlock(nn.Module):
    """BN/ReLU before each convolution, no activation after addition.

    Identity shortcut carries raw input; a projection uses the first activated
    input. bottleneck=True uses expansion=4 and stride on the 3x3 convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 norm: str | None = 'batch', bottleneck: bool = False, dropout: float = 0.) -> None:
        super().__init__()
        n = _settings(in_channels,out_channels,stride,norm)
        if type(bottleneck) is not bool or type(dropout) not in (int,float) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('bottleneck must be bool and dropout in [0,1)')
        if bottleneck and out_channels % 4: raise ValueError('bottleneck output must be divisible by 4')
        self.expansion = 4 if bottleneck else 1
        hidden = out_channels//self.expansion
        self.pre = nn.Sequential(n(in_channels),nn.ReLU())
        if bottleneck:
            self.main = nn.Sequential(nn.Conv2d(in_channels,hidden,1,bias=False), n(hidden), nn.ReLU(),
                nn.Conv2d(hidden,hidden,3,stride,1,bias=False),n(hidden),nn.ReLU(),nn.Conv2d(hidden,out_channels,1,bias=False))
        else:
            self.main = nn.Sequential(nn.Conv2d(in_channels,hidden,3,stride,1,bias=False),n(hidden),nn.ReLU(),
                nn.Dropout(dropout),nn.Conv2d(hidden,out_channels,3,1,1,bias=False))
        self.shortcut = None if in_channels==out_channels and stride==1 else nn.Conv2d(in_channels,out_channels,1,stride,bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = self.pre(x)
        return self.main(activated) + (x if self.shortcut is None else self.shortcut(activated))


class WideResidualBlock(PreActivationBlock):
    """Two preactivated 3x3 convolutions with optional inter-convolution dropout."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 norm: str | None = 'batch', dropout: float = 0.) -> None:
        super().__init__(in_channels,out_channels,stride,norm,False,dropout)
