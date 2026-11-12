# Adapters use torchvision, Copyright (c) Soumith Chintala 2016, BSD-3-Clause.
# Ghost portions adapted from Huawei Technologies Co., Ltd. (c) 2020, Apache-2.0.
# HYSo changes (2026) add standalone validation and torchvision operator adapters.
# Full upstream licenses and notices are retained in THIRD_PARTY_NOTICES.md.
"""Independent mobile/edge operators with closed NCHW constructor contracts."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.mobilenetv2 import InvertedResidual as TVInvertedResidual
from torchvision.models.efficientnet import (
    MBConv as TVMBConv, FusedMBConv as TVFusedMBConv, MBConvConfig, FusedMBConvConfig,
)
from torchvision.models.shufflenetv2 import InvertedResidual as TVShuffleUnit
from torchvision.ops import Conv2dNormActivation, SqueezeExcitation

from .blocks import _positive_int
from .attention import _check_input


def probability(value: float, name: str = 'probability', *, inclusive: bool = False) -> None:
    """Validate a finite Python probability before allocating a module."""
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError(f'{name} must be finite and in [0,1]')
    if not inclusive and value == 1:
        raise ValueError(f'{name} must be less than 1')


def block_channels(in_channels: int, out_channels: int, stride: int) -> None:
    """Check positive channel counts and the supported downsampling strides."""
    _positive_int(in_channels, 'in_channels')
    _positive_int(out_channels, 'out_channels')
    if type(stride) is not int or stride not in (1, 2):
        raise ValueError('stride must be 1 or 2')


def expansion_ratio(value: float) -> None:
    """Validate a bounded positive expansion used by mobile blocks."""
    if type(value) not in (int, float) or not 1 <= value <= 16:
        raise ValueError('expand_ratio must be finite and in [1,16]')


def odd_kernel(value: int, name: str = 'kernel_size') -> None:
    """Validate the odd kernels supported by these standalone blocks."""
    if type(value) is not int or value not in (1, 3, 5, 7):
        raise ValueError(f'{name} must be 1,3,5 or 7')


class InvertedResidual(TVInvertedResidual):
    """MobileNetV2 expansion/ReLU6, depthwise and linear projection adapter.

    Residual addition requires stride=1 and equal input/output channels.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 expand_ratio: float = 6.) -> None:
        block_channels(in_channels, out_channels, stride)
        expansion_ratio(expand_ratio)
        super().__init__(in_channels, out_channels, stride, expand_ratio)


class MBConv(TVMBConv):
    """Torchvision SiLU MBConv with input-width/4 SE and row stochastic depth.

    Expanded width follows torchvision's divisor-eight rounding convention.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 expand_ratio: float = 6., kernel_size: int = 3,
                 stochastic_depth: float = 0.) -> None:
        block_channels(in_channels, out_channels, stride)
        expansion_ratio(expand_ratio)
        odd_kernel(kernel_size)
        probability(stochastic_depth, 'stochastic_depth')
        config = MBConvConfig(expand_ratio, kernel_size, stride, in_channels, out_channels, 1)
        # Standalone adapters preserve the caller's exact external channel counts.
        config.input_channels = in_channels
        config.out_channels = out_channels
        super().__init__(config, stochastic_depth, nn.BatchNorm2d)


class FusedMBConv(TVFusedMBConv):
    """Torchvision fused spatial expansion and linear projection with SiLU."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 expand_ratio: float = 4., kernel_size: int = 3,
                 stochastic_depth: float = 0.) -> None:
        block_channels(in_channels, out_channels, stride)
        expansion_ratio(expand_ratio)
        odd_kernel(kernel_size)
        probability(stochastic_depth, 'stochastic_depth')
        config = FusedMBConvConfig(expand_ratio, kernel_size, stride, in_channels, out_channels, 1)
        super().__init__(config, stochastic_depth, nn.BatchNorm2d)


class GhostConv(nn.Module):
    """GhostNetV1 primary features and depthwise cheap features, concatenated.

    ratio>=2 selects ceil(out/ratio) primary channels. The final concatenation
    is sliced to exactly out_channels. relu=False preserves a linear projection.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 kernel_size: int = 1, ratio: int = 2, cheap_kernel: int = 3,
                 relu: bool = True) -> None:
        super().__init__()
        block_channels(in_channels, out_channels, stride)
        odd_kernel(kernel_size)
        odd_kernel(cheap_kernel, 'cheap_kernel')
        if type(ratio) is not int or not 2 <= ratio <= 8 or type(relu) is not bool:
            raise ValueError('ratio must be 2..8 and relu must be bool')
        self.in_channels = in_channels
        self.out_channels = out_channels
        primary = math.ceil(out_channels / ratio)
        activation = nn.ReLU if relu else None
        self.primary = Conv2dNormActivation(in_channels, primary, kernel_size, stride,
                                           activation_layer=activation)
        self.cheap = Conv2dNormActivation(primary, primary*(ratio-1), cheap_kernel,
                                         groups=primary, activation_layer=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_input(x, self.in_channels)
        primary = self.primary(x)
        return torch.cat((primary, self.cheap(primary)), dim=1)[:, :self.out_channels]


class GhostBottleneck(nn.Module):
    """GhostNetV1 two Ghost modules, optional depthwise downsample and hard-gated SE.

    The shortcut uses depthwise plus pointwise convolution on shape changes.
    hidden_channels is explicit so published expansion widths can be preserved.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int,
                 stride: int = 1, kernel_size: int = 3, se_ratio: float = 0.) -> None:
        super().__init__()
        block_channels(in_channels, out_channels, stride)
        _positive_int(hidden_channels, 'hidden_channels')
        odd_kernel(kernel_size)
        if type(se_ratio) not in (int, float) or not 0 <= se_ratio <= 1:
            raise ValueError('se_ratio must be finite and in [0,1]')
        from torchvision.models._utils import _make_divisible
        self.in_channels = in_channels
        self.expand = GhostConv(in_channels, hidden_channels)
        self.depthwise = (Conv2dNormActivation(hidden_channels, hidden_channels, kernel_size,
                          stride, groups=hidden_channels, activation_layer=None)
                          if stride == 2 else nn.Identity())
        self.se = (SqueezeExcitation(hidden_channels, _make_divisible(hidden_channels*se_ratio, 4),
                   activation=nn.ReLU, scale_activation=nn.Hardsigmoid) if se_ratio else nn.Identity())
        self.project = GhostConv(hidden_channels, out_channels, relu=False)
        self.shortcut = (nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            Conv2dNormActivation(in_channels, in_channels, kernel_size, stride,
                                 groups=in_channels, activation_layer=None),
            Conv2dNormActivation(in_channels, out_channels, 1, activation_layer=None)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_input(x, self.in_channels)
        return self.project(self.se(self.depthwise(self.expand(x)))) + self.shortcut(x)


class ChannelShuffle(nn.Module):
    """Reshape/transpose channel groups while preserving NCHW dimensions."""
    def __init__(self, channels: int, groups: int = 2) -> None:
        super().__init__()
        _positive_int(channels, 'channels')
        _positive_int(groups, 'groups')
        if channels % groups:
            raise ValueError('groups must divide channels')
        self.channels = channels
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_input(x, self.channels)
        n, c, h, w = x.shape
        return x.reshape(n, self.groups, c//self.groups, h, w).transpose(1, 2).reshape(n, c, h, w)


class ShuffleUnit(TVShuffleUnit):
    """Torchvision ShuffleNetV2 split/concat/shuffle unit with even output width."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        block_channels(in_channels, out_channels, stride)
        if out_channels % 2 or (stride == 1 and in_channels != out_channels):
            raise ValueError('even out_channels required, with equal channels for stride=1')
        super().__init__(in_channels, out_channels, stride)




class DropBlock2D(nn.Module):
    """Explicit contiguous spatial regularization with per-tensor mask normalization.

    block_size is clipped to H/W, including even spatial extents. Valid top-left
    seed positions spread into exact square blocks. gamma is clipped to one.
    p=1 yields zeros, p=0 and eval return the input. An all-dropped mask yields
    finite zeros. Half-precision mask statistics accumulate in float32.
    """
    def __init__(self, p: float = .1, block_size: int = 3) -> None:
        super().__init__()
        probability(p, 'p', inclusive=True)
        _positive_int(block_size, 'block_size')
        self.p = p
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or not x.is_floating_point() or min(x.shape) < 1:
            raise ValueError('DropBlock2D requires a nonempty floating NCHW tensor')
        if not self.training or self.p == 0:
            return x
        if self.p == 1:
            return x * 0
        n, c, h, w = x.shape
        size = min(self.block_size, h, w)
        gamma = min(1., self.p*h*w/(size**2*(h-size+1)*(w-size+1)))
        seeds = (torch.rand((n, c, h-size+1, w-size+1), device=x.device) < gamma).float()
        padded = F.pad(seeds, (size-1, size-1, size-1, size-1))
        mask = 1 - F.max_pool2d(padded, size, stride=1)
        scale = mask.numel() / mask.sum().clamp_min(1)
        value = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        return (value * mask * scale).to(x.dtype)
