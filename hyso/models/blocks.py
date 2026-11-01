"""Standalone CNN building blocks; legacy model architectures remain unchanged."""

import torch
from torch import nn

from .bcam import CBAMBlock
from .WithSE import SEBlock
from .attention import ECAAttention, CoordinateAttention

__all__ = ["ConvNormAct", "ResidualBlock"]


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def _pair(value, name, minimum=1):
    pair = (value, value) if type(value) is int else value
    if (type(pair) not in (tuple, list) or len(pair) != 2
            or any(type(item) is not int or item < minimum for item in pair)):
        raise ValueError(name + " must be an integer or a pair of integers >= " + str(minimum))
    return tuple(pair)


class ConvNormAct(nn.Module):
    """Conv2d -> normalization -> activation, with explicit spatial settings.

    kernel_size/stride accept positive integers or pairs; padding accepts
    nonnegative integers or pairs (no implicit 'same' padding). Norm is 'batch',
    'group' or None. GroupNorm uses the largest divisor of out_channels <= 8.
    Conv bias is enabled only when norm=None. Activation is 'silu', 'relu' or
    None; no in-place activation is used. Device/dtype follow normal Module.to().
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, groups=1, norm="batch", activation="silu"):
        super().__init__()
        _positive_int(in_channels, "in_channels")
        _positive_int(out_channels, "out_channels")
        _positive_int(groups, "groups")
        if in_channels % groups or out_channels % groups:
            raise ValueError("groups must divide both in_channels and out_channels")
        kernel_size = _pair(kernel_size, "kernel_size")
        stride = _pair(stride, "stride")
        padding = _pair(padding, "padding", minimum=0)
        if norm not in ("batch", "group", None):
            raise ValueError("norm must be 'batch', 'group' or None")
        if activation not in ("silu", "relu", None):
            raise ValueError("activation must be 'silu', 'relu' or None")

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=norm is None)
        if norm == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "group":
            count = min(8, out_channels)
            while out_channels % count:
                count -= 1
            self.norm = nn.GroupNorm(count, out_channels)
        else:
            self.norm = nn.Identity()
        self.activation = (nn.SiLU() if activation == "silu" else
                           nn.ReLU() if activation == "relu" else nn.Identity())

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions, optional attention, then SiLU(main + shortcut).

    The second convolution has no activation. A 1x1 normalized projection is
    used when channels change or either stride differs from 1. Spatial output
    is ceil(H/stride_h), ceil(W/stride_w) on both paths.

    attention accepts None, nn.Identity, SEBlock, CBAMBlock, ECAAttention or
    CoordinateAttention with out_channels channels. The supplied instance is registered, never
    copied or recreated in forward. Sharing that instance between blocks shares
    weights and is the caller's responsibility; use separate instances for
    independent attention. No stochastic depth or training API is included.
    """

    def __init__(self, in_channels, out_channels, stride=1, norm="batch",
                 attention=None):
        super().__init__()
        _positive_int(in_channels, "in_channels")
        _positive_int(out_channels, "out_channels")
        stride = _pair(stride, "stride")
        if attention is not None:
            if type(attention) not in (nn.Identity, SEBlock, CBAMBlock, ECAAttention, CoordinateAttention):
                raise TypeError("attention must be None, nn.Identity, SEBlock, CBAMBlock, "
                                "ECAAttention or CoordinateAttention")
            if type(attention) in (ECAAttention, CoordinateAttention) and attention.channels != out_channels:
                raise ValueError("attention channels must match out_channels")
            if type(attention) in (SEBlock, CBAMBlock):
                if (attention.fc1.in_features != out_channels
                        or attention.fc2.out_features != out_channels):
                    raise ValueError("attention channels must match out_channels")
                if type(attention) is CBAMBlock and any(
                        size % 2 == 0 for size in attention.conv_spatial.kernel_size):
                    raise ValueError("CBAM attention requires odd spatial kernels to preserve shape")

        self.conv1 = ConvNormAct(in_channels, out_channels, stride=stride, norm=norm)
        self.conv2 = ConvNormAct(out_channels, out_channels, norm=norm, activation=None)
        self.attention = nn.Identity() if attention is None else attention
        self.shortcut = (nn.Identity() if in_channels == out_channels and stride == (1, 1)
                         else ConvNormAct(in_channels, out_channels, kernel_size=1,
                                          stride=stride, padding=0, norm=norm, activation=None))
        self.activation = nn.SiLU()

    def forward(self, x):
        main = self.conv2(self.conv1(x))
        attended = self.attention(main)
        if not isinstance(attended, torch.Tensor) or attended.shape != main.shape:
            raise ValueError("attention must preserve the main path tensor shape")
        return self.activation(attended + self.shortcut(x))
