"""Standalone CNN building blocks; legacy model architectures remain unchanged."""
import typing as _typing

import torch
from torch import nn

from .bcam import CBAMBlock
from .WithSE import SEBlock
from .attention import ECAAttention, CoordinateAttention
from ._profile import DEVELOPMENT

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

    Args:
        in_channels (_typing.Any): Expected input feature-channel count. Required.
        out_channels (_typing.Any): Number of output feature channels. Required.
        kernel_size (int): Spatial filter/window extent; accepted integer/pair and oddness rules depend on the component. Default: 3.
        stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
        padding (int): Spatial padding under the closed registered component contract. Default: 1.
        groups (int): Convolution/group count; divisibility requirements remain unchanged. Default: 1.
        norm (str): Supported normalization choice; native training-statistics requirements still apply. Default: 'batch'.
        activation (str): Supported activation family. Default: 'silu'.

    Returns:
        An instance of ConvNormAct with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        ConvNormAct(in_channels, out_channels)

    HYSo Guide: components.convolution
    Offline: hyso docs show components.convolution --lang en
    Open: hyso docs open components.convolution --lang en
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int | tuple[int, int] | list[int]'=3, stride: 'int | tuple[int, int] | list[int]'=1,
                 padding: 'int | tuple[int, int] | list[int]'=1, groups: 'int'=1, norm: _typing.Literal['batch', 'group', None]="batch", activation: _typing.Literal['silu', 'relu', None]="silu") -> 'None':
        """Initialize ConvNormAct with the supplied settings.

        Args:
            in_channels (int): Expected input feature-channel count. Required.
            out_channels (int): Number of output feature channels. Required.
            kernel_size (int | tuple[int, int] | list[int]): Spatial filter/window extent; accepted integer/pair and oddness rules depend on the component. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step; must produce a valid output shape. Default: 1.
            padding (int | tuple[int, int] | list[int]): Spatial padding under the closed registered component contract. Default: 1.
            groups (int): Convolution/group count; divisibility requirements remain unchanged. Default: 1.
            norm (str): Supported normalization choice; native training-statistics requirements still apply. Default: 'batch'.
            activation (str): activation Default: 'silu'.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            ConvNormAct(in_channels, out_channels)

        HYSo Guide: components.convolution
        Offline: hyso docs show components.convolution --lang en
        Open: hyso docs open components.convolution --lang en
        """
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

    def forward(self, x: '_typing.Any') -> '_typing.Any':
        """Apply this module to inputs with its declared shape, dtype and device.

        Args:
            x (_typing.Any): Input Tensor(s) obeying the model/block shape, dtype and device contract. Required.

        Returns:
            Native Tensor output(s) of this model/block; this call performs the forward computation.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.forward(x)

        HYSo Guide: components.convolution
        Offline: hyso docs show components.convolution --lang en
        Open: hyso docs open components.convolution --lang en
        """
        return self.activation(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions, optional attention, then SiLU(main + shortcut).

    The second convolution has no activation. A 1x1 normalized projection is
    used when channels change or either stride differs from 1. Spatial output
    is ceil(H/stride_h), ceil(W/stride_w) on both paths.

    attention accepts None, nn.Identity, SEBlock, CBAMBlock, ECAAttention or
    CoordinateAttention, SimAM, TripletAttention or SelectiveKernelAttention with out_channels channels. The supplied instance is registered, never
    copied or recreated in forward. Sharing that instance between blocks shares
    weights and is the caller's responsibility; use separate instances for
    independent attention. No stochastic depth or training API is included.

    Args:
        in_channels (_typing.Any): Expected input feature-channel count. Required.
        out_channels (_typing.Any): Number of output feature channels. Required.
        stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
        norm (str): Supported normalization choice; native training-statistics requirements still apply. Default: 'batch'.
        attention (_typing.Any): Supported attention object/configuration for this constructor; no external plugin is inferred. Default: None.

    Returns:
        An instance of ResidualBlock with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        ResidualBlock(in_channels, out_channels)

    HYSo Guide: components.residual
    Offline: hyso docs show components.residual --lang en
    Open: hyso docs open components.residual --lang en
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', stride: 'int | tuple[int, int] | list[int]'=1, norm: _typing.Literal['batch', 'group', None]="batch",
                 attention: '_typing.Any'=None) -> 'None':
        """Initialize ResidualBlock with the supplied settings.

        Args:
            in_channels (int): Expected input feature-channel count. Required.
            out_channels (int): Number of output feature channels. Required.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step; must produce a valid output shape. Default: 1.
            norm (str): Supported normalization choice; native training-statistics requirements still apply. Default: 'batch'.
            attention (_typing.Any): Supported attention object/configuration for this constructor; no external plugin is inferred. Default: None.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            ResidualBlock(in_channels, out_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        super().__init__()
        _positive_int(in_channels, "in_channels")
        _positive_int(out_channels, "out_channels")
        stride = _pair(stride, "stride")
        development_attention = ()
        if DEVELOPMENT:
            from .m3_attention import SimAM, TripletAttention, SelectiveKernelAttention
            development_attention = (SimAM, TripletAttention, SelectiveKernelAttention)
        if attention is not None:
            if type(attention) not in development_attention + (nn.Identity, SEBlock, CBAMBlock, ECAAttention, CoordinateAttention):
                raise TypeError("attention must be None, nn.Identity, SEBlock, CBAMBlock, "
                                "ECAAttention, CoordinateAttention, SimAM, TripletAttention or SelectiveKernelAttention")
            if type(attention) in development_attention + (ECAAttention, CoordinateAttention) and attention.channels != out_channels:
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

    def forward(self, x: '_typing.Any') -> '_typing.Any':
        """Apply this module to inputs with its declared shape, dtype and device.

        Args:
            x (_typing.Any): Input Tensor(s) obeying the model/block shape, dtype and device contract. Required.

        Returns:
            Native Tensor output(s) of this model/block; this call performs the forward computation.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.forward(x)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        main = self.conv2(self.conv1(x))
        attended = self.attention(main)
        if not isinstance(attended, torch.Tensor) or attended.shape != main.shape:
            raise ValueError("attention must preserve the main path tensor shape")
        return self.activation(attended + self.shortcut(x))
