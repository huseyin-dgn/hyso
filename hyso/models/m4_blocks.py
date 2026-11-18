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
from .regularization import ChannelShuffle, DropBlock2D, probability




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

    Args:
        in_channels (int): Expected input feature-channel count. Required.
        out_channels (int): Number of output feature channels. Required.
        stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
        expand_ratio (float): Internal channel expansion ratio. Default: 6.0.

    Returns:
        An instance of InvertedResidual with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        InvertedResidual(in_channels, out_channels)

    HYSo Guide: components.mobile
    Offline: hyso docs show components.mobile --lang en
    Open: hyso docs open components.mobile --lang en
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 expand_ratio: float = 6.) -> None:
        """Initialize InvertedResidual with the supplied settings.

        Args:
            in_channels (int): Expected input feature-channel count. Required.
            out_channels (int): Number of output feature channels. Required.
            stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
            expand_ratio (float): Expansion ratio in [1,16]. Default: 6.0.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            InvertedResidual(in_channels, out_channels)

        HYSo Guide: components.mobile
        Offline: hyso docs show components.mobile --lang en
        Open: hyso docs open components.mobile --lang en
        """
        block_channels(in_channels, out_channels, stride)
        expansion_ratio(expand_ratio)
        super().__init__(in_channels, out_channels, stride, expand_ratio)


class MBConv(TVMBConv):
    """Torchvision SiLU MBConv with input-width/4 SE and row stochastic depth.

    Expanded width follows torchvision's divisor-eight rounding convention.

    Args:
        in_channels (int): Expected input feature-channel count. Required.
        out_channels (int): Number of output feature channels. Required.
        stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
        expand_ratio (float): Internal channel expansion ratio. Default: 6.0.
        kernel_size (int): Spatial filter/window extent; accepted integer/pair and oddness rules depend on the component. Default: 3.
        stochastic_depth (float): Probability/schedule for dropping residual paths during training. Default: 0.0.

    Returns:
        An instance of MBConv with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        MBConv(in_channels, out_channels)

    HYSo Guide: components.mobile
    Offline: hyso docs show components.mobile --lang en
    Open: hyso docs open components.mobile --lang en
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 expand_ratio: float = 6., kernel_size: int = 3,
                 stochastic_depth: float = 0.) -> None:
        """Initialize MBConv with the supplied settings.

        Args:
            in_channels (int): Expected input feature-channel count. Required.
            out_channels (int): Number of output feature channels. Required.
            stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
            expand_ratio (float): Expansion ratio in [1,16]. Default: 6.0.
            kernel_size (int): Spatial filter/window extent; accepted integer/pair and oddness rules depend on the component. Default: 3.
            stochastic_depth (float): stochastic depth Default: 0.0.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            MBConv(in_channels, out_channels)

        HYSo Guide: components.mobile
        Offline: hyso docs show components.mobile --lang en
        Open: hyso docs open components.mobile --lang en
        """
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
    """Torchvision fused spatial expansion and linear projection with SiLU.

    Args:
        in_channels (int): Expected input feature-channel count. Required.
        out_channels (int): Number of output feature channels. Required.
        stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
        expand_ratio (float): Internal channel expansion ratio. Default: 4.0.
        kernel_size (int): Spatial filter/window extent; accepted integer/pair and oddness rules depend on the component. Default: 3.
        stochastic_depth (float): Probability/schedule for dropping residual paths during training. Default: 0.0.

    Returns:
        An instance of FusedMBConv with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        FusedMBConv(in_channels, out_channels)

    HYSo Guide: components.mobile
    Offline: hyso docs show components.mobile --lang en
    Open: hyso docs open components.mobile --lang en
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 expand_ratio: float = 4., kernel_size: int = 3,
                 stochastic_depth: float = 0.) -> None:
        """Initialize FusedMBConv with the supplied settings.

        Args:
            in_channels (int): Expected input feature-channel count. Required.
            out_channels (int): Number of output feature channels. Required.
            stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
            expand_ratio (float): expand ratio Default: 4.0.
            kernel_size (int): Spatial filter/window extent; accepted integer/pair and oddness rules depend on the component. Default: 3.
            stochastic_depth (float): stochastic depth Default: 0.0.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            FusedMBConv(in_channels, out_channels)

        HYSo Guide: components.mobile
        Offline: hyso docs show components.mobile --lang en
        Open: hyso docs open components.mobile --lang en
        """
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

    Args:
        in_channels (int): Expected input feature-channel count. Required.
        out_channels (int): Number of output feature channels. Required.
        stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
        kernel_size (int): Spatial filter/window extent; accepted integer/pair and oddness rules depend on the component. Default: 1.
        ratio (int): Feature expansion/split ratio used by this block. Default: 2.
        cheap_kernel (int): Spatial kernel of the inexpensive feature-generation branch. Default: 3.
        relu (bool): Enable the block's ReLU activation. Default: True.

    Returns:
        An instance of GhostConv with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        GhostConv(in_channels, out_channels)

    HYSo Guide: components.mobile
    Offline: hyso docs show components.mobile --lang en
    Open: hyso docs open components.mobile --lang en
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 kernel_size: int = 1, ratio: int = 2, cheap_kernel: int = 3,
                 relu: bool = True) -> None:
        """Initialize GhostConv with the supplied settings.

        Args:
            in_channels (int): Expected input feature-channel count. Required.
            out_channels (int): Number of output feature channels. Required.
            stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
            kernel_size (int): Spatial filter/window extent; accepted integer/pair and oddness rules depend on the component. Default: 1.
            ratio (int): ratio Default: 2.
            cheap_kernel (int): cheap kernel Default: 3.
            relu (bool): relu Default: True.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            GhostConv(in_channels, out_channels)

        HYSo Guide: components.mobile
        Offline: hyso docs show components.mobile --lang en
        Open: hyso docs open components.mobile --lang en
        """
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
        """Apply this module to inputs with its declared shape, dtype and device.

        Args:
            x (torch.Tensor): Input Tensor(s) obeying the model/block shape, dtype and device contract. Required.

        Returns:
            Native Tensor output(s) of this model/block; this call performs the forward computation.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.forward(x)

        HYSo Guide: components.mobile
        Offline: hyso docs show components.mobile --lang en
        Open: hyso docs open components.mobile --lang en
        """
        _check_input(x, self.in_channels)
        primary = self.primary(x)
        return torch.cat((primary, self.cheap(primary)), dim=1)[:, :self.out_channels]


class GhostBottleneck(nn.Module):
    """GhostNetV1 two Ghost modules, optional depthwise downsample and hard-gated SE.

    The shortcut uses depthwise plus pointwise convolution on shape changes.
    hidden_channels is explicit so published expansion widths can be preserved.

    Args:
        in_channels (int): Expected input feature-channel count. Required.
        out_channels (int): Number of output feature channels. Required.
        hidden_channels (int): Internal feature width. Required.
        stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
        kernel_size (int): Spatial filter/window extent; accepted integer/pair and oddness rules depend on the component. Default: 3.
        se_ratio (float): Squeeze-and-excitation reduction setting. Default: 0.0.

    Returns:
        An instance of GhostBottleneck with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        GhostBottleneck(in_channels, out_channels, hidden_channels)

    HYSo Guide: components.mobile
    Offline: hyso docs show components.mobile --lang en
    Open: hyso docs open components.mobile --lang en
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int,
                 stride: int = 1, kernel_size: int = 3, se_ratio: float = 0.) -> None:
        """Initialize GhostBottleneck with the supplied settings.

        Args:
            in_channels (int): Expected input feature-channel count. Required.
            out_channels (int): Number of output feature channels. Required.
            hidden_channels (int): hidden channels Required.
            stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
            kernel_size (int): Spatial filter/window extent; accepted integer/pair and oddness rules depend on the component. Default: 3.
            se_ratio (float): se ratio Default: 0.0.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            GhostBottleneck(in_channels, out_channels, hidden_channels)

        HYSo Guide: components.mobile
        Offline: hyso docs show components.mobile --lang en
        Open: hyso docs open components.mobile --lang en
        """
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
        """Apply this module to inputs with its declared shape, dtype and device.

        Args:
            x (torch.Tensor): Input Tensor(s) obeying the model/block shape, dtype and device contract. Required.

        Returns:
            Native Tensor output(s) of this model/block; this call performs the forward computation.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.forward(x)

        HYSo Guide: components.mobile
        Offline: hyso docs show components.mobile --lang en
        Open: hyso docs open components.mobile --lang en
        """
        _check_input(x, self.in_channels)
        return self.project(self.se(self.depthwise(self.expand(x)))) + self.shortcut(x)




class ShuffleUnit(TVShuffleUnit):
    """Torchvision ShuffleNetV2 split/concat/shuffle unit with even output width.

    Args:
        in_channels (int): Expected input feature-channel count. Required.
        out_channels (int): Number of output feature channels. Required.
        stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.

    Returns:
        An instance of ShuffleUnit with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        ShuffleUnit(in_channels, out_channels)

    HYSo Guide: components.mobile
    Offline: hyso docs show components.mobile --lang en
    Open: hyso docs open components.mobile --lang en
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize ShuffleUnit with the supplied settings.

        Args:
            in_channels (int): Expected input feature-channel count. Required.
            out_channels (int): Number of output feature channels. Required.
            stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            ShuffleUnit(in_channels, out_channels)

        HYSo Guide: components.mobile
        Offline: hyso docs show components.mobile --lang en
        Open: hyso docs open components.mobile --lang en
        """
        block_channels(in_channels, out_channels, stride)
        if out_channels % 2 or (stride == 1 and in_channels != out_channels):
            raise ValueError('even out_channels required, with equal channels for stride=1')
        super().__init__(in_channels, out_channels, stride)
