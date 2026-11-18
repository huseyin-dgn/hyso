"""Independent NCHW attention modules; no legacy model or training changes.

Mechanisms: https://arxiv.org/abs/1910.03151 (ECA) and
https://arxiv.org/abs/2103.02907 (Coordinate Attention).
"""
import typing as _typing

import torch
from torch import nn

__all__ = ["ECAAttention", "CoordinateAttention"]


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def _check_input(x, channels):
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise ValueError("attention input must be an NCHW tensor")
    if x.shape[1] != channels or any(size <= 0 for size in x.shape):
        raise ValueError("attention input must have matching channels and nonempty dimensions")


class ECAAttention(nn.Module):
    """Global spatial mean -> channel-axis Conv1d -> sigmoid reweighting.

    kernel_size is an explicit positive odd integer, default 3; there is no
    implicit adaptive kernel selection or channel reduction. Padding preserves
    the C-element descriptor even for C < kernel_size. Parameters are created
    once; device/dtype follow normal Module.to(), including CPU float32/float64.

    Args:
        channels (_typing.Any): Feature-channel count required by this block. Required.
        kernel_size (int): Spatial filter/window extent; accepted integer/pair and oddness rules depend on the component. Default: 3.

    Returns:
        An instance of ECAAttention with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        ECAAttention(channels)

    HYSo Guide: components.attention
    Offline: hyso docs show components.attention --lang en
    Open: hyso docs open components.attention --lang en
    """

    def __init__(self, channels: 'int', kernel_size: 'int'=3) -> 'None':
        """Initialize ECAAttention with the supplied settings.

        Args:
            channels (int): Feature-channel count required by this block. Required.
            kernel_size (int): Spatial filter/window extent; accepted integer/pair and oddness rules depend on the component. Default: 3.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            ECAAttention(channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        super().__init__()
        self.channels = _positive_int(channels, "channels")
        self.kernel_size = _positive_int(kernel_size, "kernel_size")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

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

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        _check_input(x, self.channels)
        descriptor = self.pool(x).squeeze(-1).transpose(1, 2)  # B,1,C
        gate = self.sigmoid(self.conv(descriptor)).transpose(1, 2).unsqueeze(-1)
        return x * gate  # gate: B,C,1,1


class CoordinateAttention(nn.Module):
    """Separate height/width descriptors and gates, preserving NCHW shape.

    The shared width is max(8, channels // reduction), as in the authors'
    implementation. A 1x1 convolution, BatchNorm2d and h-swish transform the
    concatenated H+W descriptors. Gate convolutions preserve channels. For
    B=1,H=W=1 the shared BN still sees two spatial values (H+W=2); no small-batch
    fallback or frozen statistics are used. Train/eval retain normal BN semantics.

    Args:
        channels (_typing.Any): Feature-channel count required by this block. Required.
        reduction (int): Channel reduction ratio used by the attention bottleneck. Default: 32.

    Returns:
        An instance of CoordinateAttention with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        CoordinateAttention(channels)

    HYSo Guide: components.attention
    Offline: hyso docs show components.attention --lang en
    Open: hyso docs open components.attention --lang en
    """

    def __init__(self, channels: 'int', reduction: 'int'=32) -> 'None':
        """Initialize CoordinateAttention with the supplied settings.

        Args:
            channels (int): Feature-channel count required by this block. Required.
            reduction (int): Channel reduction ratio used by the attention bottleneck. Default: 32.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            CoordinateAttention(channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        super().__init__()
        self.channels = _positive_int(channels, "channels")
        self.reduction = _positive_int(reduction, "reduction")
        hidden = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.bn = nn.BatchNorm2d(hidden)
        self.activation = nn.Hardswish()
        self.conv_h = nn.Conv2d(hidden, channels, 1)
        self.conv_w = nn.Conv2d(hidden, channels, 1)

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

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        _check_input(x, self.channels)
        height, width = x.shape[2:]
        along_h = self.pool_h(x)                         # B,C,H,1
        along_w = self.pool_w(x).transpose(2, 3)         # B,C,W,1
        shared = self.activation(self.bn(self.conv1(torch.cat((along_h, along_w), dim=2))))
        part_h, part_w = torch.split(shared, (height, width), dim=2)
        gate_h = torch.sigmoid(self.conv_h(part_h))
        gate_w = torch.sigmoid(self.conv_w(part_w.transpose(2, 3)))
        return x * gate_w * gate_h
