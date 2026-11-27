"""Shared channel permutation and explicit spatial regularization.

Extracted without operator changes from the Stage 13 HYSo implementation.
Torch operators retain their upstream licenses in THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from .blocks import _positive_int
from .attention import _check_input


def probability(value: float, name: str = 'probability', *, inclusive: bool = False) -> None:
    """Validate a finite Python probability before allocating a module."""
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError(f'{name} must be finite and in [0,1]')
    if not inclusive and value == 1:
        raise ValueError(f'{name} must be less than 1')

class ChannelShuffle(nn.Module):
    """Reshape/transpose channel groups while preserving NCHW dimensions.

    Args:
        channels (int): Feature channels C; gates and normalization preserve this channel count.
            Positive integer; the Registry/Builder profile additionally caps it at 8192. Required.
        groups (int): Channel partitions; must divide both convolution channel counts. ChannelShuffle
            requires channels % groups == 0. Positive integer; the Registry/Builder profile additionally
            caps it at 8192. Default: 2.

    Returns:
        A ChannelShuffle module with the declared geometry.

    Raises:
        ValueError, TypeError: Constructor channel, shape or option validation fails; native PyTorch
        errors propagate.

    Notes:
        Floating NCHW with matching channels.
        Same NCHW shape; validated channel partition.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        ChannelShuffle(channels)

    HYSo Guide: components.sampling
    Offline: hyso docs show components.sampling --lang en
    Open: hyso docs open components.sampling --lang en
    """
    def __init__(self, channels: int, groups: int = 2) -> None:
        """Initialize ChannelShuffle with the supplied settings.

        Args:
            channels (int): Feature channels C; gates and normalization preserve this channel count.
                Positive integer; the Registry/Builder profile additionally caps it at 8192. Required.
            groups (int): Channel partitions; must divide both convolution channel counts. ChannelShuffle
                requires channels % groups == 0. Positive integer; the Registry/Builder profile additionally
                caps it at 8192. Default: 2.

        Returns:
            None; initializes the module.

        Raises:
            ValueError, TypeError: Constructor channel, shape or option validation fails; native PyTorch
            errors propagate.

        Notes:
            Floating NCHW with matching channels.
            Same NCHW shape; validated channel partition.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            ChannelShuffle(channels)

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        super().__init__()
        _positive_int(channels, 'channels')
        _positive_int(groups, 'groups')
        if channels % groups:
            raise ValueError('groups must divide channels')
        self.channels = channels
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply this module to inputs with its declared shape, dtype and device.

        Args:
            x (torch.Tensor): Input Tensor(s) obeying the model/block shape, dtype and device contract.
                Required.

        Returns:
            Native Tensor output(s) of this model/block; this call performs the forward computation.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.forward(x)

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        _check_input(x, self.channels)
        n, c, h, w = x.shape
        return x.reshape(n, self.groups, c//self.groups, h, w).transpose(1, 2).reshape(n, c, h, w)

class DropBlock2D(nn.Module):
    """Explicit contiguous spatial regularization with per-tensor mask normalization.

    Args:
        p (float): Probability of dropping elements, channels or paths during training; evaluation
            disables masking. Number in [0,1]. Default: 0.1.
        block_size (int): Requested spatial square side for contiguous DropBlock masks; runtime clips it
            to min(block_size,H,W). Positive integer; the Registry/Builder profile additionally caps it
            at 8192. Default: 3.

    Returns:
        A DropBlock2D module with the declared geometry.

    Raises:
        ValueError, TypeError: Constructor channel, shape or option validation fails; native PyTorch
        errors propagate.

    Notes:
        Nonempty floating NCHW.
        Same shape and dtype; explicit training-only masking.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        DropBlock2D()

    HYSo Guide: components.regularization
    Offline: hyso docs show components.regularization --lang en
    Open: hyso docs open components.regularization --lang en
    """
    def __init__(self, p: float = .1, block_size: int = 3) -> None:
        """Initialize DropBlock2D with the supplied settings.

        Args:
            p (float): Probability of dropping elements, channels or paths during training; evaluation
                disables masking. Number in [0,1]. Default: 0.1.
            block_size (int): Requested spatial square side for contiguous DropBlock masks; runtime clips it
                to min(block_size,H,W). Positive integer; the Registry/Builder profile additionally caps it
                at 8192. Default: 3.

        Returns:
            None; initializes the module.

        Raises:
            ValueError, TypeError: Constructor channel, shape or option validation fails; native PyTorch
            errors propagate.

        Notes:
            Nonempty floating NCHW.
            Same shape and dtype; explicit training-only masking.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            DropBlock2D()

        HYSo Guide: components.regularization
        Offline: hyso docs show components.regularization --lang en
        Open: hyso docs open components.regularization --lang en
        """
        super().__init__()
        probability(p, 'p', inclusive=True)
        _positive_int(block_size, 'block_size')
        self.p = p
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply this module to inputs with its declared shape, dtype and device.

        Args:
            x (torch.Tensor): Input Tensor(s) obeying the model/block shape, dtype and device contract.
                Required.

        Returns:
            Native Tensor output(s) of this model/block; this call performs the forward computation.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.forward(x)

        HYSo Guide: components.regularization
        Offline: hyso docs show components.regularization --lang en
        Open: hyso docs open components.regularization --lang en
        """
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
