"""Canonical torchvision block adapters and full-preactivation residual blocks."""
import typing as _typing
import math
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck as TVBottleneck
from .blocks import _positive_int


def normalization(norm: _typing.Literal['batch', 'group', None]):
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
    """ResNet v1 two 3x3/ReLU block, expansion=1, automatic projection.

    Args:
        in_channels (int): Expected input feature-channel count. Required.
        out_channels (int): Number of output feature channels. Required.
        stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
        norm (str | None): Supported normalization choice; native training-statistics requirements still apply. Default: 'batch'.

    Returns:
        An instance of BasicResidualBlock with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        BasicResidualBlock(in_channels, out_channels)

    HYSo Guide: components.residual
    Offline: hyso docs show components.residual --lang en
    Open: hyso docs open components.residual --lang en
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, norm: _typing.Literal['batch', 'group', None] = 'batch') -> None:
        """Initialize BasicResidualBlock with the supplied settings.

        Args:
            in_channels (int): Expected input feature-channel count. Required.
            out_channels (int): Number of output feature channels. Required.
            stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
            norm (str | None): Supported normalization choice; native training-statistics requirements still apply. Default: 'batch'.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            BasicResidualBlock(in_channels, out_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        n = _settings(in_channels,out_channels,stride,norm)
        projection = None if in_channels==out_channels and stride==1 else nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,stride,bias=False),n(out_channels))
        super().__init__(in_channels,out_channels,stride,projection,norm_layer=n)


class Bottleneck(TVBottleneck):
    """ResNet v1.5 1x1/3x3/1x1, expansion=4; out_channels includes expansion.

    Args:
        in_channels (int): Expected input feature-channel count. Required.
        out_channels (int): Number of output feature channels. Required.
        stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
        norm (str | None): Supported normalization choice; native training-statistics requirements still apply. Default: 'batch'.
        groups (int): Convolution/group count; divisibility requirements remain unchanged. Default: 1.
        width_per_group (int): Feature width within each convolution group. Default: 64.

    Returns:
        An instance of Bottleneck with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        Bottleneck(in_channels, out_channels)

    HYSo Guide: components.residual
    Offline: hyso docs show components.residual --lang en
    Open: hyso docs open components.residual --lang en
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 norm: _typing.Literal['batch', 'group', None] = 'batch', groups: int = 1, width_per_group: int = 64) -> None:
        """Initialize Bottleneck with the supplied settings.

        Args:
            in_channels (int): Expected input feature-channel count. Required.
            out_channels (int): Number of output feature channels. Required.
            stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
            norm (str | None): Supported normalization choice; native training-statistics requirements still apply. Default: 'batch'.
            groups (int): Convolution/group count; divisibility requirements remain unchanged. Default: 1.
            width_per_group (int): width per group Default: 64.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            Bottleneck(in_channels, out_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        n = _settings(in_channels,out_channels,stride,norm)
        _positive_int(groups,'groups'); _positive_int(width_per_group,'width_per_group')
        if out_channels % 4 or int(out_channels/4*(width_per_group/64))*groups < 1:
            raise ValueError('out_channels must be divisible by expansion=4; inner width must be positive')
        projection = None if in_channels==out_channels and stride==1 else nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,stride,bias=False),n(out_channels))
        super().__init__(in_channels,out_channels//4,stride,projection,groups,width_per_group,norm_layer=n)


class ResNeXtBottleneck(Bottleneck):
    """Grouped ResNeXt bottleneck; default cardinality 32, base width 4.

    Args:
        in_channels (int): Expected input feature-channel count. Required.
        out_channels (int): Number of output feature channels. Required.
        stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
        norm (str | None): Supported normalization choice; native training-statistics requirements still apply. Default: 'batch'.
        groups (int): Convolution/group count; divisibility requirements remain unchanged. Default: 32.
        width_per_group (int): Feature width within each convolution group. Default: 4.

    Returns:
        An instance of ResNeXtBottleneck with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        ResNeXtBottleneck(in_channels, out_channels)

    HYSo Guide: components.residual
    Offline: hyso docs show components.residual --lang en
    Open: hyso docs open components.residual --lang en
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 norm: _typing.Literal['batch', 'group', None] = 'batch', groups: int = 32, width_per_group: int = 4) -> None:
        """Initialize ResNeXtBottleneck with the supplied settings.

        Args:
            in_channels (int): Expected input feature-channel count. Required.
            out_channels (int): Number of output feature channels. Required.
            stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
            norm (str | None): Supported normalization choice; native training-statistics requirements still apply. Default: 'batch'.
            groups (int): Convolution/group count; divisibility requirements remain unchanged. Default: 32.
            width_per_group (int): width per group Default: 4.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            ResNeXtBottleneck(in_channels, out_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        super().__init__(in_channels,out_channels,stride,norm,groups,width_per_group)


class PreActivationBlock(nn.Module):
    """BN/ReLU before each convolution, no activation after addition.

    Identity shortcut carries raw input; a projection uses the first activated
    input. bottleneck=True uses expansion=4 and stride on the 3x3 convolution.

    Args:
        in_channels (int): Expected input feature-channel count. Required.
        out_channels (int): Number of output feature channels. Required.
        stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
        norm (str | None): Supported normalization choice; native training-statistics requirements still apply. Default: 'batch'.
        bottleneck (bool): Select the supported bottleneck block form. Default: False.
        dropout (float): Drop probability; this regularization is active only in training mode. Default: 0.0.

    Returns:
        An instance of PreActivationBlock with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        PreActivationBlock(in_channels, out_channels)

    HYSo Guide: components.residual
    Offline: hyso docs show components.residual --lang en
    Open: hyso docs open components.residual --lang en
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 norm: _typing.Literal['batch', 'group', None] = 'batch', bottleneck: bool = False, dropout: float = 0.) -> None:
        """Initialize PreActivationBlock with the supplied settings.

        Args:
            in_channels (int): Expected input feature-channel count. Required.
            out_channels (int): Number of output feature channels. Required.
            stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
            norm (str | None): Supported normalization choice; native training-statistics requirements still apply. Default: 'batch'.
            bottleneck (bool): bottleneck Default: False.
            dropout (float): dropout Default: 0.0.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            PreActivationBlock(in_channels, out_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
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

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        activated = self.pre(x)
        return self.main(activated) + (x if self.shortcut is None else self.shortcut(activated))


class WideResidualBlock(PreActivationBlock):
    """Two preactivated 3x3 convolutions with optional inter-convolution dropout.

    Args:
        in_channels (int): Expected input feature-channel count. Required.
        out_channels (int): Number of output feature channels. Required.
        stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
        norm (str | None): Supported normalization choice; native training-statistics requirements still apply. Default: 'batch'.
        dropout (float): Drop probability; this regularization is active only in training mode. Default: 0.0.

    Returns:
        An instance of WideResidualBlock with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        WideResidualBlock(in_channels, out_channels)

    HYSo Guide: components.residual
    Offline: hyso docs show components.residual --lang en
    Open: hyso docs open components.residual --lang en
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 norm: _typing.Literal['batch', 'group', None] = 'batch', dropout: float = 0.) -> None:
        """Initialize WideResidualBlock with the supplied settings.

        Args:
            in_channels (int): Expected input feature-channel count. Required.
            out_channels (int): Number of output feature channels. Required.
            stride (int): Spatial sampling step; must produce a valid output shape. Default: 1.
            norm (str | None): Supported normalization choice; native training-statistics requirements still apply. Default: 'batch'.
            dropout (float): dropout Default: 0.0.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            WideResidualBlock(in_channels, out_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        super().__init__(in_channels,out_channels,stride,norm,False,dropout)
