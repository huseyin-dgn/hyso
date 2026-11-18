"""NCHW SimAM, convolutional Triplet and selective-kernel attention.

No weights are downloaded. Native normalization train/eval semantics apply.
"""
import math
import torch
from torch import nn
from .attention import _check_input
from .blocks import _positive_int


class SimAM(nn.Module):
    """Parameter-free spatial energy gates; H*W=1 uses variance zero.

    Half/bfloat16 statistics accumulate in float32. epsilon must be finite >0.

    Args:
        channels (int): Feature-channel count required by this block. Required.
        epsilon (float): Numerical tolerance/stabilizer used by this operation. Default: 0.0001.

    Returns:
        An instance of SimAM with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        SimAM(channels)

    HYSo Guide: components.attention
    Offline: hyso docs show components.attention --lang en
    Open: hyso docs open components.attention --lang en
    """
    def __init__(self, channels: int, epsilon: float = 1e-4) -> None:
        """Initialize SimAM with the supplied settings.

        Args:
            channels (int): Feature-channel count required by this block. Required.
            epsilon (float): epsilon Default: 0.0001.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            SimAM(channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        super().__init__()
        self.channels = _positive_int(channels, 'channels')
        if type(epsilon) not in (int, float) or abs(epsilon)>1e308 or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('epsilon must be finite and positive')
        self.epsilon = epsilon

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

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        _check_input(x, self.channels)
        if not x.is_floating_point(): raise ValueError('floating attention input required')
        value = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        # Normalize first to avoid overflow when squaring large finite inputs.
        scale = value.abs().amax((2, 3), keepdim=True).clamp_min(1).detach()
        z = value / scale
        delta = (z - z.mean((2, 3), keepdim=True)).square()
        variance = delta.sum((2, 3), keepdim=True) / max(1, x.shape[2]*x.shape[3]-1)
        regularizer = (self.epsilon / scale / scale).clamp_min(torch.finfo(value.dtype).tiny)
        gate = torch.sigmoid(delta / (4 * (variance + regularizer)) + .5)
        return x * gate.to(x.dtype)


class _TripletGate(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.norm = nn.BatchNorm2d(1, eps=1e-5, momentum=.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat((x.amax(1, keepdim=True), x.mean(1, keepdim=True)), 1)
        return x * torch.sigmoid(self.norm(self.conv(pooled)))


class TripletAttention(nn.Module):
    """Rotate-to-attend C/H, C/W and optional H/W gates, averaged equally.

    Default kernel 7 and BN match the convolutional WACV architecture, not
    transformer Triplet Attention. BN requires sufficient statistics in train.

    Args:
        channels (int): Feature-channel count required by this block. Required.
        kernel_size (int): Spatial filter/window extent; accepted integer/pair and oddness rules depend on the component. Default: 7.
        spatial (bool): Include spatial attention where supported. Default: True.

    Returns:
        An instance of TripletAttention with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        TripletAttention(channels)

    HYSo Guide: components.attention
    Offline: hyso docs show components.attention --lang en
    Open: hyso docs open components.attention --lang en
    """
    def __init__(self, channels: int, kernel_size: int = 7, spatial: bool = True) -> None:
        """Initialize TripletAttention with the supplied settings.

        Args:
            channels (int): Feature-channel count required by this block. Required.
            kernel_size (int): Spatial filter/window extent; accepted integer/pair and oddness rules depend on the component. Default: 7.
            spatial (bool): spatial Default: True.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            TripletAttention(channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        super().__init__()
        self.channels = _positive_int(channels, 'channels')
        _positive_int(kernel_size, 'kernel_size')
        if kernel_size % 2 != 1 or type(spatial) is not bool:
            raise ValueError('kernel_size must be odd; spatial must be bool')
        self.cw, self.hc = _TripletGate(kernel_size), _TripletGate(kernel_size)
        self.hw = _TripletGate(kernel_size) if spatial else None

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

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        _check_input(x, self.channels)
        if not x.is_floating_point(): raise ValueError('floating attention input required')
        a = self.cw(x.transpose(1, 2)).transpose(1, 2)
        b = self.hc(x.transpose(1, 3)).transpose(1, 3)
        return (a+b+self.hw(x))/3 if self.hw is not None else (a+b)/2


class SelectiveKernelAttention(nn.Module):
    """SK split/fuse/select: grouped 3x3 dilated branches and channel softmax.

    Output is a learned branch fusion, not x multiplied by a scalar gate.
    Default branches=2, groups=1 is an explicit dense-convolution SK variant;
    use groups=32 where channels permit the grouped SKNet setting. Squeeze uses
    a biased linear/ReLU descriptor (no descriptor BN, supports batch size one).

    Args:
        channels (int): Feature-channel count required by this block. Required.
        branches (int): Explicit branch modules/configurations. Default: 2.
        groups (int): Convolution/group count; divisibility requirements remain unchanged. Default: 1.
        reduction (int): Channel reduction ratio used by the attention bottleneck. Default: 16.
        minimum (int): Lower bound applied by this operation. Default: 32.

    Returns:
        An instance of SelectiveKernelAttention with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        SelectiveKernelAttention(channels)

    HYSo Guide: components.attention
    Offline: hyso docs show components.attention --lang en
    Open: hyso docs open components.attention --lang en
    """
    def __init__(self, channels: int, branches: int = 2, groups: int = 1,
                 reduction: int = 16, minimum: int = 32) -> None:
        """Initialize SelectiveKernelAttention with the supplied settings.

        Args:
            channels (int): Feature-channel count required by this block. Required.
            branches (int): branches Default: 2.
            groups (int): Convolution/group count; divisibility requirements remain unchanged. Default: 1.
            reduction (int): Channel reduction ratio used by the attention bottleneck. Default: 16.
            minimum (int): minimum Default: 32.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            SelectiveKernelAttention(channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        super().__init__()
        for key, value in locals().copy().items():
            if key in ('channels','branches','groups','reduction','minimum'): _positive_int(value,key)
        if not 2 <= branches <= 4 or channels % groups:
            raise ValueError('branches must be 2..4 and groups must divide channels')
        self.channels = channels
        self.branches = nn.ModuleList(nn.Sequential(nn.Conv2d(channels, channels, 3,
            padding=i+1, dilation=i+1, groups=groups, bias=False), nn.BatchNorm2d(channels), nn.ReLU())
            for i in range(branches))
        hidden = max(minimum, channels//reduction)
        self.squeeze = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU())
        self.select = nn.ModuleList(nn.Linear(hidden, channels) for _ in range(branches))

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

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        _check_input(x, self.channels)
        if not x.is_floating_point(): raise ValueError('floating attention input required')
        branches = torch.stack([branch(x) for branch in self.branches], 1)
        descriptor = self.squeeze(branches.sum(1).mean((2, 3)))
        weights = torch.stack([head(descriptor) for head in self.select], 1).softmax(1)
        return (branches * weights[..., None, None]).sum(1)
