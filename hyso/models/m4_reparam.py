# RepVGG topology informed by Copyright (c) 2020 DingXiaoH, MIT.
# MobileOne topology informed by Copyright (C) 2022 Apple Inc. All Rights Reserved.
# HYSo changes (2026) share PyTorch fusion, transactional validation and phase checks.
# Full upstream licenses and notices are retained in THIRD_PARTY_NOTICES.md.
"""Eval-time structural reparameterization for distinct RepVGG/MobileOne graphs."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from .WithSE import SEBlock
from .m4_blocks import block_channels, odd_kernel


def _conv_bn(in_channels: int, out_channels: int, kernel: int,
             stride: int, groups: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, stride,
                                  kernel//2, groups=groups, bias=False),
                         nn.BatchNorm2d(out_channels))


def _check_bn(bn: nn.BatchNorm2d) -> None:
    if type(bn) is not nn.BatchNorm2d or bn.training or not bn.track_running_stats:
        raise ValueError('fusion requires eval BatchNorm2d with running statistics')
    if not math.isfinite(bn.eps) or bn.eps <= 0 or bn.running_mean is None or bn.running_var is None:
        raise ValueError('fusion requires valid BatchNorm epsilon and running statistics')
    tensors = [bn.running_mean, bn.running_var]
    if bn.affine:
        tensors.extend((bn.weight, bn.bias))
    if any(not torch.isfinite(t).all() for t in tensors) or (bn.running_var < 0).any():
        raise ValueError('fusion requires finite BatchNorm state and nonnegative variance')


class _ReparamBlock(nn.Module):
    """Shared fusion mechanics over an explicitly supplied training-branch list."""
    deploy: bool

    def _branches(self) -> list[nn.Module]:
        raise NotImplementedError

    def _reject_phase(self, state_dict: Mapping, prefix: str = '') -> None:
        keys = [key[len(prefix):] for key in state_dict if key.startswith(prefix)]
        wrong = ('branches.', 'scale.', 'identity.') if self.deploy else ('reparam.',)
        if any(key.startswith(wrong) for key in keys):
            phase = 'deploy' if self.deploy else 'training'
            raise RuntimeError(f'{prefix or self.__class__.__name__}: checkpoint topology does not match {phase} block')

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self._reject_phase(state_dict, prefix)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def load_state_dict(self, state_dict: Mapping, strict: bool = True, assign: bool = False):
        """Reject cross-phase weights even when strict=False is requested."""
        self._reject_phase(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @torch.no_grad()
    def _deployment_conv(self) -> nn.Conv2d | None:
        if self.training:
            raise ValueError('call eval() before structural reparameterization')
        if self.deploy:
            return None
        if self.branches[0][0].weight.dtype not in (torch.float32, torch.float64):
            raise ValueError('fusion supports float32 and float64 weights')
        weights, biases = [], []
        for branch in self._branches():
            if type(branch) is nn.BatchNorm2d:
                _check_bn(branch)
                if self.in_channels != self.out_channels or self.stride != 1:
                    raise ValueError('identity branch requires equal channels and stride=1')
                reference = self.branches[0][0].weight
                kernel = reference.new_zeros(self.out_channels, self.in_channels//self.groups,
                                             self.kernel_size, self.kernel_size)
                indices = torch.arange(self.out_channels, device=kernel.device)
                kernel[indices, indices % (self.in_channels//self.groups),
                       self.kernel_size//2, self.kernel_size//2] = 1
                gamma = branch.weight if branch.affine else torch.ones_like(branch.running_var)
                beta = branch.bias if branch.affine else torch.zeros_like(branch.running_var)
                factor = gamma / torch.sqrt(branch.running_var + branch.eps)
                weights.append(kernel * factor[:, None, None, None])
                biases.append(beta - branch.running_mean * factor)
            else:
                if type(branch) is not nn.Sequential or len(branch) != 2:
                    raise ValueError('unsupported fusion branch structure')
                conv, bn = branch
                if type(conv) is not nn.Conv2d:
                    raise ValueError('fusion requires ordinary Conv2d branches')
                _check_bn(bn)
                if (conv.training or conv.in_channels != self.in_channels or conv.out_channels != self.out_channels
                        or conv.groups != self.groups or conv.stride != (self.stride,)*2
                        or conv.dilation != (1, 1) or conv.padding_mode != 'zeros'
                        or conv.kernel_size not in ((1, 1), (self.kernel_size,)*2)
                        or conv.padding != tuple(k//2 for k in conv.kernel_size)):
                    raise ValueError('fusion branch geometry does not match the block contract')
                if any(not torch.isfinite(t).all() for t in conv.parameters()):
                    raise ValueError('fusion requires finite convolution parameters')
                fused = fuse_conv_bn_eval(conv, bn)
                padding = (self.kernel_size - conv.kernel_size[0]) // 2
                weights.append(F.pad(fused.weight, (padding,)*4))
                biases.append(fused.bias)
        kernel = torch.stack(weights).sum(0)
        bias = torch.stack(biases).sum(0)
        if not torch.isfinite(kernel).all() or not torch.isfinite(bias).all():
            raise ValueError('fusion produced nonfinite weights')
        result = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size,
                           self.stride, self.kernel_size//2, groups=self.groups, bias=True,
                           device=kernel.device, dtype=kernel.dtype)
        result.weight.copy_(kernel)
        result.bias.copy_(bias)
        return result.eval()

    def _apply_deployment(self, convolution: nn.Conv2d | None) -> None:
        if convolution is None:
            return
        self.reparam = convolution
        del self.branches, self.scale, self.identity
        self.deploy = True

    def switch_to_deploy(self) -> _ReparamBlock:
        """Fuse this eval block in place, preserving its device, dtype and activation."""
        self._apply_deployment(self._deployment_conv())
        return self


class RepVGGBlock(_ReparamBlock):
    """3x3+BN, 1x1+BN and optional identity BN, summed before ReLU.

    Groups divide both channel counts. deploy=True constructs the single biased
    3x3 topology used to load explicitly converted weights.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 groups: int = 1, deploy: bool = False) -> None:
        super().__init__()
        block_channels(in_channels, out_channels, stride)
        if type(groups) is not int or groups < 1 or in_channels % groups or out_channels % groups:
            raise ValueError('groups must divide both channel counts')
        if type(deploy) is not bool:
            raise ValueError('deploy must be bool')
        self.in_channels, self.out_channels = in_channels, out_channels
        self.stride, self.groups, self.kernel_size = stride, groups, 3
        self.deploy = deploy
        self.activation = nn.ReLU()
        if deploy:
            self.reparam = nn.Conv2d(in_channels, out_channels, 3, stride, 1, groups=groups, bias=True)
        else:
            self.branches = nn.ModuleList([_conv_bn(in_channels, out_channels, 3, stride, groups)])
            self.scale = _conv_bn(in_channels, out_channels, 1, stride, groups)
            self.identity = nn.BatchNorm2d(in_channels) if in_channels == out_channels and stride == 1 else None

    def _branches(self) -> list[nn.Module]:
        return [*self.branches, self.scale] + ([] if self.identity is None else [self.identity])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.activation(self.reparam(x))
        value = self.branches[0](x) + self.scale(x)
        if self.identity is not None:
            value = value + self.identity(x)
        return self.activation(value)


class MobileOneBlock(_ReparamBlock):
    """K Conv+BN branches, optional 1x1 scale and identity BN, then SE/ReLU.

    kernel_size=1 has no scale branch. use_se reuses HYSo's sigmoid SE with
    reduction=16 and a minimum hidden width of one for small configurations.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, groups: int = 1, num_conv_branches: int = 1,
                 use_se: bool = False, deploy: bool = False) -> None:
        super().__init__()
        block_channels(in_channels, out_channels, stride)
        odd_kernel(kernel_size)
        if type(groups) is not int or groups < 1 or in_channels % groups or out_channels % groups:
            raise ValueError('groups must divide both channel counts')
        if type(num_conv_branches) is not int or not 1 <= num_conv_branches <= 8:
            raise ValueError('num_conv_branches must be 1..8')
        if type(use_se) is not bool or type(deploy) is not bool:
            raise ValueError('use_se and deploy must be bool')
        self.in_channels, self.out_channels = in_channels, out_channels
        self.stride, self.groups, self.kernel_size = stride, groups, kernel_size
        self.deploy = deploy
        self.activation = nn.ReLU()
        self.se = SEBlock(out_channels, reduction=16) if use_se else nn.Identity()
        if deploy:
            self.reparam = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                     kernel_size//2, groups=groups, bias=True)
        else:
            self.branches = nn.ModuleList(_conv_bn(in_channels, out_channels, kernel_size, stride, groups)
                                          for _ in range(num_conv_branches))
            self.scale = _conv_bn(in_channels, out_channels, 1, stride, groups) if kernel_size > 1 else None
            self.identity = nn.BatchNorm2d(in_channels) if in_channels == out_channels and stride == 1 else None

    def _branches(self) -> list[nn.Module]:
        result = list(self.branches)
        if self.scale is not None:
            result.append(self.scale)
        if self.identity is not None:
            result.append(self.identity)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.activation(self.se(self.reparam(x)))
        value = self.branches[0](x)
        for branch in self.branches[1:]:
            value = value + branch(x)
        if self.scale is not None:
            value = value + self.scale(x)
        if self.identity is not None:
            value = value + self.identity(x)
        return self.activation(self.se(value))


def reparameterize_model(model: nn.Module, *, inplace: bool = False) -> nn.Module:
    """Convert all supported blocks after validating every branch, defaulting to a copy.

    Planning/fusion finishes for the entire graph before any topology changes.
    Optimizers attached to the old training parameters must be recreated by the
    caller if further training of the converted graph is explicitly intended.
    """
    if not isinstance(model, nn.Module) or type(inplace) is not bool:
        raise TypeError('model must be nn.Module and inplace must be bool')
    if any(module.training for module in model.modules()):
        raise ValueError('call eval() on the complete model before conversion')
    converted = model if inplace else copy.deepcopy(model)
    blocks = [module for module in converted.modules() if isinstance(module, _ReparamBlock)]
    if not blocks:
        raise ValueError('no supported RepVGG/MobileOne blocks found')
    plans = [block._deployment_conv() for block in blocks]
    for block, plan in zip(blocks, plans):
        block._apply_deployment(plan)
    if isinstance(converted, ReparameterizableModel):
        converted.deploy = True
    return converted


class ReparameterizableModel:
    """Public in-place conversion and early checkpoint phase checking for families."""
    def switch_to_deploy(self) -> nn.Module:
        """Convert the complete eval graph after all fusion plans succeed."""
        return reparameterize_model(self, inplace=True)

    def load_state_dict(self, state_dict: Mapping, strict: bool = True, assign: bool = False):
        """Check every block's train/deploy phase before copying model parameters."""
        for name, module in self.named_modules():
            if isinstance(module, _ReparamBlock):
                module._reject_phase(state_dict, name + '.' if name else '')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
