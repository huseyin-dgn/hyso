# torchvision wrappers, Copyright (c) Soumith Chintala 2016, BSD-3-Clause.
# GhostNetV1 adapted from Huawei Technologies Co., Ltd. (c) 2020, Apache-2.0.
# HYSo changes (2026) add closed configuration, input adapters and HYSo state keys.
# Full upstream licenses and notices are retained in THIRD_PARTY_NOTICES.md.
"""MobileNet, EfficientNetV2, GhostNetV1, ShuffleNetV2 families."""
from __future__ import annotations

import copy
from functools import partial
import torch
from torch import nn
from torchvision.models._utils import _make_divisible
from torchvision.models.mobilenetv2 import MobileNetV2 as TVMobileNetV2
from torchvision.models.mobilenetv3 import MobileNetV3 as TVMobileNetV3, _mobilenet_v3_conf
from torchvision.models.efficientnet import EfficientNet as TVEfficientNet
from torchvision.models.shufflenetv2 import ShuffleNetV2 as TVShuffleNetV2
from torchvision.ops import Conv2dNormActivation

from .m4_blocks import GhostBottleneck
from .m4_config import (GHOST_STAGES, SHUFFLE_WIDTHS, validate_mobile,
                        efficient_settings)


class MobileArchitectureConfig:
    """Architecture-only schema1 snapshot, including the current deployment phase."""
    def _record_architecture(self, name: str, params: dict) -> None:
        self._architecture_config = dict(format='hyso.model_config', schema_version=1,
                                         kind='registered_model', name=name, params=copy.deepcopy(params))

    def to_config(self) -> dict:
        """Return a detached Builder configuration for fresh weights of this topology."""
        config = copy.deepcopy(self._architecture_config)
        if 'deploy' in config['params']:
            config['params']['deploy'] = self.deploy
        return config


def _replace_input_conv(container: nn.Module, key: str | int, input_channels: int) -> None:
    old = container[key] if isinstance(key, int) else getattr(container,key)
    if old.in_channels == input_channels:
        return
    new = nn.Conv2d(input_channels, old.out_channels, old.kernel_size, old.stride,
                    old.padding, bias=False, device=old.weight.device, dtype=old.weight.dtype)
    nn.init.kaiming_normal_(new.weight, mode='fan_out')
    if isinstance(key, int):
        container[key] = new
    else:
        setattr(container,key,new)


class MobileNetV2(MobileArchitectureConfig, TVMobileNetV2):
    """Torchvision canonical MobileNetV2, divisor-eight channels, linear bottlenecks.

    width_mult=1 is the standard layout. Input/head/width changes construct
    explicit untrained variants. dropout affects the classification head.
    """
    def __init__(self, input_channels: int = 3, num_classes: int = 1000,
                 width_mult: float = 1., dropout: float = .2) -> None:
        params = dict(input_channels=input_channels,num_classes=num_classes,width_mult=width_mult,dropout=dropout)
        validate_mobile('mobilenet_v2',params)
        super().__init__(num_classes=num_classes,width_mult=width_mult,dropout=dropout)
        _replace_input_conv(self.features[0],0,input_channels)
        self._record_architecture('mobilenet_v2',params)


class MobileNetV3(MobileArchitectureConfig, TVMobileNetV3):
    """Torchvision Small/Large MobileNetV3 with native SE, ReLU and HardSwish.

    variant='large', width_mult=1 selects the standard Large configuration.
    """
    def __init__(self, variant: str = 'large', input_channels: int = 3,
                 num_classes: int = 1000, width_mult: float = 1., dropout: float = .2) -> None:
        params = dict(variant=variant,input_channels=input_channels,num_classes=num_classes,
                      width_mult=width_mult,dropout=dropout)
        validate_mobile('mobilenet_v3',params)
        settings,last = _mobilenet_v3_conf('mobilenet_v3_'+variant,width_mult=width_mult)
        super().__init__(settings,last,num_classes=num_classes,dropout=dropout)
        _replace_input_conv(self.features[0],0,input_channels)
        self._record_architecture('mobilenet_v3',params)


class EfficientNetV2(MobileArchitectureConfig, TVEfficientNet):
    """Torchvision V2 S/M/L layouts with FusedMBConv/MBConv and native SE/SiLU.

    width_mult=depth_mult=1 preserves the selected canonical stage configuration.
    Smaller multipliers explicitly select derived width/repeat configurations.
    dropout=None uses torchvision's S=.2, M=.3, L=.4 head defaults.
    """
    def __init__(self, variant: str = 's', input_channels: int = 3, num_classes: int = 1000,
                 width_mult: float = 1., depth_mult: float = 1.,
                 dropout: float | None = None, stochastic_depth: float = .2) -> None:
        params = dict(variant=variant,input_channels=input_channels,num_classes=num_classes,
                      width_mult=width_mult,depth_mult=depth_mult,dropout=dropout,
                      stochastic_depth=stochastic_depth)
        validate_mobile('efficientnet_v2',params)
        settings,last = efficient_settings(variant,width_mult,depth_mult)
        rate = {'s':.2,'m':.3,'l':.4}[variant] if dropout is None else dropout
        super().__init__(settings,rate,stochastic_depth_prob=stochastic_depth,
                         num_classes=num_classes,last_channel=last,norm_layer=partial(nn.BatchNorm2d,eps=1e-3))
        _replace_input_conv(self.features[0],0,input_channels)
        self._record_architecture('efficientnet_v2',params)


class ShuffleNetV2(MobileArchitectureConfig, TVShuffleNetV2):
    """Torchvision x0_5/x1_0/x1_5/x2_0 stage widths and repeats [4,8,4]."""
    def __init__(self, variant: str = 'x1_0', input_channels: int = 3, num_classes: int = 1000) -> None:
        params = dict(variant=variant,input_channels=input_channels,num_classes=num_classes)
        validate_mobile('shufflenet_v2',params)
        super().__init__([4,8,4],SHUFFLE_WIDTHS[variant],num_classes=num_classes)
        _replace_input_conv(self.conv1,0,input_channels)
        self._record_architecture('shufflenet_v2',params)


class GhostNet(MobileArchitectureConfig, nn.Module):
    """GhostNetV1 author stage widths, Ghost expansion/projection and hard-gated SE.

    width_mult rounds to divisor four. The post-pool head has 1280 channels.
    HYSo module keys are used for strict HYSo state_dict serialization.
    """
    def __init__(self, input_channels: int = 3, num_classes: int = 1000,
                 width_mult: float = 1., dropout: float = .2) -> None:
        super().__init__()
        params = dict(input_channels=input_channels,num_classes=num_classes,width_mult=width_mult,dropout=dropout)
        validate_mobile('ghostnet',params)
        previous = _make_divisible(16*width_mult,4)
        self.stem = Conv2dNormActivation(input_channels,previous,3,2,activation_layer=nn.ReLU)
        blocks = []
        for kernel,expanded,channels,se,stride in GHOST_STAGES:
            output = _make_divisible(channels*width_mult,4)
            hidden = _make_divisible(expanded*width_mult,4)
            blocks.append(GhostBottleneck(previous,output,hidden,stride,kernel,se))
            previous = output
        head = _make_divisible(960*width_mult,4)
        self.blocks = nn.Sequential(*blocks)
        self.features = Conv2dNormActivation(previous,head,1,activation_layer=nn.ReLU)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(head,1280,1),nn.ReLU(),
                                  nn.Flatten(),nn.Dropout(dropout),nn.Linear(1280,num_classes))
        self._record_architecture('ghostnet',params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(self.blocks(self.stem(x))))




