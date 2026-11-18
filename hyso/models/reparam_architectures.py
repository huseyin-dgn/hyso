# RepVGG layout adapted from Copyright (c) 2020 DingXiaoH, MIT.
# MobileOne layout adapted from Copyright (C) 2022 Apple Inc. All Rights Reserved.
# HYSo changes (2026) add explicit small variants, Builder JSON and deploy phase guards.
# Full upstream licenses and notices are retained in THIRD_PARTY_NOTICES.md.
"""RepVGG A0/A1 and MobileOne s0/s1/s2 classifiers with explicit deploy topology."""
from __future__ import annotations

import torch
from torch import nn
from .mobile_architectures import MobileArchitectureConfig
from .m4_reparam import RepVGGBlock, MobileOneBlock, ReparameterizableModel
from .m4_config import validate_mobile, rep_settings


class RepVGG(ReparameterizableModel, MobileArchitectureConfig, nn.Module):
    """Author A0/A1 [2,4,14,1] stages with 3x3/1x1/identity training branches.

    width_mult=depth_mult=1 selects the published stage layout. deploy=True
    constructs the single-convolution graph for converted HYSo checkpoints.

    Args:
        variant (str): Supported closed architecture variant. Default: 'a0'.
        input_channels (int): Number of input image/feature channels. Default: 3.
        num_classes (int): Classifier output width; binary/multiclass loss requirements must match. Default: 1000.
        width_mult (float): Multiplier for architecture channel widths. Default: 1.0.
        depth_mult (float): Multiplier for architecture stage depth. Default: 1.0.
        deploy (bool): Use the deployment form of this reparameterized block; conversion support remains explicit. Default: False.

    Returns:
        An instance of RepVGG with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        RepVGG()

    HYSo Guide: models.development-families
    Offline: hyso docs show models.development-families --lang en
    Open: hyso docs open models.development-families --lang en
    """
    def __init__(self, variant: str = 'a0', input_channels: int = 3, num_classes: int = 1000,
                 width_mult: float = 1., depth_mult: float = 1., deploy: bool = False) -> None:
        """Initialize RepVGG with the supplied settings.

        Args:
            variant (str): variant Default: 'a0'.
            input_channels (int): input channels Default: 3.
            num_classes (int): Classifier output width; binary/multiclass loss requirements must match. Default: 1000.
            width_mult (float): Explicit width multiplier in [0.125,2]. Default: 1.0.
            depth_mult (float): Explicit repeat multiplier in [0.1,1]. Default: 1.0.
            deploy (bool): deploy Default: False.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            RepVGG()

        HYSo Guide: models.development-families
        Offline: hyso docs show models.development-families --lang en
        Open: hyso docs open models.development-families --lang en
        """
        super().__init__()
        params = dict(variant=variant,input_channels=input_channels,num_classes=num_classes,
                      width_mult=width_mult,depth_mult=depth_mult,deploy=deploy)
        validate_mobile('repvgg',params)
        stem,widths,depths,_ = rep_settings('repvgg',variant,width_mult,depth_mult)
        self.deploy = deploy
        self.stage_widths, self.stage_depths = widths, depths
        self.stem = RepVGGBlock(input_channels,stem,2,deploy=deploy)
        stages = []
        previous = stem
        for channels,repeats in zip(widths,depths):
            blocks = []
            for index in range(repeats):
                blocks.append(RepVGGBlock(previous,channels,2 if index == 0 else 1,deploy=deploy))
                previous = channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(previous,num_classes))
        self._record_architecture('repvgg',params)

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

        HYSo Guide: models.development-families
        Offline: hyso docs show models.development-families --lang en
        Open: hyso docs open models.development-families --lang en
        """
        return self.head(self.stages(self.stem(x)))


class MobileOne(ReparameterizableModel, MobileArchitectureConfig, nn.Module):
    """Author s0/s1/s2 widths and [2,8,10,1] depthwise/pointwise pair stages.

    s0 uses four convolution branches per stage block, s1/s2 use one. The
    initial stem uses one branch as in the released implementation. These
    variants use plain ReLU and exclude the later s4 SE configuration.

    Args:
        variant (str): Supported closed architecture variant. Default: 's0'.
        input_channels (int): Number of input image/feature channels. Default: 3.
        num_classes (int): Classifier output width; binary/multiclass loss requirements must match. Default: 1000.
        width_mult (float): Multiplier for architecture channel widths. Default: 1.0.
        depth_mult (float): Multiplier for architecture stage depth. Default: 1.0.
        deploy (bool): Use the deployment form of this reparameterized block; conversion support remains explicit. Default: False.

    Returns:
        An instance of MobileOne with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        MobileOne()

    HYSo Guide: models.development-families
    Offline: hyso docs show models.development-families --lang en
    Open: hyso docs open models.development-families --lang en
    """
    def __init__(self, variant: str = 's0', input_channels: int = 3, num_classes: int = 1000,
                 width_mult: float = 1., depth_mult: float = 1., deploy: bool = False) -> None:
        """Initialize MobileOne with the supplied settings.

        Args:
            variant (str): variant Default: 's0'.
            input_channels (int): input channels Default: 3.
            num_classes (int): Classifier output width; binary/multiclass loss requirements must match. Default: 1000.
            width_mult (float): Explicit width multiplier in [0.125,2]. Default: 1.0.
            depth_mult (float): Explicit repeat multiplier in [0.1,1]. Default: 1.0.
            deploy (bool): deploy Default: False.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            MobileOne()

        HYSo Guide: models.development-families
        Offline: hyso docs show models.development-families --lang en
        Open: hyso docs open models.development-families --lang en
        """
        super().__init__()
        params = dict(variant=variant,input_channels=input_channels,num_classes=num_classes,
                      width_mult=width_mult,depth_mult=depth_mult,deploy=deploy)
        validate_mobile('mobileone',params)
        stem,widths,depths,branches = rep_settings('mobileone',variant,width_mult,depth_mult)
        self.deploy = deploy
        self.stage_widths, self.stage_depths = widths, depths
        self.stem = MobileOneBlock(input_channels,stem,3,2,deploy=deploy)
        stages = []
        previous = stem
        for channels,repeats in zip(widths,depths):
            blocks = []
            for index in range(repeats):
                blocks.extend((
                    MobileOneBlock(previous,previous,3,2 if index == 0 else 1,groups=previous,
                                   num_conv_branches=branches,deploy=deploy),
                    MobileOneBlock(previous,channels,1,num_conv_branches=branches,deploy=deploy)))
                previous = channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(previous,num_classes))
        self._record_architecture('mobileone',params)

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

        HYSo Guide: models.development-families
        Offline: hyso docs show models.development-families --lang en
        Open: hyso docs open models.development-families --lang en
        """
        return self.head(self.stages(self.stem(x)))
