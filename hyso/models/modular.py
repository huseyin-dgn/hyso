"""Standalone CNN composition from existing blocks; no training wrapper."""

import torch
from torch import nn

from .blocks import ConvNormAct, ResidualBlock
from .bcam import CBAMBlock
from .WithSE import SEBlock
from .attention import ECAAttention, CoordinateAttention

__all__ = ["ModularCNN"]


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def _stage_values(values, name):
    if type(values) not in (tuple, list) or not values:
        raise ValueError(name + " must be a nonempty tuple or list of positive integers")
    for value in values:
        _positive_int(value, name + " entries")
    return tuple(values)


def _attention(name, channels):
    if name == "none":
        return nn.Identity()
    if name == "se":
        return SEBlock(channels)
    if name == "cbam":
        return CBAMBlock(channels)
    if name == "eca":
        return ECAAttention(channels)
    return CoordinateAttention(channels)  # Constructor validates the selection.


class ModularCNN(nn.Module):
    """Stem -> residual stages -> global average pool -> linear logits.

    widths/repeats/strides are equally sized, nonempty tuples or lists of
    positive Python integers. Lists are copied into immutable tuples; values
    are never coerced. The stem is a stride-1 ConvNormAct. Only each stage's
    first ResidualBlock uses that stage's stride; all subsequent strides are 1.

    attention is 'none', 'se', 'cbam', 'eca' or 'coordinate', with a fresh instance
    for every block. norm is 'batch', 'group' or None for convolution blocks;
    CoordinateAttention retains its own BatchNorm. Attention defaults come from
    the existing modules. No device selection, fit/optimizer/AMP/checkpoint API,
    or EBO epoch-boundary resume support is provided. Use normal Module.to().
    """

    def __init__(self, input_channels=3, num_classes=10, widths=(32, 64, 128),
                 repeats=(1, 1, 1), strides=(1, 2, 2), attention="none", norm="batch"):
        super().__init__()
        self.input_channels = _positive_int(input_channels, "input_channels")
        self.num_classes = _positive_int(num_classes, "num_classes")
        self.widths = _stage_values(widths, "widths")
        self.repeats = _stage_values(repeats, "repeats")
        self.strides = _stage_values(strides, "strides")
        if not (len(self.widths) == len(self.repeats) == len(self.strides)):
            raise ValueError("widths, repeats and strides must describe the same number of stages")
        if type(attention) is not str or attention not in ("none", "se", "cbam", "eca", "coordinate"):
            raise ValueError("attention must be 'none', 'se', 'cbam', 'eca' or 'coordinate'")
        if norm is not None and (type(norm) is not str or norm not in ("batch", "group")):
            raise ValueError("norm must be 'batch', 'group' or None")
        self.attention_name = attention
        self.norm_name = norm

        self.stem = ConvNormAct(input_channels, self.widths[0], norm=norm)
        self.stages = nn.ModuleList()
        in_channels = self.widths[0]
        for width, repeat, stride in zip(self.widths, self.repeats, self.strides):
            blocks = []
            for index in range(repeat):
                blocks.append(ResidualBlock(in_channels, width,
                                            stride=stride if index == 0 else 1,
                                            norm=norm, attention=_attention(attention, width)))
                in_channels = width
            self.stages.append(nn.Sequential(*blocks))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.widths[-1], num_classes)

    def forward_features(self, x):
        """Return the final stage's NCHW feature map, before global pooling."""
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("input must be an NCHW tensor")
        if x.shape[1] != self.input_channels or any(size <= 0 for size in x.shape):
            raise ValueError("input must have matching input_channels and nonempty dimensions")
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward(self, x):
        return self.classifier(torch.flatten(self.pool(self.forward_features(x)), 1))
