"""Closed, inspectable constructors for existing HYSo/PyTorch building blocks.

No plugin loading or global registration. Schema/shape restrictions describe this
builder profile, not every option supported by the original standalone classes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Callable

import torch
from torch import nn

from .blocks import ConvNormAct, ResidualBlock
from .bcam import CBAMBlock
from .WithSE import SEBlock
from .attention import ECAAttention, CoordinateAttention
from .modular import ModularCNN


class ModelConfigError(ValueError):
    """Invalid/unsupported registry configuration or architectural connection."""


def _plain(value: Any, depth: int = 0) -> Any:
    if depth > 16:
        raise ModelConfigError('configuration exceeds nesting limit 16')
    if value is None or type(value) in (str, bool, int):
        if type(value) is str and len(value) > 256:
            raise ModelConfigError('configuration string exceeds 256 characters')
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (tuple, list) and len(value) <= 64:
        return [_plain(v, depth+1) for v in value]
    if type(value) is dict and len(value) <= 64 and all(type(k) is str for k in value):
        return {k: _plain(v, depth+1) for k, v in value.items()}
    raise ModelConfigError('configuration requires bounded plain finite JSON values')


@dataclass(frozen=True)
class ParameterSpec:
    """Serializable parameter contract; defaults never contain Python objects."""
    name: str
    kind: str
    description: str
    required: bool = True
    default: Any = None
    choices: tuple = ()


@dataclass(frozen=True)
class ComponentSpec:
    """Read-only component metadata; constructor is the actual existing class.

    ``to_dict()`` describes it without serializing executable callables. Category
    filters and parameter contracts support menus/help without building a model.
    """
    name: str
    category: str
    description: str
    constructor: Callable[..., nn.Module]
    parameters: tuple[ParameterSpec, ...]
    input_condition: str
    output_condition: str
    compatibility: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        params = []
        for p in self.parameters:
            row = dict(name=p.name, kind=p.kind, description=p.description,
                       required=p.required, choices=list(p.choices))
            if p.kind in ('positive_int', 'odd_int', 'pair', 'padding', 'int_list'):
                row['constraints'] = dict(minimum=0 if p.kind == 'padding' else 1,
                                          maximum=8192, odd=p.kind == 'odd_int')
                if p.kind == 'int_list': row['constraints']['max_items'] = 16
            if not p.required:
                row['default'] = _plain(p.default)
            params.append(row)
        return dict(name=self.name, category=self.category, description=self.description,
                    constructor=self.constructor.__module__+'.'+self.constructor.__qualname__,
                    parameters=params, input_condition=self.input_condition,
                    output_condition=self.output_condition, compatibility=list(self.compatibility))


def _p(name, kind='positive_int', default=..., choices=(), description=None):
    return ParameterSpec(name, kind, description or name.replace('_', ' '),
                         default is ..., None if default is ... else default, choices)


_NORM = _p('norm', 'choice', 'batch', ('batch', 'group', None), 'Convolution normalization; group uses largest divisor <=8.')
_IN = _p('in_channels')
_OUT = _p('out_channels')
_CHANNELS = _p('channels')
_SPECS = (
    ('conv_norm_act', 'convolution', 'Existing Conv2d → norm → activation block.', ConvNormAct,
     (_IN, _OUT, _p('kernel_size', 'pair', 3), _p('stride', 'pair', 1), _p('padding', 'padding', 1),
      _p('groups', default=1), _NORM, _p('activation', 'choice', 'silu', ('silu', 'relu', None)))),
    ('conv2d', 'convolution', 'PyTorch Conv2d primitive used by HYSo; grouped/depthwise via groups.', nn.Conv2d,
     (_IN, _OUT, _p('kernel_size', 'pair', 3), _p('stride', 'pair', 1), _p('padding', 'padding', 1),
      _p('groups', default=1), _p('bias', 'bool', True))),
    ('residual', 'residual', 'Existing two-convolution ResidualBlock with internal projection/addition.', ResidualBlock,
     (_IN, _OUT, _p('stride', 'pair', 1), _NORM,
      _p('attention', 'attention', None, description='None or {name, params} for a registered attention with matching output channels.'))),
    ('se', 'attention', 'Existing squeeze/excitation channel gates.', SEBlock,
     (_CHANNELS, _p('reduction', default=8))),
    ('cbam', 'attention', 'Existing channel and spatial attention.', CBAMBlock,
     (_CHANNELS, _p('reduction', default=16), _p('kernel_size', 'odd_int', 7))),
    ('eca', 'attention', 'Existing channel-axis Conv1d gates; no channel reduction.', ECAAttention,
     (_CHANNELS, _p('kernel_size', 'odd_int', 3))),
    ('coordinate', 'attention', 'Existing height/width attention with internal BatchNorm.', CoordinateAttention,
     (_CHANNELS, _p('reduction', default=32))),
    ('batch_norm', 'normalization', 'PyTorch BatchNorm2d as used in existing HYSo blocks.', nn.BatchNorm2d,
     (_p('num_features'),)),
    ('group_norm', 'normalization', 'PyTorch GroupNorm with explicit groups/channels.', nn.GroupNorm,
     (_p('num_groups'), _p('num_channels'))),
    ('relu', 'activation', 'Out-of-place ReLU used by ConvNormAct.', nn.ReLU, ()),
    ('silu', 'activation', 'Out-of-place SiLU used by existing residual/convolution blocks.', nn.SiLU, ()),
    ('hardswish', 'activation', 'Out-of-place HardSwish used by CoordinateAttention.', nn.Hardswish, ()),
    ('identity', 'identity', 'PyTorch Identity used by existing optional branches.', nn.Identity, ()),
    ('modular_cnn', 'model', 'Original ModularCNN constructor; unchanged class and state_dict layout.', ModularCNN,
     (_p('input_channels', default=3), _p('num_classes', default=10),
      _p('widths', 'int_list', (32, 64, 128)), _p('repeats', 'int_list', (1, 1, 1)),
      _p('strides', 'int_list', (1, 2, 2)),
      _p('attention', 'choice', 'none', ('none', 'se', 'cbam', 'eca', 'coordinate')), _NORM)),
)

_IO = {
    'convolution': ('Floating [N,in_channels,H,W]; groups divides both channel counts.',
                    '[N,out_channels,floor((H+2*pH-kH)/sH)+1,floor((W+2*pW-kW)/sW)+1]; positive spatial output.'),
    'residual': ('Floating [N,in_channels,H,W]; attention channels equal out_channels.',
                 '[N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.'),
    'attention': ('Floating [N,channels,H,W], all dimensions positive.',
                  'Same NCHW shape; multiplicative gates, no channel or spatial resizing.'),
    'normalization': ('Floating NCHW; C=num_features (BatchNorm) or num_channels (GroupNorm); groups divide C.',
                      'Same NCHW shape; native batch/statistics constraints apply.'),
    'activation': ('Floating nonempty NCHW in Builder v1.', 'Same NCHW shape; out-of-place activation.'),
    'identity': ('Floating nonempty NCHW in Builder v1.', 'Same tensor/shape; no parameters.'),
    'model': ('Floating nonempty [N,input_channels,H,W]; normal model.to(device/dtype) semantics.',
              '[N,num_classes] unnormalized classification logits.'),
}


class ModelRegistry:
    """Discover a fixed whitelist, validate parameters and construct CPU modules.

    ``list(category=None)`` and ``get(name)`` allocate no tensors. ``create``
    constructs only existing classes on CPU, consuming normal constructor RNG.
    No device/dtype, arbitrary callable, import path or plugin is accepted in JSON.
    Integer dimensions are bounded to 8192; complete Builder models additionally
    have an estimated 10-million-parameter limit. Native normalization retains
    PyTorch's training batch/statistics requirements.
    """

    def __init__(self) -> None:
        self._entries = MappingProxyType({name: ComponentSpec(
            name, category, description, ctor, params,
            _IO[category][0], _IO[category][1],
            ('Existing canonical constructor; CPU construction only.',
             'Builder v1: linear connections; no external skip/concat/DAG.',
             'BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.',
             'SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.',
             'Serialization saves architecture only, not trained weights or optimizer state.'))
            for name, category, description, ctor, params in _SPECS})

    def list(self, category: str | None = None) -> tuple[ComponentSpec, ...]:
        """Return immutable metadata in stable catalog order, optionally filtered."""
        if category is not None and (type(category) is not str or category not in {s.category for s in self._entries.values()}):
            raise ModelConfigError(f'unknown category: {category!r}')
        return tuple(s for s in self._entries.values() if category is None or s.category == category)

    def get(self, name: str) -> ComponentSpec:
        """Return schema, constructor, shape conditions and compatibility notes."""
        if type(name) is not str or name not in self._entries:
            raise ModelConfigError(f'unknown component: {name!r}')
        return self._entries[name]

    def validate(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Validate plain parameters without constructing a module; return a copy."""
        spec = self.get(name)
        p = _plain(params)
        if type(p) is not dict:
            raise ModelConfigError('params must be a dictionary')
        unknown = set(p)-{v.name for v in spec.parameters}
        if unknown:
            raise ModelConfigError(f'{name}: unknown parameters {sorted(unknown)}')
        result = {}
        for field in spec.parameters:
            if field.name not in p and field.required:
                raise ModelConfigError(f'{name}: missing parameter {field.name}')
            v = p.get(field.name, _plain(field.default))
            kind = field.kind
            if kind in ('positive_int', 'odd_int'):
                valid = type(v) is int and 1 <= v <= 8192
                if kind == 'odd_int': valid = valid and v % 2 == 1
            elif kind in ('pair', 'padding'):
                values = [v, v] if type(v) is int else v
                valid = type(values) is list and len(values) == 2 and all(
                    type(x) is int and (0 if kind == 'padding' else 1) <= x <= 8192 for x in values)
                if valid: v = values
            elif kind == 'int_list':
                valid = type(v) is list and 1 <= len(v) <= 16 and all(type(x) is int and 1 <= x <= 8192 for x in v)
            elif kind == 'bool': valid = type(v) is bool
            elif kind == 'choice': valid = (type(v) is str or v is None) and v in field.choices
            elif kind == 'attention':
                valid = v is None or type(v) is dict and set(v) == {'name', 'params'}
                if valid and v is not None:
                    child = self.get(v['name'])
                    if child.category != 'attention':
                        raise ModelConfigError('residual attention must name an attention component')
                    v = dict(name=v['name'], params=self.validate(v['name'], v['params']))
            else: valid = False
            if not valid:
                raise ModelConfigError(f'{name}.{field.name}: invalid {kind} value')
            result[field.name] = v
        if name in ('conv_norm_act', 'conv2d') and any(result[k] % result['groups'] for k in ('in_channels', 'out_channels')):
            raise ModelConfigError('groups must divide both channel counts')
        if name == 'group_norm' and result['num_channels'] % result['num_groups']:
            raise ModelConfigError('num_groups must divide num_channels')
        if name == 'residual' and result['attention'] is not None:
            if result['attention']['params']['channels'] != result['out_channels']:
                raise ModelConfigError('residual attention channels must equal out_channels')
        if name == 'modular_cnn':
            if len({len(result[k]) for k in ('widths', 'repeats', 'strides')}) != 1 or sum(result['repeats']) > 64:
                raise ModelConfigError('ModularCNN stage lengths must match; at most 64 total blocks')
        return result

    def create(self, name: str, **params: Any) -> nn.Module:
        """Construct a whitelisted existing module on CPU after full validation.

        For a chain with channel/spatial validation and an aggregate size budget,
        use ModelBuilder. Direct construction has original PyTorch allocation cost.
        """
        p = self.validate(name, params)
        if name == 'residual' and p['attention'] is not None:
            child = p['attention']
            p['attention'] = self.create(child['name'], **child['params'])
        with torch.device('cpu'):
            return self.get(name).constructor(**p)
