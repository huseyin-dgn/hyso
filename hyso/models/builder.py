"""Validated linear composition and safe architecture-only JSON round trips.

The builder reuses existing blocks; it does not implement a new convolution,
attention family, checkpoint format, arbitrary graph compiler or code loader.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .registry import ModelConfigError, ModelRegistry, _plain

_MAX_BYTES = 2 * 1024**2
_MAX_PARAMETERS = 10_000_000


def _cost(name, p):
    """Conservative parameter bound, computed before any tensor allocation."""
    if name in ('conv2d', 'conv_norm_act'):
        k = p['kernel_size']
        return p['in_channels']*p['out_channels']*k[0]*k[1]//p['groups'] + 3*p['out_channels']
    if name == 'residual':
        a, b = p['in_channels'], p['out_channels']
        cost = 9*a*b + 9*b*b + a*b + 6*b
        return cost + (_cost(p['attention']['name'], p['attention']['params']) if p['attention'] else 0)
    if name in ('se', 'cbam', 'eca', 'coordinate'):
        c = p['channels']
        if name == 'eca': return p['kernel_size']
        hidden = max(8 if name == 'coordinate' else 1, c//p['reduction'])
        if name == 'coordinate': return 3*c*hidden + 3*hidden + 2*c
        return 2*c*hidden + hidden + c + (2*p['kernel_size']**2+1 if name == 'cbam' else 0)
    if name in ('batch_norm', 'group_norm'):
        return 2*p.get('num_features', p.get('num_channels', 0))
    return 0


def _shape(name, p, shape):
    c, h, w = shape
    expected = p.get('in_channels', p.get('channels', p.get('num_features', p.get('num_channels', c))))
    if c != expected:
        raise ModelConfigError(f'{name}: incoming channels {c}, expected {expected}')
    if name in ('conv2d', 'conv_norm_act'):
        k, s, pad = p['kernel_size'], p['stride'], p['padding']
        h, w = ((v + 2*padding - kernel)//stride + 1 for v, padding, kernel, stride in zip((h,w),pad,k,s))
        c = p['out_channels']
    elif name == 'residual':
        h, w = ((v+s-1)//s for v,s in zip((h,w),p['stride']))
        c = p['out_channels']
    if min(c,h,w) <= 0:
        raise ModelConfigError(f'{name}: kernel/stride/padding produces an empty spatial output')
    if max(c,h,w) > 8192:
        raise ModelConfigError(f'{name}: inferred output dimension exceeds 8192')
    return c,h,w


def _pairs(items):
    out = {}
    for k,v in items:
        if k in out: raise ModelConfigError(f'duplicate JSON key: {k}')
        out[k] = v
    return out


class BuiltClassifier(nn.Module):
    """Existing blocks in a linear NCHW chain, global average pool, linear logits.

    Input CHW must exactly match the declared shape; batch size may vary. Native
    BatchNorm/GroupNorm statistical constraints still apply. No implicit dtype
    casting, device movement or arbitrary graph connections are performed.
    Standard state_dict/save_training_checkpoint can store component state, but
    Trainer's automatic ModularCNN-only epoch-resume profile does not cover this
    class. Configuration JSON contains no weights or executable source.
    """
    def __init__(self, features: list[nn.Module], config: dict[str, Any], output_channels: int):
        super().__init__()
        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(output_channels, config['num_classes'])
        self._architecture = json.dumps(config, sort_keys=True, allow_nan=False)
        self.input_shape = tuple(config['input_shape'])

    def to_config(self) -> dict[str, Any]:
        """Return an independent architecture description, never model weights."""
        return json.loads(self._architecture)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return [N, num_classes] logits; reject incompatible input before work."""
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[0] < 1 or tuple(x.shape[1:]) != self.input_shape:
            raise ModelConfigError(f'expected nonempty NCHW with CHW={self.input_shape}')
        if x.device != self.classifier.weight.device or x.dtype != self.classifier.weight.dtype:
            raise ModelConfigError('input device/dtype must match model; move/cast explicitly')
        for i, block in enumerate(self.features):
            try:
                x = block(x)
            except ValueError as error:
                raise ModelConfigError(f'block {i} rejected its input/statistics: {error}') from error
        return self.classifier(torch.flatten(self.pool(x), 1))


class ModelBuilder:
    """Configure a bounded CPU classifier without rewriting registered blocks.

    Args:
        input_shape: Exact positive (C,H,W), each dimension <=8192.
        num_classes: Positive logit count <=8192.

    ``add(name, **params)`` validates the whole proposed chain before changing
    this builder. Only linear connections are supported; the existing residual
    block alone owns its internal skip/projection. At most 64 blocks and a
    conservative 10-million-parameter estimate are allowed. ``build`` allocates
    fresh CPU weights; call .to(device/dtype) explicitly afterwards. Arbitrary
    user nn.Module objects remain usable directly with Trainer/BenchmarkRunner.
    """
    def __init__(self, input_shape: tuple[int, int, int], num_classes: int):
        config = dict(format='hyso.model_config', schema_version=1, kind='sequential_classifier',
                      input_shape=input_shape, num_classes=num_classes, blocks=[])
        self._config, _ = self._validate(config)

    @staticmethod
    def _validate(config):
        c = _plain(config)
        if type(c) is not dict or c.get('format') != 'hyso.model_config' or type(c.get('schema_version')) is not int or c['schema_version'] != 1:
            raise ModelConfigError('unsupported model configuration format/version')
        reg = ModelRegistry()
        if c.get('kind') == 'modular_cnn':
            if set(c) != {'format','schema_version','kind','params'}:
                raise ModelConfigError('unknown/missing ModularCNN configuration fields')
            p = reg.validate('modular_cnn', c['params'])
            cost = 9*p['input_channels']*p['widths'][0] + 3*p['widths'][0]
            previous = p['widths'][0]
            for width, repeat in zip(p['widths'],p['repeats']):
                for _ in range(repeat):
                    attention = None if p['attention'] == 'none' else dict(name=p['attention'], params=reg.validate(p['attention'],dict(channels=width)))
                    cost += _cost('residual',dict(in_channels=previous,out_channels=width,attention=attention))
                    previous = width
            cost += (previous+1)*p['num_classes']
            if cost > _MAX_PARAMETERS: raise ModelConfigError('model exceeds conservative parameter budget 10000000')
            c['params'] = p
            return c, (previous,)
        if c.get('kind') != 'sequential_classifier' or set(c) != {'format','schema_version','kind','input_shape','num_classes','blocks'}:
            raise ModelConfigError('unsupported connection/model kind or unknown configuration fields')
        shape = c['input_shape']
        if type(shape) is not list or len(shape) != 3 or any(type(v) is not int or not 1 <= v <= 8192 for v in shape):
            raise ModelConfigError('input_shape must be positive [C,H,W], dimensions <=8192')
        if type(c['num_classes']) is not int or not 1 <= c['num_classes'] <= 8192:
            raise ModelConfigError('num_classes must be a positive integer <=8192')
        if type(c['blocks']) is not list or len(c['blocks']) > 64:
            raise ModelConfigError('blocks must be a list of at most 64 entries')
        normalized = []
        cost = 0
        for i, block in enumerate(c['blocks']):
            if type(block) is not dict or set(block) != {'name','params'}:
                raise ModelConfigError(f'block {i}: only name/params are allowed; no custom connections')
            spec = reg.get(block['name'])
            if spec.category == 'model': raise ModelConfigError('a complete model cannot be inserted as an NCHW block')
            p = reg.validate(block['name'],block['params'])
            try: shape = _shape(block['name'],p,shape)
            except ModelConfigError as error: raise ModelConfigError(f'block {i}: {error}') from error
            cost += _cost(block['name'],p)
            normalized.append(dict(name=block['name'],params=p))
        if cost + (shape[0]+1)*c['num_classes'] > _MAX_PARAMETERS:
            raise ModelConfigError('model exceeds conservative parameter budget 10000000')
        c['blocks'] = normalized
        return c, tuple(shape)

    @classmethod
    def modular(cls, **params: Any) -> ModelBuilder:
        """Describe the original ModularCNN; build() returns exactly that class."""
        return cls.from_config(dict(format='hyso.model_config', schema_version=1,kind='modular_cnn',params=params))

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ModelBuilder:
        """Validate/copy plain architecture data; never imports configuration names."""
        checked, _ = cls._validate(config)
        result = cls.__new__(cls)
        result._config = checked
        return result

    def to_config(self) -> dict[str, Any]:
        """Return a detached JSON-compatible configuration including defaults."""
        return _plain(self._config)

    def add(self, name: str, **params: Any) -> ModelBuilder:
        """Append a registered block transactionally; return self for chaining."""
        if self._config['kind'] != 'sequential_classifier':
            raise ModelConfigError('add is unavailable for the fixed ModularCNN profile')
        candidate = self.to_config()
        candidate['blocks'].append(dict(name=name,params=params))
        self._config, _ = self._validate(candidate)
        return self

    def build(self) -> nn.Module:
        """Allocate a fresh CPU model; architecture round trips do not restore weights."""
        config, shape = self._validate(self._config)
        reg = ModelRegistry()
        with torch.device('cpu'):
            if config['kind'] == 'modular_cnn':
                return reg.create('modular_cnn',**config['params'])
            blocks = [reg.create(b['name'],**b['params']) for b in config['blocks']]
            return BuiltClassifier(blocks,config,shape[0])

    def to_json(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Save architecture only, UTF-8 JSON; refuse to overwrite by default."""
        if type(overwrite) is not bool: raise ModelConfigError('overwrite must be bool')
        raw = json.dumps(self.to_config(),indent=2,allow_nan=False)+'\n'
        with Path(path).open('w' if overwrite else 'x',encoding='utf-8',newline='\n') as stream:
            stream.write(raw)

    @classmethod
    def from_json(cls, path: str | Path) -> ModelBuilder:
        """Read bounded UTF-8 JSON; reject duplicates, NaN and unsupported names."""
        with Path(path).open('rb') as stream: raw = stream.read(_MAX_BYTES+1)
        if len(raw) > _MAX_BYTES: raise ModelConfigError('configuration exceeds 2 MiB')
        def invalid(value): raise ModelConfigError(f'nonfinite JSON number: {value}')
        try:
            data = json.loads(raw.decode('utf-8'),object_pairs_hook=_pairs,parse_constant=invalid)
            return cls.from_config(data)
        except (ValueError, TypeError, RecursionError) as error:
            raise ModelConfigError(str(error)) from error
