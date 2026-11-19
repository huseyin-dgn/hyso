"""Validated linear composition and safe architecture-only JSON round trips.

The builder reuses existing blocks; it does not implement a new convolution,
attention family, checkpoint format, arbitrary graph compiler or code loader.
"""
from __future__ import annotations
import typing as _typing

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .registry import ModelConfigError, ModelRegistry, _plain
from ._profile import DEVELOPMENT
from . import components
from . import advanced
from . import sampling

M4_FAMILIES = ()
M4_BLOCKS = ()
if DEVELOPMENT:
    from .m4_registry import M4_FAMILIES, M4_BLOCKS, block_cost, block_shape

_MAX_BYTES = 2 * 1024**2
_MAX_PARAMETERS = 10_000_000


def _cost(name, p):
    """Conservative parameter bound, computed before any tensor allocation."""
    if name in advanced.NEW_NAMES:
        return advanced.describe(name,p,(p.get('in_channels',p.get('channels',1)),1,1),1,ModelConfigError)['parameter_count']
    if name in components.NEW_NAMES:
        return components.parameter_count(name, p)
    if name in M4_BLOCKS:
        return block_cost(name,p)
    if name in ('basic_residual','bottleneck','preactivation','wide_residual','resnext_bottleneck'):
        a,b=p['in_channels'],p['out_channels']
        expanded=name in ('bottleneck','resnext_bottleneck') or p.get('bottleneck',False)
        groups=p.get('groups',1);inner=int(b/4*p.get('width_per_group',64)/64)*groups if expanded else b
        main=a*inner+9*inner*inner//groups+inner*b if expanded else 9*a*b+9*b*b
        return main+a*b+4*(a+b+inner)
    if name=='simam': return 0
    if name=='triplet': return 3*(2*p['kernel_size']**2+2)
    if name=='selective_kernel':
        c=p['channels'];h=max(p['minimum'],c//p['reduction']);m=p['branches']
        return m*(9*c*c//p['groups']+2*c)+(c+1)*h+m*(h+1)*c
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
        return components.parameter_count(name, p)
    return 0


def _shape(name, p, shape):
    if name in sampling.SAMPLING | sampling.UTILITIES:
        return sampling.shape(name,p,shape,ModelConfigError)
    if len(shape)!=3:
        raise ModelConfigError('NCHW block requires rank-4 features',code='INVALID_BLOCK_ORDER',
                               expected=['N','C','H','W'],actual=['N',*shape])
    c, h, w = shape
    expected = p.get('in_channels', p.get('channels', p.get('num_features', p.get('num_channels', c))))
    if c != expected:
        raise ModelConfigError(f'{name}: incoming channels {c}, expected {expected}',
                               code='CHANNEL_MISMATCH', expected=expected, actual=c)
    if name == 'prelu' and p['num_parameters'] not in (1, c):
        raise ModelConfigError('PReLU num_parameters must be 1 or incoming channels',
                               code='CHANNEL_MISMATCH', expected=[1,c], actual=p['num_parameters'])
    if name == 'instance_norm' and h*w == 1 and not p['track_running_stats']:
        raise ModelConfigError('InstanceNorm without running statistics requires H*W > 1 in train and eval',
                               code='STATISTICS_REQUIREMENT', expected='H*W > 1', actual=h*w,
                               suggestion='H*W değerini 1 üzerine çıkarın veya track_running_stats=True ile eval modunu kullanın.')
    if name in advanced.NEW_NAMES:
        c,h,w = advanced.describe(name,p,shape,1,ModelConfigError)['stages'][-1]
    elif name in components.CONVOLUTIONS | components.POOLING:
        stages = components.stages(name, p, shape)
        for index, stage in enumerate(stages):
            if min(stage) <= 0 or max(stage) > 8192:
                code = 'INTERMEDIATE_SHAPE_ERROR' if index < len(stages)-1 else 'SPATIAL_DIMENSION_ERROR'
                raise ModelConfigError(f'{name}: stage {index} produces invalid CHW {stage}',
                                       code=code, expected='CHW in 1..8192', actual=list(stage))
        c,h,w = stages[-1]
    elif name in M4_BLOCKS:
        c,h,w = block_shape(name,p,shape)
    elif name == 'conv_norm_act':
        k, s, pad = p['kernel_size'], p['stride'], p['padding']
        h, w = ((v + 2*padding - kernel)//stride + 1 for v, padding, kernel, stride in zip((h,w),pad,k,s))
        c = p['out_channels']
    elif name in ('basic_residual','bottleneck','preactivation','wide_residual','resnext_bottleneck'):
        h,w=((v+p['stride']-1)//p['stride'] for v in (h,w));c=p['out_channels']
    elif name == 'residual':
        h, w = ((v+s-1)//s for v,s in zip((h,w),p['stride']))
        c = p['out_channels']
    if min(c,h,w) <= 0:
        raise ModelConfigError(f'{name}: kernel/stride/padding produces an empty spatial output',
                               code='SPATIAL_DIMENSION_ERROR', expected='positive CHW', actual=[c,h,w])
    if max(c,h,w) > 8192:
        raise ModelConfigError(f'{name}: inferred output dimension exceeds 8192',
                               code='SPATIAL_DIMENSION_ERROR', expected='CHW <= 8192', actual=[c,h,w])
    return c,h,w


def _pairs(items):
    out = {}
    for k,v in items:
        if k in out: raise ModelConfigError(f'duplicate JSON key: {k}')
        out[k] = v
    return out


def _validate_head(head, shape):
    if (type(head) is not dict or set(head)!={'name','params'} or type(head['name']) is not str
            or head['name'] not in sampling.HEADS or type(head['params']) is not dict):
        raise ModelConfigError('schema2 head requires a whitelisted name and params',code='INVALID_OUTPUT_HEAD',
                               expected=sorted(sampling.HEADS),actual=head)
    if 'in_features' in head['params']:
        raise ModelConfigError('Builder infers head in_features from the feature chain',code='INVALID_OUTPUT_HEAD',
                               expected='head params without in_features',actual=head['params'])
    features = math.prod(shape) if head['params'].get('pooling','avg')=='none' else shape[0]
    try:
        params = ModelRegistry().validate(head['name'],dict(in_features=features,**head['params']))
        info = sampling.head_geometry(head['name'],params,shape,ModelConfigError)
    except ModelConfigError as error:
        raise ModelConfigError(str(error),code='INVALID_OUTPUT_HEAD',expected=error.expected,actual=error.actual) from error
    params.pop('in_features')
    return dict(name=head['name'],params=params),info


class BuiltClassifier(nn.Module):
    """Existing blocks in a linear NCHW chain, global average pool, linear logits.

    Input CHW must exactly match the declared shape; batch size may vary. Native
    BatchNorm/GroupNorm statistical constraints still apply. No implicit dtype
    casting, device movement or arbitrary graph connections are performed.
    Standard state_dict/save_training_checkpoint can store component state, but
    Trainer's automatic ModularCNN-only epoch-resume profile does not cover this
    class. Configuration JSON contains no weights or executable source.

    Args:
        features (list[nn.Module]): Explicit feature IDs to compare/plan; prerequisites are never silently executed. Required.
        config (dict[str, Any]): Plain versioned configuration; never executable code or trained state. Required.
        output_channels (int): Number of output feature channels. Required.

    Returns:
        An instance of BuiltClassifier with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        BuiltClassifier(features, config, output_channels)

    HYSo Guide: models.model-builder
    Offline: hyso docs show models.model-builder --lang en
    Open: hyso docs open models.model-builder --lang en
    """
    def __init__(self, features: list[nn.Module], config: dict[str, Any], output_channels: int) -> 'None':
        """Initialize BuiltClassifier with the supplied settings.

        Args:
            features (list[nn.Module]): Explicit feature IDs to compare/plan; prerequisites are never silently executed. Required.
            config (dict[str, Any]): Plain versioned configuration; never executable code or trained state. Required.
            output_channels (int): Number of output feature channels. Required.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            BuiltClassifier(features, config, output_channels)

        HYSo Guide: models.model-builder
        Offline: hyso docs show models.model-builder --lang en
        Open: hyso docs open models.model-builder --lang en
        """
        super().__init__()
        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(output_channels, config['num_classes'])
        self._architecture = json.dumps(config, sort_keys=True, allow_nan=False)
        self.input_shape = tuple(config['input_shape'])

    def to_config(self) -> dict[str, Any]:
        """Return an independent architecture description, never model weights.

        Returns:
            Detached JSON-compatible architecture metadata including normalized defaults; no weights.

        Raises:
            No additional validation for a valid object; corrupt retained state or native copy/serialization failures propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_config()

        HYSo Guide: models.model-builder
        Offline: hyso docs show models.model-builder --lang en
        Open: hyso docs open models.model-builder --lang en
        """
        config = json.loads(self._architecture)
        for block,entry in zip(self.features,config['blocks']):
            if entry['name'] in ('repvgg_block','mobileone_block'):
                entry['params']['deploy'] = block.deploy
        return config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return [N, num_classes] logits; reject incompatible input before work.

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

        HYSo Guide: models.model-builder
        Offline: hyso docs show models.model-builder --lang en
        Open: hyso docs open models.model-builder --lang en
        """
        x = self._forward_features(x)
        return self.classifier(torch.flatten(self.pool(x), 1))

    def _forward_features(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[0] < 1 or tuple(x.shape[1:]) != self.input_shape:
            raise ModelConfigError(f'expected nonempty NCHW with CHW={self.input_shape}',
                                   code='INVALID_INPUT_SHAPE', expected=['N', *self.input_shape],
                                   actual=list(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__)
        if x.device != self.classifier.weight.device or x.dtype != self.classifier.weight.dtype:
            raise ModelConfigError('input device/dtype must match model; move/cast explicitly',
                                   expected=dict(device=str(self.classifier.weight.device), dtype=str(self.classifier.weight.dtype)),
                                   actual=dict(device=str(x.device), dtype=str(x.dtype)))
        for i, block in enumerate(self.features):
            try:
                x = block(x)
            except ModelConfigError as error:
                raise error.at_layer(i)
            except ValueError as error:
                code = 'STATISTICS_REQUIREMENT' if isinstance(block, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)) else 'INVALID_PARAMETER'
                raise ModelConfigError(f'block {i} rejected its input/statistics: {error}',
                                       code=code, layer_index=i, expected='valid block input/statistics', actual=list(x.shape)) from error
        return x


class BuiltSequential(BuiltClassifier):
    """Opt-in schema2 features and explicit prediction head.

    Args:
        features (_typing.Any): Explicit feature IDs to compare/plan; prerequisites are never silently executed. Required.
        config (dict[str, _typing.Any]): Plain versioned configuration; never executable code or trained state. Required.
        head (_typing.Any): Explicit prediction-head kind, such as classification, regression or embedding. Required.

    Returns:
        An instance of BuiltSequential with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        BuiltSequential(features, config, head)

    HYSo Guide: models.model-builder
    Offline: hyso docs show models.model-builder --lang en
    Open: hyso docs open models.model-builder --lang en
    """
    def __init__(self, features: '_typing.Any', config: 'dict[str, _typing.Any]', head: '_typing.Any') -> 'None':
        """Initialize BuiltSequential with the supplied settings.

        Args:
            features (_typing.Any): Explicit feature IDs to compare/plan; prerequisites are never silently executed. Required.
            config (dict[str, _typing.Any]): Plain versioned configuration; never executable code or trained state. Required.
            head (_typing.Any): Explicit prediction-head kind, such as classification, regression or embedding. Required.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            BuiltSequential(features, config, head)

        HYSo Guide: models.model-builder
        Offline: hyso docs show models.model-builder --lang en
        Open: hyso docs open models.model-builder --lang en
        """
        nn.Module.__init__(self)
        self.features = nn.Sequential(*features)
        self.head = head
        self.input_shape = tuple(config['input_shape'])
        self._architecture = json.dumps(config,sort_keys=True,allow_nan=False)

    @property
    def classifier(self) -> 'nn.Linear':
        """Return the built classifier module.

        Returns:
            The existing Linear classification projection. Heads without a linear attribute do not support this accessor.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.classifier

        HYSo Guide: models.model-builder
        Offline: hyso docs show models.model-builder --lang en
        Open: hyso docs open models.model-builder --lang en
        """
        return self.head.linear

    def forward(self,x: '_typing.Any') -> 'torch.Tensor':
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

        HYSo Guide: models.model-builder
        Offline: hyso docs show models.model-builder --lang en
        Open: hyso docs open models.model-builder --lang en
        """
        return self.head(self._forward_features(x))


class ModelBuilder:
    """Assemble a sequential classifier or explicit sequential model from registered components.

    Args:
        input_shape: Exact positive (C,H,W), each dimension <=8192.
        num_classes: Positive logit count <=8192.

    ``add(name, **params)`` validates the whole proposed chain before changing
    this builder. Only linear connections are supported; the existing residual
    block alone owns its internal skip/projection. At most 64 blocks and a
    conservative 10-million-parameter estimate are allowed for sequential chains.
    Named M3 families use a separate conservative 200-million parameter bound. ``build`` allocates
    fresh CPU weights; call .to(device/dtype) explicitly afterwards. Arbitrary
    user nn.Module objects remain usable directly with Trainer/BenchmarkRunner.

    Returns:
        An instance of ModelBuilder with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        builder = ModelBuilder((1, 4, 4), 2)

    HYSo Guide: models.model-builder
    Offline: hyso docs show models.model-builder --lang en
    Open: hyso docs open models.model-builder --lang en
    """
    def __init__(self, input_shape: tuple[int, int, int], num_classes: int) -> 'None':
        """Initialize ModelBuilder with the supplied settings.

        Args:
            input_shape (tuple[int, int, int]): Per-sample (C, H, W) dimensions; batch is excluded. Required.
            num_classes (int): Classifier output width; binary/multiclass loss requirements must match. Required.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            ModelBuilder(input_shape, num_classes)

        HYSo Guide: models.model-builder
        Offline: hyso docs show models.model-builder --lang en
        Open: hyso docs open models.model-builder --lang en
        """
        config = dict(format='hyso.model_config', schema_version=1, kind='sequential_classifier',
                      input_shape=input_shape, num_classes=num_classes, blocks=[])
        self._config, _ = self._validate(config)

    @staticmethod
    def _validate(config):
        c = _plain(config)
        if type(c) is not dict or c.get('format') != 'hyso.model_config' or type(c.get('schema_version')) is not int or c['schema_version'] not in (1,2):
            raise ModelConfigError('unsupported model configuration format/version')
        explicit_head = c['schema_version']==2
        if explicit_head and c.get('kind')!='sequential':
            raise ModelConfigError('schema2 requires kind=sequential',code='INVALID_BLOCK_ORDER')
        reg = ModelRegistry()
        if c.get('kind') == 'registered_model':
            if set(c)!={'format','schema_version','kind','name','params'} or c['name'] not in (('resnet','preact_resnet','wide_resnet','resnext','densenet')+M4_FAMILIES if DEVELOPMENT else ()):
                raise ModelConfigError('unknown/missing registered_model family fields', code='UNSUPPORTED_COMPONENT',
                                       expected='family allowed by the installed profile', actual=c.get('name'))
            c['params']=reg.validate(c['name'],c['params'])
            return c, ()
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
            if cost > _MAX_PARAMETERS:
                raise ModelConfigError('model exceeds conservative parameter budget 10000000',
                                       code='PARAMETER_BUDGET_EXCEEDED', expected=_MAX_PARAMETERS, actual=cost)
            c['params'] = p
            return c, (previous,)
        fields = {'format','schema_version','kind','input_shape','blocks'} | ({'head'} if explicit_head else {'num_classes'})
        if c.get('kind') != ('sequential' if explicit_head else 'sequential_classifier') or set(c) != fields:
            raise ModelConfigError('unsupported connection/model kind or unknown configuration fields',
                                   code='INVALID_BLOCK_ORDER', expected='schema1 sequential_classifier fields', actual=list(c))
        shape = c['input_shape']
        if type(shape) is not list or len(shape) != 3 or any(type(v) is not int or not 1 <= v <= 8192 for v in shape):
            raise ModelConfigError('input_shape must be positive [C,H,W], dimensions <=8192',
                                   code='INVALID_INPUT_SHAPE', expected='three integers in 1..8192', actual=shape)
        if not explicit_head and (type(c['num_classes']) is not int or not 1 <= c['num_classes'] <= 8192):
            raise ModelConfigError('num_classes must be a positive integer <=8192',
                                   code='INVALID_OUTPUT_HEAD', expected='integer in 1..8192', actual=c['num_classes'])
        if type(c['blocks']) is not list or len(c['blocks']) > 64:
            raise ModelConfigError('blocks must be a list of at most 64 entries', code='INVALID_BLOCK_ORDER',
                                   expected='list with at most 64 NCHW blocks', actual=c['blocks'])
        normalized = []
        cost = 0
        for i, block in enumerate(c['blocks']):
            if type(block) is not dict or set(block) != {'name','params'}:
                raise ModelConfigError(f'block {i}: only name/params are allowed; no custom connections',
                                       code='INVALID_BLOCK_ORDER', layer_index=i,
                                       expected=['name', 'params'], actual=block)
            try:
                spec = reg.get(block['name'])
                if spec.category in ('model','prediction_head'):
                    raise ModelConfigError('a complete model cannot be inserted as an NCHW block',
                                           code='INVALID_BLOCK_ORDER', expected='NCHW block', actual=block['name'])
                p = reg.validate(block['name'],block['params'])
                if block['name']=='flatten' and (not explicit_head or i!=len(c['blocks'])-1):
                    raise ModelConfigError('flatten is supported only as the final schema2 feature block',
                                           code='INVALID_BLOCK_ORDER',expected='terminal schema2 flatten',actual=i)
                shape = _shape(block['name'],p,shape)
            except ModelConfigError as error:
                raise error.at_layer(i)
            cost += _cost(block['name'],p)
            normalized.append(dict(name=block['name'],params=p))
        if explicit_head:
            head, head_info = _validate_head(c['head'],shape)
            c['head'] = head
            head_cost = head_info['parameter_count']
        else:
            head_cost = (shape[0]+1)*c['num_classes']
        if cost + head_cost > _MAX_PARAMETERS:
            raise ModelConfigError('model exceeds conservative parameter budget 10000000',
                                   code='PARAMETER_BUDGET_EXCEEDED', expected=_MAX_PARAMETERS,
                                   actual=cost + head_cost)
        c['blocks'] = normalized
        return c, tuple(shape)

    @classmethod
    def sequential(cls, input_shape: '_typing.Sequence[int]', *, head: _typing.Literal['classification', 'regression', 'embedding']='classification', **head_params: '_typing.Any') -> 'ModelBuilder':
        """Create an opt-in schema2 chain with an explicit terminal head.

        Args:
            input_shape (_typing.Sequence[int]): Per-sample (C, H, W) dimensions; batch is excluded. Required.
            head (str): Explicit prediction-head kind, such as classification, regression or embedding. Default: 'classification'.
            head_params (_typing.Any): Parameters for the explicit prediction head; schema2 rules remain unchanged. Required.

        Returns:
            The retained sequential value, without recomputing measurements.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            ModelBuilder.sequential(input_shape)

        HYSo Guide: models.model-builder
        Offline: hyso docs show models.model-builder --lang en
        Open: hyso docs open models.model-builder --lang en
        """
        return cls.from_config(dict(format='hyso.model_config',schema_version=2,kind='sequential',
                                    input_shape=input_shape,blocks=[],head=dict(name=f'{head}_head',params=head_params)))

    @classmethod
    def classification(cls, input_shape: _typing.Sequence[int], *, num_classes: int,
                       pooling: _typing.Literal['avg', 'max', 'none'] = 'avg',
                       dropout: float = 0) -> ModelBuilder:
        """Start the existing schema2 sequential classifier with an explicit head.

        Args:
            input_shape: Per-sample (C, H, W) shape.
            num_classes: Positive output class count.
            pooling: Existing head pooling; none flattens the inferred feature shape.
            dropout: Existing classification-head dropout probability.
        Returns:
            An architecture-only builder; model construction and training are explicit.
        Raises:
            ModelConfigError: The existing schema2 head/shape validator rejects options.
        Example:
            builder = ModelBuilder.classification((1, 8, 8), num_classes=2)

        HYSo Guide: components.heads
        Offline: hyso docs show components.heads --lang en
        Open: hyso docs open components.heads --lang en
        """
        return cls.sequential(input_shape, head='classification', num_classes=num_classes,
                              pooling=pooling, dropout=dropout)

    @classmethod
    def regression(cls, input_shape: _typing.Sequence[int], *, output_features: int,
                   pooling: _typing.Literal['avg', 'max', 'none'] = 'avg') -> ModelBuilder:
        """Start the existing schema2 sequential regressor with an explicit head.

        Args:
            input_shape: Per-sample (C, H, W) shape.
            output_features: Positive number of output values.
            pooling: Existing head pooling strategy.
        Returns:
            An architecture-only builder with inferred head input features.
        Raises:
            ModelConfigError: The existing schema2 head/shape validator rejects options.
        Example:
            builder = ModelBuilder.regression((1, 8, 8), output_features=1)

        HYSo Guide: components.heads
        Offline: hyso docs show components.heads --lang en
        Open: hyso docs open components.heads --lang en
        """
        return cls.sequential(input_shape, head='regression', output_features=output_features, pooling=pooling)

    @classmethod
    def embedding(cls, input_shape: _typing.Sequence[int], *, embedding_dim: int,
                  pooling: _typing.Literal['avg', 'max', 'none'] = 'avg',
                  normalize: bool = False, eps: float = 1e-12) -> ModelBuilder:
        """Start the existing schema2 embedding model with explicit normalization options.

        Args:
            input_shape: Per-sample (C, H, W) shape.
            embedding_dim: Positive embedding output width.
            pooling: Existing head pooling strategy.
            normalize: Apply the existing output L2 normalization.
            eps: Positive native normalization epsilon.
        Returns:
            An architecture-only builder with inferred head input features.
        Raises:
            ModelConfigError: The existing schema2 head/shape validator rejects options.
        Example:
            builder = ModelBuilder.embedding((1, 8, 8), embedding_dim=4, normalize=True)

        HYSo Guide: components.heads
        Offline: hyso docs show components.heads --lang en
        Open: hyso docs open components.heads --lang en
        """
        return cls.sequential(input_shape, head='embedding', embedding_dim=embedding_dim,
                              pooling=pooling, normalize=normalize, eps=eps)

    @classmethod
    def named(cls, name: str, **params: Any) -> ModelBuilder:
        """Describe a closed model family; validation allocates no model or weights.

        Args:
            name (str): Registered component name, exact metric label, or stored entity name in this API. Required.
            params (Any): Closed registered keyword parameters; inspect ModelRegistry.get(name) for supported kinds/defaults. Required.

        Returns:
            The retained named value, without recomputing measurements.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            ModelBuilder.named(name)

        HYSo Guide: models.model-builder
        Offline: hyso docs show models.model-builder --lang en
        Open: hyso docs open models.model-builder --lang en
        """
        return cls.from_config(dict(format='hyso.model_config',schema_version=1,kind='registered_model',name=name,params=params))

    @classmethod
    def modular(cls, **params: Any) -> ModelBuilder:
        """Describe the original ModularCNN; build() returns exactly that class.

        Args:
            params (Any): Closed registered keyword parameters; inspect ModelRegistry.get(name) for supported kinds/defaults. Required.

        Returns:
            The retained modular value, without recomputing measurements.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            ModelBuilder.modular()

        HYSo Guide: models.model-builder
        Offline: hyso docs show models.model-builder --lang en
        Open: hyso docs open models.model-builder --lang en
        """
        return cls.from_config(dict(format='hyso.model_config', schema_version=1,kind='modular_cnn',params=params))

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ModelBuilder:
        """Validate/copy plain architecture data; never imports configuration names.

        Args:
            config (dict[str, Any]): Plain versioned configuration; never executable code or trained state. Required.

        Returns:
            A validated builder/configuration object; no executable names are imported.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            ModelBuilder.from_config(config)

        HYSo Guide: models.model-builder
        Offline: hyso docs show models.model-builder --lang en
        Open: hyso docs open models.model-builder --lang en
        """
        checked, _ = cls._validate(config)
        result = cls.__new__(cls)
        result._config = checked
        return result

    def to_config(self) -> dict[str, Any]:
        """Return a detached JSON-compatible configuration including defaults.

        Returns:
            Detached JSON-compatible architecture metadata including normalized defaults; no weights.

        Raises:
            No additional validation for a valid object; corrupt retained state or native copy/serialization failures propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_config()

        HYSo Guide: models.model-builder
        Offline: hyso docs show models.model-builder --lang en
        Open: hyso docs open models.model-builder --lang en
        """
        return _plain(self._config)

    @classmethod
    def preflight(cls, config: dict[str, Any], *, batch_size: int = 1,
                  activation_limit: int = 16_777_216) -> dict[str, Any]:
        """Validate architecture data and infer symbolic-N shapes without weights.

        Sequential profiles expose exact per-layer CHW geometry and the existing
        conservative parameter estimate. Other profiles retain their existing
        configuration validation, with spatial shapes explicitly unspecified.
        After a failure, independent scalar/block errors may also be reported.
        Downstream shapes are omitted when a predecessor cannot be validated.
        batch_size is an explicit activation-count assumption, while statistical
        conditions retain symbolic N. activation_limit is an advisory per-tensor
        element threshold and preserves acceptance of existing configurations.

        Args:
            config (dict[str, Any]): Plain versioned configuration; never executable code or trained state. Required.
            batch_size (int): Positive batch-size setting or metadata assumption; no implicit batch search. Default: 1.
            activation_limit (int): Advisory per-tensor element threshold used by metadata preflight, not a physical-memory quota. Default: 16777216.

        Returns:
            Metadata validation, shapes, diagnostics and advisory resource estimates; no model is built.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            ModelBuilder.preflight(config)

        HYSo Guide: models.model-builder
        Offline: hyso docs show models.model-builder --lang en
        Open: hyso docs open models.model-builder --lang en
        """
        report = dict(valid=False, diagnostics=[], warnings=[], input_shape=None, output_shape=None,
                      feature_shape=None, layers=[], parameter_estimate=None,
                      shape_scope='configuration_only')
        try:
            if type(batch_size) is not int or batch_size < 1 or type(activation_limit) is not int or activation_limit < 1:
                raise ModelConfigError('preflight batch_size and activation_limit must be positive integers',
                                       expected='positive integers', actual=dict(batch_size=batch_size, activation_limit=activation_limit))
            checked, shape = cls._validate(config)
        except ModelConfigError as error:
            report['diagnostics'].append(error.to_dict())
            # Scalar/Registry checks do not require a valid predecessor shape.
            try:
                plain = _plain(config)
            except ModelConfigError:
                return report
            if type(plain) is not dict or plain.get('kind') not in ('sequential_classifier','sequential') or type(plain.get('blocks')) is not list:
                return report
            reg = ModelRegistry()
            for index, block in enumerate(plain['blocks'][:64]):
                try:
                    if type(block) is not dict or set(block) != {'name', 'params'}:
                        raise ModelConfigError(f'block {index}: only name/params are allowed; no custom connections',
                                               code='INVALID_BLOCK_ORDER', layer_index=index,
                                               expected=['name', 'params'], actual=block)
                    if reg.get(block['name']).category == 'model':
                        raise ModelConfigError('a complete model cannot be inserted as an NCHW block',
                                               code='INVALID_BLOCK_ORDER', expected='NCHW block', actual=block['name'])
                    reg.validate(block['name'], block['params'])
                except ModelConfigError as independent:
                    diagnostic = independent.at_layer(index).to_dict()
                    if diagnostic not in report['diagnostics']:
                        report['diagnostics'].append(diagnostic)
            return report
        report['valid'] = True
        classes = checked.get('num_classes', checked.get('params', {}).get('num_classes'))
        if checked['kind']=='sequential':
            _, head_info = _validate_head(checked['head'],shape)
            classes = head_info['output_features']
        report['output_shape'] = ['N', classes]
        if checked['kind'] not in ('sequential_classifier','sequential'):
            report['input_shape'] = ['N', checked['params']['input_channels'], None, None]
            return report
        report['shape_scope'] = 'sequential_nchw'
        current = tuple(checked['input_shape'])
        report['input_shape'] = ['N', *current]
        elements = [batch_size*current[0]*current[1]*current[2]]
        report['parameter_estimate_kind'] = 'conservative_upper_bound'
        report['activation_estimate'] = dict(batch_size=batch_size, warning_threshold=activation_limit,
                                              scope='input, sequential boundaries and new convolution internal stages',
                                              peak_tensor_elements=elements[0], summed_tensor_elements=elements[0],
                                              memory_guarantee=False)
        if elements[0] > activation_limit:
            report['warnings'].append(ModelConfigError('input activation exceeds the advisory element threshold',
                code='RESOURCE_LIMIT_WARNING', expected=activation_limit, actual=elements[0]).to_dict())
        cost = 0
        for index, block in enumerate(checked['blocks']):
            output = _shape(block['name'], block['params'], current)
            name, params = block['name'], block['params']
            stages = components.stages(name, params, current) if name in components.CONVOLUTIONS | components.POOLING else [output]
            details = None
            if name in advanced.NEW_NAMES:
                try:
                    details = advanced.describe(name,params,current,batch_size,ModelConfigError)
                except ModelConfigError as error:
                    report.update(valid=False, output_shape=None, feature_shape=None, layers=[])
                    report['diagnostics'].append(error.at_layer(index).to_dict())
                    return report
                stages = details['stages']
            stage_elements = [batch_size*math.prod(stage) for stage in stages]
            elements.extend(stage_elements)
            warnings = [ModelConfigError(requirement, code='STATISTICS_REQUIREMENT', layer_index=index,
                        expected=requirement, actual=dict(shape=['N', *current], batch='symbolic')).to_dict()
                        for requirement in components.statistics_requirements(name, params, current)]
            if details is not None:
                warnings.extend(ModelConfigError(requirement,code='STATISTICS_REQUIREMENT',layer_index=index,
                                expected=requirement,actual=dict(batch='symbolic')).to_dict()
                                for requirement in details['requirements'])
            if max(stage_elements) > activation_limit:
                warnings.append(ModelConfigError('layer activation exceeds the advisory element threshold',
                                code='RESOURCE_LIMIT_WARNING', layer_index=index,
                                expected=activation_limit, actual=max(stage_elements)).to_dict())
            estimate = _cost(name, params)
            exact = components.parameter_count(name, params) if name in components.NEW_NAMES | {'conv2d','batch_norm','group_norm','relu','silu','hardswish','identity'} else None
            if details is not None:
                exact = details['parameter_count']
            if name in sampling.SAMPLING | sampling.UTILITIES:
                exact = 0
            report['layers'].append(dict(index=index, name=block['name'],
                                         input_shape=['N', *current], output_shape=['N', *output],
                                         intermediate_shapes=[['N', *s] for s in stages[:-1]],
                                         parameter_estimate=estimate, parameter_estimate_kind='conservative_upper_bound',
                                         parameter_count=exact, activation_elements=stage_elements, warnings=warnings))
            if details is not None:
                report['layers'][-1].update(attention_cost=details['attention'],projection=details['projection'],
                                            residual_active=details['residual_active'])
            report['warnings'].extend(warnings)
            cost += estimate
            current = output
        report['feature_shape'] = ['N', *shape]
        if checked['kind']=='sequential':
            report['head'] = dict(checked['head'],**head_info,input_shape=['N',*shape],output_shape=['N',classes])
            report['shape_scope'] = 'sequential_features'
            report['parameter_estimate'] = cost + head_info['parameter_count']
            head_elements = [batch_size*head_info['in_features'],batch_size*classes]
            elements.extend(head_elements)
            report['head']['activation_elements'] = head_elements
            report['head']['warnings'] = []
            report['activation_estimate']['scope'] += ', explicit head feature/output boundaries'
            if max(head_elements)>activation_limit:
                warning = ModelConfigError('head activation exceeds the advisory element threshold',
                            code='RESOURCE_LIMIT_WARNING',layer_id='head',expected=activation_limit,
                            actual=max(head_elements)).to_dict()
                report['head']['warnings'].append(warning)
                report['warnings'].append(warning)
        else:
            report['parameter_estimate'] = cost + (shape[0]+1)*classes
        report['activation_estimate']['peak_tensor_elements'] = max(elements)
        report['activation_estimate']['summed_tensor_elements'] = sum(elements)
        return report

    def conv2d(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] | list[int] = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        padding: int | tuple[int, int] | list[int] = 1,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, int] | list[int] = 1,
    ) -> ModelBuilder:
        """Append a registered 2D convolution.

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a block without building or training a model.

        Args:
            in_channels: Expected input channel count. Required.
            out_channels: Output channel count. Required.
            kernel_size: Positive spatial filter/window size, as an integer or H/W pair. Default: 3.
            stride: Positive spatial sampling step. Default: 1.
            padding: Nonnegative spatial padding. Default: 1.
            groups: Group count; input and output channels must be divisible by it. Default: 1.
            bias: Include the convolution bias. Default: True.
            dilation: Positive spacing between filter/window taps. Default: 1.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
            builder.conv2d(in_channels=1, out_channels=4)

        HYSo Guide: components.convolution
        Offline: hyso docs show components.convolution --lang en
        Open: hyso docs open components.convolution --lang en
        """
        return self.add(
            'conv2d',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )

    def conv_norm_act(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] | list[int] = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        padding: int | tuple[int, int] | list[int] = 1,
        groups: int = 1,
        norm: _typing.Literal['batch', 'group', None] = 'batch',
        activation: _typing.Literal['silu', 'relu', None] = 'silu',
    ) -> ModelBuilder:
        """Append convolution, optional normalization and activation.

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a block without building or training a model.

        Args:
            in_channels: Expected input channel count. Required.
            out_channels: Output channel count. Required.
            kernel_size: Positive spatial filter/window size, as an integer or H/W pair. Default: 3.
            stride: Positive spatial sampling step. Default: 1.
            padding: Nonnegative spatial padding. Default: 1.
            groups: Group count; input and output channels must be divisible by it. Default: 1.
            norm: Convolution normalization: batch, group, or None. Default: 'batch'.
            activation: Convolution activation: silu, relu, or None. Default: 'silu'.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
            builder.conv_norm_act(in_channels=1, out_channels=4)

        HYSo Guide: components.convolution
        Offline: hyso docs show components.convolution --lang en
        Open: hyso docs open components.convolution --lang en
        """
        return self.add(
            'conv_norm_act',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            norm=norm,
            activation=activation,
        )

    def relu(self) -> ModelBuilder:
        """Append out-of-place ReLU activation.

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a block without building or training a model.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
            builder.relu()

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add('relu')

    def silu(self) -> ModelBuilder:
        """Append out-of-place SiLU activation.

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a block without building or training a model.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
            builder.silu()

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add('silu')

    def gelu(self, *, approximate: _typing.Literal['none', 'tanh'] = 'none') -> ModelBuilder:
        """Append GELU activation with the selected approximation.

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a block without building or training a model.

        Args:
            approximate: GELU approximation: none (exact) or tanh. Default: 'none'.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
            builder.gelu()

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add('gelu', approximate=approximate)

    def batch_norm(
        self,
        *,
        num_features: int,
        eps: float = 1e-05,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> ModelBuilder:
        """Append registered 2D batch normalization.

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a block without building or training a model.

        Args:
            num_features: Expected feature-channel count. Required.
            eps: Positive normalization stabilizer. Default: 1e-05.
            momentum: Running-statistic factor in [0, 1]; None uses the cumulative average. Default: 0.1.
            affine: Learn normalization scale and offset. Default: True.
            track_running_stats: Keep running mean/variance for evaluation. Default: True.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
            builder.batch_norm(num_features=4)

        HYSo Guide: components.normalization
        Offline: hyso docs show components.normalization --lang en
        Open: hyso docs open components.normalization --lang en
        """
        return self.add(
            'batch_norm',
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def group_norm(
        self,
        *,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-05,
        affine: bool = True,
    ) -> ModelBuilder:
        """Append normalization across explicit channel groups.

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a block without building or training a model.

        Args:
            num_groups: Positive normalization group count dividing num_channels. Required.
            num_channels: Expected channel count for group normalization. Required.
            eps: Positive normalization stabilizer. Default: 1e-05.
            affine: Learn normalization scale and offset. Default: True.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
            builder.group_norm(num_groups=1, num_channels=4)

        HYSo Guide: components.normalization
        Offline: hyso docs show components.normalization --lang en
        Open: hyso docs open components.normalization --lang en
        """
        return self.add('group_norm', num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)

    def max_pool2d(
        self,
        *,
        kernel_size: int | tuple[int, int] | list[int],
        stride: int | tuple[int, int] | list[int] | None = None,
        padding: int | tuple[int, int] | list[int] = 0,
        dilation: int | tuple[int, int] | list[int] = 1,
        ceil_mode: bool = False,
    ) -> ModelBuilder:
        """Append spatial maximum pooling.

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a block without building or training a model.

        Args:
            kernel_size: Positive spatial filter/window size, as an integer or H/W pair. Required.
            stride: Spatial sampling step; None in pooling uses the kernel size. Default: None.
            padding: Nonnegative spatial padding; pooling limits still apply. Default: 0.
            dilation: Positive spacing between filter/window taps. Default: 1.
            ceil_mode: Use the native ceil-window pooling rule instead of floor. Default: False.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
            builder.max_pool2d(kernel_size=2)

        HYSo Guide: components.pooling
        Offline: hyso docs show components.pooling --lang en
        Open: hyso docs open components.pooling --lang en
        """
        return self.add(
            'max_pool2d',
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )

    def avg_pool2d(
        self,
        *,
        kernel_size: int | tuple[int, int] | list[int],
        stride: int | tuple[int, int] | list[int] | None = None,
        padding: int | tuple[int, int] | list[int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> ModelBuilder:
        """Append spatial average pooling.

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a block without building or training a model.

        Args:
            kernel_size: Positive spatial filter/window size, as an integer or H/W pair. Required.
            stride: Spatial sampling step; None in pooling uses the kernel size. Default: None.
            padding: Nonnegative spatial padding; pooling limits still apply. Default: 0.
            ceil_mode: Use the native ceil-window pooling rule instead of floor. Default: False.
            count_include_pad: Include padded positions in the average divisor. Default: True.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
            builder.avg_pool2d(kernel_size=2)

        HYSo Guide: components.pooling
        Offline: hyso docs show components.pooling --lang en
        Open: hyso docs open components.pooling --lang en
        """
        return self.add(
            'avg_pool2d',
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )

    def adaptive_avg_pool2d(self, *, output_size: int | tuple[int, int] | list[int]) -> ModelBuilder:
        """Append average pooling to a fixed spatial size.

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a block without building or training a model.

        Args:
            output_size: Positive target spatial H/W size. Required.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
            builder.adaptive_avg_pool2d(output_size=2)

        HYSo Guide: components.pooling
        Offline: hyso docs show components.pooling --lang en
        Open: hyso docs open components.pooling --lang en
        """
        return self.add('adaptive_avg_pool2d', output_size=output_size)

    def dropout(self, *, p: float = 0.5) -> ModelBuilder:
        """Append elementwise dropout; evaluation leaves values unchanged.

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a block without building or training a model.

        Args:
            p: Element drop probability in [0, 1]; active only during training. Default: 0.5.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
            builder.dropout()

        HYSo Guide: components.regularization
        Offline: hyso docs show components.regularization --lang en
        Open: hyso docs open components.regularization --lang en
        """
        return self.add('dropout', p=p)

    def residual(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int] | list[int] = 1,
        norm: _typing.Literal['batch', 'group', None] = 'batch',
        attention: dict[str, _typing.Any] | None = None,
    ) -> ModelBuilder:
        """Append the registered residual block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            in_channels: in channels Required.
            out_channels: out channels Required.
            stride: stride Default: 1.
            norm: Convolution normalization; group uses largest divisor <=8. Default: 'batch'.
            attention: None or {name, params} for a registered attention with matching output channels. Default: None.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.residual(in_channels=in_channels, out_channels=out_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        return self.add(
            'residual',
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            norm=norm,
            attention=attention,
        )

    def se(self, *, channels: int, reduction: int = 8) -> ModelBuilder:
        """Append the registered se block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            channels: channels Required.
            reduction: reduction Default: 8.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.se(channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add('se', channels=channels, reduction=reduction)

    def cbam(self, *, channels: int, reduction: int = 16, kernel_size: int = 7) -> ModelBuilder:
        """Append the registered cbam block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            channels: channels Required.
            reduction: reduction Default: 16.
            kernel_size: kernel size Default: 7.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.cbam(channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add('cbam', channels=channels, reduction=reduction, kernel_size=kernel_size)

    def eca(self, *, channels: int, kernel_size: int = 3) -> ModelBuilder:
        """Append the registered eca block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            channels: channels Required.
            kernel_size: kernel size Default: 3.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.eca(channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add('eca', channels=channels, kernel_size=kernel_size)

    def coordinate(self, *, channels: int, reduction: int = 32) -> ModelBuilder:
        """Append the registered coordinate block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            channels: channels Required.
            reduction: reduction Default: 32.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.coordinate(channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add('coordinate', channels=channels, reduction=reduction)

    def hardswish(self) -> ModelBuilder:
        """Append the registered hardswish block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.hardswish()

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add('hardswish')

    def identity(self) -> ModelBuilder:
        """Append the registered identity block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.identity()

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add('identity')

    @classmethod
    def modular_cnn(
        cls,
        *,
        input_channels: int = 3,
        num_classes: int = 10,
        widths: tuple[int, ...] | list[int] = (32, 64, 128),
        repeats: tuple[int, ...] | list[int] = (1, 1, 1),
        strides: tuple[int, ...] | list[int] = (1, 2, 2),
        attention: _typing.Literal['none', 'se', 'cbam', 'eca', 'coordinate'] = 'none',
        norm: _typing.Literal['batch', 'group', None] = 'batch',
    ) -> ModelBuilder:
        """Configure the complete modular_cnn model via the existing modular factory.

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            input_channels: Number of input image/feature channels. Default: 3.
            num_classes: num classes Default: 10.
            widths: Ordered feature widths for the architecture stages. Default: (32, 64, 128).
            repeats: Number of independently recorded measurement windows/repeats. Default: (1, 1, 1).
            strides: Ordered spatial sampling steps for the architecture stages. Default: (1, 2, 2).
            attention: attention Default: 'none'.
            norm: Convolution normalization; group uses largest divisor <=8. Default: 'batch'.

        Returns:
            A fresh ModelBuilder configuration.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            ModelBuilder.modular_cnn()

        HYSo Guide: models.support
        Offline: hyso docs show models.support --lang en
        Open: hyso docs open models.support --lang en
        """
        return cls.modular(
            input_channels=input_channels,
            num_classes=num_classes,
            widths=widths,
            repeats=repeats,
            strides=strides,
            attention=attention,
            norm=norm,
        )

    def channel_shuffle(self, *, channels: int, groups: int = 2) -> ModelBuilder:
        """Append the registered channel_shuffle block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            channels: channels Required.
            groups: groups Default: 2.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.channel_shuffle(channels=channels)

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add('channel_shuffle', channels=channels, groups=groups)

    def dropblock2d(self, *, p: float = 0.1, block_size: int = 3) -> ModelBuilder:
        """Append the registered dropblock2d block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            p: Probability of dropping an element/path during training. Default: 0.1.
            block_size: Spatial side length of the dropped block. Default: 3.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.dropblock2d()

        HYSo Guide: components.regularization
        Offline: hyso docs show components.regularization --lang en
        Open: hyso docs open components.regularization --lang en
        """
        return self.add('dropblock2d', p=p, block_size=block_size)

    def pointwise_conv2d(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int] | list[int] = 1,
        bias: bool = True,
    ) -> ModelBuilder:
        """Append the registered pointwise_conv2d block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            in_channels: in channels Required.
            out_channels: out channels Required.
            stride: stride Default: 1.
            bias: bias Default: True.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.pointwise_conv2d(in_channels=in_channels, out_channels=out_channels)

        HYSo Guide: components.convolution
        Offline: hyso docs show components.convolution --lang en
        Open: hyso docs open components.convolution --lang en
        """
        return self.add(
            'pointwise_conv2d',
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bias=bias,
        )

    def depthwise_conv2d(
        self,
        *,
        in_channels: int,
        depth_multiplier: int = 1,
        kernel_size: int | tuple[int, int] | list[int] = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        padding: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        bias: bool = True,
    ) -> ModelBuilder:
        """Append the registered depthwise_conv2d block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            in_channels: in channels Required.
            depth_multiplier: Depthwise filters per input channel. Default: 1.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            padding: padding Default: 1.
            dilation: dilation Default: 1.
            bias: bias Default: True.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.depthwise_conv2d(in_channels=in_channels)

        HYSo Guide: components.convolution
        Offline: hyso docs show components.convolution --lang en
        Open: hyso docs open components.convolution --lang en
        """
        return self.add(
            'depthwise_conv2d',
            in_channels=in_channels,
            depth_multiplier=depth_multiplier,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def depthwise_separable_conv2d(
        self,
        *,
        in_channels: int,
        out_channels: int,
        depth_multiplier: int = 1,
        kernel_size: int | tuple[int, int] | list[int] = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        padding: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        bias: bool = True,
    ) -> ModelBuilder:
        """Append the registered depthwise_separable_conv2d block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            in_channels: in channels Required.
            out_channels: out channels Required.
            depth_multiplier: Depthwise filters per input channel. Default: 1.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            padding: padding Default: 1.
            dilation: dilation Default: 1.
            bias: bias Default: True.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.depthwise_separable_conv2d(in_channels=in_channels, out_channels=out_channels)

        HYSo Guide: components.convolution
        Offline: hyso docs show components.convolution --lang en
        Open: hyso docs open components.convolution --lang en
        """
        return self.add(
            'depthwise_separable_conv2d',
            in_channels=in_channels,
            out_channels=out_channels,
            depth_multiplier=depth_multiplier,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def spatial_separable_conv2d(
        self,
        *,
        in_channels: int,
        out_channels: int,
        intermediate_channels: int,
        kernel_size: int | tuple[int, int] | list[int] = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        padding: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        bias: bool = True,
    ) -> ModelBuilder:
        """Append the registered spatial_separable_conv2d block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            in_channels: in channels Required.
            out_channels: out channels Required.
            intermediate_channels: Internal feature width between projections. Required.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            padding: padding Default: 1.
            dilation: dilation Default: 1.
            bias: bias Default: True.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.spatial_separable_conv2d(in_channels=in_channels, out_channels=out_channels, intermediate_channels=intermediate_channels)

        HYSo Guide: components.convolution
        Offline: hyso docs show components.convolution --lang en
        Open: hyso docs open components.convolution --lang en
        """
        return self.add(
            'spatial_separable_conv2d',
            in_channels=in_channels,
            out_channels=out_channels,
            intermediate_channels=intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def factorized_conv2d(
        self,
        *,
        in_channels: int,
        out_channels: int,
        intermediate_channels: int,
        kernel_size: int | tuple[int, int] | list[int] = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        padding: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        bias: bool = True,
    ) -> ModelBuilder:
        """Append the registered factorized_conv2d block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            in_channels: in channels Required.
            out_channels: out channels Required.
            intermediate_channels: Internal feature width between projections. Required.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            padding: padding Default: 1.
            dilation: dilation Default: 1.
            bias: bias Default: True.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.factorized_conv2d(in_channels=in_channels, out_channels=out_channels, intermediate_channels=intermediate_channels)

        HYSo Guide: components.convolution
        Offline: hyso docs show components.convolution --lang en
        Open: hyso docs open components.convolution --lang en
        """
        return self.add(
            'factorized_conv2d',
            in_channels=in_channels,
            out_channels=out_channels,
            intermediate_channels=intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def conv_transpose2d(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] | list[int] = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        padding: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        bias: bool = True,
        groups: int = 1,
        output_padding: int | tuple[int, int] | list[int] = 0,
    ) -> ModelBuilder:
        """Append the registered conv_transpose2d block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            in_channels: in channels Required.
            out_channels: out channels Required.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            padding: padding Default: 1.
            dilation: dilation Default: 1.
            bias: bias Default: True.
            groups: groups Default: 1.
            output_padding: Extra output size on one side of each transposed-convolution axis. Default: 0.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.conv_transpose2d(in_channels=in_channels, out_channels=out_channels)

        HYSo Guide: components.convolution
        Offline: hyso docs show components.convolution --lang en
        Open: hyso docs open components.convolution --lang en
        """
        return self.add(
            'conv_transpose2d',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
            output_padding=output_padding,
        )

    def channel_layer_norm(self, *, channels: int, eps: float = 1e-05, affine: bool = True) -> ModelBuilder:
        """Append the registered channel_layer_norm block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            channels: channels Required.
            eps: Small numerical stabilizer for normalization. Default: 1e-05.
            affine: Learn a normalization scale and offset. Default: True.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.channel_layer_norm(channels=channels)

        HYSo Guide: components.normalization
        Offline: hyso docs show components.normalization --lang en
        Open: hyso docs open components.normalization --lang en
        """
        return self.add('channel_layer_norm', channels=channels, eps=eps, affine=affine)

    def instance_norm(
        self,
        *,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> ModelBuilder:
        """Append the registered instance_norm block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            num_features: num features Required.
            eps: Small numerical stabilizer for normalization. Default: 1e-05.
            momentum: Running-statistic update factor in normalization. Default: 0.1.
            affine: Learn a normalization scale and offset. Default: False.
            track_running_stats: Maintain running mean/variance for evaluation. Default: False.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.instance_norm(num_features=num_features)

        HYSo Guide: components.normalization
        Offline: hyso docs show components.normalization --lang en
        Open: hyso docs open components.normalization --lang en
        """
        return self.add(
            'instance_norm',
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def leaky_relu(self, *, negative_slope: float = 0.01) -> ModelBuilder:
        """Append the registered leaky_relu block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            negative_slope: negative slope Default: 0.01.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.leaky_relu()

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add('leaky_relu', negative_slope=negative_slope)

    def prelu(self, *, num_parameters: int = 1, init: float = 0.25) -> ModelBuilder:
        """Append the registered prelu block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            num_parameters: num parameters Default: 1.
            init: init Default: 0.25.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.prelu()

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add('prelu', num_parameters=num_parameters, init=init)

    def mish(self) -> ModelBuilder:
        """Append the registered mish block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.mish()

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add('mish')

    def hardsigmoid(self) -> ModelBuilder:
        """Append the registered hardsigmoid block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.hardsigmoid()

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add('hardsigmoid')

    def adaptive_max_pool2d(self, *, output_size: int | tuple[int, int] | list[int]) -> ModelBuilder:
        """Append the registered adaptive_max_pool2d block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            output_size: output size Required.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.adaptive_max_pool2d(output_size=output_size)

        HYSo Guide: components.pooling
        Offline: hyso docs show components.pooling --lang en
        Open: hyso docs open components.pooling --lang en
        """
        return self.add('adaptive_max_pool2d', output_size=output_size)

    def global_avg_pool2d(self) -> ModelBuilder:
        """Append the registered global_avg_pool2d block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.global_avg_pool2d()

        HYSo Guide: components.pooling
        Offline: hyso docs show components.pooling --lang en
        Open: hyso docs open components.pooling --lang en
        """
        return self.add('global_avg_pool2d')

    def bottleneck_residual(
        self,
        *,
        in_channels: int,
        out_channels: int,
        intermediate_channels: int,
        kernel_size: int = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        normalization: _typing.Literal['batch', 'group', None] = 'batch',
        activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu',
        attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none',
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> ModelBuilder:
        """Append the registered bottleneck_residual block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            in_channels: in channels Required.
            out_channels: out channels Required.
            intermediate_channels: Internal feature width between projections. Required.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            dilation: dilation Default: 1.
            normalization: Closed normalization choice for the residual path. Default: 'batch'.
            activation: Supported activation family. Default: 'silu'.
            attention: attention Default: 'none'.
            dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.
            drop_path: drop path Default: 0.0.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.bottleneck_residual(in_channels=in_channels, out_channels=out_channels, intermediate_channels=intermediate_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        return self.add(
            'bottleneck_residual',
            in_channels=in_channels,
            out_channels=out_channels,
            intermediate_channels=intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            normalization=normalization,
            activation=activation,
            attention=attention,
            dropout=dropout,
            drop_path=drop_path,
        )

    def preactivation_residual(
        self,
        *,
        in_channels: int,
        out_channels: int,
        intermediate_channels: int,
        kernel_size: int = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        normalization: _typing.Literal['batch', 'group', None] = 'batch',
        activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu',
        attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none',
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> ModelBuilder:
        """Append the registered preactivation_residual block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            in_channels: in channels Required.
            out_channels: out channels Required.
            intermediate_channels: Internal feature width between projections. Required.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            dilation: dilation Default: 1.
            normalization: Closed normalization choice for the residual path. Default: 'batch'.
            activation: Supported activation family. Default: 'silu'.
            attention: attention Default: 'none'.
            dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.
            drop_path: drop path Default: 0.0.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.preactivation_residual(in_channels=in_channels, out_channels=out_channels, intermediate_channels=intermediate_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        return self.add(
            'preactivation_residual',
            in_channels=in_channels,
            out_channels=out_channels,
            intermediate_channels=intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            normalization=normalization,
            activation=activation,
            attention=attention,
            dropout=dropout,
            drop_path=drop_path,
        )

    def wide_residual_block(
        self,
        *,
        in_channels: int,
        out_channels: int,
        intermediate_channels: int,
        kernel_size: int = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        normalization: _typing.Literal['batch', 'group', None] = 'batch',
        activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu',
        attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none',
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> ModelBuilder:
        """Append the registered wide_residual_block block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            in_channels: in channels Required.
            out_channels: out channels Required.
            intermediate_channels: Internal feature width between projections. Required.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            dilation: dilation Default: 1.
            normalization: Closed normalization choice for the residual path. Default: 'batch'.
            activation: Supported activation family. Default: 'silu'.
            attention: attention Default: 'none'.
            dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.
            drop_path: drop path Default: 0.0.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.wide_residual_block(in_channels=in_channels, out_channels=out_channels, intermediate_channels=intermediate_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        return self.add(
            'wide_residual_block',
            in_channels=in_channels,
            out_channels=out_channels,
            intermediate_channels=intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            normalization=normalization,
            activation=activation,
            attention=attention,
            dropout=dropout,
            drop_path=drop_path,
        )

    def grouped_bottleneck(
        self,
        *,
        in_channels: int,
        out_channels: int,
        intermediate_channels: int,
        kernel_size: int = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        normalization: _typing.Literal['batch', 'group', None] = 'batch',
        activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu',
        attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none',
        dropout: float = 0.0,
        drop_path: float = 0.0,
        groups: int = 1,
    ) -> ModelBuilder:
        """Append the registered grouped_bottleneck block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            in_channels: in channels Required.
            out_channels: out channels Required.
            intermediate_channels: Internal feature width between projections. Required.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            dilation: dilation Default: 1.
            normalization: Closed normalization choice for the residual path. Default: 'batch'.
            activation: Supported activation family. Default: 'silu'.
            attention: attention Default: 'none'.
            dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.
            drop_path: drop path Default: 0.0.
            groups: groups Default: 1.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.grouped_bottleneck(in_channels=in_channels, out_channels=out_channels, intermediate_channels=intermediate_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        return self.add(
            'grouped_bottleneck',
            in_channels=in_channels,
            out_channels=out_channels,
            intermediate_channels=intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            normalization=normalization,
            activation=activation,
            attention=attention,
            dropout=dropout,
            drop_path=drop_path,
            groups=groups,
        )

    def inverted_bottleneck(
        self,
        *,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int = 2,
        kernel_size: int = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        normalization: _typing.Literal['batch', 'group', None] = 'batch',
        activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu',
        attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none',
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> ModelBuilder:
        """Append the registered inverted_bottleneck block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            in_channels: in channels Required.
            out_channels: out channels Required.
            expansion_ratio: Internal positive integer channel expansion factor. Default: 2.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            dilation: dilation Default: 1.
            normalization: Closed normalization choice for the residual path. Default: 'batch'.
            activation: Supported activation family. Default: 'silu'.
            attention: attention Default: 'none'.
            dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.
            drop_path: drop path Default: 0.0.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.inverted_bottleneck(in_channels=in_channels, out_channels=out_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        return self.add(
            'inverted_bottleneck',
            in_channels=in_channels,
            out_channels=out_channels,
            expansion_ratio=expansion_ratio,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            normalization=normalization,
            activation=activation,
            attention=attention,
            dropout=dropout,
            drop_path=drop_path,
        )

    def gated_residual(
        self,
        *,
        in_channels: int,
        out_channels: int,
        intermediate_channels: int,
        kernel_size: int = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        normalization: _typing.Literal['batch', 'group', None] = 'batch',
        activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu',
        attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none',
        dropout: float = 0.0,
        drop_path: float = 0.0,
        gate_init: float = 1.0,
        learnable_gate: bool = True,
    ) -> ModelBuilder:
        """Append the registered gated_residual block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            in_channels: in channels Required.
            out_channels: out channels Required.
            intermediate_channels: Internal feature width between projections. Required.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            dilation: dilation Default: 1.
            normalization: Closed normalization choice for the residual path. Default: 'batch'.
            activation: Supported activation family. Default: 'silu'.
            attention: attention Default: 'none'.
            dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.
            drop_path: drop path Default: 0.0.
            gate_init: Initial residual gate value in [0, 1]. Default: 1.0.
            learnable_gate: Make the residual gate trainable. Default: True.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.gated_residual(in_channels=in_channels, out_channels=out_channels, intermediate_channels=intermediate_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        return self.add(
            'gated_residual',
            in_channels=in_channels,
            out_channels=out_channels,
            intermediate_channels=intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            normalization=normalization,
            activation=activation,
            attention=attention,
            dropout=dropout,
            drop_path=drop_path,
            gate_init=gate_init,
            learnable_gate=learnable_gate,
        )

    def spatial_attention(self, *, channels: int, kernel_size: int = 7) -> ModelBuilder:
        """Append the registered spatial_attention block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            channels: channels Required.
            kernel_size: kernel size Default: 7.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.spatial_attention(channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add('spatial_attention', channels=channels, kernel_size=kernel_size)

    def channel_attention(self, *, channels: int, reduction: int = 8) -> ModelBuilder:
        """Append the registered channel_attention block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            channels: channels Required.
            reduction: reduction Default: 8.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.channel_attention(channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add('channel_attention', channels=channels, reduction=reduction)

    def spatial_self_attention(
        self,
        *,
        channels: int,
        embedding_dim: int | None = None,
        dropout: float = 0.0,
    ) -> ModelBuilder:
        """Append the registered spatial_self_attention block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            channels: channels Required.
            embedding_dim: embedding dim Default: None.
            dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.spatial_self_attention(channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add(
            'spatial_self_attention',
            channels=channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

    def multihead_spatial_attention(
        self,
        *,
        channels: int,
        embedding_dim: int | None = None,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> ModelBuilder:
        """Append the registered multihead_spatial_attention block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            channels: channels Required.
            embedding_dim: embedding dim Default: None.
            num_heads: num heads Default: 4.
            dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.multihead_spatial_attention(channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add(
            'multihead_spatial_attention',
            channels=channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def dropout2d(self, *, p: float = 0.5) -> ModelBuilder:
        """Append the registered dropout2d block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            p: Probability of dropping an element/path during training. Default: 0.5.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.dropout2d()

        HYSo Guide: components.regularization
        Offline: hyso docs show components.regularization --lang en
        Open: hyso docs open components.regularization --lang en
        """
        return self.add('dropout2d', p=p)

    def drop_path(self, *, p: float = 0.0) -> ModelBuilder:
        """Append the registered drop_path block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            p: Probability of dropping an element/path during training. Default: 0.0.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.drop_path()

        HYSo Guide: components.regularization
        Offline: hyso docs show components.regularization --lang en
        Open: hyso docs open components.regularization --lang en
        """
        return self.add('drop_path', p=p)

    def gaussian_noise(self, *, std: float = 0.1) -> ModelBuilder:
        """Append the registered gaussian_noise block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            std: Standard deviation of sampled noise. Default: 0.1.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.gaussian_noise()

        HYSo Guide: components.regularization
        Offline: hyso docs show components.regularization --lang en
        Open: hyso docs open components.regularization --lang en
        """
        return self.add('gaussian_noise', std=std)

    def resize(
        self,
        *,
        output_size: int | tuple[int, int] | list[int] | None = None,
        scale_factor: float | tuple[float, float] | list[float] | None = None,
        mode: _typing.Literal['nearest', 'bilinear', 'bicubic', 'area'] = 'nearest',
        align_corners: bool | None = None,
    ) -> ModelBuilder:
        """Append the registered resize block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            output_size: output size Default: None.
            scale_factor: scale factor Default: None.
            mode: mode Default: 'nearest'.
            align_corners: align corners Default: None.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.resize(output_size=(4, 4))

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add(
            'resize',
            output_size=output_size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )

    def pixel_shuffle(self, *, upscale_factor: int) -> ModelBuilder:
        """Append the registered pixel_shuffle block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            upscale_factor: Positive spatial upscaling factor; channels must be divisible by its square. Required.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.pixel_shuffle(upscale_factor=upscale_factor)

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add('pixel_shuffle', upscale_factor=upscale_factor)

    def pixel_unshuffle(self, *, downscale_factor: int) -> ModelBuilder:
        """Append the registered pixel_unshuffle block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            downscale_factor: Positive spatial downscaling factor dividing both spatial dimensions. Required.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.pixel_unshuffle(downscale_factor=downscale_factor)

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add('pixel_unshuffle', downscale_factor=downscale_factor)

    def flatten(self) -> ModelBuilder:
        """Append the registered flatten block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.
        Requires schema2 and must remain the final feature block before its explicit head.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.flatten()

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add('flatten')

    def permute(self, *, dims: tuple[int, ...] | list[int] = (0, 1, 3, 2)) -> ModelBuilder:
        """Append the registered permute block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            dims: Explicit axis permutation/order. Default: (0, 1, 3, 2).

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.permute()

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add('permute', dims=dims)

    def reshape(self, *, shape: tuple[int, ...] | list[int]) -> ModelBuilder:
        """Append the registered reshape block via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            shape: shape Required.

        Returns:
            This ModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.reshape(shape=shape)

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add('reshape', shape=shape)

    # BEGIN DEVELOPMENT REGISTRY WRAPPERS
    if DEVELOPMENT:
        def simam(self, *, channels: int, epsilon: float = 0.0001) -> ModelBuilder:
            """Append the registered simam block via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                channels: channels Required.
                epsilon: Numerical tolerance/stabilizer used by this operation. Default: 0.0001.

            Returns:
                This ModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.simam(channels=channels)

            HYSo Guide: components.attention
            Offline: hyso docs show components.attention --lang en
            Open: hyso docs open components.attention --lang en
            """
            return self.add('simam', channels=channels, epsilon=epsilon)
        def triplet(self, *, channels: int, kernel_size: int = 7, spatial: bool = True) -> ModelBuilder:
            """Append the registered triplet block via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                channels: channels Required.
                kernel_size: kernel size Default: 7.
                spatial: Include spatial attention where supported. Default: True.

            Returns:
                This ModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.triplet(channels=channels)

            HYSo Guide: components.attention
            Offline: hyso docs show components.attention --lang en
            Open: hyso docs open components.attention --lang en
            """
            return self.add('triplet', channels=channels, kernel_size=kernel_size, spatial=spatial)
        def selective_kernel(
            self,
            *,
            channels: int,
            branches: int = 2,
            groups: int = 1,
            reduction: int = 16,
            minimum: int = 32,
        ) -> ModelBuilder:
            """Append the registered selective_kernel block via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                channels: channels Required.
                branches: Explicit branch modules/configurations. Default: 2.
                groups: groups Default: 1.
                reduction: reduction Default: 16.
                minimum: Lower bound applied by this operation. Default: 32.

            Returns:
                This ModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.selective_kernel(channels=channels)

            HYSo Guide: components.attention
            Offline: hyso docs show components.attention --lang en
            Open: hyso docs open components.attention --lang en
            """
            return self.add(
                'selective_kernel',
                channels=channels,
                branches=branches,
                groups=groups,
                reduction=reduction,
                minimum=minimum,
            )
        def basic_residual(
            self,
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            norm: _typing.Literal['batch', 'group', None] = 'batch',
        ) -> ModelBuilder:
            """Append the registered basic_residual block via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                norm: M3 normalization; group uses gcd(8, channels). Default: 'batch'.

            Returns:
                This ModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.basic_residual(in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.residual
            Offline: hyso docs show components.residual --lang en
            Open: hyso docs open components.residual --lang en
            """
            return self.add(
                'basic_residual',
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                norm=norm,
            )
        def bottleneck(
            self,
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            norm: _typing.Literal['batch', 'group', None] = 'batch',
            groups: int = 1,
            width_per_group: int = 64,
        ) -> ModelBuilder:
            """Append the registered bottleneck block via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                norm: M3 normalization; group uses gcd(8, channels). Default: 'batch'.
                groups: groups Default: 1.
                width_per_group: Feature width within each convolution group. Default: 64.

            Returns:
                This ModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.bottleneck(in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.residual
            Offline: hyso docs show components.residual --lang en
            Open: hyso docs open components.residual --lang en
            """
            return self.add(
                'bottleneck',
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                norm=norm,
                groups=groups,
                width_per_group=width_per_group,
            )
        def preactivation(
            self,
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            norm: _typing.Literal['batch', 'group', None] = 'batch',
            bottleneck: bool = False,
            dropout: float = 0.0,
        ) -> ModelBuilder:
            """Append the registered preactivation block via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                norm: M3 normalization; group uses gcd(8, channels). Default: 'batch'.
                bottleneck: Select the supported bottleneck block form. Default: False.
                dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.

            Returns:
                This ModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.preactivation(in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.residual
            Offline: hyso docs show components.residual --lang en
            Open: hyso docs open components.residual --lang en
            """
            return self.add(
                'preactivation',
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                norm=norm,
                bottleneck=bottleneck,
                dropout=dropout,
            )
        def wide_residual(
            self,
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            norm: _typing.Literal['batch', 'group', None] = 'batch',
            dropout: float = 0.0,
        ) -> ModelBuilder:
            """Append the registered wide_residual block via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                norm: M3 normalization; group uses gcd(8, channels). Default: 'batch'.
                dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.

            Returns:
                This ModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.wide_residual(in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.residual
            Offline: hyso docs show components.residual --lang en
            Open: hyso docs open components.residual --lang en
            """
            return self.add(
                'wide_residual',
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                norm=norm,
                dropout=dropout,
            )
        def resnext_bottleneck(
            self,
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            norm: _typing.Literal['batch', 'group', None] = 'batch',
            groups: int = 32,
            width_per_group: int = 4,
        ) -> ModelBuilder:
            """Append the registered resnext_bottleneck block via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                norm: M3 normalization; group uses gcd(8, channels). Default: 'batch'.
                groups: groups Default: 32.
                width_per_group: Feature width within each convolution group. Default: 4.

            Returns:
                This ModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.resnext_bottleneck(in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.residual
            Offline: hyso docs show components.residual --lang en
            Open: hyso docs open components.residual --lang en
            """
            return self.add(
                'resnext_bottleneck',
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                norm=norm,
                groups=groups,
                width_per_group=width_per_group,
            )
        @classmethod
        def resnet(
            cls,
            *,
            depth: int = 18,
            input_channels: int = 3,
            num_classes: int = 1000,
            norm: _typing.Literal['batch', 'group', None] = 'batch',
        ) -> ModelBuilder:
            """Configure the complete resnet model via the existing named factory.

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                depth: Declared architecture depth. Default: 18.
                input_channels: Number of input image/feature channels. Default: 3.
                num_classes: num classes Default: 1000.
                norm: M3 normalization; group uses gcd(8, channels). Default: 'batch'.

            Returns:
                A fresh ModelBuilder configuration.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                ModelBuilder.resnet()

            HYSo Guide: models.development-families
            Offline: hyso docs show models.development-families --lang en
            Open: hyso docs open models.development-families --lang en
            """
            return cls.named(
                'resnet',
                depth=depth,
                input_channels=input_channels,
                num_classes=num_classes,
                norm=norm,
            )
        @classmethod
        def preact_resnet(
            cls,
            *,
            depth: int = 18,
            input_channels: int = 3,
            num_classes: int = 1000,
            base_channels: int = 64,
            norm: _typing.Literal['batch', 'group', None] = 'batch',
        ) -> ModelBuilder:
            """Configure the complete preact_resnet model via the existing named factory.

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                depth: Declared architecture depth. Default: 18.
                input_channels: Number of input image/feature channels. Default: 3.
                num_classes: num classes Default: 1000.
                base_channels: Initial feature-channel width. Default: 64.
                norm: M3 normalization; group uses gcd(8, channels). Default: 'batch'.

            Returns:
                A fresh ModelBuilder configuration.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                ModelBuilder.preact_resnet()

            HYSo Guide: models.development-families
            Offline: hyso docs show models.development-families --lang en
            Open: hyso docs open models.development-families --lang en
            """
            return cls.named(
                'preact_resnet',
                depth=depth,
                input_channels=input_channels,
                num_classes=num_classes,
                base_channels=base_channels,
                norm=norm,
            )
        @classmethod
        def wide_resnet(
            cls,
            *,
            depth: int = 28,
            input_channels: int = 3,
            num_classes: int = 10,
            widen_factor: int = 10,
            dropout: float = 0.0,
            norm: _typing.Literal['batch', 'group', None] = 'batch',
        ) -> ModelBuilder:
            """Configure the complete wide_resnet model via the existing named factory.

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                depth: Declared architecture depth. Default: 28.
                input_channels: Number of input image/feature channels. Default: 3.
                num_classes: num classes Default: 10.
                widen_factor: Channel-width multiplier of the wide residual architecture. Default: 10.
                dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.
                norm: M3 normalization; group uses gcd(8, channels). Default: 'batch'.

            Returns:
                A fresh ModelBuilder configuration.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                ModelBuilder.wide_resnet()

            HYSo Guide: models.development-families
            Offline: hyso docs show models.development-families --lang en
            Open: hyso docs open models.development-families --lang en
            """
            return cls.named(
                'wide_resnet',
                depth=depth,
                input_channels=input_channels,
                num_classes=num_classes,
                widen_factor=widen_factor,
                dropout=dropout,
                norm=norm,
            )
        @classmethod
        def resnext(
            cls,
            *,
            depth: int = 50,
            input_channels: int = 3,
            num_classes: int = 1000,
            groups: int = 32,
            width_per_group: int = 4,
            norm: _typing.Literal['batch', 'group', None] = 'batch',
        ) -> ModelBuilder:
            """Configure the complete resnext model via the existing named factory.

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                depth: Declared architecture depth. Default: 50.
                input_channels: Number of input image/feature channels. Default: 3.
                num_classes: num classes Default: 1000.
                groups: groups Default: 32.
                width_per_group: Feature width within each convolution group. Default: 4.
                norm: M3 normalization; group uses gcd(8, channels). Default: 'batch'.

            Returns:
                A fresh ModelBuilder configuration.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                ModelBuilder.resnext()

            HYSo Guide: models.development-families
            Offline: hyso docs show models.development-families --lang en
            Open: hyso docs open models.development-families --lang en
            """
            return cls.named(
                'resnext',
                depth=depth,
                input_channels=input_channels,
                num_classes=num_classes,
                groups=groups,
                width_per_group=width_per_group,
                norm=norm,
            )
        @classmethod
        def densenet(
            cls,
            *,
            depth: int = 121,
            input_channels: int = 3,
            num_classes: int = 1000,
            growth_rate: int = 32,
            dropout: float = 0.0,
        ) -> ModelBuilder:
            """Configure the complete densenet model via the existing named factory.

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                depth: Declared architecture depth. Default: 121.
                input_channels: Number of input image/feature channels. Default: 3.
                num_classes: num classes Default: 1000.
                growth_rate: Feature channels added by each dense unit. Default: 32.
                dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.

            Returns:
                A fresh ModelBuilder configuration.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                ModelBuilder.densenet()

            HYSo Guide: models.development-families
            Offline: hyso docs show models.development-families --lang en
            Open: hyso docs open models.development-families --lang en
            """
            return cls.named(
                'densenet',
                depth=depth,
                input_channels=input_channels,
                num_classes=num_classes,
                growth_rate=growth_rate,
                dropout=dropout,
            )
        def inverted_residual(
            self,
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            expand_ratio: float = 6.0,
        ) -> ModelBuilder:
            """Append the registered inverted_residual block via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                expand_ratio: Internal channel expansion ratio. Default: 6.0.

            Returns:
                This ModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.inverted_residual(in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.mobile
            Offline: hyso docs show components.mobile --lang en
            Open: hyso docs open components.mobile --lang en
            """
            return self.add(
                'inverted_residual',
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
            )
        def mbconv(
            self,
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            expand_ratio: float = 6.0,
            kernel_size: int = 3,
            stochastic_depth: float = 0.0,
        ) -> ModelBuilder:
            """Append the registered mbconv block via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                expand_ratio: Internal channel expansion ratio. Default: 6.0.
                kernel_size: kernel size Default: 3.
                stochastic_depth: Probability/schedule for dropping residual paths during training. Default: 0.0.

            Returns:
                This ModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.mbconv(in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.mobile
            Offline: hyso docs show components.mobile --lang en
            Open: hyso docs open components.mobile --lang en
            """
            return self.add(
                'mbconv',
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                kernel_size=kernel_size,
                stochastic_depth=stochastic_depth,
            )
        def fused_mbconv(
            self,
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            expand_ratio: float = 4.0,
            kernel_size: int = 3,
            stochastic_depth: float = 0.0,
        ) -> ModelBuilder:
            """Append the registered fused_mbconv block via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                expand_ratio: Internal channel expansion ratio. Default: 4.0.
                kernel_size: kernel size Default: 3.
                stochastic_depth: Probability/schedule for dropping residual paths during training. Default: 0.0.

            Returns:
                This ModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.fused_mbconv(in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.mobile
            Offline: hyso docs show components.mobile --lang en
            Open: hyso docs open components.mobile --lang en
            """
            return self.add(
                'fused_mbconv',
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                kernel_size=kernel_size,
                stochastic_depth=stochastic_depth,
            )
        def ghost_conv(
            self,
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            kernel_size: int = 1,
            ratio: int = 2,
            cheap_kernel: int = 3,
            relu: bool = True,
        ) -> ModelBuilder:
            """Append the registered ghost_conv block via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                kernel_size: kernel size Default: 1.
                ratio: Feature expansion/split ratio used by this block. Default: 2.
                cheap_kernel: Spatial kernel of the inexpensive feature-generation branch. Default: 3.
                relu: Enable the block's ReLU activation. Default: True.

            Returns:
                This ModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.ghost_conv(in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.mobile
            Offline: hyso docs show components.mobile --lang en
            Open: hyso docs open components.mobile --lang en
            """
            return self.add(
                'ghost_conv',
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=kernel_size,
                ratio=ratio,
                cheap_kernel=cheap_kernel,
                relu=relu,
            )
        def ghost_bottleneck(
            self,
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            hidden_channels: int,
            kernel_size: int = 3,
            se_ratio: float = 0.0,
        ) -> ModelBuilder:
            """Append the registered ghost_bottleneck block via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                hidden_channels: Internal feature width. Required.
                kernel_size: kernel size Default: 3.
                se_ratio: Squeeze-and-excitation reduction setting. Default: 0.0.

            Returns:
                This ModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.ghost_bottleneck(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels)

            HYSo Guide: components.mobile
            Offline: hyso docs show components.mobile --lang en
            Open: hyso docs open components.mobile --lang en
            """
            return self.add(
                'ghost_bottleneck',
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                se_ratio=se_ratio,
            )
        def shuffle_unit(self, *, in_channels: int, out_channels: int, stride: int = 1) -> ModelBuilder:
            """Append the registered shuffle_unit block via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.

            Returns:
                This ModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.shuffle_unit(in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.mobile
            Offline: hyso docs show components.mobile --lang en
            Open: hyso docs open components.mobile --lang en
            """
            return self.add('shuffle_unit', in_channels=in_channels, out_channels=out_channels, stride=stride)
        def repvgg_block(
            self,
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            groups: int = 1,
            deploy: bool = False,
        ) -> ModelBuilder:
            """Append the registered repvgg_block block via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                groups: groups Default: 1.
                deploy: Use the deployment form of this reparameterized block; conversion support remains explicit. Default: False.

            Returns:
                This ModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.repvgg_block(in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.mobile
            Offline: hyso docs show components.mobile --lang en
            Open: hyso docs open components.mobile --lang en
            """
            return self.add(
                'repvgg_block',
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                groups=groups,
                deploy=deploy,
            )
        def mobileone_block(
            self,
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            kernel_size: int = 3,
            groups: int = 1,
            num_conv_branches: int = 1,
            use_se: bool = False,
            deploy: bool = False,
        ) -> ModelBuilder:
            """Append the registered mobileone_block block via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                kernel_size: kernel size Default: 3.
                groups: groups Default: 1.
                num_conv_branches: Parallel convolution branch count in the training form. Default: 1.
                use_se: Enable squeeze-and-excitation attention. Default: False.
                deploy: Use the deployment form of this reparameterized block; conversion support remains explicit. Default: False.

            Returns:
                This ModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.mobileone_block(in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.mobile
            Offline: hyso docs show components.mobile --lang en
            Open: hyso docs open components.mobile --lang en
            """
            return self.add(
                'mobileone_block',
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=kernel_size,
                groups=groups,
                num_conv_branches=num_conv_branches,
                use_se=use_se,
                deploy=deploy,
            )
        @classmethod
        def mobilenet_v2(
            cls,
            *,
            input_channels: int = 3,
            num_classes: int = 1000,
            width_mult: float = 1.0,
            dropout: float = 0.2,
        ) -> ModelBuilder:
            """Configure the complete mobilenet_v2 model via the existing named factory.

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                input_channels: Number of input image/feature channels. Default: 3.
                num_classes: num classes Default: 1000.
                width_mult: Multiplier for architecture channel widths. Default: 1.0.
                dropout: Drop probability; this regularization is active only in training mode. Default: 0.2.

            Returns:
                A fresh ModelBuilder configuration.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                ModelBuilder.mobilenet_v2()

            HYSo Guide: models.development-families
            Offline: hyso docs show models.development-families --lang en
            Open: hyso docs open models.development-families --lang en
            """
            return cls.named(
                'mobilenet_v2',
                input_channels=input_channels,
                num_classes=num_classes,
                width_mult=width_mult,
                dropout=dropout,
            )
        @classmethod
        def mobilenet_v3(
            cls,
            *,
            input_channels: int = 3,
            num_classes: int = 1000,
            variant: _typing.Literal['small', 'large'] = 'large',
            width_mult: float = 1.0,
            dropout: float = 0.2,
        ) -> ModelBuilder:
            """Configure the complete mobilenet_v3 model via the existing named factory.

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                input_channels: Number of input image/feature channels. Default: 3.
                num_classes: num classes Default: 1000.
                variant: Supported closed architecture variant. Default: 'large'.
                width_mult: Multiplier for architecture channel widths. Default: 1.0.
                dropout: Drop probability; this regularization is active only in training mode. Default: 0.2.

            Returns:
                A fresh ModelBuilder configuration.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                ModelBuilder.mobilenet_v3()

            HYSo Guide: models.development-families
            Offline: hyso docs show models.development-families --lang en
            Open: hyso docs open models.development-families --lang en
            """
            return cls.named(
                'mobilenet_v3',
                input_channels=input_channels,
                num_classes=num_classes,
                variant=variant,
                width_mult=width_mult,
                dropout=dropout,
            )
        @classmethod
        def efficientnet_v2(
            cls,
            *,
            input_channels: int = 3,
            num_classes: int = 1000,
            variant: _typing.Literal['s', 'm', 'l'] = 's',
            width_mult: float = 1.0,
            depth_mult: float = 1.0,
            dropout: float | None = None,
            stochastic_depth: float = 0.2,
        ) -> ModelBuilder:
            """Configure the complete efficientnet_v2 model via the existing named factory.

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                input_channels: Number of input image/feature channels. Default: 3.
                num_classes: num classes Default: 1000.
                variant: Supported closed architecture variant. Default: 's'.
                width_mult: Multiplier for architecture channel widths. Default: 1.0.
                depth_mult: Multiplier for architecture stage depth. Default: 1.0.
                dropout: Drop probability; this regularization is active only in training mode. Default: None.
                stochastic_depth: Probability/schedule for dropping residual paths during training. Default: 0.2.

            Returns:
                A fresh ModelBuilder configuration.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                ModelBuilder.efficientnet_v2()

            HYSo Guide: models.development-families
            Offline: hyso docs show models.development-families --lang en
            Open: hyso docs open models.development-families --lang en
            """
            return cls.named(
                'efficientnet_v2',
                input_channels=input_channels,
                num_classes=num_classes,
                variant=variant,
                width_mult=width_mult,
                depth_mult=depth_mult,
                dropout=dropout,
                stochastic_depth=stochastic_depth,
            )
        @classmethod
        def ghostnet(
            cls,
            *,
            input_channels: int = 3,
            num_classes: int = 1000,
            width_mult: float = 1.0,
            dropout: float = 0.2,
        ) -> ModelBuilder:
            """Configure the complete ghostnet model via the existing named factory.

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                input_channels: Number of input image/feature channels. Default: 3.
                num_classes: num classes Default: 1000.
                width_mult: Multiplier for architecture channel widths. Default: 1.0.
                dropout: Drop probability; this regularization is active only in training mode. Default: 0.2.

            Returns:
                A fresh ModelBuilder configuration.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                ModelBuilder.ghostnet()

            HYSo Guide: models.development-families
            Offline: hyso docs show models.development-families --lang en
            Open: hyso docs open models.development-families --lang en
            """
            return cls.named(
                'ghostnet',
                input_channels=input_channels,
                num_classes=num_classes,
                width_mult=width_mult,
                dropout=dropout,
            )
        @classmethod
        def shufflenet_v2(
            cls,
            *,
            input_channels: int = 3,
            num_classes: int = 1000,
            variant: _typing.Literal['x0_5', 'x1_0', 'x1_5', 'x2_0'] = 'x1_0',
        ) -> ModelBuilder:
            """Configure the complete shufflenet_v2 model via the existing named factory.

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                input_channels: Number of input image/feature channels. Default: 3.
                num_classes: num classes Default: 1000.
                variant: Supported closed architecture variant. Default: 'x1_0'.

            Returns:
                A fresh ModelBuilder configuration.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                ModelBuilder.shufflenet_v2()

            HYSo Guide: models.development-families
            Offline: hyso docs show models.development-families --lang en
            Open: hyso docs open models.development-families --lang en
            """
            return cls.named(
                'shufflenet_v2',
                input_channels=input_channels,
                num_classes=num_classes,
                variant=variant,
            )
        @classmethod
        def repvgg(
            cls,
            *,
            input_channels: int = 3,
            num_classes: int = 1000,
            variant: _typing.Literal['a0', 'a1'] = 'a0',
            width_mult: float = 1.0,
            depth_mult: float = 1.0,
            deploy: bool = False,
        ) -> ModelBuilder:
            """Configure the complete repvgg model via the existing named factory.

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                input_channels: Number of input image/feature channels. Default: 3.
                num_classes: num classes Default: 1000.
                variant: Supported closed architecture variant. Default: 'a0'.
                width_mult: Multiplier for architecture channel widths. Default: 1.0.
                depth_mult: Multiplier for architecture stage depth. Default: 1.0.
                deploy: Use the deployment form of this reparameterized block; conversion support remains explicit. Default: False.

            Returns:
                A fresh ModelBuilder configuration.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                ModelBuilder.repvgg()

            HYSo Guide: models.development-families
            Offline: hyso docs show models.development-families --lang en
            Open: hyso docs open models.development-families --lang en
            """
            return cls.named(
                'repvgg',
                input_channels=input_channels,
                num_classes=num_classes,
                variant=variant,
                width_mult=width_mult,
                depth_mult=depth_mult,
                deploy=deploy,
            )
        @classmethod
        def mobileone(
            cls,
            *,
            input_channels: int = 3,
            num_classes: int = 1000,
            variant: _typing.Literal['s0', 's1', 's2'] = 's0',
            width_mult: float = 1.0,
            depth_mult: float = 1.0,
            deploy: bool = False,
        ) -> ModelBuilder:
            """Configure the complete mobileone model via the existing named factory.

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                input_channels: Number of input image/feature channels. Default: 3.
                num_classes: num classes Default: 1000.
                variant: Supported closed architecture variant. Default: 's0'.
                width_mult: Multiplier for architecture channel widths. Default: 1.0.
                depth_mult: Multiplier for architecture stage depth. Default: 1.0.
                deploy: Use the deployment form of this reparameterized block; conversion support remains explicit. Default: False.

            Returns:
                A fresh ModelBuilder configuration.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                ModelBuilder.mobileone()

            HYSo Guide: models.development-families
            Offline: hyso docs show models.development-families --lang en
            Open: hyso docs open models.development-families --lang en
            """
            return cls.named(
                'mobileone',
                input_channels=input_channels,
                num_classes=num_classes,
                variant=variant,
                width_mult=width_mult,
                depth_mult=depth_mult,
                deploy=deploy,
            )
    # END DEVELOPMENT REGISTRY WRAPPERS

    def add(self, name: str, **params: Any) -> ModelBuilder:
        """Append a registered block transactionally; return self for chaining.

        Args:
            name (str): Registered component name, exact metric label, or stored entity name in this API. Required.
            params (Any): Closed registered keyword parameters; inspect ModelRegistry.get(name) for supported kinds/defaults. Required.

        Returns:
            This builder after the existing component/reference validation.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.add(name)

        HYSo Guide: models.model-builder
        Offline: hyso docs show models.model-builder --lang en
        Open: hyso docs open models.model-builder --lang en
        """
        if self._config['kind'] not in ('sequential_classifier','sequential'):
            raise ModelConfigError('add is unavailable for fixed complete-model profiles', code='INVALID_BLOCK_ORDER')
        candidate = self.to_config()
        candidate['blocks'].append(dict(name=name,params=params))
        self._config, _ = self._validate(candidate)
        return self

    def build(self) -> nn.Module:
        """Allocate a fresh CPU model; architecture round trips do not restore weights.

        Returns:
            A fresh CPU model. No forward, training, CUDA move or checkpoint load is requested.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.build()

        HYSo Guide: models.model-builder
        Offline: hyso docs show models.model-builder --lang en
        Open: hyso docs open models.model-builder --lang en
        """
        config, shape = self._validate(self._config)
        reg = ModelRegistry()
        with torch.device('cpu'):
            if config['kind'] == 'registered_model':
                return reg.create(config['name'],**config['params'])
            if config['kind'] == 'modular_cnn':
                return reg.create('modular_cnn',**config['params'])
            blocks = [reg.create(b['name'],**b['params']) for b in config['blocks']]
            if config['kind']=='sequential':
                _, info = _validate_head(config['head'],shape)
                head = reg.create(config['head']['name'],in_features=info['in_features'],**config['head']['params'])
                return BuiltSequential(blocks,config,head)
            return BuiltClassifier(blocks,config,shape[0])

    def to_json(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Save architecture only, UTF-8 JSON; refuse to overwrite by default.

        Args:
            path (str | Path): Filesystem path for the specific reader/exporter; no configuration code is executed. Required.
            overwrite (bool): Whether an existing output file may be replaced explicitly; inputs remain protected. Default: False.

        Returns:
            None after the native file export succeeds.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_json(path)

        HYSo Guide: models.model-builder
        Offline: hyso docs show models.model-builder --lang en
        Open: hyso docs open models.model-builder --lang en
        """
        if type(overwrite) is not bool: raise ModelConfigError('overwrite must be bool')
        raw = json.dumps(self.to_config(),indent=2,allow_nan=False)+'\n'
        with Path(path).open('w' if overwrite else 'x',encoding='utf-8',newline='\n') as stream:
            stream.write(raw)

    @classmethod
    def from_json(cls, path: str | Path) -> ModelBuilder:
        """Read bounded UTF-8 JSON; reject duplicates, NaN and unsupported names.

        Args:
            path (str | Path): Filesystem path for the specific reader/exporter; no configuration code is executed. Required.

        Returns:
            A validated ModelBuilder instance reconstructed from the supplied data.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            ModelBuilder.from_json(path)

        HYSo Guide: models.model-builder
        Offline: hyso docs show models.model-builder --lang en
        Open: hyso docs open models.model-builder --lang en
        """
        with Path(path).open('rb') as stream: raw = stream.read(_MAX_BYTES+1)
        if len(raw) > _MAX_BYTES: raise ModelConfigError('configuration exceeds 2 MiB')
        def invalid(value): raise ModelConfigError(f'nonfinite JSON number: {value}')
        try:
            data = json.loads(raw.decode('utf-8'),object_pairs_hook=_pairs,parse_constant=invalid)
            return cls.from_config(data)
        except ModelConfigError:
            raise
        except (ValueError, TypeError, RecursionError) as error:
            raise ModelConfigError(str(error)) from error
