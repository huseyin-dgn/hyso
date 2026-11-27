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
        features (list[nn.Module]): Explicit feature IDs to compare/plan; prerequisites are never
            silently executed. Required.
        config (dict[str, Any]): Plain versioned configuration; never executable code or trained state.
            Required.
        output_channels (int): Number of output feature channels. Required.

    Returns:
        An instance of BuiltClassifier with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        BuiltClassifier(features, config, output_channels)

    HYSo Guide: models.model-builder
    Offline: hyso docs show models.model-builder --lang en
    Open: hyso docs open models.model-builder --lang en
    """
    def __init__(self, features: list[nn.Module], config: dict[str, Any], output_channels: int) -> 'None':
        """Initialize BuiltClassifier with the supplied settings.

        Args:
            features (list[nn.Module]): Explicit feature IDs to compare/plan; prerequisites are never
                silently executed. Required.
            config (dict[str, Any]): Plain versioned configuration; never executable code or trained state.
                Required.
            output_channels (int): Number of output feature channels. Required.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
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
            No additional validation for a valid object; corrupt retained state or native copy/serialization
            failures propagate.

        Examples:
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
        features (_typing.Any): Explicit feature IDs to compare/plan; prerequisites are never silently
            executed. Required.
        config (dict[str, _typing.Any]): Plain versioned configuration; never executable code or trained
            state. Required.
        head (_typing.Any): Explicit prediction-head kind, such as classification, regression or
            embedding. Required.

    Returns:
        An instance of BuiltSequential with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        BuiltSequential(features, config, head)

    HYSo Guide: models.model-builder
    Offline: hyso docs show models.model-builder --lang en
    Open: hyso docs open models.model-builder --lang en
    """
    def __init__(self, features: '_typing.Any', config: 'dict[str, _typing.Any]', head: '_typing.Any') -> 'None':
        """Initialize BuiltSequential with the supplied settings.

        Args:
            features (_typing.Any): Explicit feature IDs to compare/plan; prerequisites are never silently
                executed. Required.
            config (dict[str, _typing.Any]): Plain versioned configuration; never executable code or trained
                state. Required.
            head (_typing.Any): Explicit prediction-head kind, such as classification, regression or
                embedding. Required.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
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
            The existing Linear classification projection. Heads without a linear attribute do not support
            this accessor.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
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
            x (_typing.Any): Input Tensor(s) obeying the model/block shape, dtype and device contract.
                Required.

        Returns:
            Native Tensor output(s) of this model/block; this call performs the forward computation.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
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

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        builder = ModelBuilder((1, 4, 4), 2)

    HYSo Guide: models.model-builder
    Offline: hyso docs show models.model-builder --lang en
    Open: hyso docs open models.model-builder --lang en
    """
    def __init__(self, input_shape: tuple[int, int, int], num_classes: int) -> 'None':
        """Initialize ModelBuilder with the supplied settings.

        Args:
            input_shape (tuple[int, int, int]): Per-sample (C, H, W) dimensions; batch is excluded.
                Required.
            num_classes (int): Classifier output width; binary/multiclass loss requirements must match.
                Required.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
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
    @_typing.overload
    def sequential(cls, input_shape: _typing.Sequence[int], *, head: _typing.Literal['classification']='classification', num_classes: int, pooling: _typing.Literal['avg','max','none']='avg', dropout: float=0) -> ModelBuilder:
        """Keyword contract for classification_head; execution uses the validated implementation.

        Args:
            input_shape: Per-sample (C,H,W), excluding batch; each dimension must be an integer in 1..8192.
            head: Terminal prediction head selected by this overload.
            num_classes: Number of output class logits. Outputs are unnormalized scores, not probabilities.
            pooling: avg/max globally pool NCHW before flattening; none flattens NCHW or consumes flat NF.
            dropout: Training-only activation/head dropout probability; disabled in eval.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @classmethod
    @_typing.overload
    def sequential(cls, input_shape: _typing.Sequence[int], *, head: _typing.Literal['regression'], output_features: int, pooling: _typing.Literal['avg','max','none']='avg') -> ModelBuilder:
        """Keyword contract for regression_head; execution uses the validated implementation.

        Args:
            input_shape: Per-sample (C,H,W), excluding batch; each dimension must be an integer in 1..8192.
            head: Terminal prediction head selected by this overload.
            output_features: Width of the unrestricted numeric regression output; no classification objective is inferred.
            pooling: avg/max globally pool NCHW before flattening; none flattens NCHW or consumes flat NF.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @classmethod
    @_typing.overload
    def sequential(cls, input_shape: _typing.Sequence[int], *, head: _typing.Literal['embedding'], embedding_dim: int, pooling: _typing.Literal['avg','max','none']='avg', normalize: bool=False, eps: float=1e-12) -> ModelBuilder:
        """Keyword contract for embedding_head; execution uses the validated implementation.

        Args:
            input_shape: Per-sample (C,H,W), excluding batch; each dimension must be an integer in 1..8192.
            head: Terminal prediction head selected by this overload.
            embedding_dim: Attention projection width or output embedding width, according to the component. None in self-attention uses C; QKV requires 3*embedding_dim<=8192.
            pooling: avg/max globally pool NCHW before flattening; none flattens NCHW or consumes flat NF.
            normalize: Apply safe per-sample L2 normalization to output embeddings.
            eps: Positive numerical stabilizer used in normalization. Embedding normalization also floors it at the dtype tiny value.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @classmethod
    def sequential(cls, input_shape: '_typing.Sequence[int]', *, head: _typing.Literal['classification', 'regression', 'embedding']='classification', **head_params: '_typing.Any') -> 'ModelBuilder':
        """Create an opt-in schema2 chain with an explicit terminal head.

        Args:
            input_shape (_typing.Sequence[int]): Per-sample (C, H, W) dimensions; batch is excluded.
                Required.
            head (str): Explicit prediction-head kind, such as classification, regression or embedding.
                Default: 'classification'.
            head_params (_typing.Any): Closed head keywords: classification requires num_classes, regression
                requires output_features, embedding requires embedding_dim. pooling defaults to avg;
                classification also supports dropout=0 and embedding normalize=False, eps=1e-12. in_features
                is inferred.

        Returns:
            A validated architecture-only ModelBuilder; build() allocates a fresh model.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            ModelBuilder.sequential((1, 8, 8), num_classes=2)

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
        Examples:
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
        Examples:
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
        Examples:
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
            name (str): Registered component name, exact metric label, or stored entity name in this API.
                Required.
            params (Any): Closed registered keyword parameters; inspect ModelRegistry.get(name) for
                supported kinds/defaults. Required.

        Returns:
            A validated architecture-only ModelBuilder; build() allocates a fresh model.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            ModelBuilder.named("modular_cnn", input_channels=1, num_classes=2)

        HYSo Guide: models.model-builder
        Offline: hyso docs show models.model-builder --lang en
        Open: hyso docs open models.model-builder --lang en
        """
        return cls.from_config(dict(format='hyso.model_config',schema_version=1,kind='registered_model',name=name,params=params))

    @classmethod
    def modular(cls, **params: Any) -> ModelBuilder:
        """Describe the original ModularCNN; build() returns exactly that class.

        Args:
            params (Any): Closed registered keyword parameters; inspect ModelRegistry.get(name) for
                supported kinds/defaults. Required.

        Returns:
            A validated architecture-only ModelBuilder; build() allocates a fresh model.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
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
            config (dict[str, Any]): Plain versioned configuration; never executable code or trained state.
                Required.

        Returns:
            A validated builder/configuration object; no executable names are imported.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
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
            No additional validation for a valid object; corrupt retained state or native copy/serialization
            failures propagate.

        Examples:
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
            config (dict[str, Any]): Plain versioned configuration; never executable code or trained state.
                Required.
            batch_size (int): Positive batch-size setting or metadata assumption; no implicit batch search.
                Default: 1.
            activation_limit (int): Advisory per-tensor element threshold used by metadata preflight, not a
                physical-memory quota. Default: 16777216.

        Returns:
            Metadata validation, shapes, diagnostics and advisory resource estimates; no model is built.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
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

    def _infer_channels(self, parameter: str) -> int:
        """Read exact current CHW metadata through the existing validator."""
        if self._config['kind'] not in ('sequential_classifier', 'sequential'):
            raise ModelConfigError('add is unavailable for fixed complete-model profiles', code='INVALID_BLOCK_ORDER')
        _, shape = self._validate(self._config)
        if len(shape) != 3:
            raise ModelConfigError(f'Cannot infer {parameter}: current features are not NCHW; provide an explicit value and a compatible shape',
                                   code='SHAPE_INFERENCE_UNAVAILABLE', expected='known CHW features', actual=list(shape),
                                   suggestion=f'Use an explicit {parameter} after restoring a compatible NCHW shape.')
        return shape[0]

    def conv2d(
        self,
        *,
        in_channels: int | None = None,
        out_channels: int,
        kernel_size: int | tuple[int, int] | list[int] = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        padding: int | tuple[int, int] | list[int] = 1,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, int] | list[int] = 1,
    ) -> ModelBuilder:
        """Append a registered 2D convolution.

        Args:
            in_channels (int | None): Input feature channels; must match the incoming tensor C dimension.
                Integer in 1..8192; bool is rejected. None infers C from the current validated feature
                shape; ambiguous/non-NCHW shapes are rejected. Default: None.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            kernel_size (int | tuple[int, int] | list[int]): Spatial filter/window extent in H and W; an
                integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            padding (int | tuple[int, int] | list[int]): Symmetric padding on each spatial axis; changes
                output geometry without changing channel count. Integer or H/W pair, each in 0..8192.
                Default: 1.
            groups (int): Channel partitions; must divide both convolution channel counts. ChannelShuffle
                requires channels % groups == 0. Integer in 1..8192; bool is rejected. Default: 1.
            bias (bool): Include a learned additive convolution bias for each output channel. Boolean only.
                Default: True.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; groups divides both channel counts.
            Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See
            parameter_dependencies and preflight.
            groups divides in_channels and out_channels
            dilation changes the effective kernel
            Ordinary convolution: out=floor((in+2*padding-dilation*(kernel-1)-1)/stride)+1 per axis;
            transposed convolution: out=(in-1)*stride-2*padding+dilation*(kernel-1)+output_padding+1.
            Multi-stage blocks validate every intermediate shape.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.conv2d(in_channels=1, out_channels=4)

        HYSo Guide: components.convolution
        Offline: hyso docs show components.convolution --lang en
        Open: hyso docs open components.convolution --lang en
        """
        if in_channels is None:
            in_channels = self._infer_channels('in_channels')
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

        Args:
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            kernel_size (int | tuple[int, int] | list[int]): Spatial filter/window extent in H and W; an
                integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            padding (int | tuple[int, int] | list[int]): Symmetric padding on each spatial axis; changes
                output geometry without changing channel count. Integer or H/W pair, each in 0..8192.
                Default: 1.
            groups (int): Channel partitions; must divide both convolution channel counts. ChannelShuffle
                requires channels % groups == 0. Integer in 1..8192; bool is rejected. Default: 1.
            norm (_typing.Literal['batch', 'group', None]): Optional post-convolution normalization: batch,
                group or None. Group normalization chooses the largest channel divisor <=8; convolution bias
                is enabled only without normalization. Choices: 'batch', 'group', None. Default: 'batch'.
            activation (_typing.Literal['silu', 'relu', None]): Nonlinearity applied inside this block; only
                the declared registry choices are accepted. Choices: 'silu', 'relu', None. Default: 'silu'.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; groups divides both channel counts.
            Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See
            parameter_dependencies and preflight.
            groups divides both channel counts
            normalization determines convolution bias
            Ordinary convolution: out=floor((in+2*padding-dilation*(kernel-1)-1)/stride)+1 per axis;
            transposed convolution: out=(in-1)*stride-2*padding+dilation*(kernel-1)+output_padding+1.
            Multi-stage blocks validate every intermediate shape.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW in Builder v1.
            Same NCHW shape; out-of-place activation.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.relu()

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add('relu')

    def silu(self) -> ModelBuilder:
        """Append out-of-place SiLU activation.

        Args:

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW in Builder v1.
            Same NCHW shape; out-of-place activation.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.silu()

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add('silu')

    def gelu(self, *, approximate: _typing.Literal['none', 'tanh'] = 'none') -> ModelBuilder:
        """Append GELU activation with the selected approximation.

        Args:
            approximate (_typing.Literal['none', 'tanh']): GELU calculation: none uses the exact error
                function; tanh uses the approximation. Choices: 'none', 'tanh'. Default: 'none'.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW in Builder v1.
            Same NCHW shape; out-of-place activation.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.gelu()

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add('gelu', approximate=approximate)

    def batch_norm(
        self,
        *,
        num_features: int | None = None,
        eps: float = 1e-05,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> ModelBuilder:
        """Append registered 2D batch normalization.

        Args:
            num_features (int | None): Normalization channel count; must match input C. Integer in 1..8192;
                bool is rejected. Default: None.
            eps (float): Positive numerical stabilizer used in normalization. Embedding normalization also
                floors it at the dtype tiny value. Finite positive number. Default: 1e-05.
            momentum (float | None): Running-statistic update factor; None selects cumulative averaging for
                BatchNorm. None or number in [0,1]. Default: 0.1.
            affine (bool): Learn per-channel scale and bias when True; otherwise use normalization without
                learned affine parameters. Boolean only. Default: True.
            track_running_stats (bool): Track running mean/variance for evaluation. Without tracking,
                input-statistic sample requirements also apply in eval. Boolean only. Default: True.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW; C matches num_features, num_channels or channels. GroupNorm groups divide C.
            Same NCHW shape; channel LayerNorm normalizes C per spatial position. Native statistics
            constraints apply.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.batch_norm(num_features=4)

        HYSo Guide: components.normalization
        Offline: hyso docs show components.normalization --lang en
        Open: hyso docs open components.normalization --lang en
        """
        if num_features is None:
            num_features = self._infer_channels('num_features')
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

        Args:
            num_groups (int): GroupNorm channel partitions; must divide num_channels. Integer in 1..8192;
                bool is rejected. Required.
            num_channels (int): Total GroupNorm channels C; must be divisible by num_groups. Integer in
                1..8192; bool is rejected. Required.
            eps (float): Positive numerical stabilizer used in normalization. Embedding normalization also
                floors it at the dtype tiny value. Finite positive number. Default: 1e-05.
            affine (bool): Learn per-channel scale and bias when True; otherwise use normalization without
                learned affine parameters. Boolean only. Default: True.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW; C matches num_features, num_channels or channels. GroupNorm groups divide C.
            Same NCHW shape; channel LayerNorm normalizes C per spatial position. Native statistics
            constraints apply.
            num_groups divides num_channels
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            kernel_size (int | tuple[int, int] | list[int]): Spatial filter/window extent in H and W; an
                integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. Required.
            stride (int | tuple[int, int] | list[int] | None): Spatial sampling step on each axis; larger
                values reduce ordinary convolution/pooling resolution. None or integer/HW pair, each in
                1..8192. Default: None.
            padding (int | tuple[int, int] | list[int]): Symmetric padding on each spatial axis; changes
                output geometry without changing channel count. Integer or H/W pair, each in 0..8192.
                Default: 0.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            ceil_mode (bool): Round pooling output sizes upward, then discard windows beginning wholly in
                right/bottom padding. Boolean only. Default: False.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW; legal window/padding or positive adaptive output_size.
            Floating NCHW with unchanged channels; global average returns [N,C,1,1].
            stride=None uses kernel_size
            padding[i] <= floor(kernel_size[i]/2)
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            kernel_size (int | tuple[int, int] | list[int]): Spatial filter/window extent in H and W; an
                integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. Required.
            stride (int | tuple[int, int] | list[int] | None): Spatial sampling step on each axis; larger
                values reduce ordinary convolution/pooling resolution. None or integer/HW pair, each in
                1..8192. Default: None.
            padding (int | tuple[int, int] | list[int]): Symmetric padding on each spatial axis; changes
                output geometry without changing channel count. Integer or H/W pair, each in 0..8192.
                Default: 0.
            ceil_mode (bool): Round pooling output sizes upward, then discard windows beginning wholly in
                right/bottom padding. Boolean only. Default: False.
            count_include_pad (bool): Include zero-padded elements in the average-pooling divisor. Boolean
                only. Default: True.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW; legal window/padding or positive adaptive output_size.
            Floating NCHW with unchanged channels; global average returns [N,C,1,1].
            stride=None uses kernel_size
            padding[i] <= floor(kernel_size[i]/2)
            count_include_pad controls the mean divisor
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            output_size (int | tuple[int, int] | list[int]): Target H/W grid; integer means a square. Resize
                requires exactly one of output_size and scale_factor. Integer or H/W pair, each in 1..8192.
                Required.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW; legal window/padding or positive adaptive output_size.
            Floating NCHW with unchanged channels; global average returns [N,C,1,1].
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.adaptive_avg_pool2d(output_size=2)

        HYSo Guide: components.pooling
        Offline: hyso docs show components.pooling --lang en
        Open: hyso docs open components.pooling --lang en
        """
        return self.add('adaptive_avg_pool2d', output_size=output_size)

    def dropout(self, *, p: float = 0.5) -> ModelBuilder:
        """Append elementwise dropout; evaluation leaves values unchanged.

        Args:
            p (float): Probability of dropping elements, channels or paths during training; evaluation
                disables masking. Number in [0,1]. Default: 0.5.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Nonempty floating NCHW.
            Same shape and dtype; explicit training-only masking.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            norm (_typing.Literal['batch', 'group', None]): Optional post-convolution normalization: batch,
                group or None. Group normalization chooses the largest channel divisor <=8; convolution bias
                is enabled only without normalization. Choices: 'batch', 'group', None. Default: 'batch'.
            attention (dict[str, _typing.Any] | None): Optional attention applied after the residual branch;
                nested legacy attention must use matching out_channels. Default: None.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; attention channels equal out_channels.
            [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
            Odd spatial kernels use dilation-aware symmetric padding. Channel/stride changes use an explicit
            projection shortcut; intermediate channels remain bounded to 8192.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            reduction (int): Channel-gate reduction divisor; hidden width is max(1,C//reduction), or
                max(8,C//reduction) for coordinate attention. Integer in 1..8192; bool is rejected. Default:
                8.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,channels,H,W], all dimensions positive.
            Same NCHW shape; multiplicative gates, no channel or spatial resizing.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.se(channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add('se', channels=channels, reduction=reduction)

    def cbam(self, *, channels: int, reduction: int = 16, kernel_size: int = 7) -> ModelBuilder:
        """Append the registered cbam block via add().

        Args:
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            reduction (int): Channel-gate reduction divisor; hidden width is max(1,C//reduction), or
                max(8,C//reduction) for coordinate attention. Integer in 1..8192; bool is rejected. Default:
                16.
            kernel_size (int): Spatial filter/window extent in H and W; an integer uses the same extent on
                both axes. Odd integer in 1..8192; bool is rejected. Default: 7.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,channels,H,W], all dimensions positive.
            Same NCHW shape; multiplicative gates, no channel or spatial resizing.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.cbam(channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add('cbam', channels=channels, reduction=reduction, kernel_size=kernel_size)

    def eca(self, *, channels: int, kernel_size: int = 3) -> ModelBuilder:
        """Append the registered eca block via add().

        Args:
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            kernel_size (int): Spatial filter/window extent in H and W; an integer uses the same extent on
                both axes. Odd integer in 1..8192; bool is rejected. Default: 3.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,channels,H,W], all dimensions positive.
            Same NCHW shape; multiplicative gates, no channel or spatial resizing.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.eca(channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add('eca', channels=channels, kernel_size=kernel_size)

    def coordinate(self, *, channels: int, reduction: int = 32) -> ModelBuilder:
        """Append the registered coordinate block via add().

        Args:
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            reduction (int): Channel-gate reduction divisor; hidden width is max(1,C//reduction), or
                max(8,C//reduction) for coordinate attention. Integer in 1..8192; bool is rejected. Default:
                32.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,channels,H,W], all dimensions positive.
            Same NCHW shape; multiplicative gates, no channel or spatial resizing.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.coordinate(channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add('coordinate', channels=channels, reduction=reduction)

    def hardswish(self) -> ModelBuilder:
        """Append the registered hardswish block via add().

        Args:

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW in Builder v1.
            Same NCHW shape; out-of-place activation.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.hardswish()

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add('hardswish')

    def identity(self) -> ModelBuilder:
        """Append the registered identity block via add().

        Args:

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW in Builder v1.
            Same tensor/shape; no parameters.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            input_channels (int): Input image channels C; must match the first model stage. Integer in
                1..8192; bool is rejected. Default: 3.
            num_classes (int): Number of output class logits. Outputs are unnormalized scores, not
                probabilities. Integer in 1..8192; bool is rejected. Default: 10.
            widths (tuple[int, ...] | list[int]): Per-stage output channels; length must match repeats and
                strides. Sequence of 1..16 integers, each in 1..8192. Default: (32, 64, 128).
            repeats (tuple[int, ...] | list[int]): Number of blocks per stage; same length as
                widths/strides, with at most 64 blocks in total. Sequence of 1..16 integers, each in
                1..8192. Default: (1, 1, 1).
            strides (tuple[int, ...] | list[int]): First-block spatial stride for each stage; remaining
                blocks use stride 1. Sequence of 1..16 integers, each in 1..8192. Default: (1, 2, 2).
            attention (_typing.Literal['none', 'se', 'cbam', 'eca', 'coordinate']): Attention selector
                inside the block; applies channel/spatial gates without resizing. Choices: 'none', 'se',
                'cbam', 'eca', 'coordinate'. Default: 'none'.
            norm (_typing.Literal['batch', 'group', None]): Optional post-convolution normalization: batch,
                group or None. Group normalization chooses the largest channel divisor <=8; convolution bias
                is enabled only without normalization. Choices: 'batch', 'group', None. Default: 'batch'.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty [N,input_channels,H,W]; normal model.to(device/dtype) semantics.
            [N,num_classes] unnormalized classification logits.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            groups (int): Channel partitions; must divide both convolution channel counts. ChannelShuffle
                requires channels % groups == 0. Integer in 1..8192; bool is rejected. Default: 2.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW with matching channels.
            Same NCHW shape; validated channel partition.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.channel_shuffle(channels=channels)

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add('channel_shuffle', channels=channels, groups=groups)

    def dropblock2d(self, *, p: float = 0.1, block_size: int = 3) -> ModelBuilder:
        """Append the registered dropblock2d block via add().

        Args:
            p (float): Probability of dropping elements, channels or paths during training; evaluation
                disables masking. Number in [0,1]. Default: 0.1.
            block_size (int): Requested spatial square side for contiguous DropBlock masks; runtime clips it
                to min(block_size,H,W). Integer in 1..8192; bool is rejected. Default: 3.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Nonempty floating NCHW.
            Same shape and dtype; explicit training-only masking.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            bias (bool): Include a learned additive convolution bias for each output channel. Boolean only.
                Default: True.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; groups divides both channel counts.
            Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See
            parameter_dependencies and preflight.
            Ordinary convolution: out=floor((in+2*padding-dilation*(kernel-1)-1)/stride)+1 per axis;
            transposed convolution: out=(in-1)*stride-2*padding+dilation*(kernel-1)+output_padding+1.
            Multi-stage blocks validate every intermediate shape.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            depth_multiplier (int): Independent filters per input channel; depthwise output/intermediate C
                equals in_channels*depth_multiplier and must not exceed 8192. Integer in 1..8192; bool is
                rejected. Default: 1.
            kernel_size (int | tuple[int, int] | list[int]): Spatial filter/window extent in H and W; an
                integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            padding (int | tuple[int, int] | list[int]): Symmetric padding on each spatial axis; changes
                output geometry without changing channel count. Integer or H/W pair, each in 0..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            bias (bool): Include a learned additive convolution bias for each output channel. Boolean only.
                Default: True.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; groups divides both channel counts.
            Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See
            parameter_dependencies and preflight.
            groups=in_channels
            out_channels=in_channels*depth_multiplier
            Ordinary convolution: out=floor((in+2*padding-dilation*(kernel-1)-1)/stride)+1 per axis;
            transposed convolution: out=(in-1)*stride-2*padding+dilation*(kernel-1)+output_padding+1.
            Multi-stage blocks validate every intermediate shape.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            depth_multiplier (int): Independent filters per input channel; depthwise output/intermediate C
                equals in_channels*depth_multiplier and must not exceed 8192. Integer in 1..8192; bool is
                rejected. Default: 1.
            kernel_size (int | tuple[int, int] | list[int]): Spatial filter/window extent in H and W; an
                integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            padding (int | tuple[int, int] | list[int]): Symmetric padding on each spatial axis; changes
                output geometry without changing channel count. Integer or H/W pair, each in 0..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            bias (bool): Include a learned additive convolution bias for each output channel. Boolean only.
                Default: True.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; groups divides both channel counts.
            Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See
            parameter_dependencies and preflight.
            intermediate_channels=in_channels*depth_multiplier
            Ordinary convolution: out=floor((in+2*padding-dilation*(kernel-1)-1)/stride)+1 per axis;
            transposed convolution: out=(in-1)*stride-2*padding+dilation*(kernel-1)+output_padding+1.
            Multi-stage blocks validate every intermediate shape.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            intermediate_channels (int): Internal feature width. Factorized convolution requires this <=
                min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in
                1..8192; bool is rejected. Required.
            kernel_size (int | tuple[int, int] | list[int]): Spatial filter/window extent in H and W; an
                integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            padding (int | tuple[int, int] | list[int]): Symmetric padding on each spatial axis; changes
                output geometry without changing channel count. Integer or H/W pair, each in 0..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            bias (bool): Include a learned additive convolution bias for each output channel. Boolean only.
                Default: True.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; groups divides both channel counts.
            Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See
            parameter_dependencies and preflight.
            H settings apply to the vertical stage, W settings to the horizontal stage
            Ordinary convolution: out=floor((in+2*padding-dilation*(kernel-1)-1)/stride)+1 per axis;
            transposed convolution: out=(in-1)*stride-2*padding+dilation*(kernel-1)+output_padding+1.
            Multi-stage blocks validate every intermediate shape.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            intermediate_channels (int): Internal feature width. Factorized convolution requires this <=
                min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in
                1..8192; bool is rejected. Required.
            kernel_size (int | tuple[int, int] | list[int]): Spatial filter/window extent in H and W; an
                integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            padding (int | tuple[int, int] | list[int]): Symmetric padding on each spatial axis; changes
                output geometry without changing channel count. Integer or H/W pair, each in 0..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            bias (bool): Include a learned additive convolution bias for each output channel. Boolean only.
                Default: True.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; groups divides both channel counts.
            Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See
            parameter_dependencies and preflight.
            intermediate_channels <= min(in_channels,out_channels)
            1x1 reduction then KxK convolution
            Ordinary convolution: out=floor((in+2*padding-dilation*(kernel-1)-1)/stride)+1 per axis;
            transposed convolution: out=(in-1)*stride-2*padding+dilation*(kernel-1)+output_padding+1.
            Multi-stage blocks validate every intermediate shape.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            kernel_size (int | tuple[int, int] | list[int]): Spatial filter/window extent in H and W; an
                integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            padding (int | tuple[int, int] | list[int]): Symmetric padding on each spatial axis; changes
                output geometry without changing channel count. Integer or H/W pair, each in 0..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            bias (bool): Include a learned additive convolution bias for each output channel. Boolean only.
                Default: True.
            groups (int): Channel partitions; must divide both convolution channel counts. ChannelShuffle
                requires channels % groups == 0. Integer in 1..8192; bool is rejected. Default: 1.
            output_padding (int | tuple[int, int] | list[int]): Additional output extent for transposed
                convolution, not input padding; each axis must be less than stride or dilation on that axis.
                Integer or H/W pair, each in 0..8192. Default: 0.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; groups divides both channel counts.
            Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See
            parameter_dependencies and preflight.
            groups divides both channel counts
            output_padding[i] < stride[i] or dilation[i]
            Ordinary convolution: out=floor((in+2*padding-dilation*(kernel-1)-1)/stride)+1 per axis;
            transposed convolution: out=(in-1)*stride-2*padding+dilation*(kernel-1)+output_padding+1.
            Multi-stage blocks validate every intermediate shape.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            eps (float): Positive numerical stabilizer used in normalization. Embedding normalization also
                floors it at the dtype tiny value. Finite positive number. Default: 1e-05.
            affine (bool): Learn per-channel scale and bias when True; otherwise use normalization without
                learned affine parameters. Boolean only. Default: True.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW; C matches num_features, num_channels or channels. GroupNorm groups divide C.
            Same NCHW shape; channel LayerNorm normalizes C per spatial position. Native statistics
            constraints apply.
            normalizes C at each N,H,W position, external layout stays NCHW
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            num_features (int): Normalization channel count; must match input C. Integer in 1..8192; bool is
                rejected. Required.
            eps (float): Positive numerical stabilizer used in normalization. Embedding normalization also
                floors it at the dtype tiny value. Finite positive number. Default: 1e-05.
            momentum (float): Running-statistic update factor; None selects cumulative averaging for
                BatchNorm. Number in [0,1]. Default: 0.1.
            affine (bool): Learn per-channel scale and bias when True; otherwise use normalization without
                learned affine parameters. Boolean only. Default: False.
            track_running_stats (bool): Track running mean/variance for evaluation. Without tracking,
                input-statistic sample requirements also apply in eval. Boolean only. Default: False.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW; C matches num_features, num_channels or channels. GroupNorm groups divide C.
            Same NCHW shape; channel LayerNorm normalizes C per spatial position. Native statistics
            constraints apply.
            spatial sample count H*W > 1 when using input statistics
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            negative_slope (float): Nonnegative multiplier for negative LeakyReLU inputs. Finite nonnegative
                number. Default: 0.01.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW in Builder v1.
            Same NCHW shape; out-of-place activation.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.leaky_relu()

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add('leaky_relu', negative_slope=negative_slope)

    def prelu(self, *, num_parameters: int = 1, init: float = 0.25) -> ModelBuilder:
        """Append the registered prelu block via add().

        Args:
            num_parameters (int): PReLU slope count; must be 1 (shared) or equal to incoming C. Integer in
                1..8192; bool is rejected. Default: 1.
            init (float): Initial learned PReLU negative slope; finite and representable as float32. Finite
                number. Default: 0.25.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW in Builder v1.
            Same NCHW shape; out-of-place activation.
            num_parameters is 1 or the incoming channel count
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.prelu()

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add('prelu', num_parameters=num_parameters, init=init)

    def mish(self) -> ModelBuilder:
        """Append the registered mish block via add().

        Args:

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW in Builder v1.
            Same NCHW shape; out-of-place activation.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.mish()

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add('mish')

    def hardsigmoid(self) -> ModelBuilder:
        """Append the registered hardsigmoid block via add().

        Args:

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW in Builder v1.
            Same NCHW shape; out-of-place activation.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.hardsigmoid()

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add('hardsigmoid')

    def adaptive_max_pool2d(self, *, output_size: int | tuple[int, int] | list[int]) -> ModelBuilder:
        """Append the registered adaptive_max_pool2d block via add().

        Args:
            output_size (int | tuple[int, int] | list[int]): Target H/W grid; integer means a square. Resize
                requires exactly one of output_size and scale_factor. Integer or H/W pair, each in 1..8192.
                Required.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW; legal window/padding or positive adaptive output_size.
            Floating NCHW with unchanged channels; global average returns [N,C,1,1].
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.adaptive_max_pool2d(output_size=output_size)

        HYSo Guide: components.pooling
        Offline: hyso docs show components.pooling --lang en
        Open: hyso docs open components.pooling --lang en
        """
        return self.add('adaptive_max_pool2d', output_size=output_size)

    def global_avg_pool2d(self) -> ModelBuilder:
        """Append the registered global_avg_pool2d block via add().

        Args:

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW; legal window/padding or positive adaptive output_size.
            Floating NCHW with unchanged channels; global average returns [N,C,1,1].
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            intermediate_channels (int): Internal feature width. Factorized convolution requires this <=
                min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in
                1..8192; bool is rejected. Required.
            kernel_size (int): Spatial filter/window extent in H and W; an integer uses the same extent on
                both axes. Odd integer in 1..8192; bool is rejected. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            normalization (_typing.Literal['batch', 'group', None]): Normalization of residual main-branch
                features; batch, group or None. Choices: 'batch', 'group', None. Default: 'batch'.
            activation (_typing.Literal['relu', 'silu', 'gelu']): Nonlinearity applied inside this block;
                only the declared registry choices are accepted. Choices: 'relu', 'silu', 'gelu'. Default:
                'silu'.
            attention (_typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention']): Attention
                selector inside the block; applies channel/spatial gates without resizing. Choices: 'none',
                'eca', 'spatial_attention', 'channel_attention'. Default: 'none'.
            dropout (float): Training-only activation/head dropout probability; disabled in eval. Finite
                number in [0,1). Default: 0.0.
            drop_path (float): Training-only per-sample residual path drop probability; disabled in eval.
                Finite number in [0,1). Default: 0.0.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; attention channels equal out_channels.
            [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
            Odd spatial kernels use dilation-aware symmetric padding. Channel/stride changes use an explicit
            projection shortcut; intermediate channels remain bounded to 8192.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            intermediate_channels (int): Internal feature width. Factorized convolution requires this <=
                min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in
                1..8192; bool is rejected. Required.
            kernel_size (int): Spatial filter/window extent in H and W; an integer uses the same extent on
                both axes. Odd integer in 1..8192; bool is rejected. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            normalization (_typing.Literal['batch', 'group', None]): Normalization of residual main-branch
                features; batch, group or None. Choices: 'batch', 'group', None. Default: 'batch'.
            activation (_typing.Literal['relu', 'silu', 'gelu']): Nonlinearity applied inside this block;
                only the declared registry choices are accepted. Choices: 'relu', 'silu', 'gelu'. Default:
                'silu'.
            attention (_typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention']): Attention
                selector inside the block; applies channel/spatial gates without resizing. Choices: 'none',
                'eca', 'spatial_attention', 'channel_attention'. Default: 'none'.
            dropout (float): Training-only activation/head dropout probability; disabled in eval. Finite
                number in [0,1). Default: 0.0.
            drop_path (float): Training-only per-sample residual path drop probability; disabled in eval.
                Finite number in [0,1). Default: 0.0.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; attention channels equal out_channels.
            [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
            Odd spatial kernels use dilation-aware symmetric padding. Channel/stride changes use an explicit
            projection shortcut; intermediate channels remain bounded to 8192.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            intermediate_channels (int): Internal feature width. Factorized convolution requires this <=
                min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in
                1..8192; bool is rejected. Required.
            kernel_size (int): Spatial filter/window extent in H and W; an integer uses the same extent on
                both axes. Odd integer in 1..8192; bool is rejected. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            normalization (_typing.Literal['batch', 'group', None]): Normalization of residual main-branch
                features; batch, group or None. Choices: 'batch', 'group', None. Default: 'batch'.
            activation (_typing.Literal['relu', 'silu', 'gelu']): Nonlinearity applied inside this block;
                only the declared registry choices are accepted. Choices: 'relu', 'silu', 'gelu'. Default:
                'silu'.
            attention (_typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention']): Attention
                selector inside the block; applies channel/spatial gates without resizing. Choices: 'none',
                'eca', 'spatial_attention', 'channel_attention'. Default: 'none'.
            dropout (float): Training-only activation/head dropout probability; disabled in eval. Finite
                number in [0,1). Default: 0.0.
            drop_path (float): Training-only per-sample residual path drop probability; disabled in eval.
                Finite number in [0,1). Default: 0.0.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; attention channels equal out_channels.
            [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
            Odd spatial kernels use dilation-aware symmetric padding. Channel/stride changes use an explicit
            projection shortcut; intermediate channels remain bounded to 8192.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            intermediate_channels (int): Internal feature width. Factorized convolution requires this <=
                min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in
                1..8192; bool is rejected. Required.
            kernel_size (int): Spatial filter/window extent in H and W; an integer uses the same extent on
                both axes. Odd integer in 1..8192; bool is rejected. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            normalization (_typing.Literal['batch', 'group', None]): Normalization of residual main-branch
                features; batch, group or None. Choices: 'batch', 'group', None. Default: 'batch'.
            activation (_typing.Literal['relu', 'silu', 'gelu']): Nonlinearity applied inside this block;
                only the declared registry choices are accepted. Choices: 'relu', 'silu', 'gelu'. Default:
                'silu'.
            attention (_typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention']): Attention
                selector inside the block; applies channel/spatial gates without resizing. Choices: 'none',
                'eca', 'spatial_attention', 'channel_attention'. Default: 'none'.
            dropout (float): Training-only activation/head dropout probability; disabled in eval. Finite
                number in [0,1). Default: 0.0.
            drop_path (float): Training-only per-sample residual path drop probability; disabled in eval.
                Finite number in [0,1). Default: 0.0.
            groups (int): Channel partitions; must divide both convolution channel counts. ChannelShuffle
                requires channels % groups == 0. Integer in 1..8192; bool is rejected. Default: 1.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; attention channels equal out_channels.
            [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
            Odd spatial kernels use dilation-aware symmetric padding. Channel/stride changes use an explicit
            projection shortcut; intermediate channels remain bounded to 8192.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            expansion_ratio (int): Inverted-bottleneck channel multiplier; intermediate
                C=in_channels*expansion_ratio must not exceed 8192. Integer in 1..8192; bool is rejected.
                Default: 2.
            kernel_size (int): Spatial filter/window extent in H and W; an integer uses the same extent on
                both axes. Odd integer in 1..8192; bool is rejected. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            normalization (_typing.Literal['batch', 'group', None]): Normalization of residual main-branch
                features; batch, group or None. Choices: 'batch', 'group', None. Default: 'batch'.
            activation (_typing.Literal['relu', 'silu', 'gelu']): Nonlinearity applied inside this block;
                only the declared registry choices are accepted. Choices: 'relu', 'silu', 'gelu'. Default:
                'silu'.
            attention (_typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention']): Attention
                selector inside the block; applies channel/spatial gates without resizing. Choices: 'none',
                'eca', 'spatial_attention', 'channel_attention'. Default: 'none'.
            dropout (float): Training-only activation/head dropout probability; disabled in eval. Finite
                number in [0,1). Default: 0.0.
            drop_path (float): Training-only per-sample residual path drop probability; disabled in eval.
                Finite number in [0,1). Default: 0.0.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; attention channels equal out_channels.
            [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
            Odd spatial kernels use dilation-aware symmetric padding. Channel/stride changes use an explicit
            projection shortcut; intermediate channels remain bounded to 8192.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            intermediate_channels (int): Internal feature width. Factorized convolution requires this <=
                min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in
                1..8192; bool is rejected. Required.
            kernel_size (int): Spatial filter/window extent in H and W; an integer uses the same extent on
                both axes. Odd integer in 1..8192; bool is rejected. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            normalization (_typing.Literal['batch', 'group', None]): Normalization of residual main-branch
                features; batch, group or None. Choices: 'batch', 'group', None. Default: 'batch'.
            activation (_typing.Literal['relu', 'silu', 'gelu']): Nonlinearity applied inside this block;
                only the declared registry choices are accepted. Choices: 'relu', 'silu', 'gelu'. Default:
                'silu'.
            attention (_typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention']): Attention
                selector inside the block; applies channel/spatial gates without resizing. Choices: 'none',
                'eca', 'spatial_attention', 'channel_attention'. Default: 'none'.
            dropout (float): Training-only activation/head dropout probability; disabled in eval. Finite
                number in [0,1). Default: 0.0.
            drop_path (float): Training-only per-sample residual path drop probability; disabled in eval.
                Finite number in [0,1). Default: 0.0.
            gate_init (float): Initial scalar residual contribution in [0,1]; zero suppresses the residual
                main branch initially. Number in [0,1]. Default: 1.0.
            learnable_gate (bool): Store the residual gate as a learned scalar when True, otherwise a fixed
                scalar. Boolean only. Default: True.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; attention channels equal out_channels.
            [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
            Odd spatial kernels use dilation-aware symmetric padding. Channel/stride changes use an explicit
            projection shortcut; intermediate channels remain bounded to 8192.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            kernel_size (int): Spatial filter/window extent in H and W; an integer uses the same extent on
                both axes. Odd integer in 1..8192; bool is rejected. Default: 7.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,channels,H,W], all dimensions positive.
            Same NCHW shape; multiplicative gates, no channel or spatial resizing.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.spatial_attention(channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add('spatial_attention', channels=channels, kernel_size=kernel_size)

    def channel_attention(self, *, channels: int, reduction: int = 8) -> ModelBuilder:
        """Append the registered channel_attention block via add().

        Args:
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            reduction (int): Channel-gate reduction divisor; hidden width is max(1,C//reduction), or
                max(8,C//reduction) for coordinate attention. Integer in 1..8192; bool is rejected. Default:
                8.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,channels,H,W], all dimensions positive.
            Same NCHW shape; multiplicative gates, no channel or spatial resizing.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            embedding_dim (int | None): Attention projection width or output embedding width, according to
                the component. None in self-attention uses C; QKV requires 3*embedding_dim<=8192. None or
                integer in 1..8192. Default: None.
            dropout (float): Training-only activation/head dropout probability; disabled in eval. Finite
                number in [0,1). Default: 0.0.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,channels,H,W], all dimensions positive.
            Same NCHW shape; multiplicative gates, no channel or spatial resizing.
            QKV channels <=8192, H*W<=1024 and N*heads*(H*W)^2<=4194304 are mandatory resource bounds.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            embedding_dim (int | None): Attention projection width or output embedding width, according to
                the component. None in self-attention uses C; QKV requires 3*embedding_dim<=8192. None or
                integer in 1..8192. Default: None.
            num_heads (int): Parallel attention heads; must divide embedding_dim. Spatial attention is
                limited to 1024 tokens and N*heads*tokens^2<=4194304. Integer in 1..8192; bool is rejected.
                Default: 4.
            dropout (float): Training-only activation/head dropout probability; disabled in eval. Finite
                number in [0,1). Default: 0.0.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,channels,H,W], all dimensions positive.
            Same NCHW shape; multiplicative gates, no channel or spatial resizing.
            QKV channels <=8192, H*W<=1024 and N*heads*(H*W)^2<=4194304 are mandatory resource bounds.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            p (float): Probability of dropping elements, channels or paths during training; evaluation
                disables masking. Number in [0,1]. Default: 0.5.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Nonempty floating NCHW.
            Same shape and dtype; explicit training-only masking.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.dropout2d()

        HYSo Guide: components.regularization
        Offline: hyso docs show components.regularization --lang en
        Open: hyso docs open components.regularization --lang en
        """
        return self.add('dropout2d', p=p)

    def drop_path(self, *, p: float = 0.0) -> ModelBuilder:
        """Append the registered drop_path block via add().

        Args:
            p (float): Probability of dropping elements, channels or paths during training; evaluation
                disables masking. Finite number in [0,1). Default: 0.0.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Nonempty floating NCHW.
            Same shape and dtype; explicit training-only masking.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.drop_path()

        HYSo Guide: components.regularization
        Offline: hyso docs show components.regularization --lang en
        Open: hyso docs open components.regularization --lang en
        """
        return self.add('drop_path', p=p)

    def gaussian_noise(self, *, std: float = 0.1) -> ModelBuilder:
        """Append the registered gaussian_noise block via add().

        Args:
            std (float): Standard deviation of training-only zero-mean Gaussian noise; finite, nonnegative
                and at most 1e6. Finite nonnegative number. Default: 0.1.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Nonempty floating NCHW.
            Same shape and dtype; explicit training-only masking.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            output_size (int | tuple[int, int] | list[int] | None): Target H/W grid; integer means a square.
                Resize requires exactly one of output_size and scale_factor. None or integer/HW pair, each
                in 1..8192. Default: None.
            scale_factor (float | tuple[float, float] | list[float] | None): Positive H/W resize
                multipliers, each <=8192. Exactly one of this and output_size must be provided. None or
                positive scalar/HW factors, each <=8192. Default: None.
            mode (_typing.Literal['nearest', 'bilinear', 'bicubic', 'area']): Resize interpolation method;
                nearest/area require align_corners=None. Choices: 'nearest', 'bilinear', 'bicubic', 'area'.
                Default: 'nearest'.
            align_corners (bool | None): Align endpoint pixel centers for bilinear/bicubic resizing; must be
                None for nearest/area. Boolean or None. Default: None.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW.
            Floating NCHW with explicit resize or pixel rearrangement geometry.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

        Args:
            upscale_factor (int): PixelShuffle factor r: C must be divisible by r^2; output is
                (C/r^2,H*r,W*r). Integer in 1..8192; bool is rejected. Required.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW.
            Floating NCHW with explicit resize or pixel rearrangement geometry.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.pixel_shuffle(upscale_factor=upscale_factor)

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add('pixel_shuffle', upscale_factor=upscale_factor)

    def pixel_unshuffle(self, *, downscale_factor: int) -> ModelBuilder:
        """Append the registered pixel_unshuffle block via add().

        Args:
            downscale_factor (int): PixelUnshuffle factor r: H and W must be divisible by r; output is
                (C*r^2,H/r,W/r). Integer in 1..8192; bool is rejected. Required.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW.
            Floating NCHW with explicit resize or pixel rearrangement geometry.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.pixel_unshuffle(downscale_factor=downscale_factor)

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add('pixel_unshuffle', downscale_factor=downscale_factor)

    def flatten(self) -> ModelBuilder:
        """Append the registered flatten block via add().

        Args:

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW with preserved batch axis.
            NCHW or terminal NF for flatten, see shape_contract.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.flatten()

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add('flatten')

    def permute(self, *, dims: tuple[int, ...] | list[int] = (0, 1, 3, 2)) -> ModelBuilder:
        """Append the registered permute block via add().

        Args:
            dims (tuple[int, ...] | list[int]): Permutation of NCHW axes; only (0,1,2,3) and (0,1,3,2) are
                accepted, preserving batch/channel axes. Only identity or spatial-axis swap is supported.
                Default: (0, 1, 3, 2).

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW with preserved batch axis.
            NCHW or terminal NF for flatten, see shape_contract.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.permute()

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add('permute', dims=dims)

    def reshape(self, *, shape: tuple[int, ...] | list[int]) -> ModelBuilder:
        """Append the registered reshape block via add().

        Args:
            shape (tuple[int, ...] | list[int]): Target per-sample CHW dimensions; reshape must preserve the
                number of sample elements and every dimension is 1..8192. Three integers in 1..8192.
                Required.

        Returns:
            This ModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW with preserved batch axis.
            NCHW or terminal NF for flatten, see shape_contract.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
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

    # Static keyword contracts mirror the closed runtime registry.
    @_typing.overload
    def add(self, name: _typing.Literal['conv_norm_act'], *, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int] | list[int] = 3, stride: int | tuple[int, int] | list[int] = 1, padding: int | tuple[int, int] | list[int] = 1, groups: int = 1, norm: _typing.Literal['batch', 'group', None] = 'batch', activation: _typing.Literal['silu', 'relu', None] = 'silu') -> ModelBuilder:
        """Keyword contract for conv_norm_act; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'conv_norm_act'.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            padding: Symmetric padding on each spatial axis; changes output geometry without changing channel count.
            groups: Channel partitions; must divide both convolution channel counts. ChannelShuffle requires channels % groups == 0.
            norm: Optional post-convolution normalization: batch, group or None. Group normalization chooses the largest channel divisor <=8; convolution bias is enabled only without normalization.
            activation: Nonlinearity applied inside this block; only the declared registry choices are accepted.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['conv2d'], *, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int] | list[int] = 3, stride: int | tuple[int, int] | list[int] = 1, padding: int | tuple[int, int] | list[int] = 1, groups: int = 1, bias: bool = True, dilation: int | tuple[int, int] | list[int] = 1) -> ModelBuilder:
        """Keyword contract for conv2d; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'conv2d'.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            padding: Symmetric padding on each spatial axis; changes output geometry without changing channel count.
            groups: Channel partitions; must divide both convolution channel counts. ChannelShuffle requires channels % groups == 0.
            bias: Include a learned additive convolution bias for each output channel.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['residual'], *, in_channels: int, out_channels: int, stride: int | tuple[int, int] | list[int] = 1, norm: _typing.Literal['batch', 'group', None] = 'batch', attention: dict[str, _typing.Any] | None = None) -> ModelBuilder:
        """Keyword contract for residual; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'residual'.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            norm: Optional post-convolution normalization: batch, group or None. Group normalization chooses the largest channel divisor <=8; convolution bias is enabled only without normalization.
            attention: Optional attention applied after the residual branch; nested legacy attention must use matching out_channels.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['se'], *, channels: int, reduction: int = 8) -> ModelBuilder:
        """Keyword contract for se; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'se'.
            channels: Feature channels C; gates and normalization preserve this channel count.
            reduction: Channel-gate reduction divisor; hidden width is max(1,C//reduction), or max(8,C//reduction) for coordinate attention.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['cbam'], *, channels: int, reduction: int = 16, kernel_size: int = 7) -> ModelBuilder:
        """Keyword contract for cbam; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'cbam'.
            channels: Feature channels C; gates and normalization preserve this channel count.
            reduction: Channel-gate reduction divisor; hidden width is max(1,C//reduction), or max(8,C//reduction) for coordinate attention.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['eca'], *, channels: int, kernel_size: int = 3) -> ModelBuilder:
        """Keyword contract for eca; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'eca'.
            channels: Feature channels C; gates and normalization preserve this channel count.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['coordinate'], *, channels: int, reduction: int = 32) -> ModelBuilder:
        """Keyword contract for coordinate; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'coordinate'.
            channels: Feature channels C; gates and normalization preserve this channel count.
            reduction: Channel-gate reduction divisor; hidden width is max(1,C//reduction), or max(8,C//reduction) for coordinate attention.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['batch_norm'], *, num_features: int, eps: float = 1e-05, momentum: float | None = 0.1, affine: bool = True, track_running_stats: bool = True) -> ModelBuilder:
        """Keyword contract for batch_norm; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'batch_norm'.
            num_features: Normalization channel count; must match input C.
            eps: Positive numerical stabilizer used in normalization. Embedding normalization also floors it at the dtype tiny value.
            momentum: Running-statistic update factor; None selects cumulative averaging for BatchNorm.
            affine: Learn per-channel scale and bias when True; otherwise use normalization without learned affine parameters.
            track_running_stats: Track running mean/variance for evaluation. Without tracking, input-statistic sample requirements also apply in eval.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['group_norm'], *, num_groups: int, num_channels: int, eps: float = 1e-05, affine: bool = True) -> ModelBuilder:
        """Keyword contract for group_norm; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'group_norm'.
            num_groups: GroupNorm channel partitions; must divide num_channels.
            num_channels: Total GroupNorm channels C; must be divisible by num_groups.
            eps: Positive numerical stabilizer used in normalization. Embedding normalization also floors it at the dtype tiny value.
            affine: Learn per-channel scale and bias when True; otherwise use normalization without learned affine parameters.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['relu']) -> ModelBuilder:
        """Keyword contract for relu; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'relu'.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['silu']) -> ModelBuilder:
        """Keyword contract for silu; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'silu'.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['hardswish']) -> ModelBuilder:
        """Keyword contract for hardswish; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'hardswish'.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['identity']) -> ModelBuilder:
        """Keyword contract for identity; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'identity'.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['channel_shuffle'], *, channels: int, groups: int = 2) -> ModelBuilder:
        """Keyword contract for channel_shuffle; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'channel_shuffle'.
            channels: Feature channels C; gates and normalization preserve this channel count.
            groups: Channel partitions; must divide both convolution channel counts. ChannelShuffle requires channels % groups == 0.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['dropblock2d'], *, p: float = 0.1, block_size: int = 3) -> ModelBuilder:
        """Keyword contract for dropblock2d; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'dropblock2d'.
            p: Probability of dropping elements, channels or paths during training; evaluation disables masking.
            block_size: Requested mask side, clipped at runtime to min(block_size,H,W).

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['pointwise_conv2d'], *, in_channels: int, out_channels: int, stride: int | tuple[int, int] | list[int] = 1, bias: bool = True) -> ModelBuilder:
        """Keyword contract for pointwise_conv2d; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'pointwise_conv2d'.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            bias: Include a learned additive convolution bias for each output channel.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['depthwise_conv2d'], *, in_channels: int, depth_multiplier: int = 1, kernel_size: int | tuple[int, int] | list[int] = 3, stride: int | tuple[int, int] | list[int] = 1, padding: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, bias: bool = True) -> ModelBuilder:
        """Keyword contract for depthwise_conv2d; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'depthwise_conv2d'.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            depth_multiplier: Independent filters per input channel; depthwise output/intermediate C equals in_channels*depth_multiplier and must not exceed 8192.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            padding: Symmetric padding on each spatial axis; changes output geometry without changing channel count.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            bias: Include a learned additive convolution bias for each output channel.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['depthwise_separable_conv2d'], *, in_channels: int, out_channels: int, depth_multiplier: int = 1, kernel_size: int | tuple[int, int] | list[int] = 3, stride: int | tuple[int, int] | list[int] = 1, padding: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, bias: bool = True) -> ModelBuilder:
        """Keyword contract for depthwise_separable_conv2d; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'depthwise_separable_conv2d'.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            depth_multiplier: Independent filters per input channel; depthwise output/intermediate C equals in_channels*depth_multiplier and must not exceed 8192.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            padding: Symmetric padding on each spatial axis; changes output geometry without changing channel count.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            bias: Include a learned additive convolution bias for each output channel.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['spatial_separable_conv2d'], *, in_channels: int, out_channels: int, intermediate_channels: int, kernel_size: int | tuple[int, int] | list[int] = 3, stride: int | tuple[int, int] | list[int] = 1, padding: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, bias: bool = True) -> ModelBuilder:
        """Keyword contract for spatial_separable_conv2d; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'spatial_separable_conv2d'.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            intermediate_channels: Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            padding: Symmetric padding on each spatial axis; changes output geometry without changing channel count.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            bias: Include a learned additive convolution bias for each output channel.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['factorized_conv2d'], *, in_channels: int, out_channels: int, intermediate_channels: int, kernel_size: int | tuple[int, int] | list[int] = 3, stride: int | tuple[int, int] | list[int] = 1, padding: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, bias: bool = True) -> ModelBuilder:
        """Keyword contract for factorized_conv2d; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'factorized_conv2d'.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            intermediate_channels: Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            padding: Symmetric padding on each spatial axis; changes output geometry without changing channel count.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            bias: Include a learned additive convolution bias for each output channel.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['conv_transpose2d'], *, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int] | list[int] = 3, stride: int | tuple[int, int] | list[int] = 1, padding: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, bias: bool = True, groups: int = 1, output_padding: int | tuple[int, int] | list[int] = 0) -> ModelBuilder:
        """Keyword contract for conv_transpose2d; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'conv_transpose2d'.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            padding: Symmetric padding on each spatial axis; changes output geometry without changing channel count.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            bias: Include a learned additive convolution bias for each output channel.
            groups: Channel partitions; must divide both convolution channel counts. ChannelShuffle requires channels % groups == 0.
            output_padding: Additional output extent for transposed convolution, not input padding; each axis must be less than stride or dilation on that axis.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['channel_layer_norm'], *, channels: int, eps: float = 1e-05, affine: bool = True) -> ModelBuilder:
        """Keyword contract for channel_layer_norm; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'channel_layer_norm'.
            channels: Feature channels C; gates and normalization preserve this channel count.
            eps: Positive numerical stabilizer used in normalization. Embedding normalization also floors it at the dtype tiny value.
            affine: Learn per-channel scale and bias when True; otherwise use normalization without learned affine parameters.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['instance_norm'], *, num_features: int, eps: float = 1e-05, momentum: float = 0.1, affine: bool = False, track_running_stats: bool = False) -> ModelBuilder:
        """Keyword contract for instance_norm; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'instance_norm'.
            num_features: Normalization channel count; must match input C.
            eps: Positive numerical stabilizer used in normalization. Embedding normalization also floors it at the dtype tiny value.
            momentum: Running-statistic update factor; None selects cumulative averaging for BatchNorm.
            affine: Learn per-channel scale and bias when True; otherwise use normalization without learned affine parameters.
            track_running_stats: Track running mean/variance for evaluation. Without tracking, input-statistic sample requirements also apply in eval.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['leaky_relu'], *, negative_slope: float = 0.01) -> ModelBuilder:
        """Keyword contract for leaky_relu; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'leaky_relu'.
            negative_slope: Nonnegative multiplier for negative LeakyReLU inputs.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['prelu'], *, num_parameters: int = 1, init: float = 0.25) -> ModelBuilder:
        """Keyword contract for prelu; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'prelu'.
            num_parameters: PReLU slope count; must be 1 (shared) or equal to incoming C.
            init: Initial learned PReLU negative slope; finite and representable as float32.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['gelu'], *, approximate: _typing.Literal['none', 'tanh'] = 'none') -> ModelBuilder:
        """Keyword contract for gelu; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'gelu'.
            approximate: GELU calculation: none uses the exact error function; tanh uses the approximation.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['mish']) -> ModelBuilder:
        """Keyword contract for mish; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'mish'.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['hardsigmoid']) -> ModelBuilder:
        """Keyword contract for hardsigmoid; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'hardsigmoid'.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['max_pool2d'], *, kernel_size: int | tuple[int, int] | list[int], stride: int | tuple[int, int] | list[int] | None = None, padding: int | tuple[int, int] | list[int] = 0, dilation: int | tuple[int, int] | list[int] = 1, ceil_mode: bool = False) -> ModelBuilder:
        """Keyword contract for max_pool2d; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'max_pool2d'.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            padding: Symmetric padding on each spatial axis; changes output geometry without changing channel count.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            ceil_mode: Round pooling output sizes upward, then discard windows beginning wholly in right/bottom padding.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['avg_pool2d'], *, kernel_size: int | tuple[int, int] | list[int], stride: int | tuple[int, int] | list[int] | None = None, padding: int | tuple[int, int] | list[int] = 0, ceil_mode: bool = False, count_include_pad: bool = True) -> ModelBuilder:
        """Keyword contract for avg_pool2d; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'avg_pool2d'.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            padding: Symmetric padding on each spatial axis; changes output geometry without changing channel count.
            ceil_mode: Round pooling output sizes upward, then discard windows beginning wholly in right/bottom padding.
            count_include_pad: Include zero-padded elements in the average-pooling divisor.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['adaptive_avg_pool2d'], *, output_size: int | tuple[int, int] | list[int]) -> ModelBuilder:
        """Keyword contract for adaptive_avg_pool2d; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'adaptive_avg_pool2d'.
            output_size: Target H/W grid; integer means a square. Resize requires exactly one of output_size and scale_factor.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['adaptive_max_pool2d'], *, output_size: int | tuple[int, int] | list[int]) -> ModelBuilder:
        """Keyword contract for adaptive_max_pool2d; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'adaptive_max_pool2d'.
            output_size: Target H/W grid; integer means a square. Resize requires exactly one of output_size and scale_factor.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['global_avg_pool2d']) -> ModelBuilder:
        """Keyword contract for global_avg_pool2d; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'global_avg_pool2d'.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['bottleneck_residual'], *, in_channels: int, out_channels: int, intermediate_channels: int, kernel_size: int = 3, stride: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, normalization: _typing.Literal['batch', 'group', None] = 'batch', activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu', attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none', dropout: float = 0.0, drop_path: float = 0.0) -> ModelBuilder:
        """Keyword contract for bottleneck_residual; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'bottleneck_residual'.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            intermediate_channels: Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            normalization: Normalization of residual main-branch features; batch, group or None.
            activation: Nonlinearity applied inside this block; only the declared registry choices are accepted.
            attention: Optional attention applied after the residual branch; nested legacy attention must use matching out_channels.
            dropout: Training-only activation/head dropout probability; disabled in eval.
            drop_path: Training-only per-sample residual path drop probability; disabled in eval.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['preactivation_residual'], *, in_channels: int, out_channels: int, intermediate_channels: int, kernel_size: int = 3, stride: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, normalization: _typing.Literal['batch', 'group', None] = 'batch', activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu', attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none', dropout: float = 0.0, drop_path: float = 0.0) -> ModelBuilder:
        """Keyword contract for preactivation_residual; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'preactivation_residual'.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            intermediate_channels: Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            normalization: Normalization of residual main-branch features; batch, group or None.
            activation: Nonlinearity applied inside this block; only the declared registry choices are accepted.
            attention: Optional attention applied after the residual branch; nested legacy attention must use matching out_channels.
            dropout: Training-only activation/head dropout probability; disabled in eval.
            drop_path: Training-only per-sample residual path drop probability; disabled in eval.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['wide_residual_block'], *, in_channels: int, out_channels: int, intermediate_channels: int, kernel_size: int = 3, stride: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, normalization: _typing.Literal['batch', 'group', None] = 'batch', activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu', attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none', dropout: float = 0.0, drop_path: float = 0.0) -> ModelBuilder:
        """Keyword contract for wide_residual_block; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'wide_residual_block'.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            intermediate_channels: Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            normalization: Normalization of residual main-branch features; batch, group or None.
            activation: Nonlinearity applied inside this block; only the declared registry choices are accepted.
            attention: Optional attention applied after the residual branch; nested legacy attention must use matching out_channels.
            dropout: Training-only activation/head dropout probability; disabled in eval.
            drop_path: Training-only per-sample residual path drop probability; disabled in eval.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['grouped_bottleneck'], *, in_channels: int, out_channels: int, intermediate_channels: int, kernel_size: int = 3, stride: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, normalization: _typing.Literal['batch', 'group', None] = 'batch', activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu', attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none', dropout: float = 0.0, drop_path: float = 0.0, groups: int = 1) -> ModelBuilder:
        """Keyword contract for grouped_bottleneck; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'grouped_bottleneck'.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            intermediate_channels: Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            normalization: Normalization of residual main-branch features; batch, group or None.
            activation: Nonlinearity applied inside this block; only the declared registry choices are accepted.
            attention: Optional attention applied after the residual branch; nested legacy attention must use matching out_channels.
            dropout: Training-only activation/head dropout probability; disabled in eval.
            drop_path: Training-only per-sample residual path drop probability; disabled in eval.
            groups: Channel partitions; must divide both convolution channel counts. ChannelShuffle requires channels % groups == 0.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['inverted_bottleneck'], *, in_channels: int, out_channels: int, expansion_ratio: int = 2, kernel_size: int = 3, stride: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, normalization: _typing.Literal['batch', 'group', None] = 'batch', activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu', attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none', dropout: float = 0.0, drop_path: float = 0.0) -> ModelBuilder:
        """Keyword contract for inverted_bottleneck; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'inverted_bottleneck'.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            expansion_ratio: Inverted-bottleneck channel multiplier; intermediate C=in_channels*expansion_ratio must not exceed 8192.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            normalization: Normalization of residual main-branch features; batch, group or None.
            activation: Nonlinearity applied inside this block; only the declared registry choices are accepted.
            attention: Optional attention applied after the residual branch; nested legacy attention must use matching out_channels.
            dropout: Training-only activation/head dropout probability; disabled in eval.
            drop_path: Training-only per-sample residual path drop probability; disabled in eval.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['gated_residual'], *, in_channels: int, out_channels: int, intermediate_channels: int, kernel_size: int = 3, stride: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, normalization: _typing.Literal['batch', 'group', None] = 'batch', activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu', attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none', dropout: float = 0.0, drop_path: float = 0.0, gate_init: float = 1.0, learnable_gate: bool = True) -> ModelBuilder:
        """Keyword contract for gated_residual; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'gated_residual'.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            intermediate_channels: Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            normalization: Normalization of residual main-branch features; batch, group or None.
            activation: Nonlinearity applied inside this block; only the declared registry choices are accepted.
            attention: Optional attention applied after the residual branch; nested legacy attention must use matching out_channels.
            dropout: Training-only activation/head dropout probability; disabled in eval.
            drop_path: Training-only per-sample residual path drop probability; disabled in eval.
            gate_init: Initial scalar residual contribution in [0,1]; zero suppresses the residual main branch initially.
            learnable_gate: Store the residual gate as a learned scalar when True, otherwise a fixed scalar.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['spatial_attention'], *, channels: int, kernel_size: int = 7) -> ModelBuilder:
        """Keyword contract for spatial_attention; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'spatial_attention'.
            channels: Feature channels C; gates and normalization preserve this channel count.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['channel_attention'], *, channels: int, reduction: int = 8) -> ModelBuilder:
        """Keyword contract for channel_attention; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'channel_attention'.
            channels: Feature channels C; gates and normalization preserve this channel count.
            reduction: Channel-gate reduction divisor; hidden width is max(1,C//reduction), or max(8,C//reduction) for coordinate attention.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['spatial_self_attention'], *, channels: int, embedding_dim: int | None = None, dropout: float = 0.0) -> ModelBuilder:
        """Keyword contract for spatial_self_attention; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'spatial_self_attention'.
            channels: Feature channels C; gates and normalization preserve this channel count.
            embedding_dim: Attention projection width or output embedding width, according to the component. None in self-attention uses C; QKV requires 3*embedding_dim<=8192.
            dropout: Training-only activation/head dropout probability; disabled in eval.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['multihead_spatial_attention'], *, channels: int, embedding_dim: int | None = None, num_heads: int = 4, dropout: float = 0.0) -> ModelBuilder:
        """Keyword contract for multihead_spatial_attention; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'multihead_spatial_attention'.
            channels: Feature channels C; gates and normalization preserve this channel count.
            embedding_dim: Attention projection width or output embedding width, according to the component. None in self-attention uses C; QKV requires 3*embedding_dim<=8192.
            num_heads: Parallel attention heads; must divide embedding_dim. Spatial attention is limited to 1024 tokens and N*heads*tokens^2<=4194304.
            dropout: Training-only activation/head dropout probability; disabled in eval.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['dropout'], *, p: float = 0.5) -> ModelBuilder:
        """Keyword contract for dropout; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'dropout'.
            p: Probability of dropping elements, channels or paths during training; evaluation disables masking.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['dropout2d'], *, p: float = 0.5) -> ModelBuilder:
        """Keyword contract for dropout2d; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'dropout2d'.
            p: Probability of dropping elements, channels or paths during training; evaluation disables masking.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['drop_path'], *, p: float = 0.0) -> ModelBuilder:
        """Keyword contract for drop_path; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'drop_path'.
            p: Probability of dropping elements, channels or paths during training; evaluation disables masking.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['gaussian_noise'], *, std: float = 0.1) -> ModelBuilder:
        """Keyword contract for gaussian_noise; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'gaussian_noise'.
            std: Standard deviation of training-only zero-mean Gaussian noise; finite, nonnegative and at most 1e6.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['resize'], *, output_size: int | tuple[int, int] | list[int] | None = None, scale_factor: float | tuple[float, float] | list[float] | None = None, mode: _typing.Literal['nearest', 'bilinear', 'bicubic', 'area'] = 'nearest', align_corners: bool | None = None) -> ModelBuilder:
        """Keyword contract for resize; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'resize'.
            output_size: Target H/W grid; integer means a square. Resize requires exactly one of output_size and scale_factor.
            scale_factor: Positive H/W resize multipliers, each <=8192. Exactly one of this and output_size must be provided.
            mode: Interpolation or transformation strategy; the closed choices are specific to this operation.
            align_corners: Align endpoint pixel centers for bilinear/bicubic resizing; must be None for nearest/area.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['pixel_shuffle'], *, upscale_factor: int) -> ModelBuilder:
        """Keyword contract for pixel_shuffle; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'pixel_shuffle'.
            upscale_factor: PixelShuffle factor r: C must be divisible by r^2; output is (C/r^2,H*r,W*r).

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['pixel_unshuffle'], *, downscale_factor: int) -> ModelBuilder:
        """Keyword contract for pixel_unshuffle; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'pixel_unshuffle'.
            downscale_factor: PixelUnshuffle factor r: H and W must be divisible by r; output is (C*r^2,H/r,W/r).

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['flatten']) -> ModelBuilder:
        """Keyword contract for flatten; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'flatten'.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['permute'], *, dims: tuple[int, ...] | list[int] = [0, 1, 3, 2]) -> ModelBuilder:
        """Keyword contract for permute; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'permute'.
            dims: Permutation of NCHW axes; only (0,1,2,3) and (0,1,3,2) are accepted, preserving batch/channel axes.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, name: _typing.Literal['reshape'], *, shape: tuple[int, int, int] | list[int]) -> ModelBuilder:
        """Keyword contract for reshape; execution uses the validated implementation.

        Args:
            name: Exact component identifier 'reshape'.
            shape: Target per-sample CHW dimensions; reshape must preserve the number of sample elements and every dimension is 1..8192.

        Returns:
            This ModelBuilder for chaining; build() allocates separately.
        """
        ...

    def add(self, name: str, **params: Any) -> ModelBuilder:
        """Append a registered block transactionally; return self for chaining.

        Args:
            name (str): Registered component name, exact metric label, or stored entity name in this API.
                Required.
            params (Any): Closed registered keyword parameters; inspect ModelRegistry.get(name) for
                supported kinds/defaults. Required.

        Returns:
            This builder after the existing component/reference validation.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
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

        Examples:
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
            path (str | Path): Filesystem path for the specific reader/exporter; no configuration code is
                executed. Required.
            overwrite (bool): Whether an existing output file may be replaced explicitly; inputs remain
                protected. Default: False.

        Returns:
            None after the native file export succeeds.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Examples:
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
            path (str | Path): Filesystem path for the specific reader/exporter; no configuration code is
                executed. Required.

        Returns:
            A validated ModelBuilder instance reconstructed from the supplied data.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Examples:
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
