"""Validated linear composition and safe architecture-only JSON round trips.

The builder reuses existing blocks; it does not implement a new convolution,
attention family, checkpoint format, arbitrary graph compiler or code loader.
"""
from __future__ import annotations

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
        config = json.loads(self._architecture)
        for block,entry in zip(self.features,config['blocks']):
            if entry['name'] in ('repvgg_block','mobileone_block'):
                entry['params']['deploy'] = block.deploy
        return config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return [N, num_classes] logits; reject incompatible input before work."""
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
    """Opt-in schema2 features and explicit prediction head."""
    def __init__(self, features, config, head):
        nn.Module.__init__(self)
        self.features = nn.Sequential(*features)
        self.head = head
        self.input_shape = tuple(config['input_shape'])
        self._architecture = json.dumps(config,sort_keys=True,allow_nan=False)

    @property
    def classifier(self):
        return self.head.linear

    def forward(self,x):
        return self.head(self._forward_features(x))


class ModelBuilder:
    """Configure a bounded CPU classifier without rewriting registered blocks.

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
    """
    def __init__(self, input_shape: tuple[int, int, int], num_classes: int):
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
    def sequential(cls, input_shape, *, head='classification', **head_params):
        """Create an opt-in schema2 chain with an explicit terminal head."""
        return cls.from_config(dict(format='hyso.model_config',schema_version=2,kind='sequential',
                                    input_shape=input_shape,blocks=[],head=dict(name=f'{head}_head',params=head_params)))

    @classmethod
    def named(cls, name: str, **params: Any) -> ModelBuilder:
        """Describe a closed model family; validation allocates no model or weights."""
        return cls.from_config(dict(format='hyso.model_config',schema_version=1,kind='registered_model',name=name,params=params))

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

    def add(self, name: str, **params: Any) -> ModelBuilder:
        """Append a registered block transactionally; return self for chaining."""
        if self._config['kind'] not in ('sequential_classifier','sequential'):
            raise ModelConfigError('add is unavailable for fixed complete-model profiles', code='INVALID_BLOCK_ORDER')
        candidate = self.to_config()
        candidate['blocks'].append(dict(name=name,params=params))
        self._config, _ = self._validate(candidate)
        return self

    def build(self) -> nn.Module:
        """Allocate a fresh CPU model; architecture round trips do not restore weights."""
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
        except ModelConfigError:
            raise
        except (ValueError, TypeError, RecursionError) as error:
            raise ModelConfigError(str(error)) from error
