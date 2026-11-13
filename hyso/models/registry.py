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
from ._profile import DEVELOPMENT
from .regularization import ChannelShuffle, DropBlock2D
from . import components

M4_FAMILIES = ()
M4_BLOCKS = ()
if DEVELOPMENT:
    from .m3_attention import SimAM, TripletAttention, SelectiveKernelAttention
    from .m3_residual import BasicResidualBlock, Bottleneck, PreActivationBlock, WideResidualBlock, ResNeXtBottleneck
    from .architectures import ResNet, PreActResNet, WideResNet, ResNeXt, DenseNet, validate_family
    from .m4_registry import specifications as mobile_specifications, M4_FAMILIES, M4_BLOCKS, validate_mobile, validate_block
from .modular import ModularCNN


_DIAGNOSTICS = {
    'INVALID_OUTPUT_PADDING': ('Çıkış padding değeri geçersiz.', 'Her eksende output_padding değerini stride veya dilation değerinden küçük seçin.'),
    'INVALID_POOLING_CONFIGURATION': ('Pooling ayarları geçersiz.', 'Pozitif kernel/stride ve kernel yarısını aşmayan padding kullanın.'),
    'INTERMEDIATE_SHAPE_ERROR': ('Ara tensor boyutu geçersiz.', 'Ara kanal ve spatial boyutların 1..8192 aralığında olmasını sağlayın.'),
    'STATISTICS_REQUIREMENT': ('Normalizasyon istatistik koşulu çalışma anında kontrol edilir.', 'Belirtilen örnek sayısı koşulunu sağlayın. Running statistics yalnız desteklenen eval modunda kullanılabilir.'),
    'RESOURCE_LIMIT_WARNING': ('Tahmini aktivasyon eleman sayısı uyarı eşiğini aşıyor.', 'Batch, kanal veya spatial boyutları azaltın ve ön doğrulamayı yeniden çalıştırın.'),
    'INVALID_INPUT_SHAPE': ('Giriş boyutu geçersiz.', 'CHW boyutlarını 1..8192 aralığında üç tam sayı olarak belirtin.'),
    'INVALID_PARAMETER': ('Bileşen parametresi veya mimari verisi geçersiz.', 'Bileşenin desteklediği parametreleri ve JSON şemasını kontrol edin.'),
    'CHANNEL_MISMATCH': ('Katmanların kanal sayıları uyuşmuyor.', 'Katmanın giriş kanalını önceki katmanın çıkış kanalıyla eşleştirin.'),
    'GROUP_DIVISIBILITY_ERROR': ('Kanal sayısı grup sayısına bölünemiyor.', 'Grup sayısını ilgili kanal sayılarını bölen pozitif bir tam sayı seçin.'),
    'SPATIAL_DIMENSION_ERROR': ('Katman geçersiz bir uzamsal boyut üretiyor.', 'Kernel, padding ve stride değerlerini giriş yüksekliği ve genişliğine göre düzenleyin.'),
    'INVALID_BLOCK_ORDER': ('Katman sırası veya bağlantı türü desteklenmiyor.', 'Lineer zincirde yalnız NCHW blokları kullanın. Tam modeli ayrı oluşturun.'),
    'PARAMETER_BUDGET_EXCEEDED': ('Model parametre bütçesini aşıyor.', 'Kanal, kernel veya blok sayısını azaltın.'),
    'UNSUPPORTED_COMPONENT': ('Bileşen bu dağıtım profilinde desteklenmiyor.', 'ModelRegistry.list() ile kullanılabilir bileşenleri kontrol edin.'),
    'INVALID_OUTPUT_HEAD': ('Sınıflandırma başlığının çıkış sayısı geçersiz.', 'num_classes değerini 1..8192 aralığında bir tam sayı seçin.'),
}


def _diagnostic_value(value, depth=0):
    """Bound diagnostic JSON values without invoking user repr or callables."""
    if depth > 8:
        return '<depth limit>'
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is str:
        return value[:512]
    if type(value) is float:
        return value if math.isfinite(value) else str(value)
    if type(value) in (list, tuple):
        return [_diagnostic_value(v, depth+1) for v in value[:64]]
    if type(value) is dict:
        return {k[:256]: _diagnostic_value(v, depth+1)
                for k, v in list(value.items())[:64] if type(k) is str}
    return {'type': type(value).__name__}


class ModelConfigError(ValueError):
    """Backward-compatible English exception with JSON-safe bilingual diagnostics."""

    def __init__(self, *messages: Any, code: str = 'INVALID_PARAMETER',
                 layer_index: int | None = None, layer_id: str | None = None,
                 expected: Any = None, actual: Any = None,
                 suggestion: str | None = None) -> None:
        super().__init__(*messages)
        self.code = code
        self.layer_index = layer_index
        self.layer_id = layer_id
        self.expected = _diagnostic_value(expected)
        self.actual = _diagnostic_value(actual)
        self.message_tr, default_suggestion = _DIAGNOSTICS[code]
        self.suggestion = suggestion or default_suggestion

    def at_layer(self, index: int) -> ModelConfigError:
        """Attach zero-based chain context while retaining the English detail."""
        if self.layer_index is None:
            self.layer_index = index
            self.args = (f'block {index}: {self}',)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a detached diagnostic, suitable for CLI or a future editor."""
        return dict(code=self.code, message_tr=self.message_tr, message_en=str(self),
                    layer_index=self.layer_index, layer_id=self.layer_id,
                    expected=_diagnostic_value(self.expected), actual=_diagnostic_value(self.actual),
                    suggestion=self.suggestion)


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
            if p.kind in ('positive_int', 'odd_int', 'pair', 'optional_pair', 'padding', 'int_list'):
                row['constraints'] = dict(minimum=0 if p.kind == 'padding' else 1,
                                          maximum=8192, odd=p.kind == 'odd_int')
                if p.kind == 'int_list': row['constraints']['max_items'] = 16
                if p.kind in ('pair', 'optional_pair', 'padding'):
                    row['constraints']['pair_length'] = 2
                    row['constraints']['nullable'] = p.kind == 'optional_pair'
            elif p.kind in ('positive_float', 'nonnegative_float', 'finite_float', 'probability_closed', 'optional_momentum'):
                row['constraints'] = dict(finite=True, minimum=0 if p.kind != 'finite_float' else None,
                                          exclusive_minimum=p.kind == 'positive_float',
                                          maximum=1 if p.kind in ('probability_closed', 'optional_momentum') else None,
                                          nullable=p.kind == 'optional_momentum')
            elif p.kind == 'bool':
                row['constraints'] = dict(type='boolean')
            if self.name == 'prelu' and p.name == 'init':
                row['constraints'].update(minimum=-3.4028234663852886e38, maximum=3.4028234663852886e38)
            if not p.required:
                row['default'] = _plain(p.default)
            params.append(row)
        return dict(name=self.name, category=self.category, description=self.description,
                    constructor=self.constructor.__module__+'.'+self.constructor.__qualname__,
                    parameters=params, input_condition=self.input_condition,
                    output_condition=self.output_condition, compatibility=list(self.compatibility),
                    layout='NCHW', parameter_dependencies=_dependencies(self.name),
                    common_errors=_common_errors(self.category))


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
      _p('groups', default=1), _p('bias', 'bool', True), _p('dilation', 'pair', 1))),
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
     (_p('num_features'), _p('eps', 'positive_float', 1e-5), _p('momentum', 'optional_momentum', .1),
      _p('affine', 'bool', True), _p('track_running_stats', 'bool', True))),
    ('group_norm', 'normalization', 'PyTorch GroupNorm with explicit groups/channels.', nn.GroupNorm,
     (_p('num_groups'), _p('num_channels'), _p('eps', 'positive_float', 1e-5), _p('affine', 'bool', True))),
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


_SPECS += (
    ('channel_shuffle', 'channel_mixing', ChannelShuffle.__doc__, ChannelShuffle,
     (_CHANNELS, _p('groups', default=2))),
    ('dropblock2d', 'regularization', DropBlock2D.__doc__, DropBlock2D,
     (_p('p', 'probability_closed', .1), _p('block_size', default=3))),
)

_SPECS += components.specifications(_p)


def _dependencies(name):
    dependencies = {
        'conv2d': ['groups divides in_channels and out_channels', 'dilation changes the effective kernel'],
        'conv_norm_act': ['groups divides both channel counts', 'normalization determines convolution bias'],
        'depthwise_conv2d': ['groups=in_channels', 'out_channels=in_channels*depth_multiplier'],
        'depthwise_separable_conv2d': ['intermediate_channels=in_channels*depth_multiplier'],
        'spatial_separable_conv2d': ['H settings apply to the vertical stage, W settings to the horizontal stage'],
        'factorized_conv2d': ['intermediate_channels <= min(in_channels,out_channels)', '1x1 reduction then KxK convolution'],
        'conv_transpose2d': ['groups divides both channel counts', 'output_padding[i] < stride[i] or dilation[i]'],
        'group_norm': ['num_groups divides num_channels'],
        'prelu': ['num_parameters is 1 or the incoming channel count'],
        'max_pool2d': ['stride=None uses kernel_size', 'padding[i] <= floor(kernel_size[i]/2)'],
        'avg_pool2d': ['stride=None uses kernel_size', 'padding[i] <= floor(kernel_size[i]/2)', 'count_include_pad controls the mean divisor'],
        'channel_layer_norm': ['normalizes C at each N,H,W position, external layout stays NCHW'],
        'instance_norm': ['spatial sample count H*W > 1 when using input statistics'],
    }
    return dependencies.get(name, [])


def _common_errors(category):
    return {
        'convolution': ['Match incoming channels and group divisors.', 'Use positive output geometry including every internal stage.'],
        'normalization': ['Match channels and use positive finite eps.', 'Check training/eval statistics requirements.'],
        'activation': ['Use only the documented parameters.', 'PReLU requires one slope or one per channel.'],
        'pooling': ['Use positive output sizes and legal padding.', 'Check ceil-mode right-boundary correction.'],
    }.get(category, ['Use the closed parameter schema described by models show.'])

if DEVELOPMENT:
    _M3_NORM = _p('norm', 'choice', 'batch', ('batch', 'group', None), 'M3 normalization; group uses gcd(8, channels).')
    _SPECS += (
     ('simam','attention','Parameter-free spatial energy gates; singleton variance is zero.',SimAM,(_CHANNELS,_p('epsilon','positive_float',1e-4))),
     ('triplet','attention','Convolutional rotated-axis Triplet Attention with BN.',TripletAttention,(_CHANNELS,_p('kernel_size','odd_int',7),_p('spatial','bool',True))),
     ('selective_kernel','attention','Dilated branch fusion with per-channel branch softmax.',SelectiveKernelAttention,(_CHANNELS,_p('branches',default=2),_p('groups',default=1),_p('reduction',default=16),_p('minimum',default=32))),
     *tuple((name,'residual',ctor.__doc__,ctor,(_IN,_OUT,_p('stride',default=1),_M3_NORM)+opts) for name,ctor,opts in (
      ('basic_residual',BasicResidualBlock,()),('bottleneck',Bottleneck,(_p('groups',default=1),_p('width_per_group',default=64))),
      ('preactivation',PreActivationBlock,(_p('bottleneck','bool',False),_p('dropout','probability',0.))),
      ('wide_residual',WideResidualBlock,(_p('dropout','probability',0.),)),
      ('resnext_bottleneck',ResNeXtBottleneck,(_p('groups',default=32),_p('width_per_group',default=4))))),
     *tuple((name,'model',ctor.__doc__,ctor,(_p('depth',default=depth),_p('input_channels',default=3),_p('num_classes',default=classes))+opts)
      for name,ctor,depth,classes,opts in (
       ('resnet',ResNet,18,1000,(_M3_NORM,)),('preact_resnet',PreActResNet,18,1000,(_p('base_channels',default=64),_M3_NORM)),
       ('wide_resnet',WideResNet,28,10,(_p('widen_factor',default=10),_p('dropout','probability',0.),_M3_NORM)),
       ('resnext',ResNeXt,50,1000,(_p('groups',default=32),_p('width_per_group',default=4),_M3_NORM)),
       ('densenet',DenseNet,121,1000,(_p('growth_rate',default=32),_p('dropout','probability',0.)))))
    )

    _SPECS += tuple(row for row in mobile_specifications(_p) if row[0] not in ('channel_shuffle', 'dropblock2d'))

_IO = {
    'mobile': ('Floating NCHW with matching in_channels; native BN statistics constraints apply.',
               '[N,out_channels,ceil(H/stride),ceil(W/stride)]; closed internal branch topology.'),
    'channel_mixing': ('Floating NCHW with matching channels.', 'Same NCHW shape; validated channel partition.'),
    'regularization': ('Nonempty floating NCHW.', 'Same shape and dtype; explicit training-only masking.'),
    'convolution': ('Floating [N,in_channels,H,W]; groups divides both channel counts.',
                    'Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See parameter_dependencies and preflight.'),
    'pooling': ('Floating nonempty NCHW; legal window/padding or positive adaptive output_size.',
                'Floating NCHW with unchanged channels; global average returns [N,C,1,1].'),
    'residual': ('Floating [N,in_channels,H,W]; attention channels equal out_channels.',
                 '[N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.'),
    'attention': ('Floating [N,channels,H,W], all dimensions positive.',
                  'Same NCHW shape; multiplicative gates, no channel or spatial resizing.'),
    'normalization': ('Floating NCHW; C matches num_features, num_channels or channels. GroupNorm groups divide C.',
                      'Same NCHW shape; channel LayerNorm normalizes C per spatial position. Native statistics constraints apply.'),
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
    have an estimated 10-million-parameter limit for sequential chains; named
    M3 families use a separate 200-million bound. Native normalization retains
    PyTorch's training batch/statistics requirements.
    """

    def __init__(self) -> None:
        self._entries = MappingProxyType({name: ComponentSpec(
            name, category, description, ctor, params,
            _IO[category][0], _IO[category][1],
            (('HYSo/PyTorch component; CPU construction only.' if name in components.NEW_NAMES or name in ('conv_norm_act','conv2d','residual','se','cbam','eca','coordinate','batch_norm','group_norm','relu','silu','hardswish','identity','modular_cnn','channel_shuffle','dropblock2d') else 'Closed mobile/edge constructor or torchvision adapter; CPU construction only.' if name in M4_FAMILIES+M4_BLOCKS else 'Closed M3 constructor / torchvision adapter; CPU construction only.'),
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
            raise ModelConfigError(f'unknown component: {name!r}', code='UNSUPPORTED_COMPONENT',
                                   expected='registered component name', actual=name)
        return self._entries[name]

    def validate(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Validate plain parameters without constructing a module; return a copy."""
        spec = self.get(name)
        p = _plain(params)
        if type(p) is not dict:
            raise ModelConfigError('params must be a dictionary')
        unknown = set(p)-{v.name for v in spec.parameters}
        if unknown:
            raise ModelConfigError(f'{name}: unknown parameters {sorted(unknown)}',
                                   expected=[v.name for v in spec.parameters], actual=sorted(unknown))
        result = {}
        for field in spec.parameters:
            if field.name not in p and field.required:
                raise ModelConfigError(f'{name}: missing parameter {field.name}',
                                       expected=field.name, actual='missing')
            v = p.get(field.name, _plain(field.default))
            kind = field.kind
            if kind in ('positive_int', 'odd_int'):
                valid = type(v) is int and 1 <= v <= 8192
                if kind == 'odd_int': valid = valid and v % 2 == 1
            elif kind in ('pair', 'padding', 'optional_pair'):
                values = [v, v] if type(v) is int else v
                valid = type(values) is list and len(values) == 2 and all(
                    type(x) is int and (0 if kind == 'padding' else 1) <= x <= 8192 for x in values)
                if valid: v = values
                if kind == 'optional_pair' and v is None:
                    valid = True
            elif kind == 'int_list':
                valid = type(v) is list and 1 <= len(v) <= 16 and all(type(x) is int and 1 <= x <= 8192 for x in v)
            elif kind in ('positive_float', 'probability'):
                valid = type(v) in (int,float) and abs(v)<=1e308 and math.isfinite(v) and (v>0 if kind=='positive_float' else 0<=v<1)
            elif kind == 'optional_probability':
                valid = v is None or type(v) in (int,float) and 0 <= v < 1
            elif kind == 'optional_momentum':
                valid = v is None or type(v) in (int,float) and 0 <= v <= 1
            elif kind in ('finite_float', 'nonnegative_float'):
                valid = type(v) in (int,float) and abs(v) <= 1e308 and math.isfinite(v)
                if kind == 'nonnegative_float': valid = valid and v >= 0
            elif kind == 'probability_closed':
                valid = type(v) in (int,float) and 0 <= v <= 1
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
                raise ModelConfigError(f'{name}.{field.name}: invalid {kind} value',
                                       code='INVALID_OUTPUT_HEAD' if field.name == 'num_classes' else 'INVALID_PARAMETER',
                                       expected=dict(parameter=field.name, kind=kind, choices=list(field.choices)), actual=v)
            result[field.name] = v
        components.validate(name, result, ModelConfigError)
        if name == 'channel_shuffle' and result['channels'] % result['groups']:
            divisors = [v for v in range(1, result['channels']+1) if result['channels'] % v == 0]
            raise ModelConfigError('groups must divide channels', code='GROUP_DIVISIBILITY_ERROR',
                                   expected=divisors, actual=result['groups'],
                                   suggestion=f'Geçerli groups değerleri {divisors}.')
        if name == 'group_norm' and result['num_channels'] % result['num_groups']:
            divisors = [v for v in range(1, result['num_channels']+1) if result['num_channels'] % v == 0]
            raise ModelConfigError('num_groups must divide num_channels', code='GROUP_DIVISIBILITY_ERROR',
                                   expected=divisors, actual=result['num_groups'],
                                   suggestion=f'Geçerli num_groups değerleri {divisors}.')
        if name == 'residual' and result['attention'] is not None:
            if result['attention']['params']['channels'] != result['out_channels']:
                raise ModelConfigError('residual attention channels must equal out_channels', code='CHANNEL_MISMATCH',
                                       expected=result['out_channels'], actual=result['attention']['params']['channels'])
        if name == 'modular_cnn':
            if len({len(result[k]) for k in ('widths', 'repeats', 'strides')}) != 1 or sum(result['repeats']) > 64:
                raise ModelConfigError('ModularCNN stage lengths must match; at most 64 total blocks')
        if name in ('resnet','preact_resnet','wide_resnet','resnext','densenet'):
            try: validate_family(name,result)
            except ValueError as exc: raise ModelConfigError(str(exc)) from exc
        if name in ('basic_residual','bottleneck','preactivation','wide_residual','resnext_bottleneck'):
            if result['stride'] not in (1,2): raise ModelConfigError('residual stride must be 1 or 2')
            if (name in ('bottleneck','resnext_bottleneck') or result.get('bottleneck')) and result['out_channels']%4:
                raise ModelConfigError('expanded output channels must be divisible by 4')
            if name in ('bottleneck','resnext_bottleneck') and int(result['out_channels']/4*result['width_per_group']/64)<1:
                raise ModelConfigError('bottleneck inner width must be positive')
        if name=='selective_kernel' and (not 2<=result['branches']<=4 or result['channels']%result['groups']):
            raise ModelConfigError('SK branches must be 2..4; groups must divide channels')
        try:
            if name in M4_FAMILIES:
                validate_mobile(name,result)
            if name in M4_BLOCKS:
                validate_block(name,result)
        except ValueError as exc:
            raise ModelConfigError(str(exc)) from exc
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
