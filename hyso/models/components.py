"""HYSo NCHW components composed from public PyTorch operators.

Written for Stage 13.5.2 from the operator contracts, without copied model
implementations. The root MIT license and PyTorch dependency notices apply.
All geometry and parameter accounting below operate on Python integers.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _parameters(name, **params):
    from .registry import ModelRegistry
    return ModelRegistry().validate(name, params)


def _input(x, channels):
    from .registry import ModelConfigError
    if not isinstance(x, torch.Tensor) or x.ndim != 4 or min(x.shape) < 1 or not x.is_floating_point():
        raise ModelConfigError('expected a nonempty floating NCHW tensor', code='INVALID_INPUT_SHAPE',
                               expected=['N', channels, 'H', 'W'], actual=getattr(x, 'shape', None))
    if x.shape[1] != channels:
        raise ModelConfigError('input channels do not match the component', code='CHANNEL_MISMATCH',
                               expected=channels, actual=x.shape[1])


class PointwiseConv2D(nn.Conv2d):
    """1x1 channel mixing with zero padding and configurable stride/bias."""
    def __init__(self, in_channels, out_channels, stride=1, bias=True):
        p = _parameters('pointwise_conv2d', in_channels=in_channels, out_channels=out_channels,
                        stride=stride, bias=bias)
        super().__init__(**p, kernel_size=1, padding=0)


class DepthwiseConv2D(nn.Conv2d):
    """Independent spatial filters, groups=C and output=C*depth_multiplier."""
    def __init__(self, in_channels, depth_multiplier=1, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=True):
        p = _parameters('depthwise_conv2d', in_channels=in_channels, depth_multiplier=depth_multiplier,
                        kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        multiplier = p.pop('depth_multiplier')
        super().__init__(**p, out_channels=in_channels*multiplier, groups=in_channels)


class DepthwiseSeparableConv2D(nn.Module):
    """Depthwise spatial filtering followed by dense 1x1 channel mixing."""
    def __init__(self, in_channels, out_channels, depth_multiplier=1, kernel_size=3,
                 stride=1, padding=1, dilation=1, bias=True):
        p = _parameters('depthwise_separable_conv2d', in_channels=in_channels, out_channels=out_channels,
                        depth_multiplier=depth_multiplier, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias)
        super().__init__()
        middle = p['in_channels']*p['depth_multiplier']
        self.depthwise = nn.Conv2d(p['in_channels'], middle, p['kernel_size'], p['stride'],
                                   p['padding'], p['dilation'], p['in_channels'], bias=p['bias'])
        self.pointwise = nn.Conv2d(middle, p['out_channels'], 1, bias=p['bias'])

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SpatialSeparableConv2D(nn.Module):
    """(kH,1) -> (1,kW), with separate H/W stride, padding and dilation."""
    def __init__(self, in_channels, out_channels, intermediate_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, bias=True):
        p = _parameters('spatial_separable_conv2d', in_channels=in_channels, out_channels=out_channels,
                        intermediate_channels=intermediate_channels, kernel_size=kernel_size,
                        stride=stride, padding=padding, dilation=dilation, bias=bias)
        super().__init__()
        kh, kw = p['kernel_size']
        sh, sw = p['stride']
        ph, pw = p['padding']
        dh, dw = p['dilation']
        self.vertical = nn.Conv2d(in_channels, intermediate_channels, (kh, 1), (sh, 1),
                                  (ph, 0), (dh, 1), bias=bias)
        self.horizontal = nn.Conv2d(intermediate_channels, out_channels, (1, kw), (1, sw),
                                    (0, pw), (1, dw), bias=bias)

    def forward(self, x):
        return self.horizontal(self.vertical(x))


class FactorizedConv2D(nn.Module):
    """Dense 1x1 channel bottleneck -> dense KxK spatial convolution.

    intermediate_channels must not exceed either boundary channel count.
    Spatial stride/padding/dilation belong exclusively to the second operator.
    There is no hidden normalization, activation or residual connection.
    """
    def __init__(self, in_channels, out_channels, intermediate_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, bias=True):
        p = _parameters('factorized_conv2d', in_channels=in_channels, out_channels=out_channels,
                        intermediate_channels=intermediate_channels, kernel_size=kernel_size,
                        stride=stride, padding=padding, dilation=dilation, bias=bias)
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, intermediate_channels, 1, bias=bias)
        self.spatial = nn.Conv2d(intermediate_channels, out_channels, p['kernel_size'],
                                 p['stride'], p['padding'], p['dilation'], bias=bias)

    def forward(self, x):
        return self.spatial(self.reduce(x))


class ChannelLayerNorm(nn.Module):
    """Normalize C independently at each (N,H,W), preserving public NCHW layout.

    An internal movedim exposes C to torch LayerNorm, then restores NCHW.
    This differs from LayerNorm over the entire CHW sample.
    """
    def __init__(self, channels, eps=1e-5, affine=True):
        p = _parameters('channel_layer_norm', channels=channels, eps=eps, affine=affine)
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels, eps=p['eps'], elementwise_affine=p['affine'])

    def forward(self, x):
        _input(x, self.channels)
        return self.norm(x.movedim(1, -1)).movedim(-1, 1)


class InstanceNorm2D(nn.InstanceNorm2d):
    """NCHW InstanceNorm with mandatory channel matching even without affine."""
    def __init__(self, num_features, eps=1e-5, momentum=.1, affine=False, track_running_stats=False):
        p = _parameters('instance_norm', num_features=num_features, eps=eps, momentum=momentum,
                        affine=affine, track_running_stats=track_running_stats)
        super().__init__(**p)

    def forward(self, x):
        _input(x, self.num_features)
        return super().forward(x)


class GlobalAveragePool2D(nn.AdaptiveAvgPool2d):
    """Average spatial axes and retain [N,C,1,1]."""
    def __init__(self):
        super().__init__((1, 1))


CONVOLUTIONS = frozenset(('conv2d', 'pointwise_conv2d', 'depthwise_conv2d',
                          'depthwise_separable_conv2d', 'spatial_separable_conv2d',
                          'factorized_conv2d', 'conv_transpose2d'))
POOLING = frozenset(('max_pool2d', 'avg_pool2d', 'adaptive_avg_pool2d',
                    'adaptive_max_pool2d', 'global_avg_pool2d'))
NEW_NAMES = frozenset(('pointwise_conv2d', 'depthwise_conv2d', 'depthwise_separable_conv2d',
                       'spatial_separable_conv2d', 'factorized_conv2d', 'conv_transpose2d',
                       'channel_layer_norm', 'instance_norm', 'leaky_relu', 'prelu',
                       'gelu', 'mish', 'hardsigmoid')) | POOLING


def specifications(p):
    """Closed additions to the shared Registry. All defaults are plain data."""
    cin, cout = p('in_channels'), p('out_channels')
    spatial = (p('kernel_size', 'pair', 3), p('stride', 'pair', 1),
               p('padding', 'padding', 1), p('dilation', 'pair', 1), p('bias', 'bool', True))
    middle = p('intermediate_channels')
    return (
        ('pointwise_conv2d', 'convolution', PointwiseConv2D.__doc__, PointwiseConv2D,
         (cin, cout, p('stride', 'pair', 1), p('bias', 'bool', True))),
        ('depthwise_conv2d', 'convolution', DepthwiseConv2D.__doc__, DepthwiseConv2D,
         (cin, p('depth_multiplier', default=1), *spatial)),
        ('depthwise_separable_conv2d', 'convolution', DepthwiseSeparableConv2D.__doc__, DepthwiseSeparableConv2D,
         (cin, cout, p('depth_multiplier', default=1), *spatial)),
        ('spatial_separable_conv2d', 'convolution', SpatialSeparableConv2D.__doc__, SpatialSeparableConv2D,
         (cin, cout, middle, *spatial)),
        ('factorized_conv2d', 'convolution', FactorizedConv2D.__doc__, FactorizedConv2D,
         (cin, cout, middle, *spatial)),
        ('conv_transpose2d', 'convolution', 'Transposed convolution with explicit output padding.', nn.ConvTranspose2d,
         (cin, cout, *spatial, p('groups', default=1), p('output_padding', 'padding', 0))),
        ('channel_layer_norm', 'normalization', ChannelLayerNorm.__doc__, ChannelLayerNorm,
         (p('channels'), p('eps', 'positive_float', 1e-5), p('affine', 'bool', True))),
        ('instance_norm', 'normalization', InstanceNorm2D.__doc__, InstanceNorm2D,
         (p('num_features'), p('eps', 'positive_float', 1e-5), p('momentum', 'probability_closed', .1),
          p('affine', 'bool', False), p('track_running_stats', 'bool', False))),
        ('leaky_relu', 'activation', 'ReLU with an explicit nonnegative negative slope.', nn.LeakyReLU,
         (p('negative_slope', 'nonnegative_float', .01),)),
        ('prelu', 'activation', 'Learnable slope shared by all channels or one per channel.', nn.PReLU,
         (p('num_parameters', default=1), p('init', 'finite_float', .25))),
        ('gelu', 'activation', 'Gaussian error linear unit with exact or tanh approximation.', nn.GELU,
         (p('approximate', 'choice', 'none', ('none', 'tanh')),)),
        ('mish', 'activation', 'x*tanh(softplus(x)), out of place.', nn.Mish, ()),
        ('hardsigmoid', 'activation', 'Piecewise linear bounded sigmoid, out of place.', nn.Hardsigmoid, ()),
        ('max_pool2d', 'pooling', 'Spatial maximum with floor/ceil windows and optional dilation.', nn.MaxPool2d,
         (p('kernel_size', 'pair'), p('stride', 'optional_pair', None), p('padding', 'padding', 0),
          p('dilation', 'pair', 1), p('ceil_mode', 'bool', False))),
        ('avg_pool2d', 'pooling', 'Spatial mean with explicit padding divisor behavior.', nn.AvgPool2d,
         (p('kernel_size', 'pair'), p('stride', 'optional_pair', None), p('padding', 'padding', 0),
          p('ceil_mode', 'bool', False), p('count_include_pad', 'bool', True))),
        ('adaptive_avg_pool2d', 'pooling', 'Spatial averages into a fixed positive H/W grid.', nn.AdaptiveAvgPool2d,
         (p('output_size', 'pair'),)),
        ('adaptive_max_pool2d', 'pooling', 'Spatial maxima into a fixed positive H/W grid.', nn.AdaptiveMaxPool2d,
         (p('output_size', 'pair'),)),
        ('global_avg_pool2d', 'pooling', GlobalAveragePool2D.__doc__, GlobalAveragePool2D, ()),
    )


def validate(name, p, error):
    """Cross-field validation after the Registry's scalar/type validation."""
    if name in CONVOLUTIONS | {'conv_norm_act'} and 'groups' in p:
        channels = (p['in_channels'], p['out_channels'])
        if any(c % p['groups'] for c in channels):
            common = math.gcd(*channels)
            divisors = [v for v in range(1, common+1) if common % v == 0]
            raise error('groups must divide both channel counts', code='GROUP_DIVISIBILITY_ERROR',
                        expected=divisors, actual=p['groups'], suggestion=f'Geçerli groups değerleri {divisors}.')
    if name in ('depthwise_conv2d', 'depthwise_separable_conv2d') and p['in_channels']*p['depth_multiplier'] > 8192:
        raise error('depthwise intermediate channels exceed 8192', code='INTERMEDIATE_SHAPE_ERROR',
                    expected='C <= 8192', actual=p['in_channels']*p['depth_multiplier'])
    if name == 'factorized_conv2d' and p['intermediate_channels'] > min(p['in_channels'], p['out_channels']):
        raise error('factorized bottleneck exceeds boundary channels', expected=min(p['in_channels'], p['out_channels']),
                    actual=p['intermediate_channels'], suggestion='intermediate_channels değerini min(in_channels,out_channels) sınırına indirin.')
    if name == 'prelu' and abs(p['init']) > 3.4028234663852886e38:
        raise error('PReLU init must be representable in default CPU float32 weights',
                    expected='abs(init) <= float32 maximum', actual=p['init'])
    if name == 'conv_transpose2d' and any(o >= s and o >= d for o, s, d in zip(p['output_padding'], p['stride'], p['dilation'])):
        raise error('output_padding must be less than stride or dilation on each axis', code='INVALID_OUTPUT_PADDING',
                    expected='output_padding[i] < stride[i] or dilation[i]', actual=p['output_padding'])
    if name in ('max_pool2d', 'avg_pool2d') and any(pad > k//2 for pad, k in zip(p['padding'], p['kernel_size'])):
        raise error('pool padding must not exceed floor(kernel_size/2)', code='INVALID_POOLING_CONFIGURATION',
                    expected=[k//2 for k in p['kernel_size']], actual=p['padding'])


def _conv_geometry(shape, out_channels, kernel, stride, padding, dilation):
    return (out_channels, *((v+2*p-d*(k-1)-1)//s+1
                           for v, p, d, k, s in zip(shape[1:], padding, dilation, kernel, stride)))


def stages(name, p, shape):
    """Return every materialized convolution stage, or the pooling output."""
    c, h, w = shape
    if name == 'pointwise_conv2d':
        return [_conv_geometry(shape, p['out_channels'], (1,1), p['stride'], (0,0), (1,1))]
    if name == 'conv_transpose2d':
        spatial = []
        for size, stride, pad, dilation, kernel, extra in zip(
                (h,w), p['stride'], p['padding'], p['dilation'], p['kernel_size'], p['output_padding']):
            spatial.append((size-1)*stride-2*pad+dilation*(kernel-1)+extra+1)
        return [(p['out_channels'], *spatial)]
    if name in CONVOLUTIONS:
        output = p.get('out_channels', c*p.get('depth_multiplier', 1))
        if name == 'depthwise_separable_conv2d':
            first = _conv_geometry(shape, c*p['depth_multiplier'], p['kernel_size'], p['stride'], p['padding'], p['dilation'])
            return [first, (output, *first[1:])]
        if name == 'spatial_separable_conv2d':
            kh, kw = p['kernel_size']
            sh, sw = p['stride']
            ph, pw = p['padding']
            dh, dw = p['dilation']
            first = _conv_geometry(shape, p['intermediate_channels'], (kh,1), (sh,1), (ph,0), (dh,1))
            return [first, _conv_geometry(first, output, (1,kw), (1,sw), (0,pw), (1,dw))]
        final = _conv_geometry(shape, output, p['kernel_size'], p['stride'], p['padding'], p['dilation'])
        if name == 'factorized_conv2d':
            return [(p['intermediate_channels'], h, w), final]
        return [final]
    if name.startswith('adaptive_'):
        return [(c, *p['output_size'])]
    if name == 'global_avg_pool2d':
        return [(c,1,1)]
    if name in ('max_pool2d', 'avg_pool2d'):
        stride = p['stride'] or p['kernel_size']
        dilation = p.get('dilation', (1,1))
        spatial = []
        for v,k,s,pad,d in zip((h,w),p['kernel_size'],stride,p['padding'],dilation):
            size = (v+2*pad-d*(k-1)-1+(s-1 if p['ceil_mode'] else 0))//s+1
            if p['ceil_mode'] and (size-1)*s >= v+pad:
                size -= 1
            spatial.append(size)
        return [(c, *spatial)]
    return [tuple(shape)]


def parameter_count(name, p):
    """Exact trainable parameter count for the Stage 13.5.2 operators."""
    if name in CONVOLUTIONS:
        a = p['in_channels']
        b = p.get('out_channels', a*p.get('depth_multiplier', 1))
        kh, kw = p.get('kernel_size', (1,1))
        bias = int(p['bias'])
        if name == 'depthwise_separable_conv2d':
            middle = a*p['depth_multiplier']
            return middle*(kh*kw+bias)+(middle+bias)*b
        if name == 'spatial_separable_conv2d':
            middle = p['intermediate_channels']
            return (a*kh+bias)*middle+(middle*kw+bias)*b
        if name == 'factorized_conv2d':
            middle = p['intermediate_channels']
            return (a+bias)*middle+(middle*kh*kw+bias)*b
        groups = a if name == 'depthwise_conv2d' else p.get('groups', 1)
        return a*b*kh*kw//groups+b*bias
    if name in ('batch_norm', 'instance_norm', 'group_norm', 'channel_layer_norm'):
        channels = p.get('num_features', p.get('num_channels', p.get('channels')))
        return 2*channels if p.get('affine', True) else 0
    if name == 'prelu':
        return p['num_parameters']
    return 0


def statistics_requirements(name, p, shape):
    """Describe runtime conditions without assuming a concrete symbolic batch."""
    c, h, w = shape
    if name == 'batch_norm':
        mode = 'train and eval' if not p.get('track_running_stats', True) else 'train'
        return [f'{mode}: N*H*W > 1 (H*W={h*w})']
    if name == 'instance_norm':
        mode = 'train' if p['track_running_stats'] else 'train and eval'
        return [f'{mode}: H*W > 1 (H*W={h*w})']
    if name == 'group_norm':
        return [f'train and eval: N*(C/num_groups)*H*W > 1 (per-sample group elements={c//p["num_groups"]*h*w})']
    if name in ('conv_norm_act', 'residual') and p.get('norm') is not None:
        return ['Internal normalization retains native training/statistics requirements.']
    if name == 'coordinate':
        return ['Internal descriptor BatchNorm requires N*(H+W) > 1 while training.']
    return []
