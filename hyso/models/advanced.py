"""Original HYSo generic blocks built from public torch operators, root MIT.

No ready-model implementation or development-only model module is imported.
Plans are shared by construction, geometry and exact parameter accounting.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from .components import _parameters, _input

MAX_TOKENS = 1024
MAX_ATTENTION_ELEMENTS = 4_194_304
RESIDUALS = frozenset(('bottleneck_residual', 'preactivation_residual', 'wide_residual_block',
                       'grouped_bottleneck', 'inverted_bottleneck', 'gated_residual'))
SELF_ATTENTION = frozenset(('spatial_self_attention', 'multihead_spatial_attention'))
ATTENTION = SELF_ATTENTION | {'spatial_attention', 'channel_attention'}
REGULARIZATION = frozenset(('dropout', 'dropout2d', 'drop_path', 'gaussian_noise'))
NEW_NAMES = RESIDUALS | ATTENTION | REGULARIZATION


class DropPath(nn.Module):
    """Sample-wise tensor masking, x*Bernoulli(1-p)/(1-p) in train, identity in eval."""
    def __init__(self, p=0.):
        p = _parameters('drop_path', p=p)['p']
        super().__init__()
        self.p = p

    def forward(self, x):
        _input(x, x.shape[1] if isinstance(x, torch.Tensor) and x.ndim == 4 else None)
        if not self.training or self.p == 0:
            return x
        mask = torch.rand((x.shape[0],1,1,1), device=x.device) >= self.p
        return x*mask.to(x.dtype)/(1-self.p)


class GaussianNoise(nn.Module):
    """Add independent zero-mean Gaussian noise in train, identity in eval."""
    def __init__(self, std=.1):
        std = _parameters('gaussian_noise', std=std)['std']
        super().__init__()
        self.std = std

    def forward(self, x, *, generator=None):
        _input(x, x.shape[1] if isinstance(x, torch.Tensor) and x.ndim == 4 else None)
        if not self.training or self.std == 0:
            return x
        return x + torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)*self.std


class SpatialAttention(nn.Module):
    """Channel mean/max descriptors -> odd spatial convolution -> sigmoid gate."""
    def __init__(self, channels, kernel_size=7):
        p = _parameters('spatial_attention', channels=channels, kernel_size=kernel_size)
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(2,1,p['kernel_size'],padding=p['kernel_size']//2,bias=False)

    def forward(self, x):
        _input(x, self.channels)
        descriptor = torch.cat((x.mean(1,keepdim=True),x.amax(1,keepdim=True)),dim=1)
        return x*torch.sigmoid(self.conv(descriptor))


class ChannelAttention(nn.Module):
    """Shared 1x1 MLP of spatial mean/max descriptors, summed sigmoid channel gate."""
    def __init__(self, channels, reduction=8):
        p = _parameters('channel_attention', channels=channels, reduction=reduction)
        super().__init__()
        self.channels = channels
        hidden = max(1, channels//p['reduction'])
        self.mlp = nn.Sequential(nn.Conv2d(channels,hidden,1),nn.ReLU(),nn.Conv2d(hidden,channels,1))

    def forward(self, x):
        _input(x, self.channels)
        gate = self.mlp(x.mean((2,3),keepdim=True))+self.mlp(x.amax((2,3),keepdim=True))
        return x*torch.sigmoid(gate)


def attention_budget(name, p, shape, batch_size, error):
    tokens = shape[1]*shape[2]
    heads = p.get('num_heads',1)
    matrix = batch_size*heads*tokens*tokens
    if tokens > MAX_TOKENS or matrix > MAX_ATTENTION_ELEMENTS:
        raise error(f'{name}: {tokens} tokens produce {matrix} attention matrix elements (N*heads*(H*W)^2)',
                    code='ATTENTION_RESOURCE_LIMIT', expected=dict(tokens=MAX_TOKENS,matrix_elements=MAX_ATTENTION_ELEMENTS),
                    actual=dict(tokens=tokens,matrix_elements=matrix,batch_size=batch_size,heads=heads),
                    suggestion='H/W veya batch/head sayısını azaltın. Önce pooling uygulayın veya spatial_attention/channel_attention/eca seçin.')
    return dict(tokens=tokens,num_heads=heads,batch_size=batch_size,matrix_elements=matrix,
                token_limit=MAX_TOKENS,matrix_limit=MAX_ATTENTION_ELEMENTS,memory_guarantee=False)


class SpatialSelfAttention(nn.Module):
    """NCHW -> H*W tokens -> scaled dot-product attention -> projected NCHW."""
    registry_name = 'spatial_self_attention'

    def __init__(self, **params):
        p = _parameters(self.registry_name, **params)
        super().__init__()
        self.channels = p['channels']
        self.embedding_dim = p['embedding_dim']
        self.heads = p.get('num_heads',1)
        self.config = p
        self.qkv = nn.Conv2d(self.channels,3*self.embedding_dim,1)
        self.projection = nn.Conv2d(self.embedding_dim,self.channels,1)
        self.dropout = nn.Dropout(p['dropout'])

    def forward(self, x):
        from .registry import ModelConfigError
        _input(x, self.channels)
        n,c,h,w = x.shape
        attention_budget(self.registry_name,self.config,(c,h,w),n,ModelConfigError)
        q,k,v = self.qkv(x).reshape(n,3,self.heads,self.embedding_dim//self.heads,h*w).unbind(1)
        q,k,v = (t.transpose(-2,-1) for t in (q,k,v))
        scores = (q @ k.transpose(-2,-1))/math.sqrt(self.embedding_dim//self.heads)
        weights = self.dropout(scores.softmax(dim=-1))
        attended = (weights @ v).transpose(-2,-1).reshape(n,self.embedding_dim,h,w)
        return self.projection(attended)


class MultiHeadSpatialAttention(SpatialSelfAttention):
    """Independent spatial attention heads with an explicit embedding projection."""
    registry_name = 'multihead_spatial_attention'


def residual_plan(name, p):
    """Ordered main-branch operators with explicit convolution geometry."""
    a,b = p['in_channels'],p['out_channels']
    middle = a*p['expansion_ratio'] if name=='inverted_bottleneck' else p['intermediate_channels']
    kernel, stride, dilation = p['kernel_size'],p['stride'],p['dilation']
    padding = [d*(kernel-1)//2 for d in dilation]
    def conv(cin,cout,k=1,s=(1,1),d=(1,1),pad=(0,0),groups=1):
        return ('conv',dict(in_channels=cin,out_channels=cout,kernel_size=[k,k],stride=s,
                            dilation=d,padding=pad,groups=groups,bias=False))
    def norm(c): return ('norm',c)
    def act(): return ('activation',None)
    spatial = dict(k=kernel,s=stride,d=dilation,pad=padding)
    if name == 'preactivation_residual':
        return [norm(a),act(),conv(a,middle,**spatial),norm(middle),act(),
                conv(middle,b,k=kernel,d=dilation,pad=padding)]
    if name in ('wide_residual_block','gated_residual'):
        return [conv(a,middle,**spatial),norm(middle),act(),
                conv(middle,b,k=kernel,d=dilation,pad=padding),norm(b)]
    groups = middle if name=='inverted_bottleneck' else p.get('groups',1)
    return [conv(a,middle),norm(middle),act(),conv(middle,middle,groups=groups,**spatial),
            norm(middle),act(),conv(middle,b),norm(b)]


def projection_required(p):
    return p['in_channels']!=p['out_channels'] or list(p['stride'])!=[1,1]


def _normalization(name, channels):
    if name == 'batch': return nn.BatchNorm2d(channels)
    if name == 'group': return nn.GroupNorm(math.gcd(8,channels),channels)
    return nn.Identity()


class AdvancedResidual(nn.Module):
    """Explicit main branch plus raw-input identity or 1x1 projection shortcut."""
    registry_name = 'bottleneck_residual'

    def __init__(self, **params):
        p = _parameters(self.registry_name, **params)
        super().__init__()
        self.in_channels = p['in_channels']
        layers = []
        for kind, settings in residual_plan(self.registry_name,p):
            if kind == 'conv': layers.append(nn.Conv2d(**settings))
            elif kind == 'norm': layers.append(_normalization(p['normalization'],settings))
            else: layers.append({'relu':nn.ReLU,'silu':nn.SiLU,'gelu':nn.GELU}[p['activation']]())
        self.main = nn.Sequential(*layers)
        self.shortcut = (nn.Conv2d(p['in_channels'],p['out_channels'],1,stride=p['stride'],bias=False)
                         if projection_required(p) else nn.Identity())
        from .registry import ModelRegistry
        self.attention = (nn.Identity() if p['attention']=='none' else
                          ModelRegistry().create(p['attention'],channels=p['out_channels']))
        self.dropout = nn.Dropout(p['dropout'])
        self.drop_path = DropPath(p['drop_path'])
        gate = p.get('gate_init',1.)
        if p.get('learnable_gate',False): self.gate = nn.Parameter(torch.tensor(float(gate)))
        else: self.gate = float(gate)

    def forward(self, x):
        from .registry import ModelConfigError
        _input(x,self.in_channels)
        try:
            branch = self.drop_path(self.dropout(self.attention(self.main(x))))
        except ValueError as exc:
            if isinstance(exc,ModelConfigError): raise
            raise ModelConfigError(str(exc),code='STATISTICS_REQUIREMENT',
                                   expected='valid internal normalization statistics',actual=list(x.shape)) from exc
        skip = self.shortcut(x)
        if branch.shape != skip.shape:
            raise ModelConfigError('residual main and shortcut shapes differ',code='CHANNEL_MISMATCH',
                                   expected=list(skip.shape),actual=list(branch.shape))
        return skip + self.gate*branch


class PreActivationResidual(AdvancedResidual):
    registry_name = 'preactivation_residual'


class WideResidual(AdvancedResidual):
    registry_name = 'wide_residual_block'


class GroupedBottleneck(AdvancedResidual):
    registry_name = 'grouped_bottleneck'


class InvertedBottleneck(AdvancedResidual):
    registry_name = 'inverted_bottleneck'


class GatedResidual(AdvancedResidual):
    registry_name = 'gated_residual'


def specifications(p):
    rows = []
    constructors = (AdvancedResidual,PreActivationResidual,WideResidual,GroupedBottleneck,InvertedBottleneck,GatedResidual)
    for ctor in constructors:
        width = p('expansion_ratio',default=2) if ctor is InvertedBottleneck else p('intermediate_channels')
        params = (p('in_channels'),p('out_channels'),width,p('kernel_size','odd_int',3),
                  p('stride','pair',1),p('dilation','pair',1),
                  p('normalization','choice','batch',('batch','group',None)),
                  p('activation','choice','silu',('relu','silu','gelu')),
                  p('attention','choice','none',('none','eca','spatial_attention','channel_attention')),
                  p('dropout','probability',0.),p('drop_path','probability',0.))
        if ctor is GroupedBottleneck: params += (p('groups',default=1),)
        if ctor is GatedResidual: params += (p('gate_init','probability_closed',1.),p('learnable_gate','bool',True))
        rows.append((ctor.registry_name,'residual','HYSo generic residual main branch with explicit projection policy.',ctor,params))
    rows.extend((
        ('spatial_attention','attention',SpatialAttention.__doc__,SpatialAttention,(p('channels'),p('kernel_size','odd_int',7))),
        ('channel_attention','attention',ChannelAttention.__doc__,ChannelAttention,(p('channels'),p('reduction',default=8))),
        ('spatial_self_attention','attention',SpatialSelfAttention.__doc__,SpatialSelfAttention,
         (p('channels'),p('embedding_dim','optional_positive_int',None),p('dropout','probability',0.))),
        ('multihead_spatial_attention','attention',MultiHeadSpatialAttention.__doc__,MultiHeadSpatialAttention,
         (p('channels'),p('embedding_dim','optional_positive_int',None),p('num_heads',default=4),p('dropout','probability',0.))),
        ('dropout','regularization','Elementwise train masking, identity in eval.',nn.Dropout,(p('p','probability_closed',.5),)),
        ('dropout2d','regularization','Channelwise train masking, identity in eval.',nn.Dropout2d,(p('p','probability_closed',.5),)),
        ('drop_path','regularization',DropPath.__doc__,DropPath,(p('p','probability',0.),)),
        ('gaussian_noise','regularization',GaussianNoise.__doc__,GaussianNoise,(p('std','nonnegative_float',.1),)),
    ))
    return tuple(rows)


def validate(name,p,error):
    if name in SELF_ATTENTION:
        if p['embedding_dim'] is None:
            p['embedding_dim'] = p['channels']
        if 3*p['embedding_dim'] > 8192:
            raise error('QKV intermediate channels exceed 8192',code='INTERMEDIATE_SHAPE_ERROR',
                        expected='embedding_dim <= 2730',actual=p['embedding_dim'])
    if name in RESIDUALS:
        middle = p['in_channels']*p['expansion_ratio'] if name=='inverted_bottleneck' else p['intermediate_channels']
        if middle > 8192:
            raise error('expanded intermediate_channels exceed 8192',code='INTERMEDIATE_SHAPE_ERROR',expected=8192,actual=middle)
        if middle % p.get('groups',1):
            divisors = [v for v in range(1,middle+1) if middle%v==0]
            raise error('groups must divide intermediate_channels',code='GROUP_DIVISIBILITY_ERROR',
                        expected=divisors,actual=p['groups'],suggestion=f'Geçerli groups değerleri {divisors}.')
    if name=='multihead_spatial_attention' and p['embedding_dim']%p['num_heads']:
        divisors = [v for v in range(1,p['embedding_dim']+1) if p['embedding_dim']%v==0]
        raise error('embedding_dim must be divisible by num_heads',code='GROUP_DIVISIBILITY_ERROR',
                    expected=divisors,actual=p['num_heads'],suggestion=f'embedding_dim için geçerli num_heads değerleri {divisors}.')
    if name=='gaussian_noise' and p['std']>1e6:
        raise error('Gaussian std exceeds the supported finite-noise range',expected='std <= 1e6',actual=p['std'])


def describe(name,p,shape,batch_size,error):
    """Allocation-free geometry, parameter and attention resource contract."""
    from .components import stages as primitive_stages, parameter_count as primitive_count
    c,h,w = shape
    result = dict(stages=[],parameter_count=0,attention=None,projection=False,residual_active=False,requirements=[])
    if name in RESIDUALS:
        current = tuple(shape)
        for kind,settings in residual_plan(name,p):
            if kind=='conv':
                current = primitive_stages('conv2d',settings,current)[0]
                result['parameter_count'] += primitive_count('conv2d',settings)
            elif kind=='norm' and p['normalization'] is not None:
                result['parameter_count'] += 2*settings
                result['requirements'].append(f'Internal {p["normalization"]} normalization at CHW={current} requires native train/eval statistics.')
            result['stages'].append(current)
        projection = projection_required(p)
        skip = (p['out_channels'],(h+p['stride'][0]-1)//p['stride'][0],(w+p['stride'][1]-1)//p['stride'][1])
        if current != skip:
            raise error('main and projection geometry differ',code='CHANNEL_MISMATCH',expected=list(skip),actual=list(current))
        result['projection'] = projection
        if projection:
            result['parameter_count'] += c*p['out_channels']
            result['stages'].append(skip)
        if p['attention']!='none':
            from .registry import ModelRegistry
            ap = ModelRegistry().validate(p['attention'],dict(channels=current[0]))
            nested = describe(p['attention'],ap,current,batch_size,error) if p['attention'] in ATTENTION else None
            result['parameter_count'] += nested['parameter_count'] if nested else ap['kernel_size']
            result['stages'].extend(nested['stages'] if nested else [current])
        result['parameter_count'] += int(p.get('learnable_gate',False))
        result['residual_active'] = p.get('learnable_gate',False) or p.get('gate_init',1.)!=0
        result['stages'].append(current)
    elif name in SELF_ATTENTION:
        result['attention'] = attention_budget(name,p,shape,batch_size,error)
        e = p['embedding_dim']
        result['parameter_count'] = 4*c*e+3*e+c
        result['stages'] = [(3*e,h,w),(e,h,w),(c,h,w)]
    elif name=='spatial_attention':
        result['parameter_count'] = 2*p['kernel_size']**2
        result['stages'] = [(2,h,w),(1,h,w),(c,h,w)]
    elif name=='channel_attention':
        hidden = max(1,c//p['reduction'])
        result['parameter_count'] = 2*c*hidden+hidden+c
        result['stages'] = [(c,1,1),(hidden,1,1),(c,1,1),(c,h,w)]
    else:
        result['stages'] = [tuple(shape)]
    for stage in result['stages']:
        if min(stage)<1 or max(stage)>8192:
            raise error('advanced intermediate shape outside 1..8192',code='INTERMEDIATE_SHAPE_ERROR',
                        expected='CHW in 1..8192',actual=list(stage))
    return result


def metadata(name):
    return dict(description_tr={
        'bottleneck_residual':'Kanal daraltma, spatial filtre ve kanal çıkışı ile residual toplama.',
        'preactivation_residual':'Her convolution öncesinde normalizasyon ve aktivasyon.',
        'wide_residual_block':'Ara kanal genişliği açıkça seçilen iki convolution residual bloğu.',
        'grouped_bottleneck':'Ara kanalları gruplara ayıran bottleneck.',
        'inverted_bottleneck':'Kanal genişletme, depthwise filtre ve projection.',
        'gated_residual':'Residual katkısını sabit veya öğrenilen skaler ile ölçekler.',
        'spatial_attention':'Kanal özetlerinden konumsal sigmoid kapısı üretir.',
        'channel_attention':'Spatial özetlerden kanal sigmoid kapısı üretir.',
        'spatial_self_attention':'Spatial konumları token olarak tek attention başlığıyla işler.',
        'multihead_spatial_attention':'Spatial tokenları bölünebilir embedding ile çok başlıklı işler.',
        'dropout':'Eğitimde elemanları rastgele sıfırlar.',
        'dropout2d':'Eğitimde kanalları rastgele sıfırlar.',
        'drop_path':'Eğitimde örnek başına tensor katkısını rastgele sıfırlar.',
        'gaussian_noise':'Eğitimde bağımsız sıfır ortalamalı Gaussian gürültüsü ekler.',
    }[name], train_eval='Dropout/noise are disabled in eval. BatchNorm uses running statistics in eval.',
    dependencies=(['embedding_dim % num_heads == 0','tokens <= 1024','N*heads*tokens^2 <= 4194304'] if name in SELF_ATTENTION else
                  ['raw input shortcut, 1x1 projection when channels or stride change','odd kernel with dilation-aware symmetric padding'] if name in RESIDUALS else []),
    memory_notes='Attention matrix limits are mandatory at preflight and runtime. Activation estimates do not guarantee peak memory.',
    corrections='Match incoming channels, use positive dimensions and supported probability ranges. Use Registry parameter constraints.')
