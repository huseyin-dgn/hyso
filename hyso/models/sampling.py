"""Original HYSo sampling, restricted tensor utilities and prediction heads.

Public torch operators, root MIT license. Geometry never allocates tensors.
"""
from __future__ import annotations
import typing as _typing
import math
import torch
from torch import nn
from torch.nn import functional as F
from .components import _parameters, _input

SAMPLING = frozenset(('resize','pixel_shuffle','pixel_unshuffle'))
UTILITIES = frozenset(('flatten','permute','reshape'))
HEADS = frozenset(('classification_head','regression_head','embedding_head'))
NEW_NAMES = SAMPLING | UTILITIES | HEADS


def scalar(kind, value):
    if kind == 'scale_pair':
        values = [value,value] if type(value) in (int,float) else value
        valid = value is None or type(values) is list and len(values)==2 and all(
            type(v) in (int,float) and 0 < v <= 8192 and math.isfinite(v) for v in values)
        return valid, None if value is None else values
    if kind == 'optional_bool':
        return value is None or type(value) is bool, value
    if kind == 'chw':
        return type(value) is list and len(value)==3 and all(type(v) is int and 1<=v<=8192 for v in value),value
    if kind == 'axes':
        return type(value) is list and value in ([0,1,2,3],[0,1,3,2]),value
    return False,value


def validate(name,p,error):
    if name=='resize':
        if (p['output_size'] is None)==(p['scale_factor'] is None):
            raise error('resize requires exactly one of output_size or scale_factor',
                        expected='one explicit spatial target',actual=p)
        if p['mode'] in ('nearest','area') and p['align_corners'] is not None:
            raise error('align_corners must be None for nearest/area',expected=None,actual=p['align_corners'])


def shape(name,p,source,error):
    if len(source)!=3:
        raise error(f'{name} requires rank-4 NCHW input',code='INVALID_BLOCK_ORDER',
                    expected=['N','C','H','W'],actual=['N',*source])
    c,h,w=source
    if name=='resize':
        spatial=p['output_size'] or [math.floor(v*s) for v,s in zip((h,w),p['scale_factor'])]
        output=(c,*spatial)
    elif name=='pixel_shuffle':
        factor=p['upscale_factor']
        if c%(factor*factor):
            raise error('PixelShuffle C must be divisible by upscale_factor squared',code='CHANNEL_MISMATCH',
                        expected=f'C divisible by {factor*factor}',actual=c)
        output=(c//(factor*factor),h*factor,w*factor)
    elif name=='pixel_unshuffle':
        factor=p['downscale_factor']
        if h%factor or w%factor:
            raise error('PixelUnshuffle H and W must be divisible by downscale_factor',code='SPATIAL_DIMENSION_ERROR',
                        expected=f'H/W divisible by {factor}',actual=[h,w])
        output=(c*factor*factor,h//factor,w//factor)
    elif name=='flatten':
        output=(c*h*w,)
    elif name=='permute':
        output=tuple(source[index-1] for index in p['dims'][1:])
    else:
        output=tuple(p['shape'])
        if math.prod(output)!=c*h*w:
            raise error('reshape must preserve per-sample element count',code='INVALID_INPUT_SHAPE',
                        expected=c*h*w,actual=math.prod(output))
    if min(output)<1 or max(output)>8192:
        raise error(f'{name} produces a dimension outside 1..8192',code='SPATIAL_DIMENSION_ERROR',
                    expected='output dimensions in 1..8192',actual=list(output))
    return output


class Resize(nn.Module):
    """NCHW interpolation with exactly one target and explicit floor scale geometry.

    Args:
        output_size (int | tuple[int, int] | list[int] | None): Target H/W grid; integer means a square.
            Resize requires exactly one of output_size and scale_factor. None or integer/HW pair, each
            in 1..8192. Default: None.
        scale_factor (_typing.Any): Positive H/W resize multipliers, each <=8192. Exactly one of this
            and output_size must be provided. None or positive scalar/HW factors, each <=8192. Default:
            None.
        mode (_typing.Literal['nearest', 'bilinear', 'bicubic', 'area']): Resize interpolation method;
            nearest/area require align_corners=None. Choices: 'nearest', 'bilinear', 'bicubic', 'area'.
            Default: 'nearest'.
        align_corners (_typing.Any): Align endpoint pixel centers for bilinear/bicubic resizing; must be
            None for nearest/area. Boolean or None. Default: None.

    Returns:
        A Resize module with the declared geometry.

    Raises:
        ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

    Notes:
        Floating nonempty NCHW.
        Floating NCHW with explicit resize or pixel rearrangement geometry.
        resize requires exactly one target
        pixel factors require divisibility
        reshape preserves sample element count
        heads are terminal, in_features is inferred by Builder

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        Resize()

    HYSo Guide: components.sampling
    Offline: hyso docs show components.sampling --lang en
    Open: hyso docs open components.sampling --lang en
    """
    def __init__(self,output_size: 'int | tuple[int, int] | list[int] | None'=None,scale_factor: '_typing.Any'=None,mode: _typing.Literal['nearest', 'bilinear', 'bicubic', 'area']='nearest',align_corners: '_typing.Any'=None) -> 'None':
        """Initialize Resize with the supplied settings.

        Args:
            output_size (int | tuple[int, int] | list[int] | None): Target H/W grid; integer means a square.
                Resize requires exactly one of output_size and scale_factor. None or integer/HW pair, each
                in 1..8192. Default: None.
            scale_factor (_typing.Any): Positive H/W resize multipliers, each <=8192. Exactly one of this
                and output_size must be provided. None or positive scalar/HW factors, each <=8192. Default:
                None.
            mode (_typing.Literal['nearest', 'bilinear', 'bicubic', 'area']): Resize interpolation method;
                nearest/area require align_corners=None. Choices: 'nearest', 'bilinear', 'bicubic', 'area'.
                Default: 'nearest'.
            align_corners (_typing.Any): Align endpoint pixel centers for bilinear/bicubic resizing; must be
                None for nearest/area. Boolean or None. Default: None.

        Returns:
            None; initializes the module.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW.
            Floating NCHW with explicit resize or pixel rearrangement geometry.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            Resize()

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        p=_parameters('resize',output_size=output_size,scale_factor=scale_factor,mode=mode,align_corners=align_corners)
        super().__init__()
        self.config=p

    def forward(self,x: '_typing.Any') -> '_typing.Any':
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

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        from .registry import ModelConfigError
        _input(x,x.shape[1] if isinstance(x,torch.Tensor) and x.ndim==4 else None)
        shape('resize',self.config,tuple(x.shape[1:]),ModelConfigError)
        p=self.config
        return F.interpolate(x,size=p['output_size'],scale_factor=p['scale_factor'],
                             mode=p['mode'],align_corners=p['align_corners'])


class Permute(nn.Module):
    """Keep N/C axes fixed and optionally exchange spatial H/W axes.

    Args:
        dims (_typing.Sequence[int]): Permutation of NCHW axes; only (0,1,2,3) and (0,1,3,2) are
            accepted, preserving batch/channel axes. Only identity or spatial-axis swap is supported.
            Default: (0, 1, 3, 2).

    Returns:
        A Permute module with the declared geometry.

    Raises:
        ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

    Notes:
        Floating NCHW with preserved batch axis.
        NCHW or terminal NF for flatten, see shape_contract.
        resize requires exactly one target
        pixel factors require divisibility
        reshape preserves sample element count
        heads are terminal, in_features is inferred by Builder

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        Permute()

    HYSo Guide: components.sampling
    Offline: hyso docs show components.sampling --lang en
    Open: hyso docs open components.sampling --lang en
    """
    def __init__(self,dims: '_typing.Sequence[int]'=(0,1,3,2)) -> 'None':
        """Initialize Permute with the supplied settings.

        Args:
            dims (_typing.Sequence[int]): Permutation of NCHW axes; only (0,1,2,3) and (0,1,3,2) are
                accepted, preserving batch/channel axes. Only identity or spatial-axis swap is supported.
                Default: (0, 1, 3, 2).

        Returns:
            None; initializes the module.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW with preserved batch axis.
            NCHW or terminal NF for flatten, see shape_contract.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            Permute()

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        p=_parameters('permute',dims=dims)
        super().__init__()
        self.dims=tuple(p['dims'])

    def forward(self,x: '_typing.Any') -> '_typing.Any':
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

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        _input(x,x.shape[1] if isinstance(x,torch.Tensor) and x.ndim==4 else None)
        return x.permute(self.dims)


class Reshape(nn.Module):
    """Reinterpret each sample as explicit positive CHW, preserving batch and elements.

    Args:
        shape (_typing.Any): Target per-sample CHW dimensions; reshape must preserve the number of
            sample elements and every dimension is 1..8192. Three integers in 1..8192. Required.

    Returns:
        A Reshape module with the declared geometry.

    Raises:
        ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

    Notes:
        Floating NCHW with preserved batch axis.
        NCHW or terminal NF for flatten, see shape_contract.
        resize requires exactly one target
        pixel factors require divisibility
        reshape preserves sample element count
        heads are terminal, in_features is inferred by Builder

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        Reshape(shape)

    HYSo Guide: components.sampling
    Offline: hyso docs show components.sampling --lang en
    Open: hyso docs open components.sampling --lang en
    """
    def __init__(self,shape: '_typing.Any') -> 'None':
        """Initialize Reshape with the supplied settings.

        Args:
            shape (_typing.Any): Target per-sample CHW dimensions; reshape must preserve the number of
                sample elements and every dimension is 1..8192. Three integers in 1..8192. Required.

        Returns:
            None; initializes the module.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW with preserved batch axis.
            NCHW or terminal NF for flatten, see shape_contract.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            Reshape(shape)

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        p=_parameters('reshape',shape=shape)
        super().__init__()
        self.shape=tuple(p['shape'])

    def forward(self,x: '_typing.Any') -> '_typing.Any':
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

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        from .registry import ModelConfigError
        _input(x,x.shape[1] if isinstance(x,torch.Tensor) and x.ndim==4 else None)
        shape('reshape',dict(shape=self.shape),tuple(x.shape[1:]),ModelConfigError)
        return x.reshape(x.shape[0],*self.shape)


class Flatten(nn.Flatten):
    """Flatten CHW to one feature axis, preserving symbolic N.

    Args:

    Returns:
        A Flatten module with the declared geometry.

    Raises:
        ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

    Notes:
        Floating NCHW with preserved batch axis.
        NCHW or terminal NF for flatten, see shape_contract.
        resize requires exactly one target
        pixel factors require divisibility
        reshape preserves sample element count
        heads are terminal, in_features is inferred by Builder

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        Flatten()

    HYSo Guide: components.sampling
    Offline: hyso docs show components.sampling --lang en
    Open: hyso docs open components.sampling --lang en
    """
    def __init__(self) -> 'None':
        """Initialize Flatten with the supplied settings.

        Args:

        Returns:
            None; initializes the module.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW with preserved batch axis.
            NCHW or terminal NF for flatten, see shape_contract.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            Flatten()

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        super().__init__(start_dim=1)

    def forward(self,x: '_typing.Any') -> '_typing.Any':
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

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        _input(x,x.shape[1] if isinstance(x,torch.Tensor) and x.ndim==4 else None)
        return super().forward(x)


def head_geometry(name,p,source,error):
    if len(source)==3:
        features=math.prod(source) if p['pooling']=='none' else source[0]
    elif len(source)==1 and p['pooling']=='none':
        features=source[0]
    else:
        raise error('head pooling requires NCHW, vector features require pooling=none',
                    code='INVALID_OUTPUT_HEAD',expected='NCHW or vector with pooling=none',actual=['N',*source])
    if p['in_features']!=features:
        raise error('prediction head in_features mismatch',code='INVALID_OUTPUT_HEAD',expected=features,actual=p['in_features'])
    outputs=p.get('num_classes',p.get('output_features',p.get('embedding_dim')))
    return dict(in_features=features,output_features=outputs,parameter_count=(features+1)*outputs)


class PredictionHead(nn.Module):
    """Project pooled features to the declared output width; training task support is separate.

    Args:
        in_features (int): Linear-head input width after pooling/flattening; must match the inferred
            feature width. Integer in 1..8192; bool is rejected. Required.
        pooling (_typing.Literal['avg', 'max', 'none']): avg/max globally pool NCHW before flattening;
            none flattens NCHW or consumes flat NF. NF requires none. Choices: 'avg', 'max', 'none'.
            Default: 'avg'.
        num_classes (int): Number of output class logits. Outputs are unnormalized scores, not
            probabilities. Integer in 1..8192; bool is rejected. Required.
        dropout (float): Training-only activation/head dropout probability; disabled in eval. Finite
            number in [0,1). Default: 0.0.

    Returns:
        A PredictionHead module with the declared geometry.

    Raises:
        ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

    Notes:
        Floating NCHW or NF with pooling=none.
        Floating NF prediction, head type controls output semantics.
        resize requires exactly one target
        pixel factors require divisibility
        reshape preserves sample element count
        heads are terminal, in_features is inferred by Builder

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        PredictionHead(in_features=4, num_classes=2)

    HYSo Guide: components.heads
    Offline: hyso docs show components.heads --lang en
    Open: hyso docs open components.heads --lang en
    """
    registry_name='classification_head'

    def __init__(self, *, in_features: int, pooling: _typing.Literal['avg', 'max', 'none']='avg', num_classes: int, dropout: float=0.0) -> None:
        """Construct PredictionHead through its existing registry validator.

        Args:
            in_features (int): Linear-head input width after pooling/flattening; must match the inferred
                feature width. Integer in 1..8192; bool is rejected. Required.
            pooling (_typing.Literal['avg', 'max', 'none']): avg/max globally pool NCHW before flattening;
                none flattens NCHW or consumes flat NF. NF requires none. Choices: 'avg', 'max', 'none'.
                Default: 'avg'.
            num_classes (int): Number of output class logits. Outputs are unnormalized scores, not
                probabilities. Integer in 1..8192; bool is rejected. Required.
            dropout (float): Training-only activation/head dropout probability; disabled in eval. Finite
                number in [0,1). Default: 0.0.

        Returns:
            None; initializes the module.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW or NF with pooling=none.
            Floating NF prediction, head type controls output semantics.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder

        Examples:
            model = PredictionHead(in_features=4, num_classes=2)

        HYSo Guide: components.heads
        Offline: hyso docs show components.heads --lang en
        Open: hyso docs open components.heads --lang en
        """
        self._initialize(dict(in_features=in_features, pooling=pooling, num_classes=num_classes, dropout=dropout))
    def _initialize(self, params: dict[str, _typing.Any]) -> None:
        """Initialize PredictionHead with the supplied settings.

        Args:
            **params (_typing.Any): Component keyword parameters; registered defaults apply to omitted optional keys.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            PredictionHead()

        HYSo Guide: components.heads
        Offline: hyso docs show components.heads --lang en
        Open: hyso docs open components.heads --lang en
        """
        p=_parameters(self.registry_name,**params)
        super().__init__()
        self.config=p
        outputs=p.get('num_classes',p.get('output_features',p.get('embedding_dim')))
        self.pool={'avg':nn.AdaptiveAvgPool2d,'max':nn.AdaptiveMaxPool2d}.get(p['pooling'],lambda size:nn.Identity())(1)
        self.dropout=nn.Dropout(p.get('dropout',0.))
        self.linear=nn.Linear(p['in_features'],outputs)

    def forward(self,x: '_typing.Any') -> '_typing.Any':
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

        HYSo Guide: components.heads
        Offline: hyso docs show components.heads --lang en
        Open: hyso docs open components.heads --lang en
        """
        from .registry import ModelConfigError
        if not isinstance(x,torch.Tensor) or x.ndim not in (2,4) or min(x.shape)<1 or not x.is_floating_point():
            raise ModelConfigError('prediction head requires nonempty floating NCHW or NF',code='INVALID_OUTPUT_HEAD',
                                   expected='NCHW or NF',actual=list(x.shape) if isinstance(x,torch.Tensor) else type(x).__name__)
        head_geometry(self.registry_name,self.config,tuple(x.shape[1:]),ModelConfigError)
        output=self.linear(self.dropout(torch.flatten(self.pool(x),1)))
        if self.config.get('normalize',False):
            epsilon=max(self.config['eps'],torch.finfo(output.dtype).tiny)
            output=F.normalize(output,p=2,dim=1,eps=epsilon)
        return output


class RegressionHead(PredictionHead):
    """Produce numeric predictions; the public Trainer remains classification-only.

    Args:
        in_features (int): Linear-head input width after pooling/flattening; must match the inferred
            feature width. Integer in 1..8192; bool is rejected. Required.
        pooling (_typing.Literal['avg', 'max', 'none']): avg/max globally pool NCHW before flattening;
            none flattens NCHW or consumes flat NF. NF requires none. Choices: 'avg', 'max', 'none'.
            Default: 'avg'.
        output_features (int): Width of the unrestricted numeric regression output; no classification
            objective is inferred. Integer in 1..8192; bool is rejected. Required.

    Returns:
        A RegressionHead module with the declared geometry.

    Raises:
        ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

    Notes:
        Floating NCHW or NF with pooling=none.
        Floating NF prediction, head type controls output semantics.
        resize requires exactly one target
        pixel factors require divisibility
        reshape preserves sample element count
        heads are terminal, in_features is inferred by Builder

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        RegressionHead(in_features=4, output_features=4)

    HYSo Guide: components.heads
    Offline: hyso docs show components.heads --lang en
    Open: hyso docs open components.heads --lang en
    """
    registry_name='regression_head'

    def __init__(self, *, in_features: int, pooling: _typing.Literal['avg', 'max', 'none']='avg', output_features: int) -> None:
        """Construct RegressionHead through its existing registry validator.

        Args:
            in_features (int): Linear-head input width after pooling/flattening; must match the inferred
                feature width. Integer in 1..8192; bool is rejected. Required.
            pooling (_typing.Literal['avg', 'max', 'none']): avg/max globally pool NCHW before flattening;
                none flattens NCHW or consumes flat NF. NF requires none. Choices: 'avg', 'max', 'none'.
                Default: 'avg'.
            output_features (int): Width of the unrestricted numeric regression output; no classification
                objective is inferred. Integer in 1..8192; bool is rejected. Required.

        Returns:
            None; initializes the module.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW or NF with pooling=none.
            Floating NF prediction, head type controls output semantics.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder

        Examples:
            model = RegressionHead(in_features=4, output_features=4)

        HYSo Guide: components.heads
        Offline: hyso docs show components.heads --lang en
        Open: hyso docs open components.heads --lang en
        """
        self._initialize(dict(in_features=in_features, pooling=pooling, output_features=output_features))


class EmbeddingHead(PredictionHead):
    """Produce feature embeddings; the public Trainer does not infer an embedding objective.

    Args:
        in_features (int): Linear-head input width after pooling/flattening; must match the inferred
            feature width. Integer in 1..8192; bool is rejected. Required.
        pooling (_typing.Literal['avg', 'max', 'none']): avg/max globally pool NCHW before flattening;
            none flattens NCHW or consumes flat NF. NF requires none. Choices: 'avg', 'max', 'none'.
            Default: 'avg'.
        embedding_dim (int): Output embedding width for a head, or internal attention projection width
            as declared by this component. Integer in 1..8192; bool is rejected. Required.
        normalize (bool): Apply safe per-sample L2 normalization to output embeddings. Boolean only.
            Default: False.
        eps (float): Positive numerical stabilizer used in normalization. Embedding normalization also
            floors it at the dtype tiny value. Finite positive number. Default: 1e-12.

    Returns:
        A EmbeddingHead module with the declared geometry.

    Raises:
        ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

    Notes:
        Floating NCHW or NF with pooling=none.
        Floating NF prediction, head type controls output semantics.
        resize requires exactly one target
        pixel factors require divisibility
        reshape preserves sample element count
        heads are terminal, in_features is inferred by Builder

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        EmbeddingHead(in_features=4, embedding_dim=2)

    HYSo Guide: components.heads
    Offline: hyso docs show components.heads --lang en
    Open: hyso docs open components.heads --lang en
    """
    registry_name='embedding_head'

    def __init__(self, *, in_features: int, pooling: _typing.Literal['avg', 'max', 'none']='avg', embedding_dim: int, normalize: bool=False, eps: float=1e-12) -> None:
        """Construct EmbeddingHead through its existing registry validator.

        Args:
            in_features (int): Linear-head input width after pooling/flattening; must match the inferred
                feature width. Integer in 1..8192; bool is rejected. Required.
            pooling (_typing.Literal['avg', 'max', 'none']): avg/max globally pool NCHW before flattening;
                none flattens NCHW or consumes flat NF. NF requires none. Choices: 'avg', 'max', 'none'.
                Default: 'avg'.
            embedding_dim (int): Output embedding width for a head, or internal attention projection width
                as declared by this component. Integer in 1..8192; bool is rejected. Required.
            normalize (bool): Apply safe per-sample L2 normalization to output embeddings. Boolean only.
                Default: False.
            eps (float): Positive numerical stabilizer used in normalization. Embedding normalization also
                floors it at the dtype tiny value. Finite positive number. Default: 1e-12.

        Returns:
            None; initializes the module.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW or NF with pooling=none.
            Floating NF prediction, head type controls output semantics.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder

        Examples:
            model = EmbeddingHead(in_features=4, embedding_dim=2)

        HYSo Guide: components.heads
        Offline: hyso docs show components.heads --lang en
        Open: hyso docs open components.heads --lang en
        """
        self._initialize(dict(in_features=in_features, pooling=pooling, embedding_dim=embedding_dim, normalize=normalize, eps=eps))


def specifications(p):
    common=(p('in_features'),p('pooling','choice','avg',('avg','max','none')))
    return (
        ('resize','sampling',Resize.__doc__,Resize,(p('output_size','optional_pair',None),p('scale_factor','scale_pair',None),
         p('mode','choice','nearest',('nearest','bilinear','bicubic','area')),p('align_corners','optional_bool',None))),
        ('pixel_shuffle','sampling','C/r² channels, H*r and W*r spatial dimensions.',nn.PixelShuffle,(p('upscale_factor'),)),
        ('pixel_unshuffle','sampling','C*r² channels, H/r and W/r spatial dimensions.',nn.PixelUnshuffle,(p('downscale_factor'),)),
        ('flatten','tensor_utility',Flatten.__doc__,Flatten,()),
        ('permute','tensor_utility',Permute.__doc__,Permute,(p('dims','axes',(0,1,3,2)),)),
        ('reshape','tensor_utility',Reshape.__doc__,Reshape,(p('shape','chw'),)),
        ('classification_head','prediction_head','Global pool or flat features, optional dropout, linear logits.',PredictionHead,
         (*common,p('num_classes'),p('dropout','probability',0.))),
        ('regression_head','prediction_head','Global pool or flat features, unrestricted linear regression output.',RegressionHead,
         (*common,p('output_features'))),
        ('embedding_head','prediction_head','Global pool or flat features, linear embedding, optional safe L2 normalization.',EmbeddingHead,
         (*common,p('embedding_dim'),p('normalize','bool',False),p('eps','positive_float',1e-12))),
    )


def metadata(name):
    return dict(train_eval='Sampling/utilities are deterministic in train/eval. Head dropout is train-only.',
                parameter_dependencies=['resize requires exactly one target','pixel factors require divisibility',
                                        'reshape preserves sample element count','heads are terminal, in_features is inferred by Builder'],
                shape_contract='Batch is preserved. NCHW utilities keep rank 4, flatten produces NF at schema2 terminal feature boundary.',
                cost_contract='Sampling/utilities have zero trainable parameters. Heads have (in_features+1)*output_features parameters.',
                common_errors=['Choose positive bounded dimensions and legal mode/align_corners.',
                               'Keep N/C axes fixed in permute. Use pooling=none after flatten.'])
