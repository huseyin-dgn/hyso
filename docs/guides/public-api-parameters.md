# Public builder parameter reference

HYSo 0.2.0 commercial distribution: 57 registered components, 222 component parameter slots. This reference distinguishes registry keyword defaults from fluent channel inference. Definitions come from the actual closed registry and its validators.

Use `help(ModelBuilder.conv2d)` or `help(GraphModelBuilder.conv2d)` for fluent methods. `add` and `ModelRegistry.create` expose literal overload signatures; generic add requires every registry-required keyword, including in_channels. Fluent conv2d may infer in_channels when None.

ModelBuilder chains accept 53 non-model/non-head components; schema2 terminal heads use classification/regression/embedding factories. GraphModelBuilder supports 56 non-model components and four fusions. The modular_cnn whole-model constructor is available through named/modular/modular_cnn, not as a chain block.

Every registry dimension is bounded to 8192; bool is not an integer dimension. Direct standalone legacy constructors can have broader native bounds: their constructor docstrings are authoritative. Graph inputs contain 1..8 names and concrete per-sample C/CHW shapes; build/preflight validate topology, fan-in/out and geometry. The limits are not memory-use guarantees.

```python
from hyso import ModelBuilder, GraphModelBuilder, ModelRegistry
model = ModelBuilder((1, 8, 8), 2).conv2d(out_channels=4, kernel_size=3, padding=1).relu().build()
graph = (GraphModelBuilder({"image": [1, 8, 8]})
         .conv2d("stem", ["image"], out_channels=4)
         .outputs(features="stem").build())
help(ModelBuilder.conv2d)
help(GraphModelBuilder.add)
print(ModelRegistry().get("conv2d").to_dict())
```

Ordinary convolution per axis: `floor((in+2*padding-dilation*(kernel_size-1)-1)/stride)+1`. Transposed convolution: `(in-1)*stride-2*padding+dilation*(kernel_size-1)+output_padding+1`. Multi-stage blocks validate intermediate geometry as well as final geometry.

Generic graph fusions: add/multiply require 2..8 equal-shaped inputs and forbid broadcasting; concat accepts axis=1 (non-batch axis 1..3); weighted_sum requires 2..8 finite weights in [-1e6,1e6] with one weight per input and learnable=False by default. Graph IDs match `[A-Za-z][A-Za-z0-9_]{0,47}`; topology is validated by preflight/build.

## conv_norm_act

Constructor: `hyso.models.blocks.ConvNormAct`.

Floating [N,in_channels,H,W]; groups divides both channel counts. Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See parameter_dependencies and preflight.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `in_channels` | `int` | required | Input feature channels; must match the incoming tensor C dimension. Integer in 1..8192; bool is rejected. |
| `out_channels` | `int` | required | Output feature channels after this convolution or block. Integer in 1..8192; bool is rejected. |
| `kernel_size` | `int  /  tuple[int, int]  /  list[int]` | 3 | Spatial filter/window extent in H and W; an integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. |
| `stride` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192. |
| `padding` | `int  /  tuple[int, int]  /  list[int]` | 1 | Symmetric padding on each spatial axis; changes output geometry without changing channel count. Integer or H/W pair, each in 0..8192. |
| `groups` | `int` | 1 | Channel partitions; must divide both convolution channel counts. ChannelShuffle requires channels % groups == 0. Integer in 1..8192; bool is rejected. |
| `norm` | `Literal['batch', 'group', None]` | 'batch' | Optional post-convolution normalization: batch, group or None. Group normalization chooses the largest channel divisor <=8; convolution bias is enabled only without normalization. Choices: 'batch', 'group', None. |
| `activation` | `Literal['silu', 'relu', None]` | 'silu' | Nonlinearity applied inside this block; only the declared registry choices are accepted. Choices: 'silu', 'relu', None. |

Cross-parameter requirements: groups divides both channel counts; normalization determines convolution bias.

## conv2d

Constructor: `torch.nn.modules.conv.Conv2d`.

Floating [N,in_channels,H,W]; groups divides both channel counts. Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See parameter_dependencies and preflight.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `in_channels` | `int` | required | Input feature channels; must match the incoming tensor C dimension. Integer in 1..8192; bool is rejected. |
| `out_channels` | `int` | required | Output feature channels after this convolution or block. Integer in 1..8192; bool is rejected. |
| `kernel_size` | `int  /  tuple[int, int]  /  list[int]` | 3 | Spatial filter/window extent in H and W; an integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. |
| `stride` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192. |
| `padding` | `int  /  tuple[int, int]  /  list[int]` | 1 | Symmetric padding on each spatial axis; changes output geometry without changing channel count. Integer or H/W pair, each in 0..8192. |
| `groups` | `int` | 1 | Channel partitions; must divide both convolution channel counts. ChannelShuffle requires channels % groups == 0. Integer in 1..8192; bool is rejected. |
| `bias` | `bool` | True | Include a learned additive convolution bias for each output channel. Boolean only. |
| `dilation` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. |

Cross-parameter requirements: groups divides in_channels and out_channels; dilation changes the effective kernel.

## residual

Constructor: `hyso.models.blocks.ResidualBlock`.

Floating [N,in_channels,H,W]; attention channels equal out_channels. [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `in_channels` | `int` | required | Input feature channels; must match the incoming tensor C dimension. Integer in 1..8192; bool is rejected. |
| `out_channels` | `int` | required | Output feature channels after this convolution or block. Integer in 1..8192; bool is rejected. |
| `stride` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192. |
| `norm` | `Literal['batch', 'group', None]` | 'batch' | Optional post-convolution normalization: batch, group or None. Group normalization chooses the largest channel divisor <=8; convolution bias is enabled only without normalization. Choices: 'batch', 'group', None. |
| `attention` | `dict[str, Any]  /  None` | None | Optional attention applied after the residual branch; nested legacy attention must use matching out_channels. |

## se

Constructor: `hyso.models.WithSE.SEBlock`.

Floating [N,channels,H,W], all dimensions positive. Same NCHW shape; multiplicative gates, no channel or spatial resizing.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `channels` | `int` | required | Feature channels C; gates and normalization preserve this channel count. Integer in 1..8192; bool is rejected. |
| `reduction` | `int` | 8 | Channel-gate reduction divisor; hidden width is max(1,C//reduction), or max(8,C//reduction) for coordinate attention. Integer in 1..8192; bool is rejected. |

## cbam

Constructor: `hyso.models.bcam.CBAMBlock`.

Floating [N,channels,H,W], all dimensions positive. Same NCHW shape; multiplicative gates, no channel or spatial resizing.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `channels` | `int` | required | Feature channels C; gates and normalization preserve this channel count. Integer in 1..8192; bool is rejected. |
| `reduction` | `int` | 16 | Channel-gate reduction divisor; hidden width is max(1,C//reduction), or max(8,C//reduction) for coordinate attention. Integer in 1..8192; bool is rejected. |
| `kernel_size` | `int` | 7 | Spatial filter/window extent in H and W; an integer uses the same extent on both axes. Odd integer in 1..8192; bool is rejected. |

## eca

Constructor: `hyso.models.attention.ECAAttention`.

Floating [N,channels,H,W], all dimensions positive. Same NCHW shape; multiplicative gates, no channel or spatial resizing.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `channels` | `int` | required | Feature channels C; gates and normalization preserve this channel count. Integer in 1..8192; bool is rejected. |
| `kernel_size` | `int` | 3 | Spatial filter/window extent in H and W; an integer uses the same extent on both axes. Odd integer in 1..8192; bool is rejected. |

## coordinate

Constructor: `hyso.models.attention.CoordinateAttention`.

Floating [N,channels,H,W], all dimensions positive. Same NCHW shape; multiplicative gates, no channel or spatial resizing.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `channels` | `int` | required | Feature channels C; gates and normalization preserve this channel count. Integer in 1..8192; bool is rejected. |
| `reduction` | `int` | 32 | Channel-gate reduction divisor; hidden width is max(1,C//reduction), or max(8,C//reduction) for coordinate attention. Integer in 1..8192; bool is rejected. |

## batch_norm

Constructor: `torch.nn.modules.batchnorm.BatchNorm2d`.

Floating NCHW; C matches num_features, num_channels or channels. GroupNorm groups divide C. Same NCHW shape; channel LayerNorm normalizes C per spatial position. Native statistics constraints apply.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `num_features` | `int` | required | Normalization channel count; must match input C. Integer in 1..8192; bool is rejected. |
| `eps` | `float` | 1e-05 | Positive numerical stabilizer used in normalization. Embedding normalization also floors it at the dtype tiny value. Finite positive number. |
| `momentum` | `float  /  None` | 0.1 | Running-statistic update factor; None selects cumulative averaging for BatchNorm. None or number in [0,1]. |
| `affine` | `bool` | True | Learn per-channel scale and bias when True; otherwise use normalization without learned affine parameters. Boolean only. |
| `track_running_stats` | `bool` | True | Track running mean/variance for evaluation. Without tracking, input-statistic sample requirements also apply in eval. Boolean only. |

## group_norm

Constructor: `torch.nn.modules.normalization.GroupNorm`.

Floating NCHW; C matches num_features, num_channels or channels. GroupNorm groups divide C. Same NCHW shape; channel LayerNorm normalizes C per spatial position. Native statistics constraints apply.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `num_groups` | `int` | required | GroupNorm channel partitions; must divide num_channels. Integer in 1..8192; bool is rejected. |
| `num_channels` | `int` | required | Total GroupNorm channels C; must be divisible by num_groups. Integer in 1..8192; bool is rejected. |
| `eps` | `float` | 1e-05 | Positive numerical stabilizer used in normalization. Embedding normalization also floors it at the dtype tiny value. Finite positive number. |
| `affine` | `bool` | True | Learn per-channel scale and bias when True; otherwise use normalization without learned affine parameters. Boolean only. |

Cross-parameter requirements: num_groups divides num_channels.

## relu

Constructor: `torch.nn.modules.activation.ReLU`.

Floating nonempty NCHW in Builder v1. Same NCHW shape; out-of-place activation.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| — | — | No keyword parameters | Uses the component identity/activation contract. |

## silu

Constructor: `torch.nn.modules.activation.SiLU`.

Floating nonempty NCHW in Builder v1. Same NCHW shape; out-of-place activation.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| — | — | No keyword parameters | Uses the component identity/activation contract. |

## hardswish

Constructor: `torch.nn.modules.activation.Hardswish`.

Floating nonempty NCHW in Builder v1. Same NCHW shape; out-of-place activation.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| — | — | No keyword parameters | Uses the component identity/activation contract. |

## identity

Constructor: `torch.nn.modules.linear.Identity`.

Floating nonempty NCHW in Builder v1. Same tensor/shape; no parameters.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| — | — | No keyword parameters | Uses the component identity/activation contract. |

## modular_cnn

Constructor: `hyso.models.modular.ModularCNN`.

Floating nonempty [N,input_channels,H,W]; normal model.to(device/dtype) semantics. [N,num_classes] unnormalized classification logits.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `input_channels` | `int` | 3 | Input image channels C; must match the first model stage. Integer in 1..8192; bool is rejected. |
| `num_classes` | `int` | 10 | Number of output class logits. Outputs are unnormalized scores, not probabilities. Integer in 1..8192; bool is rejected. |
| `widths` | `tuple[int, ...]  /  list[int]` | [32, 64, 128] | Per-stage output channels; length must match repeats and strides. Sequence of 1..16 integers, each in 1..8192. |
| `repeats` | `tuple[int, ...]  /  list[int]` | [1, 1, 1] | Number of blocks per stage; same length as widths/strides, with at most 64 blocks in total. Sequence of 1..16 integers, each in 1..8192. |
| `strides` | `tuple[int, ...]  /  list[int]` | [1, 2, 2] | First-block spatial stride for each stage; remaining blocks use stride 1. Sequence of 1..16 integers, each in 1..8192. |
| `attention` | `Literal['none', 'se', 'cbam', 'eca', 'coordinate']` | 'none' | Attention selector inside the block; applies channel/spatial gates without resizing. Choices: 'none', 'se', 'cbam', 'eca', 'coordinate'. |
| `norm` | `Literal['batch', 'group', None]` | 'batch' | Optional post-convolution normalization: batch, group or None. Group normalization chooses the largest channel divisor <=8; convolution bias is enabled only without normalization. Choices: 'batch', 'group', None. |

## channel_shuffle

Constructor: `hyso.models.regularization.ChannelShuffle`.

Floating NCHW with matching channels. Same NCHW shape; validated channel partition.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `channels` | `int` | required | Feature channels C; gates and normalization preserve this channel count. Integer in 1..8192; bool is rejected. |
| `groups` | `int` | 2 | Channel partitions; must divide both convolution channel counts. ChannelShuffle requires channels % groups == 0. Integer in 1..8192; bool is rejected. |

## dropblock2d

Constructor: `hyso.models.regularization.DropBlock2D`.

Nonempty floating NCHW. Same shape and dtype; explicit training-only masking.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `p` | `float` | 0.1 | Probability of dropping elements, channels or paths during training; evaluation disables masking. Number in [0,1]. |
| `block_size` | `int` | 3 | Requested mask square side; runtime clips it to min(block_size,H,W). Integer in 1..8192; bool is rejected. |

## pointwise_conv2d

Constructor: `hyso.models.components.PointwiseConv2D`.

Floating [N,in_channels,H,W]; groups divides both channel counts. Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See parameter_dependencies and preflight.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `in_channels` | `int` | required | Input feature channels; must match the incoming tensor C dimension. Integer in 1..8192; bool is rejected. |
| `out_channels` | `int` | required | Output feature channels after this convolution or block. Integer in 1..8192; bool is rejected. |
| `stride` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192. |
| `bias` | `bool` | True | Include a learned additive convolution bias for each output channel. Boolean only. |

## depthwise_conv2d

Constructor: `hyso.models.components.DepthwiseConv2D`.

Floating [N,in_channels,H,W]; groups divides both channel counts. Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See parameter_dependencies and preflight.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `in_channels` | `int` | required | Input feature channels; must match the incoming tensor C dimension. Integer in 1..8192; bool is rejected. |
| `depth_multiplier` | `int` | 1 | Independent filters per input channel; depthwise output/intermediate C equals in_channels*depth_multiplier and must not exceed 8192. Integer in 1..8192; bool is rejected. |
| `kernel_size` | `int  /  tuple[int, int]  /  list[int]` | 3 | Spatial filter/window extent in H and W; an integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. |
| `stride` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192. |
| `padding` | `int  /  tuple[int, int]  /  list[int]` | 1 | Symmetric padding on each spatial axis; changes output geometry without changing channel count. Integer or H/W pair, each in 0..8192. |
| `dilation` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. |
| `bias` | `bool` | True | Include a learned additive convolution bias for each output channel. Boolean only. |

Cross-parameter requirements: groups=in_channels; out_channels=in_channels*depth_multiplier.

## depthwise_separable_conv2d

Constructor: `hyso.models.components.DepthwiseSeparableConv2D`.

Floating [N,in_channels,H,W]; groups divides both channel counts. Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See parameter_dependencies and preflight.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `in_channels` | `int` | required | Input feature channels; must match the incoming tensor C dimension. Integer in 1..8192; bool is rejected. |
| `out_channels` | `int` | required | Output feature channels after this convolution or block. Integer in 1..8192; bool is rejected. |
| `depth_multiplier` | `int` | 1 | Independent filters per input channel; depthwise output/intermediate C equals in_channels*depth_multiplier and must not exceed 8192. Integer in 1..8192; bool is rejected. |
| `kernel_size` | `int  /  tuple[int, int]  /  list[int]` | 3 | Spatial filter/window extent in H and W; an integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. |
| `stride` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192. |
| `padding` | `int  /  tuple[int, int]  /  list[int]` | 1 | Symmetric padding on each spatial axis; changes output geometry without changing channel count. Integer or H/W pair, each in 0..8192. |
| `dilation` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. |
| `bias` | `bool` | True | Include a learned additive convolution bias for each output channel. Boolean only. |

Cross-parameter requirements: intermediate_channels=in_channels*depth_multiplier.

## spatial_separable_conv2d

Constructor: `hyso.models.components.SpatialSeparableConv2D`.

Floating [N,in_channels,H,W]; groups divides both channel counts. Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See parameter_dependencies and preflight.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `in_channels` | `int` | required | Input feature channels; must match the incoming tensor C dimension. Integer in 1..8192; bool is rejected. |
| `out_channels` | `int` | required | Output feature channels after this convolution or block. Integer in 1..8192; bool is rejected. |
| `intermediate_channels` | `int` | required | Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in 1..8192; bool is rejected. |
| `kernel_size` | `int  /  tuple[int, int]  /  list[int]` | 3 | Spatial filter/window extent in H and W; an integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. |
| `stride` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192. |
| `padding` | `int  /  tuple[int, int]  /  list[int]` | 1 | Symmetric padding on each spatial axis; changes output geometry without changing channel count. Integer or H/W pair, each in 0..8192. |
| `dilation` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. |
| `bias` | `bool` | True | Include a learned additive convolution bias for each output channel. Boolean only. |

Cross-parameter requirements: H settings apply to the vertical stage, W settings to the horizontal stage.

## factorized_conv2d

Constructor: `hyso.models.components.FactorizedConv2D`.

Floating [N,in_channels,H,W]; groups divides both channel counts. Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See parameter_dependencies and preflight.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `in_channels` | `int` | required | Input feature channels; must match the incoming tensor C dimension. Integer in 1..8192; bool is rejected. |
| `out_channels` | `int` | required | Output feature channels after this convolution or block. Integer in 1..8192; bool is rejected. |
| `intermediate_channels` | `int` | required | Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in 1..8192; bool is rejected. |
| `kernel_size` | `int  /  tuple[int, int]  /  list[int]` | 3 | Spatial filter/window extent in H and W; an integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. |
| `stride` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192. |
| `padding` | `int  /  tuple[int, int]  /  list[int]` | 1 | Symmetric padding on each spatial axis; changes output geometry without changing channel count. Integer or H/W pair, each in 0..8192. |
| `dilation` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. |
| `bias` | `bool` | True | Include a learned additive convolution bias for each output channel. Boolean only. |

Cross-parameter requirements: intermediate_channels <= min(in_channels,out_channels); 1x1 reduction then KxK convolution.

## conv_transpose2d

Constructor: `torch.nn.modules.conv.ConvTranspose2d`.

Floating [N,in_channels,H,W]; groups divides both channel counts. Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See parameter_dependencies and preflight.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `in_channels` | `int` | required | Input feature channels; must match the incoming tensor C dimension. Integer in 1..8192; bool is rejected. |
| `out_channels` | `int` | required | Output feature channels after this convolution or block. Integer in 1..8192; bool is rejected. |
| `kernel_size` | `int  /  tuple[int, int]  /  list[int]` | 3 | Spatial filter/window extent in H and W; an integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. |
| `stride` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192. |
| `padding` | `int  /  tuple[int, int]  /  list[int]` | 1 | Symmetric padding on each spatial axis; changes output geometry without changing channel count. Integer or H/W pair, each in 0..8192. |
| `dilation` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. |
| `bias` | `bool` | True | Include a learned additive convolution bias for each output channel. Boolean only. |
| `groups` | `int` | 1 | Channel partitions; must divide both convolution channel counts. ChannelShuffle requires channels % groups == 0. Integer in 1..8192; bool is rejected. |
| `output_padding` | `int  /  tuple[int, int]  /  list[int]` | 0 | Additional output extent for transposed convolution, not input padding; each axis must be less than stride or dilation on that axis. Integer or H/W pair, each in 0..8192. |

Cross-parameter requirements: groups divides both channel counts; output_padding[i] < stride[i] or dilation[i].

## channel_layer_norm

Constructor: `hyso.models.components.ChannelLayerNorm`.

Floating NCHW; C matches num_features, num_channels or channels. GroupNorm groups divide C. Same NCHW shape; channel LayerNorm normalizes C per spatial position. Native statistics constraints apply.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `channels` | `int` | required | Feature channels C; gates and normalization preserve this channel count. Integer in 1..8192; bool is rejected. |
| `eps` | `float` | 1e-05 | Positive numerical stabilizer used in normalization. Embedding normalization also floors it at the dtype tiny value. Finite positive number. |
| `affine` | `bool` | True | Learn per-channel scale and bias when True; otherwise use normalization without learned affine parameters. Boolean only. |

Cross-parameter requirements: normalizes C at each N,H,W position, external layout stays NCHW.

## instance_norm

Constructor: `hyso.models.components.InstanceNorm2D`.

Floating NCHW; C matches num_features, num_channels or channels. GroupNorm groups divide C. Same NCHW shape; channel LayerNorm normalizes C per spatial position. Native statistics constraints apply.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `num_features` | `int` | required | Normalization channel count; must match input C. Integer in 1..8192; bool is rejected. |
| `eps` | `float` | 1e-05 | Positive numerical stabilizer used in normalization. Embedding normalization also floors it at the dtype tiny value. Finite positive number. |
| `momentum` | `float` | 0.1 | Running-statistic update factor; None selects cumulative averaging for BatchNorm. Number in [0,1]. |
| `affine` | `bool` | False | Learn per-channel scale and bias when True; otherwise use normalization without learned affine parameters. Boolean only. |
| `track_running_stats` | `bool` | False | Track running mean/variance for evaluation. Without tracking, input-statistic sample requirements also apply in eval. Boolean only. |

Cross-parameter requirements: spatial sample count H*W > 1 when using input statistics.

## leaky_relu

Constructor: `torch.nn.modules.activation.LeakyReLU`.

Floating nonempty NCHW in Builder v1. Same NCHW shape; out-of-place activation.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `negative_slope` | `float` | 0.01 | Nonnegative multiplier for negative LeakyReLU inputs. Finite nonnegative number. |

## prelu

Constructor: `torch.nn.modules.activation.PReLU`.

Floating nonempty NCHW in Builder v1. Same NCHW shape; out-of-place activation.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `num_parameters` | `int` | 1 | PReLU slope count; must be 1 (shared) or equal to incoming C. Integer in 1..8192; bool is rejected. |
| `init` | `float` | 0.25 | Initial learned PReLU negative slope; finite and representable as float32. Finite number. |

Cross-parameter requirements: num_parameters is 1 or the incoming channel count.

## gelu

Constructor: `torch.nn.modules.activation.GELU`.

Floating nonempty NCHW in Builder v1. Same NCHW shape; out-of-place activation.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `approximate` | `Literal['none', 'tanh']` | 'none' | GELU calculation: none uses the exact error function; tanh uses the approximation. Choices: 'none', 'tanh'. |

## mish

Constructor: `torch.nn.modules.activation.Mish`.

Floating nonempty NCHW in Builder v1. Same NCHW shape; out-of-place activation.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| — | — | No keyword parameters | Uses the component identity/activation contract. |

## hardsigmoid

Constructor: `torch.nn.modules.activation.Hardsigmoid`.

Floating nonempty NCHW in Builder v1. Same NCHW shape; out-of-place activation.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| — | — | No keyword parameters | Uses the component identity/activation contract. |

## max_pool2d

Constructor: `torch.nn.modules.pooling.MaxPool2d`.

Floating nonempty NCHW; legal window/padding or positive adaptive output_size. Floating NCHW with unchanged channels; global average returns [N,C,1,1].

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `kernel_size` | `int  /  tuple[int, int]  /  list[int]` | required | Spatial filter/window extent in H and W; an integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. |
| `stride` | `int  /  tuple[int, int]  /  list[int]  /  None` | None | Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution. None or integer/HW pair, each in 1..8192. |
| `padding` | `int  /  tuple[int, int]  /  list[int]` | 0 | Symmetric padding on each spatial axis; changes output geometry without changing channel count. Integer or H/W pair, each in 0..8192. |
| `dilation` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. |
| `ceil_mode` | `bool` | False | Round pooling output sizes upward, then discard windows beginning wholly in right/bottom padding. Boolean only. |

Cross-parameter requirements: stride=None uses kernel_size; padding[i] <= floor(kernel_size[i]/2).

## avg_pool2d

Constructor: `torch.nn.modules.pooling.AvgPool2d`.

Floating nonempty NCHW; legal window/padding or positive adaptive output_size. Floating NCHW with unchanged channels; global average returns [N,C,1,1].

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `kernel_size` | `int  /  tuple[int, int]  /  list[int]` | required | Spatial filter/window extent in H and W; an integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. |
| `stride` | `int  /  tuple[int, int]  /  list[int]  /  None` | None | Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution. None or integer/HW pair, each in 1..8192. |
| `padding` | `int  /  tuple[int, int]  /  list[int]` | 0 | Symmetric padding on each spatial axis; changes output geometry without changing channel count. Integer or H/W pair, each in 0..8192. |
| `ceil_mode` | `bool` | False | Round pooling output sizes upward, then discard windows beginning wholly in right/bottom padding. Boolean only. |
| `count_include_pad` | `bool` | True | Include zero-padded elements in the average-pooling divisor. Boolean only. |

Cross-parameter requirements: stride=None uses kernel_size; padding[i] <= floor(kernel_size[i]/2); count_include_pad controls the mean divisor.

## adaptive_avg_pool2d

Constructor: `torch.nn.modules.pooling.AdaptiveAvgPool2d`.

Floating nonempty NCHW; legal window/padding or positive adaptive output_size. Floating NCHW with unchanged channels; global average returns [N,C,1,1].

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `output_size` | `int  /  tuple[int, int]  /  list[int]` | required | Target H/W grid; integer means a square. Resize requires exactly one of output_size and scale_factor. Integer or H/W pair, each in 1..8192. |

## adaptive_max_pool2d

Constructor: `torch.nn.modules.pooling.AdaptiveMaxPool2d`.

Floating nonempty NCHW; legal window/padding or positive adaptive output_size. Floating NCHW with unchanged channels; global average returns [N,C,1,1].

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `output_size` | `int  /  tuple[int, int]  /  list[int]` | required | Target H/W grid; integer means a square. Resize requires exactly one of output_size and scale_factor. Integer or H/W pair, each in 1..8192. |

## global_avg_pool2d

Constructor: `hyso.models.components.GlobalAveragePool2D`.

Floating nonempty NCHW; legal window/padding or positive adaptive output_size. Floating NCHW with unchanged channels; global average returns [N,C,1,1].

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| — | — | No keyword parameters | Uses the component identity/activation contract. |

## bottleneck_residual

Constructor: `hyso.models.advanced.AdvancedResidual`.

Floating [N,in_channels,H,W]; attention channels equal out_channels. [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `in_channels` | `int` | required | Input feature channels; must match the incoming tensor C dimension. Integer in 1..8192; bool is rejected. |
| `out_channels` | `int` | required | Output feature channels after this convolution or block. Integer in 1..8192; bool is rejected. |
| `intermediate_channels` | `int` | required | Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in 1..8192; bool is rejected. |
| `kernel_size` | `int` | 3 | Spatial filter/window extent in H and W; an integer uses the same extent on both axes. Odd integer in 1..8192; bool is rejected. |
| `stride` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192. |
| `dilation` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. |
| `normalization` | `Literal['batch', 'group', None]` | 'batch' | Normalization of residual main-branch features; batch, group or None. Choices: 'batch', 'group', None. |
| `activation` | `Literal['relu', 'silu', 'gelu']` | 'silu' | Nonlinearity applied inside this block; only the declared registry choices are accepted. Choices: 'relu', 'silu', 'gelu'. |
| `attention` | `Literal['none', 'eca', 'spatial_attention', 'channel_attention']` | 'none' | Attention selector inside the block; applies channel/spatial gates without resizing. Choices: 'none', 'eca', 'spatial_attention', 'channel_attention'. |
| `dropout` | `float` | 0.0 | Training-only activation/head dropout probability; disabled in eval. Finite number in [0,1). |
| `drop_path` | `float` | 0.0 | Training-only per-sample residual path drop probability; disabled in eval. Finite number in [0,1). |

## preactivation_residual

Constructor: `hyso.models.advanced.PreActivationResidual`.

Floating [N,in_channels,H,W]; attention channels equal out_channels. [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `in_channels` | `int` | required | Input feature channels; must match the incoming tensor C dimension. Integer in 1..8192; bool is rejected. |
| `out_channels` | `int` | required | Output feature channels after this convolution or block. Integer in 1..8192; bool is rejected. |
| `intermediate_channels` | `int` | required | Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in 1..8192; bool is rejected. |
| `kernel_size` | `int` | 3 | Spatial filter/window extent in H and W; an integer uses the same extent on both axes. Odd integer in 1..8192; bool is rejected. |
| `stride` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192. |
| `dilation` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. |
| `normalization` | `Literal['batch', 'group', None]` | 'batch' | Normalization of residual main-branch features; batch, group or None. Choices: 'batch', 'group', None. |
| `activation` | `Literal['relu', 'silu', 'gelu']` | 'silu' | Nonlinearity applied inside this block; only the declared registry choices are accepted. Choices: 'relu', 'silu', 'gelu'. |
| `attention` | `Literal['none', 'eca', 'spatial_attention', 'channel_attention']` | 'none' | Attention selector inside the block; applies channel/spatial gates without resizing. Choices: 'none', 'eca', 'spatial_attention', 'channel_attention'. |
| `dropout` | `float` | 0.0 | Training-only activation/head dropout probability; disabled in eval. Finite number in [0,1). |
| `drop_path` | `float` | 0.0 | Training-only per-sample residual path drop probability; disabled in eval. Finite number in [0,1). |

## wide_residual_block

Constructor: `hyso.models.advanced.WideResidual`.

Floating [N,in_channels,H,W]; attention channels equal out_channels. [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `in_channels` | `int` | required | Input feature channels; must match the incoming tensor C dimension. Integer in 1..8192; bool is rejected. |
| `out_channels` | `int` | required | Output feature channels after this convolution or block. Integer in 1..8192; bool is rejected. |
| `intermediate_channels` | `int` | required | Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in 1..8192; bool is rejected. |
| `kernel_size` | `int` | 3 | Spatial filter/window extent in H and W; an integer uses the same extent on both axes. Odd integer in 1..8192; bool is rejected. |
| `stride` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192. |
| `dilation` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. |
| `normalization` | `Literal['batch', 'group', None]` | 'batch' | Normalization of residual main-branch features; batch, group or None. Choices: 'batch', 'group', None. |
| `activation` | `Literal['relu', 'silu', 'gelu']` | 'silu' | Nonlinearity applied inside this block; only the declared registry choices are accepted. Choices: 'relu', 'silu', 'gelu'. |
| `attention` | `Literal['none', 'eca', 'spatial_attention', 'channel_attention']` | 'none' | Attention selector inside the block; applies channel/spatial gates without resizing. Choices: 'none', 'eca', 'spatial_attention', 'channel_attention'. |
| `dropout` | `float` | 0.0 | Training-only activation/head dropout probability; disabled in eval. Finite number in [0,1). |
| `drop_path` | `float` | 0.0 | Training-only per-sample residual path drop probability; disabled in eval. Finite number in [0,1). |

## grouped_bottleneck

Constructor: `hyso.models.advanced.GroupedBottleneck`.

Floating [N,in_channels,H,W]; attention channels equal out_channels. [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `in_channels` | `int` | required | Input feature channels; must match the incoming tensor C dimension. Integer in 1..8192; bool is rejected. |
| `out_channels` | `int` | required | Output feature channels after this convolution or block. Integer in 1..8192; bool is rejected. |
| `intermediate_channels` | `int` | required | Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in 1..8192; bool is rejected. |
| `kernel_size` | `int` | 3 | Spatial filter/window extent in H and W; an integer uses the same extent on both axes. Odd integer in 1..8192; bool is rejected. |
| `stride` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192. |
| `dilation` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. |
| `normalization` | `Literal['batch', 'group', None]` | 'batch' | Normalization of residual main-branch features; batch, group or None. Choices: 'batch', 'group', None. |
| `activation` | `Literal['relu', 'silu', 'gelu']` | 'silu' | Nonlinearity applied inside this block; only the declared registry choices are accepted. Choices: 'relu', 'silu', 'gelu'. |
| `attention` | `Literal['none', 'eca', 'spatial_attention', 'channel_attention']` | 'none' | Attention selector inside the block; applies channel/spatial gates without resizing. Choices: 'none', 'eca', 'spatial_attention', 'channel_attention'. |
| `dropout` | `float` | 0.0 | Training-only activation/head dropout probability; disabled in eval. Finite number in [0,1). |
| `drop_path` | `float` | 0.0 | Training-only per-sample residual path drop probability; disabled in eval. Finite number in [0,1). |
| `groups` | `int` | 1 | Channel partitions; must divide both convolution channel counts. ChannelShuffle requires channels % groups == 0. Integer in 1..8192; bool is rejected. |

## inverted_bottleneck

Constructor: `hyso.models.advanced.InvertedBottleneck`.

Floating [N,in_channels,H,W]; attention channels equal out_channels. [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `in_channels` | `int` | required | Input feature channels; must match the incoming tensor C dimension. Integer in 1..8192; bool is rejected. |
| `out_channels` | `int` | required | Output feature channels after this convolution or block. Integer in 1..8192; bool is rejected. |
| `expansion_ratio` | `int` | 2 | Inverted-bottleneck channel multiplier; intermediate C=in_channels*expansion_ratio must not exceed 8192. Integer in 1..8192; bool is rejected. |
| `kernel_size` | `int` | 3 | Spatial filter/window extent in H and W; an integer uses the same extent on both axes. Odd integer in 1..8192; bool is rejected. |
| `stride` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192. |
| `dilation` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. |
| `normalization` | `Literal['batch', 'group', None]` | 'batch' | Normalization of residual main-branch features; batch, group or None. Choices: 'batch', 'group', None. |
| `activation` | `Literal['relu', 'silu', 'gelu']` | 'silu' | Nonlinearity applied inside this block; only the declared registry choices are accepted. Choices: 'relu', 'silu', 'gelu'. |
| `attention` | `Literal['none', 'eca', 'spatial_attention', 'channel_attention']` | 'none' | Attention selector inside the block; applies channel/spatial gates without resizing. Choices: 'none', 'eca', 'spatial_attention', 'channel_attention'. |
| `dropout` | `float` | 0.0 | Training-only activation/head dropout probability; disabled in eval. Finite number in [0,1). |
| `drop_path` | `float` | 0.0 | Training-only per-sample residual path drop probability; disabled in eval. Finite number in [0,1). |

## gated_residual

Constructor: `hyso.models.advanced.GatedResidual`.

Floating [N,in_channels,H,W]; attention channels equal out_channels. [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `in_channels` | `int` | required | Input feature channels; must match the incoming tensor C dimension. Integer in 1..8192; bool is rejected. |
| `out_channels` | `int` | required | Output feature channels after this convolution or block. Integer in 1..8192; bool is rejected. |
| `intermediate_channels` | `int` | required | Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in 1..8192; bool is rejected. |
| `kernel_size` | `int` | 3 | Spatial filter/window extent in H and W; an integer uses the same extent on both axes. Odd integer in 1..8192; bool is rejected. |
| `stride` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192. |
| `dilation` | `int  /  tuple[int, int]  /  list[int]` | 1 | Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. |
| `normalization` | `Literal['batch', 'group', None]` | 'batch' | Normalization of residual main-branch features; batch, group or None. Choices: 'batch', 'group', None. |
| `activation` | `Literal['relu', 'silu', 'gelu']` | 'silu' | Nonlinearity applied inside this block; only the declared registry choices are accepted. Choices: 'relu', 'silu', 'gelu'. |
| `attention` | `Literal['none', 'eca', 'spatial_attention', 'channel_attention']` | 'none' | Attention selector inside the block; applies channel/spatial gates without resizing. Choices: 'none', 'eca', 'spatial_attention', 'channel_attention'. |
| `dropout` | `float` | 0.0 | Training-only activation/head dropout probability; disabled in eval. Finite number in [0,1). |
| `drop_path` | `float` | 0.0 | Training-only per-sample residual path drop probability; disabled in eval. Finite number in [0,1). |
| `gate_init` | `float` | 1.0 | Initial scalar residual contribution in [0,1]; zero suppresses the residual main branch initially. Number in [0,1]. |
| `learnable_gate` | `bool` | True | Store the residual gate as a learned scalar when True, otherwise a fixed scalar. Boolean only. |

## spatial_attention

Constructor: `hyso.models.advanced.SpatialAttention`.

Floating [N,channels,H,W], all dimensions positive. Same NCHW shape; multiplicative gates, no channel or spatial resizing.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `channels` | `int` | required | Feature channels C; gates and normalization preserve this channel count. Integer in 1..8192; bool is rejected. |
| `kernel_size` | `int` | 7 | Spatial filter/window extent in H and W; an integer uses the same extent on both axes. Odd integer in 1..8192; bool is rejected. |

## channel_attention

Constructor: `hyso.models.advanced.ChannelAttention`.

Floating [N,channels,H,W], all dimensions positive. Same NCHW shape; multiplicative gates, no channel or spatial resizing.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `channels` | `int` | required | Feature channels C; gates and normalization preserve this channel count. Integer in 1..8192; bool is rejected. |
| `reduction` | `int` | 8 | Channel-gate reduction divisor; hidden width is max(1,C//reduction), or max(8,C//reduction) for coordinate attention. Integer in 1..8192; bool is rejected. |

## spatial_self_attention

Constructor: `hyso.models.advanced.SpatialSelfAttention`.

Floating [N,channels,H,W], all dimensions positive. Same NCHW shape; multiplicative gates, no channel or spatial resizing.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `channels` | `int` | required | Feature channels C; gates and normalization preserve this channel count. Integer in 1..8192; bool is rejected. |
| `embedding_dim` | `int  /  None` | None | Attention projection width or output embedding width, according to the component. None in self-attention uses C; QKV requires 3*embedding_dim<=8192. None or integer in 1..8192. |
| `dropout` | `float` | 0.0 | Training-only activation/head dropout probability; disabled in eval. Finite number in [0,1). |

## multihead_spatial_attention

Constructor: `hyso.models.advanced.MultiHeadSpatialAttention`.

Floating [N,channels,H,W], all dimensions positive. Same NCHW shape; multiplicative gates, no channel or spatial resizing.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `channels` | `int` | required | Feature channels C; gates and normalization preserve this channel count. Integer in 1..8192; bool is rejected. |
| `embedding_dim` | `int  /  None` | None | Attention projection width or output embedding width, according to the component. None in self-attention uses C; QKV requires 3*embedding_dim<=8192. None or integer in 1..8192. |
| `num_heads` | `int` | 4 | Parallel attention heads; must divide embedding_dim. Spatial attention is limited to 1024 tokens and N*heads*tokens^2<=4194304. Integer in 1..8192; bool is rejected. |
| `dropout` | `float` | 0.0 | Training-only activation/head dropout probability; disabled in eval. Finite number in [0,1). |

## dropout

Constructor: `torch.nn.modules.dropout.Dropout`.

Nonempty floating NCHW. Same shape and dtype; explicit training-only masking.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `p` | `float` | 0.5 | Probability of dropping elements, channels or paths during training; evaluation disables masking. Number in [0,1]. |

## dropout2d

Constructor: `torch.nn.modules.dropout.Dropout2d`.

Nonempty floating NCHW. Same shape and dtype; explicit training-only masking.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `p` | `float` | 0.5 | Probability of dropping elements, channels or paths during training; evaluation disables masking. Number in [0,1]. |

## drop_path

Constructor: `hyso.models.advanced.DropPath`.

Nonempty floating NCHW. Same shape and dtype; explicit training-only masking.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `p` | `float` | 0.0 | Probability of dropping elements, channels or paths during training; evaluation disables masking. Finite number in [0,1). |

## gaussian_noise

Constructor: `hyso.models.advanced.GaussianNoise`.

Nonempty floating NCHW. Same shape and dtype; explicit training-only masking.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `std` | `float` | 0.1 | Standard deviation of training-only zero-mean Gaussian noise; finite, nonnegative and at most 1e6. Finite nonnegative number. |

## resize

Constructor: `hyso.models.sampling.Resize`.

Floating nonempty NCHW. Floating NCHW with explicit resize or pixel rearrangement geometry.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `output_size` | `int  /  tuple[int, int]  /  list[int]  /  None` | None | Target H/W grid; integer means a square. Resize requires exactly one of output_size and scale_factor. None or integer/HW pair, each in 1..8192. |
| `scale_factor` | `float  /  tuple[float, float]  /  list[float]  /  None` | None | Positive H/W resize multipliers, each <=8192. Exactly one of this and output_size must be provided. None or positive scalar/HW factors, each <=8192. |
| `mode` | `Literal['nearest', 'bilinear', 'bicubic', 'area']` | 'nearest' | Resize interpolation method; nearest/area require align_corners=None. Choices: 'nearest', 'bilinear', 'bicubic', 'area'. |
| `align_corners` | `bool  /  None` | None | Align endpoint pixel centers for bilinear/bicubic resizing; must be None for nearest/area. Boolean or None. |

Cross-parameter requirements: resize requires exactly one target; pixel factors require divisibility; reshape preserves sample element count; heads are terminal, in_features is inferred by Builder.

## pixel_shuffle

Constructor: `torch.nn.modules.pixelshuffle.PixelShuffle`.

Floating nonempty NCHW. Floating NCHW with explicit resize or pixel rearrangement geometry.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `upscale_factor` | `int` | required | PixelShuffle factor r: C must be divisible by r^2; output is (C/r^2,H*r,W*r). Integer in 1..8192; bool is rejected. |

Cross-parameter requirements: resize requires exactly one target; pixel factors require divisibility; reshape preserves sample element count; heads are terminal, in_features is inferred by Builder.

## pixel_unshuffle

Constructor: `torch.nn.modules.pixelshuffle.PixelUnshuffle`.

Floating nonempty NCHW. Floating NCHW with explicit resize or pixel rearrangement geometry.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `downscale_factor` | `int` | required | PixelUnshuffle factor r: H and W must be divisible by r; output is (C*r^2,H/r,W/r). Integer in 1..8192; bool is rejected. |

Cross-parameter requirements: resize requires exactly one target; pixel factors require divisibility; reshape preserves sample element count; heads are terminal, in_features is inferred by Builder.

## flatten

Constructor: `hyso.models.sampling.Flatten`.

Floating NCHW with preserved batch axis. NCHW or terminal NF for flatten, see shape_contract.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| — | — | No keyword parameters | Uses the component identity/activation contract. |

Cross-parameter requirements: resize requires exactly one target; pixel factors require divisibility; reshape preserves sample element count; heads are terminal, in_features is inferred by Builder.

## permute

Constructor: `hyso.models.sampling.Permute`.

Floating NCHW with preserved batch axis. NCHW or terminal NF for flatten, see shape_contract.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `dims` | `tuple[int, ...]  /  list[int]` | [0, 1, 3, 2] | Permutation of NCHW axes; only (0,1,2,3) and (0,1,3,2) are accepted, preserving batch/channel axes. Only identity or spatial-axis swap is supported. |

Cross-parameter requirements: resize requires exactly one target; pixel factors require divisibility; reshape preserves sample element count; heads are terminal, in_features is inferred by Builder.

## reshape

Constructor: `hyso.models.sampling.Reshape`.

Floating NCHW with preserved batch axis. NCHW or terminal NF for flatten, see shape_contract.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `shape` | `tuple[int, int, int]  /  list[int]` | required | Target per-sample CHW dimensions; reshape must preserve the number of sample elements and every dimension is 1..8192. Three integers in 1..8192. |

Cross-parameter requirements: resize requires exactly one target; pixel factors require divisibility; reshape preserves sample element count; heads are terminal, in_features is inferred by Builder.

## classification_head

Constructor: `hyso.models.sampling.PredictionHead`.

Floating NCHW or NF with pooling=none. Floating NF prediction, head type controls output semantics.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `in_features` | `int` | required | Linear-head input width after pooling/flattening; must match the inferred feature width. Integer in 1..8192; bool is rejected. |
| `pooling` | `Literal['avg', 'max', 'none']` | 'avg' | avg/max globally pool NCHW before flattening; none flattens NCHW or consumes flat NF. Choices: 'avg', 'max', 'none'. |
| `num_classes` | `int` | required | Number of output class logits. Outputs are unnormalized scores, not probabilities. Integer in 1..8192; bool is rejected. |
| `dropout` | `float` | 0.0 | Training-only activation/head dropout probability; disabled in eval. Finite number in [0,1). |

Cross-parameter requirements: resize requires exactly one target; pixel factors require divisibility; reshape preserves sample element count; heads are terminal, in_features is inferred by Builder.

## regression_head

Constructor: `hyso.models.sampling.RegressionHead`.

Floating NCHW or NF with pooling=none. Floating NF prediction, head type controls output semantics.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `in_features` | `int` | required | Linear-head input width after pooling/flattening; must match the inferred feature width. Integer in 1..8192; bool is rejected. |
| `pooling` | `Literal['avg', 'max', 'none']` | 'avg' | avg/max globally pool NCHW before flattening; none flattens NCHW or consumes flat NF. Choices: 'avg', 'max', 'none'. |
| `output_features` | `int` | required | Width of the unrestricted numeric regression output; no classification objective is inferred. Integer in 1..8192; bool is rejected. |

Cross-parameter requirements: resize requires exactly one target; pixel factors require divisibility; reshape preserves sample element count; heads are terminal, in_features is inferred by Builder.

## embedding_head

Constructor: `hyso.models.sampling.EmbeddingHead`.

Floating NCHW or NF with pooling=none. Floating NF prediction, head type controls output semantics.

| Parameter | Type | Required / default | Meaning and validation |
|---|---|---|---|
| `in_features` | `int` | required | Linear-head input width after pooling/flattening; must match the inferred feature width. Integer in 1..8192; bool is rejected. |
| `pooling` | `Literal['avg', 'max', 'none']` | 'avg' | avg/max globally pool NCHW before flattening; none flattens NCHW or consumes flat NF. Choices: 'avg', 'max', 'none'. |
| `embedding_dim` | `int` | required | Output embedding width for a head, or internal attention projection width as declared by this component. Integer in 1..8192; bool is rejected. |
| `normalize` | `bool` | False | Apply safe per-sample L2 normalization to output embeddings. Boolean only. |
| `eps` | `float` | 1e-12 | Positive numerical stabilizer used in normalization. Embedding normalization also floors it at the dtype tiny value. Finite positive number. |

Cross-parameter requirements: resize requires exactly one target; pixel factors require divisibility; reshape preserves sample element count; heads are terminal, in_features is inferred by Builder.
