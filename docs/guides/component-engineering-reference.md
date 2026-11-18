# Component engineering reference

Source-backed notes retained from the retired browser documentation. Defaults, bounds and validation remain defined by ModelRegistry and GraphModelBuilder. These notes do not claim measured performance.

## conv_norm_act

Applies the selected normalization and activation after Conv2d. Group normalization uses the largest channel divisor at most eight.

**Operation:** y = activation(norm(conv(x)))

**Use:** Use when convolution needs explicit normalization and activation.

**Limits:** Batch norm requires training statistics; norm=None enables convolution bias.

**Cost:** Convolution, normalization and activation run sequentially; group norm uses the largest divisor <=8.

Source: `hyso/models/blocks.py`.

- `in_channels`: Must match the preceding tensor's C dimension.
- `out_channels`: C dimension produced by the learned filters.
- `kernel_size`: Spatial filter window. Any oddness requirement is specified in this component's metadata.
- `stride`: Spacing between window origins. Nullable pooling uses kernel size when null.
- `padding`: Padding applied to both sides of the spatial boundaries.
- `groups`: Applicable channel widths must be divisible by this value.
- `norm`: Internal normalization selector restricted by metadata choices.
- `activation`: Internal activation function.

## conv2d

Learned local spatial filters produce output channels. groups must divide both input and output channels.

**Operation:** Hout = floor((H + 2p - d(k-1) - 1)/s + 1); parameters = Cout(Cin/groups)kHkW + bias*Cout

**Use:** Learn local patterns and change channel width.

**Limits:** groups must divide both channel widths; padding is explicit.

**Cost:** Filter cost grows with Cin*Cout*kH*kW/groups; activations depend on output geometry.

Source: `torch/nn/modules/conv.py`.

- `in_channels`: Must match the preceding tensor's C dimension.
- `out_channels`: C dimension produced by the learned filters.
- `kernel_size`: Spatial filter window. Any oddness requirement is specified in this component's metadata.
- `stride`: Spacing between window origins. Nullable pooling uses kernel size when null.
- `padding`: Padding applied to both sides of the spatial boundaries.
- `groups`: Applicable channel widths must be divisible by this value.
- `bias`: Adds a learnable constant per output channel.
- `dilation`: Spacing between kernel samples.

## residual

Adds a two-convolution main branch to a skip branch. This block constructs its internal projection for channel or stride changes. Graph-level Add requires a separate explicit projection node.

**Operation:** y = activation(F(x) + projection(x))

**Use:** Use two 3x3 convolutions with a skip and final SiLU.

**Limits:** Internal attention must preserve main-branch shape; graph Add does not project.

**Cost:** Two convolutions, selected norm/attention and an optional normalized 1x1 skip.

Source: `hyso/models/blocks.py`.

- `in_channels`: Must match the preceding tensor's C dimension.
- `out_channels`: C dimension produced by the learned filters.
- `stride`: Spacing between window origins. Nullable pooling uses kernel size when null.
- `norm`: Internal normalization selector restricted by metadata choices.
- `attention`: Internal attention selector or supported name/params specification. Channel contracts must match.

## se

Builds channel descriptors with global spatial averages and learns sigmoid gates through a reduced MLP.

**Operation:** y = x * sigmoid(W2 activation(W1 GAP(x)))

**Use:** Learn channel importance from global descriptors.

**Limits:** Each channel gate is shared over H,W; it does not perform position-to-position self-attention.

**Cost:** Global mean and two linear layers; hidden width max(1,C//reduction). It creates no quadratic H*W attention matrix.

Source: `hyso/models/WithSE.py`.

- `channels`: Channel contract matching input C.
- `reduction`: The channel descriptor hidden width is max(1, C//reduction). Increasing the ratio reduces the representation until the floor is reached.

## cbam

Applies channel attention followed by spatial attention. The spatial filter kernel must be odd.

**Operation:** y = spatial_gate(channel_gate(x)*x) * (channel_gate(x)*x)

**Use:** Apply channel and spatial gates sequentially.

**Limits:** Spatial kernel must be odd; this is not QK token interaction.

**Cost:** Shared channel MLP mean/max passes and a 2-to-1 spatial convolution; no quadratic token matrix.

Source: `hyso/models/bcam.py`.

- `channels`: Channel contract matching input C.
- `reduction`: The channel descriptor hidden width is max(1, C//reduction). Increasing the ratio reduces the representation until the floor is reached.
- `kernel_size`: Spatial filter window. Any oddness requirement is specified in this component's metadata.

## eca

Applies an odd Conv1d kernel along the channel axis of a global-average descriptor. It uses no channel reduction.

**Operation:** y = x * sigmoid(Conv1d(GAP(x)))

**Use:** Mix neighbouring channel descriptors without reduction.

**Limits:** Kernel is an explicit positive odd value; no adaptive kernel selection is performed.

**Cost:** kernel_size Conv1d weights and global mean; no spatial token matrix.

Source: `hyso/models/attention.py`.

- `channels`: Channel contract matching input C.
- `kernel_size`: Spatial filter window. Any oddness requirement is specified in this component's metadata.

## coordinate

Builds position-sensitive gates from separate height and width descriptors. Internal BatchNorm requires N*(H+W)>1 during training.

**Operation:** y[n,c,h,w] = x[n,c,h,w] * gateH[n,c,h,1] * gateW[n,c,1,w]

**Use:** Use separate height- and width-dependent gates.

**Limits:** Internal BatchNorm retains train/eval semantics; descriptors span H+W.

**Cost:** Hidden width max(8,C//reduction); H+W descriptors and three 1x1 convolutions, no QK matrix.

Source: `hyso/models/attention.py`.

- `channels`: Channel contract matching input C.
- `reduction`: The channel descriptor hidden width is max(8, C//reduction). Increasing the ratio reduces the representation until the floor is reached.

## batch_norm

Normalizes each channel using N,H,W statistics. Eval uses retained statistics when enabled. Training needs N*H*W>1.

**Operation:** y = gamma*(x-mean)/sqrt(var+eps)+beta

**Use:** Normalize channels with batch/spatial statistics.

**Limits:** Training needs N*H*W>1; eval uses retained values when running statistics are enabled.

**Cost:** Statistics/affine passes over input; affine adds 2*C parameters, running statistics are buffers.

Source: `torch/nn/modules/batchnorm.py`.

- `num_features`: Channel width expected by normalization.
- `eps`: Positive denominator constant preventing division by zero.
- `momentum`: Running-statistics update factor. Null uses cumulative averaging where supported.
- `affine`: Learns per-channel scale and shift after normalization.
- `track_running_stats`: Retains training statistics in buffers for eval.

## group_norm

Partitions channels within each sample. Channels must divide by groups. Train and eval use the same grouped statistics.

**Operation:** mean,var over (C/groups,H,W) per (N,group)

**Use:** Compute statistics within per-sample channel groups.

**Limits:** C must divide by num_groups; each group needs sufficient statistics.

**Cost:** Same grouping in train/eval; affine adds 2*C parameters and input-sized passes.

Source: `torch/nn/modules/normalization.py`.

- `num_groups`: Number of equal channel groups.
- `num_channels`: Total channel width of group normalization.
- `eps`: Positive denominator constant preventing division by zero.
- `affine`: Learns per-channel scale and shift after normalization.

## relu

Clips negative inputs to zero. The operation is out of place.

**Operation:** y=max(0,x)

**Use:** Use an activation that zeroes negative features.

**Limits:** Derivative is zero on the negative branch; it changes a logits contract.

**Cost:** No parameters; elementwise thresholding.

Source: `torch/nn/modules/activation.py`.


## silu

Multiplies the input by its sigmoid gate. The operation is out of place.

**Operation:** y=x*sigmoid(x)

**Use:** Use a smooth input-dependent sigmoid gate.

**Limits:** Output is not bounded to [0,1]; this is not sigmoid output.

**Cost:** No parameters; per-element sigmoid and multiplication.

Source: `torch/nn/modules/activation.py`.


## hardswish

Computes a swish-like gate using a piecewise linear expression.

**Operation:** y=x*clamp(x+3,0,6)/6

**Use:** Use a swish-like piecewise linear gate.

**Limits:** It is not numerically identical to SiLU; hardware speed gains require measurement.

**Cost:** Parameter-free elementwise clamp and multiplication.

Source: `torch/nn/modules/activation.py`.


## identity

Returns the input tensor unchanged. It performs no shape or channel conversion.

**Operation:** y=x

**Use:** Use a shape-preserving passage or named output.

**Limits:** It does not correct channel, shape or value mismatches.

**Cost:** No learned parameters or transformation.

Source: `torch/nn/modules/linear.py`.


## modular_cnn

The original complete-model constructor. Width, repeat and stride lists must share length. It is not inserted as a graph node.

**Operation:** stem -> residual stages -> global pool -> classifier

**Use:** Use the original complete-model and state_dict contract.

**Limits:** It is not a graph node and supplies no concrete input geometry for the short Studio CPU workflow.

**Cost:** Stem, repeated residual stages and classifier; exact counting requires a concrete Core build.

Source: `hyso/models/modular.py`.

- `input_channels`: Input channel width expected by the complete model.
- `num_classes`: Logit width. Binary tasks need one, multiclass needs at least two.
- `widths`: Channel width of each model stage.
- `repeats`: Number of block repetitions in each stage.
- `strides`: Spatial stride of each stage's first block.
- `attention`: Internal attention selector or supported name/params specification. Channel contracts must match.
- `norm`: Internal normalization selector restricted by metadata choices.

## channel_shuffle

Reshapes channels into groups and transposes group and within-group axes. Channel count must divide by groups.

**Operation:** [N,g,C/g,H,W] -> transpose(1,2) -> [N,C,H,W]

**Use:** Rearrange channel order between grouped filters.

**Limits:** C must divide by groups; this is not learned dense channel mixing.

**Cost:** Parameter-free reshape/transpose; the final reshape may copy data.

Source: `hyso/models/regularization.py`.

- `channels`: Channel contract matching input C.
- `groups`: Applicable channel widths must be divisible by this value.

## dropblock2d

Spreads square masks from valid top-left seeds. Block size is clipped to H,W. An all-dropped mask yields finite zeros.

**Operation:** y=x*mask*(mask.numel()/mask.sum()); empty mask -> 0

**Use:** Drop contiguous spatial regions during training.

**Limits:** Block size is clipped to H,W; all-dropped input returns finite zeros; eval is identity.

**Cost:** Parameter-free seed mask with max-pool square expansion.

Source: `hyso/models/regularization.py`.

- `p`: Training-mask probability. Drop path has an exclusive upper bound of one.
- `block_size`: Side of the square DropBlock mask, clipped to H/W at runtime.

## pointwise_conv2d

A 1x1 filter mixes channels at each location. Padding is zero. Stride above one also subsamples spatial positions.

**Operation:** y[n,o,h,w] = sum_c W[o,c] x[n,c,h*s,w*s] + b[o]

**Use:** Learn channel mixing at each spatial location.

**Limits:** A 1x1 window does not gather spatial context from neighbouring locations.

**Cost:** Cin*Cout weights plus optional Cout bias; stride affects spatial output.

Source: `hyso/models/components.py`.

- `in_channels`: Must match the preceding tensor's C dimension.
- `out_channels`: C dimension produced by the learned filters.
- `stride`: Spacing between window origins. Nullable pooling uses kernel size when null.
- `bias`: Adds a learnable constant per output channel.

## depthwise_conv2d

Filters each input channel independently. groups=Cin and Cout=Cin*depth_multiplier. It does not mix information between input channels.

**Operation:** groups = Cin; Cout = Cin * depth_multiplier

**Use:** Filter each channel spatially and independently.

**Limits:** It learns no cross-channel mixing; Cin*depth_multiplier must be <=8192.

**Cost:** Cin*depth_multiplier*kH*kW filter weights; groups is fixed to Cin.

Source: `hyso/models/components.py`.

- `in_channels`: Must match the preceding tensor's C dimension.
- `depth_multiplier`: Independent depthwise filters per input channel.
- `kernel_size`: Spatial filter window. Any oddness requirement is specified in this component's metadata.
- `stride`: Spacing between window origins. Nullable pooling uses kernel size when null.
- `padding`: Padding applied to both sides of the spatial boundaries.
- `dilation`: Spacing between kernel samples.
- `bias`: Adds a learnable constant per output channel.

## depthwise_separable_conv2d

Runs depthwise spatial filtering followed by dense 1x1 channel mixing. There is no hidden normalization between the operators.

**Operation:** y = Conv1x1(Depthwise(x)); middle = Cin*depth_multiplier

**Use:** Combine channelwise filtering with a 1x1 channel transformation.

**Limits:** There is no normalization or activation between the convolutions.

**Cost:** Depthwise and pointwise weights add; the intermediate tensor has Cin*depth_multiplier channels.

Source: `hyso/models/components.py`.

- `in_channels`: Must match the preceding tensor's C dimension.
- `out_channels`: C dimension produced by the learned filters.
- `depth_multiplier`: Independent depthwise filters per input channel.
- `kernel_size`: Spatial filter window. Any oddness requirement is specified in this component's metadata.
- `stride`: Spacing between window origins. Nullable pooling uses kernel size when null.
- `padding`: Padding applied to both sides of the spatial boundaries.
- `dilation`: Spacing between kernel samples.
- `bias`: Adds a learnable constant per output channel.

## spatial_separable_conv2d

Factors the spatial kernel into kHx1 and 1xkW operators with explicit intermediate channels. Each spatial axis receives its own stride, padding and dilation.

**Operation:** y = Conv(1,kW)(Conv(kH,1)(x))

**Use:** Design vertical and horizontal spatial filters with explicit middle width.

**Limits:** Do not assume equivalence to a learned dense kHxkW convolution.

**Cost:** Cin*Cmid*kH + Cmid*Cout*kW weights plus selected biases.

Source: `hyso/models/components.py`.

- `in_channels`: Must match the preceding tensor's C dimension.
- `out_channels`: C dimension produced by the learned filters.
- `intermediate_channels`: Width of the internal tensor. Core validates additional dependencies.
- `kernel_size`: Spatial filter window. Any oddness requirement is specified in this component's metadata.
- `stride`: Spacing between window origins. Nullable pooling uses kernel size when null.
- `padding`: Padding applied to both sides of the spatial boundaries.
- `dilation`: Spacing between kernel samples.
- `bias`: Adds a learnable constant per output channel.

## factorized_conv2d

A dense 1x1 bottleneck precedes dense kxk convolution. Intermediate channels cannot exceed either boundary width. There is no hidden skip, normalization or activation.

**Operation:** y = ConvKxK(Conv1x1(x)); Cmid <= min(Cin,Cout)

**Use:** Use a learned channel bottleneck before spatial filtering.

**Limits:** Cmid <= min(Cin,Cout); there is no hidden skip or activation.

**Cost:** Cin*Cmid + Cmid*Cout*kH*kW weights plus selected biases.

Source: `hyso/models/components.py`.

- `in_channels`: Must match the preceding tensor's C dimension.
- `out_channels`: C dimension produced by the learned filters.
- `intermediate_channels`: Width of the internal tensor. Core validates additional dependencies.
- `kernel_size`: Spatial filter window. Any oddness requirement is specified in this component's metadata.
- `stride`: Spacing between window origins. Nullable pooling uses kernel size when null.
- `padding`: Padding applied to both sides of the spatial boundaries.
- `dilation`: Spacing between kernel samples.
- `bias`: Adds a learnable constant per output channel.

## conv_transpose2d

Learned filters can expand spatial dimensions. output_padding resolves output-size ambiguity and must be smaller than stride or dilation on each axis.

**Operation:** Hout = (H-1)s - 2p + d(k-1) + output_padding + 1

**Use:** Expand spatial output geometry with learned filters.

**Limits:** output_padding must be below stride or dilation per axis; detail quality is not guaranteed.

**Cost:** Learned kernel and output activations; output_padding adds no weights.

Source: `torch/nn/modules/conv.py`.

- `in_channels`: Must match the preceding tensor's C dimension.
- `out_channels`: C dimension produced by the learned filters.
- `kernel_size`: Spatial filter window. Any oddness requirement is specified in this component's metadata.
- `stride`: Spacing between window origins. Nullable pooling uses kernel size when null.
- `padding`: Padding applied to both sides of the spatial boundaries.
- `dilation`: Spacing between kernel samples.
- `bias`: Adds a learnable constant per output channel.
- `groups`: Applicable channel widths must be divisible by this value.
- `output_padding`: Resolves transposed-convolution output size. Must be smaller than stride or dilation per axis.

## channel_layer_norm

Normalizes only C at each spatial position. An internal movedim is reversed to preserve the public NCHW layout.

**Operation:** mean,var over C per (N,H,W)

**Use:** Normalize only channels at each spatial position.

**Limits:** This is not LayerNorm over all CHW; public NCHW layout is preserved.

**Cost:** Channel statistics and movedim; affine adds 2*C parameters.

Source: `hyso/models/components.py`.

- `channels`: Channel contract matching input C.
- `eps`: Positive denominator constant preventing division by zero.
- `affine`: Learns per-channel scale and shift after normalization.

## instance_norm

Computes H,W statistics independently per sample and channel. Running statistics are disabled by default. Spatial statistics require H*W>1.

**Operation:** mean,var over H,W per (N,C)

**Use:** Normalize each sample/channel with its own spatial statistics.

**Limits:** Computing statistics requires H*W>1; increasing batch does not resolve this.

**Cost:** N*C spatial statistics groups; affine adds 2*C parameters.

Source: `hyso/models/components.py`.

- `num_features`: Channel width expected by normalization.
- `eps`: Positive denominator constant preventing division by zero.
- `momentum`: Running-statistics update factor. Null uses cumulative averaging where supported.
- `affine`: Learns per-channel scale and shift after normalization.
- `track_running_stats`: Retains training statistics in buffers for eval.

## leaky_relu

Uses the selected nonnegative slope on the negative branch.

**Operation:** y=x if x>=0 else negative_slope*x

**Use:** Use an explicit fixed negative-branch slope.

**Limits:** negative_slope is not learned; PReLU provides a learned slope.

**Cost:** Parameter-free elementwise piecewise linear operation.

Source: `torch/nn/modules/activation.py`.

- `negative_slope`: Coefficient on Leaky ReLU's negative branch.

## prelu

Learns the negative slope globally or per channel. num_parameters must be 1 or C.

**Operation:** y=max(0,x)+a*min(0,x)

**Use:** Learn a shared or per-channel negative slope.

**Limits:** num_parameters must be 1 or input C; init must be representable in fp32.

**Cost:** 1 or C learned slopes and an elementwise activation.

Source: `torch/nn/modules/activation.py`.

- `num_parameters`: PReLU uses one shared slope or C per-channel slopes.
- `init`: Initial value of the learnable negative slope.

## gelu

Uses the Gaussian distribution function as a smooth gate. approximate selects exact or tanh computation.

**Operation:** y=x*Phi(x)

**Use:** Use a smooth Gaussian-distribution-based activation.

**Limits:** approximate=none and tanh are different numerical computations.

**Cost:** Parameter-free elementwise distribution function or tanh approximation.

Source: `torch/nn/modules/activation.py`.

- `approximate`: Selects GELU's none or tanh calculation path.

## mish

Applies a smooth non-monotonic activation through softplus and tanh.

**Operation:** y=x*tanh(softplus(x))

**Use:** Use a smooth softplus/tanh activation.

**Limits:** It is non-monotonic and its output is not a probability.

**Cost:** No parameters; per-element softplus, tanh and multiplication.

Source: `torch/nn/modules/activation.py`.


## hardsigmoid

Bounds output between zero and one. Saturated regions have zero derivative.

**Operation:** y=clamp(x+3,0,6)/6

**Use:** Use a bounded piecewise linear gate.

**Limits:** Gradients vanish in saturated tails; this is not a replacement for logits.

**Cost:** Parameter-free elementwise clamp.

Source: `torch/nn/modules/activation.py`.


## max_pool2d

Selects the maximum in each spatial window. Dilation and ceil_mode affect window geometry.

**Operation:** y[n,c,i,j]=max(window(n,c,i,j))

**Use:** Select strongest responses in local windows.

**Limits:** Padding cannot exceed half the kernel; ceil_mode affects the last window.

**Cost:** No learned parameters; maximum scan per window.

Source: `torch/nn/modules/pooling.py`.

- `kernel_size`: Spatial filter window. Any oddness requirement is specified in this component's metadata.
- `stride`: Spacing between window origins. Nullable pooling uses kernel size when null.
- `padding`: Padding applied to both sides of the spatial boundaries.
- `dilation`: Spacing between kernel samples.
- `ceil_mode`: Selects ceil rounding in pooling-window geometry.

## avg_pool2d

Averages each spatial window. count_include_pad controls whether padded zeros contribute to the divisor.

**Operation:** y=sum(window)/divisor

**Use:** Summarize local windows with their mean.

**Limits:** count_include_pad changes whether padded zeros enter the divisor.

**Cost:** No learned parameters; window sums and division.

Source: `torch/nn/modules/pooling.py`.

- `kernel_size`: Spatial filter window. Any oddness requirement is specified in this component's metadata.
- `stride`: Spacing between window origins. Nullable pooling uses kernel size when null.
- `padding`: Padding applied to both sides of the spatial boundaries.
- `ceil_mode`: Selects ceil rounding in pooling-window geometry.
- `count_include_pad`: Includes padded zeros in average-pooling divisors.

## adaptive_avg_pool2d

Computes average regions for an explicitly chosen positive output H,W grid.

**Operation:** output spatial shape = output_size

**Use:** Map variable input to an explicit H,W mean grid.

**Limits:** Grid size must be positive; resampling does not learn detail.

**Cost:** Parameter-free regional means; output storage depends on C*Hout*Wout.

Source: `torch/nn/modules/pooling.py`.

- `output_size`: Positive target H,W pair.

## adaptive_max_pool2d

Selects maxima from regions mapped to the chosen positive H,W grid.

**Operation:** output spatial shape = output_size

**Use:** Use regional maxima on a fixed H,W grid.

**Limits:** Mean information is not retained; only strongest responses are selected.

**Cost:** Parameter-free regional scans and the selected output grid.

Source: `torch/nn/modules/pooling.py`.

- `output_size`: Positive target H,W pair.

## global_avg_pool2d

Averages H and W and retains both spatial axes at size one.

**Operation:** [N,C,H,W] -> [N,C,1,1]

**Use:** Produce one spatial mean per channel.

**Limits:** Location information is lost; output is N,C,1,1 rather than NC.

**Cost:** Parameter-free H*W reduction with N*C output elements.

Source: `hyso/models/components.py`.


## bottleneck_residual

Uses 1x1 reduction, spatial convolution and 1x1 expansion in the main branch, added to the projected skip. The intermediate width is explicit.

**Operation:** y = P(x) + F(x); F = Conv1x1 -> ConvKxK -> Conv1x1

**Use:** Place spatial transformation between 1x1 reduction and expansion.

**Limits:** Main and skip shapes must match; this variant has no post-add activation.

**Cost:** Three main convolutions, norm/attention and an optional bias-free 1x1 projection.

Source: `hyso/models/advanced.py`.

- `in_channels`: Must match the preceding tensor's C dimension.
- `out_channels`: C dimension produced by the learned filters.
- `intermediate_channels`: Width of the internal tensor. Core validates additional dependencies.
- `kernel_size`: Spatial filter window. Any oddness requirement is specified in this component's metadata.
- `stride`: Spacing between window origins. Nullable pooling uses kernel size when null.
- `dilation`: Spacing between kernel samples.
- `normalization`: Normalization in the residual main branch.
- `activation`: Internal activation function.
- `attention`: Internal attention selector or supported name/params specification. Channel contracts must match.
- `dropout`: Finite internal dropout probability in 0<=p<1.
- `drop_path`: Sample-wise residual-branch drop probability, strictly below one.

## preactivation_residual

Places normalization and activation before main-branch convolutions. Core checks the skip and main-branch geometry.

**Operation:** F = Norm -> Act -> Conv; y = P(x) + F(x)

**Use:** Apply normalization and activation before convolutions.

**Limits:** The skip uses raw input; normalization statistics requirements still apply.

**Cost:** Two spatial convolutions and pre-normalizations; width/stride changes add a 1x1 projection.

Source: `hyso/models/advanced.py`.

- `in_channels`: Must match the preceding tensor's C dimension.
- `out_channels`: C dimension produced by the learned filters.
- `intermediate_channels`: Width of the internal tensor. Core validates additional dependencies.
- `kernel_size`: Spatial filter window. Any oddness requirement is specified in this component's metadata.
- `stride`: Spacing between window origins. Nullable pooling uses kernel size when null.
- `dilation`: Spacing between kernel samples.
- `normalization`: Normalization in the residual main branch.
- `activation`: Internal activation function.
- `attention`: Internal attention selector or supported name/params specification. Channel contracts must match.
- `dropout`: Finite internal dropout probability in 0<=p<1.
- `drop_path`: Sample-wise residual-branch drop probability, strictly below one.

## wide_residual_block

Exposes the intermediate width of two spatial convolutions. A wider intermediate tensor can increase parameter and activation requirements.

**Operation:** F = ConvKxK(Cin,Cmid) -> ConvKxK(Cmid,Cout)

**Use:** Directly choose the middle width of two spatial convolutions.

**Limits:** Increasing middle width does not by itself demonstrate an accuracy gain.

**Cost:** Spatial Cin*Cmid and Cmid*Cout filters; wider intermediate activations also consume storage.

Source: `hyso/models/advanced.py`.

- `in_channels`: Must match the preceding tensor's C dimension.
- `out_channels`: C dimension produced by the learned filters.
- `intermediate_channels`: Width of the internal tensor. Core validates additional dependencies.
- `kernel_size`: Spatial filter window. Any oddness requirement is specified in this component's metadata.
- `stride`: Spacing between window origins. Nullable pooling uses kernel size when null.
- `dilation`: Spacing between kernel samples.
- `normalization`: Normalization in the residual main branch.
- `activation`: Internal activation function.
- `attention`: Internal attention selector or supported name/params specification. Channel contracts must match.
- `dropout`: Finite internal dropout probability in 0<=p<1.
- `drop_path`: Sample-wise residual-branch drop probability, strictly below one.

## grouped_bottleneck

Groups the spatial convolution inside a bottleneck. The intermediate width must be divisible by the group count.

**Operation:** spatial parameters = Cmid*Cmid*kH*kW/groups

**Use:** Use grouped channel interaction in the bottleneck's middle convolution.

**Limits:** Middle width must divide by groups; outer 1x1 convolutions remain dense.

**Cost:** Middle filter Cmid*Cmid*k²/groups; outer projections and norms add costs.

Source: `hyso/models/advanced.py`.

- `in_channels`: Must match the preceding tensor's C dimension.
- `out_channels`: C dimension produced by the learned filters.
- `intermediate_channels`: Width of the internal tensor. Core validates additional dependencies.
- `kernel_size`: Spatial filter window. Any oddness requirement is specified in this component's metadata.
- `stride`: Spacing between window origins. Nullable pooling uses kernel size when null.
- `dilation`: Spacing between kernel samples.
- `normalization`: Normalization in the residual main branch.
- `activation`: Internal activation function.
- `attention`: Internal attention selector or supported name/params specification. Channel contracts must match.
- `dropout`: Finite internal dropout probability in 0<=p<1.
- `drop_path`: Sample-wise residual-branch drop probability, strictly below one.
- `groups`: Applicable channel widths must be divisible by this value.

## inverted_bottleneck

Expands channels, applies depthwise filtering and projects to output width. This HYSo block uses a 1x1 skip projection when widths or stride differ.

**Operation:** Cexpanded = Cin*expansion_ratio; y = P(x) + F(x)

**Use:** Expand channels before depthwise filtering.

**Limits:** Expanded width is Cin*expansion_ratio; this HYSo variant projects the skip on shape changes.

**Cost:** Expanded activations, two dense 1x1 convolutions and one depthwise kernel.

Source: `hyso/models/advanced.py`.

- `in_channels`: Must match the preceding tensor's C dimension.
- `out_channels`: C dimension produced by the learned filters.
- `expansion_ratio`: Multiplied by Cin to obtain inverted-bottleneck expanded width.
- `kernel_size`: Spatial filter window. Any oddness requirement is specified in this component's metadata.
- `stride`: Spacing between window origins. Nullable pooling uses kernel size when null.
- `dilation`: Spacing between kernel samples.
- `normalization`: Normalization in the residual main branch.
- `activation`: Internal activation function.
- `attention`: Internal attention selector or supported name/params specification. Channel contracts must match.
- `dropout`: Finite internal dropout probability in 0<=p<1.
- `drop_path`: Sample-wise residual-branch drop probability, strictly below one.

## gated_residual

Scales the residual branch with a scalar gate whose initial value is explicit. Gate learnability is separately configurable.

**Operation:** y = P(x) + gate*F(x)

**Use:** Control a two-convolution residual branch with an explicit scalar gate.

**Limits:** A zero gate can affect initial main-branch gradients; its value is not an efficacy measurement.

**Cost:** Two spatial convolutions gain one scalar parameter when learnable_gate=True.

Source: `hyso/models/advanced.py`.

- `in_channels`: Must match the preceding tensor's C dimension.
- `out_channels`: C dimension produced by the learned filters.
- `intermediate_channels`: Width of the internal tensor. Core validates additional dependencies.
- `kernel_size`: Spatial filter window. Any oddness requirement is specified in this component's metadata.
- `stride`: Spacing between window origins. Nullable pooling uses kernel size when null.
- `dilation`: Spacing between kernel samples.
- `normalization`: Normalization in the residual main branch.
- `activation`: Internal activation function.
- `attention`: Internal attention selector or supported name/params specification. Channel contracts must match.
- `dropout`: Finite internal dropout probability in 0<=p<1.
- `drop_path`: Sample-wise residual-branch drop probability, strictly below one.
- `gate_init`: Initial scalar gate on the residual branch.
- `learnable_gate`: Selects a learnable parameter or fixed buffer for the scalar gate.

## spatial_attention

Combines channel mean and max descriptors, then uses an odd convolution and sigmoid for spatial gating.

**Operation:** y = x * sigmoid(Conv(cat(meanC(x),maxC(x))))

**Use:** Produce position gates from channel mean/max descriptors.

**Limits:** The same position gate is shared by channels; it does not match global token pairs.

**Cost:** Bias-free 2*k² kernel weights and two spatial descriptors; no quadratic token matrix.

Source: `hyso/models/advanced.py`.

- `channels`: Channel contract matching input C.
- `kernel_size`: Spatial filter window. Any oddness requirement is specified in this component's metadata.

## channel_attention

Spatial mean and max descriptors share a 1x1 MLP. Their sum becomes a sigmoid channel gate.

**Operation:** y = x * sigmoid(MLP(avgHW(x)) + MLP(maxHW(x)))

**Use:** Process mean/max channel descriptors through a shared MLP.

**Limits:** Unlike SE it also uses max descriptors; gates do not vary by position.

**Cost:** Hidden width max(1,C//reduction); the same MLP runs twice with shared weights.

Source: `hyso/models/advanced.py`.

- `channels`: Channel contract matching input C.
- `reduction`: The channel descriptor hidden width is max(1, C//reduction). Increasing the ratio reduces the representation until the floor is reached.

## spatial_self_attention

Treats H*W positions as tokens and applies scaled QK attention to V. Limits are 1024 tokens and 4194304 attention-matrix elements.

**Operation:** Attention(Q,K,V) = softmax(QK^T/sqrt(d))V; T=H*W

**Use:** Use data-dependent interaction across all spatial positions.

**Limits:** H*W <=1024 and N*(H*W)² <=4194304; this constraint does not apply to SE.

**Cost:** QKV/projection weights and an N*T*T attention matrix; cost is quadratic in T=H*W.

Source: `hyso/models/advanced.py`.

- `channels`: Channel contract matching input C.
- `embedding_dim`: Embedding width. Attention uses channel width for null and multihead attention requires head divisibility.
- `dropout`: Finite internal dropout probability in 0<=p<1.

## multihead_spatial_attention

Splits embedding channels into independent attention heads. Embedding width must divide by head count. The matrix budget includes heads and batch size.

**Operation:** dhead=embedding_dim/num_heads; matrix_elements=N*num_heads*T*T

**Use:** Split spatial interaction across multiple attention heads.

**Limits:** embedding_dim must divide by head count; token and batch*head matrix budgets apply.

**Cost:** N*num_heads*T*T attention elements; more heads increase this intermediate matrix storage.

Source: `hyso/models/advanced.py`.

- `channels`: Channel contract matching input C.
- `embedding_dim`: Embedding width. Attention uses channel width for null and multihead attention requires head divisibility.
- `num_heads`: Number of parallel attention heads dividing embedding width.
- `dropout`: Finite internal dropout probability in 0<=p<1.

## dropout

Masks independent elements during training and rescales survivors. Eval is identity.

**Operation:** train y=x*Bernoulli(1-p)/(1-p); p=1 -> 0

**Use:** Mask independent feature elements during training.

**Limits:** Eval is identity; this Registry profile requires p<1.

**Cost:** No learned parameters; input-sized random mask during training.

Source: `torch/nn/modules/dropout.py`.

- `p`: Training-mask probability. Drop path has an exclusive upper bound of one.

## dropout2d

Masks entire spatial channel maps per sample during training. Eval is identity.

**Operation:** mask shape=[N,C,1,1]

**Use:** Mask complete channel maps during training.

**Limits:** This differs from elementwise Dropout; eval is identity.

**Cost:** No parameters; N*C channel masks broadcast spatially.

Source: `torch/nn/modules/dropout.py`.

- `p`: Training-mask probability. Drop path has an exclusive upper bound of one.

## drop_path

Drops a sample's whole path with a shared mask during training. This profile requires p<1. Eval is identity.

**Operation:** mask shape=[N,1,1,1]; y=x*mask/(1-p)

**Use:** Drop a whole branch per sample during training.

**Limits:** Requires p<1; it does not add a skip; eval is identity.

**Cost:** Parameter-free N,1,1,1 mask with 1/(1-p) scaling.

Source: `hyso/models/advanced.py`.

- `p`: Training-mask probability. Drop path has an exclusive upper bound of one.

## gaussian_noise

Adds independent zero-mean Gaussian noise during training. Eval is identity.

**Operation:** train y=x+std*Normal(0,1)

**Use:** Add training noise with explicit standard deviation.

**Limits:** This is not masking; eval is identity; std controls noise amplitude.

**Cost:** No parameters; input-sized normal random sampling.

Source: `hyso/models/advanced.py`.

- `std`: Scale of zero-mean noise added during training.

## resize

Choose exactly one target size or scale pair. nearest, bilinear and bicubic follow Core rules. Scaled dimensions use floor.

**Operation:** Hout=floor(H*scaleH); Wout=floor(W*scaleW)

**Use:** Match branch spatial dimensions to an explicit target.

**Limits:** Choose output_size or scale_factor; mode/align_corners must be compatible.

**Cost:** No learned parameters; cost depends on interpolation mode and output grid.

Source: `hyso/models/sampling.py`.

- `output_size`: Positive target H,W pair.
- `scale_factor`: Positive finite H,W scales. Cannot be combined with output_size.
- `mode`: Sampling method for resizing.
- `align_corners`: Controls bilinear/bicubic coordinate mapping. Must be null for nearest.

## pixel_shuffle

Moves channel elements into spatial positions. C must be divisible by the upscale factor squared.

**Operation:** [N,C,H,W] -> [N,C/r^2,H*r,W*r]

**Use:** Rearrange channel elements into spatial upscaling.

**Limits:** C must divide by upscale_factor²; it learns no filter.

**Cost:** Parameter-free element rearrangement preserving per-sample element count.

Source: `torch/nn/modules/pixelshuffle.py`.

- `upscale_factor`: Positive integer expansion for pixel shuffle.

## pixel_unshuffle

Moves spatial elements into channels. H and W must divide by the downscale factor.

**Operation:** [N,C,H,W] -> [N,C*r^2,H/r,W/r]

**Use:** Move spatial resolution into channel width.

**Limits:** H and W must divide by downscale_factor; channels grow by its square.

**Cost:** Parameter-free inverse pixel rearrangement preserving per-sample element count.

Source: `torch/nn/modules/pixelshuffle.py`.

- `downscale_factor`: Positive integer dividing H and W for pixel unshuffle.

## flatten

Combines CHW into a single feature axis while preserving batch N.

**Operation:** [N,C,H,W] -> [N,C*H*W]

**Use:** Convert image features into an NC vector.

**Limits:** Schema2 allows it only at the final feature boundary; graph consumers must support NC.

**Cost:** Parameter-free flattening; a following dense head may scale with C*H*W.

Source: `hyso/models/sampling.py`.


## permute

This profile keeps N and C fixed. H and W may exchange places.

**Operation:** dims in {[0,1,2,3],[0,1,3,2]}

**Use:** Explicitly exchange H and W axes.

**Limits:** This profile fixes N and C; it is not an arbitrary axis permutation.

**Cost:** Parameter-free axis view; later operations may require a contiguous copy.

Source: `hyso/models/sampling.py`.

- `dims`: Allowed permutation preserving batch and channel axes.

## reshape

Reinterprets each sample as an explicit positive CHW shape. Element count per sample must be preserved.

**Operation:** Cin*Hin*Win = Cout*Hout*Wout

**Use:** Reinterpret per-sample elements in another positive CHW layout.

**Limits:** CHW product must match; batch is preserved and no semantic mixing is learned.

**Cost:** Parameter-free reshape; incompatible strides may require a copy.

Source: `hyso/models/sampling.py`.

- `shape`: Three positive integers preserving per-sample element count.

## classification_head

Uses pooling or flat features, optional dropout and linear logits. It applies no output softmax.

**Operation:** logits = Linear(Dropout(Pool(x))); shape=[N,num_classes]

**Use:** Produce per-class logits or one score for a binary task.

**Limits:** No softmax; NC input requires pooling=none and matching in_features.

**Cost:** (in_features+1)*num_classes parameters; pooling=none uses CHW image features.

Source: `hyso/models/sampling.py`.

- `in_features`: Must equal pooled channel width or flat NC feature width.
- `pooling`: Selects avg, max or the supported flat-input path for the head.
- `num_classes`: Logit width. Binary tasks need one, multiclass needs at least two.
- `dropout`: Finite internal dropout probability in 0<=p<1.

## regression_head

Produces unrestricted linear regression outputs from pooled or flat features. The application must choose loss and target units.

**Operation:** y=Linear(Pool(x)); shape=[N,output_features]

**Use:** Produce unrestricted linear outputs for continuous targets.

**Limits:** The short Studio TrainingSession workflow lacks regression loss support; Core build is available.

**Cost:** (in_features+1)*output_features parameters and selected pooling.

Source: `hyso/models/sampling.py`.

- `in_features`: Must equal pooled channel width or flat NC feature width.
- `pooling`: Selects avg, max or the supported flat-input path for the head.
- `output_features`: Width of the regression output.

## embedding_head

Produces a linear embedding. Optional L2 normalization uses epsilon to avoid division by zero.

**Operation:** z=Linear(Pool(x)); normalized z=z/max(norm2(z),eps)

**Use:** Produce fixed-width embeddings with optional L2 normalization.

**Limits:** Short Studio training/evaluation does not support embedding tasks; epsilon bounds zero norms.

**Cost:** (in_features+1)*embedding_dim parameters; normalization computes a vector norm.

Source: `hyso/models/sampling.py`.

- `in_features`: Must equal pooled channel width or flat NC feature width.
- `pooling`: Selects avg, max or the supported flat-input path for the head.
- `embedding_dim`: Embedding width. Attention uses channel width for null and multihead attention requires head divisibility.
- `normalize`: Applies epsilon-protected L2 normalization to the embedding.
- `eps`: Positive denominator constant preventing division by zero.

## add

Adds two to eight inputs with identical shape, rank, batch, dtype and device. Channel or spatial conversion requires separate projection/resize nodes.

**Operation:** y=sum_i x_i; no broadcasting

**Use:** Sum equal-shaped branches or external skips.

**Limits:** 2–8 inputs; no broadcasting, hidden projection or resize.

**Cost:** No parameters; elementwise addition for each extra input.

Source: `hyso/models/graph_ops.py`.


## concat

Concatenates two to eight inputs along a selected non-batch axis. Other dimensions must match. NC permits only axis one.

**Operation:** shape[axis]=sum_i shape_i[axis]

**Use:** Join branches along one axis while retaining their features.

**Limits:** Batch axis is forbidden; other dimensions must match; NC supports only axis=1.

**Cost:** No parameters; output storage follows the sum along the selected axis.

Source: `hyso/models/graph_ops.py`.

- `axis`: Non-batch tensor axis. One for NC, one through three for NCHW.

## multiply

Multiplies two to eight equal-shaped inputs elementwise. Broadcasting is forbidden.

**Operation:** y=product_i x_i

**Use:** Apply elementwise gating between equal-shaped branches.

**Limits:** 2–8 inputs; no broadcasting; small factors can attenuate values and gradients.

**Cost:** No parameters; elementwise multiplication per extra input.

Source: `hyso/models/graph_ops.py`.


## weighted_sum

Sums ordered inputs multiplied by raw weights. There is no softmax or sum-to-one normalization. Weights are fixed buffers or learnable parameters.

**Operation:** y=sum_i w_i*x_i; 2<=len(weights)=len(inputs)<=8; abs(w_i)<=1e6

**Use:** Sum ordered branches with explicit fixed or learned coefficients.

**Limits:** 2–8 finite raw weights, |w|<=1e6; no softmax or automatic normalization.

**Cost:** One parameter per input when learnable=True; otherwise weights are buffers.

Source: `hyso/models/graph_ops.py`.

- `weights`: One finite raw coefficient per ordered input. No automatic normalization is applied.
- `learnable`: Learns raw coefficients as parameters. Otherwise retains them as buffers.
