# Mobile, edge and reparameterizable CNN architectures

Stage 13 adds seven usable model families and ten independent operators.
FasterNet and PartialConv are excluded following the source-license audit.
Read the [license audit](../reports/stage13-license-audit.md) and
[retained notices](../../THIRD_PARTY_NOTICES.md) before redistribution.
Patent freedom to operate and commercial deployment clearance remain unverified.

## Model selection

```python
from hyso import ModelBuilder
from hyso.models import MobileNetV2, MobileNetV3, EfficientNetV2

model = MobileNetV2(input_channels=1, num_classes=3, width_mult=.5)
small = MobileNetV3(variant='small', input_channels=1, num_classes=3)
large = MobileNetV3(variant='large', input_channels=1, num_classes=3)
efficient = EfficientNetV2(variant='s', input_channels=1, num_classes=3,
                           width_mult=.125, depth_mult=.1)
builder = ModelBuilder.named('mobilenet_v2', input_channels=1,
                             num_classes=3, width_mult=.5)
assert model.to_config() == builder.to_config()
```

| Registry name | Public class | Supported canonical variants | Default |
|---|---|---|---|
| `mobilenet_v2` | MobileNetV2 | Standard width 1 | width 1, dropout .2 |
| `mobilenet_v3` | MobileNetV3 | `small`, `large` | large, width 1, dropout .2 |
| `efficientnet_v2` | EfficientNetV2 | `s`, `m`, `l` | s, width/depth 1, stochastic depth .2 |
| `ghostnet` | GhostNet | GhostNetV1 width 1 | width 1, dropout .2 |
| `shufflenet_v2` | ShuffleNetV2 | `x0_5`, `x1_0`, `x1_5`, `x2_0` | x1_0 |
| `repvgg` | RepVGG | `a0`, `a1` | a0, train topology |
| `mobileone` | MobileOne | `s0`, `s1`, `s2` | s0, train topology |

All families expose `input_channels=3`, `num_classes=1000`. Counts are positive
integers in 1..8192. Output is `(N, num_classes)` after adaptive global pooling.
Width multipliers range from .125 to 2 except ShuffleNetV2, which uses four
discrete widths. EfficientNetV2, RepVGG and MobileOne accept `depth_mult` in
[.1, 1]. Scaled models are explicit derived configurations. Standard settings
preserve the named topology. A 32×32 input supports the bounded smoke protocols.
BatchNorm training needs sufficient batch/spatial observations, including batch
size two at final 1×1 feature maps.

MobileNetV2 preserves the `(expansion, channels, repeats, stride)` sequence
`(1,16,1,1), (6,24,2,2), (6,32,3,2), (6,64,4,2), (6,96,3,1),
(6,160,3,2), (6,320,1,1)`. Channels use divisor-eight rounding. The head width is
`make_divisible(1280*max(1,width),8)`. MobileNetV3 Small/Large use distinct
torchvision layouts with the original SE, HardSigmoid, HardSwish and ReLU
selections, classifier widths and normalization.

EfficientNetV2 reuses torchvision S/M/L FusedMBConv and MBConv stage records.
The repeats are S `[2,4,4,6,9,15]`, M `[3,5,5,7,14,18,5]`, L
`[4,7,7,10,19,25,7]`. BatchNorm epsilon is .001, the classifier feature width
is 1280, and dropout defaults to .2/.3/.4 for S/M/L when `None`. Width scaling
rounds channels to eight. Depth scaling uses `ceil(repeats*depth)` with minimum
one. Stochastic depth increases over the block sequence.

GhostNetV1 has 16 Ghost bottlenecks, divisor-four channel rounding, depthwise
downsample shortcuts and SE on the specified source blocks. Its head uses a
width-scaled 960-channel convolution before pooling and a 1280-channel
convolution after pooling. HYSo composes torchvision Conv/BN/activation and
hard-gated SE operators. HYSo state keys differ from the author's implementation.
GhostNetV2/V3 and attention extensions are outside the implemented family.

ShuffleNetV2 retains `[4,8,4]` stage repeats and canonical channel tables. Its
stride-one unit splits channels, transforms the second half, concatenates and
shuffles. Stride-two units transform both branches before concatenation.

RepVGG A0/A1 use `[2,4,14,1]` depths. Width multipliers over
`[64,128,256,512]` are A0 `[.75,.75,.75,2.5]` and A1 `[1,1,1,2.5]`.
The stem is `min(64, first stage width)`.
MobileOne uses `[2,8,10,1]` depthwise/pointwise pairs. Multipliers are
s0 `[.75,1,1,2]`, s1 `[1.5,1.5,2,2.5]`, s2 `[1.5,2,2.5,4]`.
s0 stage blocks have four convolution branches. s1/s2 use one. All three use
a single-branch stem. These variants omit the SE placement of the author's s4.
Standalone MobileOneBlock optionally uses existing HYSo sigmoid SE with
reduction 16 and a minimum hidden width of one.

## Operator distinctions

| Operator | Computation and channel behavior |
|---|---|
| Standard convolution | Mixes input channels in each group across the spatial kernel. |
| Depthwise convolution | `groups=in_channels` gives a separate spatial filter per channel, with an optional channel multiplier. |
| Pointwise convolution | A 1×1 kernel mixes channels at each spatial location. |
| Inverted residual | Expansion, depthwise spatial mixing and linear projection. Addition requires equal channels and stride one. MobileNetV2 uses ReLU6 before projection. |
| MBConv | Expansion, depthwise convolution, SE and linear projection with SiLU and optional row stochastic depth. |
| FusedMBConv | Regular spatial convolution for expansion and optional pointwise projection. The standalone adapter has no SE. |
| Ghost features | Primary learned features are extended by cheap grouped spatial features, concatenated and sliced to the output width. |
| Channel shuffle | Reshape/transpose exchanges channel-group membership without learned parameters. |
| Partial convolution | Spatial convolution processes a subset of channels. The consulted implementation is excluded by the license decision. |
| Structural reparameterization | Parallel linear Conv/BN branches are fused using eval running statistics. Activation and SE retain their positions after the sum. |

Hardware speed, FPS, trained accuracy and power are NOT_EVALUATED.

## Independent blocks and Builder

All ten included blocks have pure Builder shape and parameter contracts.
Registry contains the previous 27 components plus seven families and ten blocks,
for 44 components.

| Registry identifier | Parameters beyond channel counts |
|---|---|
| `inverted_residual` | stride 1/2, expand_ratio [1,16], default 6 |
| `mbconv` | stride, expand_ratio default 6, odd kernel 1/3/5/7, stochastic_depth [0,1) |
| `fused_mbconv` | Same, expand_ratio default 4 |
| `ghost_conv` | stride, kernel, ratio 2..8, cheap_kernel, relu bool |
| `ghost_bottleneck` | hidden_channels, stride, kernel, se_ratio [0,1] |
| `channel_shuffle` | channels, groups dividing channels |
| `shuffle_unit` | stride 1/2, even output channels, equal input/output at stride one |
| `repvgg_block` | stride, groups dividing both widths, deploy bool |
| `mobileone_block` | stride, kernel, groups, num_conv_branches 1..8, use_se, deploy |
| `dropblock2d` | p [0,1], positive block_size |

```python
from hyso import ModelBuilder
model = (ModelBuilder((4, 16, 16), 3)
         .add('mbconv', in_channels=4, out_channels=8, expand_ratio=2)
         .add('ghost_conv', in_channels=8, out_channels=8)
         .add('dropblock2d', p=.1, block_size=3).build())
```

DropBlock is explicit. Training samples valid top-left square-block origins,
clips block size to the spatial extent and normalizes by retained mask elements
across the tensor. Small and even spatial dimensions preserve shape. An
all-dropped mask and `p=1` produce finite zeros. `p=0` and eval return the input.
Half/bfloat16 mask application accumulates in float32 before restoring dtype.

## Deployment conversion

```python
from hyso import ModelBuilder
from hyso.models import reparameterize_model

training = ModelBuilder.named('repvgg', width_mult=.125, depth_mult=.1,
                              input_channels=1, num_classes=3).build().eval()
# Load trained HYSo train-topology weights explicitly before conversion.
deployed = reparameterize_model(training)  # copy, source preserved
# training.switch_to_deploy() performs explicit in-place conversion.
config = deployed.to_config()             # params.deploy is now True
fresh = ModelBuilder.from_config(config).build().eval()
fresh.load_state_dict(deployed.state_dict(), strict=True)
```

For each output channel, fusion computes `scale=gamma/sqrt(running_var+epsilon)`,
`W_fused=W*scale` and `b_fused=beta+(b-running_mean)*scale`. Grouped identity
uses a center impulse at `output_index % input_channels_per_group`. A 1×1 branch
is zero-padded to the spatial kernel. Fused kernels and biases are summed.
RepVGG has one 3×3 and one 1×1 branch. MobileOne has K spatial branches,
an optional scale branch for kernels larger than one and optional identity BN.
Its pointwise blocks have no separate scale branch.

Conversion requires the complete graph in eval mode, tracked finite BN state,
positive epsilon, nonnegative variance and supported branch geometry. Float32
and float64 preserve dtype/device. Other conversion dtypes raise an error.
Verification covers CPU float32/float64. All block fusion plans finish before
graph mutation. Repeated conversion is idempotent. Recreate optimizers for the
new parameters if further training of the converted graph is explicitly intended.

Training state keys contain `branches`, `scale` and optional `identity` BN.
Deploy keys contain `reparam.weight` and `reparam.bias`, alongside unchanged
activation/SE/classifier state. Save deploy architecture JSON with deploy weights.
Family loaders reject cross-phase state even with `strict=False` and check every
block before copying family weights. PyTorch errors still apply to other missing
keys and shapes. Generic enclosing modules retain PyTorch's ordinary partial-load
semantics for unrelated invalid entries.

Tolerances are float32 `atol=2e-6, rtol=2e-5` and float64
`atol=3e-12, rtol=3e-12`. Observed absolute and norm-relative errors are recorded
in the validation report. Equivalence concerns eval outputs using the current
running statistics.

## JSON, training and CLI

`hyso.model_config` schema1 remains architecture-only. The closed
`registered_model` whitelist accepts seven families. Validation and summary use
scalar parameters and source configuration records without constructing a model,
executing a forward, downloading weights or querying CUDA. Named models have a
conservative 200-million-parameter limit. Sequential chains retain the 10-million
limit. Runtime activation memory and available RAM require separate evaluation.

```console
hyso models list --json
hyso models show mobileone --json
hyso models validate architecture.json --json
hyso models summary architecture.json --json
hyso models inspect architecture.json --execute --json
```

Explicit inspect constructs a CPU model and performs no forward or conversion.
CLI exit codes and JSON envelopes retain their existing contracts.

TrainingSession accepts these models as external nn.Modules. Validation uses
one two-sample CPU optimizer step per family. Match input/model dtype and device.
Automatic `save_last_to` / `resume_from` remains scoped to ModularCNN. Save
architecture JSON and explicit strict `state_dict` weights separately for the
new families. Weights omit optimizer, RNG, loader and epoch-resume state.
Examples use `torch.load(..., weights_only=True)`.

Canonical torchvision state-key and eval equivalence is tested for representative
MobileNetV2, MobileNetV3 Small, EfficientNetV2-S and ShuffleNetV2 x0.5 wrappers.
Changed channels, classes, widths, depths and deploy phases require matching
weights. GhostNet, RepVGG and MobileOne use HYSo-specific state keys.
Pretrained adaptation, full-model pickles, ONNX/TorchScript export, mobile runtime
deployment and hardware benchmarks remain unverified.

## References and examples

Canonical wrappers reuse the existing torchvision dependency. Private
configuration helpers remain a torchvision version compatibility boundary.
[MobileNetV2](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/mobilenetv2.py),
[MobileNetV3](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/mobilenetv3.py),
[EfficientNet](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/efficientnet.py),
[ShuffleNetV2](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/shufflenetv2.py).

Additional topologies follow the identified author implementations.
[GhostNetV1](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/f90e129b645c3b1684fe07cd361cd557d0ad71f7/ghostnet_pytorch),
[RepVGG](https://github.com/DingXiaoH/RepVGG/tree/eae7c5204001eaf195bbe2ee72fb6a37855cce33),
[MobileOne](https://github.com/apple/ml-mobileone/tree/3cebdf04a1524c447ab5e44a5235524f8da46f8d).

The excluded source is recorded for audit traceability.
[FasterNet](https://github.com/JierunChen/FasterNet/tree/25b5dc75d047d61878dc59061d31ef4608507950).

Runnable commands and expected outputs are in the
[examples README](../../examples/models/README.md).
