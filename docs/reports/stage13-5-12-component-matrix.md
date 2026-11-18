# 13.5.12 — Component and combination matrix

Observed source development Registry: **85** entries. Installed wheel/sdist commercial Registry: **57** entries.
Four Graph fusion operators are separate from both catalogs. Runtime module count remains **164**.
The full parameter/default/bounds/constructor metadata snapshot is in [the audited fixture](../../tests/fixtures/builder-graph-compatibility-v1.json).
Detailed measurements and source-class discovery are in QA/full-inventory.json. Counts below are for the stated small fixture, not constants of a component family.

## Registered components

| Name | Actual constructor | Commercial | Direct construction | Sequential Builder | Graph node | Graph v2 CPU profile | Fixture parameters / trainable | Output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conv_norm_act | hyso.models.blocks.ConvNormAct | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | SUPPORTED conditional; typed admission PASS | 152 / 152 | [2, 4, 8, 8] |
| conv2d | torch.nn.modules.conv.Conv2d | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | SUPPORTED conditional; typed admission PASS | 148 / 148 | [2, 4, 8, 8] |
| residual | hyso.models.blocks.ResidualBlock | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 304 / 304 | [2, 4, 8, 8] |
| se | hyso.models.WithSE.SEBlock | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 13 / 13 | [2, 4, 8, 8] |
| cbam | hyso.models.bcam.CBAMBlock | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 112 / 112 | [2, 4, 8, 8] |
| eca | hyso.models.attention.ECAAttention | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 3 / 3 | [2, 4, 8, 8] |
| coordinate | hyso.models.attention.CoordinateAttention | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 128 / 128 | [2, 4, 8, 8] |
| batch_norm | torch.nn.modules.batchnorm.BatchNorm2d | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | SUPPORTED conditional; typed admission PASS | 8 / 8 | [2, 4, 8, 8] |
| group_norm | torch.nn.modules.normalization.GroupNorm | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 8 / 8 | [2, 4, 8, 8] |
| relu | torch.nn.modules.activation.ReLU | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | SUPPORTED conditional; typed admission PASS | 0 / 0 | [2, 4, 8, 8] |
| silu | torch.nn.modules.activation.SiLU | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | SUPPORTED conditional; typed admission PASS | 0 / 0 | [2, 4, 8, 8] |
| hardswish | torch.nn.modules.activation.Hardswish | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 8, 8] |
| identity | torch.nn.modules.linear.Identity | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | SUPPORTED conditional; typed admission PASS | 0 / 0 | [2, 4, 8, 8] |
| modular_cnn | hyso.models.modular.ModularCNN | yes | Catalogued constructor; whole-family execution NOT_TESTED here | SUPPORTED fixed modular()/named() profile; no add() feature insertion | UNSUPPORTED complete model node | UNSUPPORTED | NOT_TESTED / variable | family-specific |
| channel_shuffle | hyso.models.regularization.ChannelShuffle | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 8, 8] |
| dropblock2d | hyso.models.regularization.DropBlock2D | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 8, 8] |
| pointwise_conv2d | hyso.models.components.PointwiseConv2D | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | SUPPORTED conditional; typed admission PASS | 20 / 20 | [2, 4, 8, 8] |
| depthwise_conv2d | hyso.models.components.DepthwiseConv2D | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 40 / 40 | [2, 4, 8, 8] |
| depthwise_separable_conv2d | hyso.models.components.DepthwiseSeparableConv2D | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 60 / 60 | [2, 4, 8, 8] |
| spatial_separable_conv2d | hyso.models.components.SpatialSeparableConv2D | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 104 / 104 | [2, 4, 8, 8] |
| factorized_conv2d | hyso.models.components.FactorizedConv2D | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 168 / 168 | [2, 4, 8, 8] |
| conv_transpose2d | torch.nn.modules.conv.ConvTranspose2d | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 148 / 148 | [2, 4, 8, 8] |
| channel_layer_norm | hyso.models.components.ChannelLayerNorm | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 8 / 8 | [2, 4, 8, 8] |
| instance_norm | hyso.models.components.InstanceNorm2D | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 8, 8] |
| leaky_relu | torch.nn.modules.activation.LeakyReLU | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 8, 8] |
| prelu | torch.nn.modules.activation.PReLU | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 1 / 1 | [2, 4, 8, 8] |
| gelu | torch.nn.modules.activation.GELU | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 8, 8] |
| mish | torch.nn.modules.activation.Mish | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 8, 8] |
| hardsigmoid | torch.nn.modules.activation.Hardsigmoid | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 8, 8] |
| max_pool2d | torch.nn.modules.pooling.MaxPool2d | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 4, 4] |
| avg_pool2d | torch.nn.modules.pooling.AvgPool2d | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 4, 4] |
| adaptive_avg_pool2d | torch.nn.modules.pooling.AdaptiveAvgPool2d | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 4, 4] |
| adaptive_max_pool2d | torch.nn.modules.pooling.AdaptiveMaxPool2d | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 4, 4] |
| global_avg_pool2d | hyso.models.components.GlobalAveragePool2D | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | SUPPORTED conditional; typed admission PASS | 0 / 0 | [2, 4, 1, 1] |
| bottleneck_residual | hyso.models.advanced.AdvancedResidual | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 200 / 200 | [2, 4, 8, 8] |
| preactivation_residual | hyso.models.advanced.PreActivationResidual | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 304 / 304 | [2, 4, 8, 8] |
| wide_residual_block | hyso.models.advanced.WideResidual | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 304 / 304 | [2, 4, 8, 8] |
| grouped_bottleneck | hyso.models.advanced.GroupedBottleneck | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 200 / 200 | [2, 4, 8, 8] |
| inverted_bottleneck | hyso.models.advanced.InvertedBottleneck | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 176 / 176 | [2, 4, 8, 8] |
| gated_residual | hyso.models.advanced.GatedResidual | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 305 / 305 | [2, 4, 8, 8] |
| spatial_attention | hyso.models.advanced.SpatialAttention | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 98 / 98 | [2, 4, 8, 8] |
| channel_attention | hyso.models.advanced.ChannelAttention | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 13 / 13 | [2, 4, 8, 8] |
| spatial_self_attention | hyso.models.advanced.SpatialSelfAttention | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 80 / 80 | [2, 4, 8, 8] |
| multihead_spatial_attention | hyso.models.advanced.MultiHeadSpatialAttention | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 80 / 80 | [2, 4, 8, 8] |
| dropout | torch.nn.modules.dropout.Dropout | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 8, 8] |
| dropout2d | torch.nn.modules.dropout.Dropout2d | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 8, 8] |
| drop_path | hyso.models.advanced.DropPath | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 8, 8] |
| gaussian_noise | hyso.models.advanced.GaussianNoise | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 8, 8] |
| resize | hyso.models.sampling.Resize | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 4, 4] |
| pixel_shuffle | torch.nn.modules.pixelshuffle.PixelShuffle | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 1, 16, 16] |
| pixel_unshuffle | torch.nn.modules.pixelshuffle.PixelUnshuffle | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 16, 4, 4] |
| flatten | hyso.models.sampling.Flatten | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED terminal schema2 feature | SUPPORTED unary node | SUPPORTED conditional; typed admission PASS | 0 / 0 | [2, 256] |
| permute | hyso.models.sampling.Permute | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 8, 8] |
| reshape | hyso.models.sampling.Reshape | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 8, 8] |
| classification_head | hyso.models.sampling.PredictionHead | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED terminal head | SUPPORTED unary node | SUPPORTED conditional; typed admission PASS | 15 / 15 | [2, 3] |
| regression_head | hyso.models.sampling.RegressionHead | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED terminal head | SUPPORTED unary node | UNSUPPORTED | 15 / 15 | [2, 3] |
| embedding_head | hyso.models.sampling.EmbeddingHead | yes | SUPPORTED fixture PASS via Registry.create | SUPPORTED terminal head | SUPPORTED unary node | UNSUPPORTED | 40 / 40 | [2, 8] |
| simam | hyso.models.m3_attention.SimAM | no (development) | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 0 / 0 | [2, 4, 8, 8] |
| triplet | hyso.models.m3_attention.TripletAttention | no (development) | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 300 / 300 | [2, 4, 8, 8] |
| selective_kernel | hyso.models.m3_attention.SelectiveKernelAttention | no (development) | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 728 / 728 | [2, 4, 8, 8] |
| basic_residual | hyso.models.m3_residual.BasicResidualBlock | no (development) | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 304 / 304 | [2, 4, 8, 8] |
| bottleneck | hyso.models.m3_residual.Bottleneck | no (development) | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 29 / 29 | [2, 4, 8, 8] |
| preactivation | hyso.models.m3_residual.PreActivationBlock | no (development) | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 304 / 304 | [2, 4, 8, 8] |
| wide_residual | hyso.models.m3_residual.WideResidualBlock | no (development) | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 304 / 304 | [2, 4, 8, 8] |
| resnext_bottleneck | hyso.models.m3_residual.ResNeXtBottleneck | no (development) | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 200 / 200 | [2, 8, 8, 8] |
| resnet | hyso.models.architectures.ResNet | no (development) | Catalogued constructor; whole-family execution NOT_TESTED here | SUPPORTED fixed modular()/named() profile; no add() feature insertion | UNSUPPORTED complete model node | UNSUPPORTED | NOT_TESTED / variable | family-specific |
| preact_resnet | hyso.models.architectures.PreActResNet | no (development) | Catalogued constructor; whole-family execution NOT_TESTED here | SUPPORTED fixed modular()/named() profile; no add() feature insertion | UNSUPPORTED complete model node | UNSUPPORTED | NOT_TESTED / variable | family-specific |
| wide_resnet | hyso.models.architectures.WideResNet | no (development) | Catalogued constructor; whole-family execution NOT_TESTED here | SUPPORTED fixed modular()/named() profile; no add() feature insertion | UNSUPPORTED complete model node | UNSUPPORTED | NOT_TESTED / variable | family-specific |
| resnext | hyso.models.architectures.ResNeXt | no (development) | Catalogued constructor; whole-family execution NOT_TESTED here | SUPPORTED fixed modular()/named() profile; no add() feature insertion | UNSUPPORTED complete model node | UNSUPPORTED | NOT_TESTED / variable | family-specific |
| densenet | hyso.models.architectures.DenseNet | no (development) | Catalogued constructor; whole-family execution NOT_TESTED here | SUPPORTED fixed modular()/named() profile; no add() feature insertion | UNSUPPORTED complete model node | UNSUPPORTED | NOT_TESTED / variable | family-specific |
| inverted_residual | hyso.models.m4_blocks.InvertedResidual | no (development) | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 512 / 512 | [2, 4, 8, 8] |
| mbconv | hyso.models.m4_blocks.MBConv | no (development) | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 585 / 585 | [2, 4, 8, 8] |
| fused_mbconv | hyso.models.m4_blocks.FusedMBConv | no (development) | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 680 / 680 | [2, 4, 8, 8] |
| ghost_conv | hyso.models.m4_blocks.GhostConv | no (development) | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 34 / 34 | [2, 4, 8, 8] |
| ghost_bottleneck | hyso.models.m4_blocks.GhostBottleneck | no (development) | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 68 / 68 | [2, 4, 8, 8] |
| shuffle_unit | hyso.models.m4_blocks.ShuffleUnit | no (development) | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 38 / 38 | [2, 4, 8, 8] |
| repvgg_block | hyso.models.m4_reparam.RepVGGBlock | no (development) | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 184 / 184 | [2, 4, 8, 8] |
| mobileone_block | hyso.models.m4_reparam.MobileOneBlock | no (development) | SUPPORTED fixture PASS via Registry.create | SUPPORTED feature block | SUPPORTED unary node | UNSUPPORTED | 184 / 184 | [2, 4, 8, 8] |
| mobilenet_v2 | hyso.models.mobile_architectures.MobileNetV2 | no (development) | Catalogued constructor; whole-family execution NOT_TESTED here | SUPPORTED fixed modular()/named() profile; no add() feature insertion | UNSUPPORTED complete model node | UNSUPPORTED | NOT_TESTED / variable | family-specific |
| mobilenet_v3 | hyso.models.mobile_architectures.MobileNetV3 | no (development) | Catalogued constructor; whole-family execution NOT_TESTED here | SUPPORTED fixed modular()/named() profile; no add() feature insertion | UNSUPPORTED complete model node | UNSUPPORTED | NOT_TESTED / variable | family-specific |
| efficientnet_v2 | hyso.models.mobile_architectures.EfficientNetV2 | no (development) | Catalogued constructor; whole-family execution NOT_TESTED here | SUPPORTED fixed modular()/named() profile; no add() feature insertion | UNSUPPORTED complete model node | UNSUPPORTED | NOT_TESTED / variable | family-specific |
| ghostnet | hyso.models.mobile_architectures.GhostNet | no (development) | Catalogued constructor; whole-family execution NOT_TESTED here | SUPPORTED fixed modular()/named() profile; no add() feature insertion | UNSUPPORTED complete model node | UNSUPPORTED | NOT_TESTED / variable | family-specific |
| shufflenet_v2 | hyso.models.mobile_architectures.ShuffleNetV2 | no (development) | Catalogued constructor; whole-family execution NOT_TESTED here | SUPPORTED fixed modular()/named() profile; no add() feature insertion | UNSUPPORTED complete model node | UNSUPPORTED | NOT_TESTED / variable | family-specific |
| repvgg | hyso.models.reparam_architectures.RepVGG | no (development) | Catalogued constructor; whole-family execution NOT_TESTED here | SUPPORTED fixed modular()/named() profile; no add() feature insertion | UNSUPPORTED complete model node | UNSUPPORTED | NOT_TESTED / variable | family-specific |
| mobileone | hyso.models.reparam_architectures.MobileOne | no (development) | Catalogued constructor; whole-family execution NOT_TESTED here | SUPPORTED fixed modular()/named() profile; no add() feature insertion | UNSUPPORTED complete model node | UNSUPPORTED | NOT_TESTED / variable | family-specific |

Direct construction above invokes the real registered constructor through Registry.create; it does not certify every optional argument of a bare Torch constructor. The positive execution columns describe the source profile. Commercial absence rejects regardless of source support.

Graph v2 is a separate closed typed contract: only the 10 listed Registry names and the four fusion operations are eligible, subject to little-endian CPU fp32/fp64, exact types/constructor state, no hooks/custom callables/aliasing and a conservative 1M reference parameter budget. All 10 concrete fixtures and four existing fusion topology fixtures passed typed contract admission during closure; every other concrete component rejected. This is not per-component receiver VERIFIED. Actual source-to-receiver verification remains the explicit supported branching example and receiver regressions. Details: QA/graph-v2-component-proof.json.

## Parameters and observed contracts

For each component below, the signature is the actual constructor signature; Registry parameters are the smaller JSON whitelist. Standalone Torch constructors may accept more arguments. Device/dtype are not JSON constructor options.
All 72 non-model fixture rows ran through Registry, sequential and unary Graph, with real finite forward/backward, input gradients, used parameter gradients, an optimizer update when parameters exist, architecture/state serialization and strict checkpoint restore. Parameter-free rows have no optimizer update requirement.
Only CPU fp32 is claimed for every fixture. Selected fp64 and stochastic-mode behavior are separate regressions. No CUDA/general parameter sweep is claimed.

### conv_norm_act

Existing Conv2d → norm → activation block.

- Constructor: `hyso.models.blocks.ConvNormAct(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, norm='batch', activation='silu')`
- Input: Floating [N,in_channels,H,W]; groups divides both channel counts. Output: Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See parameter_dependencies and preflight.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: groups divides both channel counts; normalization determines convolution bias; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | pair | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| stride | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| padding | padding | 1 | {'constraints': {'minimum': 0, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| groups | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| norm | choice | batch | {'constraints': {}, 'choices': ['batch', 'group', None]} |
| activation | choice | silu | {'constraints': {}, 'choices': ['silu', 'relu', None]} |

### conv2d

PyTorch Conv2d primitive used by HYSo; grouped/depthwise via groups.

- Constructor: `torch.nn.modules.conv.Conv2d(in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]] = 1, padding: Union[str, int, Tuple[int, int]] = 0, dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None`
- Input: Floating [N,in_channels,H,W]; groups divides both channel counts. Output: Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See parameter_dependencies and preflight.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: groups divides in_channels and out_channels; dilation changes the effective kernel; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | pair | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| stride | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| padding | padding | 1 | {'constraints': {'minimum': 0, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| groups | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| bias | bool | True | {'constraints': {'type': 'boolean'}, 'choices': []} |
| dilation | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |

### residual

Existing two-convolution ResidualBlock with internal projection/addition.

- Constructor: `hyso.models.blocks.ResidualBlock(in_channels, out_channels, stride=1, norm='batch', attention=None)`
- Input: Floating [N,in_channels,H,W]; attention channels equal out_channels. Output: [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| stride | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| norm | choice | batch | {'constraints': {}, 'choices': ['batch', 'group', None]} |
| attention | attention | None | {'constraints': {}, 'choices': []} |

### se

Existing squeeze/excitation channel gates.

- Constructor: `hyso.models.WithSE.SEBlock(channels: int, reduction: int = 8)`
- Input: Floating [N,channels,H,W], all dimensions positive. Output: Same NCHW shape; multiplicative gates, no channel or spatial resizing.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| reduction | positive_int | 8 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |

### cbam

Existing channel and spatial attention.

- Constructor: `hyso.models.bcam.CBAMBlock(channels, reduction=16, kernel_size=7)`
- Input: Floating [N,channels,H,W], all dimensions positive. Output: Same NCHW shape; multiplicative gates, no channel or spatial resizing.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| reduction | positive_int | 16 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | odd_int | 7 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': True}, 'choices': []} |

### eca

Existing channel-axis Conv1d gates; no channel reduction.

- Constructor: `hyso.models.attention.ECAAttention(channels, kernel_size=3)`
- Input: Floating [N,channels,H,W], all dimensions positive. Output: Same NCHW shape; multiplicative gates, no channel or spatial resizing.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | odd_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': True}, 'choices': []} |

### coordinate

Existing height/width attention with internal BatchNorm.

- Constructor: `hyso.models.attention.CoordinateAttention(channels, reduction=32)`
- Input: Floating [N,channels,H,W], all dimensions positive. Output: Same NCHW shape; multiplicative gates, no channel or spatial resizing.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| reduction | positive_int | 32 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |

### batch_norm

PyTorch BatchNorm2d as used in existing HYSo blocks.

- Constructor: `torch.nn.modules.batchnorm.BatchNorm2d(num_features: int, eps: float = 1e-05, momentum: Optional[float] = 0.1, affine: bool = True, track_running_stats: bool = True, device=None, dtype=None) -> None`
- Input: Floating NCHW; C matches num_features, num_channels or channels. GroupNorm groups divide C. Output: Same NCHW shape; channel LayerNorm normalizes C per spatial position. Native statistics constraints apply.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| num_features | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| eps | positive_float | 1e-05 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': True, 'maximum': None, 'nullable': False}, 'choices': []} |
| momentum | optional_momentum | 0.1 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': False, 'maximum': 1, 'nullable': True}, 'choices': []} |
| affine | bool | True | {'constraints': {'type': 'boolean'}, 'choices': []} |
| track_running_stats | bool | True | {'constraints': {'type': 'boolean'}, 'choices': []} |

### group_norm

PyTorch GroupNorm with explicit groups/channels.

- Constructor: `torch.nn.modules.normalization.GroupNorm(num_groups: int, num_channels: int, eps: float = 1e-05, affine: bool = True, device=None, dtype=None) -> None`
- Input: Floating NCHW; C matches num_features, num_channels or channels. GroupNorm groups divide C. Output: Same NCHW shape; channel LayerNorm normalizes C per spatial position. Native statistics constraints apply.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: num_groups divides num_channels; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| num_groups | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| num_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| eps | positive_float | 1e-05 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': True, 'maximum': None, 'nullable': False}, 'choices': []} |
| affine | bool | True | {'constraints': {'type': 'boolean'}, 'choices': []} |

### relu

Out-of-place ReLU used by ConvNormAct.

- Constructor: `torch.nn.modules.activation.ReLU(inplace: bool = False)`
- Input: Floating nonempty NCHW in Builder v1. Output: Same NCHW shape; out-of-place activation.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| none | — | — | — |

### silu

Out-of-place SiLU used by existing residual/convolution blocks.

- Constructor: `torch.nn.modules.activation.SiLU(inplace: bool = False)`
- Input: Floating nonempty NCHW in Builder v1. Output: Same NCHW shape; out-of-place activation.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| none | — | — | — |

### hardswish

Out-of-place HardSwish used by CoordinateAttention.

- Constructor: `torch.nn.modules.activation.Hardswish(inplace: bool = False) -> None`
- Input: Floating nonempty NCHW in Builder v1. Output: Same NCHW shape; out-of-place activation.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| none | — | — | — |

### identity

PyTorch Identity used by existing optional branches.

- Constructor: `torch.nn.modules.linear.Identity(*args: Any, **kwargs: Any) -> None`
- Input: Floating nonempty NCHW in Builder v1. Output: Same tensor/shape; no parameters.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| none | — | — | — |

### modular_cnn

Original ModularCNN constructor; unchanged class and state_dict layout.

- Constructor: `hyso.models.modular.ModularCNN(input_channels: 'int' = 3, num_classes: 'int' = 10, widths: 'tuple[int, ...] | list[int]' = (32, 64, 128), repeats: 'tuple[int, ...] | list[int]' = (1, 1, 1), strides: 'tuple[int, ...] | list[int]' = (1, 2, 2), attention: 'str' = 'none', norm: 'str | None' = 'batch') -> 'None'`
- Input: Floating nonempty [N,input_channels,H,W]; normal model.to(device/dtype) semantics. Output: [N,num_classes] unnormalized classification logits.
- Roles: SUPPORTED fixed modular()/named() profile; no add() feature insertion; UNSUPPORTED complete model node
- Checkpoint: NOT_TESTED here for all family variants; bounded existing named-family regressions separate
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| input_channels | positive_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| num_classes | positive_int | 10 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| widths | int_list | [32, 64, 128] | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'max_items': 16}, 'choices': []} |
| repeats | int_list | [1, 1, 1] | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'max_items': 16}, 'choices': []} |
| strides | int_list | [1, 2, 2] | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'max_items': 16}, 'choices': []} |
| attention | choice | none | {'constraints': {}, 'choices': ['none', 'se', 'cbam', 'eca', 'coordinate']} |
| norm | choice | batch | {'constraints': {}, 'choices': ['batch', 'group', None]} |

### channel_shuffle

Reshape/transpose channel groups while preserving NCHW dimensions.

- Constructor: `hyso.models.regularization.ChannelShuffle(channels: 'int', groups: 'int' = 2) -> 'None'`
- Input: Floating NCHW with matching channels. Output: Same NCHW shape; validated channel partition.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| groups | positive_int | 2 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |

### dropblock2d

Explicit contiguous spatial regularization with per-tensor mask normalization.

    block_size is clipped to H/W, including even spatial extents. Valid top-left
    seed positions spread into exact square blocks. gamma is clipped to one.
    p=1 yields zeros, p=0 and eval return the input. An all-dropped mask yields
    finite zeros. Half-precision mask statistics accumulate in float32.


- Constructor: `hyso.models.regularization.DropBlock2D(p: 'float' = 0.1, block_size: 'int' = 3) -> 'None'`
- Input: Nonempty floating NCHW. Output: Same shape and dtype; explicit training-only masking.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| p | probability_closed | 0.1 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': False, 'maximum': 1, 'nullable': False}, 'choices': []} |
| block_size | positive_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |

### pointwise_conv2d

1x1 channel mixing with zero padding and configurable stride/bias.

- Constructor: `hyso.models.components.PointwiseConv2D(in_channels, out_channels, stride=1, bias=True)`
- Input: Floating [N,in_channels,H,W]; groups divides both channel counts. Output: Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See parameter_dependencies and preflight.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| stride | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| bias | bool | True | {'constraints': {'type': 'boolean'}, 'choices': []} |

### depthwise_conv2d

Independent spatial filters, groups=C and output=C*depth_multiplier.

- Constructor: `hyso.models.components.DepthwiseConv2D(in_channels, depth_multiplier=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)`
- Input: Floating [N,in_channels,H,W]; groups divides both channel counts. Output: Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See parameter_dependencies and preflight.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: groups=in_channels; out_channels=in_channels*depth_multiplier; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| depth_multiplier | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | pair | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| stride | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| padding | padding | 1 | {'constraints': {'minimum': 0, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| dilation | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| bias | bool | True | {'constraints': {'type': 'boolean'}, 'choices': []} |

### depthwise_separable_conv2d

Depthwise spatial filtering followed by dense 1x1 channel mixing.

- Constructor: `hyso.models.components.DepthwiseSeparableConv2D(in_channels, out_channels, depth_multiplier=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)`
- Input: Floating [N,in_channels,H,W]; groups divides both channel counts. Output: Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See parameter_dependencies and preflight.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: intermediate_channels=in_channels*depth_multiplier; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| depth_multiplier | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | pair | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| stride | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| padding | padding | 1 | {'constraints': {'minimum': 0, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| dilation | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| bias | bool | True | {'constraints': {'type': 'boolean'}, 'choices': []} |

### spatial_separable_conv2d

(kH,1) -> (1,kW), with separate H/W stride, padding and dilation.

- Constructor: `hyso.models.components.SpatialSeparableConv2D(in_channels, out_channels, intermediate_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)`
- Input: Floating [N,in_channels,H,W]; groups divides both channel counts. Output: Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See parameter_dependencies and preflight.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: H settings apply to the vertical stage, W settings to the horizontal stage; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| intermediate_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | pair | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| stride | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| padding | padding | 1 | {'constraints': {'minimum': 0, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| dilation | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| bias | bool | True | {'constraints': {'type': 'boolean'}, 'choices': []} |

### factorized_conv2d

Dense 1x1 channel bottleneck -> dense KxK spatial convolution.

    intermediate_channels must not exceed either boundary channel count.
    Spatial stride/padding/dilation belong exclusively to the second operator.
    There is no hidden normalization, activation or residual connection.


- Constructor: `hyso.models.components.FactorizedConv2D(in_channels, out_channels, intermediate_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)`
- Input: Floating [N,in_channels,H,W]; groups divides both channel counts. Output: Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See parameter_dependencies and preflight.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: intermediate_channels <= min(in_channels,out_channels); 1x1 reduction then KxK convolution; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| intermediate_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | pair | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| stride | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| padding | padding | 1 | {'constraints': {'minimum': 0, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| dilation | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| bias | bool | True | {'constraints': {'type': 'boolean'}, 'choices': []} |

### conv_transpose2d

Transposed convolution with explicit output padding.

- Constructor: `torch.nn.modules.conv.ConvTranspose2d(in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0, output_padding: Union[int, Tuple[int, int]] = 0, groups: int = 1, bias: bool = True, dilation: Union[int, Tuple[int, int]] = 1, padding_mode: str = 'zeros', device=None, dtype=None) -> None`
- Input: Floating [N,in_channels,H,W]; groups divides both channel counts. Output: Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See parameter_dependencies and preflight.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: groups divides both channel counts; output_padding[i] < stride[i] or dilation[i]; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | pair | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| stride | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| padding | padding | 1 | {'constraints': {'minimum': 0, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| dilation | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| bias | bool | True | {'constraints': {'type': 'boolean'}, 'choices': []} |
| groups | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| output_padding | padding | 0 | {'constraints': {'minimum': 0, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |

### channel_layer_norm

Normalize C independently at each (N,H,W), preserving public NCHW layout.

    An internal movedim exposes C to torch LayerNorm, then restores NCHW.
    This differs from LayerNorm over the entire CHW sample.


- Constructor: `hyso.models.components.ChannelLayerNorm(channels, eps=1e-05, affine=True)`
- Input: Floating NCHW; C matches num_features, num_channels or channels. GroupNorm groups divide C. Output: Same NCHW shape; channel LayerNorm normalizes C per spatial position. Native statistics constraints apply.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: normalizes C at each N,H,W position, external layout stays NCHW; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| eps | positive_float | 1e-05 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': True, 'maximum': None, 'nullable': False}, 'choices': []} |
| affine | bool | True | {'constraints': {'type': 'boolean'}, 'choices': []} |

### instance_norm

NCHW InstanceNorm with mandatory channel matching even without affine.

- Constructor: `hyso.models.components.InstanceNorm2D(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)`
- Input: Floating NCHW; C matches num_features, num_channels or channels. GroupNorm groups divide C. Output: Same NCHW shape; channel LayerNorm normalizes C per spatial position. Native statistics constraints apply.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: spatial sample count H*W > 1 when using input statistics; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| num_features | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| eps | positive_float | 1e-05 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': True, 'maximum': None, 'nullable': False}, 'choices': []} |
| momentum | probability_closed | 0.1 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': False, 'maximum': 1, 'nullable': False}, 'choices': []} |
| affine | bool | False | {'constraints': {'type': 'boolean'}, 'choices': []} |
| track_running_stats | bool | False | {'constraints': {'type': 'boolean'}, 'choices': []} |

### leaky_relu

ReLU with an explicit nonnegative negative slope.

- Constructor: `torch.nn.modules.activation.LeakyReLU(negative_slope: float = 0.01, inplace: bool = False) -> None`
- Input: Floating nonempty NCHW in Builder v1. Output: Same NCHW shape; out-of-place activation.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| negative_slope | nonnegative_float | 0.01 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': False, 'maximum': None, 'nullable': False}, 'choices': []} |

### prelu

Learnable slope shared by all channels or one per channel.

- Constructor: `torch.nn.modules.activation.PReLU(num_parameters: int = 1, init: float = 0.25, device=None, dtype=None) -> None`
- Input: Floating nonempty NCHW in Builder v1. Output: Same NCHW shape; out-of-place activation.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: num_parameters is 1 or the incoming channel count; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| num_parameters | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| init | finite_float | 0.25 | {'constraints': {'finite': True, 'minimum': -3.4028234663852886e+38, 'exclusive_minimum': False, 'maximum': 3.4028234663852886e+38, 'nullable': False}, 'choices': []} |

### gelu

Gaussian error linear unit with exact or tanh approximation.

- Constructor: `torch.nn.modules.activation.GELU(approximate: str = 'none') -> None`
- Input: Floating nonempty NCHW in Builder v1. Output: Same NCHW shape; out-of-place activation.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| approximate | choice | none | {'constraints': {}, 'choices': ['none', 'tanh']} |

### mish

x*tanh(softplus(x)), out of place.

- Constructor: `torch.nn.modules.activation.Mish(inplace: bool = False)`
- Input: Floating nonempty NCHW in Builder v1. Output: Same NCHW shape; out-of-place activation.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| none | — | — | — |

### hardsigmoid

Piecewise linear bounded sigmoid, out of place.

- Constructor: `torch.nn.modules.activation.Hardsigmoid(inplace: bool = False) -> None`
- Input: Floating nonempty NCHW in Builder v1. Output: Same NCHW shape; out-of-place activation.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| none | — | — | — |

### max_pool2d

Spatial maximum with floor/ceil windows and optional dilation.

- Constructor: `torch.nn.modules.pooling.MaxPool2d(kernel_size: Union[int, Tuple[int, ...]], stride: Union[int, Tuple[int, ...], NoneType] = None, padding: Union[int, Tuple[int, ...]] = 0, dilation: Union[int, Tuple[int, ...]] = 1, return_indices: bool = False, ceil_mode: bool = False) -> None`
- Input: Floating nonempty NCHW; legal window/padding or positive adaptive output_size. Output: Floating NCHW with unchanged channels; global average returns [N,C,1,1].
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: stride=None uses kernel_size; padding[i] <= floor(kernel_size[i]/2); HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| kernel_size | pair | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| stride | optional_pair | None | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': True}, 'choices': []} |
| padding | padding | 0 | {'constraints': {'minimum': 0, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| dilation | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| ceil_mode | bool | False | {'constraints': {'type': 'boolean'}, 'choices': []} |

### avg_pool2d

Spatial mean with explicit padding divisor behavior.

- Constructor: `torch.nn.modules.pooling.AvgPool2d(kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int], NoneType] = None, padding: Union[int, Tuple[int, int]] = 0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: Optional[int] = None) -> None`
- Input: Floating nonempty NCHW; legal window/padding or positive adaptive output_size. Output: Floating NCHW with unchanged channels; global average returns [N,C,1,1].
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: stride=None uses kernel_size; padding[i] <= floor(kernel_size[i]/2); count_include_pad controls the mean divisor; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| kernel_size | pair | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| stride | optional_pair | None | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': True}, 'choices': []} |
| padding | padding | 0 | {'constraints': {'minimum': 0, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| ceil_mode | bool | False | {'constraints': {'type': 'boolean'}, 'choices': []} |
| count_include_pad | bool | True | {'constraints': {'type': 'boolean'}, 'choices': []} |

### adaptive_avg_pool2d

Spatial averages into a fixed positive H/W grid.

- Constructor: `torch.nn.modules.pooling.AdaptiveAvgPool2d(output_size: Union[int, NoneType, Tuple[Optional[int], ...]]) -> None`
- Input: Floating nonempty NCHW; legal window/padding or positive adaptive output_size. Output: Floating NCHW with unchanged channels; global average returns [N,C,1,1].
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| output_size | pair | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |

### adaptive_max_pool2d

Spatial maxima into a fixed positive H/W grid.

- Constructor: `torch.nn.modules.pooling.AdaptiveMaxPool2d(output_size: Union[int, NoneType, Tuple[Optional[int], ...]], return_indices: bool = False) -> None`
- Input: Floating nonempty NCHW; legal window/padding or positive adaptive output_size. Output: Floating NCHW with unchanged channels; global average returns [N,C,1,1].
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| output_size | pair | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |

### global_avg_pool2d

Average spatial axes and retain [N,C,1,1].

- Constructor: `hyso.models.components.GlobalAveragePool2D()`
- Input: Floating nonempty NCHW; legal window/padding or positive adaptive output_size. Output: Floating NCHW with unchanged channels; global average returns [N,C,1,1].
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| none | — | — | — |

### bottleneck_residual

HYSo generic residual main branch with explicit projection policy.

- Constructor: `hyso.models.advanced.AdvancedResidual(**params)`
- Input: Floating [N,in_channels,H,W]; attention channels equal out_channels. Output: [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| intermediate_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | odd_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': True}, 'choices': []} |
| stride | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| dilation | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| normalization | choice | batch | {'constraints': {}, 'choices': ['batch', 'group', None]} |
| activation | choice | silu | {'constraints': {}, 'choices': ['relu', 'silu', 'gelu']} |
| attention | choice | none | {'constraints': {}, 'choices': ['none', 'eca', 'spatial_attention', 'channel_attention']} |
| dropout | probability | 0.0 | {'constraints': {}, 'choices': []} |
| drop_path | probability | 0.0 | {'constraints': {}, 'choices': []} |

### preactivation_residual

HYSo generic residual main branch with explicit projection policy.

- Constructor: `hyso.models.advanced.PreActivationResidual(**params)`
- Input: Floating [N,in_channels,H,W]; attention channels equal out_channels. Output: [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| intermediate_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | odd_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': True}, 'choices': []} |
| stride | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| dilation | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| normalization | choice | batch | {'constraints': {}, 'choices': ['batch', 'group', None]} |
| activation | choice | silu | {'constraints': {}, 'choices': ['relu', 'silu', 'gelu']} |
| attention | choice | none | {'constraints': {}, 'choices': ['none', 'eca', 'spatial_attention', 'channel_attention']} |
| dropout | probability | 0.0 | {'constraints': {}, 'choices': []} |
| drop_path | probability | 0.0 | {'constraints': {}, 'choices': []} |

### wide_residual_block

HYSo generic residual main branch with explicit projection policy.

- Constructor: `hyso.models.advanced.WideResidual(**params)`
- Input: Floating [N,in_channels,H,W]; attention channels equal out_channels. Output: [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| intermediate_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | odd_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': True}, 'choices': []} |
| stride | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| dilation | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| normalization | choice | batch | {'constraints': {}, 'choices': ['batch', 'group', None]} |
| activation | choice | silu | {'constraints': {}, 'choices': ['relu', 'silu', 'gelu']} |
| attention | choice | none | {'constraints': {}, 'choices': ['none', 'eca', 'spatial_attention', 'channel_attention']} |
| dropout | probability | 0.0 | {'constraints': {}, 'choices': []} |
| drop_path | probability | 0.0 | {'constraints': {}, 'choices': []} |

### grouped_bottleneck

HYSo generic residual main branch with explicit projection policy.

- Constructor: `hyso.models.advanced.GroupedBottleneck(**params)`
- Input: Floating [N,in_channels,H,W]; attention channels equal out_channels. Output: [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| intermediate_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | odd_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': True}, 'choices': []} |
| stride | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| dilation | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| normalization | choice | batch | {'constraints': {}, 'choices': ['batch', 'group', None]} |
| activation | choice | silu | {'constraints': {}, 'choices': ['relu', 'silu', 'gelu']} |
| attention | choice | none | {'constraints': {}, 'choices': ['none', 'eca', 'spatial_attention', 'channel_attention']} |
| dropout | probability | 0.0 | {'constraints': {}, 'choices': []} |
| drop_path | probability | 0.0 | {'constraints': {}, 'choices': []} |
| groups | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |

### inverted_bottleneck

HYSo generic residual main branch with explicit projection policy.

- Constructor: `hyso.models.advanced.InvertedBottleneck(**params)`
- Input: Floating [N,in_channels,H,W]; attention channels equal out_channels. Output: [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| expansion_ratio | positive_int | 2 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | odd_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': True}, 'choices': []} |
| stride | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| dilation | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| normalization | choice | batch | {'constraints': {}, 'choices': ['batch', 'group', None]} |
| activation | choice | silu | {'constraints': {}, 'choices': ['relu', 'silu', 'gelu']} |
| attention | choice | none | {'constraints': {}, 'choices': ['none', 'eca', 'spatial_attention', 'channel_attention']} |
| dropout | probability | 0.0 | {'constraints': {}, 'choices': []} |
| drop_path | probability | 0.0 | {'constraints': {}, 'choices': []} |

### gated_residual

HYSo generic residual main branch with explicit projection policy.

- Constructor: `hyso.models.advanced.GatedResidual(**params)`
- Input: Floating [N,in_channels,H,W]; attention channels equal out_channels. Output: [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| intermediate_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | odd_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': True}, 'choices': []} |
| stride | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| dilation | pair | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': False}, 'choices': []} |
| normalization | choice | batch | {'constraints': {}, 'choices': ['batch', 'group', None]} |
| activation | choice | silu | {'constraints': {}, 'choices': ['relu', 'silu', 'gelu']} |
| attention | choice | none | {'constraints': {}, 'choices': ['none', 'eca', 'spatial_attention', 'channel_attention']} |
| dropout | probability | 0.0 | {'constraints': {}, 'choices': []} |
| drop_path | probability | 0.0 | {'constraints': {}, 'choices': []} |
| gate_init | probability_closed | 1.0 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': False, 'maximum': 1, 'nullable': False}, 'choices': []} |
| learnable_gate | bool | True | {'constraints': {'type': 'boolean'}, 'choices': []} |

### spatial_attention

Channel mean/max descriptors -> odd spatial convolution -> sigmoid gate.

- Constructor: `hyso.models.advanced.SpatialAttention(channels, kernel_size=7)`
- Input: Floating [N,channels,H,W], all dimensions positive. Output: Same NCHW shape; multiplicative gates, no channel or spatial resizing.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | odd_int | 7 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': True}, 'choices': []} |

### channel_attention

Shared 1x1 MLP of spatial mean/max descriptors, summed sigmoid channel gate.

- Constructor: `hyso.models.advanced.ChannelAttention(channels, reduction=8)`
- Input: Floating [N,channels,H,W], all dimensions positive. Output: Same NCHW shape; multiplicative gates, no channel or spatial resizing.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| reduction | positive_int | 8 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |

### spatial_self_attention

NCHW -> H*W tokens -> scaled dot-product attention -> projected NCHW.

- Constructor: `hyso.models.advanced.SpatialSelfAttention(**params)`
- Input: Floating [N,channels,H,W], all dimensions positive. Output: Same NCHW shape; multiplicative gates, no channel or spatial resizing.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| embedding_dim | optional_positive_int | None | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'nullable': True}, 'choices': []} |
| dropout | probability | 0.0 | {'constraints': {}, 'choices': []} |

### multihead_spatial_attention

Independent spatial attention heads with an explicit embedding projection.

- Constructor: `hyso.models.advanced.MultiHeadSpatialAttention(**params)`
- Input: Floating [N,channels,H,W], all dimensions positive. Output: Same NCHW shape; multiplicative gates, no channel or spatial resizing.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| embedding_dim | optional_positive_int | None | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'nullable': True}, 'choices': []} |
| num_heads | positive_int | 4 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| dropout | probability | 0.0 | {'constraints': {}, 'choices': []} |

### dropout

Elementwise train masking, identity in eval.

- Constructor: `torch.nn.modules.dropout.Dropout(p: float = 0.5, inplace: bool = False) -> None`
- Input: Nonempty floating NCHW. Output: Same shape and dtype; explicit training-only masking.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| p | probability_closed | 0.5 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': False, 'maximum': 1, 'nullable': False}, 'choices': []} |

### dropout2d

Channelwise train masking, identity in eval.

- Constructor: `torch.nn.modules.dropout.Dropout2d(p: float = 0.5, inplace: bool = False) -> None`
- Input: Nonempty floating NCHW. Output: Same shape and dtype; explicit training-only masking.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| p | probability_closed | 0.5 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': False, 'maximum': 1, 'nullable': False}, 'choices': []} |

### drop_path

Sample-wise tensor masking, x*Bernoulli(1-p)/(1-p) in train, identity in eval.

- Constructor: `hyso.models.advanced.DropPath(p=0.0)`
- Input: Nonempty floating NCHW. Output: Same shape and dtype; explicit training-only masking.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| p | probability | 0.0 | {'constraints': {}, 'choices': []} |

### gaussian_noise

Add independent zero-mean Gaussian noise in train, identity in eval.

- Constructor: `hyso.models.advanced.GaussianNoise(std=0.1)`
- Input: Nonempty floating NCHW. Output: Same shape and dtype; explicit training-only masking.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| std | nonnegative_float | 0.1 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': False, 'maximum': None, 'nullable': False}, 'choices': []} |

### resize

NCHW interpolation with exactly one target and explicit floor scale geometry.

- Constructor: `hyso.models.sampling.Resize(output_size=None, scale_factor=None, mode='nearest', align_corners=None)`
- Input: Floating nonempty NCHW. Output: Floating NCHW with explicit resize or pixel rearrangement geometry.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: resize requires exactly one target; pixel factors require divisibility; reshape preserves sample element count; heads are terminal, in_features is inferred by Builder; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| output_size | optional_pair | None | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False, 'pair_length': 2, 'nullable': True}, 'choices': []} |
| scale_factor | scale_pair | None | {'constraints': {'exclusive_minimum': 0, 'maximum': 8192, 'length': 2, 'nullable': True, 'finite': True}, 'choices': []} |
| mode | choice | nearest | {'constraints': {}, 'choices': ['nearest', 'bilinear', 'bicubic', 'area']} |
| align_corners | optional_bool | None | {'constraints': {'type': 'boolean', 'nullable': True}, 'choices': []} |

### pixel_shuffle

C/r² channels, H*r and W*r spatial dimensions.

- Constructor: `torch.nn.modules.pixelshuffle.PixelShuffle(upscale_factor: int) -> None`
- Input: Floating nonempty NCHW. Output: Floating NCHW with explicit resize or pixel rearrangement geometry.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: resize requires exactly one target; pixel factors require divisibility; reshape preserves sample element count; heads are terminal, in_features is inferred by Builder; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| upscale_factor | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |

### pixel_unshuffle

C*r² channels, H/r and W/r spatial dimensions.

- Constructor: `torch.nn.modules.pixelshuffle.PixelUnshuffle(downscale_factor: int) -> None`
- Input: Floating nonempty NCHW. Output: Floating NCHW with explicit resize or pixel rearrangement geometry.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: resize requires exactly one target; pixel factors require divisibility; reshape preserves sample element count; heads are terminal, in_features is inferred by Builder; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| downscale_factor | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |

### flatten

Flatten CHW to one feature axis, preserving symbolic N.

- Constructor: `hyso.models.sampling.Flatten()`
- Input: Floating NCHW with preserved batch axis. Output: NCHW or terminal NF for flatten, see shape_contract.
- Roles: SUPPORTED terminal schema2 feature; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: resize requires exactly one target; pixel factors require divisibility; reshape preserves sample element count; heads are terminal, in_features is inferred by Builder; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| none | — | — | — |

### permute

Keep N/C axes fixed and optionally exchange spatial H/W axes.

- Constructor: `hyso.models.sampling.Permute(dims=(0, 1, 3, 2))`
- Input: Floating NCHW with preserved batch axis. Output: NCHW or terminal NF for flatten, see shape_contract.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: resize requires exactly one target; pixel factors require divisibility; reshape preserves sample element count; heads are terminal, in_features is inferred by Builder; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| dims | axes | [0, 1, 3, 2] | {'constraints': {'choices': [[0, 1, 2, 3], [0, 1, 3, 2]], 'batch_preserved': True}, 'choices': []} |

### reshape

Reinterpret each sample as explicit positive CHW, preserving batch and elements.

- Constructor: `hyso.models.sampling.Reshape(shape)`
- Input: Floating NCHW with preserved batch axis. Output: NCHW or terminal NF for flatten, see shape_contract.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: resize requires exactly one target; pixel factors require divisibility; reshape preserves sample element count; heads are terminal, in_features is inferred by Builder; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| shape | chw | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'length': 3, 'batch_preserved': True}, 'choices': []} |

### classification_head

Global pool or flat features, optional dropout, linear logits.

- Constructor: `hyso.models.sampling.PredictionHead(**params)`
- Input: Floating NCHW or NF with pooling=none. Output: Floating NF prediction, head type controls output semantics.
- Roles: SUPPORTED terminal head; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: resize requires exactly one target; pixel factors require divisibility; reshape preserves sample element count; heads are terminal, in_features is inferred by Builder; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_features | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| pooling | choice | avg | {'constraints': {}, 'choices': ['avg', 'max', 'none']} |
| num_classes | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| dropout | probability | 0.0 | {'constraints': {}, 'choices': []} |

### regression_head

Global pool or flat features, unrestricted linear regression output.

- Constructor: `hyso.models.sampling.RegressionHead(**params)`
- Input: Floating NCHW or NF with pooling=none. Output: Floating NF prediction, head type controls output semantics.
- Roles: SUPPORTED terminal head; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: resize requires exactly one target; pixel factors require divisibility; reshape preserves sample element count; heads are terminal, in_features is inferred by Builder; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_features | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| pooling | choice | avg | {'constraints': {}, 'choices': ['avg', 'max', 'none']} |
| output_features | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |

### embedding_head

Global pool or flat features, linear embedding, optional safe L2 normalization.

- Constructor: `hyso.models.sampling.EmbeddingHead(**params)`
- Input: Floating NCHW or NF with pooling=none. Output: Floating NF prediction, head type controls output semantics.
- Roles: SUPPORTED terminal head; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: resize requires exactly one target; pixel factors require divisibility; reshape preserves sample element count; heads are terminal, in_features is inferred by Builder; HYSo/PyTorch component; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_features | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| pooling | choice | avg | {'constraints': {}, 'choices': ['avg', 'max', 'none']} |
| embedding_dim | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| normalize | bool | False | {'constraints': {'type': 'boolean'}, 'choices': []} |
| eps | positive_float | 1e-12 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': True, 'maximum': None, 'nullable': False}, 'choices': []} |

### simam

Parameter-free spatial energy gates; singleton variance is zero.

- Constructor: `hyso.models.m3_attention.SimAM(channels: int, epsilon: float = 0.0001) -> None`
- Input: Floating [N,channels,H,W], all dimensions positive. Output: Same NCHW shape; multiplicative gates, no channel or spatial resizing.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: Closed M3 constructor / torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| epsilon | positive_float | 0.0001 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': True, 'maximum': None, 'nullable': False}, 'choices': []} |

### triplet

Convolutional rotated-axis Triplet Attention with BN.

- Constructor: `hyso.models.m3_attention.TripletAttention(channels: int, kernel_size: int = 7, spatial: bool = True) -> None`
- Input: Floating [N,channels,H,W], all dimensions positive. Output: Same NCHW shape; multiplicative gates, no channel or spatial resizing.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: Closed M3 constructor / torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | odd_int | 7 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': True}, 'choices': []} |
| spatial | bool | True | {'constraints': {'type': 'boolean'}, 'choices': []} |

### selective_kernel

Dilated branch fusion with per-channel branch softmax.

- Constructor: `hyso.models.m3_attention.SelectiveKernelAttention(channels: int, branches: int = 2, groups: int = 1, reduction: int = 16, minimum: int = 32) -> None`
- Input: Floating [N,channels,H,W], all dimensions positive. Output: Same NCHW shape; multiplicative gates, no channel or spatial resizing.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: Closed M3 constructor / torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| branches | positive_int | 2 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| groups | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| reduction | positive_int | 16 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| minimum | positive_int | 32 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |

### basic_residual

ResNet v1 two 3x3/ReLU block, expansion=1, automatic projection.

- Constructor: `hyso.models.m3_residual.BasicResidualBlock(in_channels: int, out_channels: int, stride: int = 1, norm: str | None = 'batch') -> None`
- Input: Floating [N,in_channels,H,W]; attention channels equal out_channels. Output: [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: Closed M3 constructor / torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| stride | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| norm | choice | batch | {'constraints': {}, 'choices': ['batch', 'group', None]} |

### bottleneck

ResNet v1.5 1x1/3x3/1x1, expansion=4; out_channels includes expansion.

- Constructor: `hyso.models.m3_residual.Bottleneck(in_channels: int, out_channels: int, stride: int = 1, norm: str | None = 'batch', groups: int = 1, width_per_group: int = 64) -> None`
- Input: Floating [N,in_channels,H,W]; attention channels equal out_channels. Output: [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: Closed M3 constructor / torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| stride | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| norm | choice | batch | {'constraints': {}, 'choices': ['batch', 'group', None]} |
| groups | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| width_per_group | positive_int | 64 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |

### preactivation

BN/ReLU before each convolution, no activation after addition.

    Identity shortcut carries raw input; a projection uses the first activated
    input. bottleneck=True uses expansion=4 and stride on the 3x3 convolution.


- Constructor: `hyso.models.m3_residual.PreActivationBlock(in_channels: int, out_channels: int, stride: int = 1, norm: str | None = 'batch', bottleneck: bool = False, dropout: float = 0.0) -> None`
- Input: Floating [N,in_channels,H,W]; attention channels equal out_channels. Output: [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: Closed M3 constructor / torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| stride | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| norm | choice | batch | {'constraints': {}, 'choices': ['batch', 'group', None]} |
| bottleneck | bool | False | {'constraints': {'type': 'boolean'}, 'choices': []} |
| dropout | probability | 0.0 | {'constraints': {}, 'choices': []} |

### wide_residual

Two preactivated 3x3 convolutions with optional inter-convolution dropout.

- Constructor: `hyso.models.m3_residual.WideResidualBlock(in_channels: int, out_channels: int, stride: int = 1, norm: str | None = 'batch', dropout: float = 0.0) -> None`
- Input: Floating [N,in_channels,H,W]; attention channels equal out_channels. Output: [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: Closed M3 constructor / torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| stride | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| norm | choice | batch | {'constraints': {}, 'choices': ['batch', 'group', None]} |
| dropout | probability | 0.0 | {'constraints': {}, 'choices': []} |

### resnext_bottleneck

Grouped ResNeXt bottleneck; default cardinality 32, base width 4.

- Constructor: `hyso.models.m3_residual.ResNeXtBottleneck(in_channels: int, out_channels: int, stride: int = 1, norm: str | None = 'batch', groups: int = 32, width_per_group: int = 4) -> None`
- Input: Floating [N,in_channels,H,W]; attention channels equal out_channels. Output: [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: Closed M3 constructor / torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| stride | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| norm | choice | batch | {'constraints': {}, 'choices': ['batch', 'group', None]} |
| groups | positive_int | 32 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| width_per_group | positive_int | 4 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |

### resnet

Torchvision ResNet 18/34 or v1.5 bottleneck 50/101/152; ImageNet stem.

    input_channels and num_classes change only input/output contracts. No weights.
    State keys match torchvision when dimensions/norm match. Not original v1
    stride placement for bottleneck models: stride is on 3x3 (v1.5).


- Constructor: `hyso.models.architectures.ResNet(depth: 'int' = 18, input_channels: 'int' = 3, num_classes: 'int' = 1000, norm: 'str | None' = 'batch') -> 'None'`
- Input: Floating nonempty [N,input_channels,H,W]; normal model.to(device/dtype) semantics. Output: [N,num_classes] unnormalized classification logits.
- Roles: SUPPORTED fixed modular()/named() profile; no add() feature insertion; UNSUPPORTED complete model node
- Checkpoint: NOT_TESTED here for all family variants; bounded existing named-family regressions separate
- Limits: Closed M3 constructor / torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| depth | positive_int | 18 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| input_channels | positive_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| num_classes | positive_int | 1000 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| norm | choice | batch | {'constraints': {}, 'choices': ['batch', 'group', None]} |

### preact_resnet

Four-stage full-preactivation ResNet with final BN/ReLU and global pool.

    ImageNet 7x7/stride2/maxpool stem; no stem BN/ReLU, block preactivation
    supplies it. base_channels=64 default; custom widths are explicit variants.
    Bottlenecks use stride on their 3x3, not a claimed exact paper checkpoint.


- Constructor: `hyso.models.architectures.PreActResNet(depth: 'int' = 18, input_channels: 'int' = 3, num_classes: 'int' = 1000, base_channels: 'int' = 64, norm: 'str | None' = 'batch') -> 'None'`
- Input: Floating nonempty [N,input_channels,H,W]; normal model.to(device/dtype) semantics. Output: [N,num_classes] unnormalized classification logits.
- Roles: SUPPORTED fixed modular()/named() profile; no add() feature insertion; UNSUPPORTED complete model node
- Checkpoint: NOT_TESTED here for all family variants; bounded existing named-family regressions separate
- Limits: Closed M3 constructor / torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| depth | positive_int | 18 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| input_channels | positive_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| num_classes | positive_int | 1000 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| base_channels | positive_int | 64 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| norm | choice | batch | {'constraints': {}, 'choices': ['batch', 'group', None]} |

### wide_resnet

CIFAR-style 6n+4 WideResNet, 3x3 stem, widths 16k/32k/64k.

    This is a basic-block WideResNet, not torchvision wide_resnet50_2 bottlenecks.
    Projection uses activated input; optional dropout sits between convolutions.


- Constructor: `hyso.models.architectures.WideResNet(depth: 'int' = 28, input_channels: 'int' = 3, num_classes: 'int' = 10, widen_factor: 'int' = 10, dropout: 'float' = 0.0, norm: 'str | None' = 'batch') -> 'None'`
- Input: Floating nonempty [N,input_channels,H,W]; normal model.to(device/dtype) semantics. Output: [N,num_classes] unnormalized classification logits.
- Roles: SUPPORTED fixed modular()/named() profile; no add() feature insertion; UNSUPPORTED complete model node
- Checkpoint: NOT_TESTED here for all family variants; bounded existing named-family regressions separate
- Limits: Closed M3 constructor / torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| depth | positive_int | 28 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| input_channels | positive_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| num_classes | positive_int | 10 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| widen_factor | positive_int | 10 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| dropout | probability | 0.0 | {'constraints': {}, 'choices': []} |
| norm | choice | batch | {'constraints': {}, 'choices': ['batch', 'group', None]} |

### resnext

Torchvision grouped bottleneck topology; default ResNeXt-50 32x4d.

    For the common 101-32x8d variant explicitly select width_per_group=8.


- Constructor: `hyso.models.architectures.ResNeXt(depth: 'int' = 50, input_channels: 'int' = 3, num_classes: 'int' = 1000, groups: 'int' = 32, width_per_group: 'int' = 4, norm: 'str | None' = 'batch') -> 'None'`
- Input: Floating nonempty [N,input_channels,H,W]; normal model.to(device/dtype) semantics. Output: [N,num_classes] unnormalized classification logits.
- Roles: SUPPORTED fixed modular()/named() profile; no add() feature insertion; UNSUPPORTED complete model node
- Checkpoint: NOT_TESTED here for all family variants; bounded existing named-family regressions separate
- Limits: Closed M3 constructor / torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| depth | positive_int | 50 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| input_channels | positive_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| num_classes | positive_int | 1000 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| groups | positive_int | 32 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| width_per_group | positive_int | 4 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| norm | choice | batch | {'constraints': {}, 'choices': ['batch', 'group', None]} |

### densenet

Torchvision DenseNet-BC (bottleneck factor 4, transition compression 0.5).

    Depth layouts 121/169/201, default growth=32 and initial channels=64 preserve
    standard topology. A custom growth_rate is explicitly a width variant.


- Constructor: `hyso.models.architectures.DenseNet(depth: 'int' = 121, input_channels: 'int' = 3, num_classes: 'int' = 1000, growth_rate: 'int' = 32, dropout: 'float' = 0.0) -> 'None'`
- Input: Floating nonempty [N,input_channels,H,W]; normal model.to(device/dtype) semantics. Output: [N,num_classes] unnormalized classification logits.
- Roles: SUPPORTED fixed modular()/named() profile; no add() feature insertion; UNSUPPORTED complete model node
- Checkpoint: NOT_TESTED here for all family variants; bounded existing named-family regressions separate
- Limits: Closed M3 constructor / torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| depth | positive_int | 121 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| input_channels | positive_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| num_classes | positive_int | 1000 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| growth_rate | positive_int | 32 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| dropout | probability | 0.0 | {'constraints': {}, 'choices': []} |

### inverted_residual

MobileNetV2 expansion/ReLU6, depthwise and linear projection adapter.

    Residual addition requires stride=1 and equal input/output channels.


- Constructor: `hyso.models.m4_blocks.InvertedResidual(in_channels: 'int', out_channels: 'int', stride: 'int' = 1, expand_ratio: 'float' = 6.0) -> 'None'`
- Input: Floating NCHW with matching in_channels; native BN statistics constraints apply. Output: [N,out_channels,ceil(H/stride),ceil(W/stride)]; closed internal branch topology.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: Closed mobile/edge constructor or torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| stride | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| expand_ratio | positive_float | 6.0 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': True, 'maximum': None, 'nullable': False}, 'choices': []} |

### mbconv

Torchvision SiLU MBConv with input-width/4 SE and row stochastic depth.

    Expanded width follows torchvision's divisor-eight rounding convention.


- Constructor: `hyso.models.m4_blocks.MBConv(in_channels: 'int', out_channels: 'int', stride: 'int' = 1, expand_ratio: 'float' = 6.0, kernel_size: 'int' = 3, stochastic_depth: 'float' = 0.0) -> 'None'`
- Input: Floating NCHW with matching in_channels; native BN statistics constraints apply. Output: [N,out_channels,ceil(H/stride),ceil(W/stride)]; closed internal branch topology.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: Closed mobile/edge constructor or torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| stride | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| expand_ratio | positive_float | 6.0 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': True, 'maximum': None, 'nullable': False}, 'choices': []} |
| kernel_size | odd_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': True}, 'choices': []} |
| stochastic_depth | probability | 0.0 | {'constraints': {}, 'choices': []} |

### fused_mbconv

Torchvision fused spatial expansion and linear projection with SiLU.

- Constructor: `hyso.models.m4_blocks.FusedMBConv(in_channels: 'int', out_channels: 'int', stride: 'int' = 1, expand_ratio: 'float' = 4.0, kernel_size: 'int' = 3, stochastic_depth: 'float' = 0.0) -> 'None'`
- Input: Floating NCHW with matching in_channels; native BN statistics constraints apply. Output: [N,out_channels,ceil(H/stride),ceil(W/stride)]; closed internal branch topology.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: Closed mobile/edge constructor or torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| stride | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| expand_ratio | positive_float | 4.0 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': True, 'maximum': None, 'nullable': False}, 'choices': []} |
| kernel_size | odd_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': True}, 'choices': []} |
| stochastic_depth | probability | 0.0 | {'constraints': {}, 'choices': []} |

### ghost_conv

GhostNetV1 primary features and depthwise cheap features, concatenated.

    ratio>=2 selects ceil(out/ratio) primary channels. The final concatenation
    is sliced to exactly out_channels. relu=False preserves a linear projection.


- Constructor: `hyso.models.m4_blocks.GhostConv(in_channels: 'int', out_channels: 'int', stride: 'int' = 1, kernel_size: 'int' = 1, ratio: 'int' = 2, cheap_kernel: 'int' = 3, relu: 'bool' = True) -> 'None'`
- Input: Floating NCHW with matching in_channels; native BN statistics constraints apply. Output: [N,out_channels,ceil(H/stride),ceil(W/stride)]; closed internal branch topology.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: Closed mobile/edge constructor or torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| stride | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | odd_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': True}, 'choices': []} |
| ratio | positive_int | 2 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| cheap_kernel | odd_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': True}, 'choices': []} |
| relu | bool | True | {'constraints': {'type': 'boolean'}, 'choices': []} |

### ghost_bottleneck

GhostNetV1 two Ghost modules, optional depthwise downsample and hard-gated SE.

    The shortcut uses depthwise plus pointwise convolution on shape changes.
    hidden_channels is explicit so published expansion widths can be preserved.


- Constructor: `hyso.models.m4_blocks.GhostBottleneck(in_channels: 'int', out_channels: 'int', hidden_channels: 'int', stride: 'int' = 1, kernel_size: 'int' = 3, se_ratio: 'float' = 0.0) -> 'None'`
- Input: Floating NCHW with matching in_channels; native BN statistics constraints apply. Output: [N,out_channels,ceil(H/stride),ceil(W/stride)]; closed internal branch topology.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: Closed mobile/edge constructor or torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| stride | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| hidden_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | odd_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': True}, 'choices': []} |
| se_ratio | probability_closed | 0.0 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': False, 'maximum': 1, 'nullable': False}, 'choices': []} |

### shuffle_unit

Torchvision ShuffleNetV2 split/concat/shuffle unit with even output width.

- Constructor: `hyso.models.m4_blocks.ShuffleUnit(in_channels: 'int', out_channels: 'int', stride: 'int' = 1) -> 'None'`
- Input: Floating NCHW with matching in_channels; native BN statistics constraints apply. Output: [N,out_channels,ceil(H/stride),ceil(W/stride)]; closed internal branch topology.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: Closed mobile/edge constructor or torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| stride | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |

### repvgg_block

3x3+BN, 1x1+BN and optional identity BN, summed before ReLU.

    Groups divide both channel counts. deploy=True constructs the single biased
    3x3 topology used to load explicitly converted weights.


- Constructor: `hyso.models.m4_reparam.RepVGGBlock(in_channels: 'int', out_channels: 'int', stride: 'int' = 1, groups: 'int' = 1, deploy: 'bool' = False) -> 'None'`
- Input: Floating NCHW with matching in_channels; native BN statistics constraints apply. Output: [N,out_channels,ceil(H/stride),ceil(W/stride)]; closed internal branch topology.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: Closed mobile/edge constructor or torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| stride | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| groups | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| deploy | bool | False | {'constraints': {'type': 'boolean'}, 'choices': []} |

### mobileone_block

K Conv+BN branches, optional 1x1 scale and identity BN, then SE/ReLU.

    kernel_size=1 has no scale branch. use_se reuses HYSo's sigmoid SE with
    reduction=16 and a minimum hidden width of one for small configurations.


- Constructor: `hyso.models.m4_reparam.MobileOneBlock(in_channels: 'int', out_channels: 'int', kernel_size: 'int' = 3, stride: 'int' = 1, groups: 'int' = 1, num_conv_branches: 'int' = 1, use_se: 'bool' = False, deploy: 'bool' = False) -> 'None'`
- Input: Floating NCHW with matching in_channels; native BN statistics constraints apply. Output: [N,out_channels,ceil(H/stride),ceil(W/stride)]; closed internal branch topology.
- Roles: SUPPORTED feature block; SUPPORTED unary node
- Checkpoint: Observed strict model-only v1 roundtrip; no universal architecture/optimizer resume claim
- Limits: Closed mobile/edge constructor or torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| in_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| out_channels | positive_int | required | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| stride | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| kernel_size | odd_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': True}, 'choices': []} |
| groups | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| num_conv_branches | positive_int | 1 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| use_se | bool | False | {'constraints': {'type': 'boolean'}, 'choices': []} |
| deploy | bool | False | {'constraints': {'type': 'boolean'}, 'choices': []} |

### mobilenet_v2

Torchvision canonical MobileNetV2, divisor-eight channels, linear bottlenecks.

    width_mult=1 is the standard layout. Input/head/width changes construct
    explicit untrained variants. dropout affects the classification head.


- Constructor: `hyso.models.mobile_architectures.MobileNetV2(input_channels: 'int' = 3, num_classes: 'int' = 1000, width_mult: 'float' = 1.0, dropout: 'float' = 0.2) -> 'None'`
- Input: Floating nonempty [N,input_channels,H,W]; normal model.to(device/dtype) semantics. Output: [N,num_classes] unnormalized classification logits.
- Roles: SUPPORTED fixed modular()/named() profile; no add() feature insertion; UNSUPPORTED complete model node
- Checkpoint: NOT_TESTED here for all family variants; bounded existing named-family regressions separate
- Limits: Closed mobile/edge constructor or torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| input_channels | positive_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| num_classes | positive_int | 1000 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| width_mult | positive_float | 1.0 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': True, 'maximum': None, 'nullable': False}, 'choices': []} |
| dropout | probability | 0.2 | {'constraints': {}, 'choices': []} |

### mobilenet_v3

Torchvision Small/Large MobileNetV3 with native SE, ReLU and HardSwish.

    variant='large', width_mult=1 selects the standard Large configuration.


- Constructor: `hyso.models.mobile_architectures.MobileNetV3(variant: 'str' = 'large', input_channels: 'int' = 3, num_classes: 'int' = 1000, width_mult: 'float' = 1.0, dropout: 'float' = 0.2) -> 'None'`
- Input: Floating nonempty [N,input_channels,H,W]; normal model.to(device/dtype) semantics. Output: [N,num_classes] unnormalized classification logits.
- Roles: SUPPORTED fixed modular()/named() profile; no add() feature insertion; UNSUPPORTED complete model node
- Checkpoint: NOT_TESTED here for all family variants; bounded existing named-family regressions separate
- Limits: Closed mobile/edge constructor or torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| input_channels | positive_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| num_classes | positive_int | 1000 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| variant | choice | large | {'constraints': {}, 'choices': ['small', 'large']} |
| width_mult | positive_float | 1.0 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': True, 'maximum': None, 'nullable': False}, 'choices': []} |
| dropout | probability | 0.2 | {'constraints': {}, 'choices': []} |

### efficientnet_v2

Torchvision V2 S/M/L layouts with FusedMBConv/MBConv and native SE/SiLU.

    width_mult=depth_mult=1 preserves the selected canonical stage configuration.
    Smaller multipliers explicitly select derived width/repeat configurations.
    dropout=None uses torchvision's S=.2, M=.3, L=.4 head defaults.


- Constructor: `hyso.models.mobile_architectures.EfficientNetV2(variant: 'str' = 's', input_channels: 'int' = 3, num_classes: 'int' = 1000, width_mult: 'float' = 1.0, depth_mult: 'float' = 1.0, dropout: 'float | None' = None, stochastic_depth: 'float' = 0.2) -> 'None'`
- Input: Floating nonempty [N,input_channels,H,W]; normal model.to(device/dtype) semantics. Output: [N,num_classes] unnormalized classification logits.
- Roles: SUPPORTED fixed modular()/named() profile; no add() feature insertion; UNSUPPORTED complete model node
- Checkpoint: NOT_TESTED here for all family variants; bounded existing named-family regressions separate
- Limits: Closed mobile/edge constructor or torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| input_channels | positive_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| num_classes | positive_int | 1000 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| variant | choice | s | {'constraints': {}, 'choices': ['s', 'm', 'l']} |
| width_mult | positive_float | 1.0 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': True, 'maximum': None, 'nullable': False}, 'choices': []} |
| depth_mult | positive_float | 1.0 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': True, 'maximum': None, 'nullable': False}, 'choices': []} |
| dropout | optional_probability | None | {'constraints': {}, 'choices': []} |
| stochastic_depth | probability | 0.2 | {'constraints': {}, 'choices': []} |

### ghostnet

GhostNetV1 author stage widths, Ghost expansion/projection and hard-gated SE.

    width_mult rounds to divisor four. The post-pool head has 1280 channels.
    HYSo module keys are used for strict HYSo state_dict serialization.


- Constructor: `hyso.models.mobile_architectures.GhostNet(input_channels: 'int' = 3, num_classes: 'int' = 1000, width_mult: 'float' = 1.0, dropout: 'float' = 0.2) -> 'None'`
- Input: Floating nonempty [N,input_channels,H,W]; normal model.to(device/dtype) semantics. Output: [N,num_classes] unnormalized classification logits.
- Roles: SUPPORTED fixed modular()/named() profile; no add() feature insertion; UNSUPPORTED complete model node
- Checkpoint: NOT_TESTED here for all family variants; bounded existing named-family regressions separate
- Limits: Closed mobile/edge constructor or torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| input_channels | positive_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| num_classes | positive_int | 1000 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| width_mult | positive_float | 1.0 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': True, 'maximum': None, 'nullable': False}, 'choices': []} |
| dropout | probability | 0.2 | {'constraints': {}, 'choices': []} |

### shufflenet_v2

Torchvision x0_5/x1_0/x1_5/x2_0 stage widths and repeats [4,8,4].

- Constructor: `hyso.models.mobile_architectures.ShuffleNetV2(variant: 'str' = 'x1_0', input_channels: 'int' = 3, num_classes: 'int' = 1000) -> 'None'`
- Input: Floating nonempty [N,input_channels,H,W]; normal model.to(device/dtype) semantics. Output: [N,num_classes] unnormalized classification logits.
- Roles: SUPPORTED fixed modular()/named() profile; no add() feature insertion; UNSUPPORTED complete model node
- Checkpoint: NOT_TESTED here for all family variants; bounded existing named-family regressions separate
- Limits: Closed mobile/edge constructor or torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| input_channels | positive_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| num_classes | positive_int | 1000 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| variant | choice | x1_0 | {'constraints': {}, 'choices': ['x0_5', 'x1_0', 'x1_5', 'x2_0']} |

### repvgg

Author A0/A1 [2,4,14,1] stages with 3x3/1x1/identity training branches.

    width_mult=depth_mult=1 selects the published stage layout. deploy=True
    constructs the single-convolution graph for converted HYSo checkpoints.


- Constructor: `hyso.models.reparam_architectures.RepVGG(variant: 'str' = 'a0', input_channels: 'int' = 3, num_classes: 'int' = 1000, width_mult: 'float' = 1.0, depth_mult: 'float' = 1.0, deploy: 'bool' = False) -> 'None'`
- Input: Floating nonempty [N,input_channels,H,W]; normal model.to(device/dtype) semantics. Output: [N,num_classes] unnormalized classification logits.
- Roles: SUPPORTED fixed modular()/named() profile; no add() feature insertion; UNSUPPORTED complete model node
- Checkpoint: NOT_TESTED here for all family variants; bounded existing named-family regressions separate
- Limits: Closed mobile/edge constructor or torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| input_channels | positive_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| num_classes | positive_int | 1000 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| variant | choice | a0 | {'constraints': {}, 'choices': ['a0', 'a1']} |
| width_mult | positive_float | 1.0 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': True, 'maximum': None, 'nullable': False}, 'choices': []} |
| depth_mult | positive_float | 1.0 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': True, 'maximum': None, 'nullable': False}, 'choices': []} |
| deploy | bool | False | {'constraints': {'type': 'boolean'}, 'choices': []} |

### mobileone

Author s0/s1/s2 widths and [2,8,10,1] depthwise/pointwise pair stages.

    s0 uses four convolution branches per stage block, s1/s2 use one. The
    initial stem uses one branch as in the released implementation. These
    variants use plain ReLU and exclude the later s4 SE configuration.


- Constructor: `hyso.models.reparam_architectures.MobileOne(variant: 'str' = 's0', input_channels: 'int' = 3, num_classes: 'int' = 1000, width_mult: 'float' = 1.0, depth_mult: 'float' = 1.0, deploy: 'bool' = False) -> 'None'`
- Input: Floating nonempty [N,input_channels,H,W]; normal model.to(device/dtype) semantics. Output: [N,num_classes] unnormalized classification logits.
- Roles: SUPPORTED fixed modular()/named() profile; no add() feature insertion; UNSUPPORTED complete model node
- Checkpoint: NOT_TESTED here for all family variants; bounded existing named-family regressions separate
- Limits: Closed mobile/edge constructor or torchvision adapter; CPU construction only.; Builder v1: linear connections; no external skip/concat/DAG.; BatchNorm requires N*H*W>1 while training; GroupNorm requires valid group statistics.; SE/CBAM keep max(1, channels//reduction); CBAM/ECA kernels must be odd.; Serialization saves architecture only, not trained weights or optimizer state.

| Parameter | Kind | Required / default | Bounds / choices |
| --- | --- | --- | --- |
| input_channels | positive_int | 3 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| num_classes | positive_int | 1000 | {'constraints': {'minimum': 1, 'maximum': 8192, 'odd': False}, 'choices': []} |
| variant | choice | s0 | {'constraints': {}, 'choices': ['s0', 's1', 's2']} |
| width_mult | positive_float | 1.0 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': True, 'maximum': None, 'nullable': False}, 'choices': []} |
| depth_mult | positive_float | 1.0 | {'constraints': {'finite': True, 'minimum': 0, 'exclusive_minimum': True, 'maximum': None, 'nullable': False}, 'choices': []} |
| deploy | bool | False | {'constraints': {'type': 'boolean'}, 'choices': []} |

## Graph fusion operators

| Name | Actual constructor | Parameters and restrictions |
| --- | --- | --- |
| add | hyso.models.graph_ops.Fusion | {}; 2..8 ordered inputs; no implicit broadcasting |
| concat | hyso.models.graph_ops.Concat | {'axis': {'default': 1, 'minimum': 1, 'maximum': 3}}; 2..8 ordered inputs; no implicit broadcasting |
| multiply | hyso.models.graph_ops.Multiply | {}; 2..8 ordered inputs; no implicit broadcasting |
| weighted_sum | hyso.models.graph_ops.WeightedSum | {'weights': {'min_items': 2, 'max_items': 8, 'minimum': -1000000.0, 'maximum': 1000000.0}, 'learnable': {'default': False}}; 2..8 ordered inputs; no implicit broadcasting |

The four fusion operations also passed conditional Graph v2 typed contract admission; Registry.create and linear ModelBuilder reject their names. Full-Graph/receiver constraints above still apply.

## Defined but not Registry-bound classes

This is exhaustive top-level class discovery in hyso/models/*.py, not a claim that helper/base classes are model components. Inherited constructors may have no local initializer. Public spelling is distinguished from a documented public namespace or Builder support.

| Definition | Bases | Registry alias | Commercial module | Forward defined locally |
| --- | --- | --- | --- | --- |
| hyso.models.bcam.BottleneckResidualBlock | ['nn.Module'] | not wired / helper | True | True |
| hyso.models.bcam.ELO | ['nn.Module'] | not wired / helper | True | True |
| hyso.models.bcam.EBO | ['nn.Module'] | not wired / helper | True | True |
| hyso.models.builder.BuiltClassifier | ['nn.Module'] | not wired / helper | True | True |
| hyso.models.builder.BuiltSequential | ['BuiltClassifier'] | not wired / helper | True | True |
| hyso.models.builder.ModelBuilder | [] | not wired / helper | True | False |
| hyso.models.fit.CNNSENormalWithFit | ['CEMO'] | not wired / helper | True | False |
| hyso.models.graph.GraphModel | ['nn.Module'] | not wired / helper | True | True |
| hyso.models.graph.GraphModelBuilder | [] | not wired / helper | True | False |
| hyso.models.graph_ops.Fusion | ['nn.Module'] | not wired / helper | True | True |
| hyso.models.graph_ops.Concat | ['Fusion'] | not wired / helper | True | False |
| hyso.models.graph_ops.Multiply | ['Fusion'] | not wired / helper | True | False |
| hyso.models.graph_ops.WeightedSum | ['Fusion'] | not wired / helper | True | False |
| hyso.models.m3_attention._TripletGate | ['nn.Module'] | not wired / helper | False | True |
| hyso.models.m4_reparam._ReparamBlock | ['nn.Module'] | not wired / helper | False | False |
| hyso.models.m4_reparam.ReparameterizableModel | [] | not wired / helper | False | False |
| hyso.models.mobile_architectures.MobileArchitectureConfig | [] | not wired / helper | False | False |
| hyso.models.output.GraphTensorOutput | ['nn.Module'] | not wired / helper | True | True |
| hyso.models.registry.ModelConfigError | ['ValueError'] | not wired / helper | True | False |
| hyso.models.registry.ParameterSpec | [] | not wired / helper | True | False |
| hyso.models.registry.ComponentSpec | [] | not wired / helper | True | False |
| hyso.models.registry.ModelRegistry | [] | not wired / helper | True | False |
| hyso.models.WithSE.EMA | [] | not wired / helper | True | False |
| hyso.models.WithSE.BottleneckSEBlock | ['nn.Module'] | not wired / helper | True | True |
| hyso.models.WithSE.CEMO | ['nn.Module'] | not wired / helper | True | True |

## Controlled combination universe

Factors: convolution (standard/grouped/depthwise), activation (ReLU/SiLU/GELU), normalization (BatchNorm/GroupNorm), pool (max/average/adaptive average). The 54 configurations form an explicitly bounded universe. **18 SUPPORTED observed, 36 NOT_TESTED**. The 18 cover every two-factor combination. This does not enumerate all legal architectures or parameter values.

| Conv index | Activation index | Norm index | Pool index | Status | Passing source XML test ID |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | SUPPORTED / PASS | `test_builder_graph_compatibility::test_b_e_risk_pairwise_cnn_combinations[0-0-0-0]` |
| 0 | 0 | 0 | 1 | NOT_TESTED | Not executed by bounded schedule |
| 0 | 0 | 0 | 2 | NOT_TESTED | Not executed by bounded schedule |
| 0 | 0 | 1 | 0 | NOT_TESTED | Not executed by bounded schedule |
| 0 | 0 | 1 | 1 | SUPPORTED / PASS | `test_builder_graph_compatibility::test_b_e_risk_pairwise_cnn_combinations[0-0-1-1]` |
| 0 | 0 | 1 | 2 | NOT_TESTED | Not executed by bounded schedule |
| 0 | 1 | 0 | 0 | NOT_TESTED | Not executed by bounded schedule |
| 0 | 1 | 0 | 1 | SUPPORTED / PASS | `test_builder_graph_compatibility::test_b_e_risk_pairwise_cnn_combinations[0-1-0-1]` |
| 0 | 1 | 0 | 2 | NOT_TESTED | Not executed by bounded schedule |
| 0 | 1 | 1 | 0 | NOT_TESTED | Not executed by bounded schedule |
| 0 | 1 | 1 | 1 | NOT_TESTED | Not executed by bounded schedule |
| 0 | 1 | 1 | 2 | SUPPORTED / PASS | `test_builder_graph_compatibility::test_b_e_risk_pairwise_cnn_combinations[0-1-1-2]` |
| 0 | 2 | 0 | 0 | NOT_TESTED | Not executed by bounded schedule |
| 0 | 2 | 0 | 1 | NOT_TESTED | Not executed by bounded schedule |
| 0 | 2 | 0 | 2 | SUPPORTED / PASS | `test_builder_graph_compatibility::test_b_e_risk_pairwise_cnn_combinations[0-2-0-2]` |
| 0 | 2 | 1 | 0 | SUPPORTED / PASS | `test_builder_graph_compatibility::test_b_e_risk_pairwise_cnn_combinations[0-2-1-0]` |
| 0 | 2 | 1 | 1 | NOT_TESTED | Not executed by bounded schedule |
| 0 | 2 | 1 | 2 | NOT_TESTED | Not executed by bounded schedule |
| 1 | 0 | 0 | 0 | NOT_TESTED | Not executed by bounded schedule |
| 1 | 0 | 0 | 1 | SUPPORTED / PASS | `test_builder_graph_compatibility::test_b_e_risk_pairwise_cnn_combinations[1-0-0-1]` |
| 1 | 0 | 0 | 2 | NOT_TESTED | Not executed by bounded schedule |
| 1 | 0 | 1 | 0 | NOT_TESTED | Not executed by bounded schedule |
| 1 | 0 | 1 | 1 | NOT_TESTED | Not executed by bounded schedule |
| 1 | 0 | 1 | 2 | SUPPORTED / PASS | `test_builder_graph_compatibility::test_b_e_risk_pairwise_cnn_combinations[1-0-1-2]` |
| 1 | 1 | 0 | 0 | NOT_TESTED | Not executed by bounded schedule |
| 1 | 1 | 0 | 1 | NOT_TESTED | Not executed by bounded schedule |
| 1 | 1 | 0 | 2 | SUPPORTED / PASS | `test_builder_graph_compatibility::test_b_e_risk_pairwise_cnn_combinations[1-1-0-2]` |
| 1 | 1 | 1 | 0 | SUPPORTED / PASS | `test_builder_graph_compatibility::test_b_e_risk_pairwise_cnn_combinations[1-1-1-0]` |
| 1 | 1 | 1 | 1 | NOT_TESTED | Not executed by bounded schedule |
| 1 | 1 | 1 | 2 | NOT_TESTED | Not executed by bounded schedule |
| 1 | 2 | 0 | 0 | SUPPORTED / PASS | `test_builder_graph_compatibility::test_b_e_risk_pairwise_cnn_combinations[1-2-0-0]` |
| 1 | 2 | 0 | 1 | NOT_TESTED | Not executed by bounded schedule |
| 1 | 2 | 0 | 2 | NOT_TESTED | Not executed by bounded schedule |
| 1 | 2 | 1 | 0 | NOT_TESTED | Not executed by bounded schedule |
| 1 | 2 | 1 | 1 | SUPPORTED / PASS | `test_builder_graph_compatibility::test_b_e_risk_pairwise_cnn_combinations[1-2-1-1]` |
| 1 | 2 | 1 | 2 | NOT_TESTED | Not executed by bounded schedule |
| 2 | 0 | 0 | 0 | NOT_TESTED | Not executed by bounded schedule |
| 2 | 0 | 0 | 1 | NOT_TESTED | Not executed by bounded schedule |
| 2 | 0 | 0 | 2 | SUPPORTED / PASS | `test_builder_graph_compatibility::test_b_e_risk_pairwise_cnn_combinations[2-0-0-2]` |
| 2 | 0 | 1 | 0 | SUPPORTED / PASS | `test_builder_graph_compatibility::test_b_e_risk_pairwise_cnn_combinations[2-0-1-0]` |
| 2 | 0 | 1 | 1 | NOT_TESTED | Not executed by bounded schedule |
| 2 | 0 | 1 | 2 | NOT_TESTED | Not executed by bounded schedule |
| 2 | 1 | 0 | 0 | SUPPORTED / PASS | `test_builder_graph_compatibility::test_b_e_risk_pairwise_cnn_combinations[2-1-0-0]` |
| 2 | 1 | 0 | 1 | NOT_TESTED | Not executed by bounded schedule |
| 2 | 1 | 0 | 2 | NOT_TESTED | Not executed by bounded schedule |
| 2 | 1 | 1 | 0 | NOT_TESTED | Not executed by bounded schedule |
| 2 | 1 | 1 | 1 | SUPPORTED / PASS | `test_builder_graph_compatibility::test_b_e_risk_pairwise_cnn_combinations[2-1-1-1]` |
| 2 | 1 | 1 | 2 | NOT_TESTED | Not executed by bounded schedule |
| 2 | 2 | 0 | 0 | NOT_TESTED | Not executed by bounded schedule |
| 2 | 2 | 0 | 1 | SUPPORTED / PASS | `test_builder_graph_compatibility::test_b_e_risk_pairwise_cnn_combinations[2-2-0-1]` |
| 2 | 2 | 0 | 2 | NOT_TESTED | Not executed by bounded schedule |
| 2 | 2 | 1 | 0 | NOT_TESTED | Not executed by bounded schedule |
| 2 | 2 | 1 | 1 | NOT_TESTED | Not executed by bounded schedule |
| 2 | 2 | 1 | 2 | SUPPORTED / PASS | `test_builder_graph_compatibility::test_b_e_risk_pairwise_cnn_combinations[2-2-1-2]` |

### Explicit pairwise proof

Index order follows the factor lists above (zero based). These six factor-pair groups contain 45 distinct value pairs; all 45 are covered by the 18 measured models. Each source XML ID above maps to the acceptance measurement and exact factor vector in QA/pairwise-test-id-proof.json, also available in the byte-identical installed tests. No new combination test was added at closure.

| Factor pair | Expected / observed distinct values | Covered value pairs |
| --- | --- | --- |
| Convolution × Activation | 9 / 9 | (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) |
| Convolution × Normalization | 6 / 6 | (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1) |
| Convolution × Pooling | 9 / 9 | (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) |
| Activation × Normalization | 6 / 6 | (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1) |
| Activation × Pooling | 9 / 9 | (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2) |
| Normalization × Pooling | 6 / 6 | (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2) |

## Explicit unsupported or untested scope

- Complete model families are not unary Graph nodes or sequential feature blocks. Named families are fixed profiles; custom add() is rejected.
- Development-only names are rejected by the commercial Registry. The installed matrix passes those negative guards; it does not label the missing components SUPPORTED.
- Multiple named Graph inputs run with a dict, but the standard Tensor output adapter supports exactly one input. No new multi-input TrainingSession bridge is claimed.
- A node not contributing to any named output is rejected. A separately named output may be valid while its parameters get no gradient from a different selected loss.
- Generic graph/standard checkpoint support is broader than the closed Graph v2 source/receiver profile. Depthwise is an observed example of valid forward/training but unsupported Graph v2 capture.
- No transposed-Graph skip broadcasting, arbitrary callables/custom imported components, automatic NHWC, automatic dtype conversion, dynamic spatial shapes, CUDA, whole-family training or exhaustive hyperparameter combinations are certified.
