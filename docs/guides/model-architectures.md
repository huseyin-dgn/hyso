# M3 residual, dense and attention architectures

Use explicit imports from `hyso.models`. No pretrained weights are downloaded.
All families return unnormalized `[N,num_classes]` classification logits and
follow normal `.to(device/dtype)` semantics. Registry/Builder construct on CPU.
Tensor inputs must match the configured channels and model dtype/device.

## Inventory and reuse

| Existing implementation | Retained behavior |
| --- | --- |
| `ConvNormAct` | Conv → batch/group/identity norm → SiLU/ReLU/identity |
| `ResidualBlock` | Two 3x3, optional attention, normalized projection, post-add SiLU |
| SE / CBAM | Existing linear channel gates; CBAM adds spatial attention |
| ECA / Coordinate Attention | Existing channel Conv1d / height-width descriptors |
| ELO | CBAM bottlenecks, optional depthwise path, SiLU, custom head |
| CEMO | SE depthwise bottlenecks, SiLU, optional EMA/training conveniences |
| ModularCNN | Existing configurable residual stages and original state layout |

ELO/CEMO/ModularCNN are not relabeled as canonical ResNet/DenseNet. Their source
and state layouts stay intact except that standalone legacy ResidualBlock now
also accepts the three new attention types. ModularCNN's existing string choices
remain unchanged. New attention can be composed through ModelBuilder or modules.

## Model families

| Registry name / Python class | Supported topology/options |
| --- | --- |
| `resnet` / `ResNet` | depth 18,34: BasicBlock; 50,101,152: expansion-4 bottlenecks; input_channels=3, num_classes=1000, norm=batch/group/None |
| `preact_resnet` / `PreActResNet` | same five stage-count layouts; base_channels=64 (1..128), input_channels=3, num_classes=1000, norm=batch/group/None |
| `wide_resnet` / `WideResNet` | CIFAR basic-block depth=28, widen_factor=10; depth 6n+4 in 10..100, factor 1..16, dropout=0 in [0,1), input_channels=3, num_classes=10, norm=batch/group/None |
| `resnext` / `ResNeXt` | depth 50 or 101, groups=32, width_per_group=4; groups/width 1..64, input_channels=3, num_classes=1000, norm=batch/group/None |
| `densenet` / `DenseNet` | depth 121,169,201; growth_rate=32 (1..64), dropout=0 in [0,1), input_channels=3, num_classes=1000 |

ResNet stage counts are 18:[2,2,2,2], 34/50:[3,4,6,3],
101:[3,4,23,3], 152:[3,8,36,3]. ResNet/ResNeXt reuse torchvision classes,
7x7 stride-2 stem, maxpool, global pooling and linear head. Bottleneck downsample
stride is on the 3x3 convolution: **ResNet v1.5**, not the paper's v1 placement.
The usual ResNeXt-101 32x8d requires explicit width_per_group=8; default 101 is
32x4d. Group/width changes are declared variants, not equivalent checkpoints.

DenseNet reuses torchvision DenseNet-BC: layouts [6,12,24,16], [6,12,32,32],
[6,12,48,32], bottleneck factor 4 and transition compression 0.5. Initial width
is 2*growth_rate. Default growth=32 preserves the standard named topology;
custom growth is a documented width variant. DenseNet retains BatchNorm and
requires spatial dimensions sufficient for three transition pools (use >=32).

PreActResNet uses BN/ReLU **before** convolution, raw identity or projected
activated input, no post-add activation, then final norm/ReLU/global pooling.
It has a 7x7/maxpool stem without stem norm/activation and uses 3x3 bottleneck
stride. This explicit full-preactivation implementation is not a claim of
compatibility with every paper/repository's v2 checkpoint or CIFAR stem.
WideResNet uses a 3x3 stride-1 stem and 16k/32k/64k stages with final norm/ReLU;
projection uses activated input. It is not torchvision `wide_resnet50_2`.

```python
from hyso.models import ResNet, DenseNet, ResNeXt, PreActResNet, WideResNet
model = ResNet(depth=34, input_channels=1, num_classes=7)
dense = DenseNet(depth=121, input_channels=1, num_classes=7, growth_rate=4)
```

## Residual blocks

Registry names: `basic_residual`, `bottleneck`, `preactivation`, `wide_residual`,
`resnext_bottleneck`. Every block takes positive `in_channels`, `out_channels`,
stride 1/2 and norm=batch/group/None. Output channels always mean **expanded
output width**. Bottleneck/ResNeXt and preactivation(bottleneck=True) require
out_channels divisible by 4. Basic blocks have expansion 1. PreAct/Wide support
dropout; preactivation defaults to a two-3x3 basic block.

BasicResidualBlock and Bottleneck reuse torchvision's ReLU implementations,
adding automatic projection on stride/channel mismatch. ResNeXtBottleneck
defaults groups=32,width_per_group=4; its inner width is
floor((out_channels/4)*width_per_group/64)*groups and must be nonzero.
GroupNorm uses gcd(8,C) groups. Native batch/group statistical requirements
apply; there is no silent training/eval fallback. Spatial outputs use ceil(H/s).

## Attention

| Registry name / class | Contract and limits |
| --- | --- |
| `simam` / `SimAM` | channels, epsilon=1e-4 >0 finite; parameter-free spatial energy gates; singleton H*W uses variance zero |
| `triplet` / `TripletAttention` | channels, odd kernel_size=7, spatial=True; rotated C/H, C/W and H/W gates averaged equally; BN retained |
| `selective_kernel` / `SelectiveKernelAttention` | channels, branches=2 (2..4), groups=1 dividing C, reduction=16, minimum=32; dilated 3x3 branch fusion |

SimAM's variance denominator is H*W-1 (clamped to one for singleton input).
Half/bfloat16 statistics accumulate in FP32; scale normalization avoids squared
overflow while preserving the energy formula. Triplet is **convolutional**
Rotate-to-Attend, not transformer triplet attention. SK performs split/fuse/
select via channelwise softmax over branches. Its squeeze uses linear/ReLU
without descriptor BN, an explicit batch-one-compatible SK variant. It is not
an entire SKNet and not claimed identical to the Caffe reference. Branch BN still
requires sufficient train statistics. No attention performance gain is asserted.

```python
from hyso import ModelBuilder
builder = (ModelBuilder((1,32,32), 3)
           .add('basic_residual', in_channels=1, out_channels=8, stride=2)
           .add('simam', channels=8))
model = builder.build()
family = ModelBuilder.named('resnet', depth=18, input_channels=1, num_classes=3)
```

## JSON, training and serialization

Existing `hyso.model_config` schema1 sequential and modular kinds are unchanged.
The additive closed kind `registered_model` carries exactly format,
schema_version, kind, name, params; no callable/import path is accepted.
`ModelBuilder.named(...).to_json(path)` saves normalized architecture only.
`from_json(...).build()` creates fresh untrained weights, not a trained model.
Sequential chains retain the conservative 10M parameter budget. Named families
use a separate conservative 200M estimate before allocation. These are parameter
bounds, not hard limits on activations, memory, runtime or arbitrary user code.

```json
{"format":"hyso.model_config","schema_version":1,"kind":"registered_model","name":"resnet","params":{"depth":18,"input_channels":1,"num_classes":3,"norm":"batch"}}
```

CLI discovery/validation needs no model allocation:

```console
hyso models list
hyso models show resnet --json
hyso models validate architecture.json
hyso models summary architecture.json
```

TrainingSession accepts these models as ordinary external nn.Modules with real
loaders and matching dtype/device. Existing config/Manager/report formats are
unchanged. Automatic Trainer `save_last_to`/`resume_from` remains **ModularCNN
only**; new families are not silently admitted to its epoch/RNG/loader resume
profile. Save/load `state_dict` explicitly with an identical architecture and
`strict=True`; save the architecture JSON separately. Use weights_only=True
for Torch load. State dicts contain weights/buffers, not optimizer, loader, RNG
or complete training-resume state. Top-level legacy wildcard exports remain five
names; new names use explicit `from hyso.models import ...` imports.

Canonical torchvision state-key/shape compatibility is tested for unchanged
ResNet18/DenseNet121 configurations. Different channels/classes/norm/widths and
PreAct/Wide architectures must not be treated as interchangeable weights.
Torchvision internals remain a version dependency; exported raw model pickles,
TorchScript, ONNX and pretrained weight adaptation are not certified here.

Working CPU examples: [examples/models](../../examples/models/README.md).

Primary references: [torchvision ResNet](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/resnet.html),
[DenseNet](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/densenet.html),
[full preactivation](https://arxiv.org/abs/1603.05027),
[WideResNet authors](https://github.com/szagoruyko/wide-residual-networks),
[SimAM authors](https://github.com/ZjjConan/SimAM),
[convolutional Triplet authors](https://github.com/landskape-ai/triplet-attention),
[Selective Kernel paper](https://arxiv.org/abs/1903.06586).
