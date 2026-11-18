# Pythonic builder examples

After installing HYSo, `python examples/models/pythonic_builders.py` builds a small CNN, builds a graph,
and prepares a CPU training session. It never calls `fit()`. The three functions
can also be imported and called separately.

```python
import torch
from torch.utils.data import TensorDataset
from hyso import ModelBuilder, GraphModelBuilder, TrainingConfig

# 1. A CNN: new methods delegate to the existing add() implementation.
model = (ModelBuilder((1, 8, 8), 2)
         .conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
         .relu()
         .max_pool2d(kernel_size=2)
         .build())

# 2. A graph: names, connections and outputs stay explicit.
graph = (GraphModelBuilder({'image': [1, 8, 8]})
         .conv_norm_act('stem', ['image'], in_channels=1, out_channels=4,
                        norm=None, activation='relu')
         .max_pool2d('pool', ['stem'], kernel_size=2)
         .add('head', 'classification_head', ['pool'], in_features=4, num_classes=2)
         .outputs(logits='head')
         .build())

# 3. Prepare training with a tiny CPU Dataset; no fit() is called.
train_dataset = TensorDataset(torch.zeros(4, 1, 8, 8, device='cpu'),
                              torch.tensor([0, 1, 0, 1], device='cpu'))
session = TrainingConfig(epochs=1, batch_size=2).prepare_datasets(model, train_dataset)
```

The complete executable file supplies tiny synthetic CPU data for the third
example. No task-quality result is implied. Building initializes weights; it does
not run a forward pass or train. Graph classification output adapters and other
existing task restrictions are unchanged.

Both builders offer `conv2d`, `conv_norm_act`, `relu`, `silu`, `gelu`, `batch_norm`,
`group_norm`, `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d`, and `dropout`.
Component options are explicit keyword parameters. Graph methods first take
`node_id` and `inputs`. All other registered components and fusion operations
remain available through the unchanged `.add(...)` API.

Defaults come from HYSo's Registry: for example, `conv2d` defaults to
`kernel_size=3, padding=1`, while pooling requires a kernel and defaults to
`stride=None, padding=0`. `conv_norm_act.norm` suggests `'batch'`, `'group'`, or
`None`; its activation suggests `'silu'`, `'relu'`, or `None`. `gelu.approximate`
accepts `'none'` or `'tanh'`. The wrappers do not expose unsupported PyTorch
options such as `relu(inplace=...)`, `max_pool2d(return_indices=...)`, or
`avg_pool2d(divisor_override=...)`.

Docstrings and [`api-topics.json`](../../hyso/data/api-topics.json) point to the
same four-language offline guide: `components.convolution`,
`components.activation`, `components.normalization`, `components.pooling`, and
`components.regularization`. For builder structure and training preparation use:

```console
hyso docs show models.model-builder --lang tr
hyso docs show models.graph-builder --lang tr
hyso docs show training.configuration --lang tr
```

## M3 CPU examples

Install HYSo first (`python -m pip install -e .`). From the checkout root:

```console
python examples/models/m3_workflows.py resnet
python examples/models/m3_workflows.py variant --depth 34
python examples/models/m3_workflows.py densenet
python examples/models/m3_workflows.py block
python examples/models/m3_workflows.py attention
python examples/models/m3_workflows.py builder
python examples/models/m3_workflows.py training
python examples/models/m3_workflows.py state_dict
```

All use CPU and tiny synthetic inputs. Training is one two-sample optimizer step;
it is an API example, not a model-quality result. DenseNet uses explicit growth=4
as a small width variant. ResNet examples retain full named topology with a
one-channel input and three-class head. Large depths only perform an eval forward.
The state_dict example uses a temporary directory and strict weights-only load;
no acceptance checkpoint is read or written. The default scenario is Builder.

See [architecture guide](../../docs/guides/model-architectures.md) for defaults,
normalization, JSON contracts and epoch-resume limitations.

## M4 CPU examples

After installing HYSo, run `python examples/models/m4_workflows.py SCENARIO`
from the checkout root. The default scenario is `builder`. Every command uses
one CPU thread, a fixed CPU seed and a synthetic `(2,1,32,32)` input. Family
examples select explicit width `.125` and three classes. EfficientNetV2,
RepVGG and MobileOne use depth `.1`. ShuffleNetV2 uses canonical x0.5.
These small configurations demonstrate APIs and numerical behavior. Model
quality and hardware performance remain NOT_EVALUATED.

| SCENARIO | Purpose and selected parameters | Expected output | Technical boundary |
|---|---|---|---|
| `mobilenet_v2` | Linear bottlenecks, width .125 | `(2,3)` | Derived width configuration |
| `mobilenet_v3_small` | Small SE/activation topology | `(2,3)` | Random initialized Small weights |
| `mobilenet_v3_large` | Large SE/activation topology | `(2,3)` | Distinct Large block sequence |
| `efficientnet_v2` | S FusedMBConv/MBConv, width .125, depth .1 | `(2,3)` | Mini configuration for CPU smoke |
| `ghostnet` | V1 Ghost features and SE | `(2,3)` | V1, HYSo-specific state keys |
| `shufflenet_v2` | Canonical x0.5 split/shuffle | `(2,3)` | Custom one-channel stem and three-class head |
| `repvgg` | A0-derived eval graph, explicit copy conversion and strict deploy reload | `(2,3)`, `deploy max_abs` within tolerance | Conversion uses current BN running statistics |
| `mobileone` | s0-derived multi-branch graph and explicit conversion | `(2,3)`, `deploy max_abs` within tolerance | Train/deploy topology is explicit |
| `mbconv` | MBConv then FusedMBConv with exact external channel counts | `(2,8,32,32)` | Operators retain torchvision expansion rounding |
| `dropblock` | p .2, block 3, training mask and eval identity | `(2,1,32,32)`, finite and identity messages | Explicit stochastic regularization |
| `builder` | MBConv, GhostConv and explicit DropBlock | `(2,3)` | Sequential schema1 configuration |
| `state_dict` | Separate JSON and weights files with strict weights-only reload | `state_dict strict roundtrip OK` | Temporary example artifacts, no optimizer/epoch resume |

For example, run `python examples/models/m4_workflows.py repvgg`. All twelve
scenario entry points are covered by short execution tests. The training
integration tests separately exercise one optimizer step for each included
family using TrainingSession.

FasterNet and PartialConv were excluded following the
[license audit](../../docs/reports/stage13-license-audit.md). Their source/configuration
license scope remains unresolved. Existing M3 examples remain available above.
The [M4 architecture guide](../../docs/guides/mobile-edge-architectures.md)
documents canonical variants, constructor ranges, fusion and serialization.
