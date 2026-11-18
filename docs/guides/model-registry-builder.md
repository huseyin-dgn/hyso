# Model Registry / Builder v1

`from hyso import ModelRegistry, ModelBuilder, ModelConfigError` and the equivalent
named `hyso.models` imports are supported. Legacy wildcard exports remain frozen.
`help(ModelRegistry)`, `help(ModelBuilder.add)` and `registry.get(name).to_dict()`
describe the constructors, parameter schema/defaults, category and conditions.
Metadata is immutable; introspection does not construct a model. No plugin loading
or registration of arbitrary Python callables is provided by this JSON profile.

## Actual catalog

| Category | Names | Existing implementation |
|---|---|---|
| Convolution | conv_norm_act, conv2d, pointwise_conv2d, depthwise_conv2d, depthwise_separable_conv2d, spatial_separable_conv2d, factorized_conv2d, conv_transpose2d | HYSo composition over PyTorch convolution operators |
| Residual | residual | Existing ResidualBlock, including its own internal projection/addition |
| Attention | se, cbam, eca, coordinate | SEBlock, CBAMBlock, ECAAttention, CoordinateAttention |
| Normalization | batch_norm, group_norm, channel_layer_norm, instance_norm | Native normalization and explicit NCHW channel LayerNorm |
| Activation | relu, silu, hardswish, leaky_relu, prelu, gelu, mish, hardsigmoid | Closed out-of-place PyTorch activation operators |
| Pooling | max_pool2d, avg_pool2d, adaptive_avg_pool2d, adaptive_max_pool2d, global_avg_pool2d | NCHW spatial reductions |
| Identity | identity | torch.nn.Identity |
| Full model | modular_cnn | Exactly the original ModularCNN class |

The catalog exposes documented constructor subsets of native PyTorch operators.
JSON accepts closed component parameters with explicit dilation on
Conv2d and the new convolution entries. Device/dtype remain explicit Module.to()
operations. Activation and normalization choices use fixed Registry entries.
SE/CBAM keep the original max(1, channels//reduction) hidden width; CBAM/ECA kernels
must be odd. Groups must divide channels. BatchNorm/GroupNorm retain their native
batch/statistics requirements; `eval()` changes BatchNorm behavior normally.

Legacy BottleneckResidualBlock/BottleneckSEBlock and ELO/CEMO training wrappers
remain directly usable but are outside v1's normalized JSON adapters. Their
legacy options/callable constructors are not silently translated. EMA is a
training utility, not a model block. Absent attention/CNN families are not listed.
Stage 13.5.2 adds channel LayerNorm with an NCHW public interface, InstanceNorm,
five additional activations, five pooling entries and six convolution entries.
Pooling preserves NCHW and composes with the existing implicit classifier head.
The commercial catalog contains 57 entries including all 48 Stage 13.5.3 entries. The
[component guide](custom-cnn-builder-components.md) describes the complete
parameter, geometry, cost and runtime-statistics contracts.

Stage 13.5.3 adds six generic residual blocks, four attention entries and four
regularization entries. The [advanced guide](custom-cnn-builder-advanced-blocks.md)
documents internal projections, train/eval behavior and mandatory token budgets.

Stage 13.5.4 adds resize, pixel shuffle/unshuffle, restricted tensor utilities and
three explicit heads. ModelBuilder.sequential opts into schema2, preserving the
schema1 implicit classifier. Flatten is supported at the terminal schema2 feature
boundary with pooling=none. The [sampling/head guide](custom-cnn-builder-sampling-heads.md)
documents rank restrictions, serialization and training support.

GraphModelBuilder is a separate schema3 API for user-defined DAGs, explicit skips,
fusion and named multiple outputs. The existing ModelBuilder remains sequential.
The [Graph Builder guide](custom-cnn-graph-builder.md) documents node IDs, shape
contracts, resource limits, preview and CPU execution.

## Architecture and shape contract

`ModelBuilder((C,H,W), num_classes)` appends only linear NCHW blocks. Every add
checks channel continuity, convolution output dimensions, residual stride and
attention compatibility before changing the builder. It rejects unknown fields,
external connections, nested complete models and empty convolution outputs.
The fixed head is AdaptiveAvgPool2d(1) -> flatten -> Linear logits. This is assembly
of existing modules, not a new CNN family or unrestricted graph compiler.

`build()` creates fresh CPU parameters. Input CHW must exactly match the configured
shape; positive batch size can vary. Explicitly move/cast both model and input.
Training on a single-value BatchNorm/GroupNorm domain still raises an explicit
block/statistics error; normal runtime errors do not promise transactional state
restoration. Custom skip, concat, DAG, feature pyramids and arbitrary Python
modules in JSON are unsupported. Supply custom nn.Module objects directly to
Trainer/BenchmarkRunner as before; no registry dependency is imposed on them.

Configuration bounds: 64 chain blocks (or total ModularCNN residual blocks),
configured integer dimensions and inferred chain outputs <=8192, 16 stages,
nesting depth 16, JSON input <=2 MiB and
a conservative estimated parameter bound of 10 million for complete Builder
models. The bound may reject a model smaller than 10 million actual parameters;
it is not a measured memory quota. Direct Registry.create retains constructor
allocation cost and is not an untrusted-code or memory-exhaustion sandbox.

## JSON, weights and checkpoint boundaries

`to_config()` returns a detached dictionary with explicit defaults. `to_json(path)`
refuses existing files by default; `from_json(path)` rejects duplicate keys,
nonfinite numbers, unknown names/parameters/version and non-plain data. No eval,
dynamic class import, pickle or executable factory name is read from JSON.

The versioned architecture has either:

- `format="hyso.model_config", schema_version=1, kind="sequential_classifier"`,
  plus `input_shape`, `num_classes`, `blocks=[{name, params}, ...]`;
- the same format/version with `kind="modular_cnn", params={...}`.

Configuration round trips reproduce architecture, not learned weights or RNG.
Use normal state_dict loading or existing component checkpoint APIs separately.
`ModelBuilder.modular(**params).build()` returns type ModularCNN itself with its
original state_dict layout. Existing Trainer automatic epoch-resume restrictions
remain unchanged: only the standard ModularCNN profile with complete model_config
and compatible loaders is accepted. For that path pass
`builder.to_config()['params']` as Trainer.model_config. BuiltClassifier has no
automatic Trainer resume support; no checkpoint format was extended.

The [README](../../README.md) contains a small executable CPU example. P01–P35
engines and the existing scoped 70/70 acceptance artifacts are unchanged; the new
Builder has its own CPU tests and is not certified by that historical acceptance.
