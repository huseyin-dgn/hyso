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
| Convolution | conv_norm_act, conv2d | HYSo ConvNormAct; torch.nn.Conv2d primitive already used by HYSo |
| Residual | residual | Existing ResidualBlock, including its own internal projection/addition |
| Attention | se, cbam, eca, coordinate | SEBlock, CBAMBlock, ECAAttention, CoordinateAttention |
| Normalization | batch_norm, group_norm | torch.nn.BatchNorm2d / GroupNorm used by existing blocks |
| Activation | relu, silu, hardswish | Existing-use torch.nn.ReLU / SiLU / Hardswish; out of place |
| Identity | identity | torch.nn.Identity |
| Full model | modular_cnn | Exactly the original ModularCNN class |

The catalog exposes supported constructor subsets, not every native PyTorch
option: no JSON device/dtype, dilation, callable normalization/activation or
in-place activation. Grouped/depthwise/pointwise convolutions are configurations
of existing convolution entries, not newly implemented convolution families.
SE/CBAM keep the original max(1, channels//reduction) hidden width; CBAM/ECA kernels
must be odd. Groups must divide channels. BatchNorm/GroupNorm retain their native
batch/statistics requirements; `eval()` changes BatchNorm behavior normally.

Legacy BottleneckResidualBlock/BottleneckSEBlock and ELO/CEMO training wrappers
remain directly usable but are outside v1's normalized JSON adapters. Their
legacy options/callable constructors are not silently translated. EMA is a
training utility, not a model block. Absent attention/CNN families are not listed.
Flattened-head LayerNorm, dropout/stochastic-depth controls and internal sigmoid
gates are not advertised as selectable v1 NCHW entries. Pooling/Linear are the
fixed classifier head, not free graph nodes. Their existing implementations are
preserved; exclusion from this catalog does not mean they are missing from HYSo.

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
