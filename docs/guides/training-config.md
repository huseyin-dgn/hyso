# TrainingConfig v1: settings and training experience

`from hyso import TrainingConfig` (also `hyso.training.TrainingConfig`) exposes an
immutable `hyso.training_config`, schema_version=1 request. `to_dict()/to_json()`
and `from_dict()/from_json()` round-trip settings only. Unknown/missing schema
fields, duplicate JSON keys, nonfinite numbers, executable objects and oversized
JSON (>64 KiB) are rejected. Paths are explicit data: reading a configuration
never trains, restores weights, writes checkpoints or probes CUDA. `fit()` is
the explicit action that activates requested training/checkpoint paths.

## Supported profile

| Setting | Contract |
|---|---|
| epochs | Positive total budget, including epochs retained by resume |
| batch_size | Positive nominal DataLoader batch; tail follows explicit drop_last |
| optimizer | Native standard SGD, Adam or AdamW, single parameter group |
| learning_rate | Positive optimizer-construction LR; scheduler can subsequently change it |
| optimizer_params | SGD: momentum/dampening/weight_decay/nesterov; Adam/W: betas/eps/weight_decay |
| scheduler | None, step, cosine, onecycle, plateau (fixed mode=min validation loss) |
| precision/device | CPU fp32/fp64; explicit cuda:index fp32/fp64 or amp_fp16 with FP32 model |
| accum_steps/grad_clip | Existing Trainer accumulation and optional positive norm clipping |
| ema_decay | Optional [0,1]; validation uses Trainer.evaluation_model |
| early_stopping | patience/delta/restore_best_weights; requires validation=True |
| validation | Flag and supplied validation dataset/loader must agree |
| shuffle/drop_last | Standard training sampler and tail policy; validation is sequential/non-dropping |
| num_workers | 0 only in this v1 UX profile; original Trainer remains independently available |
| task | multiclass with CrossEntropyLoss, or binary with BCEWithLogitsLoss, native mean/unweighted losses |
| save_last_to/resume_from | Callback-free ModularCNN v1 or bounded selected Graph v2; exact model_config and admitted loaders |

Profile defaults are explicit: optimizer weight_decay=0; SGD momentum/dampening=0,
nesterov=False; Adam betas=(0.9,0.999), eps=1e-8. Step defaults step_size=1,
gamma=0.1; cosine T_max=epochs, eta_min=0. OneCycle requires max_lr, derives
total_steps=epochs*ceil(len(train_loader)/accum_steps), and defaults to pct_start=.3,
div_factor=25, final_div_factor=10000, cycle_momentum=False. It requires at least
two planned updates and a nondegenerate first phase. Plateau uses native defaults
factor=.1, patience=10, threshold=1e-4, cooldown=0, min_lr=0, eps=1e-8, mode=min.
All supplied values are retained in requested settings, never silently coerced.

`config.make_loaders(train_dataset, val_dataset)` creates ordinary finite map-style
loaders without iterating/seeding data. Seed model creation, loader randomness and
training explicitly (e.g. torch.manual_seed); there is no hidden seed reset. Models
must already have the requested device/dtype before `prepare()`. TensorDataset
input/target dtypes and known Builder model shapes/heads are checked without a
forward. Other dataset transformations and custom model output/data contracts are
validated by the original Trainer when batches are consumed, not certified by
configuration alone. Unsupported options such as CPU AMP, BF16, fused optimizers,
arbitrary scheduler classes or custom batch samplers fail clearly in this profile.
CUDA paths delegate to existing Trainer support; this stage exercised CPU only.

## One Trainer, separate architecture and effective settings

```python
config = TrainingConfig(epochs=2, batch_size=2, optimizer="adamw",
                        learning_rate=0.001, validation=True)
train, val = config.make_loaders(train_dataset, val_dataset)
model = builder.build()  # Architecture is not part of TrainingConfig JSON.
session = config.prepare(model, train, val)
print(session.settings())
result = session.fit()   # One native Trainer.fit, no replay or epoch loop here.
print(result.history)
```

`settings()` separates requested settings, resolved profile defaults/planned
update count, actual optimizer groups/LR, scheduler settings and model device/dtype.
`model_config` is a separate sibling, not embedded in the training request.
OneCycle's constructor changes the initial LR (and optional momentum); actual
native values are visible before fit. After resume, Trainer restores checkpoint
state inside fit; prepared values do not claim to be already-restored values.
Use `session.evaluation_model` (the original Trainer property) for inference;
it returns the EMA model when enabled, otherwise the live model.
Changing prepared native settings is rejected before training. A session is
one-shot: failure can leave weights advanced and never triggers an automatic retry.
Custom nn.Module objects can use this profile without Model Registry registration,
or continue to use Trainer directly for supported features outside this UX subset.

## Progress, durations, checkpoints and tracking

`on_epoch` is an optional synchronous Python callback accepting one independent
dictionary, never JSON code. Its loss, accuracy, validation metrics and LR come
from the existing Trainer epoch_end logs. `TrainingResult.history` is a detached
copy of Trainer.history; accuracies are percentages, unavailable validation is None.
`epoch_events[].interval_seconds` measures intervals between native observer
boundaries, including validation/instrumentation and previous observer overhead.
It is not exact GPU/compute time or P19 evidence. `fit_wall_seconds` includes the
whole call, including restore, finalization, observers and checkpoint I/O.

Automatic checkpoint/resume installs no callbacks, preserving the original save
profile. Epoch timings are then unavailable (empty events + explicit reason),
not fabricated from total wall time. Histories may include an unmeasured restored
prefix. File presence after success is reported, not attested as a new write:
a completed resume can be a no-op. Original Trainer still validates checkpoint
identity/schema/loader/epoch compatibility before any continuing updates.
Resume must match the original total epoch budget, architecture/data and native
constructor options, including scalar representations retained by checkpoint v1.
The adapter does not rewrite that plan or override requested options to force a
match. `applied_updates` is the cumulative Trainer counter, including any restored
prefix; it is not the number of new updates performed by this call alone.

BuiltClassifier automatic resume is **deferred**: the current v1 plan validates
standard ModularCNN structure/configuration and TensorDataset loader profiles.
Broadening that profile needs a separately versioned, tested architecture-state
binding. Existing component save/load/state_dict use stays available. With
`ModelBuilder.modular(...)`, pass `builder.to_config()['params']` separately as
`model_config`; the built model remains exactly ModularCNN.

### Selected Graph automatic resume

`GraphTensorOutput` supports CPU fp32/fp64 binary or multiclass automatic resume
using the existing v2 strict Graph contract. Pass the exact Graph config as
`model_config`; preserve topology, selected output, `graph.` state keys and total
epoch budget. The bounded profile admits SGD/Adam/AdamW with no scheduler or
StepLR, no AMP/EMA/early stopping or callbacks. The two-input examples use
accum_steps=1 without clipping.
Single-input resume retains the CHW TensorDataset profile. Two explicit inputs
(CHW or vector shapes) require `named_inputs=True`, an explicit classification
task and exactly `hyso.training.NamedTensorDataset({name: input_tensor, ...}, targets)`.
Names bind inputs, independently of dictionary insertion order. Standard
num_workers=0 loaders preserve the existing sampler/collate restrictions. Changed
input names/shapes/bytes, labels, config, runtime or checkpoint state are rejected;
this does not certify arbitrary datasets/transforms. The
[runnable Graph example](python-first-workflow.md#graph-training-and-saved-results)
shows save and resume with matching settings.

### Two named classification losses

The existing fit loop can train exactly two selected classification heads on a
raw CPU fp32 Graph with one Tensor input or two explicitly named inputs. Use
`Trainer` directly, or pass explicit `loss_fn` and `loss_weights` dictionaries to
`TrainingConfig.prepare` / `TrainingSession`. These live PyTorch losses remain
separate from TrainingConfig JSON. Existing single-loss calls are unchanged:

```python
import torch
from torch.utils.data import DataLoader
from hyso import GraphModelBuilder, Trainer

graph = (GraphModelBuilder({'image': [1, 4, 4]})
    .add('head_a', 'classification_head', ['image'], in_features=1, num_classes=2)
    .add('head_b', 'classification_head', ['image'], in_features=1, num_classes=3)
    .outputs(a='head_a', b='head_b').build())
loader = DataLoader([(torch.zeros(1, 4, 4), {'a': 0, 'b': 2}),
                     (torch.ones(1, 4, 4), {'a': 1, 'b': 0})], batch_size=2)
trainer = Trainer(graph, optimizer=torch.optim.SGD(graph.parameters(), lr=.01),
    loss_fn={'a': torch.nn.CrossEntropyLoss(), 'b': torch.nn.CrossEntropyLoss()},
    loss_weights={'a': .3, 'b': 1.2})
history = trainer.fit(loader, epochs=1)
print(history['train_loss'])  # Sample-weighted mean of .3 * loss_a + 1.2 * loss_b.
assert history['train_acc'] == [None]
```

Use explicit matching output/target/loss/weight names; extra Graph outputs are
not automatically losses. Standard mean, unweighted CrossEntropyLoss (>=2
classes) or BCEWithLogitsLoss (one logit, binary labels) are supported per head.
Weights must be finite and positive. SGD/Adam are supported without AMP,
accumulation, clipping, scheduler, EMA or early stopping. Train/validation losses
are weighted totals; accuracy series are `None`. Head loss types and weights
are retained in `settings()['effective']['named_losses']`; the effective task is
`per_head_classification`. A mixed CE/BCE pair does not become one multiclass task.

Automatic save/resume uses `NamedTargetDataset(input_tensor,
{'a': targets_a, 'b': targets_b})`, exact Graph `model_config` and the original
total epoch budget. Dataset target names must exactly match loss/output names;
their insertion order is immaterial. The existing v2 strict serializer retains
model/optimizer/history/RNG and the named loss/data plan. A continuation digest
guards optimizer/history bytes in this new profile. Missing fields, altered
mapping/weights/config/data or optimizer state fail before continuing updates.
Raw Graph state retains its original keys; GraphTensorOutput checkpoints retain
their existing `graph.` prefix. No prefix conversion or schema downgrade occurs.

For two inputs, pass `NamedTargetDataset({'image': x, 'hint': hint}, targets)`.
Inputs are routed by explicit Graph names. Two-input/two-output automatic resume
uses the same strict v2 profile, with both input names/shapes/dtypes/bytes in the
data plan. Input dictionary insertion order is immaterial; swapping the tensors
assigned to names changes dataset identity. General multi-task, CUDA, regression,
segmentation and distributed training remain outside this bounded profile.

`hyso.analysis.evaluate_graph_heads` separately evaluates two selected heads
using the original classification evaluator. Its current profile requires a
one- or two-input CPU fp32 Graph, a sequential standard NamedTargetDataset loader,
and explicit per-head target/task/class_names/split/dataset_id/preprocessing_id.
It returns separate accuracy, macro precision/recall/F1, confusion matrices,
sample counts and provenance through `result.head(output).evaluation`.
Two-input native streams retain canonical input_names and use
`sha256_ordered_named_tensor_samples_le_v1`; the digest commits names, shapes,
dtypes and ordered tensor samples, independently of batch partitioning. Existing
single-Tensor streams retain their original bytes and method. Receiver-runtime
capture for named inputs remains unsupported. There is no overall_accuracy. See the
[runnable multi-head workflow](python-first-workflow.md#multi-head-workflow).

To persist results, explicitly create/start an ExperimentManager training run
(use a dedicated non-synced store), call `result.log_to(run)`, then finish it.
Use try/finally and manager.close(); the existing manager has no context-manager
API. log_to records non-null loss/accuracy history and observed intervals only;
it does not start/finish runs or rerun training. Tracking failures propagate and
do not mean the workload should be retried. History logging includes any retained
resume prefix, with that provenance label. Checkpoint artifact linking remains
an explicit existing `run.add_artifact(...)` action.

The [README](../../README.md) contains a complete tiny synthetic CPU example.
This UX report is not a new acceptance result. P01–P35 engines, 70/70 evidence,
production checkpoint v1 and legacy public APIs are unchanged.
