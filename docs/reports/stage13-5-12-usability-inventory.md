# 13.5.12 — Public API and usability inventory

This is an inspection and backlog, not a redesigned API. Technical contracts and beginner usability are evaluated separately. No new high-level runtime API was added.

## A. Technical API quality

- The public factories are ModelBuilder and GraphModelBuilder; ModelRegistry supplies immutable metadata and constrained constructors. GraphTensorOutput explicitly selects one output for Tensor workflows.
- Sequential add validates the whole proposed chain transactionally. Graph add validates scalar params/identifier syntax while references, cycles, reachability and geometry complete at preview/from_config/build. This stage distinction is intentional and tested.
- Architecture JSON is bounded plain data without weights/imported callables. Configuration shape is fixed apart from batch N. Model build creates new random weights; matching weights require a seed or checkpoint restore.
- Bilingual structured errors expose code, node/layer context, expected/actual and suggestions. Symbolic BatchNorm statistics may remain warnings until a concrete runtime batch; train N=1,H=W=1 fails while eval can succeed.
- Parameter bounds are not exact counts. For all 72 observed non-model fixtures, measured counts stayed below the Graph/sequential conservative bounds. Whole model families have separate limits and may lack exact metadata-only geometry.
- Source development catalog has 85 entries; installed commercial has 57. The difference is deliberate exclusion of 28 M3/M4 names, not a broken wheel or a runtime module count of 57. The package still contains 164 runtime Python files.
- Classification TrainingConfig accepts multiclass and binary. Regression/embedding heads can be built, differentiated and serialized, but this does not add regression/metric-learning tasks to the classification session.
- Standard checkpoint state support is broader than Graph v2 trusted runtime support. Generic strict restore is not architecture attestation or automatic optimizer/epoch replay. Graph legacy P26 stays missing; explicit supported receiver capture is separate.

## Observed signatures, documentation and annotations

| API | Actual signature | Docstring | Annotated parameter slots | Return hint |
| --- | --- | --- | --- | --- |
| ModelBuilder.__init__ | `(self, input_shape: 'tuple[int, int, int]', num_classes: 'int')` | True | 2/2 | False |
| ModelBuilder.add | `(self, name: 'str', **params: 'Any') -> 'ModelBuilder'` | True | 2/2 | True |
| ModelBuilder.build | `(self) -> 'nn.Module'` | True | 0/0 | True |
| ModelBuilder.sequential | `(input_shape, *, head='classification', **head_params)` | True | 0/3 | False |
| ModelBuilder.modular | `(**params: 'Any') -> 'ModelBuilder'` | True | 1/1 | True |
| ModelBuilder.named | `(name: 'str', **params: 'Any') -> 'ModelBuilder'` | True | 2/2 | True |
| ModelBuilder.from_config | `(config: 'dict[str, Any]') -> 'ModelBuilder'` | True | 1/1 | True |
| ModelBuilder.to_config | `(self) -> 'dict[str, Any]'` | True | 0/0 | True |
| ModelBuilder.from_json | `(path: 'str / Path') -> 'ModelBuilder'` | True | 1/1 | True |
| ModelBuilder.to_json | `(self, path: 'str / Path', *, overwrite: 'bool' = False) -> 'None'` | True | 2/2 | True |
| ModelBuilder.preflight | `(config: 'dict[str, Any]', *, batch_size: 'int' = 1, activation_limit: 'int' = 16777216) -> 'dict[str, Any]'` | True | 3/3 | True |
| GraphModelBuilder.__init__ | `(self, inputs)` | True | 0/1 | False |
| GraphModelBuilder.add | `(self, node_id, component, inputs, **params)` | False | 0/4 | False |
| GraphModelBuilder.outputs | `(self, **references)` | False | 0/1 | False |
| GraphModelBuilder.build | `(self)` | False | 0/0 | False |
| GraphModelBuilder.preview | `(self, **options)` | False | 0/1 | False |
| GraphModelBuilder.preflight | `(config, *, batch_size=1, activation_limit=16777216)` | False | 0/3 | False |
| GraphModelBuilder.from_config | `(config)` | False | 0/1 | False |
| GraphModelBuilder.to_config | `(self)` | False | 0/0 | False |
| GraphModelBuilder.from_json | `(path)` | False | 0/1 | False |
| GraphModelBuilder.to_json | `(self, path, *, overwrite=False)` | False | 0/2 | False |
| GraphTensorOutput.__init__ | `(self, graph, output_name, *, task=None)` | True | 0/3 | False |
| GraphTensorOutput.forward | `(self, inputs)` | True | 0/1 | False |
| TrainingConfig.__init__ | `(self, *, epochs: 'int' = 5, batch_size: 'int' = 32, optimizer: 'str' = 'adamw', learning_rate: 'float' = 0.001, optimizer_params: 'dict[str, Any] / None' = None, scheduler: 'str / None' = None, scheduler_params: 'dict[str, Any] / None' = None, device: 'str' = 'cpu', precision: 'str' = 'fp32', accum_steps: 'int' = 1, grad_clip: 'float / None' = None, ema_decay: 'float / None' = None, early_stopping: 'dict[str, Any] / None' = None, validation: 'bool' = False, shuffle: 'bool' = False, drop_last: 'bool' = False, num_workers: 'int' = 0, task: 'str' = 'multiclass', save_last_to: 'str / None' = None, resume_from: 'str / None' = None) -> 'None'` | True | 20/20 | True |
| TrainingConfig.prepare | `(self, model: 'nn.Module', train_loader: 'DataLoader', val_loader: 'DataLoader / None' = None, *, model_config: 'dict[str, Any] / None' = None, on_epoch: 'Callable[[dict[str, Any]], None] / None' = None) -> 'TrainingSession'` | True | 5/5 | True |

Class-level documentation may explain a constructor whose method docstring is absent. These are annotation/docstring observations, not proof of static type-checker completeness.

## B. Beginner experience

A minimal supported recipe uses one import, a constructor and three fluent calls:

```python
from hyso import ModelBuilder
model = (ModelBuilder((1, 8, 8), 3)
         .add("conv2d", in_channels=1, out_channels=4)
         .add("relu")
         .build())
```

Graph recipes require stable node IDs, separate component names, ordered references, declared outputs and an explicit Tensor output selection. This is more verbose but avoids implicit projection/broadcasting or an ambiguous output choice.

| Existing example function | Physical source lines (including validation/branches) | Call expressions |
| --- | --- | --- |
| sequential | 10 | 7 |
| branching | 10 | 7 |
| measured_probe | 42 | 51 |
| run | 308 | 144 |

The complete acceptance script imports 31 names, including artifact/audit tooling; this is not the minimum import burden for merely defining a model. End-to-end provenance setup is substantially longer than the small model recipe.

## Deferred simplification backlog

| Priority | Observed friction | Future work, not implemented here |
| --- | --- | --- |
| High | Source 85 vs commercial 57 discovery is easy to misread | Show installed profile/capabilities prominently in discovery and recipes |
| High | Sequential add and Graph add reject invalid state at different phases | Document a common validation-phase vocabulary; consider a guided finalized-config step |
| High | Model config schema3, evaluation schema3, checkpoint v2 and association v2 are distinct formats | Provide a compact version/scope reference without silently converting old evidence |
| High | Graph builder produces dict outputs; classification workflows need Tensor selection | A future convenience may combine selection and task validation explicitly; preserve multi-output semantics |
| Medium | in_channels/in_features/output class widths repeat information | Consider validated optional inference helpers; do not remove explicit current guards |
| Medium | Graph method annotations/docstrings are sparse compared with ModelBuilder | Improve hints and task-focused examples in a later coherent API documentation pass |
| Medium | Statistics errors can surface at runtime despite valid symbolic preflight | Present batch-dependent warnings close to the concrete loader/batch configuration |
| Medium | Some runtime Graph statistics errors use broad shape/error codes | Make diagnostics more specific without changing acceptance or native authority |
| Medium | Json architecture, weights, training state and trusted receipts have different restore paths | A future workflow guide should show ownership and scope instead of implying a universal restore |
| Medium | ExperimentManager migration/artifact/provenance setup is lengthy in a standalone example | Consider documented reusable recipes; no alternate manager/storage added here |
| Low | Functionality exists under several qualified namespaces and aliases | Add a task-to-import reference and profile-specific availability; no mass rename |

## Limits and missing interfaces

- Linear Builder has no node IDs/external skip/concat; repeated component names are legal sequential blocks. Graph is the supported DAG interface.
- No arbitrary callable/plugin/dotted-import JSON, automatic NHWC handling, automatic dtype/device casting or symbolic spatial dimensions were found in these closed Builder APIs.
- Sigmoid/tanh/softmax are not standalone Registry names in this catalog even though Torch/attention internals may use those operations. Do not infer a Builder entry from an internal operator.
- Multi-input raw Graph works with dict input. The existing single-Tensor adapter rejects it; no general multi-input classification loader bridge is certified.
- Legacy public model classes and internal metadata/base classes are not automatically Builder components. Full owned class definitions are listed separately in the component inventory.
- The five new examples (basic, multilayer, branching, invalid, end-to-end) executed. This does not claim every historical example script or every constructor combination was executed.
- No broad API simplification, GUI, service, new component or automatic repair was implemented.

## Whole-workflow simplification inventory for a later stage

These observations include the actual CPU example, existing guides and selected dataset/CLI regressions. They are a concrete backlog, not a second API implementation or a claim that every interface was measured with the same recipe-line metric.

| Area | Current real entry point / user burden | Deferred improvement |
| --- | --- | --- |
| Model creation | `from hyso import ModelBuilder`; CHW, channels and output width are explicit | Short profile-aware task recipes; optional guarded dimension inference |
| Graph creation | `from hyso import GraphModelBuilder`; node IDs, reference lists and named outputs repeat context | Guided examples and consistent finalization/validation vocabulary |
| Dataset loading | `from hyso.datasets import load_image_splits`; channels, split paths, provenance and loader seed are separate concerns; synthetic example uses Torch TensorDataset/DataLoader explicitly | A task-to-dataset contract guide and a single documented provenance handoff; preserve split/class checks |
| Training | `from hyso import TrainingConfig`; `prepare(model, loader).fit()` plus model/task/loader settings; batch size appears in loader and config | Show which settings own the effective loader, and reusable short-session recipes |
| Evaluation | `evaluate_classification` versus qualified `evaluate_graph_classification`; class names, split/context, checkpoint digest and receiver opt-in must be supplied appropriately | A compact entry-point/scope decision guide with explicit opt-in and task boundaries |
| Model Comparison | Architecture comparison, registered-structure comparison and P26 `ModelComparisonSource`/EvidenceIndex have different purposes and prerequisites | One task-to-comparison guide with examples and required evidence, retaining different authority levels |
| Reporting | `from hyso.reporting import create_report`; several result types and JSON/Markdown/HTML exporters | Reusable result-to-report recipes and a documented supported-input matrix |
| Checkpoint | Qualified save/load functions require fresh model ownership, expected digest and optionally metadata/evidence guards | Explain architecture, weights, training state and process-local receipt lifetimes together |
| ExperimentManager | `from hyso import ExperimentManager`; create/close/migrate/reopen, experiment/run lifecycle, artifact registration and verification add substantial code | A standard lifecycle recipe that preserves explicit schema migration and user-store ownership |
| CLI | `hyso models`, `train`, `analysis`, `experiments`, `compare`, `advisor`, `reports`; command-specific files and factory wiring must match Python contracts | End-to-end command recipes and common context/schema help; do not silently launch workloads from discovery |
| Error messages | Structured bilingual diagnostics help; batch-dependent statistics and some broad Graph errors still need concrete runtime context | Show phase, node, actual/expected values and likely correction together |
| Documentation | Source/commercial profiles and several independent schema versions are easy to confuse | Version/profile glossary, import map and cross-linked beginner-to-technical examples |

No model, dataset, training, evaluation, comparison, report, checkpoint, storage or CLI API was redesigned during this closure.
