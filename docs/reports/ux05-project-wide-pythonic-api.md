# HYSo UX-05 ? project-wide Pythonic API

Final CPU/API/packaging verification: **PASS**.

Branch: `feature/hyso-models-next`. Starting HEAD: `7e954d1a3d2bf1bfd2d9afe44500a3c3e2d2ece5`.
No AGENTS.md was found in the repository or its ancestor directories. The UX-05
attachment was read in sections; existing UX-01/02/03/04/04B work was retained.

## Audited scope

- Original UX-03: **775 source declarations / 1,643 aliases**, all retained.
- Pre-UX-05 help inventory: **933** source declarations.
- Current help inventory: **956** source declarations / **2163** entries.
- Conservative source review: **1154** declarations across **110** source files, including **198** additional named source helpers.
- **94** validator candidates classified; no unreviewed candidate remains.
- **354** new/changed parameter or field annotations across **61** files; **215** changed/new closed-selector sites.

The review covers builders and registered factories; TrainingConfig, TrainingSession
and Trainer; Analyzer planning, execution, retained sources and P01?P35 routing;
model/architecture/structure comparisons; Advisor; Reporting; Evaluation and Quality;
Benchmark and performance protocols; ExperimentManager/workflows; augmentation,
callbacks, checkpoint/export/import helpers; and offline help. CLI parsers retain
their text-interface contract. There is no new algorithm or dashboard.

| Preserved explicit Builder coverage | Count |
| --- | ---: |
| Development | 85/85 |
| Commercial | 57/57 |
| Graph fusion | 4/4 |

`tools/docs/audit_pythonic_api.py` regenerates the source/alias matrix without
importing HYSo, torch or workload code. Full per-API decisions, signatures and
closed options are retained in `outputs/ux05-project-wide-pythonic-api/`:
`public-api-audit.json`, `closed-options.json`, `changed-api-annotations.json`,
`validator-candidate-decisions.json`, and `registry-options.json`.

## Added methods and typed choices

- TrainingConfig: `with_sgd`, `with_adam`, `with_adamw`, `with_step_scheduler`,
  `with_cosine_scheduler`, `with_onecycle_scheduler`, `with_plateau_scheduler`,
  `with_early_stopping`. Immutable replacements use the original `from_dict`
  validator and schema. Optimizer LR is retained when omitted. No preparation or fit occurs.
- ModelBuilder: `classification`, `regression`, `embedding` expose existing schema2 heads.
- Device: `cpu()` and `cuda(index)` produce plain strings without hardware queries,
  allocation or admission. Dynamic indices remain native strings.
- Twelve native constructors expose fixed parameters: eleven advanced residual,
  spatial-attention and prediction-head constructors plus legacy CNNSENormalWithFit.
  Shared native engine bodies are retained; child constructors only delegate.
- Graph preview and `measure_stages` expose their existing delegated keyword arguments.
- `GraphHeadSpec` is a typed plain dictionary for existing named-head P26 evaluation.
- Literal choices cover registry names/categories by profile; model normalization,
  activation, attention, variants and heads; training optimizers/schedulers/tasks/
  precision; P01?P35 identifiers, analysis profiles/modes/capabilities/goals; comparison
  policies/axes; Advisor contracts/units; experiment stages/statuses/roles/policies;
  evaluation predictions/tasks and quality/memory metrics; performance axes/precision/
  transfer/streaming/roofline policies; report language/level/audience; and all 70
  offline topic identifiers with independent four-language help options.
- Existing Advisor Metric/NativeMetric and other Enums are reused, including the
  resolver and goal-context metric parameters. No new Enum serialization is introduced.

New public methods/types are mapped to existing help topics in api-topics.json.
Constructor hover text and examples use the real signatures. The short guide is
`docs/guides/ux-05-pythonic-api.md`; the runnable example is `examples/pythonic_api.py`.

## Remaining intentional exceptions

Paths, model/experiment names, Graph node/output/target identities, labels, dataset
and split identities, user metadata and external runtime/provider identities remain
strings. Evidence scope labels describe native/caller records rather than selecting
new HYSo operations. Indexed devices remain strings with the Device helper.

Nine variable-keyword entry points remain: six generic Builder/Graph configuration
or output dispatchers, ModelRegistry.create, the internal Fusion constructor and the
PyTorch DataLoader adapter. Explicit Builder/Graph/native methods cover the fixed
operation choices; output names and external loader options are intentionally open.
Native validation helpers are not duplicated as new public engines. Existing legacy
case-insensitive spellings and permissive normalization fallbacks are unchanged.

Technical reports remain TR/EN; offline help remains TR/EN/DE/ES. Missing, unsupported
or insufficient evidence retains its native status and is never converted to zero
or a successful measurement. Analyzer/Advisor read-only semantics, explicit execution,
GPU admission, dataset checks, checkpoint/resume, ModelConfig schemas 1/2/3, JSON and
state_dict behavior remain governed by their original code.

## Verification

Final selected CPU/API/packaging result: **716 passed, 88 subtests passed, 31 deselected**.
The 30 warnings are existing native warnings; there are no test failures or errors.

The selected command and raw log/XML are `final-test-command.json`,
`final-targeted.log`, `final-targeted.xml` and `final-test-summary.json` under the
report output directory. Earlier attempts and their failure records are retained.
A test-only Temp path correction preserves ExperimentManager's synced-root guard;
the application guard was not relaxed. Final temporary test roots are fresh UUID
paths in local Temp, retaining earlier QA directories and artifacts.

Checks include original source/alias coverage; registry/constructor Literal parity;
real signatures and Jedi completion; native validation and immutable configuration;
legacy/new schema and JSON equivalence; CPU state_dict equivalence; unchanged native
function bodies; profile-specific wheel/sdist projection; installed imports, examples,
offline help and topics. The example tests preparation separately from one explicit
tiny CPU fit plus two P11 logical-footprint observations and retained comparison,
Advisor and report creation.

Pylance/PyCharm GUI completion was not tested. No GPU run, P31 7200-second run,
P01?P35 acceptance campaign or full suite was performed. Existing native AMP
FutureWarnings are outside this API-only scope.

## Preservation and commit

The original **47/47** dirty/untracked user files are checked by SHA-256 against the
initial snapshot. They are excluded from staging. No explicit write/delete operation
targeted the prior 19 artifacts or prior QA evidence. Existing distribution archive
placeholders were left in place; OS permissions prevent content hashing of those
placeholders, so no byte-hash verification is claimed for them.

New output is isolated to this UX-05 report directory and fresh local Temp QA roots.
There is no reset, restore, clean, rebase, push or new dependency. The implementation
commit contains this report; its concrete SHA is recorded after commit in
`outputs/ux05-project-wide-pythonic-api/final-summary.json` and the final response.

## Task files (69)

- `docs/guides/ux-05-pythonic-api.md`
- `docs/reports/ux05-project-wide-pythonic-api.md`
- `examples/pythonic_api.py`
- `hyso/__init__.py`
- `hyso/advisor/_recommendation_text.py`
- `hyso/advisor/comparison.py`
- `hyso/advisor/contracts.py`
- `hyso/advisor/goals.py`
- `hyso/advisor/model_comparison_presentation.py`
- `hyso/advisor/planning.py`
- `hyso/advisor/presentation.py`
- `hyso/advisor/profiles.py`
- `hyso/advisor/recommendations.py`
- `hyso/advisor/resolver.py`
- `hyso/analysis/__init__.py`
- `hyso/analysis/catalog.py`
- `hyso/analysis/evidence.py`
- `hyso/analysis/execution.py`
- `hyso/analysis/feature_plan.py`
- `hyso/analysis/graph_evidence.py`
- `hyso/analysis/graph_heads.py`
- `hyso/analysis/model_comparison.py`
- `hyso/analysis/sources.py`
- `hyso/augmentation/augmentation.py`
- `hyso/benchmark/runner.py`
- `hyso/data/api-topics.json`
- `hyso/evaluation.py`
- `hyso/experiments/manager.py`
- `hyso/experiments/workflow.py`
- `hyso/model_checks.py`
- `hyso/models/WithSE.py`
- `hyso/models/advanced.py`
- `hyso/models/architectures.py`
- `hyso/models/bcam.py`
- `hyso/models/blocks.py`
- `hyso/models/builder.py`
- `hyso/models/fit.py`
- `hyso/models/graph.py`
- `hyso/models/m3_residual.py`
- `hyso/models/mobile_architectures.py`
- `hyso/models/modular.py`
- `hyso/models/output.py`
- `hyso/models/registry.py`
- `hyso/models/reparam_architectures.py`
- `hyso/models/sampling.py`
- `hyso/performance/_loader_stage_trace.py`
- `hyso/performance/_loader_task_trace.py`
- `hyso/performance/capacity.py`
- `hyso/performance/cold.py`
- `hyso/performance/devices.py`
- `hyso/performance/inference.py`
- `hyso/performance/memory.py`
- `hyso/performance/pipeline.py`
- `hyso/performance/providers.py`
- `hyso/performance/roofline.py`
- `hyso/performance/roofline_counters.py`
- `hyso/performance/streaming.py`
- `hyso/performance/sweeps.py`
- `hyso/performance/training_memory.py`
- `hyso/performance/transfers.py`
- `hyso/quality.py`
- `hyso/reporting/catalog.py`
- `hyso/reporting/report.py`
- `hyso/training/config.py`
- `hyso/training/trainer.py`
- `hyso_guide.py`
- `setup.py`
- `tests/test_project_wide_pythonic_api.py`
- `tools/docs/audit_pythonic_api.py`
