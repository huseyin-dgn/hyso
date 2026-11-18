# Python-first engineering workflow

HYSo is a Python-first PyTorch CNN engineering, training, evaluation, benchmarking and performance analysis framework.

Türkçe [kullanım ve kabul sınırları](python-first-acceptance-and-limits.md), bu iş
akışının destek matrisini, strict seçim ve kanıt kimliklerini, kısa evaluation/
Reporting/store tariflerini ve güncel CUDA/B4 sınırlarını bir araya getirir.

Install Python >=3.10 and a compatible Torch/torchvision pair, then `python -m pip install -e .`. Use `python -m hyso --help` or `hyso --help`. Existing groups are models, train, analysis, compare, advisor, experiments and reports. Dry-run and explicit workload/resource authorization contracts are unchanged.

For a built wheel, install `C:/local-hyso/dist/hyso-0.2.0-py3-none-any.whl`
with `python -m pip install` in a separate venv. Public `hyso` imports and CLI
commands work outside the checkout. The `python -m examples...` commands below
require the source checkout; examples are intentionally absent from the commercial
distribution. See the [CPU beta installation scope](python-first-acceptance-and-limits.md#cpu-beta-installation).

## Select a graph output explicitly

```python
from hyso import GraphModelBuilder, TrainingConfig
from hyso.models.output import GraphTensorOutput

builder = (GraphModelBuilder({"image": [1, 4, 4]})
    .add("head", "classification_head", ["image"], in_features=1, num_classes=2)
    .outputs(logits="head"))
graph = builder.build()
model = GraphTensorOutput(graph, "logits", task="multiclass")
```

Unknown output names are rejected. Two-input CPU classification requires
`named_inputs=True`, an explicit task and input dictionaries matching the Graph
names. Existing parameters are registered by reference, not copied. Autograd,
device and dtype follow the shared graph. Move the model before optimizer
creation. All Graph outputs still execute. A wrapper state_dict has a `graph.`
prefix; restore into a matching wrapper. The optional task guard requires
classification heads with matching multiclass/binary widths. Evaluation and
benchmark support remain governed by their own input contracts.

Pass the selected Tensor module to existing TrainingConfig.prepare/TrainingSession, Trainer, evaluate_classification or BenchmarkRunner. Training changes the shared graph weights. Evaluation's isolated inference policy and benchmark state guards are unchanged. The adapter introduces no seed, dtype cast or device transfer.

<a id="graph-training-and-saved-results"></a>
## Graph training and saved results

The existing [Graph example](../../examples/models/graph_evidence_workflow.py)
now provides these small CPU workflows. Use fresh, local output directories
outside the repository and synchronized folders:

```powershell
# Two named inputs -> TrainingSession -> automatic v2 checkpoint.
python -m examples.models.graph_evidence_workflow --named-training --output C:/local-hyso/named-01
# After interruption, use the last complete epoch and the same total budget (2).
python -m examples.models.graph_evidence_workflow --named-training --resume-from C:/local-hyso/named-01/last.pth --output C:/local-hyso/named-02
# Two trained Graphs -> evaluation -> saved experiment -> comparison/advice/reports.
python -m examples.models.graph_evidence_workflow --output C:/local-hyso/graph-results-01
```

`train_named` builds a Graph with explicit `image`/`hint` inputs, wraps `logits`
with `GraphTensorOutput(..., task="multiclass", named_inputs=True)`, and uses
`NamedTensorDataset` plus `TrainingConfig.make_loaders`. It passes the exact
`builder.to_config()` as `model_config` to `prepare`. The result's `history`,
`applied_updates` and checkpoint path are written to `summary.json`. Resume
restores the original two-epoch budget; a completed checkpoint is a no-op and
may leave the new output directory without a newly written checkpoint.

The evaluation workflow trains two tiny single-input models for one epoch each,
saves/restores Graph v2 checkpoints, and calls `evaluate_graph_classification`
with explicit dataset/split/preprocessing context and `capture_structure=True`.
It saves architecture descriptions and evaluation associations in its own
ExperimentManager store. `compare_saved` reopens the associations through
`manager.open_result`, then calls `analysis_source`, `analysis.compare_models`
with `features=("P26", "P12")`, `include_structure=True`, strict context policy,
`advisor.recommend` and `report_experiment`. Reusing saved results launches no
training, forward or profiler. Architecture descriptions remain metadata;
verified file bytes do not restore process-local Graph receipt authority.

Read `model-*/training.json` for history, `evaluation.json` for accuracy/macro F1
(fractions), and `model.pth` for checkpoints. `saved-results/comparison.json`
contains eligible B-A quality differences and registered parameter/structure
counts; `architectures.json` describes the two Graphs. `advisor.json` contains
manual advice. `saved-results/tr.*` and `en.*` provide JSON/Markdown/HTML.
There is no P12 capture in this example: MACs/FLOPs differences remain `null`
with `INSUFFICIENT_EVIDENCE`. Synthetic quality is not a real-dataset claim;
logical operation counts would not establish device speed or energy use.
Presentations retain `presentation_only=true`, `authoritative_evidence=false`.

For two selected classification losses on a raw Graph, use the
[bounded Trainer recipe](training-config.md#two-named-classification-losses).
One- and two-input multi-loss automatic resume are supported; combined accuracy is undefined.

<a id="multi-head-workflow"></a>
## Multi-head workflow

The same example file provides explicit two-head CPU workflows:

```powershell
# disease (2 classes) + severity (3 classes), weighted loss, 2 epochs, v2 checkpoint.
python -m examples.models.graph_evidence_workflow --multi-head --output C:/local-hyso/heads-A
# Resume only with the same variant, names, loss weights, data and total budget.
python -m examples.models.graph_evidence_workflow --multi-head --resume-from C:/local-hyso/heads-A/last.pth --output C:/local-hyso/heads-A-resumed
# A second explicitly chosen small architecture, with the same evaluation protocol.
python -m examples.models.graph_evidence_workflow --multi-head --variant B --output C:/local-hyso/heads-B
# Saved statements only: no model construction, training, inference or profiler.
python -m examples.models.graph_evidence_workflow --compare-heads C:/local-hyso/heads-A/heads.json C:/local-hyso/heads-B/heads.json --output C:/local-hyso/head-reports
# Combined named inputs/two heads, automatic checkpoint and per-head evaluation.
python -m examples.models.graph_evidence_workflow --multi-head --two-inputs --output C:/local-hyso/named-heads
# Resume a matching combined checkpoint (after interruption, or completed no-op).
python -m examples.models.graph_evidence_workflow --multi-head --two-inputs --resume-from C:/local-hyso/named-heads/last.pth --output C:/local-hyso/named-heads-resumed
# Explicit second combined model and retained-only comparison/reporting.
python -m examples.models.graph_evidence_workflow --multi-head --two-inputs --variant B --output C:/local-hyso/named-heads-B
python -m examples.models.graph_evidence_workflow --compare-heads C:/local-hyso/named-heads/heads.json C:/local-hyso/named-heads-B/heads.json --output C:/local-hyso/named-head-reports
```

`train_heads` constructs the Graph, `NamedTargetDataset` and explicit loss/weight
dictionaries (`disease=.3`, `severity=1.2`), then calls
`TrainingConfig.prepare(..., model_config=builder.to_config(), loss_fn=...,
loss_weights=...).fit()`. Both input modes save/restore the existing strict
v2 checkpoint and explicitly calls `analysis.evaluate_graph_heads`.
The example reuses four synthetic samples for training and evaluation; it is
an API demonstration, with no held-out quality or device-performance claim.
`training.json` holds weighted total history (accuracy series are null);
`heads.json` retains each head's independent evaluation. `summary.json` contains
per-head accuracy/F1/confusion matrices and real checkpoint/store/run/artifact
references. An already completed resume remains a no-op.

The evaluation head specs explicitly supply each output's target field, task,
ordered class_names, split, dataset_id and preprocessing_id. Missing declared
identities can be explicit `None`; they do not become verified provenance.
The existing CPU evaluator runs once per selected head, with isolated copies,
native per-head streams and preserved source state/RNG. Two-input mode uses
explicit `image`/`hint` mappings, and native hashes bind both names and contents;
no positional inference or concatenation is used. This is not a shared
single-forward multi-task evaluator. Binary and multiclass heads keep separate
label spaces. No overall_accuracy is generated.

The example records training and `graph_multi_head_evaluation` generic artifacts
in its own ExperimentManager store, then uses `manager.open_result`. SHA checks
verify saved bytes. A raw Graph checkpoint receipt is kept separately and is
never transferred to per-head Tensor adapters. Imported results show
`graph_v2=unverified`; architecture metadata and artifact hashes cannot restore
process-local authority.

`compare_saved_heads` reads the two files through
`MultiHeadGraphEvaluationResult.from_json` and calls
`analysis.compare_graph_heads(..., head_pairs={'disease':'disease',
'severity':'severity'})`. That adapter uses existing `compare_models` results,
strict context gates and native P26/structure checks. Pairing is explicit and
currently requires equal output names; semantic head renaming is unsupported.
Equal names alone do not establish eligibility. Label/stream/protocol mismatch
keeps INCOMPATIBLE; missing context/evidence keeps INSUFFICIENT_EVIDENCE.
Each selected adapter executes the full Graph. Registered structure snapshots
cover that full Graph, including both heads; they are not head-only parameter
or execution counts. Repeated structure values across head pairs must not be summed.

Each head has its own `*-comparison.json` and `*-advisor.json`. TR/EN
JSON/Markdown/HTML reports retain the same values, units and eligibility, and
keep `presentation_only=true`, `authoritative_evidence=false`. Existing
`hyso reports render` can reopen the saved presentation without a workload.
No P12 profiler runs; absent native MACs/FLOPs differences remain null. Native
P12 comparisons on separately retained model-level sources keep their existing
API and scope; head metadata does not supply operation-count evidence.

## Explicit user-model checks

```python
from hyso.model_checks import preflight_model, run_model_checks, count_parameters

config = builder.to_config()
static = preflight_model(config)  # No allocation or forward.
checked = run_model_checks(config, model=graph, model_id="my-cnn",
    tests=("structure", "forward", "backward", "serialization"))
exact = count_parameters(config)  # Explicit CPU construction; no forward.
```

Checks use concrete schema1/2/3 geometry and synthetic zero inputs in CPU eval mode. Admission requires a known Core bound <=1M parameters, <=1M activation elements and <=65536 input elements; these are not process-memory guarantees. Missing geometry or exceeded bounds produces SKIP with a reason. Invalid structures retain Core diagnostics and skip runtime execution. Real failures are FAIL; failed prerequisites are SKIP with the error retained. Backward differentiates the sum of outputs, not a classification loss, and does not prove full parameter coverage. Strict state_dict roundtrip checks every output on this fixture.

A supplied model must match the configuration and already be on CPU. An alias-checked deepcopy is tested; original weights, buffers, gradients, modes and CPU RNG remain unchanged. Results retain configuration/model/state context and individual case reasons. Fresh checks use seed-0 weights; trained weights are tested only when explicitly supplied. Custom Python side effects are not sandboxed. Model checks do not execute pytest, CUDA, native P01–P35 acceptance or long training.

## Architecture, evidence and advice

`hyso.architecture_comparison.compare_architectures([config_a, config_b], exact_parameters=True)` compares 2–32 descriptions. It reports input/output contracts, nodes/connections, branches, mechanisms/categories, conservative bounds, explicit exact CPU counts and diagnostics. Default comparison allocates no weights. Invalid models remain unchanged and have no runtime counts. Missing measurements are never replaced with zeros.

Existing `capture_model_structure` / `compare_model_structures` compare registered structure snapshots. Existing **binary** `analysis.compare_models` retains its evidence-binding and compatible-condition rules. An architecture result does not imply measured speed, quality or a global winner.

`hyso.model_observations.observe_result(native_result, config=..., model_id=...)` binds native results to explicit caller context in memory. `diagnose_results` retains conservative history/gradient/test/benchmark threshold observations. `observe_model_checks` binds the new check result. `explain_diagnostic` interprets Core diagnostics; `suggest_model_corrections` validates copied channel candidates without applying them. No database, capture or workload is started. Labels/times/dataset declarations do not prove authenticity. Existing Advisor recommend/advise remains the primary evidence-linked API.

## Runnable A–I workflow

```powershell
python -m examples.models.python_first --output C:/local-hyso/python-example-01
```

Use a fresh local output directory outside synchronized folders. This explicitly runs linear CNN construction; branching graph/output selection; one CPU training epoch; **actual inference** classification evaluation; three measured benchmark iterations; checks of isolated trained weights; architecture/registered-structure comparisons; ExperimentManager metrics/artifact retention; native analysis/Advisor/report generation. It creates and explicitly migrates only its own new empty example store. No previous user workspace is accessed. Synthetic accuracy is not a quality claim.

The 60 preserved component JSON examples in `examples/models/component-configs.json` all preflight. The [61-mechanism reference](component-engineering-reference.md) preserves source-backed engineering prose, including the complete model family.

## Existing APIs and limits

- [TrainingConfig / TrainingSession](training-config.md): one Trainer.fit, requested/effective settings, ModularCNN v1 and bounded Graph v2 automatic resume.
- [Python API map](python-api.md): schema1/2/3 builders, evaluation, checkpoints and measurements.
- [Selective analysis](selective-analysis.md) and [native P01–P35 analysis](native-analysis-executor.md): explicit prerequisites/authorization; selection is not certification.
- [Comparison](two-model-comparison.md), [Advisor](performance-advisor.md), [ExperimentManager](experiment-workflow.md), [Reporting](result-reporting.md): retained evidence, explicit storage and read-only interpretation.

Historical Studio guides are marked retired. Its external SQLite file is preserved and not merged with ExperimentManager. Existing invalid drafts serialize losslessly as JSON data; Core correctly refuses to build them until their diagnostics are resolved. No automatic export, repair, migration or reset occurs. The previously launched server remains in memory until its owner presses Ctrl+C; `python -m hyso.studio` is no longer a supported module entry point.
