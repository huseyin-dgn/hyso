# Python-first engineering workflow

HYSo is a Python-first PyTorch CNN engineering, training, evaluation, benchmarking and performance analysis framework.

Install Python >=3.10 and a compatible Torch/torchvision pair, then `python -m pip install -e .`. Use `python -m hyso --help` or `hyso --help`. Existing groups are models, train, analysis, compare, advisor, experiments and reports. Dry-run and explicit workload/resource authorization contracts are unchanged.

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

Unknown names and multi-input graphs are rejected. Existing parameters are registered by reference, not copied. Autograd, device and dtype follow the shared graph. Move the model before optimizer creation. All graph outputs still execute. A wrapper state_dict has a `graph.` prefix; load into a matching wrapper or save `model.graph.state_dict()` for an unwrapped graph. The optional task guard supports classification heads with matching multiclass/binary widths. It does not advertise regression/embedding training or multi-loss orchestration.

Pass the selected Tensor module to existing TrainingConfig.prepare/TrainingSession, Trainer, evaluate_classification or BenchmarkRunner. Training changes the shared graph weights. Evaluation's isolated inference policy and benchmark state guards are unchanged. The adapter introduces no seed, dtype cast or device transfer.

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

- [TrainingConfig / TrainingSession](training-config.md): one Trainer.fit, requested/effective settings and unchanged ModularCNN-only automatic-resume scope.
- [Python API map](python-api.md): schema1/2/3 builders, evaluation, checkpoints and measurements.
- [Selective analysis](selective-analysis.md) and [native P01–P35 analysis](native-analysis-executor.md): explicit prerequisites/authorization; selection is not certification.
- [Comparison](two-model-comparison.md), [Advisor](performance-advisor.md), [ExperimentManager](experiment-workflow.md), [Reporting](result-reporting.md): retained evidence, explicit storage and read-only interpretation.

Historical Studio guides are marked retired. Its external SQLite file is preserved and not merged with ExperimentManager. Existing invalid drafts serialize losslessly as JSON data; Core correctly refuses to build them until their diagnostics are resolved. No automatic export, repair, migration or reset occurs. The previously launched server remains in memory until its owner presses Ctrl+C; `python -m hyso.studio` is no longer a supported module entry point.
