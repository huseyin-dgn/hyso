# HYSo

PyTorch classification models, model-independent training, explicit performance
analysis, evidence reports and local experiment tracking. Start with Python;
there is no general installed HYSo CLI or graphical UI yet.

## Installation and imports

Metadata minimums: Python >=3.10, Torch >=2.2.2, torchvision >=0.17.2. Install a
matching Torch/torchvision CPU or CUDA pair for your platform; minimum version
bounds do not guarantee all combinations. For this checkout: `python -m pip install -e .`.
Optional `hyso[deployment]` supplies pinned ONNX/ONNX Runtime packages, not a
general deployment converter or automatic CUDA runtime installation.

```python
from hyso import ModularCNN, Trainer, BenchmarkRunner, ExperimentManager
from hyso import performance, analysis, advisor
from hyso.models import ConvNormAct, ResidualBlock, SEBlock, CBAMBlock
```

Existing deep imports remain valid and reference the same objects. Legacy
`hyso.models.__all__` deliberately keeps its five-name published wildcard contract;
newer components are available through explicit named imports.

## Build and train a tiny CPU example

Synthetic data demonstrates API usage, not model quality. Supply compatible
finite train/validation loaders and real labels/splits in your application.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from hyso import ModularCNN, Trainer

torch.manual_seed(0)
model = ModularCNN(input_channels=1, num_classes=2, widths=(4,),
                   repeats=(1,), strides=(1,), norm=None).to("cpu")
inputs = torch.randn(4, 1, 8, 8)
targets = torch.tensor([0, 1, 0, 1])
loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
trainer = Trainer(model, optimizer=optimizer, loss_fn=torch.nn.CrossEntropyLoss(),
                  device="cpu", amp=False)
history = trainer.fit(loader, epochs=1)
assert len(history["train_loss"]) == 1
```

Move the model **before** optimizer creation; Trainer does not move it. Model
outputs are logits; history accuracy is a percentage. Use `evaluation_model`
when EMA is enabled. Legacy EBO/CEMO/CNNSENormalWithFit training APIs remain valid.
`save_last_to`/`resume_from` requires the documented standard ModularCNN profile,
complete model_config and compatible loaders, not arbitrary checkpoint replay.

## TrainingConfig: separate requested and applied settings

TrainingConfig v1 prepares the existing Trainer; it does not add another training
engine. Model architecture, Dataset and DataLoader objects stay outside its JSON.
The following synthetic train/validation split is only a CPU usage example:

```python
from hyso import TrainingConfig, ModelBuilder

config = TrainingConfig(epochs=1, batch_size=2, optimizer="sgd",
                        learning_rate=0.01, validation=True,
                        scheduler="step", scheduler_params={"step_size": 1, "gamma": 0.9})
training_inputs = torch.randn(8, 1, 8, 8)
training_targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
train_batches, val_batches = config.make_loaders(
    TensorDataset(training_inputs[:4], training_targets[:4]),
    TensorDataset(training_inputs[4:], training_targets[4:]))
configured_model = ModelBuilder((1, 8, 8), 2).add(
    "conv_norm_act", in_channels=1, out_channels=4, norm=None).build()
session = config.prepare(configured_model, train_batches, val_batches)
print(session.settings()["effective"])
training_result = session.fit()  # Exactly one native Trainer.fit call.
print(training_result.history["val_acc"])  # Percent; not a quality claim.
# config.to_json("training.json")
# config = TrainingConfig.from_json("training.json")
```

`settings()` separates requested, resolved and effective settings, including native
LR values. The v1 profile supports standard SGD/Adam/AdamW; StepLR, cosine,
OneCycle and validation-loss plateau; fp32/fp64 or explicit CUDA amp_fp16;
accumulation, clipping, EMA and validation-dependent early stopping. Default
weight decay is explicitly **0**; OneCycle can change LR at construction and that
change is reported. No implicit device movement, dtype cast or seed is applied.
CPU AMP is rejected. CUDA paths delegate to Trainer and were not exercised here.

Loss/accuracy come from Trainer.history; observer epoch intervals and total fit
wall time are UX timings, not P19/GPU evidence. Automatic save/resume retains the
callback-free ModularCNN-only v1 profile: BuiltClassifier is not supported for
that path and checkpoint-mode epoch timings are unavailable. Use
`training_result.log_to(run)` to explicitly record history in an already-started
ExperimentManager training run; manage run lifecycle and `manager.close()` yourself.
See [training configuration](docs/guides/training-config.md) for exact scope.

## Explicit performance analysis and result reading

This three-sample CPU example checks usage only: it does not establish tail
latency, independent statistical evidence or full P01?P35 acceptance.

```python
from hyso import BenchmarkRunner, performance

result = BenchmarkRunner(trainer.evaluation_model, device="cpu",
                         example_input=inputs[:1], warmup_iterations=1,
                         measure_iterations=3, repeats=1).run()
data = result.to_dict()
latency = performance.analyze_latency(result, quantiles=(0.5,))
print(data["configuration"]["measure_iterations"])
print(latency.to_dict()["feature_ids"])
# Explicit optional exports:
# result.to_json("benchmark.json")
# result.to_csv("benchmark-summary.csv")
# result.to_samples_csv("benchmark-samples.csv")
```

CUDA, streaming and long runs require explicit calls and their prerequisites.
Read missing/unsupported metric statuses and reasons; do not replace them with zero.

## Select analyses without executing them

```python
from hyso.analysis import ModelContext, FeaturePlanRequest, plan_features

plan = plan_features(FeaturePlanRequest(
    mode="selected", models=(ModelContext(model_id="my-cnn", input_contract=True),),
    features=("P01", "P11")))
assert plan.execution_authorized is False
assert [item.feature_id for item in plan.features] == ["P01", "P11"]
```

Single/selected/automatic/full planning reports prerequisites; it never executes
or authorizes workloads. The common executor adds explicit selection/preflight
and authorization around the existing producers:

```python
analysis_plan = analysis.plan_analysis(
    ModelContext(model_id="my-cnn", input_contract=True), profile="basic")
analysis_dry_run = analysis.preflight_analysis(
    analysis_plan, (analysis.LiveCPUModel("my-cnn", model, inputs[:1]),))
print(analysis_dry_run.to_dict())  # No forward, training or factory import.
# After reviewing support, reasons, workload counts and resource estimates:
# analysis_result = analysis.execute_analysis(
#     analysis_dry_run, authorized_features=("P01", "P02", "P10", "P11"),
#     explicit_authorization=True)
# index = analysis_result.to_evidence_index()
```

Profiles: basic, inference, training, memory, comparison, full. They select IDs,
not permissions. Trusted CPU execution supports P01, P02, P10 and P11; the existing
Windows Job route supports P02/P11. P10 is one process RSS/HWM snapshot with no
model forward or model-only peak claim. Other common adapters are not connected;
P09/CUDA is unsupported here. Dependencies never start automatically.
Default P01 requests 1000 measured samples to retain the native tail policy;
a smaller requested budget is never silently increased. A valid but insufficient
capture remains MISSING_PREREQUISITE, with raw samples retained.
See [selective analysis](docs/guides/selective-analysis.md) for exact statuses,
trust/resource limits and report integration. The acceptance runner remains a
separate advanced tool; full selection does not execute or certify all 35 features.

## Experiments, reports and help

Compare two **existing** reports without running missing tests. Reports must have
records associated with model IDs A/B (or use your own matching IDs):

```python
from hyso import analysis

def compare_saved_models(report_a_path, report_b_path):
    source_a = analysis.ModelComparisonSource(
        "A", analysis.UnifiedAnalysisReport.from_json(report_a_path))
    source_b = analysis.ModelComparisonSource(
        "B", analysis.UnifiedAnalysisReport.from_json(report_b_path))
    return analysis.compare_models(source_a, source_b,
        features=("P01", "P02", "P10", "P11", "P26"), include_structure=True)

# comparison = compare_saved_models("model-a-report.json", "model-b-report.json")
# comparison.to_json("comparison.json")
# comparison.to_markdown("comparison-tr.md", language="tr")
# comparison.to_markdown("comparison-en.md", language="en")
```

Differences are B minus A, with native metric directions and separate quality
percentage points. Missing/ambiguous records or incompatible conditions yield no
numerical difference. General performance comparison requires the existing
producer comparison metadata; ordinary Stage-5 captures may remain insufficient.
Caller labels never repair missing native proof. P26 bound structure comparisons
can expose parameter counts and registered logical sizes separately from P11.
P10 remains process-wide RSS; it is not model-exclusive RAM or P09 VRAM.
No overall winner, statistical superiority or acceptance PASS is inferred.
See [two-model comparison](docs/guides/two-model-comparison.md) for scope and examples.

`ExperimentManager(root_dir)` stores experiments, metrics and artifacts; it does
not schedule training. Use a dedicated non-synced local directory outside OneDrive
and call `close()` in a try/finally block. `EvidenceIndex` and `UnifiedAnalysisReport`
aggregate existing evidence. `advisor.advise(report)` interprets a compatible report
without running a model. Advisor display JSON/Markdown supports English and Turkish.

Use `help(Trainer)`, `help(Trainer.fit)`, `help(TrainingConfig)`, `help(ModularCNN)` and `help(BenchmarkRunner)`
for parameters and return values. `py.typed` enables inline typing discovery;
annotations remain incremental and heterogeneous metadata may use Any. Actual IDE
behavior depends on its selected interpreter. See the [Python API map](docs/guides/python-api.md).

## Implemented versus planned

Model Registry/Builder v1 now exposes a closed catalog of existing components and
a validated linear classifier chain (plus an unchanged ModularCNN profile):

```python
from hyso import ModelRegistry, ModelBuilder

registry = ModelRegistry()
print([spec.name for spec in registry.list("attention")])
print(registry.get("conv_norm_act").to_dict()["parameters"])
builder = (ModelBuilder(input_shape=(1, 8, 8), num_classes=2)
           .add("conv_norm_act", in_channels=1, out_channels=8, norm=None)
           .add("residual", in_channels=8, out_channels=8, norm=None,
                attention={"name": "eca", "params": {"channels": 8}}))
built_model = builder.build()  # Fresh CPU weights; no training or benchmark.
rebuilt = ModelBuilder.from_config(builder.to_config()).build()
# builder.to_json("architecture.json")  # Architecture only; no weights/code.
# rebuilt = ModelBuilder.from_json("architecture.json").build()
```

See the [Registry/Builder guide](docs/guides/model-registry-builder.md) for the
whitelist, parameter/shape contracts, limits and checkpoint compatibility.

Implemented: ELO/CBAM, CEMO/SE, ModularCNN, ECA/Coordinate Attention, Trainer,
checkpoint v1, BenchmarkRunner, measurement APIs, pairwise comparisons,
ExperimentManager, evidence reports and read-only Advisor interpretation.
ONNX provider observation exists; it is not a general export/deployment pipeline.

Not implemented: arbitrary graph/multi-family model plugins, general CLI,
graphical UI, TensorRT/OpenVINO/TFLite/CoreML/Edge deployment adapters.
See the [architecture roadmap](docs/roadmap/hyso-architecture-roadmap-v1.md) and
[performance backlog](docs/roadmap/hyso-performance-backlog-v1.1.md); dated rows are
historical and should be read alongside current code and reports.

## Accepted scope

[Final Model A/B acceptance](docs/reports/p01-p35-final-acceptance-20261112.md)
records 35/35 PASS each within declared scopes. This entry-point work does not
rerun or change it. Model A combines preserved evidence; P30 uses a separate
trained image adapter; P23 Basic excludes extended saved-activation lifetime and
production CUDA saved-activation hooks remain disabled. Acceptance is not a
universal model/hardware/backend/version guarantee.

License: [MIT](LICENSE).
