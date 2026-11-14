# Python API discovery and help

The current [Python-first workflow](python-first-workflow.md) covers graph output
selection, isolated model checks and architecture comparison. The web product
is retired; the Python API and seven existing CLI groups are the product interfaces.

| Component | Convenience import | Canonical module / principal APIs |
|---|---|---|
| Model | from hyso import ModularCNN | hyso.models.modular; named blocks in hyso.models |
| Registry/Builder | from hyso import ModelRegistry, ModelBuilder | hyso.models.registry / builder; list, get, add, build, from_config/from_json |
| Trainer | from hyso import Trainer | hyso.training.trainer; fit, evaluation_model |
| Training setup | from hyso import TrainingConfig | hyso.training.config / session; make_loaders, prepare, settings, fit |
| Benchmark | from hyso import BenchmarkRunner | hyso.benchmark.runner; run returns BenchmarkResult |
| Performance | from hyso import performance | Explicit measurement/protocol/result APIs |
| Analysis | from hyso import analysis | EvidenceIndex, UnifiedAnalysisReport, FeaturePlanRequest, plan_features |
| Selective execution | from hyso import analysis | feature_catalog, plan_analysis, preflight_analysis, execute_analysis |
| Two-model comparison | from hyso import analysis | ModelComparisonSource, compare_models, ModelComparisonResult |
| Advisor | from hyso import advisor | advise, to_display_dict/json, to_markdown; recommend, RecommendationReport.to_json/to_markdown/to_html |
| Experiments | from hyso import ExperimentManager | hyso.experiments.manager; run/metric/artifact persistence |
| Experiment history | from hyso.experiments import ExperimentRun | create_workflow_run, list_experiments, get_experiment_history, get_run_details, record_result, open_result |
| Result reporting | from hyso import reporting | create_report, report_experiment, feature_explanations; PresentationReport.to_json/to_markdown/to_html |

Root convenience aliases resolve on access to the original classes, not wrappers.
Imports do not construct models or request execution. Existing deep imports remain
valid. TYPE_CHECKING aliases expose concrete types to editors; dir(hyso) includes
lazy names. Use help(Trainer.fit), help(ModularCNN) and help(BenchmarkRunner).

Legacy from hyso.models import * keeps exactly EBO, ELO, CEMO, EMA and
CNNSENormalWithFit in the published order. Import ModularCNN, ConvNormAct,
ResidualBlock, SEBlock, CBAMBlock, ECAAttention and CoordinateAttention by name.

The wheel/sdist includes hyso/py.typed. Main entry points have inline annotations;
legacy/internal APIs are not all fully typed. Runtime validation remains authoritative
for supported shapes, devices, optimizers and tasks. No mypy/Pylance-wide clean bill
is implied by the marker. Pylance must use the environment where HYSo is installed.

Recommended workflow: construct model -> select device -> create optimizer ->
Trainer.fit (only when intended) -> evaluation_model -> selected measurement ->
read/export result -> EvidenceIndex/unified report -> Advisor -> optional Manager
persistence in a non-synced store. Selection planning is not a workload scheduler;
Manager persistence is not a model registry; display models are not a GUI.
TrainingConfig stores only versioned settings, with architecture/data supplied
separately. See [training configuration](training-config.md) for a single-Trainer
workflow, applied-value reporting, Manager metric logging and checkpoint limits.
For feature/profile selection and explicitly authorized native CPU producers,
see [selective analysis](selective-analysis.md). Legacy G1/G2/G3 APIs remain valid.
P01–P35 typed sources, shared native captures and retained analysis are covered
in the [native executor guide](native-analysis-executor.md).

## Tiny configurable CPU training

This standalone synthetic example demonstrates the user workflow, not accuracy.
Architecture and data are separate from the serialized training request.

```python
import torch
from torch.utils.data import TensorDataset
from hyso import ModelBuilder, TrainingConfig

torch.manual_seed(0)
builder = ModelBuilder((1, 4, 4), 2).add("conv2d", in_channels=1, out_channels=2)
config = TrainingConfig(epochs=1, batch_size=2, optimizer="sgd", learning_rate=0.01)
config = TrainingConfig.from_dict(config.to_dict())  # Same schema as JSON.
dataset = TensorDataset(torch.randn(4, 1, 4, 4), torch.tensor([0, 1, 0, 1]))
train_loader, val_loader = config.make_loaders(dataset)
session = config.prepare(builder.build(), train_loader, val_loader)
assert session.settings()["effective"]["device"] == "cpu"
training_result = session.fit()
assert len(training_result.history["train_loss"]) == 1
inference_model = session.evaluation_model
# config.to_json("training.json")  # Settings only; no datasets or code.
# config = TrainingConfig.from_json("training.json")
```

The [README](../../README.md) contains sequential CPU examples. Their synthetic
training and three timed calls demonstrate API usage, not quality/tail-latency or
historical 70/70 acceptance. See the [Registry/Builder guide](model-registry-builder.md)
for its closed whitelist and linear-graph scope. Deployment and task-specific support require separate
contracts; ONNX placement observation alone does not provide them.

## Compare two retained reports

This function only reads JSON and compares the requested evidence. Model IDs A/B
must match retained record associations. No model, checkpoint or factory is loaded.

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
# print(comparison.to_markdown(language="tr"))
# comparison.to_json("comparison.json")
```

For Stage-5 outputs, use `ModelComparisonSource.from_execution(result, model_id)`.
Native comparison metadata may still be insufficient; no capture is rerun or
backfilled. [Two-model comparison](two-model-comparison.md) explains native gates,
P10/P11/P09 separation, bilingual views and optional Manager artifact linking.

See [experiment history integration](experiment-workflow.md) to retain architectures,
settings and existing results in the same experiment. Workflow entries use the current
training/evaluation/benchmark run kinds and existing lifecycle states. schema1 supports
snapshots and metrics. Result artifacts use schema2 after explicit offline migration.

[Result reporting](result-reporting.md) unifies retained training, evidence, Advisor,
comparison and experiment history in bilingual beginner/technical/expert views.
The presentation contract preserves source units and eligibility statuses and
exports portable HTML with embedded charts alongside JSON and Markdown.

[Performance Advisor](performance-advisor.md) adds read-only `recommend(source,
request=None, *, context=None)`. Existing `advise` decisions remain unchanged.
`RecommendationReport` retains native sources, original plans, six recommendation
areas, missing evidence and manual verification steps. Use `to_display_dict` or
the Stage 8 exports with `language="tr"/"en"` and
`level="beginner"/"technical"/"expert"`. Reopening JSON rederives recommendations
without running a workload. It does not reread external artifact files.
