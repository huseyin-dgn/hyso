# Experiment history integration

Use the existing `ExperimentManager` to keep architecture, training settings,
training results, selected analyses and model comparisons under one experiment ID.
All operations below consume configuration or already-retained results. Run
lifecycle uses the existing `start()`, `finish()`, `fail()` and `interrupt()` calls.

## Architecture and settings

Pass a dedicated local directory outside synced folders. This function creates
a schema1 store when absent and records architecture metadata using the Builder
configuration. It allocates no model weights.

```python
from hyso import ExperimentManager, ModelBuilder, TrainingConfig

def create_study(store_dir):
    manager = ExperimentManager(store_dir)
    try:
        experiment = manager.create_experiment("CNN study")
        builder = ModelBuilder((1, 4, 4), 2).add(
            "conv2d", in_channels=1, out_channels=2)
        run = manager.create_workflow_run(
            experiment["id"], "model", model_ids=["A"],
            architectures={"A": builder},
            dataset={"id": "my-dataset-v1", "split": "train"},
            device="cpu", precision="fp32")
        run.start()
        run.finish()
        return experiment["id"]
    finally:
        manager.close()
```

`create_workflow_run()` accepts stages `model`, `training`, `analysis`, `comparison`.
They use existing run kinds `evaluation`, `training`, `benchmark`, `benchmark`,
respectively. `model` and `training` have one model ID, `analysis` supports one or
two, and `comparison` uses its two distinct source IDs in baseline/candidate order.
`architectures` maps model IDs to existing ModelBuilders. Training requires a
TrainingConfig, analysis requires an existing AnalysisDryRun, and comparison
requires an existing ModelComparisonResult. A `request_id` provides the same
idempotent creation contract as `create_run()`.

Run config stores `hyso.experiment_workflow` version 1 as ordinary application
metadata in the existing `config_json` column. ModelBuilder.to_config(),
TrainingConfig.to_dict() and existing tracking_config() projections supply these
snapshots. This adds no SQLite columns, tables, run kinds or checkpoint format.

`dataset`, `device`, `precision` and `protocol` are explicit caller context.
Omitted values stay null. Model IDs and architecture-to-result links are declared
associations. Native device/protocol/model-state facts remain in the full result
artifact with their original evidence contracts. Known training setting and
analysis plan mismatches are rejected during result recording. Callers can retain
dataset versions/hashes/splits in their context dictionary. No source code,
factory, model weight or dataset is loaded from these snapshots.

## Explicit migration and recording

New stores keep the existing schema1 policy. Snapshots, history, filtering and
`TrainingResult.log_to(run)` work with schema1. `record_result`, result reopening
and artifact relations require schema2. Calling them on schema1 raises the
existing schema2-required error before writing metrics or links.

Close handles, coordinate all writers, and use the established explicit migration.
Persisted running runs require the existing revision-checked recovery before
migration. The migration retains its backup, integrity checks and old-session
fencing. Opening a manager or reading history performs no migration.

```python
def enable_result_artifacts(store_dir):
    return ExperimentManager.migrate_store(store_dir)
```

The following function records a `TrainingResult` already produced by the user.
The builder/config must describe that result. Supply existing input artifacts
from the same experiment when relationships are available.

```python
def record_saved_training(store_dir, experiment_id, builder, config,
                          training_result, input_artifact_ids=()):
    manager = ExperimentManager(store_dir)
    try:
        run = manager.create_workflow_run(
            experiment_id, "training", model_ids=["A"],
            architectures={"A": builder}, training_config=config,
            dataset={"id": "my-dataset-v1", "split": "train"},
            device=config.to_dict()["device"],
            precision=config.to_dict()["precision"])
        run.start()
        artifact = run.record_result(
            training_result, input_artifact_ids=input_artifact_ids)
        run.finish()
        return run.id, artifact["id"]
    finally:
        manager.close()
```

For analyses, pass `analysis=dry_run` at creation and `run.record_result(execution)`
after obtaining an AnalysisExecutionResult. Its plan ID must match. The raw
native results, auxiliary P01 BenchmarkResult and receipts stay in the artifact.
For comparisons, pass `comparison=comparison_result`, matching model IDs, then
record that same result with explicit input artifacts. The existing comparison
tracking projection supplies its result hash and selected features.

`record_result()` stores one full JSON result per run as a managed copied output
artifact. It validates typed/native data and uses existing input links. Input
artifacts must already be linked to this experiment and currently verify. These
explicit input associations are caller declarations. The adapter preserves the
result's own evidence and comparability rules. Arbitrary checkpoint or code
deserialization is outside the result reader whitelist.

While the run is running, recording the identical result and inputs reuses its
artifact and metric events. A changed result or changed inputs produces a
conflict. Training history uses TrainingResult.log_to(), with the existing
256-event transaction bound applied in batches. An identical retry can complete
a partially logged history. Recording never changes lifecycle state itself.

Artifact ingestion and metric logging retain separate existing transactions.
A failure may leave input links, a pending/failed artifact, or a ready artifact
with partial metrics on a running run. Inspect details before choosing the next
action. Existing `recover_interrupted(run_id, expected_revision, reason)` and
offline `reconcile_storage()` handle abandoned writers/storage. Incomplete
artifact ingestion raises `result_storage_incomplete` on repeat recording. A
new attempt can use the existing explicit `create_run(..., retry_of_run_id=...)`
with the prior workflow config. Retained evidence and failed attempts remain
available. There is no implicit workload retry or run completion.

## Find and reopen history

```python
def read_study(store_dir, experiment_id, artifact_id):
    manager = ExperimentManager(store_dir)
    try:
        experiments = manager.list_experiments(kind="training", status="completed")
        history = manager.get_experiment_history(experiment_id)
        details = [manager.get_run_details(r["id"]) for r in history["runs"]]
        opened = manager.open_result(artifact_id)
        return experiments, history, details, opened
    finally:
        manager.close()
```

`list_experiments()` supports exact name and matching run kind/status, plus
`after_seq`/`limit`. Kind/status apply to the same run. `list_runs()` keeps its
existing experiment/kind/status filters. History pages include full run configs
and their existing states. The experiment summary counts all runs, including
those outside the page, with precedence running, created, failed, interrupted,
completed. An empty experiment has derived state created. This summary is a
view, with original per-run states retained. Running writer liveness stays
unknown until explicit operator recovery.

`get_experiment_history()` returns `next_after_seq`. `get_run_details()` returns
metric and artifact pages, roles, and `next_metric_seq`/`next_artifact_seq`.
Follow full-page cursors to obtain the complete history. Observations of file
availability are current filesystem checks, separate from persisted run status.
History/detail/read operations append no artifact-check rows and start no
training, inference, benchmarking or measurement. Comparison JSON reopening
reruns the existing pure validation/comparison derivation from retained evidence.

`open_result()` returns a dictionary with `status`, `reason`, `artifact`, `result`.
A verified TrainingResult, AnalysisExecutionResult or ModelComparisonResult
appears in `result`. The reader verifies the actual loaded bytes against the
registered SHA256 and size, then uses the result/native readers. Valid JSON can
still produce `invalid_result`. TrainingResult remains a training UX/history
record with declared associations. Native measurement eligibility comes from
the original evidence readers and comparison policies.

Missing/moved files report `missing`, changed bytes `mismatch`, inaccessible files
`unreadable`, and references without locators `unavailable`. In these cases result
is null. A moved reference can be opened using explicit `path=...`; the original
locator and expected identity stay recorded. Managed copies use their registered
location. Other artifact kinds return `unsupported`; result reads have a 128 MiB
limit. Successful byte verification alone cannot authenticate an independently
fabricated capture.
