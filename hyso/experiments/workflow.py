"""Adapters for the existing Manager snapshots, artifact store and result readers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from .manager import ExperimentError, ExperimentRun, _error, _id, _label, _page
from ..analysis._codec import canonical, decode


_KINDS = {"model": "evaluation", "training": "training", "analysis": "benchmark", "comparison": "benchmark"}
_RESULT_KINDS = {"training": "training_result", "analysis": "analysis_execution", "comparison": "model_comparison"}
_REASONS = {
    "verified": "Registered size and SHA256 match the current bytes.",
    "missing": "Registered file is missing or moved. Supply path for a relocated reference.",
    "mismatch": "Current bytes differ from the registered size or SHA256.",
    "unreadable": "Registered file could not be read safely.",
    "unavailable": "No retained locator or complete artifact identity is available.",
}
_MAX_RESULT_BYTES = 128 * 1024**2


def workflow_config(stage, model_ids, architectures, training_config, analysis, comparison,
                    dataset, device, precision, protocol):
    from ..models.builder import ModelBuilder
    from ..training import TrainingConfig
    from ..analysis import AnalysisDryRun, ModelComparisonResult

    if type(stage) is not str or stage not in _KINDS:
        _error("invalid_workflow_stage", "create_workflow_run")
    if (type(model_ids) not in (tuple, list) or not 1 <= len(model_ids) <= 2
            or any(type(name) is not str for name in model_ids) or len(set(model_ids)) != len(model_ids)):
        _error("invalid_model_ids", "create_workflow_run")
    models = [_label(name, 256) for name in model_ids]
    if stage in ("model", "training") and len(models) != 1:
        _error("single_model_required", "create_workflow_run")
    architecture_data = {}
    if architectures is not None:
        if type(architectures) is not dict or set(architectures) - set(models):
            _error("architecture_model_mismatch", "create_workflow_run")
        for name, builder in architectures.items():
            if type(builder) is not ModelBuilder:
                _error("model_builder_required", "create_workflow_run")
            architecture_data[name] = builder.to_config()
    if stage == "model" and not architecture_data:
        _error("architecture_required", "create_workflow_run")
    if (training_config is not None and type(training_config) is not TrainingConfig
            or stage == "training" and training_config is None):
        _error("training_config_required", "create_workflow_run")
    if analysis is not None and (stage != "analysis" or type(analysis) is not AnalysisDryRun):
        _error("analysis_dry_run_required", "create_workflow_run")
    if comparison is not None and (stage != "comparison" or type(comparison) is not ModelComparisonResult):
        _error("model_comparison_result_required", "create_workflow_run")
    tracking = None
    if stage == "analysis":
        if analysis is None:
            _error("analysis_dry_run_required", "create_workflow_run")
        tracking = analysis.tracking_config()
        if tracking["models"] != models or device is not None and device != tracking["device"]:
            _error("analysis_context_mismatch", "create_workflow_run")
    elif stage == "comparison":
        if comparison is None:
            _error("model_comparison_result_required", "create_workflow_run")
        tracking = comparison.tracking_config()
        if list(tracking["models"].values()) != models:
            _error("comparison_context_mismatch", "create_workflow_run")
    for value in (device, precision):
        if value is not None:
            _label(value)
    if any(value is not None and type(value) is not dict for value in (dataset, protocol)):
        _error("context_requires_dict", "create_workflow_run")
    config = dict(format="hyso.experiment_workflow", schema_version=1, stage=stage,
                  context=dict(model_ids=models, dataset=dataset, device=device, precision=precision,
                               protocol=protocol, association="caller_declared"),
                  architectures=architecture_data,
                  training_config=None if training_config is None else training_config.to_dict(),
                  tracking=tracking)
    return _KINDS[stage], config


def _restore_result(kind, data):
    """Validate data and reuse typed/native readers without a workload call."""
    from ..training import TrainingResult, TrainingConfig
    from ..analysis import AnalysisExecutionResult, ModelComparisonResult

    if kind == "model_comparison":
        return ModelComparisonResult(data)
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported_result_version")
    if kind == "training_result":
        if data.get("format") != "hyso.training_session_result" or data["settings"]["state"] != "completed":
            raise ValueError("completed_training_result_required")
        TrainingConfig.from_dict(data["settings"]["requested"])
        history = data["history"]
        if set(history) != {"train_loss", "train_acc", "val_loss", "val_acc"}:
            raise ValueError("invalid_training_history")
        epochs = data["completed_history_epochs"]
        if type(epochs) is not int or epochs < 1 or any(
                type(values) is not list or len(values) != epochs or any(
                    value is not None and type(value) not in (int, float) for value in values)
                for values in history.values()):
            raise ValueError("invalid_training_history")
        if (type(data["applied_updates"]) is not int or data["applied_updates"] < 0
                or type(data["epoch_events"]) is not list):
            raise ValueError("invalid_training_observations")
        for row in data["epoch_events"]:
            if (type(row["epoch"]) is not int or not 1 <= row["epoch"] <= epochs
                    or type(row["interval_seconds"]) not in (int, float) or row["interval_seconds"] < 0):
                raise ValueError("invalid_epoch_observation")
        return TrainingResult(canonical(data))
    if kind == "analysis_execution":
        from ..analysis.execution import STATUSES
        from ..analysis._codec import digest
        from ..benchmark import BenchmarkResult
        if data.get("format") != "hyso.analysis_execution":
            raise ValueError("analysis_execution_result_required")
        plan = data["dry_run"]["plan"]
        if plan["plan_id"] != digest({k: v for k, v in plan.items() if k != "plan_id"}):
            raise ValueError("analysis_plan_identity_mismatch")
        models = {row["model_id"] for row in plan["request"]["models"]}
        seen = set()
        for row in data["features"]:
            identity = (row["model_id"], row["feature_id"])
            if (identity in seen or identity[0] not in models or row["status"] not in STATUSES
                    or identity[1] not in {r["feature_id"] for r in plan["features"]}):
                raise ValueError("invalid_execution_row")
            seen.add(identity)
            record = row["evidence_record"]
            if record is not None and (record["scope"]["model_id"] != identity[0] or record["feature_id"] != identity[1]):
                raise ValueError("execution_record_association_mismatch")
            if row["status"] == "COMPLETED" and (record is None or identity[1] not in data["authorized_features"]):
                raise ValueError("completed_execution_evidence_required")
            if "benchmark" in row["auxiliary_evidence"]:
                BenchmarkResult(row["auxiliary_evidence"]["benchmark"])
            if plan.get('execution_mode') == 'native':
                from ..analysis.native_execution import validate_operation_row
                validate_operation_row(row, data)
        expected_counts = {s: sum(r["status"] == s for r in data["features"]) for s in STATUSES}
        if plan.get('execution_mode') == 'native' and seen != {
                (model, feature['feature_id']) for model in models for feature in plan['features']}:
            raise ValueError('native_execution_rows_incomplete')
        if data["status_counts"] != expected_counts:
            raise ValueError("execution_status_counts_mismatch")
        result = AnalysisExecutionResult(canonical(data))
        result.to_evidence_index()  # Original EvidenceRecord and native-reader checks.
        return result
    raise ValueError("unsupported_result_kind")


def _result_data(result):
    from ..training import TrainingResult
    from ..analysis import AnalysisExecutionResult, ModelComparisonResult
    types = {TrainingResult: "training_result", AnalysisExecutionResult: "analysis_execution",
             ModelComparisonResult: "model_comparison"}
    kind = types.get(type(result))
    if kind is None:
        _error("typed_workflow_result_required", "record_result")
    data = decode(canonical(result.to_dict()).encode("utf-8"))
    try:
        _restore_result(kind, data)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        raise ExperimentError("invalid_workflow_result", "record_result") from error
    return kind, data


def _match_result(config, kind, data):
    stage = config.get("stage")
    if (config.get("format") != "hyso.experiment_workflow" or config.get("schema_version") != 1
            or _RESULT_KINDS.get(stage) != kind):
        _error("workflow_result_stage_mismatch", "record_result")
    context = config["context"]
    if stage == "training":
        if config["training_config"] != data["settings"]["requested"]:
            _error("training_settings_mismatch", "record_result")
        effective = data["settings"]["effective"]
        if any(context[key] is not None and context[key] != effective[key] for key in ("device", "precision")):
            _error("training_context_mismatch", "record_result")
    elif stage == "analysis":
        plan = data["dry_run"]["plan"]
        if (config["tracking"]["analysis_plan_id"] != plan["plan_id"]
                or context["model_ids"] != [m["model_id"] for m in plan["request"]["models"]]):
            _error("analysis_plan_mismatch", "record_result")
    elif config["tracking"]["comparison_sha256"] != data["content_sha256"]:
        _error("comparison_result_mismatch", "record_result")


def record_result(run, result, input_artifact_ids):
    manager = run._manager
    manager._b2()  # Before any partial metric/link write on schema1.
    if (type(input_artifact_ids) not in (list, tuple)
            or any(type(identity) is not str for identity in input_artifact_ids)
            or len(set(input_artifact_ids)) != len(input_artifact_ids)):
        _error("invalid_input_artifact_ids", "record_result")
    inputs = sorted(_id(identity) for identity in input_artifact_ids)
    kind, data = _result_data(result)
    raw = canonical(data).encode("utf-8")
    if len(raw) > _MAX_RESULT_BYTES:
        _error("result_size_limit", "record_result")
    checksum = hashlib.sha256(raw).hexdigest()
    request = hashlib.sha256(("workflow-result:" + run.id).encode()).hexdigest()[:32]
    metadata = dict(format="hyso.workflow_result", schema_version=1, result_sha256=checksum,
                    input_artifact_ids=inputs, association="caller_declared")
    with manager._access("record_result_preflight") as db:
        state = manager._row(db, "runs", run.id)
        run._owner(state, "record_result")
        if state["status"] != "running":
            _error("run_not_running", "record_result")
        _match_result(json.loads(state["config_json"]), kind, data)
        previous = db.execute("SELECT * FROM artifacts WHERE request_id=?", (request,)).fetchone()
        artifacts = [manager._row(db, "artifacts", identity) for identity in inputs]
        for artifact in artifacts:
            linked = db.execute("SELECT 1 FROM run_artifacts a JOIN runs r ON r.id=a.run_id "
                                "WHERE a.artifact_id=? AND r.experiment_id=? LIMIT 1",
                                (artifact["id"], state["experiment_id"])).fetchone()
            if linked is None:
                _error("input_artifact_experiment_mismatch", "record_result")
    for artifact in artifacts:
        if artifact["ingest_state"] != "ready" or manager._observe(artifact)["status"] != "verified":
            _error("input_artifact_unverified", "record_result", entity_id=artifact["id"])
    if previous is not None:
        previous = dict(previous)
        if previous["kind"] != kind or json.loads(previous["metadata_json"]) != metadata:
            _error("workflow_result_conflict", "record_result")
        if previous["ingest_state"] != "ready":
            _error("result_storage_incomplete", "record_result", entity_id=previous["id"])
        if manager._observe(previous)["status"] != "verified":
            _error("result_artifact_unverified", "record_result", entity_id=previous["id"])
    # Input links commit together. Artifact ingest retains the existing multi-step
    # recovery protocol. A later failure leaves explicit state in the running run.
    with manager._access("record_result_inputs", write=True) as db:
        for identity in inputs:
            manager._link(db, run.id, identity, "input")
    if previous is None:
        with tempfile.TemporaryDirectory(prefix="hyso-result-") as directory:
            path = Path(directory) / "result.json"
            path.write_bytes(raw)
            artifact = run.add_artifact(path, kind=kind, ownership="copy", role="output",
                                        metadata=metadata, request_id=request, max_size_bytes=_MAX_RESULT_BYTES)
    else:
        artifact = manager._artifact_record(previous)
    if kind == "training_result":
        result.log_to(run)
    return artifact


def run_details(manager, run_id, after_metric_seq, after_artifact_seq, limit):
    _page(after_metric_seq, limit, 100)
    _page(after_artifact_seq, limit, 100)
    run = manager.get_run(run_id)
    state = run.to_dict()
    metrics = run.read_metrics(after_seq=after_metric_seq, limit=limit)
    artifacts = []
    if manager._schema_version == 2:
        records = manager.list_artifacts(run.id, after_seq=after_artifact_seq, limit=limit)
        with manager._access("workflow_artifact_roles") as db:
            for record in records:
                roles = [row[0] for row in db.execute(
                    "SELECT role FROM run_artifacts WHERE run_id=? AND artifact_id=? ORDER BY seq",
                    (run.id, record["id"]))]
                observation = manager._observe(record)
                artifacts.append(dict(artifact=record, roles=roles, availability=observation["status"],
                                      reason=_REASONS[observation["status"]]))
    return dict(run=state, metrics=metrics, artifacts=artifacts,
                artifact_support="available" if manager._schema_version == 2 else "schema2_required",
                next_metric_seq=metrics[-1]["seq"] if len(metrics) == limit else None,
                next_artifact_seq=artifacts[-1]["artifact"]["seq"] if len(artifacts) == limit else None)


def open_result(manager, artifact_id, path):
    record = manager.get_artifact(artifact_id)
    outcome = dict(artifact=record, status="unavailable", reason=_REASONS["unavailable"], result=None)
    if record["ingest_state"] != "ready":
        outcome.update(reason="Artifact ingestion is incomplete. Use explicit storage recovery.")
        return outcome
    observation = manager._observe(record, path)
    outcome.update(status=observation["status"], reason=_REASONS[observation["status"]])
    if observation["status"] != "verified":
        return outcome
    if record["kind"] not in _RESULT_KINDS.values():
        outcome.update(status="unsupported", reason="No workflow JSON reader is registered for this artifact kind.")
        return outcome
    if record["expected_size_bytes"] > _MAX_RESULT_BYTES:
        outcome.update(status="size_limit", reason="Result exceeds the 128 MiB read limit.")
        return outcome
    try:
        locator = manager._managed(record["id"]) if record["ownership"] == "copy" else (
            path if path is not None else record["original_locator"])
        location = manager._file_path(locator)
        with location.open("rb") as source:
            raw = source.read(_MAX_RESULT_BYTES + 1)
        if (len(raw) != record["expected_size_bytes"]
                or hashlib.sha256(raw).hexdigest() != record["expected_sha256"]):
            outcome.update(status="mismatch", reason=_REASONS["mismatch"])
            return outcome
        outcome["result"] = _restore_result(record["kind"], decode(raw))
    except (OSError, ExperimentError):
        outcome.update(status="unreadable", reason=_REASONS["unreadable"])
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, RecursionError):
        outcome.update(status="invalid_result", reason="JSON or result/native validation failed.")
    return outcome
