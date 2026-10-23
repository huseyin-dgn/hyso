"""Immutable, explicitly keyed numeric observations and all-or-none batches."""
import uuid

import numpy as np
import pytest
import torch

from hyso.experiments import ExperimentError, ExperimentManager


@pytest.fixture
def run(tmp_path):
    manager = ExperimentManager(tmp_path / "runs")
    exp = manager.create_experiment("metrics")
    run = manager.create_run(exp["id"], "evaluation", {})
    run.start()
    yield run
    manager.close()


def metric(**kwargs):
    return dict(name="train/loss", value=.4, unit="mean_loss", source="caller", epoch=1, **kwargs)


def test_epoch_duplicate_and_terminal_read_retry(run):
    first = run.log_metric(**metric())
    run.log_metric(**{**metric(), "epoch": 3})
    assert run.log_metric(**metric()) == first
    assert run.to_dict()["metric_count"] == 2
    before = run.to_dict()["revision"]
    first["value"] = 100
    assert run.read_metrics()[0]["value"] == .4
    run.finish()
    assert run.log_metric(**metric())["value"] == .4
    assert run.to_dict()["revision"] == before + 1
    with pytest.raises(ExperimentError, match="run_not_running"):
        run.log_metric(**{**metric(), "epoch": 4})


@pytest.mark.parametrize("step_kind", ["applied_update", "optimizer_attempt", "microbatch", "observation"])
def test_step_axes_context(run, step_kind):
    record = run.log_metric(name="learning_rate/group_0", value=.01, unit="lr", source="caller",
                            step=0, step_kind=step_kind, epoch=1)
    assert record["axis_index"] == 0 and record["axis_kind"] == step_kind and record["epoch_context"] == 1


@pytest.mark.parametrize("change", [{"value": .5}, {"unit": "percent"}, {"source": "other"},
                                   {"step": 1, "step_kind": "observation"}])
def test_coordinate_conflicts(run, change):
    saved = run.log_metric(**metric())
    with pytest.raises(ExperimentError, match="metric_conflict"):
        run.log_metric(**{**metric(), **change})
    assert run.read_metrics() == [saved]


@pytest.mark.parametrize("change", [{"unit": "fraction"}, {"source": "trainer_history"},
                                   {"step": 2, "step_kind": "optimizer_attempt"}])
def test_series_invariants(run, change):
    saved = run.log_metric(**metric())
    with pytest.raises(ExperimentError, match="series_conflict"):
        run.log_metric(**{**metric(), "epoch": 2, **change})
    assert run.read_metrics() == [saved]


def test_out_of_order_and_exact_old_retry(run):
    first = run.log_metric(**{**metric(), "epoch": 2})
    run.log_metric(**{**metric(), "epoch": 4})
    assert run.log_metric(**{**metric(), "epoch": 2}) == first
    with pytest.raises(ExperimentError, match="out_of_order"):
        run.log_metric(**{**metric(), "epoch": 3})
    assert run.to_dict()["metric_count"] == 2


@pytest.mark.parametrize("value", [True, False, None, "0.4", float("nan"), float("inf"), -float("inf"),
                                   2**53 + 1, -(2**53 + 1), 10**500, np.float64(.4), np.int64(2), torch.tensor(.4)])
def test_rejected_values(run, value):
    with pytest.raises(ExperimentError, match="invalid_metric_value"):
        run.log_metric(**{**metric(), "value": value})
    assert run.read_metrics() == [] and run.to_dict()["metric_count"] == 0


@pytest.mark.parametrize("value", [0, 2**53, -(2**53), .4])
def test_exact_supported_numeric_values(run, value):
    assert run.log_metric(**{**metric(), "value": value})["value"] == value


@pytest.mark.parametrize("change", [{"epoch": 0}, {"epoch": True}, {"epoch": 2**63}, {"epoch": None},
                                   {"step": -1, "step_kind": "observation"}, {"step": 0},
                                   {"step": 0, "step_kind": "wall_time"}, {"step_kind": "microbatch"},
                                   {"name": "a//b"}, {"name": "../b"}, {"name": "a/.."},
                                   {"name": "a/1b"}, {"name": "ş"}, {"name": "a" * 129},
                                   {"unit": ""}, {"unit": "ms\n"}, {"source": "x" * 129}])
def test_invalid_axes_and_labels(run, change):
    with pytest.raises(ExperimentError):
        run.log_metric(**{**metric(), **change})
    assert run.read_metrics() == []


def test_event_id_collision_and_context(run):
    event = uuid.uuid4().hex
    first = run.log_metric(**metric(event_id=event))
    assert run.log_metric(**metric(event_id=event)) == first
    for change in ({"event_id": uuid.uuid4().hex}, {"epoch": 2}, {"name": "other"}):
        with pytest.raises(ExperimentError, match="conflict"):
            run.log_metric(**{**metric(event_id=event), **change})
    other = run._manager.create_run(run.to_dict()["experiment_id"], "training", {})
    other.start()
    with pytest.raises(ExperimentError, match="event_conflict"):
        other.log_metric(**metric(event_id=event))
    assert other.read_metrics() == []


def test_atomic_batch_success_pagination_and_duplicate(run):
    records = [{**metric(), "epoch": i} for i in range(1, 257)]
    result = run.log_metrics(records)
    assert run.log_metrics(records) == result
    assert run.to_dict()["metric_count"] == 256
    pages, cursor = [], 0
    while True:
        page = run.read_metrics(after_seq=cursor, limit=31)
        if not page:
            break
        pages += page
        cursor = page[-1]["seq"]
    assert pages == result and len({r["event_id"] for r in pages}) == 256
    assert run.read_metrics(name="other") == []
    for kwargs in ({"limit": 1001}, {"after_seq": True}, {"after_seq": -1}):
        with pytest.raises(ExperimentError):
            run.read_metrics(**kwargs)


@pytest.mark.parametrize("failure", ["validation", "conflict", "order", "unit", "same_batch"])
def test_atomic_batch_failure_and_reopen(run, failure):
    original = run.log_metric(**metric())
    records = [{**metric(), "epoch": 2}, {**metric(), "epoch": 3}]
    if failure == "validation":
        records[-1]["value"] = float("nan")
    elif failure == "conflict":
        records[-1] = {**metric(), "value": 999}
    elif failure == "order":
        records = [{**metric(), "epoch": 3}, {**metric(), "epoch": 2}]
    elif failure == "unit":
        records[-1]["unit"] = "different"
    else:
        records[-1] = {**records[0], "value": 999}
    before = run.to_dict()
    with pytest.raises(ExperimentError):
        run.log_metrics(records)
    assert run.to_dict() == before and run.read_metrics() == [original]
    reopened = ExperimentManager(run._manager.root_dir)
    assert reopened.get_run(run.id).read_metrics() == [original]
    reopened.close()


def test_count_policy_persisted_and_batch_limit(tmp_path):
    manager = ExperimentManager(tmp_path / "limited", max_metrics_per_run=2)
    exp = manager.create_experiment("limits")
    run = manager.create_run(exp["id"], "benchmark", {})
    run.start()
    with pytest.raises(ExperimentError, match="metric_limit"):
        run.log_metrics([{**metric(), "epoch": i} for i in (1, 2, 3)])
    assert run.read_metrics() == []
    first = run.log_metric(**metric())
    run.log_metric(**{**metric(), "epoch": 2})
    assert run.log_metric(**metric()) == first
    for records in ([], [metric()] * 257, iter([metric()])):
        with pytest.raises(ExperimentError, match="invalid_batch"):
            run.log_metrics(records)
    manager.close()
    with pytest.raises(ExperimentError, match="policy_conflict"):
        ExperimentManager(tmp_path / "limited")
    reopened = ExperimentManager(tmp_path / "limited", max_metrics_per_run=2)
    assert reopened.get_run(run.id).to_dict()["metric_count"] == 2
    reopened.close()


def test_observed_time_is_payload_not_axis(run):
    first = run.log_metric(**metric(observed_at_utc="2026-10-19T00:00:00Z"))
    assert run.log_metric(**metric(observed_at_utc="2026-10-19T00:00:00+00:00")) == first
    with pytest.raises(ExperimentError, match="metric_conflict"):
        run.log_metric(**metric(observed_at_utc="2026-10-20T00:00:00Z"))
    for observed in ("now", "2026-10-19", "2026-10-19T00:00:00+03:00"):
        with pytest.raises(ExperimentError, match="invalid_observed_time"):
            run.log_metric(**metric(observed_at_utc=observed))


def test_percent_fraction_are_explicit_series(run):
    run.log_metric(name="validation/accuracy", value=91., unit="percent", source="caller", epoch=1)
    with pytest.raises(ExperimentError, match="series_conflict"):
        run.log_metric(name="validation/accuracy", value=.91, unit="fraction", source="caller", epoch=2)
    assert run.read_metrics()[0]["value"] == 91.


def test_metric_query_default_1000_and_exact_batch_dedup(run):
    first = run.log_metrics([metric(), metric()])
    assert first[0] == first[1] and run.to_dict()["metric_count"] == 1
    for start in range(2, 1002, 250):
        run.log_metrics([{**metric(), "epoch": i} for i in range(start, min(start + 250, 1002))])
    page = run.read_metrics()
    assert len(page) == 1000 and run.to_dict()["metric_count"] == 1001
    assert len(run.read_metrics(after_seq=page[-1]["seq"])) == 1


def test_created_and_failed_runs_cannot_append(run):
    other = run._manager.create_run(run.to_dict()["experiment_id"], "evaluation", {})
    for phase in ("created", "failed"):
        with pytest.raises(ExperimentError, match="run_not_running"):
            other.log_metric(**metric())
        assert other.to_dict()["status"] == phase and other.read_metrics() == []
        if phase == "created":
            other.fail("preflight")
