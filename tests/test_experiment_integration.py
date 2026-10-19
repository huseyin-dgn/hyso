"""Execute the manual guide against real public training/benchmark/storage APIs."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import uuid
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hyso import checkpoint
from hyso.benchmark import BenchmarkRunner, BenchmarkResult, compare_results
from hyso.experiments import ExperimentManager, ExperimentError
from hyso.models.modular import ModularCNN
from hyso.training import Trainer
from hyso.callbacks.callbacks import EarlyStopping

GUIDE = Path(__file__).resolve().parents[1] / "docs/guides/experiment-manager.md"
BLOCKS = re.findall(r"```python\n(.*?)```", GUIDE.read_text(encoding="utf-8"), re.S)
EXAMPLE = {}
exec(compile(BLOCKS[0], str(GUIDE), "exec"), EXAMPLE)
exec(compile(BLOCKS[2], str(GUIDE), "exec"), EXAMPLE)
log_history = EXAMPLE["log_history"]
train_and_record = EXAMPLE["train_and_record"]
record_benchmark = EXAMPLE["record_benchmark"]
record_comparison = EXAMPLE["record_comparison"]
retry_from_checkpoint = EXAMPLE["retry_from_checkpoint"]
CONFIG = dict(input_channels=1, num_classes=2, widths=[3], repeats=[1],
              strides=[1], attention="eca", norm="batch")


def build(dtype=torch.float32, *, ema=False, early=False):
    model = ModularCNN(**CONFIG).to(dtype=dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
    return Trainer(model, optimizer=optimizer, scheduler=scheduler,
        loss_fn=nn.CrossEntropyLoss(), model_config=CONFIG, callbacks=(),
        ema_decay=.8 if ema else None,
        early_stopping=EarlyStopping(patience=1, delta=100, restore_best_weights=True) if early else None)


def loaders(dtype=torch.float32):
    x = torch.linspace(-1, 1, 4 * 25, dtype=dtype).reshape(4, 1, 5, 5)
    y = torch.tensor([0, 1, 0, 1])
    return [DataLoader(TensorDataset(x.clone(), y.clone()), batch_size=2, num_workers=0) for _ in range(2)]


@pytest.fixture
def manager(tmp_path):
    root = tmp_path / "store"
    m = ExperimentManager(root)
    m.close()
    assert ExperimentManager.migrate_store(root)["status"] == "migrated"
    m = ExperimentManager(root)
    yield m
    m.close()


def new_run(m, kind="training"):
    return m.create_run(m.create_experiment("integration")["id"], kind, CONFIG)


def assert_mapping(run, history, start=1):
    expected = []
    for index in range(start - 1, len(history["train_loss"])):
        for key, (name, unit) in EXAMPLE["METRICS"].items():
            if history[key][index] is not None:
                expected.append((name, unit, index + 1, history[key][index]))
    actual = [(r["name"], r["unit"], r["axis_index"], r["value"]) for r in run.read_metrics()]
    assert actual == expected
    assert all(r["source"] == "trainer_history" and r["axis_kind"] == "epoch" for r in run.read_metrics())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
@pytest.mark.parametrize("validation", [False, True])
@pytest.mark.parametrize("ownership", ["reference", "copy"])
def test_trainer_history_checkpoint_reopen(manager, tmp_path, dtype, validation, ownership):
    t = build(dtype, ema=True)
    train, val = loaders(dtype)
    r = new_run(manager)
    output = tmp_path / "last.pth"
    history, a = train_and_record(r, t, train, val if validation else None,
        epochs=2, checkpoint_path=output, ownership=ownership)
    assert history == t.history and r.to_dict()["status"] == "completed"
    assert_mapping(r, history)
    assert all(0 <= v <= 100 for v in history["train_acc"])
    if not validation:
        assert history["val_loss"] == history["val_acc"] == [None, None]
        assert not any(row["name"].startswith("validation/") for row in r.read_metrics())
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert payload["format_version"] == checkpoint.FORMAT_VERSION == 1
    assert payload["history"] == history and payload["completed_epochs"] == 2
    assert a["producer_run_id"] == r.id and a["expected_size_bytes"] == output.stat().st_size
    assert a["expected_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manager.verify_artifact(a["id"], path=output)["status"] == "verified"
    snapshot, metrics = r.to_dict(), r.read_metrics()
    other = ExperimentManager(manager.root_dir)
    assert other.get_run(r.id).to_dict() == snapshot
    assert other.get_run(r.id).read_metrics() == metrics
    assert other.list_artifacts(r.id)[0]["id"] == a["id"]
    assert other.get_artifact(a["id"])["producer_run_id"] == r.id
    other.close()


def test_accuracy_is_percent_not_fraction(manager):
    model = nn.Linear(1, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(torch.tensor([1., 0.]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.)
    t = Trainer(model, optimizer=optimizer, loss_fn=nn.CrossEntropyLoss())
    train = DataLoader(TensorDataset(torch.ones(4, 1), torch.tensor([0, 0, 0, 1])), batch_size=4)
    history = t.fit(train, train, epochs=1)
    r = new_run(manager)
    r.start()
    log_history(r, history)
    assert history["train_acc"] == history["val_acc"] == [75.]
    assert r.read_metrics(name="train/accuracy")[0]["value"] == 75.
    r.finish()


def test_many_epochs_bounded_atomic_batches_and_duplicate_replay(manager, monkeypatch):
    model = nn.Linear(1, 2)
    t = Trainer(model, optimizer=torch.optim.SGD(model.parameters(), lr=.001), loss_fn=nn.CrossEntropyLoss())
    train = DataLoader(TensorDataset(torch.ones(2, 1), torch.tensor([0, 1])), batch_size=2)
    history = t.fit(train, train, epochs=65)
    r = new_run(manager)
    r.start()
    original = r.log_metrics
    batches = []
    def observe(records):
        batches.append(len(records))
        return original(records)
    monkeypatch.setattr(r, "log_metrics", observe)
    log_history(r, history)
    assert batches == [256, 4]
    snapshot = r.read_metrics()
    log_history(r, history)
    assert r.read_metrics() == snapshot
    assert_mapping(r, history)
    r.finish()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("tracking_fails", [False, True])
def test_workload_exception_identity_and_secondary_tracking_failure(manager, tmp_path, monkeypatch, error_type, tracking_fails):
    t = build()
    train, val = loaders()
    r = new_run(manager)
    original = DataLoader.__iter__
    calls = 0
    original_error = error_type("original workload failure")
    def fail_second_epoch(loader):
        nonlocal calls
        if loader is train:
            calls += 1
            if calls == 2: raise original_error
        return original(loader)
    if tracking_fails:
        def unavailable(*args, **kwargs): raise ExperimentError("storage_error", "injected")
        monkeypatch.setattr(r, "log_metrics", unavailable)
        monkeypatch.setattr(r, "fail", unavailable)
        monkeypatch.setattr(r, "interrupt", unavailable)
    with patch.object(DataLoader, "__iter__", fail_second_epoch):
        with pytest.raises(error_type) as caught:
            train_and_record(r, t, train, epochs=3, checkpoint_path=tmp_path / "partial.pth")
    assert caught.value is original_error
    assert len(t.history["train_loss"]) == 1
    assert r.to_dict()["status"] == ("running" if tracking_fails else "interrupted" if error_type is KeyboardInterrupt else "failed")
    if not tracking_fails:
        assert_mapping(r, t.history)
        if error_type is RuntimeError:
            assert r.to_dict()["reason"]["failure_domain"] == "workload"
            assert r.to_dict()["reason"]["workload_outcome"] == "failed"
    else:
        manager.recover_interrupted(r.id, r.to_dict()["revision"], "old workload stopped")
        assert r.to_dict()["status"] == "interrupted"


@pytest.mark.parametrize("fault", ["metric", "artifact", "transaction"])
def test_successful_workload_tracking_failures_keep_committed_metrics(manager, tmp_path, monkeypatch, fault):
    t = build()
    train, val = loaders()
    r = new_run(manager)
    original = r.log_metrics
    error = ExperimentError("storage_error", "injected")
    def logging_failure(records):
        original(records[:1])
        if fault == "transaction":
            manager._db.execute("CREATE TEMP TRIGGER reject_metric BEFORE INSERT ON metric_events BEGIN SELECT RAISE(ABORT,'injected'); END")
            return original(records[1:])
        raise error
    if fault in ("metric", "transaction"):
        monkeypatch.setattr(r, "log_metrics", logging_failure)
    else:
        def unavailable(*args, **kwargs): raise error
        monkeypatch.setattr(r, "add_artifact", unavailable)
    with pytest.raises(ExperimentError):
        train_and_record(r, t, train, val, epochs=2, checkpoint_path=tmp_path / "last.pth")
    assert len(t.history["train_loss"]) == 2
    assert r.to_dict()["status"] == "failed"
    reason = r.to_dict()["reason"]
    assert reason["workload_outcome"] == "succeeded"
    assert reason["failure_domain"] == ("artifact" if fault == "artifact" else "tracking")
    assert len(r.read_metrics()) == (8 if fault == "artifact" else 1)
    other = ExperimentManager(manager.root_dir)
    assert other.get_run(r.id).read_metrics() == r.read_metrics()
    other.close()


def test_closed_storage_never_fakes_completion(manager, tmp_path, monkeypatch):
    t = build()
    train, val = loaders()
    r = new_run(manager)
    original = r.log_metrics
    def storage_closed(records):
        manager.close()
        return original(records)
    monkeypatch.setattr(r, "log_metrics", storage_closed)
    with pytest.raises(ExperimentError, match="closed_session"):
        train_and_record(r, t, train, epochs=1, checkpoint_path=tmp_path / "last.pth")
    other = ExperimentManager(manager.root_dir)
    old = other.get_run(r.id)
    assert old.to_dict()["status"] == "running" and len(t.history["train_loss"]) == 1
    other.recover_interrupted(r.id, old.to_dict()["revision"], "workload already returned")
    other.close()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
def test_real_retry_resume_prefix_attribution(manager, tmp_path, dtype):
    t = build(dtype, ema=True)
    train, val = loaders(dtype)
    r = new_run(manager)
    original_iter = DataLoader.__iter__
    calls = 0
    def interrupt(loader):
        nonlocal calls
        if loader is train:
            calls += 1
            if calls == 2: raise KeyboardInterrupt()
        return original_iter(loader)
    source = tmp_path / "source.pth"
    with patch.object(DataLoader, "__iter__", interrupt), pytest.raises(KeyboardInterrupt):
        train_and_record(r, t, train, val, epochs=3, checkpoint_path=source)
    parent = r.to_dict()
    a = manager.list_artifacts(r.id)[0]
    owned = manager.root_dir / a["managed_relative_path"]
    retry, fresh, history, output = retry_from_checkpoint(manager, r.id, a["id"], owned,
        lambda: build(dtype, ema=True), train, val, epochs=3, output_path=tmp_path / "resumed.pth")
    assert retry.to_dict()["attempt_no"] == 2 and retry.to_dict()["resume_artifact_id"] == a["id"]
    assert_mapping(retry, history, 2)
    assert {row["axis_index"] for row in retry.read_metrics()} == {2, 3}
    assert history["train_loss"][:1] == t.history["train_loss"]
    assert r.to_dict() == parent and a["producer_run_id"] == r.id
    assert output["producer_run_id"] == retry.id and fresh.update_count == 6
    assert len(manager.list_artifacts(retry.id)) == 2


@pytest.mark.parametrize("early", [False, True], ids=["completed_budget", "early_stop_terminal"])
def test_terminal_resume_no_new_metrics_or_updates(manager, tmp_path, early):
    t = build(early=early)
    train, val = loaders()
    r = new_run(manager)
    source = tmp_path / "terminal.pth"
    r.start()
    t.fit(train, val, epochs=3, save_last_to=source)
    a = r.add_artifact(source, ownership="copy")
    r.fail("tracking_after_success", failure_domain="tracking", workload_outcome="succeeded")
    updates = t.update_count
    retry, fresh, history, output = retry_from_checkpoint(manager, r.id, a["id"], source,
        lambda: build(early=early), train, val, epochs=3, output_path=tmp_path / "no-new.pth")
    assert fresh.update_count == updates
    assert retry.read_metrics() == [] and output is None
    assert retry.to_dict()["reason"] == {"reason": "resume_no_new_epochs"}


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_benchmark_exports_identity_and_reopen(manager, tmp_path, precision):
    dtype = torch.float64 if precision == "fp64" else torch.float32
    model = nn.Sequential(nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2)).to(dtype)
    model.train()
    model[1].eval()
    for p in model.parameters(): p.grad = torch.ones_like(p)
    before = runtime_snapshot(model)
    r = new_run(manager, "benchmark")
    result, artifacts = record_benchmark(r, BenchmarkRunner(model, device="cpu", precision=precision,
        input_shape=(2, 1, 5, 5), warmup_iterations=1, measure_iterations=2, repeats=2), tmp_path / precision)
    assert_runtime(model, before)
    data = result.to_dict()
    assert data["schema_version"] == 1 and data["identity"]["run_id"] != r.id
    assert r.read_metrics() == []
    assert json.loads((tmp_path / precision / "result.json").read_text(encoding="utf-8")) == data
    with (tmp_path / precision / "summary.csv").open(newline="", encoding="utf-8") as f: assert len(list(csv.DictReader(f))) == 1
    with (tmp_path / precision / "samples.csv").open(newline="", encoding="utf-8") as f: assert len(list(csv.DictReader(f))) == 4
    other = ExperimentManager(manager.root_dir)
    assert other.get_run(r.id).to_dict()["status"] == "completed"
    for a in artifacts:
        assert other.verify_artifact(a["id"])["status"] == "verified"
        assert a["metadata"]["benchmark_run_id"] == data["identity"]["run_id"]
        assert a["metadata"]["experiment_run_id"] == r.id
    other.close()


def benchmark(model=None, **kwargs):
    return BenchmarkRunner(nn.Identity() if model is None else model, input_shape=(2, 1, 5, 5),
        warmup_iterations=0, measure_iterations=2, repeats=2, **kwargs).run()


@pytest.mark.parametrize("status", ["compatible", "incompatible", "insufficient_metadata"])
def test_comparison_plain_report_and_input_artifacts(manager, tmp_path, status):
    left, right = benchmark(), benchmark(nn.Conv2d(1, 2, 1))
    if status == "incompatible": right = benchmark(precision="fp64")
    if status == "insufficient_metadata":
        data = right.to_dict()
        data["environment"]["cpu"]["processor"] = "unknown"
        right = BenchmarkResult(data)
    r = new_run(manager, "evaluation")
    report, a = record_comparison(r, left, right, tmp_path / "comparison.json", [])
    assert report["status"] == report["timing_comparability"]["status"] == status
    assert report["memory_comparability"]["status"] == "not_applicable"
    assert not {"winner", "speedup", "significance", "pareto"} & report.keys()
    assert manager.verify_artifact(a["id"])["status"] == "verified"
    assert json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8")) == report


def runtime_snapshot(model):
    return dict(state={k:v.detach().clone() for k,v in model.state_dict().items()},
        refs=[id(p) for p in model.parameters()],
        grads=[None if p.grad is None else p.grad.clone() for p in model.parameters()],
        grad_refs=[None if p.grad is None else id(p.grad) for p in model.parameters()],
        modes=[m.training for m in model.modules()], python=random.getstate(), numpy=np.random.get_state(),
        torch=torch.get_rng_state().clone())


def assert_runtime(model, before):
    assert [id(p) for p in model.parameters()] == before["refs"]
    assert [m.training for m in model.modules()] == before["modes"]
    assert [None if p.grad is None else id(p.grad) for p in model.parameters()] == before["grad_refs"]
    for k,v in model.state_dict().items(): assert torch.equal(v, before["state"][k])
    for p,g in zip(model.parameters(),before["grads"]):
        assert (p.grad is None and g is None) or torch.equal(p.grad,g)
    assert random.getstate() == before["python"]
    now = np.random.get_state()
    assert now[0] == before["numpy"][0] and np.array_equal(now[1],before["numpy"][1]) and now[2:] == before["numpy"][2:]
    assert torch.equal(torch.get_rng_state(),before["torch"])


def test_tracking_never_calls_runtime_and_preserves_live_model_rng(manager, tmp_path):
    t = build(ema=True)
    train, val = loaders()
    history = t.fit(train, val, epochs=1, save_last_to=tmp_path / "last.pth")
    for p in t.model.parameters(): p.grad = torch.ones_like(p)
    t.model.train()
    t.model.stem.eval()
    result = benchmark(t.model)
    before = runtime_snapshot(t.model)
    ema_before = runtime_snapshot(t.evaluation_model)
    opt, scheduler = copy.deepcopy(t.optimizer.state_dict()), copy.deepcopy(t.scheduler.state_dict())
    def forbidden(*args, **kwargs): raise AssertionError("Manager touched runtime")
    from contextlib import ExitStack
    with ExitStack() as stack:
        for obj,name in [(t.model,"forward"),(t.evaluation_model,"forward"),(t.optimizer,"step"),
            (t.optimizer,"zero_grad"),(t.scheduler,"step"),(torch.cuda,"synchronize"),
            (torch.cuda,"Event"),(torch.cuda,"reset_peak_memory_stats"),(torch,"load")]:
            stack.enter_context(patch.object(obj,name,forbidden))
        r = new_run(manager)
        r.start()
        log_history(r,history)
        a = r.add_artifact(tmp_path / "last.pth",ownership="copy")
        assert manager.verify_artifact(a["id"])["status"] == "verified"
        result.to_json(tmp_path / "result.json")
        r.add_artifact(tmp_path / "result.json",kind="benchmark_json",ownership="reference",role="report")
        r.finish()
        other = ExperimentManager(manager.root_dir)
        assert other.get_artifact(a["id"])["expected_sha256"] == a["expected_sha256"]
        other.close()
    assert_runtime(t.model,before)
    assert_runtime(t.evaluation_model,ema_before)
    assert t.scheduler.state_dict() == scheduler
    for actual, saved in zip(t.optimizer.state_dict()["state"].values(), opt["state"].values()):
        for key in saved: assert torch.equal(actual[key],saved[key])


def test_complete_guide_executes(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    namespace = {}
    for block in BLOCKS:
        exec(compile(block,str(GUIDE),"exec"),namespace)
    manager = ExperimentManager(namespace["root"])
    linked = manager.list_artifacts(namespace["comparison"].id)
    assert len(linked) == 3
    report_artifact = manager.get_artifact(namespace["report_artifact"]["id"])
    assert report_artifact["metadata"]["input_artifact_ids"] == namespace["input_ids"]
    assert all(manager.verify_artifact(a["id"])["status"] == "verified" for a in linked)
    manager.close()
