"""P19 real training, exact paired state, bounded evidence and storage contracts."""
import copy
import csv
from dataclasses import FrozenInstanceError
import hashlib
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from hyso.training import Trainer
from hyso.performance import (TrainingTimingProtocol, TrainingPerformanceResult,
                              TrainingTimingError, measure_training)
from hyso.performance.training import _Recorder
from hyso.callbacks.callbacks import EarlyStopping
from trainer_replay_support import equal, seed, build, CRITICAL, CASES, snapshot


def make(*, epochs=3, scheduler="step", ema=True, early=False):
    model = nn.Sequential(nn.Linear(3, 4), nn.Dropout(.2), nn.Linear(4, 2))
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    sched = (torch.optim.lr_scheduler.StepLR(opt, 1, gamma=.9) if scheduler == "step" else
             torch.optim.lr_scheduler.OneCycleLR(opt, .02, total_steps=epochs * 2))
    return Trainer(model, optimizer=opt, loss_fn=nn.CrossEntropyLoss(), accum_steps=2,
                   scheduler=sched, ema_decay=.8 if ema else None,
                   early_stopping=EarlyStopping(patience=1, delta=1e6, verbose=False) if early else None)


def loader():
    return list(zip(torch.linspace(-1, 1, 15).reshape(5, 3).split(2),
                    torch.tensor([0, 1, 0, 1, 0]).split(2)))


def state(t):
    return dict(model=copy.deepcopy(t.model.state_dict()), optimizer=copy.deepcopy(t.optimizer.state_dict()),
                scheduler=copy.deepcopy(t.scheduler.state_dict()), scaler=t.scaler.state_dict(),
                ema=None if t.ema is None else copy.deepcopy(t.ema.ema_model.state_dict()),
                gradients=[None if p.grad is None else p.grad.clone() for p in t.model.parameters()],
                history=t.history, counters=[t.microbatch_count, t.attempt_count, t.update_count, t.ema_update_count],
                rng=(random.getstate(), np.random.get_state(), torch.get_rng_state()),
                modes=[m.training for m in t.model.modules()])


@pytest.mark.parametrize("epochs", [1, 2, 3])
@pytest.mark.parametrize("validation", [False, True])
@pytest.mark.parametrize("scheduler", ["step", "onecycle"])
def test_real_training_exact_state_and_ledger(epochs, validation, scheduler):
    outputs = []
    for observed in (False, True):
        seed(81)
        t = make(epochs=epochs, scheduler=scheduler)
        train, val = loader(), loader() if validation else None
        if observed:
            r = measure_training(t, train, val, epochs=epochs,
                                 protocol=TrainingTimingProtocol(warmup_epochs=1)).to_dict()
        else:
            t.fit(train, val, epochs=epochs)
        outputs.append(state(t))
        assert t.callbacks == () and not hasattr(t, "_timing_recorder")
        assert not t.optimizer._optimizer_step_post_hooks
    equal(outputs[0], outputs[1])
    assert r["actual_epochs"] == epochs
    assert len(r["observations"]) == 7 * epochs  # 3 micro + 2 window + 2 attempt
    for e in r["epochs"]:
        assert [e[k] for k in ("samples", "microbatches", "windows", "attempts", "applied", "skipped")] == [5, 3, 2, 2, 2, 0]
        assert e["total_s"] >= e["training_s"] > 0
        assert (e["validation_s"] is not None) == validation
        if validation:
            assert e["validation_s"] > 0 and e["total_s"] >= e["training_s"] + e["validation_s"]
    assert [x["samples"] for x in r["observations"] if x["kind"] == "window"] == [4, 1] * epochs
    assert all(x["cuda_event_ms"] is None and x["host_ms"] > 0 for x in r["observations"])


def test_fake_clock_exact_arithmetic(monkeypatch):
    import hyso.performance.training as module
    ticks = iter(range(100, 100000, 10))
    monkeypatch.setattr(module.time, "perf_counter_ns", lambda: next(ticks))
    data = measure_training(make(), loader(), epochs=1).to_dict()
    for row in data["observations"]:
        assert row["host_ms"] == (row["end_ns"] - row["start_ns"]) / 1e6
        assert row["start_ns"] % 10 == row["end_ns"] % 10 == 0
        if row["kind"] in ("microbatch", "attempt"):
            assert row["host_ms"] == 10 / 1e6
    e = data["epochs"][0]
    assert e["training_s"] == (e["train_end_ns"] - e["start_ns"]) / 1e9


def test_early_stop_exact():
    results = []
    for observed in (False, True):
        seed(111)
        t = make(early=True)
        if observed:
            data = measure_training(t, loader(), loader(), epochs=3).to_dict()
        else:
            t.fit(loader(), loader(), epochs=3)
        results.append(state(t))
        assert t._finalized and t.early_stopping.early_stop
    equal(*results)
    assert data["requested_epochs"] == 3 and data["actual_epochs"] == 2


def test_exception_cleanup_and_no_retry():
    t = make()
    bad = loader()
    bad[1] = (bad[1][0], torch.tensor([100, 100]))
    with pytest.raises(ValueError, match="class indices"):
        measure_training(t, bad, epochs=1)
    assert not hasattr(t, "_timing_recorder") and t.microbatch_count == 1
    assert not t.optimizer._optimizer_step_post_hooks
    with pytest.raises(ValueError, match="fresh"):
        measure_training(t, loader(), epochs=1)


def test_record_cap_fails_explicitly_without_success():
    t = make()
    with pytest.raises(TrainingTimingError, match="max_records"):
        measure_training(t, loader(), epochs=3, protocol=TrainingTimingProtocol(max_records=4))
    assert not hasattr(t, "_timing_recorder")
    assert t.update_count <= 1


def test_disabled_has_no_timer_profiler_or_recorder(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled training touched instrumentation")
    monkeypatch.setattr(_Recorder, "__init__", forbidden)
    monkeypatch.setattr(torch.profiler, "profile", forbidden)
    import hyso.performance.training as module
    monkeypatch.setattr(module.time, "perf_counter_ns", forbidden)
    t = make()
    t.fit(loader(), epochs=1)
    assert not hasattr(t, "_timing_recorder")


@pytest.mark.parametrize("kwargs", [dict(max_records=True), dict(event_buffer=0),
                                    dict(warmup_epochs=-1), dict(event_buffer=4097)])
def test_protocol_validation(kwargs):
    with pytest.raises(ValueError):
        TrainingTimingProtocol(**kwargs)


def test_immutable_export_and_tamper(tmp_path):
    r = measure_training(make(), loader(), epochs=1)
    with pytest.raises(FrozenInstanceError):
        r._payload = "bad"
    d = r.to_dict()
    d["epochs"][0]["samples"] = 0
    assert r.to_dict()["epochs"][0]["samples"] == 5
    with pytest.raises(ValueError):
        TrainingPerformanceResult(d)
    r.to_json(tmp_path / "timing.json")
    r.to_csv(tmp_path / "timing.csv")
    assert json.loads((tmp_path / "timing.json").read_text()) == r.to_dict()
    with (tmp_path / "timing.csv").open(newline="") as f:
        assert len(list(csv.DictReader(f))) == 7
    with pytest.raises(FileExistsError):
        r.to_json(tmp_path / "timing.json")
    bad = r.to_dict()
    bad["observations"][0]["host_ms"] += 1
    object.__setattr__(r, "_payload", json.dumps(bad))
    with pytest.raises(ValueError):
        r.to_csv(tmp_path / "bad.csv")
    assert not (tmp_path / "bad.csv").exists()


@pytest.mark.parametrize("mutation", ["sample", "clock", "skip", "counter", "id", "validation"])
def test_reject_inconsistent_evidence(mutation):
    d = measure_training(make(), loader(), loader(), epochs=1).to_dict()
    if mutation == "sample":
        d["observations"][0]["samples"] += 1
    elif mutation == "clock":
        d["observations"][0]["cuda_event_ms"] = 1
    elif mutation == "skip":
        next(r for r in d["observations"] if r["kind"] == "attempt")["skipped"] = True
    elif mutation == "counter":
        d["counters"]["after"]["applied"] += 1
    elif mutation == "id":
        next(r for r in reversed(d["observations"]) if r["kind"] == "microbatch")["microbatch_id"] = 1
    else:
        d["epochs"][0]["validation_start_ns"] = 0
    with pytest.raises(ValueError):
        TrainingPerformanceResult(d)


def test_experiment_manager_after_training(tmp_path):
    from hyso.experiments import ExperimentManager
    root = tmp_path / "store"
    m = ExperimentManager(root)
    m.close()
    ExperimentManager.migrate_store(root)
    m = ExperimentManager(root)
    run = m.create_run(m.create_experiment("p19")["id"], "training", {})
    run.start()
    r = measure_training(make(), loader(), epochs=1)
    path = tmp_path / "timing.json"
    r.to_json(path)
    r.to_csv(tmp_path / "timing.csv")
    artifact = run.add_artifact(path, kind="report", ownership="copy")
    assert m.verify_artifact(artifact["id"])["status"] == "verified"
    assert artifact["expected_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    run.log_metric(name="training.epoch.wall", value=r.to_dict()["epochs"][0]["training_s"],
                   unit="s", source="P19", epoch=1)
    run.finish()
    identity = run.id
    m.close()
    m = ExperimentManager(root)
    assert len(m.get_run(identity).read_metrics()) == 1
    m.close()


def test_checkpoint_resume_new_segment_exact(tmp_path):
    # Existing native replay fixture: checkpoint after epoch 2, total budget 5.
    from unittest.mock import patch
    import hyso.training.resume as resume
    case = dict(CRITICAL)
    outputs = []
    for observed in (False, True):
        seed(234)
        t, train, val = build(case)
        path = tmp_path / (str(observed) + ".pth")
        native = resume.save_last
        class Stop(Exception):
            pass
        def cut(*args):
            native(*args)
            if len(args[0].history["train_loss"]) == 2:
                raise Stop()
        with patch.object(resume, "save_last", cut), pytest.raises(Stop):
            if observed:
                measure_training(t, train, val, epochs=5, save_last_to=path)
            else:
                t.fit(train, val, epochs=5, save_last_to=path)
        assert not hasattr(t, "_timing_recorder")
        t, train, val = build(case)
        if observed:
            d = measure_training(t, train, val, epochs=5, resume_from=path).to_dict()
        else:
            t.fit(train, val, epochs=5, resume_from=path)
        outputs.append(state(t))
    equal(*outputs)
    assert d["actual_epochs"] == 3 and [e["epoch_id"] for e in d["epochs"]] == [3, 4, 5]
    assert d["counters"]["before"]["attempts"] > 0


@pytest.mark.parametrize("case", CASES)
def test_instrumented_full_replay_matrix(case):
    outputs = []
    for observed in (False, True):
        seed(567)
        t, train, val = build(case)
        if observed:
            measure_training(t, train, val, epochs=5)
        else:
            t.fit(train, val, epochs=5)
        outputs.append(snapshot(t))
    equal(*outputs)


def test_real_uneven_update_matches_independent_reference():
    model = nn.Linear(3, 2).double()
    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.03)
    refopt = torch.optim.SGD(reference.parameters(), lr=.03)
    data = [(x.double(), y) for x, y in loader()]
    for window in (data[:2], data[2:]):
        refopt.zero_grad()
        x = torch.cat([b[0] for b in window])
        y = torch.cat([b[1] for b in window])
        nn.functional.cross_entropy(reference(x), y).backward()
        refopt.step()
    t = Trainer(model, optimizer=optimizer, loss_fn=nn.CrossEntropyLoss(), accum_steps=2)
    d = measure_training(t, data, epochs=1).to_dict()
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-14, atol=1e-16)
    assert d["counters"]["after"]["applied"] == 2


def test_completed_checkpoint_no_fabricated_prefix(tmp_path):
    seed(45)
    t, train, val = build(CRITICAL)
    path = tmp_path / "complete.pth"
    t.fit(train, val, epochs=5, save_last_to=path)
    t, train, val = build(CRITICAL)
    d = measure_training(t, train, val, epochs=5, resume_from=path).to_dict()
    assert d["actual_epochs"] == 0 and d["observations"] == []
    assert d["status"] == "no_new_epochs"
    assert d["counters"]["before"] == d["counters"]["after"]


def test_recorder_failure_cleanup(monkeypatch):
    t = make()
    def fail(*args):
        raise TrainingTimingError("injected recorder failure")
    monkeypatch.setattr(_Recorder, "microbatch_end", fail)
    with pytest.raises(TrainingTimingError, match="injected"):
        measure_training(t, loader(), epochs=1)
    assert not hasattr(t, "_timing_recorder") and not t.optimizer._optimizer_step_post_hooks


def test_tracking_failure_does_not_repeat_fit(tmp_path, monkeypatch):
    t = make()
    r = measure_training(t, loader(), epochs=1)
    before = state(t)
    def forbidden(*args, **kwargs):
        raise AssertionError("must not retrain for tracking")
    monkeypatch.setattr(t, "fit", forbidden)
    with pytest.raises(FileNotFoundError):
        r.to_json(tmp_path / "absent" / "timing.json")
    equal(before, state(t))


def test_training_guide_real_workflow():
    from pathlib import Path
    import re
    guide = Path(__file__).resolve().parents[1] / "docs/guides/performance-training-timing.md"
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", guide.read_text(encoding="utf-8"), re.S):
        exec(compile(block, str(guide), "exec"), namespace)
    assert namespace["data"]["actual_epochs"] == 2


@pytest.mark.parametrize("limit,success", [(7, False), (8, True)])
def test_exact_record_budget_includes_epoch(limit, success):
    t = make()
    if success:
        d = measure_training(t, loader(), epochs=1, protocol=TrainingTimingProtocol(max_records=limit)).to_dict()
        assert len(d["observations"]) + len(d["epochs"]) == limit
    else:
        with pytest.raises(TrainingTimingError, match="max_records"):
            measure_training(t, loader(), epochs=1, protocol=TrainingTimingProtocol(max_records=limit))
