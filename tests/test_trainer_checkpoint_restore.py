"""Native restore-state checks and bounded real epoch continuation, not the B4-C matrix."""
import copy
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hyso.training import Trainer
from hyso.training import resume
from hyso import checkpoint
from hyso.models.modular import ModularCNN
from hyso.callbacks.callbacks import EarlyStopping


def dataset():
    generator = torch.Generator().manual_seed(932)
    return TensorDataset(torch.randn(5, 1, 5, 5, generator=generator), torch.tensor([0, 1, 1, 0, 1]))


def build(attention="eca", optimizer="adamw", scheduler="step", ema=.8, epochs=5,
          accum=2, early=None, cfg=None, device="cpu", amp=False):
    cfg = cfg or dict(input_channels=1, num_classes=2, widths=[3], repeats=[1], strides=[1], attention=attention, norm="batch")
    model = ModularCNN(**cfg).to(device)
    cls = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[optimizer]
    opt = cls(model.parameters(), lr=.03, **({"momentum": .9} if optimizer == "sgd" else {}))
    if scheduler == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, .05, total_steps=epochs * ((3 + accum - 1) // accum))
    elif scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=.7)
    elif scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    elif scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=0)
    else:
        sched = None
    trainer = Trainer(model, optimizer=opt, scheduler=sched, loss_fn=nn.CrossEntropyLoss(),
                      device=device, amp=amp, model_config=cfg, accum_steps=accum, ema_decay=ema, early_stopping=early)
    return trainer, DataLoader(dataset(), batch_size=2, shuffle=True), DataLoader(dataset(), batch_size=2)


def assert_state(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            assert_state(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_state(a, b)
    else:
        assert actual == expected


def interrupt_after(trainer, train, val, path, monkeypatch, epoch=3, epochs=5):
    native = resume.save_last
    def save_then_stop(*args):
        native(*args)
        if len(args[0].history["train_loss"]) == epoch:
            raise InterruptedError("controlled completed-epoch interruption")
    with monkeypatch.context() as patch:
        patch.setattr(resume, "save_last", save_then_stop)
        with pytest.raises(InterruptedError):
            trainer.fit(train, val, epochs=epochs, save_last_to=path)


@pytest.mark.parametrize("attention", ["none", "se", "cbam", "eca", "coordinate"])
@pytest.mark.parametrize("optimizer", ["sgd", "adam", "adamw"])
@pytest.mark.parametrize("scheduler", ["step", "cosine", "onecycle", "plateau"])
@pytest.mark.parametrize("ema", [None, .8])
def test_completed_restore_all_components_and_modes_without_iterator(tmp_path, monkeypatch, attention, optimizer, scheduler, ema):
    original, train, val = build(attention, optimizer, scheduler, ema, epochs=2)
    original.model.stem.norm.eval()
    if original.ema is not None:
        original.evaluation_model.train()
        original.evaluation_model.classifier.eval()
    path = tmp_path / "completed.pth"
    original.fit(train, val, epochs=2, save_last_to=path)
    saved = torch.load(path, weights_only=True)
    target, train, val = build(attention, optimizer, scheduler, ema, epochs=2)
    target.model.eval()
    target.model.classifier.train()
    native_iter = DataLoader.__iter__
    def no_new_iterator(loader):
        if loader is train or loader is val:
            raise AssertionError("completed restore must not open an iterator")
        return native_iter(loader)
    monkeypatch.setattr(DataLoader, "__iter__", no_new_iterator)
    history = target.fit(train, val, epochs=2, resume_from=path)
    assert history == saved["history"]
    assert_state(target.model.state_dict(), saved["model"]["state_dict"])
    assert_state(target.optimizer.state_dict(), saved["optimizer"]["state_dict"])
    assert_state(checkpoint._scheduler_state(target.scheduler), saved["scheduler"]["state_dict"])
    assert_state(target.scaler.state_dict(), saved["scaler"]["state_dict"])
    if target.ema is not None:
        assert_state(target.ema.ema_model.state_dict(), saved["ema"]["state_dict"])
    progress = saved["metrics"]["generic_trainer"]["progress"]
    assert target.update_count == target.attempt_count == progress["updates"] == 4
    assert target.microbatch_count == 6
    assert target.ema_update_count == progress["ema_updates"]
    assert {n: m.training for n, m in target.model.named_modules()} == progress["model_modes"]
    if target.ema is not None:
        assert {n: m.training for n, m in target.evaluation_model.named_modules()} == progress["ema_modes"]
    assert_state(checkpoint._rng_state(), saved["rng_state"])
    assert target._finished and target._epochs == 2


@pytest.mark.parametrize("scheduler", ["step", "cosine", "onecycle", "plateau"])
@pytest.mark.parametrize("ema", [None, .8])
def test_real_five_epoch_budget_three_plus_two_continuation_smoke(tmp_path, monkeypatch, scheduler, ema):
    torch.manual_seed(72)
    baseline, train, val = build(scheduler=scheduler, ema=ema)
    baseline.fit(train, val, epochs=5)
    expected_rng = checkpoint._rng_state()
    torch.manual_seed(72)
    interrupted, train, val = build(scheduler=scheduler, ema=ema)
    path = tmp_path / "last.pth"
    interrupt_after(interrupted, train, val, path, monkeypatch)
    assert len(interrupted.history["train_loss"]) == 3
    resumed, train, val = build(scheduler=scheduler, ema=ema)
    count = []
    native = resumed._update
    def update(samples):
        count.append(True)
        return native(samples)
    monkeypatch.setattr(resumed, "_update", update)
    resumed.fit(train, val, epochs=5, resume_from=path, save_last_to=path)
    assert len(count) == 4  # exactly two remaining epochs, two real updates each
    assert resumed.history == baseline.history
    assert_state(resumed.model.state_dict(), baseline.model.state_dict())
    assert_state(resumed.optimizer.state_dict(), baseline.optimizer.state_dict())
    assert_state(resumed.scheduler.state_dict(), baseline.scheduler.state_dict())
    if ema is not None:
        assert_state(resumed.ema.ema_model.state_dict(), baseline.ema.ema_model.state_dict())
    assert_state(checkpoint._rng_state(), expected_rng)
    assert resumed.update_count == resumed.attempt_count == 10
    assert torch.load(path, weights_only=True)["completed_epochs"] == 5


@pytest.mark.parametrize("ema", [None, 0., .8])
@pytest.mark.parametrize("restore", [False, True])
def test_terminal_early_stop_restores_same_best_target_without_training(tmp_path, monkeypatch, ema, restore):
    def make():
        es = EarlyStopping(patience=1, verbose=False, restore_best_weights=restore)
        trainer, _, _ = build(scheduler="plateau", optimizer="sgd", ema=ema, early=es, accum=1)
        with torch.no_grad():
            trainer.model.classifier.weight.zero_()
            trainer.model.classifier.bias.zero_()
            if trainer.ema is not None:
                trainer.evaluation_model.load_state_dict(trainer.model.state_dict())
        train = DataLoader(TensorDataset(torch.zeros(2, 1, 5, 5), torch.ones(2, dtype=torch.long)), batch_size=2)
        val = DataLoader(TensorDataset(torch.zeros(3, 1, 5, 5), torch.zeros(3, dtype=torch.long)), batch_size=2)
        return trainer, train, val
    original, train, val = make()
    path = tmp_path / "terminal.pth"
    original.fit(train, val, epochs=5, save_last_to=path)
    assert len(original.history["val_loss"]) == 2
    target, train, val = make()
    monkeypatch.setattr(DataLoader, "__iter__", lambda self: (_ for _ in ()).throw(AssertionError("terminal iterator")))
    result = target.fit(train, val, epochs=5, resume_from=path)
    assert result == original.history
    assert_state(target.model.state_dict(), original.model.state_dict())
    assert_state(target.evaluation_model.state_dict(), original.evaluation_model.state_dict())
    assert_state(vars(target.early_stopping), vars(original.early_stopping))
    assert target.early_stopping.early_stop and target._finalized
    assert target.update_count == target.attempt_count == 2


@pytest.mark.parametrize("mutation", ["schema", "missing", "unknown", "completed", "history", "nan_history", "accuracy",
                                      "attempts", "updates", "ema_count", "bool_counter", "mode_key", "mode_type", "model_tensor",
                                      "optimizer_group", "optimizer_type", "scheduler_counter", "scheduler_lr", "scheduler_config", "namespace", "rng"])
def test_corrupt_metadata_rejected_before_runtime_apply(tmp_path, mutation):
    original, train, val = build(epochs=2)
    path = tmp_path / "last.pth"
    original.fit(train, val, epochs=2, save_last_to=path)
    payload = torch.load(path, weights_only=True)
    meta = payload["metrics"]["generic_trainer"]
    if mutation == "schema": meta["schema_version"] = True
    elif mutation == "missing": del meta["progress"]
    elif mutation == "unknown": meta["extra"] = None
    elif mutation == "completed": payload["completed_epochs"] = 3
    elif mutation == "history": payload["history"]["train_loss"].pop()
    elif mutation == "nan_history": payload["history"]["val_loss"][0] = float("nan")
    elif mutation == "accuracy": payload["history"]["val_acc"][0] = 101.
    elif mutation == "attempts": meta["progress"]["attempts"] += 1
    elif mutation == "updates": meta["progress"]["updates"] -= 1
    elif mutation == "ema_count": meta["progress"]["ema_updates"] = 0
    elif mutation == "bool_counter": meta["progress"]["updates"] = True
    elif mutation == "mode_key": meta["progress"]["model_modes"].pop("stem")
    elif mutation == "mode_type": meta["progress"]["ema_modes"][""] = 1
    elif mutation == "model_tensor": next(iter(payload["model"]["state_dict"].values())).fill_(float("nan"))
    elif mutation == "optimizer_group": payload["optimizer"]["param_names"][0].reverse()
    elif mutation == "optimizer_type": payload["optimizer"]["type"] = "torch.optim.SGD"
    elif mutation == "scheduler_counter": payload["scheduler"]["state_dict"]["last_epoch"] += 1
    elif mutation == "scheduler_lr": payload["scheduler"]["state_dict"]["_last_lr"] = [999.]
    elif mutation == "scheduler_config": payload["scheduler"]["state_dict"]["gamma"] = .5
    elif mutation == "namespace": payload["metrics"] = {}
    elif mutation == "rng": payload["rng_state"] = None
    torch.save(payload, path)
    target, train, val = build(epochs=2)
    before = copy.deepcopy(target.model.state_dict())
    rng = checkpoint._rng_state()
    with pytest.raises(ValueError):
        target.fit(train, val, epochs=2, resume_from=path)
    assert_state(target.model.state_dict(), before)
    assert_state(checkpoint._rng_state(), rng)
    assert target._epochs is None and target.update_count == 0


@pytest.mark.parametrize("mismatch", ["stride", "attention", "width", "norm", "optimizer", "groups", "epochs", "accum", "dataset", "batch", "sampler", "drop_last", "validation", "callbacks"])
def test_live_plan_mismatch_rejected_without_training(tmp_path, mismatch):
    original, train, val = build(epochs=2)
    path = tmp_path / "last.pth"
    original.fit(train, val, epochs=2, save_last_to=path)
    cfg = dict(original.model_config)
    options = dict(epochs=2)
    if mismatch == "stride": cfg["strides"] = [2]
    elif mismatch == "attention": cfg["attention"] = "none"
    elif mismatch == "width": cfg["widths"] = [4]
    elif mismatch == "norm": cfg["norm"] = "group"
    elif mismatch == "optimizer": options["optimizer"] = "sgd"
    elif mismatch == "accum": options["accum"] = 1
    elif mismatch == "epochs": options["epochs"] = 3
    target, train, val = build(cfg=cfg, **options)
    if mismatch == "groups": target.optimizer.param_groups[0]["params"].reverse()
    elif mismatch == "dataset": train.dataset.tensors[0][0, 0, 0, 0] += 1
    elif mismatch == "batch": train = DataLoader(dataset(), batch_size=3, shuffle=True)
    elif mismatch == "sampler": train = DataLoader(dataset(), batch_size=2)
    elif mismatch == "drop_last": train = DataLoader(dataset(), batch_size=2, shuffle=True, drop_last=True)
    elif mismatch == "validation": val = None
    elif mismatch == "callbacks": target.callbacks = (lambda *args: None,)
    before = copy.deepcopy(target.model.state_dict())
    with pytest.raises(ValueError):
        target.fit(train, val, epochs=options["epochs"], resume_from=path)
    assert_state(target.model.state_dict(), before)
    assert target.attempt_count == 0


@pytest.mark.parametrize("unsafe", [False, True])
def test_corrupt_or_unsafe_checkpoint_is_never_unpickled_unsafely(tmp_path, unsafe):
    path = tmp_path / "bad.pth"
    if unsafe:
        torch.save({"unsafe": nn.Linear(1, 1)}, path)
    else:
        path.write_bytes(b"not a checkpoint")
    target, train, val = build()
    with pytest.raises(ValueError, match="weights-only"):
        target.fit(train, val, epochs=5, resume_from=path)
    assert target.update_count == 0


@pytest.mark.parametrize("field", ["target", "counter", "best_loss", "best_state", "extra", "delta_type"])
def test_invalid_early_snapshot_metadata_is_staged_before_apply(tmp_path, field):
    early = EarlyStopping(patience=20, delta=.01, verbose=False, restore_best_weights=False)
    source, train, val = build(epochs=2, early=early)
    path = tmp_path / "early.pth"
    source.fit(train, val, epochs=2, save_last_to=path)
    payload = torch.load(path, weights_only=True)
    state = payload["metrics"]["generic_trainer"]["early_stopping"]
    if field == "target": state["target"] = "live"
    elif field == "counter": state["counter"] = True
    elif field == "best_loss": state["best_loss"] = None
    elif field == "best_state": state["best_state"]["classifier.weight"] = torch.zeros(1)
    elif field == "delta_type": state["delta"] = True
    else: state["extra"] = None
    torch.save(payload, path)
    target, train, val = build(epochs=2, early=EarlyStopping(patience=20, delta=.01, verbose=False, restore_best_weights=False))
    with pytest.raises(ValueError):
        target.fit(train, val, epochs=2, resume_from=path)
    assert target.update_count == 0


@pytest.mark.parametrize("ema", [None, .8])
def test_partial_early_stopping_state_continues_and_finalizes_same_target(tmp_path, monkeypatch, ema):
    def make():
        return build(ema=ema, early=EarlyStopping(patience=10, delta=.001, verbose=False))
    torch.manual_seed(18)
    baseline, train, val = make()
    baseline.fit(train, val, epochs=5)
    torch.manual_seed(18)
    interrupted, train, val = make()
    path = tmp_path / "partial-early.pth"
    interrupt_after(interrupted, train, val, path, monkeypatch)
    resumed, train, val = make()
    resumed.fit(train, val, epochs=5, resume_from=path)
    assert resumed.history == baseline.history
    assert_state(vars(resumed.early_stopping), vars(baseline.early_stopping))
    assert_state(resumed.model.state_dict(), baseline.model.state_dict())
    assert_state(resumed.evaluation_model.state_dict(), baseline.evaluation_model.state_dict())
    assert resumed._finalized and resumed.update_count == 10
