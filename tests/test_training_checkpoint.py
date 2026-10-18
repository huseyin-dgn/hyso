"""Real CPU component round-trips and strict full-checkpoint boundaries."""

import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from hyso.callbacks.callbacks import ModelCheckpoint
from hyso.checkpoint import TrainingCheckpointError, load_training_checkpoint, save_training_checkpoint
from hyso.models.WithSE import EMA


def assert_state(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            assert_state(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_state(left, right)
    else:
        assert actual == expected


def components(optimizer_name="adam", schedule="step"):
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Tanh(), nn.Linear(4, 2))
    optimizer = (torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) if optimizer_name == "sgd"
                 else {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[optimizer_name](model.parameters(), lr=0.01))
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.6)
                 if schedule == "step" else torch.optim.lr_scheduler.OneCycleLR(
                     optimizer, max_lr=0.03, total_steps=6, cycle_momentum=False))
    return model, optimizer, scheduler


def train_step(model, optimizer, scheduler):
    model.train()
    inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10
    targets = torch.tensor([0, 1, 0, 1])
    optimizer.zero_grad()
    loss = nn.functional.cross_entropy(model(inputs), targets)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()


@pytest.mark.parametrize("optimizer_name", ["adam", "adamw", "sgd"])
@pytest.mark.parametrize("schedule", ["step", "onecycle"])
def test_real_training_components_round_trip(tmp_path, optimizer_name, schedule):
    model, optimizer, scheduler = components(optimizer_name, schedule)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    ema = EMA(model, decay=0.6)
    history = {"train_loss": []}
    for _ in range(2):
        history["train_loss"].append(train_step(model, optimizer, scheduler))
        ema.update(model)
    assert optimizer.state and scheduler.last_epoch == 2
    assert any(not torch.equal(value, ema.ema_model.state_dict()[key])
               for key, value in model.state_dict().items() if value.is_floating_point())
    path = tmp_path / "training.pth"
    save_training_checkpoint(path, model, optimizer=optimizer, scheduler=scheduler,
                             scaler=scaler, ema=ema, completed_epochs=2,
                             history=history, metrics={"loss": history["train_loss"][-1]})
    raw = torch.load(path, weights_only=True, map_location="cpu")
    assert raw["format_version"] == 1 and raw["scaler"]["state_dict"] == {}
    assert raw["rng_state"] is None
    target, target_optimizer, target_scheduler = components(optimizer_name, schedule)
    target_scaler = torch.cuda.amp.GradScaler(enabled=False)
    target_ema = EMA(target, decay=0.1)
    assert any(not torch.equal(value, target.state_dict()[key]) for key, value in model.state_dict().items())
    result = load_training_checkpoint(path, target, optimizer=target_optimizer,
                                     scheduler=target_scheduler, scaler=target_scaler,
                                     ema=target_ema, map_location="cpu")
    for left, right in ((target, model), (target_optimizer, optimizer),
                        (target_scheduler, scheduler), (target_scaler, scaler),
                        (target_ema.ema_model, ema.ema_model)):
        assert_state(left.state_dict(), right.state_dict())
    assert target_ema.decay == 0.6 and all(not p.requires_grad for p in target_ema.ema_model.parameters())
    assert result == {"completed_epochs": 2, "history": history,
                      "metrics": {"loss": history["train_loss"][-1]},
                      "rng_state": None, "rng_restored": False}
    model.eval()
    target.eval()
    with torch.no_grad():
        inputs = torch.ones(4, 3)
        torch.testing.assert_close(target(inputs), model(inputs), rtol=0, atol=0)
    # One manual next step checks restored moments and schedule, not fit resume.
    assert train_step(target, target_optimizer, target_scheduler) == train_step(model, optimizer, scheduler)
    assert_state(target.state_dict(), model.state_dict())
    assert_state(target_optimizer.state_dict(), optimizer.state_dict())
    assert_state(target_scheduler.state_dict(), scheduler.state_dict())
    assert not torch.cuda.is_initialized()


def test_optional_components_are_absent_not_fake_states(tmp_path):
    model, _, _ = components()
    path = tmp_path / "minimal.pth"
    save_training_checkpoint(path, model)
    raw = torch.load(path, weights_only=True)
    assert all(raw[name] is None for name in ("optimizer", "scheduler", "scaler", "ema", "rng_state"))
    target, optimizer, _ = components()
    load_training_checkpoint(path, target)
    assert_state(target.state_dict(), model.state_dict())
    with pytest.raises(TrainingCheckpointError, match="optimizer.*presence"):
        load_training_checkpoint(path, target, optimizer=optimizer)
    with pytest.raises(TrainingCheckpointError, match="rng_state is absent"):
        load_training_checkpoint(path, target, restore_rng=True)


def test_python_numpy_and_cpu_torch_rng_round_trip(tmp_path):
    model, _, _ = components()
    path = tmp_path / "rng.pth"
    save_training_checkpoint(path, model, include_rng=True)
    expected = (random.random(), np.random.random(4), torch.rand(4))
    target, _, _ = components()
    random.random()
    np.random.random(7)
    torch.rand(7)
    result = load_training_checkpoint(path, target, restore_rng=True, map_location="cpu")
    assert result["rng_restored"] is True and result["rng_state"]["cuda"] is None
    assert random.random() == expected[0]
    np.testing.assert_array_equal(np.random.random(4), expected[1])
    torch.testing.assert_close(torch.rand(4), expected[2], rtol=0, atol=0)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("field", ["model", "optimizer", "completed_epochs", "history"])
def test_missing_required_fields_rejected_before_mutation(tmp_path, field):
    model, _, _ = components()
    path = tmp_path / "missing.pth"
    save_training_checkpoint(path, model)
    raw = torch.load(path, weights_only=True)
    del raw[field]
    torch.save(raw, path)
    target, _, _ = components()
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(TrainingCheckpointError, match="fields"):
        load_training_checkpoint(path, target)
    assert_state(target.state_dict(), before)


@pytest.mark.parametrize("version", [0, 2, True, "1"])
def test_unknown_or_invalid_version_rejected(tmp_path, version):
    model, _, _ = components()
    path = tmp_path / "version.pth"
    save_training_checkpoint(path, model)
    raw = torch.load(path, weights_only=True)
    raw["format_version"] = version
    torch.save(raw, path)
    with pytest.raises(TrainingCheckpointError, match="format_version"):
        load_training_checkpoint(path, model)


@pytest.mark.parametrize("name", ["optimizer", "scheduler", "scaler", "ema"])
def test_saved_component_cannot_be_silently_omitted(tmp_path, name):
    model, optimizer, scheduler = components()
    kwargs = {"optimizer": optimizer, "scheduler": scheduler,
              "scaler": torch.cuda.amp.GradScaler(enabled=False), "ema": EMA(model)}
    path = tmp_path / "present.pth"
    save_training_checkpoint(path, model, **kwargs)
    del kwargs[name]
    with pytest.raises(TrainingCheckpointError, match=name):
        load_training_checkpoint(path, model, **kwargs)


def test_incompatible_model_and_optimizer_order_rejected(tmp_path):
    model, optimizer, _ = components()
    path = tmp_path / "schema.pth"
    save_training_checkpoint(path, model, optimizer=optimizer)
    target, target_optimizer, _ = components()
    target_optimizer.param_groups[0]["params"].reverse()
    with pytest.raises(TrainingCheckpointError, match="parameter ordering"):
        load_training_checkpoint(path, target, optimizer=target_optimizer)
    other = nn.Sequential(nn.Linear(3, 2))
    with pytest.raises(TrainingCheckpointError, match="state_dict keys"):
        load_training_checkpoint(path, other, optimizer=torch.optim.Adam(other.parameters()))
    raw = torch.load(path, weights_only=True)
    raw["model"]["state_dict"]["0.weight"] = torch.zeros(1, 1)
    torch.save(raw, path)
    with pytest.raises(TrainingCheckpointError, match="incompatible tensor"):
        load_training_checkpoint(path, model, optimizer=optimizer)


@pytest.mark.parametrize("corruption", ["optimizer_tensor", "optimizer_scalar", "optimizer_missing_moment",
                                      "optimizer_groups", "scheduler_counter", "rng", "extra_field"])
def test_malformed_state_rejected_without_model_mutation(tmp_path, corruption):
    model, optimizer, scheduler = components()
    train_step(model, optimizer, scheduler)
    path = tmp_path / "malformed.pth"
    save_training_checkpoint(path, model, optimizer=optimizer, scheduler=scheduler, include_rng=True)
    raw = torch.load(path, weights_only=True)
    if corruption == "optimizer_tensor":
        next(iter(raw["optimizer"]["state_dict"]["state"].values()))["exp_avg"] = torch.zeros(1)
    elif corruption == "optimizer_scalar":
        next(iter(raw["optimizer"]["state_dict"]["state"].values()))["exp_avg"] = torch.tensor(0.0)
    elif corruption == "optimizer_missing_moment":
        del next(iter(raw["optimizer"]["state_dict"]["state"].values()))["exp_avg"]
    elif corruption == "optimizer_groups":
        raw["optimizer"]["state_dict"]["param_groups"][0]["params"] = 123
    elif corruption == "rng":
        raw["rng_state"]["torch_cpu"] = torch.zeros(3)
    elif corruption == "scheduler_counter":
        raw["scheduler"]["state_dict"]["last_epoch"] = "invalid"
    else:
        raw["unknown"] = "not a version 1 field"
    torch.save(raw, path)
    target, target_optimizer, target_scheduler = components()
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(TrainingCheckpointError):
        load_training_checkpoint(path, target, optimizer=target_optimizer, scheduler=target_scheduler)
    assert_state(target.state_dict(), before)


def test_wrong_component_type_and_scheduler_binding_rejected(tmp_path):
    model, optimizer, scheduler = components()
    path = tmp_path / "types.pth"
    save_training_checkpoint(path, model, optimizer=optimizer)
    with pytest.raises(TrainingCheckpointError, match="optimizer.*type"):
        load_training_checkpoint(path, model, optimizer=torch.optim.SGD(model.parameters(), lr=0.01))
    wrong_optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(TrainingCheckpointError, match="scheduler.*optimizer"):
        save_training_checkpoint(path, model, optimizer=wrong_optimizer, scheduler=scheduler)


def test_corrupt_and_unsafe_files_never_use_pickle_fallback(tmp_path, monkeypatch):
    model, _, _ = components()
    path = tmp_path / "bad.pth"
    path.write_bytes(b"not a checkpoint")
    original_load = torch.load
    requests = []

    def load(*args, **kwargs):
        requests.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    with pytest.raises(TrainingCheckpointError, match="weights-only"):
        load_training_checkpoint(path, model)
    torch.save(nn.Linear(2, 2), path)  # External unsafe file, not our save format.
    with pytest.raises(TrainingCheckpointError, match="weights-only"):
        load_training_checkpoint(path, model)
    assert len(requests) == 2
    assert all(item["weights_only"] is True and item["map_location"] == "cpu" for item in requests)


def test_custom_metadata_and_invalid_epoch_rejected(tmp_path):
    model, _, _ = components()
    path = tmp_path / "invalid.pth"
    with pytest.raises(TrainingCheckpointError, match="unsupported value type"):
        save_training_checkpoint(path, model, history={"custom": object()})
    for value in (-1, True, 1.5):
        with pytest.raises(TrainingCheckpointError, match="completed_epochs"):
            save_training_checkpoint(path, model, completed_epochs=value)
    assert not path.exists()


def test_atomic_failure_preserves_existing_file_and_cleans_temp(tmp_path, monkeypatch):
    model, _, _ = components()
    path = tmp_path / "atomic.pth"
    save_training_checkpoint(path, model)
    before = path.read_bytes()

    def fail(payload, stream):
        stream.write(b"partial")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="simulated disk failure"):
        save_training_checkpoint(path, model)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_restore_failure_rolls_back_already_loaded_components(tmp_path, monkeypatch):
    model, optimizer, scheduler = components()
    train_step(model, optimizer, scheduler)
    path = tmp_path / "rollback.pth"
    save_training_checkpoint(path, model, optimizer=optimizer, scheduler=scheduler)
    target, target_optimizer, target_scheduler = components()
    objects = [target, target_optimizer, target_scheduler]
    before = [copy.deepcopy(obj.state_dict()) for obj in objects]
    original = torch.optim.lr_scheduler.StepLR.load_state_dict
    calls = []

    def fail_once(self, state):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("injected scheduler load failure")
        return original(self, state)

    monkeypatch.setattr(torch.optim.lr_scheduler.StepLR, "load_state_dict", fail_once)
    with pytest.raises(TrainingCheckpointError, match="previous state restored"):
        load_training_checkpoint(path, target, optimizer=target_optimizer, scheduler=target_scheduler)
    assert len(calls) == 2
    for obj, state in zip(objects, before):
        assert_state(obj.state_dict(), state)


def test_legacy_model_checkpoint_remains_raw_and_is_not_full_state(tmp_path):
    model, _, _ = components()
    path = tmp_path / "legacy.pth"
    callback = ModelCheckpoint(str(path), verbose=False)
    callback(0, model, 1.0)
    original = path.read_bytes()
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert_state(raw, model.state_dict())
    assert "format_version" not in raw and "model" not in raw
    callback(1, model, 2.0)
    assert path.read_bytes() == original
    target, _, _ = components()
    target.load_state_dict(raw, strict=True)
    assert_state(target.state_dict(), model.state_dict())
    with pytest.raises(TrainingCheckpointError, match="model-only"):
        load_training_checkpoint(path, target)

