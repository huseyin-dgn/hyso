"""Epoch-boundary EBO replay with the original five-epoch training budget."""

import inspect
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from hyso.callbacks.callbacks import EarlyStopping
from hyso.checkpoint import TrainingCheckpointError, save_training_checkpoint
from hyso.models import EBO, bcam
from test_training_checkpoint import assert_state
from training_helpers import tiny_ebo, tiny_cemo


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.random.default_generator.manual_seed(value)


def loader(shuffle=True):
    data = torch.randn(8, 3, 16, 16, generator=torch.Generator().manual_seed(71))
    return DataLoader(TensorDataset(data, torch.arange(8) % 2), batch_size=3, shuffle=shuffle)


def options(optimizer="adamw", scheduler="steplr", ema=False):
    return dict(epochs=5, lr=0.005, optimizer_name=optimizer, weight_decay=0.001,
                use_ema=ema, use_amp=True, accum_steps=2, grad_clip=0.5,
                scheduler_name=scheduler, max_lr=0.03,
                scheduler_params=({"step_size": 2, "gamma": 0.5} if scheduler == "steplr"
                                  else {"cycle_momentum": False}))


@pytest.fixture
def presentation(monkeypatch, quiet_training):
    calls = []

    def summary(model, shape):
        calls.append(True)
        random.random()
        np.random.random()
        with torch.no_grad():
            model(torch.randn(4, *shape))  # Real dropout/BN and RNG side effects.

    def no_plot(*args, **kwargs):
        raise RuntimeError("plot disabled for numeric regression")

    monkeypatch.setattr(bcam, "summary", summary)
    monkeypatch.setattr(bcam.plt, "figure", no_plot)
    monkeypatch.setattr(bcam, "classification_report", lambda *args, **kwargs: "")
    return calls


class EpochBoundaryStop(Exception):
    pass


def interrupted(monkeypatch, model, train, path, opts, *, validation=None, stop=3):
    real_save = bcam.save_training_checkpoint

    def save(*args, **kwargs):
        result = real_save(*args, **kwargs)
        if kwargs["completed_epochs"] == stop:
            raise EpochBoundaryStop
        return result

    with monkeypatch.context() as patch:
        patch.setattr(bcam, "save_training_checkpoint", save)
        with pytest.raises(EpochBoundaryStop):
            model.fit(train, val_loader=validation, save_last_to=path, **opts)


@pytest.mark.parametrize("optimizer", ["adam", "adamw"])
@pytest.mark.parametrize("scheduler", ["steplr", "onecycle"])
@pytest.mark.parametrize("ema", [False, True])
def test_five_epoch_uninterrupted_equals_three_plus_resume(
        tmp_path, monkeypatch, presentation, optimizer, scheduler, ema):
    opts = options(optimizer, scheduler, ema)
    full_path, cut_path, resumed_path = (tmp_path / name for name in ("full.pth", "cut.pth", "resumed.pth"))
    seed(19)
    full = tiny_ebo()
    full_history = full.fit(loader(), val_loader=loader(False), save_last_to=full_path, **opts)
    full_state = torch.load(full_path, weights_only=True, map_location="cpu")
    seed(19)
    first = tiny_ebo()
    interrupted(monkeypatch, first, loader(), cut_path, opts, validation=loader(False))
    cut_state = torch.load(cut_path, weights_only=True)
    assert cut_state["completed_epochs"] == 3
    assert len(cut_state["history"]["train_loss"]) == 3
    assert cut_state["metrics"]["ebo_fit"]["epochs"] == 5
    if scheduler == "onecycle":
        assert cut_state["scheduler"]["state_dict"]["total_steps"] == 10
        assert cut_state["scheduler"]["state_dict"]["last_epoch"] == 6
    seed(999)  # New object/setup must not determine the resumed RNG stream.
    resumed = tiny_ebo()
    presentation.clear()
    history = resumed.fit(loader(), val_loader=loader(False), resume_from=cut_path,
                          save_last_to=resumed_path, **opts)
    assert presentation == []  # No summary/sample forward on resume.
    resumed_state = torch.load(resumed_path, weights_only=True, map_location="cpu")
    assert_state(resumed_state, full_state)
    assert_state(resumed.state_dict(), full.state_dict())
    assert_state(resumed.optimizer.state_dict(), full.optimizer.state_dict())
    assert_state(resumed.lr_scheduler.scheduler.state_dict(), full.lr_scheduler.scheduler.state_dict())
    assert history == full_history and all(len(values) == 5 for values in history.values())
    assert resumed_state["completed_epochs"] == 5
    assert resumed_state["scaler"]["state_dict"] == {}  # Real disabled AMP scaler.
    assert all(state["step"].item() == 10 for state in resumed.optimizer.state.values())
    assert resumed.lr_scheduler.scheduler.last_epoch == (10 if scheduler == "onecycle" else 5)
    if ema:
        assert_state(resumed.ema.ema_model.state_dict(), full.ema.ema_model.state_dict())
        assert resumed.ema.decay == full.ema.decay
    else:
        assert not hasattr(resumed, "ema") and resumed_state["ema"] is None
    assert not torch.cuda.is_initialized()


@pytest.fixture
def saved_epoch(tmp_path, monkeypatch, presentation):
    seed(41)
    path = tmp_path / "source.pth"
    interrupted(monkeypatch, tiny_ebo(), loader(), path, options(), stop=1)
    return path


@pytest.mark.parametrize("change", [
    "accumulation", "total_epochs", "scheduler_missing", "scheduler_gamma", "optimizer",
    "learning_rate", "grad_clip", "amp", "ema", "batch_size", "shuffle", "dataset",
    "history", "history_value", "scheduler_state", "completed_epochs", "missing_plan", "missing_field", "version", "corrupt", "legacy", "model_type",
])
def test_incompatible_resume_fails_before_model_or_rng_mutation(
        saved_epoch, change, presentation):
    path = saved_epoch
    opts = options()
    train = loader()
    if change == "accumulation":
        opts["accum_steps"] = 1
    elif change == "total_epochs":
        opts["epochs"] = 6
    elif change == "scheduler_missing":
        opts["scheduler_name"] = None
    elif change == "scheduler_gamma":
        opts["scheduler_params"] = {"step_size": 2, "gamma": 0.9}
    elif change == "optimizer":
        opts["optimizer_name"] = "adam"
    elif change == "learning_rate":
        opts["lr"] = 0.02
    elif change == "grad_clip":
        opts["grad_clip"] = None
    elif change == "amp":
        opts["use_amp"] = False
    elif change == "ema":
        opts["use_ema"] = True
    elif change == "batch_size":
        train = DataLoader(train.dataset, batch_size=2, shuffle=True)
    elif change == "shuffle":
        train = loader(False)
    elif change == "dataset":
        train.dataset.tensors[0][0].add_(1)
    elif change == "corrupt":
        path.write_bytes(b"corrupt checkpoint")
    elif change == "legacy":
        torch.save(tiny_ebo().model.state_dict(), path)
    elif change == "model_type":
        save_training_checkpoint(path, tiny_cemo(), include_rng=True)
    else:
        state = torch.load(path, weights_only=True)
        if change == "history":
            state["history"]["train_loss"].pop()
        elif change == "history_value":
            state["history"]["train_loss"][0] = "invalid"
        elif change == "scheduler_state":
            state["scheduler"]["state_dict"]["gamma"] = 0.9
        elif change == "completed_epochs":
            state["completed_epochs"] = 6
        elif change == "missing_plan":
            state["metrics"] = None
        elif change == "missing_field":
            del state["scheduler"]
        elif change == "version":
            state["format_version"] = 999
        torch.save(state, path)
    target = tiny_ebo()
    before = {name: value.clone() for name, value in target.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(TrainingCheckpointError):
        target.fit(train, resume_from=path, **opts)
    assert_state(target.state_dict(), before)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)


@pytest.mark.parametrize("epochs", [3, 6])
def test_onecycle_rejects_changed_original_budget(tmp_path, monkeypatch, presentation, epochs):
    path = tmp_path / "cycle.pth"
    interrupted(monkeypatch, tiny_ebo(), loader(), path, options(scheduler="onecycle"), stop=1)
    opts = options(scheduler="onecycle")
    opts["epochs"] = epochs
    with pytest.raises(TrainingCheckpointError, match="training plan mismatch"):
        tiny_ebo().fit(loader(), resume_from=path, **opts)


@pytest.mark.parametrize("field", ["total_steps", "last_epoch", "phase"])
def test_onecycle_rejects_state_that_contradicts_saved_plan(tmp_path, monkeypatch, presentation, field):
    path = tmp_path / "inconsistent-cycle.pth"
    opts = options(scheduler="onecycle")
    interrupted(monkeypatch, tiny_ebo(), loader(), path, opts, stop=1)
    state = torch.load(path, weights_only=True)
    schedule = state["scheduler"]["state_dict"]
    if field == "phase":
        schedule["_schedule_phases"][0]["end_step"] += 1
    else:
        schedule[field] += 1
    torch.save(state, path)
    with pytest.raises(TrainingCheckpointError, match="scheduler|OneCycle"):
        tiny_ebo().fit(loader(), resume_from=path, **opts)


@pytest.mark.parametrize("case", ["generator", "batch_sampler", "workers", "sampler", "collate", "custom_loader", "callback"])
def test_unsupported_loader_and_callback_state_is_rejected(tmp_path, presentation, case):
    train = loader()
    model = tiny_ebo()
    if case == "generator":
        train = DataLoader(train.dataset, batch_size=3, generator=torch.Generator())
    elif case == "batch_sampler":
        train = DataLoader(train.dataset, batch_sampler=[[0, 1], [2, 3]])
    elif case == "workers":
        train = DataLoader(train.dataset, batch_size=3, num_workers=1)
    elif case == "sampler":
        train = DataLoader(train.dataset, batch_size=3,
                           sampler=WeightedRandomSampler(torch.ones(8), 8))
    elif case == "collate":
        train = DataLoader(train.dataset, batch_size=3, collate_fn=lambda values: values)
    elif case == "custom_loader":
        class CustomLoader(DataLoader):
            pass
        train = CustomLoader(train.dataset, batch_size=3)
    else:
        model._callbacks.append(EarlyStopping())
    path = tmp_path / "unsupported.pth"
    with pytest.raises(TrainingCheckpointError, match="checkpoint"):
        model.fit(train, save_last_to=path, **options())
    assert not path.exists()


@pytest.mark.parametrize("validation", [False, True])
def test_completed_budget_is_noop_and_history_and_mode_are_preserved(tmp_path, presentation, validation):
    path, unused = tmp_path / "complete.pth", tmp_path / "unused.pth"
    opts = options()
    opts["scheduler_name"] = None
    source = tiny_ebo()
    history = source.fit(loader(False), val_loader=loader(False) if validation else None, save_last_to=path, **opts)
    target = tiny_ebo()
    result = target.fit(loader(False), val_loader=loader(False) if validation else None,
                        resume_from=path, save_last_to=unused, **opts)
    assert result == history and len(result["train_loss"]) == 5
    assert len(result["val_loss"]) == len(result["val_acc"]) == (5 if validation else 0)
    assert target.model.training is source.model.training
    assert_state(target.state_dict(), source.state_dict())
    assert not unused.exists()


def test_resume_keywords_are_keyword_only():
    signature = inspect.signature(EBO.fit)
    for name in ("resume_from", "save_last_to"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default is None

