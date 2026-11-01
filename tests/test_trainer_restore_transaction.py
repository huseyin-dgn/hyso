"""Faults delegate to native component restores, then verify complete outer rollback."""
import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from hyso import checkpoint
from hyso.training import resume
from hyso.callbacks.callbacks import EarlyStopping
from test_trainer_checkpoint_restore import build, assert_state, interrupt_after


def snapshot(trainer):
    return dict(model=copy.deepcopy(trainer.model.state_dict()), optimizer=copy.deepcopy(trainer.optimizer.state_dict()),
                scheduler=copy.deepcopy(trainer.scheduler.state_dict()), scaler=copy.deepcopy(trainer.scaler.state_dict()),
                scaler_runtime=copy.deepcopy(vars(trainer.scaler)),
                ema=copy.deepcopy(trainer.ema.ema_model.state_dict()), decay=trainer.ema.decay,
                early=copy.deepcopy(vars(trainer.early_stopping)), history=copy.deepcopy(trainer.history),
                fields={key: copy.deepcopy(getattr(trainer, key)) for key in resume._TRAINER_FIELDS if key != "history"},
                model_modes={n: m.training for n, m in trainer.model.named_modules()},
                ema_modes={n: m.training for n, m in trainer.evaluation_model.named_modules()},
                gradients=[None if p.grad is None else p.grad.clone() for p in trainer.model.parameters()],
                rng=checkpoint._rng_state())


def fixture(tmp_path, monkeypatch, scheduler="step"):
    source, train, val = build(early=EarlyStopping(patience=20, verbose=False), scheduler=scheduler)
    path = tmp_path / "last.pth"
    interrupt_after(source, train, val, path, monkeypatch, epoch=2)
    target, train, val = build(early=EarlyStopping(patience=20, verbose=False), scheduler=scheduler)
    # Real warm native model/AdamW state to distinguish rollback from resetting.
    x, y = train.dataset.tensors
    target.optimizer.zero_grad()
    nn.functional.cross_entropy(target.model(x[:2]), y[:2]).backward()
    target.optimizer.step()
    target.model.eval()
    target.model.classifier.train()
    target.evaluation_model.train()
    target.evaluation_model.classifier.eval()
    target.history["train_loss"] = [99.]
    target.update_count, target.attempt_count, target.microbatch_count, target.ema_update_count = 7, 9, 11, 5
    for p in target.model.parameters():
        p.grad = torch.full_like(p, .375)
    random.seed(431)
    np.random.seed(432)
    torch.manual_seed(433)
    return target, train, val, path


@pytest.mark.parametrize("phase", ["model", "optimizer", "scheduler", "scaler", "ema", "early", "progress", "modes", "rng"])
def test_native_apply_and_late_commit_faults_restore_every_runtime_field(tmp_path, monkeypatch, phase):
    target, train, val, path = fixture(tmp_path, monkeypatch)
    before = snapshot(target)
    fired = []
    if phase in ("model", "ema"):
        native = nn.Module.load_state_dict
        selected = target.model if phase == "model" else target.ema.ema_model
        def fail(self, state, *args, **kwargs):
            result = native(self, state, *args, **kwargs)
            if self is selected and not fired:
                fired.append(True)
                raise RuntimeError("injected after native " + phase)
            return result
        monkeypatch.setattr(nn.Module, "load_state_dict", fail)
    elif phase in ("optimizer", "scheduler", "scaler"):
        selected = {"optimizer": target.optimizer, "scheduler": target.scheduler, "scaler": target.scaler}[phase]
        cls = type(selected)
        native = cls.load_state_dict
        def fail(self, state, *args, **kwargs):
            result = native(self, state, *args, **kwargs)
            if self is selected and not fired:
                fired.append(True)
                raise RuntimeError("injected after native " + phase)
            return result
        monkeypatch.setattr(cls, "load_state_dict", fail)
    elif phase == "rng":
        native = checkpoint._restore_rng
        def fail(state):
            native(state)
            if not fired:
                fired.append(True)
                raise RuntimeError("injected after RNG restore")
        monkeypatch.setattr(checkpoint, "_restore_rng", fail)
    else:
        name = {"early": "_apply_early", "progress": "_apply_progress", "modes": "_apply_modes"}[phase]
        native = getattr(resume, name)
        def fail(*args):
            result = native(*args)
            fired.append(True)
            # All native components have already been applied at these phases.
            saved = torch.load(path, weights_only=True)
            assert_state(target.optimizer.state_dict(), saved["optimizer"]["state_dict"])
            assert_state(target.ema.ema_model.state_dict(), saved["ema"]["state_dict"])
            raise RuntimeError("injected after Trainer " + phase)
        monkeypatch.setattr(resume, name, fail)
    with pytest.raises(checkpoint.TrainingCheckpointError, match="previous runtime restored"):
        target.fit(train, val, epochs=5, resume_from=path)
    assert fired == [True]
    assert_state(snapshot(target), before)


def test_preflight_rejection_does_not_touch_any_warm_runtime_state(tmp_path, monkeypatch):
    target, train, val, path = fixture(tmp_path, monkeypatch)
    payload = torch.load(path, weights_only=True)
    payload["metrics"]["generic_trainer"]["progress"]["attempts"] += 1
    torch.save(payload, path)
    before = snapshot(target)
    with pytest.raises(ValueError, match="progress"):
        target.fit(train, val, epochs=5, resume_from=path)
    assert_state(snapshot(target), before)


def test_failed_rollback_reports_fatal_partial_runtime_explicitly(tmp_path, monkeypatch):
    target, train, val, path = fixture(tmp_path, monkeypatch)
    native = resume._apply_progress
    def fail(*args):
        native(*args)
        raise RuntimeError("late commit failure")
    def cannot_rollback(*args):
        raise OSError("rollback failure")
    monkeypatch.setattr(resume, "_apply_progress", fail)
    monkeypatch.setattr(resume, "_rollback", cannot_rollback)
    with pytest.raises(ValueError, match="rollback failed; runtime may be partially restored") as caught:
        target.fit(train, val, epochs=5, resume_from=path)
    assert "rollback failure" in str(caught.value.__cause__)


def test_checkpoint_path_changed_after_staging_cannot_change_applied_payload(tmp_path, monkeypatch):
    target, train, val, path = fixture(tmp_path, monkeypatch)
    expected = torch.load(path, weights_only=True)
    native = resume._snapshot_runtime
    def replace_after_validation(trainer):
        backup = native(trainer)
        path.write_bytes(b"corrupt replacement after staging")
        return backup
    monkeypatch.setattr(resume, "_snapshot_runtime", replace_after_validation)
    target.fit(train, val, epochs=5, resume_from=path)
    assert target.history["train_loss"][:2] == expected["history"]["train_loss"]
    assert len(target.history["train_loss"]) == 5
    assert target.update_count == 10


def test_successful_restore_clears_inflight_gradients_and_preserves_file_modes(tmp_path, monkeypatch):
    target, train, val, path = fixture(tmp_path, monkeypatch)
    _, plan = resume.prepare_save(target, train, val, 5, path)
    saved = torch.load(path, weights_only=True)
    completed = resume.restore_checkpoint(target, path, plan)
    assert completed == 2 and not target._finished
    assert all(p.grad is None for p in target.model.parameters())
    assert_state(target.model.state_dict(), saved["model"]["state_dict"])
    assert_state(target.optimizer.state_dict(), saved["optimizer"]["state_dict"])
    assert_state(target.scaler.state_dict(), saved["scaler"]["state_dict"])
    assert_state(vars(target.early_stopping)["best_state"], saved["metrics"]["generic_trainer"]["early_stopping"]["best_state"])
    assert_state(checkpoint._rng_state(), saved["rng_state"])
