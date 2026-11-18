"""Save-only v1 integration: real updates, bounded plans and atomic failures."""
import copy
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset, SequentialSampler

from hyso import checkpoint
from hyso.callbacks.callbacks import EarlyStopping
from hyso.models.modular import ModularCNN
from hyso.training import Trainer
from hyso.training.resume import prepare_save, save_last, _early_state


def config(attention="eca", num_classes=2, **kwargs):
    result = dict(input_channels=1, num_classes=num_classes, widths=[3], repeats=[1],
                  strides=[1], attention=attention, norm="batch")
    result.update(kwargs)
    return result


def data(dtype=torch.float32, binary=False, shuffle=False):
    x = torch.randn(5, 1, 5, 5, dtype=dtype)
    y = torch.tensor([0, 1, 1, 0, 1], dtype=dtype if binary else torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=2, shuffle=shuffle)


def build(*, attention="eca", optimizer="adamw", scheduler=None, ema=None,
          early=None, dtype=torch.float32, binary=False, **kwargs):
    cfg = config(attention, 1 if binary else 2)
    model = ModularCNN(**cfg).to(dtype)
    opt = (torch.optim.SGD(model.parameters(), lr=.03, momentum=.9) if optimizer == "sgd"
           else torch.optim.AdamW(model.parameters(), lr=.03))
    if scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=.7)
    elif scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2)
    elif scheduler == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, .05, total_steps=4)
    elif scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=0)
    else:
        sched = None
    options = dict(model_config=cfg, ema_decay=ema, early_stopping=early, scheduler=sched,
                   loss_fn=nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss(),
                   task="binary" if binary else "multiclass", accum_steps=2)
    options.update(kwargs)
    return Trainer(model, optimizer=opt, **options)


def assert_state(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            assert_state(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_state(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("attention", ["none", "se", "cbam", "eca", "coordinate"])
@pytest.mark.parametrize("optimizer", ["sgd", "adamw"])
@pytest.mark.parametrize("scheduler", [None, "step", "cosine", "onecycle"])
@pytest.mark.parametrize("ema", [None, .8])
def test_v1_full_component_save_matrix(tmp_path, attention, optimizer, scheduler, ema):
    instance = build(attention=attention, optimizer=optimizer, scheduler=scheduler, ema=ema)
    train, val = data(shuffle=True), data()
    path = tmp_path / "last.pth"
    history = instance.fit(train, val, epochs=2, save_last_to=path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert set(saved) == {"format", "format_version", "model", "optimizer", "scheduler", "scaler", "ema",
                          "completed_epochs", "history", "metrics", "rng_state"}
    assert saved["format"] == "hyso.training" and saved["format_version"] == checkpoint.FORMAT_VERSION == 1
    assert saved["completed_epochs"] == 2 and saved["history"] == history
    assert all(len(values) == 2 for values in history.values())
    assert_state(saved["model"]["state_dict"], instance.model.state_dict())
    assert_state(saved["optimizer"]["state_dict"], instance.optimizer.state_dict())
    assert saved["optimizer"]["state_dict"]["state"]  # real SGD momentum / AdamW moments
    assert saved["scaler"]["state_dict"] == instance.scaler.state_dict() == {}
    if scheduler is not None:
        assert_state(saved["scheduler"]["state_dict"], checkpoint._scheduler_state(instance.scheduler))
    else:
        assert saved["scheduler"] is None
    if ema is None:
        assert saved["ema"] is None
    else:
        assert saved["ema"]["decay"] == .8
        assert_state(saved["ema"]["state_dict"], instance.ema.ema_model.state_dict())
    metadata = saved["metrics"]["generic_trainer"]
    assert set(metadata) == {"schema_version", "plan", "progress", "early_stopping"}
    assert metadata["schema_version"] == 1 and metadata["early_stopping"] is None
    plan, progress = metadata["plan"], metadata["progress"]
    assert progress["completed_epochs"] == 2
    assert progress["attempts"] == progress["updates"] == instance.update_count == 4
    assert progress["microbatches"] == 6 and progress["ema_updates"] == (0 if ema is None else 4)
    assert progress["model_modes"] == {name: m.training for name, m in instance.model.named_modules()}
    assert progress["ema_modes"] == (None if ema is None else {name: m.training for name, m in instance.ema.ema_model.named_modules()})
    assert plan["epochs"] == 2 and plan["accum_steps"] == 2
    assert plan["model"]["config"] == config(attention)
    assert plan["model"]["convolutions"] and plan["model"]["source_sha256"]
    assert plan["train_loader"]["sampler"].endswith("RandomSampler")
    assert len(plan["train_loader"]["tensors"]) == 2
    assert plan["val_loader"] is not None
    assert plan["optimizer"]["groups"][0]["names"] == saved["optimizer"]["param_names"][0]
    assert set(saved["rng_state"]) == {"python", "numpy", "torch_cpu", "cuda"}
    assert saved["rng_state"]["cuda"] is None
    assert torch.equal(saved["rng_state"]["torch_cpu"], torch.get_rng_state())
    assert any("num_batches_tracked" in name for name in saved["model"]["state_dict"])


@pytest.mark.parametrize("ema", [None, 0., .8])
@pytest.mark.parametrize("restore", [False, True])
def test_last_saved_before_best_restore_with_terminal_early_stop(tmp_path, ema, restore):
    es = EarlyStopping(patience=1, verbose=False, restore_best_weights=restore)
    instance = build(ema=ema, early=es, scheduler="plateau", accum_steps=1)
    with torch.no_grad():
        instance.model.classifier.weight.zero_()
        instance.model.classifier.bias.zero_()
        if instance.ema is not None:
            instance.ema.ema_model.load_state_dict(instance.model.state_dict())
    train = DataLoader(TensorDataset(torch.zeros(2, 1, 5, 5), torch.ones(2, dtype=torch.long)), batch_size=2)
    val = DataLoader(TensorDataset(torch.zeros(3, 1, 5, 5), torch.zeros(3, dtype=torch.long)), batch_size=2)
    path = tmp_path / "terminal.pth"
    history = instance.fit(train, val, epochs=8, save_last_to=path)
    saved = torch.load(path, weights_only=True)
    assert len(history["train_loss"]) == saved["completed_epochs"] == 2
    metadata = saved["metrics"]["generic_trainer"]
    early, progress = metadata["early_stopping"], metadata["progress"]
    assert progress["early_stop"] and early["early_stop"] and early["counter"] == 1
    assert early["target"] == ("live" if ema is None else "ema")
    assert early["patience"] == 1 and early["restore_best_weights"] is restore
    assert early["best_loss"] == history["val_loss"][0]
    assert_state(early["best_state"], es.best_state)
    saved_target = saved["model"] if ema is None else saved["ema"]
    key = next(name for name in es.best_state if name.endswith("num_batches_tracked"))
    assert saved_target["state_dict"][key].item() == 2
    assert early["best_state"][key].item() == 1
    if restore:
        assert instance.evaluation_model.state_dict()[key].item() == 1
        assert not torch.equal(saved_target["state_dict"]["classifier.bias"], instance.evaluation_model.classifier.bias)
    else:
        assert_state(saved_target["state_dict"], instance.evaluation_model.state_dict())
    if ema is not None:
        assert_state(saved["model"]["state_dict"], instance.model.state_dict())
    assert_state(saved["optimizer"]["state_dict"], instance.optimizer.state_dict())
    assert_state(saved["scheduler"]["state_dict"], instance.scheduler.state_dict())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("binary", [False, True])
def test_dtype_task_and_no_validation_save(tmp_path, dtype, binary):
    instance = build(dtype=dtype, binary=binary)
    history = instance.fit(data(dtype, binary), epochs=1, save_last_to=tmp_path / "last.pth")
    saved = torch.load(tmp_path / "last.pth", weights_only=True)
    assert saved["history"] == history and history["val_loss"] == [None]
    assert saved["metrics"]["generic_trainer"]["plan"]["task"] == ("binary" if binary else "multiclass")
    assert saved["metrics"]["generic_trainer"]["plan"]["val_loader"] is None


@pytest.mark.parametrize("kind", ["missing", "extra", "stride", "attention", "bool", "nonfinite", "object", "mutated_conv", "override", "hook", "parameter_hook"])
def test_invalid_model_config_or_structure_rejected_before_update(tmp_path, kind):
    instance = build()
    if kind == "missing":
        instance.model_config = None
    elif kind == "extra":
        instance.model_config["extra"] = 1
    elif kind == "stride":
        instance.model_config["strides"] = [2]
    elif kind == "attention":
        instance.model_config["attention"] = "none"
    elif kind == "bool":
        instance.model_config["input_channels"] = True
    elif kind == "nonfinite":
        instance.model_config["norm"] = float("nan")
    elif kind == "object":
        instance.model_config["norm"] = object()
    elif kind == "mutated_conv":
        instance.model.stages[0][0].conv1.conv.stride = (2, 2)
    elif kind == "override":
        instance.model.forward = lambda inputs: inputs
    elif kind == "parameter_hook":
        next(instance.model.parameters()).register_hook(lambda gradient: gradient)
    else:
        instance.model.register_forward_hook(lambda *args: None)
    before = copy.deepcopy(instance.model.state_dict())
    with pytest.raises(checkpoint.TrainingCheckpointError):
        instance.fit(data(), epochs=1, save_last_to=tmp_path / "bad.pth")
    assert instance.update_count == 0 and not (tmp_path / "bad.pth").exists()
    assert_state(instance.model.state_dict(), before)


@pytest.mark.parametrize("kind", ["list", "subset", "workers", "generator", "collate", "sampler", "empty", "extra_tensor", "requires_grad", "wrong_dtype", "dataset_method", "loader_method"])
def test_unsupported_loader_plan_rejected_before_update(tmp_path, kind):
    instance, loader = build(), data()
    dataset = loader.dataset
    if kind == "list":
        loader = list(loader)
    elif kind == "subset":
        loader = DataLoader(Subset(dataset, [0, 1]), batch_size=2)
    elif kind == "workers":
        loader = DataLoader(dataset, batch_size=2, num_workers=1)
    elif kind == "generator":
        loader = DataLoader(dataset, batch_size=2, generator=torch.Generator())
    elif kind == "collate":
        loader = DataLoader(dataset, batch_size=2, collate_fn=lambda items: items)
    elif kind == "sampler":
        class Custom(SequentialSampler):
            pass
        loader = DataLoader(dataset, batch_size=2, sampler=Custom(dataset))
    elif kind == "empty":
        loader = DataLoader(TensorDataset(torch.empty(0, 1, 5, 5), torch.empty(0, dtype=torch.long)), batch_size=2)
    elif kind == "extra_tensor":
        loader = DataLoader(TensorDataset(*dataset.tensors, dataset.tensors[1]), batch_size=2)
    elif kind == "requires_grad":
        dataset.tensors[0].requires_grad_(True)
    elif kind == "dataset_method":
        dataset.__getitems__ = lambda indices: []
    elif kind == "loader_method":
        loader._get_iterator = lambda: iter([])
    else:
        loader = data(torch.float64)
    with pytest.raises(ValueError):
        instance.fit(loader, epochs=1, save_last_to=tmp_path / "bad.pth")
    assert instance.update_count == 0 and not (tmp_path / "bad.pth").exists()


def test_callbacks_checkpoint_rejected_but_normal_generic_fit_unrestricted(tmp_path):
    instance = build(callbacks=(lambda event, logs: None,))
    with pytest.raises(ValueError, match="callbacks"):
        instance.fit(data(), epochs=1, save_last_to=tmp_path / "last.pth")
    assert instance.update_count == 0
    assert len(instance.fit(data(), epochs=1)["train_loss"]) == 1
    model = nn.Linear(2, 2)
    normal = Trainer(model, optimizer=torch.optim.SGD(model.parameters(), lr=.1),
                     loss_fn=nn.CrossEntropyLoss(), model_config={"arbitrary": "normal fit"})
    normal.fit([(torch.ones(2, 2), torch.tensor([0, 1]))], epochs=1)


def test_prepare_plan_determinism_and_rng_preservation(tmp_path):
    instance, train, val = build(early=EarlyStopping(verbose=False)), data(), data()
    torch_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
    _, first = prepare_save(instance, train, val, 2, tmp_path / "a.pth")
    _, second = prepare_save(instance, train, val, 2, tmp_path / "b.pth")
    assert first == second
    assert torch.equal(torch_rng, torch.get_rng_state()) and python_rng == random.getstate()
    actual = np.random.get_state()
    assert actual[0] == numpy_rng[0] and np.array_equal(actual[1], numpy_rng[1]) and actual[2:] == numpy_rng[2:]
    early = _early_state(instance)
    assert early["best_loss"] is early["best_state"] is None
    instance.model_config["strides"] = [2]
    with pytest.raises(ValueError):
        prepare_save(instance, train, val, 2, tmp_path / "c.pth")


@pytest.mark.parametrize("kind", ["optimizer_config", "scheduler_identity", "scheduler_plan"])
def test_invalid_optimizer_scheduler_metadata_before_update(tmp_path, kind):
    instance = build(scheduler="step")
    if kind == "optimizer_config":
        instance.optimizer.param_groups[0]["weight_decay"] = float("nan")
    elif kind == "scheduler_identity":
        instance.scheduler.optimizer = torch.optim.SGD(instance.model.parameters(), lr=.1)
    else:
        instance.scheduler.gamma = float("nan")
    with pytest.raises(ValueError):
        instance.fit(data(), epochs=2, save_last_to=tmp_path / "bad.pth")
    assert instance.update_count == 0


@pytest.mark.parametrize("failure", ["serialize", "replace"])
def test_atomic_failure_preserves_previous_checkpoint_and_skips_finalization(tmp_path, monkeypatch, failure):
    path = tmp_path / "last.pth"
    first = build()
    first.fit(data(), epochs=1, save_last_to=path)
    previous = path.read_bytes()
    instance = build(early=EarlyStopping(verbose=False))
    if failure == "serialize":
        def broken(payload, stream):
            stream.write(b"partial")
            raise OSError("injected serialization failure")
        monkeypatch.setattr(checkpoint.torch, "save", broken)
    else:
        def broken(*args):
            raise OSError("injected atomic replace failure")
        monkeypatch.setattr(checkpoint.os, "replace", broken)
    with pytest.raises(OSError, match="injected"):
        instance.fit(data(), data(), epochs=3, save_last_to=path)
    assert path.read_bytes() == previous
    assert torch.load(path, weights_only=True)["completed_epochs"] == 1
    assert not list(tmp_path.glob(".last.pth.*.tmp"))
    assert instance.update_count == 2 and len(instance.history["train_loss"]) == 1
    assert not instance._finalized and not instance._finished
    assert instance.model.stem.norm.num_batches_tracked.item() == 3
    with pytest.raises(RuntimeError, match="fresh Trainer"):
        instance.fit(data(), data(), epochs=3)


def test_completed_noop_does_not_create_fake_new_checkpoint(tmp_path):
    instance, train = build(), data()
    instance.fit(train, epochs=1)
    with pytest.raises(RuntimeError, match="initial fit"):
        instance.fit(train, epochs=1, save_last_to=tmp_path / "not-created.pth")
    assert not (tmp_path / "not-created.pth").exists()
    assert instance.fit(train, epochs=1) == instance.history


@pytest.mark.parametrize("optimizer", ["sgd", "adamw"])
@pytest.mark.parametrize("scheduler", [None, "onecycle"])
def test_multiple_optimizer_groups_and_parameter_name_order(tmp_path, optimizer, scheduler):
    instance = build(optimizer=optimizer)
    classifier = list(instance.model.classifier.parameters())
    others = [p for p in instance.model.parameters() if all(p is not q for q in classifier)]
    cls = torch.optim.SGD if optimizer == "sgd" else torch.optim.AdamW
    instance.optimizer = cls([{"params": classifier, "lr": .015}, {"params": others, "lr": .03}])
    if scheduler is not None:
        instance.scheduler = torch.optim.lr_scheduler.OneCycleLR(instance.optimizer, [.02, .04], total_steps=4)
    path = tmp_path / "groups.pth"
    instance.fit(data(), epochs=2, save_last_to=path)
    saved = torch.load(path, weights_only=True)
    assert saved["optimizer"]["param_names"][0] == ["classifier.weight", "classifier.bias"]
    groups = saved["metrics"]["generic_trainer"]["plan"]["optimizer"]["groups"]
    assert [g["names"] for g in groups] == saved["optimizer"]["param_names"]
    assert_state(saved["optimizer"]["state_dict"], instance.optimizer.state_dict())


def test_later_epoch_failure_keeps_first_coherent_epoch(tmp_path, monkeypatch):
    instance = build()
    path = tmp_path / "last.pth"
    real = checkpoint.torch.save
    calls = []
    def fail_second(payload, stream):
        calls.append(payload["completed_epochs"])
        if len(calls) == 2:
            stream.write(b"partial second epoch")
            raise OSError("second epoch failed")
        return real(payload, stream)
    monkeypatch.setattr(checkpoint.torch, "save", fail_second)
    with pytest.raises(OSError, match="second epoch"):
        instance.fit(data(), epochs=3, save_last_to=path)
    assert calls == [1, 2]
    saved = torch.load(path, weights_only=True)
    assert saved["completed_epochs"] == 1 and len(saved["history"]["train_loss"]) == 1
    assert saved["metrics"]["generic_trainer"]["progress"]["updates"] == 2
    assert instance.update_count == 4 and len(instance.history["train_loss"]) == 2
    assert not list(tmp_path.glob(".last.pth.*.tmp"))


def test_dataset_hash_and_stride_attention_configs_distinguish_plans(tmp_path):
    instance, loader = build(), data()
    _, original = prepare_save(instance, loader, None, 2, tmp_path / "a.pth")
    loader.dataset.tensors[0][0, 0, 0, 0] += 1
    _, changed = prepare_save(instance, loader, None, 2, tmp_path / "b.pth")
    assert original["train_loader"]["tensors"][0]["sha256"] != changed["train_loader"]["tensors"][0]["sha256"]
    cfg = config(strides=[2])
    model = ModularCNN(**cfg)
    other = Trainer(model, optimizer=torch.optim.AdamW(model.parameters()), loss_fn=nn.CrossEntropyLoss(), model_config=cfg)
    _, stride = prepare_save(other, loader, None, 2, tmp_path / "c.pth")
    assert original["model"]["convolutions"] != stride["model"]["convolutions"]
    _, attention = prepare_save(build(attention="none"), loader, None, 2, tmp_path / "d.pth")
    assert original["model"]["config"]["attention"] != attention["model"]["config"]["attention"]


def test_invalid_validation_loader_fails_before_first_training_update(tmp_path):
    instance = build()
    with pytest.raises(ValueError, match="DataLoader"):
        instance.fit(data(), list(data()), epochs=1, save_last_to=tmp_path / "bad.pth")
    assert instance.update_count == 0


@pytest.mark.parametrize("kind", ["hook", "state", "scheduler_state", "plain_scheduler_state", "nonfinite_state", "early"])
def test_unsupported_added_component_state_rejected_preflight(tmp_path, kind):
    instance = build(scheduler="step", early=EarlyStopping(verbose=False))
    if kind == "hook":
        instance.optimizer.register_step_post_hook(lambda *args: None)
    elif kind == "state":
        instance.optimizer.state[next(instance.model.parameters())]["extra"] = object()
    elif kind == "scheduler_state":
        instance.scheduler.extra = lambda: None
    elif kind == "plain_scheduler_state":
        instance.scheduler.extra = "not described by the native plan"
    elif kind == "nonfinite_state":
        parameter = next(instance.model.parameters())
        instance.optimizer.state[parameter]["exp_avg"] = torch.full_like(parameter, float("inf"))
    else:
        instance.early_stopping.patience = 0
    with pytest.raises(ValueError):
        instance.fit(data(), data(), epochs=2, save_last_to=tmp_path / "bad.pth")
    assert instance.update_count == 0


def test_save_rechecks_unscheduled_optimizer_configuration(tmp_path):
    instance, loader = build(), data()
    path, plan = prepare_save(instance, loader, None, 1, tmp_path / "last.pth")
    instance.fit(loader, epochs=1)
    instance.optimizer.param_groups[0]["lr"] *= .5
    with pytest.raises(ValueError, match="hyperparameter plan changed"):
        save_last(instance, path, plan, loader, None)
    assert not path.exists()
