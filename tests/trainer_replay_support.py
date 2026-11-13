"""Test-only native replay harness, also usable against an installed distribution."""
import copy
import math
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hyso import checkpoint
from hyso.callbacks.callbacks import EarlyStopping
from hyso.models.modular import ModularCNN
from hyso.training import Trainer, resume


def equal(actual, expected, path="state"):
    """Exact recursive comparison, including dtype, tensor bits and RNG arrays."""
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        # No tolerance: compare byte views too, including signed zero.
        assert torch.equal(actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                           expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)), path
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype and np.array_equal(actual, expected), path
    elif isinstance(expected, dict):
        assert list(actual) == list(expected), path
        for key in expected:
            equal(actual[key], expected[key], path + "." + str(key))
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for index, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, path + "." + str(index))
    else:
        assert type(actual) is type(expected) and actual == expected, path


def seed(value):
    random.seed(value)
    np.random.seed(value + 1)
    torch.manual_seed(value + 2)


def build(case, device="cpu", amp=False, overflow=False):
    config = dict(input_channels=1, num_classes=2, widths=[3], repeats=[1],
                  strides=[1], attention=case["attention"], norm="batch")
    dtype = torch.float64 if case.get("double", False) else torch.float32
    model = ModularCNN(**config).to(device=device, dtype=dtype)
    kind = case["optimizer"]
    optimizer = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam,
                 "adamw": torch.optim.AdamW}[kind](
        model.parameters(), lr=.03, **({"momentum": .9} if kind == "sgd" else {}))
    batches = math.ceil(7 / case["batch"])
    updates = math.ceil(batches / case["accum"])
    sched = case["scheduler"]
    if sched == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.7)
    elif sched == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
    elif sched == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, .05, total_steps=5 * updates)
    elif sched == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=.5)
    else:
        assert sched == "none"
        scheduler = None
    early = None
    if case.get("early"):
        early = EarlyStopping(patience=1 if case["early"] == "terminal" else 20,
                              delta=1e6 if case["early"] == "terminal" else 0.,
                              verbose=False, restore_best_weights=case.get("best", True))
    trainer = Trainer(model, optimizer=optimizer, scheduler=scheduler, loss_fn=nn.CrossEntropyLoss(),
                      model_config=config, device=device, amp=amp, ema_decay=.8 if case["ema"] else None,
                      early_stopping=early, accum_steps=case["accum"], grad_clip=1.)
    if amp:
        scaler = trainer.scaler.state_dict()
        scaler.update(scale=float(2 ** 20) if overflow else 128., growth_interval=1000)
        trainer.scaler.load_state_dict(scaler)
    generator = torch.Generator().manual_seed(919)
    x = torch.randn(7, 1, 5, 5, generator=generator, dtype=dtype) * .1
    x[:, 0, 0, 0] = torch.arange(7, dtype=dtype)  # observable native sample identity
    data = TensorDataset(x, torch.tensor([0, 1, 1, 0, 1, 0, 1]))
    train = DataLoader(data, batch_size=case["batch"], shuffle=case["shuffle"])
    val = DataLoader(data, batch_size=3) if case["validation"] else None
    # Exercise per-module mixed modes without changing the native model/profile.
    trainer.model.stem.norm.eval()
    if trainer.ema is not None:
        trainer.evaluation_model.train()
        trainer.evaluation_model.classifier.eval()
    return trainer, train, val


def snapshot(trainer):
    return copy.deepcopy(dict(
        model=trainer.model.state_dict(), parameters=dict(trainer.model.named_parameters()),
        optimizer=trainer.optimizer.state_dict(),
        scheduler=None if trainer.scheduler is None else checkpoint._scheduler_state(trainer.scheduler),
        scaler=trainer.scaler.state_dict(),
        # Native load_state_dict resets these lazy initialization seeds to the
        # restored values. They may retain different historical values once
        # active tensors exist; state_dict and all active fields stay exact.
        # Transaction rollback tests separately compare the complete vars().
        scaler_runtime={k: v for k, v in vars(trainer.scaler).items()
                        if k not in {"_init_scale", "_init_growth_tracker"}},
        gradients=[None if p.grad is None else p.grad for p in trainer.model.parameters()],
        ema=None if trainer.ema is None else trainer.ema.ema_model.state_dict(),
        decay=None if trainer.ema is None else trainer.ema.decay,
        early=None if trainer.early_stopping is None else vars(trainer.early_stopping),
        fields={key: getattr(trainer, key) for key in resume._TRAINER_FIELDS},
        modes={n: m.training for n, m in trainer.model.named_modules()},
        ema_modes=None if trainer.ema is None else {n: m.training for n, m in trainer.evaluation_model.named_modules()},
        rng=checkpoint._rng_state()))


class InterruptedReplay(Exception):
    pass


def run_replay(case, directory, boundary=3, device="cpu", amp=False, overflow=False):
    """Compare uninterrupted five-budget training with a coherent boundary restart."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "last.pth"
    native_iter, native_save, native_restore = DataLoader.__iter__, resume.save_last, resume.restore_checkpoint

    def execute(trainer, train, val, *, stop=None, restoring=False):
        trace = dict(order=[], lr=[], epochs=[])
        ready = not restoring
        def restore(*args):
            nonlocal ready
            result = native_restore(*args)
            ready = True
            return result
        def iterate(loader):
            if loader is not train and loader is not val:
                return native_iter(loader)
            assert ready, "iterator opened before restore completed"
            samples = []
            trace["order"].append(("train" if loader is train else "val", samples))
            iterator = native_iter(loader)
            def observed():
                for batch in iterator:
                    samples.extend(int(value) for value in batch[0][:, 0, 0, 0].tolist())
                    # Test-only RNG consumers exercise all supported RNG domains.
                    random.random()
                    np.random.random()
                    torch.rand(1)
                    if device != "cpu":
                        torch.rand(1, device=device)
                    yield batch
            return observed()
        native_update = trainer._update
        def update(samples):
            before = [g["lr"] for g in trainer.optimizer.param_groups]
            native_update(samples)
            trace["lr"].append((before, [g["lr"] for g in trainer.optimizer.param_groups],
                                trainer.attempt_count, trainer.update_count, trainer.ema_update_count))
        def save(*args):
            native_save(*args)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            trace["epochs"].append(payload)
            if stop == payload["completed_epochs"]:
                raise InterruptedReplay()
        trainer._update = update
        try:
            with patch.object(DataLoader, "__iter__", iterate), patch.object(resume, "save_last", save), \
                    patch.object(resume, "restore_checkpoint", restore):
                trainer.fit(train, val, epochs=5, save_last_to=path,
                            resume_from=path if restoring else None)
        except InterruptedReplay:
            assert stop is not None
        else:
            assert stop is None, "requested coherent interruption was not reached"
        finally:
            trainer._update = native_update
        return trace

    seed(331)
    baseline, train, val = build(case, device, amp, overflow)
    initial = snapshot(baseline)
    baseline_trace = execute(baseline, train, val)
    expected = snapshot(baseline)
    seed(331)
    source, train, val = build(case, device, amp, overflow)
    equal(snapshot(source), initial, "initial")
    prefix = execute(source, train, val, stop=boundary)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["completed_epochs"] == boundary
    equal(prefix["epochs"], baseline_trace["epochs"][:boundary], "prefix-checkpoints")
    # Simulate process recreation and deliberately consume every RNG before restore.
    seed(889)
    target, train, val = build(case, device, amp, overflow)
    suffix = execute(target, train, val, restoring=True)
    equal(snapshot(target), expected, "final")
    for key in baseline_trace:
        equal(prefix[key] + suffix[key], baseline_trace[key], "trace." + key)
    final_epochs = 2 if case.get("early") == "terminal" else 5
    assert len(target.history["train_loss"]) == final_epochs and target._finished
    attempts = final_epochs * math.ceil(math.ceil(7 / case["batch"]) / case["accum"])
    assert target.attempt_count == attempts
    if not overflow:
        assert target.update_count == attempts
    else:
        assert 0 < target.update_count < attempts, "native overflow must recover to applied updates"
        assert target.update_count > saved["metrics"]["generic_trainer"]["progress"]["updates"]
    assert target.ema_update_count == (target.update_count if case["ema"] else 0)
    if case["scheduler"] == "onecycle":
        assert target.scheduler.last_epoch == target.update_count
    if boundary == final_epochs:
        assert suffix == dict(order=[], lr=[], epochs=[]), "completed/terminal restore did training work"
    return dict(case=case, boundary=boundary, epochs=final_epochs, attempts=attempts,
                updates=target.update_count, resumed_attempts=len(suffix["lr"]),
                exact_bytes=True, sample_order_equal=True, rng_equal=True,
                scaler=target.scaler.state_dict())


ATTENTIONS = ["none", "se", "cbam", "eca", "coordinate"]
SCHEDULERS = ["none", "step", "cosine", "onecycle", "plateau"]
CASES = []
for a, attention in enumerate(ATTENTIONS):
    for s, scheduler in enumerate(SCHEDULERS):
        index = a * 5 + s
        early = "continue" if index % 3 == 0 else None
        CASES.append(dict(attention=attention, optimizer=["sgd", "adam", "adamw"][(a + s) % 3],
                          scheduler=scheduler, ema=bool(index % 2), early=early,
                          accum=[1, 2, 5][index % 3], batch=[2, 3, 4][(a + 2*s) % 3],
                          shuffle=bool((a+s) % 2), validation=bool(early or scheduler == "plateau" or index % 2),
                          double=bool(index % 3 == 1)))

CRITICAL = dict(attention="eca", optimizer="adamw", scheduler="onecycle", ema=True,
                early="continue", accum=2, batch=3, shuffle=True, validation=True, double=False)
