"""Independent CPU references for update math and scheduler ownership.

CUDA evidence lives in cuda_trainer_updates_validation.py, not simulated here.
"""

import copy
import math

import pytest
import torch
from torch import nn

from hyso.training import Trainer


def batches(dtype=torch.float32, task="multiclass"):
    x = torch.randn(11, 3, dtype=dtype)
    y = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
    if task == "binary":
        y = y.to(dtype)
    return list(zip(x.split([2, 1, 3, 4, 1]), y.split([2, 1, 3, 4, 1])))


def build(*, dtype=torch.float32, task="multiclass", optimizer_type=torch.optim.SGD, **kwargs):
    model = nn.Linear(3, 2 if task == "multiclass" else 1).to(dtype)
    optimizer = optimizer_type(model.parameters(), lr=.03)
    loss = nn.CrossEntropyLoss() if task == "multiclass" else nn.BCEWithLogitsLoss()
    return Trainer(model, optimizer=optimizer, loss_fn=loss, task=task, **kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW])
@pytest.mark.parametrize("accum", [1, 2, 4, 9])
@pytest.mark.parametrize("clip", [None, .07])
@pytest.mark.parametrize("task", ["multiclass", "binary"])
def test_unequal_windows_match_concatenated_batch_reference(dtype, optimizer_type, accum, clip, task):
    instance = build(dtype=dtype, task=task, optimizer_type=optimizer_type,
                     accum_steps=accum, grad_clip=clip)
    model = copy.deepcopy(instance.model)
    opt = optimizer_type(model.parameters(), lr=.03)
    data = batches(dtype, task)
    loss_fn = instance.loss_fn
    expected_loss, expected_acc, expected_gradients = [], [], []
    for _ in range(2):
        summed, correct = 0., 0
        for start in range(0, len(data), accum):
            window = data[start:start + accum]
            # Metrics retain the per-microbatch definition. Update gradients
            # come independently from a single concatenated mean loss.
            with torch.no_grad():
                for x, y in window:
                    logits = model(x)
                    if task == "binary":
                        logits = logits.flatten()
                    summed += loss_fn(logits, y).item() * len(x)
                    pred = logits.argmax(1) if task == "multiclass" else (logits >= 0)
                    correct += (pred == y).sum().item()
            opt.zero_grad()
            x = torch.cat([x for x, _ in window])
            y = torch.cat([y for _, y in window])
            logits = model(x)
            loss_fn(logits if task == "multiclass" else logits.flatten(), y).backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            expected_gradients.append([p.grad.detach().clone() for p in model.parameters()])
            opt.step()
        expected_loss.append(summed / 11)
        expected_acc.append(100 * correct / 11)
    observed = []
    handle = instance.optimizer.register_step_pre_hook(
        lambda *args: observed.append([p.grad.detach().clone() for p in instance.model.parameters()]))
    try:
        history = instance.fit(data, epochs=2)
    finally:
        handle.remove()
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    for actual, expected in zip(instance.model.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert len(observed) == len(expected_gradients)
    for actual, expected in zip(observed, expected_gradients):
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)
    assert history["train_loss"] == pytest.approx(expected_loss, rel=tolerance, abs=tolerance)
    assert history["train_acc"] == expected_acc
    assert history["val_loss"] == history["val_acc"] == [None, None]
    assert instance.attempt_count == instance.update_count == 2 * math.ceil(5 / accum)
    assert instance.microbatch_count == 10
    assert all(p.grad is None for p in instance.model.parameters())
    assert not instance.scaler.is_enabled()


def test_native_unscale_then_sample_normalize_then_clip(monkeypatch):
    instance = build(dtype=torch.float64, accum_steps=4, grad_clip=.01)
    data = batches(torch.float64)[:4]
    reference = copy.deepcopy(instance.model)
    x, y = torch.cat([x for x, _ in data]), torch.cat([y for _, y in data])
    nn.functional.cross_entropy(reference(x), y).backward()
    normalized = [p.grad.clone() for p in reference.parameters()]
    events = []
    real_unscale = instance.scaler.unscale_
    real_clip = torch.nn.utils.clip_grad_norm_

    def unscale(opt):
        events.append("unscale")
        return real_unscale(opt)

    def clip(parameters, max_norm, **kwargs):
        events.append("clip")
        parameters = list(parameters)
        for p, expected in zip(parameters, normalized):
            torch.testing.assert_close(p.grad, expected, rtol=1e-12, atol=1e-12)
        norm = real_clip(parameters, max_norm, **kwargs)
        after = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in parameters]))
        assert norm > max_norm and after <= max_norm + 1e-12
        return norm

    monkeypatch.setattr(instance.scaler, "unscale_", unscale)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    instance.fit(data, epochs=1)
    assert events == ["unscale", "clip"]


@pytest.mark.parametrize("option,value,error", [
    ("accum_steps", True, TypeError), ("accum_steps", 1.5, TypeError),
    ("accum_steps", 0, ValueError), ("accum_steps", -2, ValueError),
    ("amp", 1, TypeError), ("amp", None, TypeError), ("amp", True, TypeError),
    ("grad_clip", True, TypeError), ("grad_clip", "1", TypeError),
    ("grad_clip", 0, ValueError), ("grad_clip", -1, ValueError),
    ("grad_clip", float("nan"), ValueError), ("grad_clip", float("inf"), ValueError),
])
def test_invalid_update_options(option, value, error):
    with pytest.raises(error, match=option):
        build(**{option: value})


@pytest.mark.parametrize("kind", ["onecycle", "step", "cosine"])
@pytest.mark.parametrize("accum", [1, 2, 8])
def test_scheduler_cadence_and_lr_sequence_match_native_reference(kind, accum):
    instance = build(accum_steps=accum)
    ref_model = copy.deepcopy(instance.model)
    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=.03)
    updates = math.ceil(5 / accum)
    def schedule(opt):
        if kind == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=.1, total_steps=3 * updates,
                                                       cycle_momentum=False)
        if kind == "step":
            return torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=.7)
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3, eta_min=.001)
    instance.scheduler = schedule(instance.optimizer)
    reference = schedule(ref_opt)
    expected_lrs = []
    data = batches()
    for _ in range(3):
        for start in range(0, 5, accum):
            window = data[start:start + accum]
            x, y = torch.cat([x for x, _ in window]), torch.cat([y for _, y in window])
            ref_opt.zero_grad()
            nn.functional.cross_entropy(ref_model(x), y).backward()
            expected_lrs.append(ref_opt.param_groups[0]["lr"])
            ref_opt.step()
            if kind == "onecycle":
                reference.step()
        if kind != "onecycle":
            reference.step()
    observed = []
    instance.optimizer.register_step_pre_hook(lambda opt, *args: observed.append(opt.param_groups[0]["lr"]))
    instance.fit(data, epochs=3)
    assert observed == pytest.approx(expected_lrs, rel=1e-12)
    assert instance.scheduler.last_epoch == (3 * updates if kind == "onecycle" else 3)
    assert instance.scheduler.get_last_lr() == reference.get_last_lr()
    assert instance.update_count == instance.attempt_count == 3 * updates
    state = copy.deepcopy(instance.scheduler.state_dict())
    instance.fit(data, epochs=3)
    assert instance.scheduler.state_dict() == state


def test_scheduler_optimizer_mismatch():
    instance = build()
    other = torch.optim.SGD(instance.model.parameters(), lr=.03)
    scheduler = torch.optim.lr_scheduler.StepLR(other, step_size=1)
    with pytest.raises(ValueError, match="bound"):
        Trainer(instance.model, optimizer=instance.optimizer, loss_fn=instance.loss_fn, scheduler=scheduler)


@pytest.mark.parametrize("kind", ["plateau", "custom", "invalid_type"])
def test_unsupported_scheduler(kind):
    instance = build()
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(instance.optimizer) if kind == "plateau"
                 else torch.optim.lr_scheduler.ExponentialLR(instance.optimizer, gamma=.9) if kind == "custom"
                 else True)
    if kind == "plateau":
        supported = Trainer(instance.model, optimizer=instance.optimizer, loss_fn=instance.loss_fn, scheduler=scheduler)
        with pytest.raises(ValueError, match="val_loader"):
            supported.fit(batches(), epochs=2)
        assert supported.update_count == 0
        history = supported.fit(batches(), val_loader=batches(), epochs=2)
        assert scheduler.last_epoch == 2
        assert scheduler.best == min(history["val_loss"])
        assert supported.update_count == 10
        return
    with pytest.raises(TypeError if kind == "invalid_type" else NotImplementedError):
        Trainer(instance.model, optimizer=instance.optimizer, loss_fn=instance.loss_fn, scheduler=scheduler)


@pytest.mark.parametrize("kind", ["budget", "advanced", "lr", "total_type", "phases", "boundary", "group_plan",
                                  "step_size", "gamma", "T_max", "eta_min"])
def test_scheduler_bad_state_rejected_before_loader_or_optimizer_mutation(kind):
    instance = build(accum_steps=2)
    if kind in ("step_size", "gamma"):
        sched = torch.optim.lr_scheduler.StepLR(instance.optimizer, 1)
        setattr(sched, kind, 0 if kind == "step_size" else float("nan"))
    elif kind in ("T_max", "eta_min"):
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(instance.optimizer, 2)
        setattr(sched, kind, 0 if kind == "T_max" else float("inf"))
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(instance.optimizer, .1, total_steps=6)
        if kind == "budget":
            sched.total_steps = 9
            sched._schedule_phases[-1]["end_step"] = 8
        elif kind == "advanced":
            instance.optimizer.step()
            sched.step()
        elif kind == "lr":
            instance.optimizer.param_groups[0]["lr"] = .123
        elif kind == "total_type":
            sched.total_steps = True
        elif kind == "phases":
            sched._schedule_phases = []
        elif kind == "boundary":
            sched._schedule_phases[-1]["end_step"] = float("nan")
        elif kind == "group_plan":
            instance.optimizer.param_groups[0]["min_lr"] = -1
    instance.scheduler = sched
    before = copy.deepcopy(instance.model.state_dict())
    class NeverRead:
        def __len__(self):
            return 5
        def __iter__(self):
            raise AssertionError("preflight must run before iterator creation")
    with pytest.raises(ValueError):
        instance.fit(NeverRead(), epochs=2)
    assert instance.attempt_count == instance.update_count == 0
    assert instance.history["train_loss"] == []
    for name, value in instance.model.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize("kind", ["onecycle", "step", "cosine"])
def test_scheduler_exception_is_visible_once_and_hook_removed(monkeypatch, kind):
    instance = build(accum_steps=2)
    if kind == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(instance.optimizer, .1, total_steps=3)
    elif kind == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(instance.optimizer, 1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(instance.optimizer, 1)
    instance.scheduler = scheduler
    real_step = scheduler.step
    calls = []
    def step():
        calls.append(True)
        real_step()
        raise RuntimeError("scheduler failed after native step")
    monkeypatch.setattr(scheduler, "step", step)
    hooks = set(instance.optimizer._optimizer_step_post_hooks)
    with pytest.raises(RuntimeError, match="scheduler failed"):
        instance.fit(batches(), epochs=1)
    assert len(calls) == 1
    assert set(instance.optimizer._optimizer_step_post_hooks) == hooks
    assert all(p.grad is None for p in instance.model.parameters())


def test_optimizer_exception_removes_internal_hook_and_pending_gradients():
    instance = build()
    hooks = set(instance.optimizer._optimizer_step_post_hooks)
    def fail(*args):
        raise RuntimeError("optimizer pre-step failure")
    handle = instance.optimizer.register_step_pre_hook(fail)
    with pytest.raises(RuntimeError, match="pre-step failure"):
        instance.fit(batches(), epochs=1)
    handle.remove()
    assert set(instance.optimizer._optimizer_step_post_hooks) == hooks
    assert instance.attempt_count == 1 and instance.update_count == 0
    assert all(p.grad is None for p in instance.model.parameters())


@pytest.mark.parametrize("kind", ["fused", "custom"])
def test_unsupported_optimizer_profile(kind, record_property):
    model = nn.Linear(3, 2)
    class Custom(torch.optim.SGD):
        pass
    if kind == "fused":
        from torch.utils._foreach_utils import _get_fused_kernels_supported_devices
        if "cpu" not in _get_fused_kernels_supported_devices():
            with pytest.raises(RuntimeError, match=r"fused=True.*supported devices"):
                torch.optim.Adam(model.parameters(), fused=True)
            assert all(p.device.type == "cpu" and p.grad is None for p in model.parameters())
            supported = torch.optim.Adam(model.parameters(), fused=False)
            Trainer(model, optimizer=supported, loss_fn=nn.CrossEntropyLoss())
            assert not supported.state
            record_property("fused_CPU_boundary", "unsupported_native_optimizer_constructor")
            return
        record_property("fused_CPU_boundary", "unsupported_HYSo_admission")
    opt = torch.optim.Adam(model.parameters(), fused=True) if kind == "fused" else Custom(model.parameters(), lr=.1)
    with pytest.raises(NotImplementedError, match="fused"):
        Trainer(model, optimizer=opt, loss_fn=nn.CrossEntropyLoss())


def test_omitted_frozen_parameter_does_not_affect_clipping():
    model = nn.Linear(3, 2).double()
    model.bias.requires_grad_(False)
    model.bias.grad = torch.full_like(model.bias, float("nan"))
    reference = copy.deepcopy(model)
    opt = torch.optim.SGD([model.weight], lr=.03)
    ref_opt = torch.optim.SGD([reference.weight], lr=.03)
    dataset = batches(torch.float64)
    x, y = torch.cat([x for x, _ in dataset]), torch.cat([y for _, y in dataset])
    nn.functional.cross_entropy(reference(x), y).backward()
    torch.nn.utils.clip_grad_norm_([reference.weight], .01)
    ref_opt.step()
    Trainer(model, optimizer=opt, loss_fn=nn.CrossEntropyLoss(), accum_steps=8,
            grad_clip=.01).fit(dataset, epochs=1)
    torch.testing.assert_close(model.weight, reference.weight, rtol=1e-12, atol=1e-12)
    assert torch.equal(model.bias, reference.bias)
