"""R10/R11: real scheduler methods, LR sequences and optimizer-step hooks."""

import math

import pytest
import torch

from hyso.callbacks.callbacks import LRSchedulerCallback
from training_helpers import (
    SchedulerProbe, attach_sgd, observe_optimizer, run_ebo, synthetic_loader, tiny_ebo,
)
from regression_guards import (
    R10EpochSchedule, R10OneCycleSchedule, R11UpdateBudget, R09OptimizerUpdates,
    expect_value, expect_tensor, require_optimizer_activity,
)


SCHEDULERS = ["steplr", "reduceonplateau", "cosine"]


def epoch_scheduler_run(monkeypatch, name, route, validation):
    classes = {
        "steplr": torch.optim.lr_scheduler.StepLR,
        "reduceonplateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    }
    parameters = {
        "steplr": {"step_size": 1, "gamma": 0.5},
        "reduceonplateau": {"patience": 0, "factor": 0.5,
                            "threshold": 1e6, "threshold_mode": "abs"},
        "cosine": {"T_max": 4},
    }[name]
    probe = SchedulerProbe(monkeypatch, classes[name])
    model = tiny_ebo()
    options = {}
    if route == "callbacks":
        model.callbacks(
            optimizer="SGD", optimizer_params={"lr": 0.05, "momentum": 0, "weight_decay": 0},
            scheduler="cosineannealinglr" if name == "cosine" else name,
            scheduler_params=parameters,
        )
        optimizer_trace = observe_optimizer(model.optimizer)
    else:
        optimizer_trace = attach_sgd(model)
        options.update(scheduler_name=name, scheduler_params=parameters)
    history = run_ebo(model, synthetic_loader((2, 2)), epochs=2, validation=validation, **options)
    assert len(optimizer_trace.after) == 4
    assert len(history["train_loss"]) == 2
    assert len(probe.records) == 1
    record = probe.records[0]
    record.optimizer_trace = optimizer_trace
    record.history = history
    return record


@pytest.mark.parametrize("name", SCHEDULERS)
@pytest.mark.parametrize("route", ["callbacks", "fit"])
@pytest.mark.parametrize("validation", [False, True])
def test_r10_current_epoch_scheduler_calls_and_lr(monkeypatch, quiet_training, name, route, validation):
    record = epoch_scheduler_run(monkeypatch, name, route, validation)
    count = 0 if name == "reduceonplateau" and not validation else 2
    assert len(record.calls) == count
    assert all(event["error"] is None for event in record.calls)
    expected = []
    for step in range(1, count + 1):
        if name == "steplr":
            expected.append(0.05 * 0.5 ** step)
        elif name == "reduceonplateau":
            expected.append(0.05 * 0.5 ** (step - 1))
        else:
            expected.append(0.05 * (1 + math.cos(math.pi * step / 4)) / 2)
    assert [event["after"] for event in record.calls] == pytest.approx(expected)
    expected_update_lrs = [record.initial_lr] * 2 + [expected[0] if expected else record.initial_lr] * 2
    assert [event["lr"] for event in record.optimizer_trace.before] == pytest.approx(expected_update_lrs)
    if name == "reduceonplateau":
        assert all(len(event["args"]) == 1 and math.isfinite(event["args"][0]) for event in record.calls)
        assert [event["args"][0] for event in record.calls] == record.history["val_loss"]


@pytest.mark.parametrize("name,route,validation", [
    (name, route, validation)
    for name in SCHEDULERS for route in ("callbacks", "fit") for validation in (False, True)
])
def test_r10_epoch_scheduler_expected_cadence(monkeypatch, quiet_training, name, route, validation):
    record = epoch_scheduler_run(monkeypatch, name, route, validation)
    # Plateau needs a validation metric; absent validation, no step is required.
    expected = 0 if name == "reduceonplateau" and not validation else 2
    assert all(event["error"] is None for event in record.calls)
    expect_value(len(record.calls), expected, R10EpochSchedule, "R10: epoch cadence")
    expected_lrs = []
    for step in range(1, expected + 1):
        if name == "steplr":
            expected_lrs.append(0.05 * 0.5 ** step)
        elif name == "reduceonplateau":
            expected_lrs.append(0.05 * 0.5 ** (step - 1))
        else:
            expected_lrs.append(0.05 * (1 + math.cos(math.pi * step / 4)) / 2)
    expect_tensor(torch.tensor([event["after"] for event in record.calls], dtype=torch.float64),
                  torch.tensor(expected_lrs, dtype=torch.float64), R10EpochSchedule,
                  "R10: epoch LR reference", rtol=1e-12, atol=1e-15)


def onecycle_run(monkeypatch, accum_steps, validation, *, batch_sizes=(2, 2, 2), epochs=2):
    model = tiny_ebo()
    optimizer_trace = attach_sgd(model)
    probe = SchedulerProbe(monkeypatch, torch.optim.lr_scheduler.OneCycleLR)
    loader = synthetic_loader(batch_sizes)
    hook_ids = set(model.optimizer._optimizer_step_post_hooks)
    history = run_ebo(model, loader, epochs=epochs, validation=validation, accum_steps=accum_steps,
                      scheduler_name="onecycle", max_lr=0.1)
    assert set(model.optimizer._optimizer_step_post_hooks) == hook_ids
    assert len(probe.records) == 1
    record = probe.records[0]
    record.model = model
    record.history = history
    return record, optimizer_trace


def require_no_scheduler_errors(record):
    for event in record.calls:
        assert event["error"] is None and event["error_message"] is None
        assert event["last_epoch_after"] == event["last_epoch_before"] + 1


def onecycle_lr_reference(total_steps, updates):
    assert total_steps >= updates > 0
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=total_steps)
    before, after = [], []
    for _ in range(updates):
        before.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        after.append(optimizer.param_groups[0]["lr"])
    return before, after


@pytest.mark.parametrize("accum_steps", [1, 2])
@pytest.mark.parametrize("validation", [False, True])
def test_r10_r11_current_onecycle_attempts_errors_and_lr(monkeypatch, quiet_training, accum_steps, validation):
    # Build an independent real-scheduler oracle before installing the spy.
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.05)
    oracle = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.1, steps_per_epoch=math.ceil(3 / accum_steps), epochs=2,
    )
    record, updates = onecycle_run(monkeypatch, accum_steps, validation)
    assert len(updates.after) == 2 * math.ceil(3 / accum_steps)
    attempts = len(updates.after)
    assert len(record.calls) == attempts
    expected_lrs = []
    for _ in range(attempts):
        optimizer.step()
        oracle.step()
        expected_lrs.append(optimizer.param_groups[0]["lr"])
    successful = [event for event in record.calls if event["error"] is None]
    assert [event["after"] for event in successful] == pytest.approx(expected_lrs)
    require_no_scheduler_errors(record)


@pytest.mark.parametrize("accum_steps", [1, 2])
@pytest.mark.parametrize("validation", [False, True])
def test_r10_onecycle_step_matches_actual_optimizer_update(monkeypatch, quiet_training, accum_steps, validation):
    record, updates = onecycle_run(monkeypatch, accum_steps, validation)
    expected = 2 * math.ceil(3 / accum_steps)
    # R09 now applies the tail; scheduler cadence still has an independent oracle.
    require_optimizer_activity(updates, (expected,))
    require_no_scheduler_errors(record)
    before, after = onecycle_lr_reference(record.scheduler.total_steps, len(updates.after))
    expect_value(len(record.calls), len(updates.after), R10OneCycleSchedule, "R10: OneCycle cadence")
    expect_value([event["error"] for event in record.calls], [None] * len(updates.after),
                 R10OneCycleSchedule, "R10: no exhausted-budget exceptions")
    expect_tensor(torch.tensor([event["lr"] for event in updates.before], dtype=torch.float64),
                  torch.tensor(before, dtype=torch.float64), R10OneCycleSchedule,
                  "R10: LR used at each real optimizer update", rtol=1e-12, atol=1e-15)
    expect_tensor(torch.tensor([event["after"] for event in record.calls], dtype=torch.float64),
                  torch.tensor(after, dtype=torch.float64), R10OneCycleSchedule,
                  "R10: post-step OneCycle LR reference", rtol=1e-12, atol=1e-15)


def test_r11_onecycle_budget_counts_accumulated_updates(monkeypatch, quiet_training):
    record, updates = onecycle_run(monkeypatch, accum_steps=2, validation=False)
    expected = 2 * math.ceil(3 / 2)
    assert expected == 4
    require_optimizer_activity(updates, (expected,))
    require_no_scheduler_errors(record)
    expect_value(record.scheduler.total_steps, expected, R11UpdateBudget, "R11: optimizer-update budget")


@pytest.mark.parametrize("accum_steps", [1, 2])
@pytest.mark.parametrize("validation", [False, True])
def test_r09_onecycle_run_has_expected_positive_optimizer_updates(monkeypatch, quiet_training, accum_steps, validation):
    record, updates = onecycle_run(monkeypatch, accum_steps, validation)
    expected = 2 * math.ceil(3 / accum_steps)
    assert expected > 0
    require_optimizer_activity(updates, (expected,))
    require_no_scheduler_errors(record)
    expect_value(len(updates.after), expected, R09OptimizerUpdates, "R09: exact positive optimizer count")


@pytest.mark.parametrize("batch_count,accum_steps,epochs", [
    (4, 2, 1), (3, 2, 3), (3, 8, 2), (4, 1, 2),
])
@pytest.mark.parametrize("validation", [False, True])
def test_r11_onecycle_budget_and_lr_cover_full_tail_and_oversized_windows(
    monkeypatch, quiet_training, batch_count, accum_steps, epochs, validation,
):
    expected = epochs * math.ceil(batch_count / accum_steps)
    record, updates = onecycle_run(monkeypatch, accum_steps, validation,
                                   batch_sizes=(2,) * batch_count, epochs=epochs)
    require_optimizer_activity(updates, (expected,))
    assert record.scheduler.total_steps == expected
    assert len(record.calls) == record.scheduler.last_epoch == expected
    require_no_scheduler_errors(record)
    before, after = onecycle_lr_reference(expected, expected)
    assert record.initial_lr == pytest.approx(before[0], rel=1e-12)
    assert [event["lr"] for event in updates.before] == pytest.approx(before, rel=1e-12)
    assert [event["after"] for event in record.calls] == pytest.approx(after, rel=1e-12)


@pytest.mark.parametrize("skip_indices", [(0,), (1, 3), (0, 1, 2, 3)])
@pytest.mark.parametrize("validation", [False, True])
def test_r11_cpu_scaler_skip_preserves_optimizer_scheduler_pairing(
    monkeypatch, quiet_training, skip_indices, validation,
):
    # The real CPU scaler and autograd run; only optimizer invocation is skipped
    # at selected update opportunities, as standard GradScaler does on overflow.
    real_scaler = torch.cuda.amp.GradScaler
    attempts, results, instances = [], [], []

    class SelectiveSkipScaler:
        def __init__(self, *args, **kwargs):
            self.real = real_scaler(*args, **kwargs)
            assert not self.real.is_enabled()
            self.update_calls = 0
            instances.append(self)

        def scale(self, loss):
            return self.real.scale(loss)

        def get_scale(self):
            return self.real.get_scale()

        def unscale_(self, optimizer):
            return self.real.unscale_(optimizer)

        def step(self, optimizer):
            index = len(attempts)
            before = [p.detach().clone() for group in optimizer.param_groups for p in group["params"]]
            skipped = index in skip_indices
            result = None if skipped else self.real.step(optimizer)
            after = [p.detach().clone() for group in optimizer.param_groups for p in group["params"]]
            attempts.append((skipped, before, after))
            results.append(result)
            return result

        def update(self):
            self.update_calls += 1
            return self.real.update()

    monkeypatch.setattr(torch.cuda.amp, "GradScaler", SelectiveSkipScaler)
    record, updates = onecycle_run(monkeypatch, 2, validation)
    expected = 4 - len(skip_indices)
    assert len(attempts) == instances[0].update_calls == record.scheduler.total_steps == 4
    assert results == [None] * 4  # Successful SGD and skipped steps both return None.
    assert len(updates.before) == len(updates.after) == len(record.calls) == expected
    assert record.scheduler.last_epoch == expected
    require_no_scheduler_errors(record)
    for skipped, before, after in attempts:
        assert all(torch.equal(a, b) for a, b in zip(before, after)) is skipped
    if expected:
        require_optimizer_activity(updates, (expected,))
        before, after = onecycle_lr_reference(4, expected)
        assert [event["lr"] for event in updates.before] == pytest.approx(before, rel=1e-12)
        assert [event["after"] for event in record.calls] == pytest.approx(after, rel=1e-12)
    else:
        # This is an explicit all-skipped scenario, not a vacuous normal 0/0 pass.
        assert all(skipped for skipped, _, _ in attempts)
        assert record.scheduler.optimizer.param_groups[0]["lr"] == record.initial_lr
    assert len(record.history["train_loss"]) == 2
    assert torch.isfinite(torch.tensor(record.history["train_loss"])).all()
    assert all(parameter.grad is None for parameter in record.model.model.parameters())


@pytest.mark.parametrize("fused", [False, True])
def test_r11_real_cpu_scaler_overflow_skips_even_delegated_fused_updates(monkeypatch, quiet_training, fused, record_property):
    # Native CPU GradScaler and native SGD (including fused kernel): no CUDA
    # availability/autocast simulation. Inject one nonfinite backward gradient.
    scaler_type = getattr(torch.amp, "GradScaler", None)
    if scaler_type is None:
        # Minimum Torch has no native CPU scaler. Verify the actual HYSo boundary;
        # do not replace native overflow with a simulated successful update.
        from hyso.training import Trainer
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=.05)
        with pytest.raises(TypeError, match="CPU amp is unsupported"):
            Trainer(model, optimizer=optimizer, loss_fn=torch.nn.CrossEntropyLoss(), amp=True)
        disabled = torch.cuda.amp.GradScaler(enabled=False)
        assert not disabled.is_enabled() and disabled.state_dict() == {}
        assert not optimizer.state and all(p.grad is None for p in model.parameters())
        record_property("cpu_native_overflow", "unsupported_no_native_CPU_GradScaler")
        return
    record_property("cpu_native_overflow", "native_CPU_GradScaler")
    scaler = scaler_type("cpu", init_scale=128, growth_interval=1000)
    monkeypatch.setattr(torch.cuda.amp, "GradScaler", lambda **kwargs: scaler)
    model = tiny_ebo()
    model.optimizer = torch.optim.SGD(model.model.parameters(), lr=0.05, momentum=0, fused=fused)
    before, changed, successful_lrs, pending_lrs = [], [], [], []

    def pre_step(optimizer, args, kwargs):
        before.append([parameter.detach().clone() for parameter in model.model.parameters()])
        pending_lrs.append(optimizer.param_groups[0]["lr"])

    def post_step(optimizer, args, kwargs):
        did_change = any(not torch.equal(a, b) for a, b in zip(before[-1], model.model.parameters()))
        changed.append(did_change)
        if did_change:
            successful_lrs.append(pending_lrs[-1])

    model.optimizer.register_step_pre_hook(pre_step)
    model.optimizer.register_step_post_hook(post_step)
    hook_ids = set(model.optimizer._optimizer_step_post_hooks)
    backwards = []

    def overflow_first_gradient(gradient):
        backwards.append(True)
        return torch.full_like(gradient, float("inf")) if len(backwards) == 1 else gradient

    poison = model.model.stem[0].weight.register_hook(overflow_first_gradient)
    probe = SchedulerProbe(monkeypatch, torch.optim.lr_scheduler.OneCycleLR)
    try:
        history = run_ebo(model, synthetic_loader((2, 2, 2)), epochs=2, accum_steps=2,
                          scheduler_name="onecycle", max_lr=0.1,
                          scheduler_params={"cycle_momentum": False})
    finally:
        poison.remove()
    assert len(backwards) == 6
    assert scaler.get_scale() == 64  # The real scaler detected inf and backed off.
    assert sum(changed) == 3
    assert len(changed) == (4 if fused else 3)
    if fused:
        assert changed == [False, True, True, True]  # step() ran, kernel skipped once.
    record = probe.records[0]
    assert record.scheduler.total_steps == 4
    assert len(record.calls) == record.scheduler.last_epoch == 3
    require_no_scheduler_errors(record)
    expected_before, expected_after = onecycle_lr_reference(4, 3)
    assert successful_lrs == pytest.approx(expected_before, rel=1e-12)
    assert [event["after"] for event in record.calls] == pytest.approx(expected_after, rel=1e-12)
    assert set(model.optimizer._optimizer_step_post_hooks) == hook_ids
    assert torch.isfinite(torch.tensor(history["train_loss"])).all()
    assert all(torch.isfinite(parameter).all() for parameter in model.model.parameters())


@pytest.mark.parametrize("name,route", [
    (name, route) for name in SCHEDULERS for route in ("callbacks", "fit")
] + [("onecycle", "fit")])
def test_r16_unexpected_scheduler_error_propagates_once(monkeypatch, quiet_training, name, route):
    classes = dict(steplr=torch.optim.lr_scheduler.StepLR,
                   reduceonplateau=torch.optim.lr_scheduler.ReduceLROnPlateau,
                   cosine=torch.optim.lr_scheduler.CosineAnnealingLR,
                   onecycle=torch.optim.lr_scheduler.OneCycleLR)
    cls = classes[name]
    original_step = cls.step
    failure = RuntimeError("R16 injected failure after real scheduler step")

    def step_then_fail(scheduler, *args, **kwargs):
        result = original_step(scheduler, *args, **kwargs)
        if scheduler.last_epoch > 0:  # Keep the genuine constructor initial step.
            raise failure
        return result

    monkeypatch.setattr(cls, "step", step_then_fail)
    probe = SchedulerProbe(monkeypatch, cls)
    model = tiny_ebo()
    options = {}
    params = dict(steplr={"step_size": 1}, reduceonplateau={"patience": 0},
                  cosine={"T_max": 2}, onecycle={})[name]
    if route == "callbacks":
        model.callbacks(optimizer="SGD", optimizer_params={"lr": 0.05, "momentum": 0},
                        scheduler="cosineannealinglr" if name == "cosine" else name,
                        scheduler_params=params)
        updates = observe_optimizer(model.optimizer)
    else:
        updates = attach_sgd(model)
        options = dict(scheduler_name=name, scheduler_params=params)
        if name == "onecycle":
            options["max_lr"] = 0.1
    hook_ids = set(model.optimizer._optimizer_step_post_hooks)
    with pytest.raises(RuntimeError) as caught:
        run_ebo(model, synthetic_loader((2, 2)), epochs=2, validation=True, **options)
    assert caught.value is failure
    require_optimizer_activity(updates, (1 if name == "onecycle" else 2,))
    assert len(probe.records) == 1
    record = probe.records[0]
    assert len(record.calls) == record.scheduler.last_epoch == 1
    assert record.calls[0]["error"] == "RuntimeError"
    assert set(model.optimizer._optimizer_step_post_hooks) == hook_ids


@pytest.mark.parametrize("validation", [False, True])
def test_r16_exhausted_onecycle_budget_is_not_suppressed_or_retried(monkeypatch, quiet_training, validation):
    model = tiny_ebo()
    updates = attach_sgd(model)
    probe = SchedulerProbe(monkeypatch, torch.optim.lr_scheduler.OneCycleLR)
    hook_ids = set(model.optimizer._optimizer_step_post_hooks)
    # Preserve explicit PyTorch total_steps overrides; a deliberately short
    # caller budget must fail visibly on the first excess update.
    with pytest.raises(ValueError, match=r"Tried to step 3 times.*total steps is 2"):
        run_ebo(model, synthetic_loader((2, 2, 2)), epochs=2, validation=validation,
                scheduler_name="onecycle", max_lr=0.1, scheduler_params={"total_steps": 2})
    require_optimizer_activity(updates, (3,))
    record = probe.records[0]
    assert record.scheduler.total_steps == 2
    assert len(record.calls) == record.scheduler.last_epoch == 3
    assert [event["error"] for event in record.calls] == [None, None, "ValueError"]
    assert set(model.optimizer._optimizer_step_post_hooks) == hook_ids


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_r10_plateau_rejects_nonfinite_validation_metric(monkeypatch, quiet_training, nonfinite):
    model = tiny_ebo()
    updates = attach_sgd(model)
    probe = SchedulerProbe(monkeypatch, torch.optim.lr_scheduler.ReduceLROnPlateau)
    val_loader = synthetic_loader((2, 2))
    val_loader.dataset.tensors[0].fill_(nonfinite)
    with pytest.raises(ValueError, match="finite validation loss"):
        model.fit(synthetic_loader((2, 2)), val_loader=val_loader, epochs=1,
                  use_amp=False, scheduler_name="reduceonplateau")
    require_optimizer_activity(updates, (2,))
    assert probe.records[0].calls == []


def test_r10_fit_configuration_is_the_single_active_scheduler(monkeypatch, quiet_training):
    step_probe = SchedulerProbe(monkeypatch, torch.optim.lr_scheduler.StepLR)
    cosine_probe = SchedulerProbe(monkeypatch, torch.optim.lr_scheduler.CosineAnnealingLR)
    model = tiny_ebo()
    model.callbacks(optimizer="SGD", optimizer_params={"lr": 0.05, "momentum": 0},
                    scheduler="steplr", scheduler_params={"step_size": 1, "gamma": 0.5})
    old_callback = model.lr_scheduler
    run_ebo(model, synthetic_loader((2, 2)), epochs=2, validation=True,
            scheduler_name="cosine", scheduler_params={"T_max": 4})
    assert old_callback in model._callbacks  # Other callback storage semantics remain unchanged.
    assert model.lr_scheduler.scheduler is cosine_probe.records[0].scheduler
    assert step_probe.records[0].calls == []
    assert len(cosine_probe.records[0].calls) == 2
    assert [event["after"] for event in cosine_probe.records[0].calls] == pytest.approx([
        0.05 * (1 + math.cos(math.pi / 4)) / 2, 0.025,
    ])


@pytest.mark.parametrize("name", SCHEDULERS + ["onecycle"])
def test_r10_scheduler_callback_api_preserves_real_step_semantics(name):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=0.05)
    if name == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        expected = [0.025, 0.0125]
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
        expected = [0.05 * (1 + math.cos(math.pi / 4)) / 2, 0.025]
    elif name == "reduceonplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=0.5)
        expected = [0.05, 0.025]
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=2)
        _, expected = onecycle_lr_reference(2, 2)
    callback = LRSchedulerCallback(scheduler)
    after = []
    for metric in (1.0, 2.0):
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()
        assert callback.step(metric) is None
        after.append(optimizer.param_groups[0]["lr"])
    assert after == pytest.approx(expected, rel=1e-12)
