"""R03/R14: real EMA formula, explicit BN policies and CEMO flag matrix."""

import pytest
import torch
from torch import nn
from types import SimpleNamespace

from hyso.models import EMA
from hyso.models.WithSE import CEMO
from training_helpers import synthetic_loader, tiny_cemo, observe_optimizer
from regression_guards import (
    R03FloatBufferCopy, R03IntegerBufferCopy, R14EnableIgnored, R14DisableIgnored,
    expect_tensor, expect_value,
)


def bn_ema_case():
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(2.0)
    ema = EMA(model, decay=0.75, device="cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(6.0)
        model[1].running_mean.fill_(4)
        model[1].running_var.fill_(9)
        model[1].num_batches_tracked.fill_(7)
    return model, ema


def test_r03_parameter_formula_and_frozen_shadow():
    model, ema = bn_ema_case()
    ema.update(model)
    for parameter in ema.ema_model.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 3.0))
        assert not parameter.requires_grad
        assert parameter.grad is None


def test_r03_current_bn_float_and_integer_buffers_remain_at_initial_values():
    model, ema = bn_ema_case()
    ema.update(model)
    torch.testing.assert_close(ema.ema_model[1].running_mean, torch.full((2,), 4.0), rtol=0, atol=0)
    torch.testing.assert_close(ema.ema_model[1].running_var, torch.full((2,), 9.0), rtol=0, atol=0)
    assert ema.ema_model[1].num_batches_tracked.item() == 7
    assert model[1].num_batches_tracked.item() == 7


@pytest.mark.parametrize("name", ["running_mean", "running_var"])
def test_r03_float_bn_buffers_track_training_statistics(name):
    model, ema = bn_ema_case()
    before = getattr(ema.ema_model[1], name).clone()
    live = getattr(model[1], name).clone()
    ema.update(model)
    after = getattr(ema.ema_model[1], name)
    assert torch.all(live != before)
    expect_tensor(after, live, R03FloatBufferCopy, f"R03: exact live {name} copy")


def test_r03_integer_bn_counter_is_copied_from_live_model():
    model, ema = bn_ema_case()
    assert model[1].num_batches_tracked.dtype == torch.int64
    assert model[1].num_batches_tracked.item() == 7
    ema.update(model)
    expect_tensor(ema.ema_model[1].num_batches_tracked, model[1].num_batches_tracked,
                  R03IntegerBufferCopy, "R03: exact live integer buffer copy")


def test_r03_other_registered_buffers_copy_without_aliasing_or_schema_change():
    # Buffer-only modules also work; non-persistent state is copied but remains
    # absent from state_dict. No averaging or integer/bool arithmetic is applied.
    model = nn.Module()
    model.register_buffer("statistics", torch.tensor([1., 2.], dtype=torch.float64))
    model.register_buffer("counter", torch.tensor(1, dtype=torch.int64))
    model.register_buffer("ready", torch.tensor(False))
    model.register_buffer("cache", torch.tensor([3.]), persistent=False)
    ema = EMA(model, decay=0.75, device="cpu")
    state_keys = list(ema.ema_model.state_dict())
    assert state_keys == ["statistics", "counter", "ready"]
    shadow_buffers = dict(ema.ema_model.named_buffers())
    pointers = {name: buffer.data_ptr() for name, buffer in shadow_buffers.items()}
    for increment in (2, 3):
        model.statistics.add_(increment)
        model.counter.add_(increment)
        model.ready.logical_not_()
        model.cache.add_(increment)
        live = {name: buffer.clone() for name, buffer in model.named_buffers()}
        assert ema.update(model) is None
        assert list(ema.ema_model.state_dict()) == state_keys
        for name, buffer in ema.ema_model.named_buffers():
            torch.testing.assert_close(buffer, live[name], rtol=0, atol=0)
            assert buffer.dtype == live[name].dtype and buffer.device == live[name].device
            assert buffer.data_ptr() == pointers[name]
            assert buffer.data_ptr() != dict(model.named_buffers())[name].data_ptr()
            assert buffer.grad_fn is None
    model.statistics.add_(1)
    assert not torch.equal(ema.ema_model.statistics, model.statistics)


@pytest.mark.parametrize("training", [False, True])
def test_r03_shadow_mode_copies_source_and_is_managed_explicitly(training):
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)).train(training)
    ema = EMA(model, device="cpu")
    assert ema.ema_model.training is training
    assert ema.to("cpu") is None
    assert ema.device == "cpu"
    assert all(p.device.type == "cpu" for p in ema.ema_model.parameters())
    assert all(b.device.type == "cpu" for b in ema.ema_model.buffers())
    ema.ema_model.eval()
    assert not ema.ema_model.training
    model.train()
    assert not ema.ema_model.training


def test_r03_parent_state_and_device_traversal_do_not_include_shadow():
    model = tiny_cemo(use_ema=True)
    shadow = model.ema.ema_model
    assert "ema" not in dict(model.named_modules())
    assert not any(key.startswith("ema.") for key in model.state_dict())
    assert list(model.state_dict()) == list(shadow.state_dict())
    model.eval()
    assert shadow.training  # EMA is a plain Python object, not a child module.
    model.to(dtype=torch.float64)
    assert next(model.parameters()).dtype == torch.float64
    assert next(shadow.parameters()).dtype == torch.float32
    model.ema.to("cpu")
    assert all(p.device.type == "cpu" for p in shadow.parameters())


def observe_cemo_flags(monkeypatch, constructor_flag, fit_flag):
    model = tiny_cemo(use_ema=constructor_flag)
    updates, shadow_forwards, parent_forwards = [], [], []
    original_update, original_forward = EMA.update, CEMO.forward
    model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer_trace = observe_optimizer(model.optimizer)
    before = {key: tensor.clone() for key, tensor in model.state_dict().items()}

    def update(ema, source):
        updates.append(ema)
        return original_update(ema, source)

    def forward(instance, x):
        if instance is not model:
            shadow_forwards.append((instance.training, x.device.type))
        else:
            if fit_flag and not parent_forwards:
                assert hasattr(model, "ema") and model.ema.decay == 0.5
                for key, tensor in model.ema.ema_model.state_dict().items():
                    torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)
            parent_forwards.append((instance.training, x.device.type))
        return original_forward(instance, x)

    monkeypatch.setattr(EMA, "update", update)
    monkeypatch.setattr(CEMO, "forward", forward)
    loader = synthetic_loader((2, 2))
    error, history = None, None
    try:
        history = model.fit(loader, val_loader=loader, epochs=1, use_ema=fit_flag)
    except ValueError as exception:
        error = exception
    if error is None:
        assert len(optimizer_trace.after) == 2
        assert sum(training for training, _ in parent_forwards) == 2
        assert any(not torch.equal(parameter, before[name]) for name, parameter in model.named_parameters())
        assert all(gradient is not None and torch.isfinite(gradient).all()
                   for event in optimizer_trace.before for gradient in event["gradients"])
        assert all(torch.isfinite(torch.tensor(values)).all() for values in history.values())
    return SimpleNamespace(model=model, updates=updates, shadow_forwards=shadow_forwards,
                           parent_forwards=parent_forwards, optimizer=optimizer_trace,
                           before=before, error=error)


def require_early_ema_rejection(observation):
    assert isinstance(observation.error, ValueError)
    assert "ema" in str(observation.error).lower()
    assert observation.parent_forwards == []
    assert observation.shadow_forwards == []
    assert observation.updates == []
    assert observation.optimizer.after == []
    assert list(observation.model.state_dict()) == list(observation.before)
    for key, tensor in observation.model.state_dict().items():
        # Parameters and BN floating/integer buffers must all be unchanged.
        torch.testing.assert_close(tensor, observation.before[key], rtol=0, atol=0)


@pytest.mark.parametrize("constructor_flag,fit_flag", [(False, False), (False, True), (True, False), (True, True)])
def test_r14_current_cemo_ema_flag_matrix(monkeypatch, quiet_training, constructor_flag, fit_flag):
    observation = observe_cemo_flags(monkeypatch, constructor_flag, fit_flag)
    assert observation.error is None
    model, updates, forwards = observation.model, observation.updates, observation.shadow_forwards
    assert hasattr(model, "ema") is (constructor_flag or fit_flag)
    assert model.use_ema is constructor_flag
    assert len(updates) == (2 if fit_flag else 0)
    assert len(forwards) == (2 if fit_flag else 0)
    assert sum(not training for training, _ in observation.parent_forwards) == (0 if fit_flag else 2)
    assert all(not training and device == "cpu" for training, device in forwards)


@pytest.mark.parametrize("constructor_flag,fit_flag", [(False, False), (False, True), (True, False), (True, True)])
def test_r14_cemo_fit_flag_controls_ema_use(monkeypatch, quiet_training, constructor_flag, fit_flag):
    observation = observe_cemo_flags(monkeypatch, constructor_flag, fit_flag)
    assert observation.error is None  # Enabling EMA must really work, not merely reject early.
    model, updates, forwards = observation.model, observation.updates, observation.shadow_forwards
    if not constructor_flag and fit_flag:
        expect_value(hasattr(model, "ema"), True, R14EnableIgnored, "R14: EMA activation request")
    else:
        assert hasattr(model, "ema") is (constructor_flag or fit_flag)
    failure = R14EnableIgnored if fit_flag else R14DisableIgnored
    expect_value(len(updates), 2 if fit_flag else 0, failure, "R14: EMA update usage")
    expect_value(len(forwards), 2 if fit_flag else 0, failure, "R14: EMA validation usage")
    assert sum(not training for training, _ in observation.parent_forwards) == (0 if fit_flag else 2)


def test_r14_consecutive_fit_flags_and_default_do_not_leak(monkeypatch, quiet_training):
    model = tiny_cemo(use_ema=False)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = model.optimizer
    trace = observe_optimizer(optimizer)
    real_update, real_forward = EMA.update, CEMO.forward
    updates, forwards = [], []

    def update(ema, source):
        updates.append(len(trace.after))
        return real_update(ema, source)

    def forward(instance, inputs):
        forwards.append((instance is model, instance.training))
        return real_forward(instance, inputs)

    monkeypatch.setattr(EMA, "update", update)
    monkeypatch.setattr(CEMO, "forward", forward)
    loader = synthetic_loader((2, 2))
    keys, shadow = list(model.state_dict()), None
    for call_index, flag in enumerate((True, False, True, None)):
        enabled = flag is True
        before = {name: parameter.clone() for name, parameter in model.named_parameters()}
        shadow_before = None if shadow is None else {key: value.clone() for key, value in shadow.ema_model.state_dict().items()}
        updates.clear()
        forwards.clear()
        options = {} if flag is None else {"use_ema": flag}  # Last call tests the unchanged False default.
        history = model.fit(loader, val_loader=loader, epochs=1, **options)
        assert model.optimizer is optimizer and len(trace.after) == 2 * (call_index + 1)
        assert list(model.state_dict()) == keys and model.use_ema is False
        assert any(not torch.equal(parameter, before[name]) for name, parameter in model.named_parameters())
        assert all(len(values) == 1 and torch.isfinite(torch.tensor(values)).all() for values in history.values())
        if shadow is None:
            shadow = model.ema
        assert model.ema is shadow and shadow.decay == 0.5
        assert forwards.count((True, True)) == 2
        assert forwards.count((True, False)) == (0 if enabled else 2)
        assert forwards.count((False, False)) == (2 if enabled else 0)
        assert len(forwards) == 4
        assert updates == ([2 * call_index + 1, 2 * call_index + 2] if enabled else [])
        if enabled:
            for name, buffer in model.named_buffers():
                torch.testing.assert_close(dict(shadow.ema_model.named_buffers())[name], buffer, rtol=0, atol=0)
        else:
            for key, value in shadow.ema_model.state_dict().items():
                torch.testing.assert_close(value, shadow_before[key], rtol=0, atol=0)
