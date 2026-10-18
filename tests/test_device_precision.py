"""R12/R13: CPU precision and a bounded, explicitly simulated CUDA signal."""

import pytest
import torch

from training_helpers import attach_sgd, run_ebo, synthetic_loader, tiny_cemo, tiny_ebo
from regression_guards import R12CPUSelection, expect_value


@pytest.mark.parametrize("use_amp", [False, True])
def test_r12_r13_cpu_training_precision_and_device(monkeypatch, quiet_training, use_amp):
    assert not torch.cuda.is_available(), "This test requires the CPU-only visibility fixture"
    model = tiny_ebo()
    trace = attach_sgd(model)
    forwards, contexts, scalers = [], [], []
    real_autocast = torch.amp.autocast
    real_scaler = torch.cuda.amp.GradScaler

    def autocast(*args, **kwargs):
        contexts.append(dict(kwargs))
        return real_autocast(*args, **kwargs)

    def scaler(*args, **kwargs):
        scalers.append(dict(kwargs))
        return real_scaler(*args, **kwargs)

    def forward(module, inputs, output):
        forwards.append((inputs[0].device.type, inputs[0].dtype, output.device.type, output.dtype))

    monkeypatch.setattr(torch.amp, "autocast", autocast)
    monkeypatch.setattr(torch.cuda.amp, "GradScaler", scaler)
    model.model.register_forward_hook(forward)
    history = run_ebo(model, synthetic_loader((2, 2)), use_amp=use_amp)
    assert len(trace.after) == 2
    assert contexts == [{"device_type": "cpu", "enabled": False}] * 2
    assert scalers == [{"enabled": False}]
    assert forwards == [("cpu", torch.float32, "cpu", torch.float32)] * 2
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters())
    assert torch.isfinite(torch.tensor(history["train_loss"])).all()


class PrecisionSelectionObserved(Exception):
    pass


def simulated_cuda_selection(monkeypatch):
    model = tiny_ebo()
    updates = attach_sgd(model)  # actual CPU optimizer, before the signal is changed
    requested = {}
    real_scaler = torch.cuda.amp.GradScaler

    def scaler(*args, **kwargs):
        requested["scaler_enabled"] = kwargs["enabled"]
        # Only the request is observed. Never allocate/execute a CUDA scaler.
        return real_scaler(enabled=False)

    def autocast(*args, **kwargs):
        requested.update(kwargs)
        # Stop before entering any GPU context or model forward.
        raise PrecisionSelectionObserved

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda.amp, "GradScaler", scaler)
    monkeypatch.setattr(torch.amp, "autocast", autocast)
    with pytest.raises(PrecisionSelectionObserved):
        run_ebo(model, synthetic_loader((2,)), use_amp=True)
    assert updates.after == []
    assert all(p.device.type == "cpu" for p in model.parameters())
    return requested


def test_r12_current_cpu_model_requests_cuda_amp_on_simulated_signal(monkeypatch, quiet_training):
    requested = simulated_cuda_selection(monkeypatch)
    assert requested == {"scaler_enabled": True, "device_type": "cuda", "enabled": True}


@pytest.mark.xfail(strict=True, reason="R12: simulated global CUDA availability overrides the explicit CPU AMP path", raises=R12CPUSelection)
def test_r12_explicit_cpu_model_selects_cpu_amp_path(monkeypatch, quiet_training):
    requested = simulated_cuda_selection(monkeypatch)
    assert set(requested) == {"scaler_enabled", "device_type", "enabled"}
    assert isinstance(requested["scaler_enabled"], bool) and isinstance(requested["enabled"], bool)
    expect_value(requested, {"scaler_enabled": False, "device_type": "cpu", "enabled": False},
                 R12CPUSelection, "R12: explicit CPU precision selection")


def test_r13_current_to_meta_does_not_update_application_device_field():
    # Meta is a storage-free device sentinel; no CUDA use.
    model = tiny_ebo()
    model.to("meta")
    assert all(p.device.type == "meta" for p in model.parameters())
    assert model.device == "cpu"


def test_r13_cemo_cpu_training_input_parameter_device_agreement(quiet_training):
    model = tiny_cemo()
    forwards = []

    def check_device(module, inputs):
        forwards.append(inputs[0].device.type)
        assert inputs[0].device == next(module.parameters()).device

    model.register_forward_pre_hook(check_device)
    history = model.fit(synthetic_loader((2, 2)), epochs=1)
    assert forwards == ["cpu", "cpu"]
    assert torch.isfinite(torch.tensor(history["train_loss"])).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="R12/R13: CUDA hidden/unavailable in the CPU-first test environment")
def test_r12_r13_real_cuda_device_smoke():
    from hyso.models import EBO
    model = EBO(conv_channels=[4, 8], residual_per_block=1, num_classes=2, device="cuda")
    assert next(model.parameters()).device.type == "cuda"
