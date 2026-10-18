"""R09: real optimizer updates and sample-weighted CE normalization."""

import math
import pytest
import torch

from training_helpers import (
    attach_sgd, run_ebo, run_with_sample_weighted_reference,
    synthetic_loader, tiny_ebo,
)
from regression_guards import (
    R09TailStep, R09TailWeights, R09SampleWeighting, expect_value, expect_state, expect_tensor,
    require_optimizer_activity,
)


@pytest.mark.parametrize("batch_count,accum_steps", [(3, 1), (4, 2), (3, 2)])
def test_r09_current_step_counts_and_weight_change(quiet_training, batch_count, accum_steps):
    model = tiny_ebo()
    before = [p.detach().clone() for p in model.parameters()]
    trace = attach_sgd(model)
    history = run_ebo(model, synthetic_loader((2,) * batch_count), accum_steps=accum_steps)
    assert len(trace.after) == math.ceil(batch_count / accum_steps)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert torch.isfinite(torch.tensor(history["train_loss"])).all()


def test_r09_final_partial_window_is_stepped(quiet_training):
    model = tiny_ebo()
    trace = attach_sgd(model)
    loader = synthetic_loader((2, 2, 2))
    run_ebo(model, loader, accum_steps=2)
    expected = math.ceil(len(loader) / 2)
    assert expected == 2
    require_optimizer_activity(trace, (expected,))
    expect_value(len(trace.after), expected, R09TailStep, "R09: final window step count")


@pytest.mark.parametrize("batch_sizes,accum_steps", [((2, 2), 1), ((2, 2, 2, 2), 2)])
def test_r09_full_equal_windows_match_sample_weighted_reference(quiet_training, batch_sizes, accum_steps):
    model, reference, trace, reference_trace, _ = run_with_sample_weighted_reference(batch_sizes, accum_steps)
    assert len(trace.after) == len(reference_trace.after)
    for key, tensor in model.model.state_dict().items():
        torch.testing.assert_close(tensor, reference.state_dict()[key], rtol=1e-5, atol=1e-7)
    for actual, expected in zip(trace.before, reference_trace.before):
        for grad, ref_grad in zip(actual["gradients"], expected["gradients"]):
            torch.testing.assert_close(grad, ref_grad, rtol=1e-5, atol=1e-7)


def test_r09_current_tail_has_half_normalized_unapplied_gradient(quiet_training):
    model, _, trace, reference_trace, _ = run_with_sample_weighted_reference((2, 2, 2), 2)
    assert len(trace.after) == len(reference_trace.after) == 2
    expected = reference_trace.before[-1]["gradients"]
    for actual, grad in zip(trace.before[-1]["gradients"], expected):
        torch.testing.assert_close(actual, grad, rtol=1e-5, atol=1e-7)
    assert all(parameter.grad is None for parameter in model.model.parameters())


def test_r09_tail_parameters_match_correct_partial_window_reference(quiet_training):
    model, reference, trace, reference_trace, _ = run_with_sample_weighted_reference((2, 2, 2), 2)
    require_optimizer_activity(trace, (2,))
    require_optimizer_activity(reference_trace, (2,))
    # Keep the original parameter comparison; validate all schemas first.
    actual = dict(model.model.named_parameters())
    expected = dict(reference.named_parameters())
    expect_state(actual, expected, R09TailWeights, "R09: tail parameter values", rtol=1e-5, atol=1e-7)


def test_r09_unequal_microbatch_sizes_match_sample_weighted_mean(quiet_training):
    # CE reduction='mean': 3+1 samples require weights 3/4 and 1/4, not 1/2.
    model, reference, trace, reference_trace, _ = run_with_sample_weighted_reference((3, 1), 2)
    require_optimizer_activity(trace, (1,))
    require_optimizer_activity(reference_trace, (1,))
    assert len(trace.before[0]["gradients"]) == len(reference_trace.before[0]["gradients"])
    for grad, expected in zip(trace.before[0]["gradients"], reference_trace.before[0]["gradients"]):
        expect_tensor(grad, expected, R09SampleWeighting, "R09: sample-weighted gradient", rtol=1e-5, atol=1e-7)


def test_r09_current_zero_accumulation_raises_division_error(quiet_training):
    model = tiny_ebo()
    attach_sgd(model)
    with pytest.raises(ValueError, match="accum_steps"):
        run_ebo(model, synthetic_loader((2,)), accum_steps=0)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, False, 2.0, None, "2"])
def test_r09_invalid_accumulation_is_rejected_before_training(quiet_training, value):
    model = tiny_ebo()
    model.eval()
    trace = attach_sgd(model)
    forwards = []
    before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    model.model.register_forward_pre_hook(lambda module, inputs: forwards.append(True))
    with pytest.raises((ValueError, TypeError), match="accum_steps"):
        run_ebo(model, synthetic_loader((2, 2, 2)), accum_steps=value)
    assert trace.before == trace.after == []
    assert forwards == []
    assert not model.model.training
    assert all(torch.equal(tensor, before[key]) for key, tensor in model.state_dict().items())


@pytest.mark.parametrize("batch_sizes,accum_steps,epochs,grad_clip,num_classes", [
    ((2, 2, 2), 2, 3, None, 2),
    ((2, 2, 2), 2, 3, 0.05, 2),
    ((2, 2, 2, 2), 2, 2, 0.05, 2),
    ((3, 1), 2, 2, 0.05, 2),
    ((3, 1, 2), 8, 2, None, 2),
    ((3, 1), 2, 2, None, 1),
    ((2, 2, 2), 2, 2, 0.05, 1),
])
def test_r09_windows_epochs_and_clipping_match_microbatch_reference(
    quiet_training, batch_sizes, accum_steps, epochs, grad_clip, num_classes,
):
    model, reference, trace, expected_trace, history = run_with_sample_weighted_reference(
        batch_sizes, accum_steps, epochs=epochs, grad_clip=grad_clip, num_classes=num_classes,
    )
    expected_count = epochs * math.ceil(len(batch_sizes) / accum_steps)
    require_optimizer_activity(trace, (expected_count,))
    require_optimizer_activity(expected_trace, (expected_count,))
    for actual, expected in zip(trace.before, expected_trace.before):
        for grad, ref_grad in zip(actual["gradients"], expected["gradients"]):
            # Normalize-before/after-backward changes FP32 rounding. Across
            # these deterministic fixtures the required absolute budget is
            # < 4 eps (see p0.2-c-completion-report.md). Keep state tolerances
            # unchanged and check the same math tightly in FP64 below.
            torch.testing.assert_close(grad, ref_grad, rtol=1e-5,
                                       atol=4 * torch.finfo(torch.float32).eps)
        if grad_clip is not None:
            norm = torch.cat([grad.flatten() for grad in actual["gradients"]]).norm()
            assert norm <= grad_clip + 1e-6
    for key, tensor in model.model.state_dict().items():
        torch.testing.assert_close(tensor, reference.state_dict()[key], rtol=1e-5, atol=1e-7)
    assert all(parameter.grad is None for parameter in model.model.parameters())
    assert history["train_loss"] == pytest.approx(expected_trace.epoch_losses, rel=1e-6, abs=1e-7)
    assert history["val_loss"] == history["val_acc"] == []


def test_r09_float64_oversized_window_matches_sample_weighted_reference(quiet_training):
    # Lift identical FP32-initialized weights/data; do not combine microbatches
    # into a large batch, which would change BatchNorm and dropout semantics.
    model, reference, trace, expected_trace, history = run_with_sample_weighted_reference(
        (3, 1, 2), 8, epochs=2, dtype=torch.float64,
    )
    require_optimizer_activity(trace, (2,))
    require_optimizer_activity(expected_trace, (2,))
    for actual, expected in zip(trace.before, expected_trace.before):
        assert len(actual["gradients"]) == len(expected["gradients"])
        for grad, ref_grad in zip(actual["gradients"], expected["gradients"]):
            assert grad.dtype == ref_grad.dtype == torch.float64
            torch.testing.assert_close(grad, ref_grad, rtol=1e-11, atol=1e-13)
    for key, tensor in model.model.state_dict().items():
        torch.testing.assert_close(tensor, reference.state_dict()[key], rtol=1e-11, atol=1e-13)
    assert all(parameter.grad is None for parameter in model.model.parameters())
    assert history["train_loss"] == pytest.approx(expected_trace.epoch_losses, rel=1e-11, abs=1e-13)
    assert history["val_loss"] == history["val_acc"] == []


@pytest.mark.parametrize("grad_clip", [None, 0.05])
def test_r09_real_disabled_scaler_orders_unscale_clip_step_update(monkeypatch, quiet_training, grad_clip):
    from hyso.models import bcam

    events = []
    real_scaler = torch.cuda.amp.GradScaler
    real_clip = torch.nn.utils.clip_grad_norm_

    def scaler_factory(*args, **kwargs):
        scaler = real_scaler(*args, **kwargs)
        assert not scaler.is_enabled()
        for name in ("scale", "unscale_", "step", "update"):
            method = getattr(scaler, name)
            def observed(*args, _method=method, _name=name, **kwargs):
                events.append(_name)
                return _method(*args, **kwargs)
            setattr(scaler, name, observed)
        return scaler

    def observed_clip(*args, **kwargs):
        events.append("clip")
        return real_clip(*args, **kwargs)

    monkeypatch.setattr(bcam.torch.cuda.amp, "GradScaler", scaler_factory)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", observed_clip)
    model = tiny_ebo()
    trace = attach_sgd(model)
    run_ebo(model, synthetic_loader((2, 2, 2)), accum_steps=2, grad_clip=grad_clip)
    require_optimizer_activity(trace, (2,))
    update = ["unscale_"] + (["clip"] if grad_clip is not None else []) + ["step", "update"]
    assert events == ["scale", "scale"] + update + ["scale"] + update
