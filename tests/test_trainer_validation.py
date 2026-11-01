"""Pure validation metrics, mode/buffer restoration and EMA target selection."""
import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hyso.training import Trainer


def make(dtype=torch.float32, task="multiclass", **kwargs):
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2 if task == "multiclass" else 1)).to(dtype)
    opt = torch.optim.Adam(model.parameters(), lr=.02)
    return Trainer(model, optimizer=opt, loss_fn=nn.CrossEntropyLoss() if task == "multiclass" else nn.BCEWithLogitsLoss(),
                   task=task, **kwargs)


def data(dtype=torch.float32, task="multiclass", size=5):
    x = torch.randn(size, 2, dtype=dtype)
    y = torch.arange(size) % 2
    return DataLoader(TensorDataset(x, y if task == "multiclass" else y.to(dtype)), batch_size=2)


def assert_state(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_state(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("task", ["multiclass", "binary"])
@pytest.mark.parametrize("ema", [None, 0., .8])
def test_validation_exact_metrics_and_no_training_mutation(dtype, task, ema):
    instance = make(dtype, task, ema_decay=ema)
    # Populate real Adam/BN state, with no single-sample BN training batch.
    instance.fit(data(dtype, task, size=4), epochs=1)
    target = instance.evaluation_model
    target.train()
    target[0].eval()  # deliberate mixed mode restored independently
    modes = [m.training for m in target.modules()]
    state = copy.deepcopy(target.state_dict())
    live = copy.deepcopy(instance.model.state_dict())
    optimizer = copy.deepcopy(instance.optimizer.state_dict())
    history = copy.deepcopy(instance.history)
    counts = (instance.update_count, instance.attempt_count, instance.ema_update_count)
    val = data(dtype, task)
    x, y = val.dataset.tensors
    reference = copy.deepcopy(target).eval()
    with torch.no_grad():
        logits = reference(x)
        if task == "binary":
            logits = logits.flatten()
        expected_loss = instance.loss_fn(logits, y).item()
        pred = logits.argmax(1) if task == "multiclass" else logits >= 0
        expected_acc = 100 * (pred == y).sum().item() / len(y)
    observations = []
    handle = target.register_forward_hook(lambda module, args, out: observations.append(
        (torch.is_grad_enabled(), out.requires_grad, all(not m.training for m in module.modules()))))
    try:
        loss, acc = instance._validate(val)
    finally:
        handle.remove()
    assert loss == pytest.approx(expected_loss, rel=1e-6)
    assert acc == expected_acc
    assert observations == [(False, False, True)] * 3
    assert [m.training for m in target.modules()] == modes
    assert_state(target.state_dict(), state)
    assert_state(instance.model.state_dict(), live)
    assert_state(instance.optimizer.state_dict(), optimizer)
    assert instance.history == history
    assert counts == (instance.update_count, instance.attempt_count, instance.ema_update_count)


@pytest.mark.parametrize("ema", [None, .7])
@pytest.mark.parametrize("problem", ["label", "shape", "nonfinite", "exception", "few", "many"])
def test_validation_failure_restores_all_modes_without_fallback(ema, problem):
    instance = make(ema_decay=ema)
    target = instance.evaluation_model
    target.train()
    target[0].eval()
    modes = [m.training for m in target.modules()]
    before = copy.deepcopy(target.state_dict())
    batch = (torch.ones(2, 2), torch.tensor([0, 1]))
    expected = (ValueError, FloatingPointError, RuntimeError)
    if problem == "label":
        batch = (batch[0], torch.tensor([-100, 1]))
    elif problem == "shape":
        batch = (batch[0], torch.zeros(2, 1, dtype=torch.long))
    elif problem == "nonfinite":
        batch = (torch.full((2, 2), float("inf")), batch[1])
    elif problem == "exception":
        def fail(*args):
            raise RuntimeError("validation forward failed")
        target.register_forward_pre_hook(fail)
    class Sized:
        def __len__(self):
            return 1
        def __iter__(self):
            return iter([] if problem == "few" else [batch, batch] if problem == "many" else [batch])
    if ema is not None:
        instance.model.register_forward_pre_hook(lambda *args: (_ for _ in ()).throw(AssertionError("no live fallback")))
    with pytest.raises(expected):
        instance._validate(Sized())
    assert [m.training for m in target.modules()] == modes
    assert_state(target.state_dict(), before)
    assert instance.update_count == instance.ema_update_count == 0


@pytest.mark.parametrize("invalid", [[], None, 5, iter([])])
def test_invalid_val_loader_rejected_before_training(invalid):
    if invalid is None:
        # A valid sized iterable whose batches are empty.
        invalid = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
    instance = make()
    with pytest.raises(ValueError, match="val_loader"):
        instance.fit(data(size=4), invalid, epochs=1)
    assert instance.update_count == 0 and not instance.history["train_loss"]


@pytest.mark.parametrize("root_training", [True, False])
def test_frozen_eval_child_survives_training_and_validation(root_training):
    instance = make()
    instance.model.train(root_training)
    instance.model[1].train()
    instance.model[0].eval()
    instance.model[0].requires_grad_(False)
    buffers = copy.deepcopy(instance.model[0].state_dict())
    seen = []
    instance.model.register_forward_pre_hook(lambda model, args: seen.append([m.training for m in model.modules()]))
    history = instance.fit(data(size=4), data(), epochs=2)
    assert len(history["val_loss"]) == 2
    assert instance.model.training and not instance.model[0].training and instance.model[1].training
    assert_state(instance.model[0].state_dict(), buffers)
    assert seen[:2] == [[True, False, True]] * 2
    assert seen[2:5] == [[False, False, False]] * 3


@pytest.mark.parametrize("task", ["multiclass", "binary"])
def test_full_fit_validation_uses_ema_for_both_loss_and_accuracy(task):
    instance = make(task=task, ema_decay=1.)
    with torch.no_grad():
        instance.ema.ema_model[1].weight.zero_()
        instance.ema.ema_model[1].bias.copy_(torch.tensor([8., -8.]) if task == "multiclass" else torch.tensor([-8.]))
        instance.model[1].weight.zero_()
        instance.model[1].bias.copy_(torch.tensor([-8., 8.]) if task == "multiclass" else torch.tensor([8.]))
    val = data(task=task)
    result = instance.fit(data(task=task, size=4), val, epochs=2)
    shadow_loss, shadow_acc = instance._validate(val)
    assert result["val_loss"] == pytest.approx([shadow_loss] * 2)
    assert result["val_acc"] == [shadow_acc] * 2
    assert shadow_acc == 60.
    assert instance.ema_update_count == instance.update_count == 4
