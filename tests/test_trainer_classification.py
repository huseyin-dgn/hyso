"""Real forward/backward and strict B1 classification boundary checks."""

import copy
import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hyso.models.modular import ModularCNN
from hyso.training import Trainer


def loader(x=None, y=None, batch_size=2):
    if x is None:
        x = torch.tensor([[1., -1.], [2., 1.], [-1., 1.], [0., 2.], [1., 3.]])
    if y is None:
        y = torch.tensor([0, 1, 1, 0, 1])
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def trainer(model=None, **kwargs):
    model = nn.Linear(2, 2) if model is None else model
    options = dict(optimizer=torch.optim.SGD(model.parameters(), lr=.1),
                   loss_fn=nn.CrossEntropyLoss())
    options.update(kwargs)
    return Trainer(model, **options)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["modular", "sequential", "linear"])
@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW])
def test_real_multiclass_updates_match_native_loop(dtype, kind, optimizer_type):
    if kind == "modular":
        model = ModularCNN(input_channels=1, num_classes=2, widths=(3,),
                           repeats=(1,), strides=(1,), norm=None, attention="eca")
        x = torch.randn(5, 1, 5, 7, dtype=dtype)
    else:
        model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2)) if kind == "sequential" else nn.Linear(2, 2)
        x = torch.randn(5, 2, dtype=dtype)
    model.to(dtype=dtype)
    reference = copy.deepcopy(model)
    before = [p.detach().clone() for p in model.parameters()]
    data = loader(x=x)
    optimizer = optimizer_type(model.parameters(), lr=.03)
    reference_optimizer = optimizer_type(reference.parameters(), lr=.03)
    loss_fn = nn.CrossEntropyLoss()
    expected_loss, expected_acc = [], []
    for _ in range(3):
        summed, correct = 0., 0
        for inputs, targets in data:
            reference_optimizer.zero_grad()
            logits = reference(inputs)
            loss = loss_fn(logits, targets)
            summed += loss.item() * targets.numel()
            correct += (logits.argmax(1) == targets).sum().item()
            loss.backward()
            reference_optimizer.step()
        expected_loss.append(summed / 5)
        expected_acc.append(100 * correct / 5)
    instance = trainer(model, optimizer=optimizer, loss_fn=loss_fn)
    with torch.no_grad():
        history = instance.fit(data, epochs=3)
    assert history["train_loss"] == pytest.approx(expected_loss)
    assert history["train_acc"] == expected_acc
    assert history["val_loss"] == history["val_acc"] == [None] * 3
    assert all(len(values) == 3 for values in history.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.dtype == dtype and actual.device.type == "cpu"
    assert instance.fit(data, epochs=3) == history  # completed total budget: no restart
    with pytest.raises(RuntimeError, match="fresh Trainer"):
        instance.fit(data, epochs=4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("column_logits", [False, True])
@pytest.mark.parametrize("column_targets", [False, True])
def test_binary_shape_normalization_and_native_updates(dtype, column_logits, column_targets):
    class Binary(nn.Linear):
        def forward(self, inputs):
            result = super().forward(inputs)
            return result if column_logits else result.squeeze(1)

    model = Binary(2, 1).to(dtype)
    reference = copy.deepcopy(model)
    x = torch.randn(5, 2, dtype=dtype)
    y = torch.tensor([0., 1., 1., 0., 1.], dtype=torch.float32)
    data = loader(x, y[:, None] if column_targets else y)
    opt = torch.optim.Adam(model.parameters(), lr=.1)
    ref_opt = torch.optim.Adam(reference.parameters(), lr=.1)
    expected_loss, expected_acc = [], []
    for _ in range(2):
        summed, correct = 0., 0
        for inputs, targets in data:
            ref_opt.zero_grad()
            logits = reference(inputs).reshape(-1)
            targets = targets.reshape(-1).to(dtype)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
            summed += loss.item() * len(inputs)
            correct += ((logits >= 0) == targets.bool()).sum().item()
            loss.backward()
            ref_opt.step()
        expected_loss.append(summed / 5)
        expected_acc.append(correct * 20.)
    history = trainer(model, optimizer=opt, loss_fn=nn.BCEWithLogitsLoss(), task="binary").fit(data, epochs=2)
    assert history["train_loss"] == pytest.approx(expected_loss)
    assert history["train_acc"] == expected_acc
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("task", ["multiclass", "binary"])
@pytest.mark.parametrize("sizes", [[2, 2, 1], [1, 3, 1], [4, 1]])
def test_sample_weighted_metrics_known_logits(task, sizes):
    # The final sample is deliberately wrong with a large loss. Batch-mean
    # averaging would give both a different loss and a different accuracy.
    x = torch.tensor([[3., -2.], [-1., 2.], [1., -2.], [-1., 2.], [8., -8.]])
    y = torch.tensor([0, 1, 0, 1, 1])
    model = nn.Linear(2, 2 if task == "multiclass" else 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2) if task == "multiclass" else torch.tensor([[-1., 1.]]))
    targets = y if task == "multiclass" else y.float()
    criterion = nn.CrossEntropyLoss() if task == "multiclass" else nn.BCEWithLogitsLoss()
    logits = model(x)
    expected_loss = criterion(logits if task == "multiclass" else logits.flatten(), targets).item()
    batches = list(zip(x.split(sizes), targets.split(sizes)))
    result = trainer(model, optimizer=torch.optim.SGD(model.parameters(), lr=0),
                     loss_fn=criterion, task=task).fit(batches, epochs=1)
    assert result["train_loss"] == pytest.approx([expected_loss])
    assert result["train_acc"] == [80.]


def test_forward_only_train_mode_and_unused_parameters():
    class Model(nn.Linear):
        def fit(self, *args, **kwargs):
            raise AssertionError("model.fit must never be called")

    model = Model(2, 2).eval()
    model.register_parameter("unused", nn.Parameter(torch.ones(2)))
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    instance = trainer(model, optimizer=optimizer)
    before = model.weight.detach().clone()
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    result = instance.fit(loader(), epochs=1)
    assert model.training and not torch.equal(before, model.weight)
    assert model.unused.grad is None and math.isfinite(result["train_loss"][0])


@pytest.mark.parametrize("kwargs,error", [
    ({"task": "regression"}, ValueError),
    ({"optimizer": None}, TypeError),
    ({"loss_fn": nn.MSELoss()}, TypeError),
    ({"loss_fn": nn.BCEWithLogitsLoss()}, TypeError),
    ({"loss_fn": nn.CrossEntropyLoss(reduction="sum")}, ValueError),
    ({"loss_fn": nn.CrossEntropyLoss(reduction="none")}, ValueError),
    ({"loss_fn": nn.CrossEntropyLoss(weight=torch.ones(2))}, ValueError),
    ({"loss_fn": nn.CrossEntropyLoss(label_smoothing=.1)}, ValueError),
    ({"task": "binary", "loss_fn": nn.BCEWithLogitsLoss(pos_weight=torch.ones(1))}, ValueError),
    ({"task": "binary", "loss_fn": nn.BCEWithLogitsLoss(weight=torch.ones(1))}, ValueError),
    ({"task": "binary", "loss_fn": nn.BCEWithLogitsLoss(reduction="sum")}, ValueError),
    ({"device": "not-a-device"}, ValueError),
    ({"device": "meta"}, NotImplementedError),
])
def test_invalid_configuration(kwargs, error):
    with pytest.raises(error):
        trainer(**kwargs)


def test_cuda_unavailable_is_explicit():
    assert not torch.cuda.is_available()  # real CPU-hidden conftest profile
    with pytest.raises(ValueError, match="CUDA.*unavailable"):
        trainer(device="cuda")


@pytest.mark.parametrize("kind", ["foreign", "missing", "duplicate", "frozen", "buffer", "dtype", "changed"])
def test_model_optimizer_ownership_and_device(kind):
    model = nn.Linear(2, 2)
    if kind == "buffer":
        model.register_buffer("wrong_device", torch.empty(1, device="meta"))
    if kind == "dtype":
        model.half()
    if kind == "frozen":
        model.requires_grad_(False)
    parameters = list(model.parameters())
    if kind == "foreign":
        parameters.append(nn.Parameter(torch.ones(1)))
    if kind == "missing":
        parameters = parameters[:1]
    optimizer = torch.optim.SGD(parameters, lr=.1)
    if kind == "duplicate":
        optimizer.param_groups[0]["params"].append(parameters[0])
    if kind == "changed":
        instance = trainer(model, optimizer=optimizer)
        model.weight = nn.Parameter(model.weight.detach().clone())
        with pytest.raises(ValueError, match="foreign"):
            instance.fit(loader())
    else:
        with pytest.raises(ValueError):
            trainer(model, optimizer=optimizer)


def test_frozen_parameters_may_be_owned_or_omitted():
    for include in (False, True):
        model = nn.Linear(2, 2)
        model.bias.requires_grad_(False)
        optimizer = torch.optim.SGD(model.parameters() if include else [model.weight], lr=.1)
        assert len(trainer(model, optimizer=optimizer).fit(loader(), epochs=1)["train_loss"]) == 1


@pytest.mark.parametrize("targets,error", [
    (torch.tensor([0., 1.]), TypeError),
    (torch.tensor([0, 1], dtype=torch.int32), TypeError),
    (torch.tensor([[0], [1]]), ValueError),
    (torch.tensor([[1., 0.], [0., 1.]]), ValueError),
    (torch.tensor([-100, 1]), ValueError),
    (torch.tensor([-1, 1]), ValueError),
    (torch.tensor([0, 2]), ValueError),
])
def test_invalid_multiclass_labels(targets, error):
    with pytest.raises(error):
        trainer().fit([(torch.ones(2, 2), targets)], epochs=1)


@pytest.mark.parametrize("targets,error", [
    (torch.tensor([0, 1]), TypeError),
    (torch.ones(2, 2), ValueError),
    (torch.tensor([0., .5]), ValueError),
    (torch.tensor([0., 2.]), ValueError),
    (torch.tensor([0., float("nan")]), ValueError),
])
def test_invalid_binary_labels(targets, error):
    with pytest.raises(error):
        trainer(nn.Linear(2, 1), task="binary", loss_fn=nn.BCEWithLogitsLoss()).fit(
            [(torch.ones(2, 2), targets)], epochs=1)


@pytest.mark.parametrize("task,shape", [("multiclass", (2,)), ("multiclass", (2, 1)),
                                       ("multiclass", (1, 2)), ("multiclass", (2, 2, 1)),
                                       ("binary", (2, 2)), ("binary", (2, 1, 1))])
def test_invalid_output_shapes(task, shape):
    class Wrong(nn.Linear):
        def forward(self, inputs):
            return self.weight.sum().expand(shape)
    loss = nn.CrossEntropyLoss() if task == "multiclass" else nn.BCEWithLogitsLoss()
    targets = torch.zeros(2, dtype=torch.long if task == "multiclass" else torch.float32)
    with pytest.raises(ValueError, match="logits"):
        trainer(Wrong(2, 2), task=task, loss_fn=loss).fit([(torch.ones(2, 2), targets)])


@pytest.mark.parametrize("batch,error", [
    ({"inputs": torch.ones(2, 2)}, ValueError),
    ((torch.ones(2, 2),), ValueError),
    (([1, 2], torch.zeros(2, dtype=torch.long)), TypeError),
    ((torch.ones(2, 2), [0, 1]), TypeError),
    ((torch.tensor(1.), torch.tensor(0)), ValueError),
    ((torch.empty(0, 2), torch.empty(0, dtype=torch.long)), ValueError),
    ((torch.ones(2, 2, dtype=torch.int64), torch.zeros(2, dtype=torch.long)), TypeError),
])
def test_invalid_batches(batch, error):
    with pytest.raises(error):
        trainer().fit([batch])


@pytest.mark.parametrize("data", [None, 42, [], iter([]),
                                    DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long))),
                                    DataLoader(TensorDataset(torch.ones(1, 2), torch.zeros(1, dtype=torch.long)), batch_size=2, drop_last=True)])
def test_empty_or_invalid_loader(data):
    with pytest.raises(ValueError, match="train_loader"):
        trainer().fit(data)


@pytest.mark.parametrize("actual", [0, 2])
def test_loader_length_mismatch(actual):
    class Sized:
        def __len__(self):
            return 1
        def __iter__(self):
            return iter([(torch.ones(2, 2), torch.zeros(2, dtype=torch.long))] * actual)
    with pytest.raises(ValueError, match="train_loader"):
        trainer().fit(Sized())


@pytest.mark.parametrize("epochs", [True, False, 0, -1, 1.5, "2", None])
def test_invalid_epoch_budget(epochs):
    with pytest.raises(ValueError, match="epochs"):
        trainer().fit(loader(), epochs=epochs)


@pytest.mark.parametrize("kind", ["loss", "gradient", "logits", "output_type"])
def test_nonfinite_and_nontensor_outputs_fail_before_update(kind):
    class Bad(nn.Linear):
        def forward(self, inputs):
            logits = super().forward(inputs)
            if kind == "logits":
                return logits * float("inf")
            if kind == "output_type":
                return {"logits": logits}
            if kind == "loss":
                # Finite individual logits, overflowing cross-entropy difference.
                return logits * 0 + torch.tensor([[-3e38, 3e38]])
            return logits
    model = Bad(2, 2)
    if kind == "gradient":
        model.weight.register_hook(lambda grad: grad * float("nan"))
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    before = copy.deepcopy(model.state_dict())
    instance = trainer(model, optimizer=optimizer)
    with pytest.raises(TypeError if kind == "output_type" else FloatingPointError):
        instance.fit([(torch.ones(2, 2), torch.zeros(2, dtype=torch.long))], epochs=1)
    assert optimizer.state == {}
    assert instance.history["train_loss"] == []
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    with pytest.raises(RuntimeError, match="fresh Trainer"):
        instance.fit(loader(), epochs=1)


@pytest.mark.parametrize("option", ["amp", "accum_steps", "grad_clip", "scheduler", "ema_decay",
                                    "early_stopping", "callbacks", "model_config"])
def test_future_constructor_options_are_not_silent(option):
    with pytest.raises(TypeError, match=option):
        trainer(**{option: True})


@pytest.mark.parametrize("option", ["resume_from", "save_last_to"])
def test_future_fit_options_are_not_silent(option):
    with pytest.raises(TypeError, match=option):
        trainer().fit(loader(), **{option: "unused.pth"})


def test_validation_runs_without_additional_optimizer_updates():
    instance = trainer()
    before = copy.deepcopy(instance.model.state_dict())
    history = instance.fit(loader(), val_loader=loader(), epochs=1)
    assert instance.update_count == instance.attempt_count == len(loader())
    assert len(history["train_loss"]) == len(history["val_loss"]) == 1
    with torch.no_grad():
        data = loader().dataset.tensors
        logits = instance.model(data[0])
        assert history["val_loss"] == pytest.approx([nn.functional.cross_entropy(logits, data[1]).item()])
        assert history["val_acc"] == [100 * (logits.argmax(1) == data[1]).sum().item() / len(data[1])]
    assert any(not torch.equal(value, before[name]) for name, value in instance.model.state_dict().items())


def test_nondefault_ignore_index_only_rejects_actual_masked_targets():
    instance = trainer(loss_fn=nn.CrossEntropyLoss(ignore_index=99))
    assert len(instance.fit(loader(), epochs=1)["train_loss"]) == 1
    with pytest.raises(ValueError, match="ignored"):
        trainer(loss_fn=nn.CrossEntropyLoss(ignore_index=1)).fit(loader())


def test_invalid_input_feature_shape_propagates_without_parameter_update():
    instance = trainer()
    before = copy.deepcopy(instance.model.state_dict())
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        instance.fit([(torch.ones(2, 3), torch.zeros(2, dtype=torch.long))])
    for name, value in instance.model.state_dict().items():
        assert torch.equal(value, before[name])


def test_model_identity_is_preserved_without_automatic_moves():
    class NoMove(nn.Linear):
        def to(self, *args, **kwargs):
            raise AssertionError("Trainer must not move the model")

    model = NoMove(2, 2)
    identities = [id(p) for p in model.parameters()]
    instance = trainer(model)
    assert instance.model is model
    instance.fit(loader(), epochs=1)
    assert identities == [id(p) for p in model.parameters()]


def test_closure_optimizer_is_explicitly_unsupported():
    model = nn.Linear(2, 2)
    with pytest.raises(NotImplementedError, match="closure"):
        trainer(model, optimizer=torch.optim.LBFGS(model.parameters()))


def test_custom_loss_forward_cannot_bypass_mean_loss_contract():
    class CustomLoss(nn.CrossEntropyLoss):
        def forward(self, logits, targets):
            return super().forward(logits, targets) * 2

    with pytest.raises(TypeError, match="standard CrossEntropyLoss"):
        trainer(loss_fn=CustomLoss())


def test_model_must_be_module():
    with pytest.raises(TypeError, match="nn.Module"):
        Trainer(object(), optimizer=None, loss_fn=nn.CrossEntropyLoss())
