"""Native EMA/EarlyStopping integration and observer/Plateau lifecycle contracts."""
import copy
import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hyso.training import Trainer
from hyso.models.WithSE import EMA
from hyso.callbacks.callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback


def data():
    return DataLoader(TensorDataset(torch.randn(6, 2), torch.tensor([0, 1, 1, 0, 0, 1])), batch_size=2)


def build(**kwargs):
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2))
    return Trainer(model, optimizer=torch.optim.SGD(model.parameters(), lr=.1), loss_fn=nn.CrossEntropyLoss(), **kwargs)


@pytest.mark.parametrize("decay", [0., .8, 1.])
@pytest.mark.parametrize("accum", [1, 2, 9])
def test_native_ema_reference_per_applied_update_including_bn_buffers(decay, accum):
    instance = build(ema_decay=decay, accum_steps=accum)
    assert type(instance.ema) is EMA
    assert not instance.ema.ema_model.training
    assert all(not p.requires_grad for p in instance.ema.ema_model.parameters())
    reference = EMA(instance.model, decay=decay)
    shadows_before = [p.clone() for p in instance.ema.ema_model.parameters()]
    for source, target in zip(instance.model.parameters(), instance.ema.ema_model.parameters()):
        assert source is not target and source.data_ptr() != target.data_ptr()
    # Observer hook delegates real updates to the existing native EMA algorithm.
    handle = instance.optimizer.register_step_post_hook(lambda *args: reference.update(instance.model))
    try:
        instance.fit(data(), epochs=2)
    finally:
        handle.remove()
    assert instance.ema_update_count == instance.update_count == 2 * math.ceil(3 / accum)
    for name, tensor in instance.ema.ema_model.state_dict().items():
        torch.testing.assert_close(tensor, reference.ema_model.state_dict()[name], rtol=0, atol=0)
    for name, buffer in instance.ema.ema_model.named_buffers():
        assert torch.equal(buffer, dict(instance.model.named_buffers())[name])
    assert instance.ema.ema_model[0].num_batches_tracked.item() == 6
    assert any(not torch.equal(p, before) for p, before in zip(instance.model.parameters(), shadows_before))
    if decay == 1:
        assert all(torch.equal(p, before) for p, before in zip(instance.ema.ema_model.parameters(), shadows_before))


@pytest.mark.parametrize("value,error", [(True, TypeError), ("0.9", TypeError),
                                        (float("nan"), ValueError), (float("inf"), ValueError),
                                        (-.1, ValueError), (1.1, ValueError)])
def test_invalid_ema_decay(value, error):
    with pytest.raises(error, match="ema_decay"):
        build(ema_decay=value)


@pytest.mark.parametrize("mutation", ["name", "shape", "dtype", "alias", "grad"])
def test_ema_schema_mismatch_is_visible_before_update(mutation):
    instance = build(ema_decay=.9)
    shadow = instance.ema.ema_model
    if mutation == "name":
        shadow.register_buffer("extra", torch.zeros(1))
    elif mutation == "shape":
        shadow[1].weight = nn.Parameter(torch.zeros(3, 2), requires_grad=False)
    elif mutation == "dtype":
        shadow.double()
    elif mutation == "alias":
        shadow[1].weight = instance.model[1].weight
    else:
        shadow[1].weight.requires_grad_(True)
    with pytest.raises(ValueError, match="EMA"):
        instance.fit(data(), epochs=1)
    assert instance.attempt_count == 0


def worsening(*, ema=None, restore=True, patience=1, delta=0., callbacks=()):
    # BN outputs zero for this constant training batch. Classifier bias learns
    # class 1, while validation asks for class 0: loss increases monotonically.
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.zero_()
    es = EarlyStopping(patience=patience, delta=delta, verbose=False, restore_best_weights=restore)
    instance = Trainer(model, optimizer=torch.optim.SGD(model.parameters(), lr=.5), loss_fn=nn.CrossEntropyLoss(),
                       ema_decay=ema, early_stopping=es, callbacks=callbacks)
    train = [(torch.ones(2, 2), torch.ones(2, dtype=torch.long))]
    val = [(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long))]
    return instance, train, val


@pytest.mark.parametrize("ema", [None, 0., .8])
@pytest.mark.parametrize("restore", [False, True])
def test_early_stop_best_snapshot_and_correct_target_finalization(ema, restore):
    epochs, ended = [], []
    instance, train, val = worsening(ema=ema, restore=restore)
    def observe(event, logs):
        if event == "epoch_end":
            epochs.append(copy.deepcopy(instance.evaluation_model.state_dict()))
        if event == "train_end":
            ended.append(copy.deepcopy(instance.evaluation_model.state_dict()))
    instance.callbacks = (observe,)
    history = instance.fit(train, val, epochs=8)
    assert len(history["train_loss"]) == 2
    assert history["val_loss"][0] < history["val_loss"][1]
    es = instance.early_stopping
    assert es.early_stop and es.counter == 1 and es.best_loss == history["val_loss"][0]
    assert len(epochs) == 2 and len(ended) == 1
    expected = epochs[0] if restore else epochs[-1]
    for name, tensor in es.best_state.items():
        assert torch.equal(tensor, epochs[0][name])
        assert tensor.data_ptr() != instance.evaluation_model.state_dict()[name].data_ptr()
        assert torch.equal(ended[0][name], expected[name])
    assert es.best_state["0.num_batches_tracked"].item() == 1
    if ema is not None:
        assert instance.model[0].num_batches_tracked.item() == 2
        assert instance.evaluation_model[0].num_batches_tracked.item() == (1 if restore else 2)
    assert instance.update_count == 2
    with pytest.raises(RuntimeError, match="terminal"):
        instance.fit(train, val, epochs=8)
    saved = copy.deepcopy(es.best_state)
    with torch.no_grad():
        next(instance.evaluation_model.parameters()).add_(10)
    for name, tensor in es.best_state.items():
        assert torch.equal(tensor, saved[name])


def test_patience_delta_and_improvement_match_existing_early_stopping():
    model = nn.Linear(2, 2)
    train, val = data(), data()
    es = EarlyStopping(patience=2, delta=.01, verbose=False, restore_best_weights=False)
    reference = EarlyStopping(patience=2, delta=.01, verbose=False, restore_best_weights=False)
    opt = torch.optim.SGD(model.parameters(), lr=.05)
    instance = Trainer(model, optimizer=opt, loss_fn=nn.CrossEntropyLoss(), early_stopping=es)
    expected = []
    def observe(event, logs):
        if event == "epoch_end":
            reference(logs["val_loss"], model)
            expected.append((reference.counter, reference.best_loss, reference.early_stop))
            assert (es.counter, es.best_loss, es.early_stop) == expected[-1]
    instance.callbacks = (observe,)
    history = instance.fit(train, val, epochs=5)
    assert len(expected) == len(history["train_loss"])
    assert expected[0][0] == 0


@pytest.mark.parametrize("kwargs", [{"patience": 0}, {"patience": True}, {"delta": -1},
                                     {"delta": float("nan")}, {"restore_best_weights": 1}])
def test_invalid_early_stopping_configuration(kwargs):
    with pytest.raises((ValueError, TypeError)):
        build(early_stopping=EarlyStopping(**kwargs))


@pytest.mark.parametrize("feature", ["early", "plateau"])
def test_validation_required_before_training(feature):
    instance = build(early_stopping=EarlyStopping(verbose=False) if feature == "early" else None)
    if feature == "plateau":
        instance.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(instance.optimizer)
    before = copy.deepcopy(instance.model.state_dict())
    with pytest.raises(ValueError, match="val_loader"):
        instance.fit(data(), epochs=2)
    assert instance.update_count == 0
    for name, tensor in instance.model.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_plateau_real_loss_lr_sequence_and_event_order():
    instance, train, val = worsening(restore=False, patience=5)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(instance.optimizer, factor=.5, patience=0)
    instance.scheduler = plateau
    reference_opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=.5)
    reference = torch.optim.lr_scheduler.ReduceLROnPlateau(reference_opt, factor=.5, patience=0)
    events = []
    def observer(event, logs):
        events.append(event)
        if event == "epoch_end":
            reference.step(logs["val_loss"])
            assert plateau.last_epoch == logs["epoch"]
            assert instance.early_stopping.best_state is not None
            assert instance.optimizer.param_groups[0]["lr"] == reference_opt.param_groups[0]["lr"]
            assert logs["learning_rate"] == [reference_opt.param_groups[0]["lr"]]
    instance.callbacks = (observer,)
    history = instance.fit(train, val, epochs=4)
    assert events == ["train_begin"] + ["epoch_end"] * 4 + ["train_end"]
    assert plateau.last_epoch == 4 and reference.last_epoch == 4
    assert plateau.best == min(history["val_loss"])


def test_observer_plain_snapshots_and_per_observer_isolation():
    observed, retained = [], []
    def mutate(event, logs):
        retained.append(logs)
        logs["train_loss"] = "changed"
        logs["learning_rate"].append(999)
    def check(event, logs):
        observed.append((event, copy.deepcopy(logs)))
        assert logs["train_loss"] != "changed" and 999 not in logs["learning_rate"]
        def plain(value):
            assert not isinstance(value, torch.Tensor)
            if isinstance(value, dict):
                for item in value.values():
                    plain(item)
            elif isinstance(value, list):
                for item in value:
                    plain(item)
            else:
                assert value is None or type(value) in (int, float, bool)
        plain(logs)
    instance = build(callbacks=(mutate, check))
    history = instance.fit(data(), data(), epochs=2)
    assert [event for event, _ in observed] == ["train_begin", "epoch_end", "epoch_end", "train_end"]
    assert observed[0][1]["epoch"] == 0 and observed[0][1]["train_loss"] is None
    assert observed[-1][1]["epoch"] == 2
    assert observed[-1][1]["applied_updates"] == instance.update_count == 6
    retained[-1]["val_loss"] = -999
    assert instance.history == history and history["val_loss"][-1] != -999


@pytest.mark.parametrize("failure_event", ["train_begin", "epoch_end", "train_end"])
def test_observer_failure_propagates_without_retry_or_later_events(failure_event):
    events = []
    def fail(event, logs):
        events.append(event)
        if event == failure_event:
            raise RuntimeError("observer failure")
    instance = build(callbacks=(fail,))
    with pytest.raises(RuntimeError, match="observer failure"):
        instance.fit(data(), data(), epochs=1)
    expected = ["train_begin", "epoch_end", "train_end"]
    assert events == expected[:expected.index(failure_event) + 1]
    if failure_event == "train_begin":
        assert instance.update_count == 0
    with pytest.raises(RuntimeError, match="fresh Trainer"):
        instance.fit(data(), data(), epochs=1)


@pytest.mark.parametrize("callback", [True, ModelCheckpoint(), EarlyStopping(), LRSchedulerCallback(None), lambda: None])
def test_legacy_or_invalid_observers_rejected(callback):
    with pytest.raises(TypeError, match="callbacks"):
        build(callbacks=(callback,))


def test_nonfresh_early_stopping_not_treated_as_resume():
    es = EarlyStopping(verbose=False)
    es.counter = 1
    instance = build(early_stopping=es)
    with pytest.raises(ValueError, match="fresh state"):
        instance.fit(data(), data(), epochs=1)
    assert instance.update_count == 0


def test_epoch_observer_sees_last_state_before_best_restore_and_failure_skips_finalization():
    instance, train, val = worsening()
    snapshots = []
    def observer(event, logs):
        if event == "epoch_end":
            snapshots.append(copy.deepcopy(instance.model.state_dict()))
            if logs["early_stop"]:
                raise RuntimeError("failure at coherent boundary")
    instance.callbacks = (observer,)
    with pytest.raises(RuntimeError, match="coherent boundary"):
        instance.fit(train, val, epochs=5)
    assert len(snapshots) == 2
    assert instance.model[0].num_batches_tracked.item() == 2
    assert instance.early_stopping.best_state["0.num_batches_tracked"].item() == 1


@pytest.mark.parametrize("delta", [0., 10.])
def test_real_improvements_and_delta_patience(delta):
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=.5)
    es = EarlyStopping(patience=2, delta=delta, verbose=False, restore_best_weights=False)
    instance = Trainer(model, optimizer=optimizer, loss_fn=nn.CrossEntropyLoss(), early_stopping=es)
    batch = [(torch.eye(2), torch.tensor([0, 1]))]
    history = instance.fit(batch, batch, epochs=5)
    assert all(a > b for a, b in zip(history["val_loss"], history["val_loss"][1:]))
    if delta == 0:
        assert len(history["val_loss"]) == 5 and es.counter == 0 and not es.early_stop
        assert es.best_loss == history["val_loss"][-1]
    else:
        assert len(history["val_loss"]) == 3 and es.counter == 2 and es.early_stop
        assert es.best_loss == history["val_loss"][0]


def test_plateau_exception_propagates_once_after_validation(monkeypatch):
    instance = build()
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(instance.optimizer)
    instance.scheduler = schedule
    real = schedule.step
    calls, events = [], []
    def fail(value):
        calls.append(value)
        real(value)
        raise RuntimeError("Plateau step failed")
    monkeypatch.setattr(schedule, "step", fail)
    instance.callbacks = (lambda event, logs: events.append(event),)
    with pytest.raises(RuntimeError, match="Plateau step failed"):
        instance.fit(data(), data(), epochs=2)
    assert len(calls) == 1 and math.isfinite(calls[0])
    assert schedule.last_epoch == 1 and events == ["train_begin"]
    assert instance.model.training and instance.model[0].training


@pytest.mark.parametrize("field,value", [("mode", "max"), ("factor", float("nan")),
                                         ("patience", -1), ("eps", float("inf"))])
def test_invalid_plateau_state_is_explicit(field, value):
    instance = build()
    instance.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(instance.optimizer)
    setattr(instance.scheduler, field, value)
    with pytest.raises(ValueError, match="Plateau"):
        instance.fit(data(), data(), epochs=1)
    assert instance.update_count == 0
