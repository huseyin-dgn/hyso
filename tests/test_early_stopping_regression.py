"""R02: real parameter/buffer values, snapshot storage, restore and patience."""

import pytest
import torch
from torch import nn

from hyso.callbacks.callbacks import EarlyStopping
from regression_guards import (
    R02StorageAlias, R02SnapshotMutation, R02RestoreMismatch, expect_state, expect_value,
)


def snapshot_case():
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    callback = EarlyStopping(patience=2, delta=0.25, verbose=False)
    callback(1.0, model)
    best = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    return model, callback, best


def change_live_state(model):
    with torch.no_grad():
        for tensor in model.state_dict().values():
            tensor.add_(3)


def test_r02_snapshot_initial_values_and_current_alias():
    model, callback, best = snapshot_case()
    assert callback.best_loss == 1.0 and callback.best_state is not None
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(callback.best_state[key], best[key])
        assert callback.best_state[key].data_ptr() != tensor.data_ptr()
        assert callback.best_state[key].dtype == tensor.dtype
        assert callback.best_state[key].device == tensor.device
    change_live_state(model)
    assert all(torch.equal(callback.best_state[key], best[key]) for key in best)


def test_r02_best_snapshot_storage_is_independent():
    model, callback, _ = snapshot_case()
    assert callback.best_state is not None
    assert list(callback.best_state) == list(model.state_dict())
    aliases = [key for key, tensor in model.state_dict().items()
               if callback.best_state[key].data_ptr() == tensor.data_ptr()]
    expect_value(aliases, [], R02StorageAlias, "R02: live snapshot storage aliases")


def test_r02_best_snapshot_values_survive_later_mutations():
    model, callback, best = snapshot_case()
    change_live_state(model)
    assert callback.best_state is not None
    assert all(torch.equal(value, best[key] + 3) for key, value in model.state_dict().items())
    expect_state(callback.best_state, best, R02SnapshotMutation, "R02: snapshot values changed")


def test_r02_restore_recovers_best_parameters_and_bn_buffers():
    model, callback, best = snapshot_case()
    assert {"1.running_mean", "1.running_var", "1.num_batches_tracked"} <= set(best)
    change_live_state(model)
    callback(1.5, model)
    callback.restore_weights(model)
    expect_state(model.state_dict(), best, R02RestoreMismatch, "R02: best parameters/BN buffers not restored")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_r02_snapshot_survives_real_optimizer_and_bn_updates(dtype):
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)).to(dtype=dtype)
    callback = EarlyStopping(verbose=False)
    callback(1.0, model)
    best = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    metadata = dict(callback.best_state._metadata)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(torch.tensor([[1., 2.], [3., -1.]], dtype=dtype)).square().mean().backward()
    optimizer.step()
    assert any(not torch.equal(tensor, best[key]) for key, tensor in model.state_dict().items())
    assert model[1].num_batches_tracked.item() == best["1.num_batches_tracked"].item() + 1
    expect_state(callback.best_state, best, R02SnapshotMutation, "snapshot after real update")
    callback.restore_weights(model)
    expect_state(model.state_dict(), best, R02RestoreMismatch, "restore after real update")
    assert callback.best_state._metadata == metadata == model.state_dict()._metadata


def test_r02_patience_delta_and_counter_reset():
    model = nn.Linear(2, 2)
    callback = EarlyStopping(patience=2, delta=0.25, verbose=False)
    callback(1.0, model)
    callback(0.75, model)  # equality at the delta boundary is not improvement
    assert callback.best_loss == 1.0 and callback.counter == 1
    callback(0.5, model)
    assert callback.best_loss == 0.5 and callback.counter == 0
    callback(0.4, model)
    assert callback.counter == 1 and not callback.early_stop
    callback(0.3, model)
    assert callback.counter == 2 and callback.early_stop
