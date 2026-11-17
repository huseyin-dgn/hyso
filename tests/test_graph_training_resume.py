"""Tiny CPU Graph automatic resume; two epochs, two batches per epoch."""
import copy

import pytest
import torch
from torch.utils.data import TensorDataset

from hyso import GraphModelBuilder, TrainingConfig, TrainingConfigError
from hyso.models.output import GraphTensorOutput
from hyso.training import resume
from hyso.checkpoint import TrainingCheckpointError
from test_trainer_checkpoint_restore import assert_state


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail('Graph resume attempted CUDA')
    # CPU Adam probes availability internally; answer without querying a device.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    for name in ('init', '_lazy_init', 'device_count', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def model(dtype=torch.float32, activation='relu', task='multiclass'):
    b = (GraphModelBuilder({'image': [1, 3, 3]})
         .add('stem', 'conv2d', ['image'], in_channels=1, out_channels=2)
         .add('act', activation, ['stem'])
         .add('head', 'classification_head', ['act'], in_features=2,
              num_classes=2 if task == 'multiclass' else 1).outputs(logits='head'))
    return GraphTensorOutput(b.build().to(dtype), 'logits', task=task), b.to_config()


def prepare(m, cfg, path=None, source=None, optimizer='adamw', dtype=torch.float32, task='multiclass', epochs=2, **options):
    c = TrainingConfig(epochs=epochs, batch_size=2, optimizer=optimizer, task=task,
                       optimizer_params={'momentum': .8} if optimizer == 'sgd' else None,
                       scheduler='step', scheduler_params={'step_size': 1, 'gamma': .7},
                       precision='fp64' if dtype == torch.float64 else 'fp32',
                       save_last_to=None if path is None else str(path),
                       resume_from=None if source is None else str(source), **options)
    x = torch.arange(36, dtype=dtype).reshape(4, 1, 3, 3) / 36
    y = torch.tensor([0, 1, 1, 0], dtype=torch.long if task == 'multiclass' else dtype)
    train, _ = c.make_loaders(TensorDataset(x, y))
    return c.prepare(m, train, model_config=cfg)


@pytest.mark.parametrize('optimizer', ['sgd', 'adam', 'adamw'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_two_epoch_resume_exact_state(tmp_path, monkeypatch, optimizer, dtype):
    check_continuation(tmp_path, monkeypatch, optimizer, dtype, 'multiclass')


def test_binary_resume(tmp_path, monkeypatch):
    check_continuation(tmp_path, monkeypatch, 'adam', torch.float32, 'binary')


def check_continuation(tmp_path, monkeypatch, optimizer, dtype, task):
    torch.manual_seed(51)
    m, cfg = model(dtype, task=task)
    baseline = prepare(m, cfg, optimizer=optimizer, dtype=dtype, task=task)
    expected = baseline.fit()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(51)
    m, cfg = model(dtype, task=task)
    parameter_ids = [id(p) for p in m.parameters()]
    path = tmp_path / 'last.pth'
    interrupted = prepare(m, cfg, path=path, optimizer=optimizer, dtype=dtype, task=task)
    native = resume.save_last

    def save_then_stop(*args):
        native(*args)
        raise InterruptedError('completed epoch one')

    with monkeypatch.context() as patch:
        patch.setattr(resume, 'save_last', save_then_stop)
        with pytest.raises(InterruptedError):
            interrupted.fit()
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint['format_version'] == 2
    assert checkpoint['model']['graph_contract']['configuration'] == cfg
    assert all(n.startswith('graph.') for n in checkpoint['model']['state_dict'])
    assert all(n.startswith('graph.') for group in checkpoint['optimizer']['param_names'] for n in group)
    assert parameter_ids == [id(p) for p in m.parameters()]
    saved = torch.load(path, weights_only=True)
    assert saved['format_version'] == 2 and saved['completed_epochs'] == 1
    assert all(n.startswith('graph.') for n in saved['model']['state_dict'])
    assert saved['optimizer']['param_names'] == [[n for n, _ in m.named_parameters()]]
    target, cfg = model(dtype, task=task)
    resumed = prepare(target, cfg, path=path, source=path, optimizer=optimizer, dtype=dtype, task=task)
    ids = [id(p) for p in target.parameters()]
    result = resumed.fit()
    assert result.history == expected.history
    assert result.to_dict()['completed_history_epochs'] == 2
    assert resumed.trainer.update_count == baseline.trainer.update_count == 4
    assert ids == [id(p) for p in target.parameters()]
    assert_state(target.state_dict(), baseline.trainer.model.state_dict())
    assert_state(resumed.trainer.optimizer.state_dict(), baseline.trainer.optimizer.state_dict())
    assert_state(resumed.trainer.scheduler.state_dict(), baseline.trainer.scheduler.state_dict())
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.load(path, weights_only=True)['completed_epochs'] == 2


@pytest.mark.parametrize('change', ['architecture', 'tensor', 'missing', 'optimizer', 'version'])
def test_incompatible_checkpoint_rejected_before_mutation(tmp_path, change):
    path = tmp_path / 'source.pth'
    m, cfg = model()
    prepare(m, cfg, path=path, epochs=1).fit()
    payload = torch.load(path, weights_only=True)
    if change == 'version':
        payload['format_version'] = 1
        del payload['model']['graph_contract']
        torch.save(payload, path)
    if change in ('tensor', 'missing'):
        key = next(iter(payload['model']['state_dict']))
        if change == 'tensor':
            payload['model']['state_dict'][key].add_(1)
        else:
            del payload['model']['state_dict'][key]
        torch.save(payload, path)
    target, cfg = model(activation='silu' if change == 'architecture' else 'relu')
    s = prepare(target, cfg, source=path, epochs=1, optimizer='sgd' if change == 'optimizer' else 'adamw')
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(TrainingCheckpointError):
        s.fit()
    assert_state(target.state_dict(), before)
    assert not s.trainer.optimizer.state
    assert s.trainer.update_count == 0


def test_config_and_precision_are_explicit(tmp_path):
    m, cfg = model(torch.float64)
    with pytest.raises(TrainingConfigError, match='device/dtype'):
        prepare(m, cfg, path=tmp_path/'bad.pth')
    m, cfg = model()
    cfg['outputs'] = {'other': 'head'}
    with pytest.raises(TrainingCheckpointError, match='model_config'):
        prepare(m, cfg, path=tmp_path/'bad.pth')


def test_unsupported_ema_is_rejected(tmp_path):
    m, cfg = model()
    with pytest.raises(TrainingCheckpointError, match='no EMA'):
        prepare(m, cfg, path=tmp_path/'bad.pth', ema_decay=.9)


def test_completed_resume_without_scheduler(tmp_path):
    m, cfg = model()
    path = tmp_path / 'plain.pth'
    c = TrainingConfig(epochs=1, batch_size=2, save_last_to=str(path))
    data = TensorDataset(torch.ones(2, 1, 3, 3), torch.tensor([0, 1]))
    train, _ = c.make_loaders(data)
    saved = c.prepare(m, train, model_config=cfg).fit()
    target, cfg = model()
    restored = TrainingConfig(epochs=1, batch_size=2, resume_from=str(path)).prepare(target, train, model_config=cfg)
    assert restored.fit().history == saved.history
    assert_state(target.state_dict(), m.state_dict())
