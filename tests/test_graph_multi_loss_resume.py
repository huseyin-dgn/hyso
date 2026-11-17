"""Two explicit losses use the existing v2 serializer and transactional restore."""
import copy
import pytest
import torch
from hyso import TrainingConfig
from hyso.training import NamedTargetDataset, resume
from hyso.checkpoint import TrainingCheckpointError
from test_graph_multi_loss import model, criteria, Samples
from test_graph_training_resume import cpu_only
from test_trainer_checkpoint_restore import assert_state


def dataset(binary=False, reverse=False):
    samples = Samples(binary)
    targets = {'one': samples.a, 'two': samples.b}
    if reverse: targets = dict(reversed(list(targets.items())))
    return NamedTargetDataset(samples.x, targets)


def session(graph, path=None, source=None, optimizer='sgd', binary=False, ds=None, epochs=2, weights=None, losses=None):
    config = TrainingConfig(epochs=epochs, batch_size=2, validation=True, optimizer=optimizer,
        optimizer_params={'momentum': .8} if optimizer == 'sgd' else None,
        save_last_to=None if path is None else str(path), resume_from=None if source is None else str(source))
    data = dataset(binary) if ds is None else ds
    train, val = config.make_loaders(data, data)
    return config.prepare(graph, train, val, model_config=graph.to_config(),
        loss_fn=criteria(binary) if losses is None else losses,
        loss_weights={'one': .3, 'two': 1.2} if weights is None else weights)


@pytest.mark.parametrize('optimizer,binary', [('sgd', False), ('adam', True)])
def test_multi_loss_resume_exact_continuation(tmp_path, monkeypatch, optimizer, binary):
    torch.manual_seed(91)
    baseline = session(model(binary), optimizer=optimizer, binary=binary)
    expected = baseline.fit(); rng = torch.get_rng_state().clone()
    torch.manual_seed(91)
    path = tmp_path / 'last.pth'
    interrupted = session(model(binary), path=path, optimizer=optimizer, binary=binary)
    save = resume.save_last
    def stop(*args):
        save(*args)
        raise InterruptedError()
    with monkeypatch.context() as scoped:
        scoped.setattr(resume, 'save_last', stop)
        with pytest.raises(InterruptedError): interrupted.fit()
    payload = torch.load(path, weights_only=True)
    assert payload['format_version'] == 2 and payload['completed_epochs'] == 1
    # Raw Graph state keeps its original names; GraphTensorOutput regression
    # separately verifies that existing graph. wrapper prefixes remain unchanged.
    assert list(payload['model']['state_dict']) == list(interrupted.trainer.model.state_dict())
    graph = model(binary); identities = [id(p) for p in graph.parameters()]
    continued = session(graph, path=path, source=path, optimizer=optimizer, binary=binary,
        ds=dataset(binary, reverse=True), losses=dict(reversed(list(criteria(binary).items()))))
    actual = continued.fit()
    assert actual.history == expected.history
    assert actual.history['train_acc'] == actual.history['val_acc'] == [None, None]
    assert torch.equal(torch.get_rng_state(), rng)
    assert identities == [id(p) for p in graph.parameters()]
    assert_state(graph.state_dict(), baseline.trainer.model.state_dict())
    assert_state(continued.trainer.optimizer.state_dict(), baseline.trainer.optimizer.state_dict())
    assert continued.trainer.scheduler is baseline.trainer.scheduler is None
    assert torch.load(path, weights_only=True)['completed_epochs'] == 2
    completed = session(model(binary), source=path, optimizer=optimizer, binary=binary)
    assert completed.fit().history == actual.history


@pytest.mark.parametrize('change', ['topology','output','target_name','target_mapping','loss_type','weights',
    'optimizer','optimizer_state','commitment','corrupt','budget','input','history'])
def test_multi_loss_resume_mismatch_is_transactional(tmp_path, change):
    path = tmp_path / 'last.pth'
    session(model(), path=path).fit()
    graph = model(); ds = dataset(); options = {}
    if change == 'topology': graph.nodes['node_left'].stride = (2,2)
    if change == 'output':
        config = graph.to_config(); config['outputs']['other'] = config['outputs'].pop('two')
        from hyso import GraphModelBuilder
        graph = GraphModelBuilder.from_config(config).build()
    if change == 'target_name': ds.target_names = ('one','other')
    if change == 'target_mapping': ds.tensors = (ds.tensors[0], ds.tensors[2], ds.tensors[1])
    if change == 'loss_type': options['losses'] = {'one': torch.nn.CrossEntropyLoss(), 'two': torch.nn.BCEWithLogitsLoss()}
    if change == 'weights': options['weights'] = {'one': 1.2, 'two': .3}
    if change == 'optimizer':
        payload = torch.load(path, weights_only=True); payload['optimizer'] = None; torch.save(payload,path)
    if change == 'optimizer_state':
        payload = torch.load(path, weights_only=True); payload['optimizer']['state_dict']['state'] = {}; torch.save(payload,path)
    if change == 'commitment':
        payload = torch.load(path, weights_only=True)
        payload['metrics']['generic_trainer']['progress'].pop('continuation_sha256'); torch.save(payload,path)
    if change == 'corrupt': path.write_bytes(b'broken checkpoint')
    if change == 'budget': options['epochs'] = 3
    if change == 'input': ds.tensors = (ds.tensors[0][:,:,:2,:], *ds.tensors[1:])
    if change == 'history':
        payload = torch.load(path, weights_only=True); payload['history']['train_acc'][0] = 50.; torch.save(payload,path)
    before = copy.deepcopy(graph.state_dict()); rng = torch.get_rng_state().clone(); prepared = None
    with pytest.raises((TrainingCheckpointError, ValueError, TypeError)):
        prepared = session(graph, source=path, ds=ds, **options)
        prepared.fit()
    assert_state(graph.state_dict(), before)
    assert torch.equal(torch.get_rng_state(), rng)
    if prepared is not None:
        assert prepared.trainer.update_count == 0 and not prepared.trainer.optimizer.state
