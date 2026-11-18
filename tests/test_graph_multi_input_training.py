"""Small named-input CPU training, checked against the native loss/update loop."""
import copy

import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from hyso import GraphModelBuilder, TrainingConfig, TrainingConfigError, Trainer
from hyso.checkpoint import TrainingCheckpointError
from hyso.models.output import GraphTensorOutput
from test_graph_training_resume import cpu_only


class Samples(Dataset):
    def __init__(self, dtype=torch.float32, task='multiclass'):
        self.x = torch.arange(27, dtype=dtype).reshape(3, 1, 3, 3) / 27
        self.y = torch.tensor([0, 1, 1], dtype=torch.long if task == 'multiclass' else dtype)

    def __len__(self):
        return 3

    def __getitem__(self, i):
        return {'right': self.x[i] * .3 + .1, 'left': self.x[i]}, self.y[i]


def build(dtype=torch.float32, task='multiclass'):
    graph = (GraphModelBuilder({'left': [1, 3, 3], 'right': [1, 3, 3]})
             .add('a', 'conv2d', ['left'], in_channels=1, out_channels=2)
             .add('b', 'conv2d', ['right'], in_channels=1, out_channels=2)
             .add('sum', 'add', ['a', 'b'])
             .add('head', 'classification_head', ['sum'], in_features=2,
                  num_classes=2 if task == 'multiclass' else 1).outputs(logits='head').build().to(dtype))
    return GraphTensorOutput(graph, 'logits', task=task, named_inputs=True)


@pytest.mark.parametrize('mode', ['trainer', 'session'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('task', ['multiclass', 'binary'])
def test_named_training_matches_native_update(mode, dtype, task):
    model = build(dtype, task)
    reference = copy.deepcopy(model)
    data = Samples(dtype, task)
    loss_fn = torch.nn.CrossEntropyLoss() if task == 'multiclass' else torch.nn.BCEWithLogitsLoss()
    expected_optimizer = torch.optim.SGD(reference.parameters(), lr=.1, momentum=.8)
    expected_optimizer.zero_grad()
    summed = 0.
    for inputs, targets in DataLoader(data, batch_size=2):
        logits = reference(inputs)
        loss = loss_fn(logits, targets if task == 'multiclass' else targets[:, None])
        summed += loss.item() * len(targets)
        (loss * len(targets)).backward()
    for p in reference.parameters():
        p.grad.div_(3)
    expected_optimizer.step()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    ids = [id(p) for p in model.parameters()]
    if mode == 'session':
        c = TrainingConfig(epochs=1, batch_size=2, shuffle=False, accum_steps=2, task=task,
                           precision='fp64' if dtype == torch.float64 else 'fp32', validation=True,
                           optimizer='sgd', learning_rate=.1, optimizer_params={'momentum': .8})
        train, val = c.make_loaders(data, data)
        session = c.prepare(model, train, val)
        history = session.fit().history
        trainer = session.trainer
    else:
        trainer = Trainer(model, optimizer=torch.optim.SGD(model.parameters(), lr=.1, momentum=.8),
                          loss_fn=loss_fn, task=task, accum_steps=2)
        history = trainer.fit(DataLoader(data, batch_size=2), DataLoader(data, batch_size=2), epochs=1)
    assert history['train_loss'][0] == pytest.approx(summed / 3)
    assert history['val_loss'][0] is not None
    assert trainer.update_count == 1 and trainer.microbatch_count == 2
    assert ids == [id(p) for p in model.parameters()]
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, reference.state_dict()[name], rtol=0, atol=0)
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if 'node_a' in n)
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if 'node_b' in n)


@pytest.mark.parametrize('problem', [
    'missing', 'extra', 'value', 'batch', 'shape', 'dtype', 'tuple',
    'empty', 'scalar', 'integer', 'input_device', 'target_device',
])
def test_bad_inputs_fail_before_update(problem):
    model = build()
    inputs = {'left': torch.ones(2, 1, 3, 3), 'right': torch.zeros(2, 1, 3, 3)}
    if problem == 'missing': del inputs['left']
    if problem == 'extra': inputs['other'] = inputs['left']
    if problem == 'value': inputs['left'] = 'invalid'
    if problem == 'batch': inputs['right'] = inputs['right'][:1]
    if problem == 'shape': inputs['right'] = torch.zeros(2, 1, 2, 2)
    if problem == 'dtype': inputs['right'] = inputs['right'].double()
    if problem == 'tuple': inputs = tuple(inputs.values())
    if problem == 'empty': inputs['right'] = inputs['right'][:0]
    if problem == 'scalar': inputs['right'] = torch.tensor(0.)
    if problem == 'integer': inputs['right'] = inputs['right'].long()
    if problem == 'input_device': inputs['right'] = inputs['right'].to('meta')
    targets = torch.tensor([0, 1], device='meta' if problem == 'target_device' else 'cpu')
    before = copy.deepcopy(model.state_dict())
    trainer = Trainer(model, optimizer=torch.optim.SGD(model.parameters(), lr=.1), loss_fn=torch.nn.CrossEntropyLoss())
    with pytest.raises((ValueError, TypeError)):
        trainer.fit([(inputs, targets)], epochs=1)
    assert trainer.update_count == 0
    for n, p in model.state_dict().items(): assert torch.equal(p, before[n])


@pytest.mark.parametrize('output,task,message', [
    ('missing', 'multiclass', 'explicit graph output'),
    ('features', 'multiclass', 'classification_head'),
    (['logits', 'features'], 'multiclass', 'explicit graph output'),
    ('logits', None, 'explicit classification task'),
    ('logits', 'regression', 'explicit classification task'),
    ('logits', 'binary', 'width does not match task'),
])
def test_unsupported_output_selection(output, task, message):
    model = build()
    config = model.graph.to_config()
    config['outputs']['features'] = 'sum'
    graph = GraphModelBuilder.from_config(config).build()
    with pytest.raises(ValueError, match=message):
        GraphTensorOutput(graph, output, task=task, named_inputs=True)


def test_named_inputs_requires_explicit_opt_in():
    with pytest.raises(ValueError, match='exactly one'):
        GraphTensorOutput(build().graph, 'logits')


@pytest.mark.parametrize('count', [1, 3])
def test_named_inputs_requires_exactly_two_inputs(count):
    names = [f'input{i}' for i in range(count)]
    graph = (GraphModelBuilder({name: [1, 3, 3] for name in names})
             .add('head', 'classification_head', [names[0]], in_features=1, num_classes=2)
             .outputs(logits='head').build())
    with pytest.raises(ValueError, match='exactly two inputs'):
        GraphTensorOutput(graph, 'logits', task='multiclass', named_inputs=True)


@pytest.mark.parametrize('flag', [None, 1, 'true'])
def test_named_inputs_flag_must_be_bool(flag):
    with pytest.raises(TypeError, match='named_inputs must be bool'):
        GraphTensorOutput(build().graph, 'logits', task='multiclass', named_inputs=flag)


@pytest.mark.parametrize('mode', ['trainer', 'session'])
@pytest.mark.parametrize('with_model_config', [False, True])
@pytest.mark.parametrize('checkpoint_option', ['save_last_to', 'resume_from'])
def test_checkpoint_boundaries_before_iteration_or_mutation(
        tmp_path, monkeypatch, mode, with_model_config, checkpoint_option):
    model = build()
    before = copy.deepcopy(model.state_dict())
    path = tmp_path / 'last.pth'
    sentinel = b'checkpoint must not be read or overwritten'
    path.write_bytes(sentinel)

    def forbidden(*args, **kwargs):
        pytest.fail('unsupported multi-input checkpoint must fail before data/checkpoint access')

    monkeypatch.setattr(Samples, '__getitem__', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    options = {'model_config': model.graph.to_config()} if with_model_config else {}
    c = TrainingConfig(epochs=1, batch_size=2, **{checkpoint_option: str(path)})
    train, _ = c.make_loaders(Samples())
    rng = torch.get_rng_state().clone()
    if mode == 'session':
        with pytest.raises(TrainingConfigError, match='multi-input'):
            c.prepare(model, train, **options)
    else:
        trainer = Trainer(model, optimizer=torch.optim.SGD(model.parameters(), lr=.1),
                          loss_fn=torch.nn.CrossEntropyLoss(), **options)
        with pytest.raises(TrainingCheckpointError, match='NamedTensorDataset' if with_model_config else 'model_config'):
            trainer.fit(train, epochs=1, **{checkpoint_option: str(path)})
        assert trainer.update_count == trainer.microbatch_count == 0
        assert not trainer.optimizer.state
    assert torch.equal(torch.get_rng_state(), rng)
    assert path.read_bytes() == sentinel
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_trainer_revalidates_selected_classification_head():
    model = build(task='binary')
    with pytest.raises(ValueError, match='width does not match task'):
        Trainer(model, optimizer=torch.optim.SGD(model.parameters(), lr=.1),
                loss_fn=torch.nn.CrossEntropyLoss(), task='multiclass')


def test_raw_dict_output_is_not_a_loss_adapter():
    model = build().graph
    trainer = Trainer(model, optimizer=torch.optim.SGD(model.parameters(), lr=.1), loss_fn=torch.nn.CrossEntropyLoss())
    with pytest.raises(TypeError, match='GraphTensorOutput'):
        trainer.fit(DataLoader(Samples(), batch_size=2), epochs=1)
