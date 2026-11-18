"""Graph CLI preview delegates to the public, allocation-free planning API."""
import json

import pytest
import torch

from hyso import GraphModelBuilder, ModelBuilder
from hyso.cli import main
from hyso.models.registry import ModelRegistry


@pytest.fixture(autouse=True)
def no_workloads(monkeypatch):
    from hyso import benchmark, training

    def forbidden(*args, **kwargs):
        pytest.fail('Graph CLI preview attempted allocation, forward, training, benchmark or CUDA')

    for obj, name in [(GraphModelBuilder, 'build'), (ModelBuilder, 'build'),
                      (ModelRegistry, 'create'), (torch.nn.Module, '__init__'),
                      (torch.nn.Module, '_call_impl'),
                      (torch, 'load'), (training.Trainer, 'fit'),
                      (training.TrainingSession, 'fit'), (benchmark.BenchmarkRunner, 'run')]:
        monkeypatch.setattr(obj, name, forbidden)
    for name in ('init', '_lazy_init', 'is_available', 'device_count', 'current_device', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


@pytest.fixture
def graph_config(tmp_path):
    builder = (GraphModelBuilder({'image': [4, 5, 7]})
               .add('left', 'conv2d', ['image'], in_channels=4, out_channels=4)
               .add('right', 'pointwise_conv2d', ['image'], in_channels=4, out_channels=4)
               .add('merge', 'add', ['left', 'right']).outputs(features='merge'))
    path = tmp_path / 'graph architecture.json'
    builder.to_json(path)
    return builder, path


def invoke(capsys, *args):
    code = main([*map(str, args), '--json'])
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result['exit_code'] == code
    return code, result, captured.err


@pytest.mark.parametrize('batch_size', [1, 7])
def test_graph_batch_preview_matches_public_api(capsys, graph_config, tmp_path, monkeypatch, batch_size):
    builder, path = graph_config
    original_bytes = path.read_bytes()
    expected = builder.preview(batch_size=batch_size)
    preview = GraphModelBuilder.preview
    calls = []

    def observe(self, **options):
        calls.append((self.to_config(), options))
        return preview(self, **options)

    monkeypatch.setattr(GraphModelBuilder, 'preview', observe)
    output = tmp_path / 'preview result.json'
    code, envelope, err = invoke(capsys, 'models', 'preview', path,
                                 '--batch-size', batch_size, '--output', output)
    assert code == 0 and envelope['status'] == 'ok' and not err
    result = envelope['result']
    assert result['validation'] == expected
    assert result['validation']['activation_estimate']['batch_size'] == batch_size
    assert result['validation']['activation_estimate']['memory_guarantee'] is False
    assert calls == [(builder.to_config(), {'batch_size': batch_size})]
    assert not result['model_constructed'] and not result['weights_included']
    assert not result['trained_checkpoint']
    assert json.loads(output.read_text(encoding='utf-8')) == result
    assert GraphModelBuilder.from_config(result['architecture']).to_config() == builder.to_config()
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize('action', ['validate', 'summary', 'preview'])
def test_existing_graph_metadata_behavior(capsys, graph_config, action):
    builder, path = graph_config
    code, envelope, err = invoke(capsys, 'models', action, path)
    assert code == 0 and not err
    result = envelope['result']
    assert result['architecture'] == builder.to_config()
    assert result['validation'] == builder.preflight(builder.to_config())
    assert result['model_constructed'] is False
    if action == 'summary':
        assert result['summary']['outputs'] == result['validation']['outputs']


@pytest.mark.parametrize('batch_size', ['0', '-1', '1.5', 'invalid'])
def test_invalid_batch_has_explicit_error(capsys, graph_config, batch_size):
    _, path = graph_config
    code, envelope, err = invoke(capsys, 'models', 'preview', path, '--batch-size', batch_size)
    assert code == 2 and envelope['status'] == 'error'
    assert 'batch-size' in envelope['error'] and err
    assert 'result' not in envelope


def test_batch_option_rejects_non_graph(capsys, tmp_path):
    path = tmp_path / 'sequential.json'
    ModelBuilder((1, 4, 4), 2).add('conv2d', in_channels=1, out_channels=2).to_json(path)
    assert invoke(capsys, 'models', 'preview', path)[0] == 0
    code, envelope, _ = invoke(capsys, 'models', 'preview', path, '--batch-size', 2)
    assert code == 4 and envelope['status'] == 'unsupported'
    assert 'only for graph preview' in envelope['error']


@pytest.mark.parametrize('mutation', ['component', 'parameter', 'schema', 'duplicate'])
def test_invalid_graph_config_rejected(capsys, graph_config, mutation):
    builder, path = graph_config
    config = builder.to_config()
    if mutation == 'component':
        config['nodes'][0]['component'] = 'unsupported_graph_component'
    elif mutation == 'parameter':
        config['nodes'][0]['params']['unsupported_parameter'] = 1
    elif mutation == 'schema':
        config['schema_version'] = 999
    raw = json.dumps(config)
    if mutation == 'duplicate':
        raw = raw.replace('"kind":', '"kind": "graph", "kind":', 1)
    path.write_text(raw, encoding='utf-8')
    code, envelope, err = invoke(capsys, 'models', 'preview', path, '--batch-size', 2)
    assert code == 2 and envelope['status'] == 'error' and err
    assert envelope['diagnostics'] and 'result' not in envelope


@pytest.mark.parametrize('arguments', [('preview', '--device', 'cuda'),
                                      ('validate', '--batch-size', '2')])
def test_unsupported_cli_parameters_rejected(capsys, graph_config, arguments):
    _, path = graph_config
    action, *flags = arguments
    code, envelope, _ = invoke(capsys, 'models', action, path, *flags)
    assert code == 2 and 'unrecognized arguments' in envelope['error']
