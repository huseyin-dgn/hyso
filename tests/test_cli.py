"""Stage 11 CLI contracts. Every live workload/device API is fenced off."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import pytest
import torch

from hyso import ModelBuilder, TrainingConfig, analysis as api
from hyso.cli import main
from hyso.cli.main import build_parser
from fixture_support import load_fixture

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_workloads(monkeypatch):
    from hyso import training, benchmark, performance
    def forbidden(*a, **kw):
        pytest.fail('CLI attempted a live workload, CUDA query or checkpoint load')
    for obj, name in [(training.Trainer, 'fit'), (training.TrainingSession, 'fit'),
                      (benchmark.BenchmarkRunner, 'run'), (torch.nn.Module, '_call_impl'), (torch, 'load')]:
        monkeypatch.setattr(obj, name, forbidden)
    for name in ('init', '_lazy_init', 'is_available', 'device_count', 'current_device', 'synchronize', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    for name in ('measure_throughput', 'measure_pipeline', 'measure_stability', 'measure_streaming', 'measure_utilization'):
        if hasattr(performance, name):
            monkeypatch.setattr(performance, name, forbidden)


def invoke(capsys, *args):
    code = main([*map(str, args), '--json'])
    captured = capsys.readouterr()
    value = json.loads(captured.out)
    assert value['exit_code'] == code
    return code, value, captured.err


def write(path, data):
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


@pytest.fixture
def architecture(tmp_path):
    p = tmp_path / 'architecture.json'
    ModelBuilder((1, 4, 4), 2).add('conv2d', in_channels=1, out_channels=2).to_json(p)
    return p


@pytest.fixture
def config(tmp_path):
    p = tmp_path / 'training.json'
    TrainingConfig(epochs=1, batch_size=2).to_json(p)
    return p


@pytest.fixture
def native(tmp_path):
    p = write(tmp_path / 'native.json', load_fixture('training-p19-schema1'))
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def report(tmp_path):
    from test_reporting import native_report
    p = tmp_path / 'analysis.json'
    native_report().to_json(p)
    return p


def test_all_help_paths_are_safe(capsys, monkeypatch):
    from hyso.models.builder import ModelBuilder
    monkeypatch.setattr(ModelBuilder, 'build', lambda *a: pytest.fail('help constructed a model'))
    parser = build_parser()
    import hyso.cli.train as train_command
    import hyso.cli.analysis as analysis_command
    for module in (train_command, analysis_command):
        monkeypatch.setattr(module, 'load_factory', lambda *a: pytest.fail('help loaded a factory'))
    def walk(parser, path):
        assert main([*path, '--help']) == 0
        text = capsys.readouterr().out
        assert 'usage:' in text and '--help' in text
        for action in parser._actions:
            if hasattr(action, 'choices') and isinstance(action.choices, dict):
                for name, child in action.choices.items():
                    walk(child, [*path, name])
    walk(parser, [])


def test_models_catalog_validation_summary_inspect(capsys, architecture, monkeypatch):
    code, value, _ = invoke(capsys, 'models', 'list')
    assert code == 0 and value['result']['components']
    name = value['result']['components'][0]['name']
    assert invoke(capsys, 'models', 'show', name)[1]['result']['parameters']
    original = ModelBuilder.build
    monkeypatch.setattr(ModelBuilder, 'build', lambda *a: pytest.fail('validation constructed model'))
    for action in ('validate', 'summary'):
        code, value, _ = invoke(capsys, 'models', action, architecture)
        assert code == 0 and not value['result']['model_constructed']
        assert value['result']['trained_checkpoint'] is False
    assert invoke(capsys, 'models', 'inspect', architecture)[0] == 3
    monkeypatch.setattr(ModelBuilder, 'build', original)
    code, value, _ = invoke(capsys, 'models', 'inspect', architecture, '--execute')
    assert code == 0 and value['result']['parameters'] > 0
    assert value['result']['forward_executed'] is False
    assert value['result']['tensors'][0]['shape']


@pytest.mark.parametrize('payload', ['{', '{"a":1,"a":2}', '{"x":NaN}', '[]', '{"schema_version":9}'])
def test_bad_json_schema(capsys, tmp_path, payload):
    p = tmp_path / 'bad.json'; p.write_text(payload, encoding='utf-8')
    for group, action in [('models', 'validate'), ('train', 'validate'), ('reports', 'render')]:
        code, _, err = invoke(capsys, group, action, p)
        assert code == 2 and 'Traceback' not in err


@pytest.mark.parametrize('args', [[], ['bogus'], ['models'], ['models', 'show', 'bogus'],
    ['models', 'validate', 'missing.json'], ['analysis', 'show', 'P36'],
    ['analysis', 'plan', '--features', 'P00'], ['analysis', 'plan', '--features', 'P01', 'P01'],
    ['analysis', 'plan', '--features', 'P11', '--device', 'cuda'],
    ['analysis', 'plan', '--features', 'P11', '--device', 'mps']])
def test_usage_errors(capsys, args):
    code, _, err = invoke(capsys, *args)
    assert code == 2 and err and 'Traceback' not in err


def test_cli_syntax_python310():
    for p in [ROOT / 'hyso/__main__.py', *(ROOT / 'hyso/cli').glob('*.py')]:
        ast.parse(p.read_text(encoding='utf-8'), feature_version=(3, 10))


def test_training_options_validation_and_no_import(capsys, config):
    assert invoke(capsys, 'train', 'options')[1]['result']['optimizers'] == ['sgd', 'adam', 'adamw']
    assert invoke(capsys, 'train', 'validate', config)[0] == 0
    code, value, _ = invoke(capsys, 'train', 'run', config, '--factory', 'absent_never_import:factory')
    assert code == 3 and value['result']['factory_loaded'] is False
    assert not value['result']['workload_started']
    code, value, _ = invoke(capsys, 'train', 'run', config, '--factory', 'absent_never_import:factory', '--trust-factory', '--allow-training')
    assert code == 0 and value['status'] == 'dry_run' and value['result']['execution_authorized'] is False


@pytest.mark.parametrize('field,value', [('device', 'cuda'), ('device', 'mps'), ('precision', 'fp16'),
    ('precision', 'amp_fp16'), ('optimizer', 'unknown'), ('scheduler', 'unknown')])
def test_training_invalid_options(capsys, tmp_path, field, value):
    data = TrainingConfig().to_dict(); data[field] = value
    p = write(tmp_path / 'config.json', data)
    assert invoke(capsys, 'train', 'validate', p)[0] == 2


@pytest.mark.parametrize('flags', [[], ['--allow-training'], ['--trust-factory']])
def test_training_permission_before_import(capsys, config, flags):
    code, _, err = invoke(capsys, 'train', 'run', config, '--factory', 'absent_never_import:factory', '--execute', *flags)
    assert code == 3 and 'absent_never_import' not in err


def test_training_fake_session_delegates_once_and_clean_stdout(capsys, config, monkeypatch):
    from hyso.training import TrainingSession
    from test_reporting import training
    calls = []
    session = TrainingSession.__new__(TrainingSession)
    session.settings = lambda: dict(state='prepared', requested=TrainingConfig.from_json(config).to_dict())
    session.fit = lambda: (calls.append('fit') or training())
    module = types.ModuleType('cli_test_factory')
    def factory(c):
        print('factory diagnostics')
        return session
    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    code, value, err = invoke(capsys, 'train', 'run', config, '--factory', module.__name__ + ':factory',
                             '--execute', '--allow-training', '--trust-factory')
    assert code == 0 and calls == ['fit'] and 'factory diagnostics' in err
    assert value['result']['format'] == 'hyso.training_session_result'
    session.fit = Mock(side_effect=RuntimeError('fixture failure'))
    assert invoke(capsys, 'train', 'run', config, '--factory', module.__name__ + ':factory', '--execute',
                  '--allow-training', '--trust-factory')[0] == 5
    assert session.fit.call_count == 1


def test_training_checkpoint_and_cuda_gates(capsys, tmp_path):
    for kwargs, flag in [(dict(device='cuda:0'), '--allow-cuda'),
                          (dict(save_last_to=str(tmp_path / 'last.pth')), '--allow-file-writes')]:
        p = tmp_path / 'cfg.json'; write(p, TrainingConfig(**kwargs).to_dict())
        code, _, err = invoke(capsys, 'train', 'run', p, '--execute', '--factory', 'absent_never_import:factory', '--trust-factory', '--allow-training')
        assert code == 3 and flag in err


def test_analysis_catalog_and_plan(capsys):
    code, value, _ = invoke(capsys, 'analysis', 'list')
    assert code == 0 and [r['feature_id'] for r in value['result']] == [f'P{i:02d}' for i in range(1, 36)]
    assert all(r['executor_modes'] and r['cost_class'] for r in value['result'])
    assert invoke(capsys, 'analysis', 'show', 'P31')[1]['result']['cost_class'] == 'long_7200s'
    code, value, _ = invoke(capsys, 'analysis', 'plan', '--features', 'P22')
    assert code == 0 and value['result']['execution_mode'] == 'native'
    assert value['result']['selected_features'] == ['P22']
    assert value['result']['execution_authorized'] is False


@pytest.mark.parametrize('feature,flags,missing', [
    ('P01', [], '--trust-factory'), ('P01', ['--trust-factory'], '--allow-capture'),
    ('P19', ['--trust-factory', '--allow-capture'], '--allow-training'),
    ('P30', ['--trust-factory', '--allow-capture'], '--allow-camera'),
    ('P31', ['--trust-factory', '--allow-capture'], '--allow-long-run')])
def test_live_permissions_before_import(capsys, feature, flags, missing):
    code, _, err = invoke(capsys, 'analysis', 'run', '--features', feature, '--execute', '--authorize', feature,
                         '--factory', 'absent_never_import:factory', *flags)
    assert code == 3 and missing in err


def test_live_cuda_gate_before_import_and_dryrun(capsys):
    common = ['analysis', 'run', '--features', 'P14', '--device', 'cuda:0', '--factory', 'absent_never_import:factory', '--trust-factory', '--allow-capture']
    code, value, _ = invoke(capsys, *common)
    assert code == 3 and value['result']['cli_source']['factory_loaded'] is False
    assert invoke(capsys, *common, '--execute', '--authorize', 'P14')[0] == 3


def test_retained_dryrun_authorization_and_P22(capsys, native, tmp_path):
    path, sha = native
    base = ['analysis', 'run', '--features', 'P19', 'P22', '--source', path, '--sha256', sha]
    code, value, _ = invoke(capsys, *base)
    assert code == 0 and value['result']['execution_authorized'] is False
    assert invoke(capsys, *base, '--execute')[0] == 3
    assert invoke(capsys, *base, '--execute', '--authorize', 'P35')[0] == 3
    output = tmp_path / 'execution.json'
    code, value, _ = invoke(capsys, *base, '--execute', '--authorize', 'P22', '--output', output)
    assert code == 0
    rows = {r['feature_id']: r for r in value['result']['features']}
    assert rows['P19']['status'] == 'SKIPPED' and rows['P22']['status'] == 'COMPLETED'
    assert rows['P22']['evidence_record']['origin'] == 'reused'
    assert api.AnalysisExecutionResult.from_json(output).verify_artifacts()[0]['status'] == 'verified'


@pytest.mark.parametrize('sha', [None, '0' * 64, 'invalid'])
def test_retained_hash_failures(capsys, native, sha):
    args = ['analysis', 'run', '--features', 'P22', '--source', native[0]]
    if sha is not None: args += ['--sha256', sha]
    assert invoke(capsys, *args)[0] == 2


@pytest.mark.parametrize('feature,code', [('P11', 5), ('P22', 0)])
def test_native_outcome_exit_codes(capsys, native, feature, code):
    result = invoke(capsys, 'analysis', 'run', '--features', feature, '--source', native[0], '--sha256', native[1], '--execute', '--authorize', feature)
    assert result[0] == code


def test_partial_and_blocked_statuses(capsys, native):
    assert invoke(capsys, 'analysis', 'run', '--features', 'P11')[0] == 3
    code, value, _ = invoke(capsys, 'analysis', 'run', '--features', 'P19', 'P11', '--source', native[0], '--sha256', native[1], '--execute', '--authorize', 'P19', 'P11')
    assert code == 6 and value['status'] == 'partial'


def test_factory_policy_cannot_grant_permissions(capsys, monkeypatch):
    from hyso import performance as perf
    module = types.ModuleType('cli_bindings_factory')
    module.factory = lambda: [api.NativeBinding('fixture', ('model',), ('P14',),
        api.RecipeSource(perf.ColdStartRecipe('absent:factory', device='cuda:0')),
        policy=api.OperationPolicy(allow_cuda=True, allow_capture=True), device='cuda:0')]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    # CPU plan + CUDA binding must stay blocked, even if factory supplies consent.
    assert invoke(capsys, 'analysis', 'run', '--features', 'P14', '--factory', module.__name__ + ':factory', '--execute',
                  '--authorize', 'P14', '--trust-factory', '--allow-capture')[0] in (3, 4)


@pytest.mark.parametrize('format', ['json', 'markdown', 'html'])
@pytest.mark.parametrize('language,level', [('tr', 'beginner'), ('en', 'technical'), ('tr', 'expert')])
def test_advisor_report_exports(capsys, report, tmp_path, format, language, level):
    advice = tmp_path / 'advice.json'
    assert invoke(capsys, 'advisor', 'recommend', report, '--output', advice)[0] == 0
    retained = json.loads(advice.read_text(encoding='utf-8'))
    assert retained['claims']['numerical_improvement'] is None
    assert all(row['estimated_improvement'] is None and row['authorization'] == 'not_authorized'
               for row in retained['recommendations'])
    for group, action, path in [('advisor', 'recommend', report), ('reports', 'render', report), ('reports', 'render', advice)]:
        output = tmp_path / (group + action + path.stem + '.' + format)
        code, value, _ = invoke(capsys, group, action, path, '--format', format, '--language', language, '--level', level, '--output', output)
        assert code == 0 and output.stat().st_size
        if format == 'json': json.loads(output.read_text(encoding='utf-8'))
        if format == 'html': assert '<html' in output.read_text(encoding='utf-8')


def test_comparison_no_winner_roundtrip(capsys, report, tmp_path):
    out = tmp_path / 'comparison.json'
    args = ['compare', report, report, '--baseline-model', 'retained-A', '--candidate-model', 'retained-B', '--features', 'P01', 'P11']
    code, value, _ = invoke(capsys, *args, '--output', out)
    assert code == 0 and value['result']['claims']['global_winner'] is None
    assert value['result']['claims']['acceptance'] == 'NOT_EVALUATED'
    assert invoke(capsys, *args, '--format', 'markdown')[0] == 0
    assert invoke(capsys, 'advisor', 'recommend', out)[0] == 0
    assert invoke(capsys, 'reports', 'render', out)[0] == 0


def test_report_training_and_evidence_context(capsys, report, tmp_path):
    assert invoke(capsys, 'reports', 'render', ROOT / 'examples/reporting/fixtures/training-result-v1.json')[0] == 0
    from hyso.cli.common import read_result
    index = tmp_path / 'index.json'; read_result(report).to_dict()
    write(index, read_result(report).to_dict()['index'])
    context = write(tmp_path / 'context.json', dict(workload_id='test', task='classification', model={}, dataset={}))
    for group, action in [('advisor', 'recommend'), ('reports', 'render')]:
        assert invoke(capsys, group, action, index)[0] == 2
        assert invoke(capsys, group, action, index, '--context', context)[0] == 0


def test_output_conflicts_no_clobber(capsys, architecture, tmp_path):
    out = tmp_path / 'result.json'; out.write_text('sentinel', encoding='utf-8')
    args = ['models', 'validate', architecture, '--output', out]
    assert invoke(capsys, *args)[0] == 2 and out.read_text() == 'sentinel'
    assert invoke(capsys, *args, '--overwrite')[0] == 0
    assert invoke(capsys, 'models', 'validate', architecture, '--output', architecture, '--overwrite')[0] == 2
    assert invoke(capsys, 'models', 'validate', architecture, '--output', tmp_path)[0] == 2
    assert invoke(capsys, 'models', 'validate', architecture, '--output', tmp_path / 'missing' / 'out.json')[0] == 2


def test_preexisting_export_blocks_factory(capsys, native, tmp_path):
    p = tmp_path / 'out.json'; p.write_text('sentinel', encoding='utf-8')
    assert invoke(capsys, 'analysis', 'run', '--features', 'P22', '--source', native[0], '--sha256', native[1],
                  '--execute', '--authorize', 'P22', '--output', p)[0] == 2
    assert p.read_text() == 'sentinel'


def test_debug_and_unexpected_error(capsys, monkeypatch):
    import hyso.cli.models as models
    monkeypatch.setattr(models, 'run', Mock(side_effect=RuntimeError('fixture unexpected')))
    code, _, err = invoke(capsys, 'models', 'list')
    assert code == 1 and 'Traceback' not in err
    assert 'Traceback' in invoke(capsys, '--debug', 'models', 'list')[2]


def test_keyboard_interrupt(capsys, monkeypatch):
    import hyso.cli.models as models
    monkeypatch.setattr(models, 'run', Mock(side_effect=KeyboardInterrupt()))
    assert invoke(capsys, 'models', 'list')[0] == 130


@pytest.mark.parametrize('schema', [1, 2])
def test_manager_schema_history_open_verify_and_report(capsys, tmp_path, schema):
    from hyso.experiments import ExperimentManager
    from test_experiment_workflow import training_run
    root = tmp_path / 'store'; m = ExperimentManager(root)
    if schema == 2:
        m.close(); ExperimentManager.migrate_store(root); m = ExperimentManager(root)
    run, result = training_run(m)
    expid = run.to_dict()['experiment_id']; runid = run.id
    artifact = run.record_result(result) if schema == 2 else None
    run.finish(); m.close()
    before = hashlib.sha256((root / 'experiments.sqlite3').read_bytes()).hexdigest()
    for args in [('experiments', 'list', '--root', root), ('experiments', 'history', expid, '--root', root),
                 ('experiments', 'run', runid, '--root', root), ('reports', 'experiment', expid, '--root', root)]:
        assert invoke(capsys, *args)[0] == 0
    assert hashlib.sha256((root / 'experiments.sqlite3').read_bytes()).hexdigest() == before
    if schema == 1:
        assert invoke(capsys, 'experiments', 'artifacts', '--root', root)[0] == 4
    else:
        assert invoke(capsys, 'experiments', 'artifacts', '--root', root)[0] == 0
        assert invoke(capsys, 'experiments', 'open', artifact['id'], '--root', root)[0] == 0
        assert invoke(capsys, 'experiments', 'verify', artifact['id'], '--root', root)[0] == 0
        m = ExperimentManager(root); path = m._managed(artifact['id']); m.close()
        path.write_text('{}', encoding='utf-8')
        code, value, _ = invoke(capsys, 'experiments', 'open', artifact['id'], '--root', root)
        assert code == 3 and value['result']['status'] == 'mismatch'


def test_manager_missing_store_not_created(capsys, tmp_path):
    p = tmp_path / 'missing'
    assert invoke(capsys, 'experiments', 'list', '--root', p)[0] == 2
    assert not p.exists()


@pytest.mark.parametrize('feature', ['P32', 'P33'])
def test_offline_runtime_derive_without_capture(capsys, tmp_path, feature):
    from test_native_execution import aborted_stability
    payload = aborted_stability().to_dict()
    p = write(tmp_path / 'p31.json', payload)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    code, value, _ = invoke(capsys, 'analysis', 'run', '--features', feature, '--source', p, '--sha256', sha,
                            '--execute', '--authorize', feature)
    # Insufficient/aborted retained source cannot be promoted into valid 7200s
    # evidence. The offline derive still returns its native result/receipt.
    row = next(r for r in value['result']['features'] if r['feature_id'] == feature)
    assert row['execution_receipt']['operation_kind'] == 'retained'
    assert row['attempted'] and value['result']['acceptance'] == 'NOT_EVALUATED'


@pytest.mark.parametrize('missing', ['power', 'writes'])
def test_native_runtime_permissions_survive_factory(capsys, monkeypatch, tmp_path, missing):
    from hyso import performance as perf
    module = types.ModuleType('cli_runtime_factory')
    gpu = missing == 'power'
    source = api.RuntimeSource(perf.WorkloadRecipe('absent:factory', gpu_uuid='GPU-' + 'a'*32 if gpu else None),
        str(tmp_path / 'native-never-created'), options=dict(protocol=perf.StabilityProtocol()))
    module.factory = lambda: [api.NativeBinding('runtime', ('model',), ('P31',), source,
        device='cuda:0' if gpu else 'cpu', policy=api.OperationPolicy(allow_capture=True,
        allow_file_writes=True, allow_power_sampling=True, allow_cuda=True, allow_long_run=True))]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    flags = ['--device', 'cuda:0', '--allow-cuda', '--allow-file-writes'] if gpu else []
    code, value, _ = invoke(capsys, 'analysis', 'run', '--features', 'P31', '--factory', module.__name__ + ':factory',
        '--execute', '--authorize', 'P31', '--trust-factory', '--allow-capture', '--allow-long-run', *flags)
    assert code == 3
    row = next(r for r in value['result']['features'] if r['feature_id'] == 'P31')
    assert not row['attempted']
    assert any(('power' if gpu else 'file') in reason for reason in row['reasons'])
    assert not (tmp_path / 'native-never-created').exists()


def test_peak_reset_separate_and_pid_validation(capsys, monkeypatch):
    module = types.ModuleType('cli_allocator_factory')
    module.factory = lambda: [api.NativeBinding('allocator', ('model',), ('P09',),
        api.InferenceSource(torch.nn.Identity(), torch.ones(1)), device='cuda:0',
        policy=api.OperationPolicy(allow_capture=True, allow_cuda=True, peak_reset_pid=123))]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    args = ['analysis', 'run', '--features', 'P09', '--factory', module.__name__ + ':factory', '--device', 'cuda:0',
            '--execute', '--authorize', 'P09', '--trust-factory', '--allow-capture', '--allow-cuda']
    code, value, _ = invoke(capsys, *args)
    assert code == 3 and value['result']['features'][0]['reasons'] == ['current_process_allocator_reset_consent_required']
    assert invoke(capsys, *args, '--peak-reset-pid', '0')[0] == 2


def test_raw_advisor_validation_requires_matching_source(capsys, report, tmp_path):
    from hyso.advisor import advise
    from hyso.cli.common import read_result
    source = read_result(report)
    raw = tmp_path / 'advisor.json'; advise(source).to_json(raw)
    assert invoke(capsys, 'reports', 'render', raw)[0] == 2
    assert invoke(capsys, 'reports', 'render', raw, '--advisor-source', report)[0] == 0


def test_source_export_paths_with_spaces(capsys, tmp_path, native):
    folder = tmp_path / 'inputs with spaces'; folder.mkdir()
    p = folder / 'native input.json'; p.write_bytes(native[0].read_bytes())
    out = folder / 'execution output.json'
    assert invoke(capsys, 'analysis', 'run', '--features', 'P22', '--source', p, '--sha256', native[1],
                  '--execute', '--authorize', 'P22', '--output', out)[0] == 0
    assert out.is_file()


def test_documented_demo_runs_offline(tmp_path, capsys):
    import runpy
    module = runpy.run_path(str(ROOT / 'examples/cli/prepare_demo.py'))
    output = tmp_path / 'demo'
    module['prepare'](output)
    capsys.readouterr()
    for group, action, filename in [('models', 'validate', 'architecture.json'), ('train', 'validate', 'training.json'),
                                    ('reports', 'render', 'A-analysis.json'), ('advisor', 'recommend', 'A-analysis.json')]:
        assert invoke(capsys, group, action, output / filename)[0] == 0
    factory = runpy.run_path(str(ROOT / 'examples/cli/bindings.py'))['retained']
    p = output / 'p19.json'; sha = hashlib.sha256(p.read_bytes()).hexdigest()
    bindings = factory(path=str(p), sha256=sha)
    assert type(bindings[0]) is api.NativeBinding
    assert all(not getattr(bindings[0].policy, flag) for flag in ('allow_capture', 'allow_training', 'allow_cuda'))


def test_machine_unicode_survives_legacy_stream(capsys):
    code = main(['analysis', 'plan', '--features', 'P11', '--model-id', 'Türkçe-模型', '--json'])
    raw = capsys.readouterr().out
    assert code == 0 and raw.isascii()
    assert json.loads(raw)['result']['request']['models'][0]['model_id'] == 'Türkçe-模型'


def test_command_system_exit_cannot_report_success(capsys, monkeypatch):
    import hyso.cli.models as models
    monkeypatch.setattr(models, 'run', Mock(side_effect=SystemExit(0)))
    code, value, _ = invoke(capsys, 'models', 'list')
    assert code == 5 and value['status'] == 'failed'


def test_peak_reset_cli_process_consent_is_explicit(capsys):
    import os
    from hyso.cli.analysis import _policy
    parser = build_parser()
    args = ['analysis', 'run', '--features', 'P09']
    assert _policy(parser.parse_args(args)).peak_reset_pid is None
    assert _policy(parser.parse_args([*args, '--allow-peak-reset'])).peak_reset_pid == os.getpid()
    assert invoke(capsys, *args, '--allow-peak-reset', '--peak-reset-pid', str(os.getpid()))[0] == 2


def test_export_cannot_replace_a_planned_native_artifact(capsys, monkeypatch, tmp_path):
    module = types.ModuleType('cli_output_conflict_factory')
    output = tmp_path / 'native-footprint.json'
    module.factory = lambda: [api.NativeBinding('footprint', ('model',), ('P11',),
        api.InferenceSource(torch.nn.Identity(), options=dict(serialize_to=str(output))))]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    code, _, err = invoke(capsys, 'analysis', 'run', '--features', 'P11', '--factory', module.__name__ + ':factory',
        '--execute', '--authorize', 'P11', '--trust-factory', '--allow-capture', '--allow-file-writes',
        '--output', output, '--overwrite')
    assert code == 2 and 'Output cannot replace an input source' in err
    assert not output.exists()


def test_unsupported_native_device_is_distinct_from_failure(capsys, native):
    args = ['analysis', 'run', '--features', 'P19', '--source', native[0], '--sha256', native[1], '--device', 'cuda:0']
    for flags in ([], ['--execute', '--authorize', 'P19']):
        code, value, _ = invoke(capsys, *args, *flags)
        assert code == 4 and value['status'] == 'unsupported'
