"""Public discovery contracts and executable current README; CPU only."""
import ast
import importlib
import inspect
from pathlib import Path
import re
import typing

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name,module', [
    ('Trainer', 'hyso.training'), ('ModularCNN', 'hyso.models.modular'),
    ('BenchmarkRunner', 'hyso.benchmark'), ('ExperimentManager', 'hyso.experiments')])
def test_root_alias_identity_and_completion(name, module):
    import hyso
    assert name in dir(hyso)
    assert getattr(hyso, name) is getattr(importlib.import_module(module), name)
    assert getattr(hyso, name) is getattr(hyso, name)
    with pytest.raises(AttributeError):
        getattr(hyso, 'NoSuchPublicAPI')


def test_named_model_exports_do_not_change_legacy_wildcard():
    from hyso import models
    for name, module in [('ModularCNN', 'modular'), ('ConvNormAct', 'blocks'),
                         ('ResidualBlock', 'blocks'), ('ECAAttention', 'attention'),
                         ('CoordinateAttention', 'attention'), ('SEBlock', 'WithSE'),
                         ('CBAMBlock', 'bcam')]:
        assert getattr(models, name) is getattr(importlib.import_module('hyso.models.'+module), name)
    assert models.__all__ == ['EBO', 'ELO', 'CEMO', 'EMA', 'CNNSENormalWithFit']


def test_planner_and_presentation_exports_are_canonical():
    from hyso import analysis, advisor, performance
    from hyso.analysis.feature_plan import plan_features
    from hyso.advisor.presentation import to_display_dict
    from hyso.performance.inference import analyze_latency
    assert analysis.plan_features is plan_features
    assert advisor.to_display_dict is to_display_dict
    assert performance.analyze_latency is analyze_latency


def test_main_entry_point_annotations_and_help_are_resolvable():
    from hyso import Trainer, ModularCNN, BenchmarkRunner, ExperimentManager, advisor, analysis
    for target in (Trainer.__init__, Trainer.fit, ModularCNN.__init__,
                   ModularCNN.forward, BenchmarkRunner.__init__, BenchmarkRunner.run,
                   ExperimentManager.__init__, advisor.advise, analysis.plan_features):
        assert typing.get_type_hints(target)['return'] is not inspect.Signature.empty
        assert inspect.getdoc(target)
    for target in (Trainer.__init__, ModularCNN.__init__, BenchmarkRunner.__init__):
        assert 'Args:' in inspect.getdoc(target)
    assert typing.get_type_hints(ModularCNN.forward)['return'] is torch.Tensor
    assert (ROOT/'hyso/py.typed').is_file()


def test_current_readme_examples_execute_cpu_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # No example may initialize or measure a GPU, regardless of the host setup.
    def forbidden(*args, **kwargs):
        pytest.fail('README attempted CUDA initialization')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    text = (ROOT/'README.md').read_text(encoding='utf-8')
    examples = re.findall(r'^```python\n(.*?)^```', text, re.M | re.S)
    assert len(examples) == 8
    namespace = {}
    for example in examples:
        ast.parse(example, feature_version=(3, 10))
        exec(compile(example, 'README.md', 'exec'), namespace)
    assert namespace['model'].classifier.out_features == 2
    assert namespace['trainer'].update_count == 2
    assert namespace['history']['val_loss'] == [None]
    assert namespace['history']['val_acc'] == [None]
    assert namespace['data']['configuration']['measure_iterations'] == 3
    assert namespace['plan'].execution_authorized is False
    assert namespace['built_model'].to_config() == namespace['rebuilt'].to_config()
    assert namespace['training_result'].to_dict()['applied_updates'] == 2
    assert namespace['analysis_dry_run'].to_dict()['execution_authorized'] is False
    assert list(tmp_path.iterdir()) == []


def test_python_api_training_example_executes_cpu_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('Python API guide attempted CUDA initialization')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    path = ROOT/'docs/guides/python-api.md'
    examples = re.findall(r'^```python\n(.*?)^```', path.read_text(encoding='utf-8'), re.M | re.S)
    assert len(examples) == 2
    namespace = {}
    for example in examples:
        ast.parse(example, feature_version=(3, 10))
        exec(compile(example, str(path), 'exec'), namespace)
    assert namespace['training_result'].to_dict()['applied_updates'] == 2
    assert namespace['inference_model'] is namespace['session'].trainer.evaluation_model
    assert list(tmp_path.iterdir()) == []


def test_user_documentation_links_exist():
    for path in (ROOT/'README.md', ROOT/'hyso/README.md', ROOT/'docs/guides/python-api.md',
                 ROOT/'docs/guides/training-config.md'):
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if '://' not in link and not link.startswith('#'):
                assert (path.parent/link.split('#')[0]).exists(), (path, link)
