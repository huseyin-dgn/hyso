"""Offline P23 metadata reading and execution admission; no factory or fit."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import uuid

import pytest
import torch

from hyso.performance import isolated_training_memory as module
from hyso.training import Trainer


def saved_result(path):
    data = dict(schema_version=1, format='hyso.performance.isolated_training_memory',
                feature_ids=['P23'], probe_id=str(uuid.uuid4()),
                recipe=dict(factory='absent_factory:build', device='cpu', kwargs={},
                            factory_search_path=str(path), seed=0, epochs=1,
                            save_last_to=None, resume_from=None),
                protocol=asdict(module.IsolatedTrainingMemoryProtocol()),
                outcome='unsupported', reason='archived offline fixture', receipt=None,
                worker_receipt=None, measurement=None, limitations=[])
    data['sha256'] = module._hash(data)
    return data


@pytest.fixture
def no_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('offline reader/admission reached factory, training, CUDA or worker execution')
    monkeypatch.setattr(module.importlib, 'import_module', forbidden)
    monkeypatch.setattr(module, '_launch', forbidden)
    monkeypatch.setattr(torch.nn.Module, '_call_impl', forbidden)
    monkeypatch.setattr(torch.Tensor, 'backward', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(Trainer, 'fit', forbidden)
    for name in ('init', '_lazy_init', 'is_available', 'device_count', 'set_device', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def test_saved_recipe_survives_deleted_directory_without_filesystem_access(tmp_path, monkeypatch, no_execution):
    path = tmp_path / 'factory'
    path.mkdir()
    recipe = module.TrainingMemoryRecipe('absent_factory:build', factory_search_path=path)
    data = saved_result(path)
    data['recipe'] = recipe.to_dict()
    data['sha256'] = module._hash({k: v for k, v in data.items() if k != 'sha256'})
    path.rmdir()
    def forbidden(*args, **kwargs):
        pytest.fail('saved metadata must not resolve or query factory paths')
    with monkeypatch.context() as local:
        local.setattr(Path, 'resolve', forbidden)
        local.setattr(Path, 'is_dir', forbidden)
        result = module.IsolatedTrainingMemoryResult(data)
        assert result.to_dict() == data
        assert result._export_data() == data
        assert result.diagnostics()['core_status'] == 'not_executed'
    result.to_json(tmp_path / 'result.json')
    assert json.loads((tmp_path / 'result.json').read_text()) == data
    assert result.to_dict()['probe_id'] == data['probe_id']
    assert result.to_dict()['sha256'] == data['sha256']
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('admission', ['constructor', 'parent', 'worker'])
def test_missing_directory_rejected_before_execution(tmp_path, no_execution, admission):
    path = tmp_path / 'factory'
    path.mkdir()
    recipe = module.TrainingMemoryRecipe('absent_factory:build', factory_search_path=path)
    path.rmdir()
    with pytest.raises(ValueError, match='factory search directory missing'):
        if admission == 'constructor':
            module.TrainingMemoryRecipe(**recipe.to_dict())
        elif admission == 'parent':
            module.measure_training_memory_isolated(recipe)
        else:
            module._run_recipe(dict(recipe=recipe.to_dict()))


@pytest.mark.parametrize('field,value', [
    ('factory', '__main__:bad'), ('factory', 'invalid'), ('device', 'cuda'),
    ('seed', True), ('epochs', 0), ('kwargs', {'device': 'cpu'}),
    ('kwargs', {'value': float('inf')}), ('factory_search_path', 3),
    ('factory_search_path', ''), ('factory_search_path', 'relative/path'),
    ('factory_search_path', 'C:\\bad\x00path'), ('save_last_to', 3),
    ('execution', False), ('unexpected', 'value'),
])
def test_rehashed_invalid_recipe_still_rejected(tmp_path, no_execution, field, value):
    data = saved_result(tmp_path / 'missing')
    data['recipe'][field] = value
    if field == 'kwargs' and 'value' in value:
        # Nonfinite values are already rejected by canonical JSON hashing.
        with pytest.raises(ValueError):
            module._hash({k: v for k, v in data.items() if k != 'sha256'})
        with pytest.raises(ValueError):
            module.IsolatedTrainingMemoryResult(data)
        return
    data['sha256'] = module._hash({k: v for k, v in data.items() if k != 'sha256'})
    with pytest.raises(ValueError):
        module.IsolatedTrainingMemoryResult(data)


def test_saved_metadata_tamper_cannot_bypass_checksum(tmp_path, no_execution):
    data = saved_result(tmp_path / 'missing')
    tampered = copy.deepcopy(data)
    tampered['recipe']['epochs'] += 1
    with pytest.raises(ValueError, match='checksum mismatch'):
        module.IsolatedTrainingMemoryResult(tampered)
