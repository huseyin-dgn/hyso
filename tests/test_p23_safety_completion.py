"""CPU-only fault injection for P23 bounds, provenance and metric domains."""
import copy
import threading
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from types import SimpleNamespace

from hyso.performance import training_memory as memory
from hyso.performance import isolated_training_memory as isolated
from hyso.performance._saved_activation import _SavedActivationRecorder


def test_parameterless_graph_still_has_a_traversal_budget():
    model = torch.nn.Sequential(*[torch.nn.Identity() for _ in range(10)])
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=.1)
    with pytest.raises(memory.TrainingMemoryError, match="module traversal"):
        memory._logical(model, optimizer, 5)


def test_blocked_sampler_shutdown_is_bounded_and_explicitly_failed():
    recorder = object.__new__(memory._MemoryRecorder)
    timeouts = []
    recorder.stop = threading.Event()
    recorder.errors = []
    recorder.thread = SimpleNamespace(ident=1, join=lambda timeout:timeouts.append(timeout), is_alive=lambda:True)
    recorder._stop_sampler()
    assert recorder.stop.is_set() and timeouts == [5.]
    assert recorder.errors == ['RSS_sampler_shutdown_timeout; native query not cancelled']


@pytest.mark.parametrize("kind", ["cycle", "depth", "nodes"])
def test_recipe_container_validation_is_bounded(kind):
    value = []
    if kind == "cycle": value.append(value)
    elif kind == "nodes": value.extend([None]*10001)
    else:
        for _ in range(34): value = [value]
    with pytest.raises(ValueError, match="budget"):
        isolated.TrainingMemoryRecipe("training_memory_factories:tiny", kwargs=dict(value=value))


@pytest.mark.parametrize("records,events", [(True, 10), (1, 0), (100001, 10), (1, 400001)])
def test_saved_recorder_validates_bounds_before_hook_setup(records, events, monkeypatch):
    monkeypatch.setattr(torch.autograd.graph, "saved_tensors_hooks", lambda *a: pytest.fail("hooks reached"))
    with pytest.raises(ValueError, match="budgets"):
        _SavedActivationRecorder(torch.nn.Linear(2, 2), records, events)


def test_worker_rejects_device_mismatch_before_any_cuda_query(monkeypatch):
    def forbidden(*a, **k): pytest.fail("CUDA reached before request validation")
    monkeypatch.setattr(torch.cuda, "is_initialized", forbidden)
    recipe = isolated.TrainingMemoryRecipe("training_memory_factories:tiny")
    protocol = isolated.IsolatedTrainingMemoryProtocol(memory_protocol=memory.TrainingMemoryProtocol(device="cuda:0"))
    with pytest.raises(ValueError, match="device mismatch"):
        isolated._run_recipe(dict(recipe=recipe.to_dict(), protocol=asdict(protocol), torch_version=str(torch.__version__)))


@pytest.fixture(scope="module")
def cpu_result():
    recipe = isolated.TrainingMemoryRecipe("training_memory_factories:tiny", factory_search_path=Path(__file__).parent,
                                          epochs=2, seed=71)
    result = isolated.measure_training_memory_isolated(recipe)
    assert result.to_dict()["outcome"] == "success", result.to_dict()
    return result


@pytest.mark.parametrize("damage", ["storage_total", "alias_extent", "duplicate_tensor", "cuda_on_cpu", "wrong_epoch"])
def test_rehashed_isolated_forgery_is_rejected(cpu_result, damage):
    data = cpu_result.to_dict()
    row = data["measurement"]["boundaries"][0]
    tensor = row["logical"]["tensors"][0]
    if damage == "storage_total": row["logical"]["unique_storage_bytes"] += 1
    elif damage == "alias_extent": tensor["storage_offset"] += tensor["storage_bytes"] + 1
    elif damage == "duplicate_tensor": row["logical"]["tensors"].append(copy.deepcopy(tensor))
    elif damage == "cuda_on_cpu": row["cuda"] = dict(allocated_bytes=0)
    else:
        row = next(x for x in data["measurement"]["boundaries"] if x["boundary"] == "post_forward")
        row["epoch_id"] = 999
    data["sha256"] = isolated._hash({k:v for k,v in data.items() if k != "sha256"})
    with pytest.raises(ValueError): isolated.IsolatedTrainingMemoryResult(data)


def test_diagnostics_keep_private_commit_rss_allocator_artifacts_separate(cpu_result):
    diagnostics = cpu_result.diagnostics()
    assert diagnostics["process_private_commit"]["unit"] == "byte"
    assert diagnostics["process_private_commit"]["peak_job_commit_bytes"] >= 0
    assert diagnostics["cuda_allocator"]["value"] is None
    assert diagnostics["cuda_allocator"]["reason"] == "CPU_training"
    assert diagnostics["artifact_bytes"]["value"] is None and diagnostics["artifact_bytes"]["reason"]
    assert diagnostics["activation"]["value"] is None
