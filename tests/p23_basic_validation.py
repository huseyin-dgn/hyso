"""Bounded native basic-memory validation shared by CPU tests and CUDA runner."""
import json
from pathlib import Path
import random
import threading
from unittest.mock import patch

import numpy as np
import torch

from hyso.performance import (TrainingMemoryProtocol, TrainingMemoryResult, TrainingMemoryRecipe,
    IsolatedTrainingMemoryProtocol, IsolatedTrainingMemoryResult, measure_training_memory,
    measure_training_memory_isolated)
from hyso.performance import training_memory as memory
from hyso.performance import isolated_training_memory as isolated
from p23_basic_factories import trained_pair
from trainer_replay_support import equal, snapshot
from training_memory_factories import state_digest

SEED = 1337


def seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def no_saved_hooks(*args, **kwargs):
    raise AssertionError("P23 basic validation entered experimental saved hooks")


def check_cleanup(trainer):
    assert not memory._RUN_LOCK.locked()
    assert not hasattr(trainer, "_timing_recorder") and not hasattr(trainer, "_stage_recorder")
    assert not [t for t in threading.enumerate() if t.name == "hyso-training-rss"]


def check_basic(result, device, expected_parameters):
    basic = result.basic_metrics()
    assert basic["status"] == "complete", basic
    metrics = basic["metrics"]
    assert metrics["parameter_bytes"]["value"] == expected_parameters
    assert metrics["gradient_bytes"]["sampled_max_bytes"] == expected_parameters
    assert metrics["gradient_bytes"]["baseline_bytes"] == metrics["gradient_bytes"]["value"] == 0
    assert metrics["optimizer_state_bytes"]["baseline_bytes"] == 0 < metrics["optimizer_state_bytes"]["value"]
    assert metrics["process_sampled_peak_rss_bytes"]["value"] >= metrics["process_rss_bytes"]["value"] > 0
    assert metrics["process_lifetime_peak_rss_bytes"]["value"] >= metrics["process_sampled_peak_rss_bytes"]["value"]
    for name in ("cuda_allocated_bytes", "cuda_reserved_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        assert metrics[name]["status"] == ("not_applicable" if device == "cpu" else "measured")
        assert (metrics[name]["value"] is None) == (device == "cpu")
    return basic


def validate_pair(arm, assets_path, output, device="cpu", *, run_isolated=True, isolated_limits=None):
    output = Path(output)
    output.mkdir()
    config = json.loads(Path(assets_path).read_bytes())["models"][arm]
    expected_bytes = config["parameters"] * 4
    states = []
    native_logical = memory._logical
    oracle_calls = []
    def checked_logical(model, optimizer, budget):
        observed = native_logical(model, optimizer, budget)
        # Native live objects are an independent oracle at EVERY observed boundary.
        expected = dict(parameter=sum(p.numel()*p.element_size() for p in model.parameters()),
            gradient=sum(p.grad.numel()*p.grad.element_size() for p in model.parameters() if p.grad is not None),
            optimizer=sum(v.numel()*v.element_size() for state in optimizer.state.values()
                          for v in state.values() if isinstance(v, torch.Tensor)))
        for key, value in expected.items():
            assert observed["summary"][key]["logical_bytes"] == value
        if device != "cpu":
            # Readbacks are consecutive, with no tensor allocation in between.
            current = memory._cuda(torch.device(device))
            assert current["allocated_bytes"] == torch.cuda.memory_allocated(device)
            assert current["reserved_bytes"] == torch.cuda.memory_reserved(device)
            assert current["lifetime_peak_allocated_bytes"] == torch.cuda.max_memory_allocated(device)
            assert current["lifetime_peak_reserved_bytes"] == torch.cuda.max_memory_reserved(device)
        oracle_calls.append(expected)
        return observed
    for observed in (False, True):
        seed()
        work = trained_pair(device=device, arm=arm, assets_path=str(assets_path))
        trainer = work["trainer"]
        with patch.object(torch.autograd.graph, "saved_tensors_hooks", no_saved_hooks), \
                patch.object(isolated, "_activation_adapter", no_saved_hooks):
            if observed:
                with patch.object(memory, "_logical", checked_logical), \
                        patch.object(torch.cuda, "reset_peak_memory_stats", no_saved_hooks):
                    result = measure_training_memory(trainer, work["train_loader"], work["val_loader"], epochs=2,
                        protocol=TrainingMemoryProtocol(device=device))
            else:
                trainer.fit(work["train_loader"], work["val_loader"], epochs=2)
        states.append(snapshot(trainer))
        check_cleanup(trainer)
    result.to_json(output/"in-process.json")
    result.to_csv(output/"in-process.csv")
    equal(states[0], states[1])
    basic = check_basic(result, device, expected_bytes)
    assert not basic["reset_performed"]
    payload = result.to_dict()
    assert len(oracle_calls) == len(payload["observations"])
    assert payload["rss"]["summary"]["sampled_peak_rss_bytes"]["value"] == max(r["rss_bytes"] for r in payload["rss"]["observations"])
    assert payload["rss"]["summary"]["current_rss_bytes"]["value"] == payload["rss"]["observations"][-1]["rss_bytes"]
    assert TrainingMemoryResult(json.loads((output/"in-process.json").read_bytes())).to_dict() == payload
    (output/"basic.json").write_text(json.dumps(basic, indent=2), encoding="utf-8")
    evidence = dict(arm=arm, device=device, checkpoint_sha256=config["sha256"], parameters=config["parameters"],
        parameter_bytes=expected_bytes, exact_training_state_rng=True, oracle_boundaries=len(oracle_calls),
        in_process=basic, isolated=None)
    if run_isolated:
        # Parent live training/RNG/allocator state must be preserved by the worker.
        parent = snapshot(trainer)
        allocator = dict(torch.cuda.memory_stats(device)) if device != "cpu" else None
        path = output/"isolated-state.json"
        recipe = TrainingMemoryRecipe("p23_basic_factories:trained_pair", device=device,
            factory_search_path=Path(__file__).parent, seed=SEED, epochs=2,
            kwargs=dict(arm=arm, assets_path=str(assets_path), state_path=str(path), require_no_activation_adapter=True))
        protocol = IsolatedTrainingMemoryProtocol(process_memory_bytes=4*1024**3, job_memory_bytes=8*1024**3,
            cuda_memory_fraction=.15, timeout_seconds=60, saved_activation_lifetime=False,
            allow_peak_reset=device != "cpu", memory_protocol=TrainingMemoryProtocol(device=device))
        if isolated_limits is not None:
            from dataclasses import replace
            protocol = replace(protocol, **isolated_limits)
        isolated_result = measure_training_memory_isolated(recipe, protocol=protocol)
        data = isolated_result.to_dict()
        isolated_result.to_json(output/"isolated.json")
        assert data["outcome"] == "success", data
        if allocator is not None:
            assert allocator == dict(torch.cuda.memory_stats(device))
        equal(parent, snapshot(trainer))
        assert data["receipt"]["job"]["active_processes"] == 0
        assert json.loads(path.read_bytes()) == dict(state_sha256=state_digest(trainer), fits=1)
        ibasic = check_basic(isolated_result, device, expected_bytes)
        assert ibasic["reset_performed"] == (device != "cpu")
        if device != "cpu":
            for window in data["measurement"]["windows"]:
                assert window["reset_owner_pid"] == data["receipt"]["launcher_pid"]
                for kind in ("allocated", "reserved"):
                    assert window["after_reset"]["lifetime_peak_"+kind+"_bytes"] == window["after_reset"][kind+"_bytes"]
                    assert window["peak_"+kind+"_bytes"] >= window["end"][kind+"_bytes"]
                    assert ibasic["metrics"]["cuda_peak_"+kind+"_bytes"]["scope"] == "maximum_of_owned_training_windows"
        isolated_result.to_csv(output/"isolated.csv")
        assert IsolatedTrainingMemoryResult(json.loads((output/"isolated.json").read_bytes())).to_dict() == data
        evidence["isolated"] = ibasic
    (output/"validation.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence


def validate_failure(arm, assets_path, output, device="cpu", *, isolated_limits=None):
    output = Path(output)
    output.mkdir()
    seed()
    work = trained_pair(device=device, arm=arm, assets_path=str(assets_path), failure=True)
    try:
        with patch.object(torch.autograd.graph, "saved_tensors_hooks", no_saved_hooks):
            measure_training_memory(work["trainer"], work["train_loader"], epochs=1,
                                    protocol=TrainingMemoryProtocol(device=device))
    except RuntimeError as error:
        assert "controlled forward failure" in str(error)
    else:
        raise AssertionError("injected failure was not propagated")
    check_cleanup(work["trainer"])
    recipe = TrainingMemoryRecipe("p23_basic_factories:trained_pair", device=device, epochs=1,
        factory_search_path=Path(__file__).parent, kwargs=dict(arm=arm, assets_path=str(assets_path), failure=True,
                                                              require_no_activation_adapter=True))
    protocol = IsolatedTrainingMemoryProtocol(
        process_memory_bytes=4*1024**3, job_memory_bytes=8*1024**3, cuda_memory_fraction=.15,
        memory_protocol=TrainingMemoryProtocol(device=device))
    if isolated_limits is not None:
        from dataclasses import replace
        protocol = replace(protocol, **isolated_limits)
    result = measure_training_memory_isolated(recipe, protocol=protocol)
    data = result.to_dict()
    result.to_json(output/"failure.json")
    assert data["outcome"] == "workload_exception" and data["measurement"] is None, data
    assert data["receipt"]["job"]["active_processes"] == 0
    basic = result.basic_metrics()
    assert basic["status"] == "not_executed"
    assert all(v["value"] is None and v["reason"] for v in basic["metrics"].values())
    return dict(arm=arm, device=device, error_propagated=True, sampler_lock_recorder_cleanup=True,
                owned_active_processes=0, failed_measurements_null=True)
