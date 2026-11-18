"""Basic P23 contract and bounded trained Model 1(A)/Model 2(B) CPU acceptance."""
import copy
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path

import pytest
import torch

from hyso.performance import training_memory as memory
from fixture_support import load_fixture
from p23_basic_validation import validate_pair, validate_failure


@pytest.mark.parametrize("arm", ["A", "B"], ids=["Model1", "Model2"])
def test_trained_models_cpu_basic_memory(arm):
    assets = os.environ.get("P23_BASIC_ASSETS")
    output = os.environ.get("P23_BASIC_OUTPUT")
    if not assets or not output:
        pytest.skip("real checkpoint/data inventory requires explicit P23_BASIC_ASSETS/OUTPUT")
    validate_pair(arm, assets, Path(output)/("cpu-"+arm))
    evidence = validate_failure(arm, assets, Path(output)/("cpu-failure-"+arm))
    (Path(output)/("cpu-failure-"+arm)/"validation.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")


def test_legacy_wire_and_new_basic_view_are_independent(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline basic projection reached CUDA or activation execution")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.autograd.graph, "saved_tensors_hooks", forbidden)
    data = load_fixture("training-memory-p23-schema1")
    result = memory.TrainingMemoryResult(data)
    basic = result.basic_metrics()
    assert basic["status"] == "complete"
    assert all(m["status"] == "not_applicable" and m["value"] is None and m["reason"]
               for k, m in basic["metrics"].items() if k.startswith("cuda_"))
    basic["metrics"]["parameter_bytes"]["value"] = -1
    assert result.to_dict() == data
    assert result.to_dict()["status"] == "partial"  # historical full-scope wire label


def test_missing_rss_and_unsupported_optimizer_do_not_become_basic_complete():
    data = load_fixture("training-memory-p23-schema1")
    rows = copy.deepcopy(data["observations"])
    for row in rows:
        row["logical"]["summary"]["optimizer"].update(status="unsupported", logical_bytes=None, reason="custom_optimizer")
    rss = dict(summary={key: dict(value=None, status="unsupported", reason="backend_disappeared")
                        for key in data["rss"]["summary"]})
    basic = memory._basic_memory_report(rows, rss, "cpu")
    assert basic["status"] == "partial"
    assert basic["metrics"]["optimizer_state_bytes"]["status"] == "unsupported"
    assert basic["metrics"]["process_rss_bytes"]["status"] == "unavailable"
    assert basic["metrics"]["process_rss_bytes"]["reason"] == "backend_disappeared"
    assert all(v["value"] is None for k, v in basic["metrics"].items()
               if k in ("optimizer_state_bytes", "process_rss_bytes") or k.startswith("cuda_"))


def test_basic_cuda_peak_projection_keeps_lifetime_and_owned_window_scopes():
    data = load_fixture("training-memory-p23-schema1")
    rows = copy.deepcopy(data["observations"])
    for row in rows:
        row["cuda"] = dict(allocated_bytes=10, reserved_bytes=100,
                           lifetime_peak_allocated_bytes=900, lifetime_peak_reserved_bytes=1000)
    lifetime = memory._basic_memory_report(rows, data["rss"], "cuda:0")
    assert lifetime["metrics"]["cuda_peak_allocated_bytes"]["value"] == 900
    assert lifetime["metrics"]["cuda_peak_allocated_bytes"]["scope"] == "process_device_lifetime"
    assert not lifetime["reset_performed"]
    windows = [dict(reset_performed=True, status="measured", peak_allocated_bytes=a, peak_reserved_bytes=r)
               for a, r in ((20, 200), (30, 150))]
    owned = memory._basic_memory_report(rows, data["rss"], "cuda:0", isolated=True, windows=windows)
    assert owned["metrics"]["cuda_peak_allocated_bytes"]["value"] == 30
    assert owned["metrics"]["cuda_peak_reserved_bytes"]["value"] == 200
    assert owned["metrics"]["cuda_peak_allocated_bytes"]["scope"] == "maximum_of_owned_training_windows"
    assert owned["reset_performed"]


@pytest.mark.parametrize("invalid", ["tuple", "decreasing_hwm"])
def test_invalid_native_rss_does_not_discard_independent_training_domains(monkeypatch, invalid):
    from hyso.training import Trainer
    from torch.utils.data import DataLoader, TensorDataset
    original = memory._observe
    calls = []
    def observe():
        calls.append(True)
        if invalid == "tuple":
            raise ValueError("inconsistent native memory observation/clock")
        sample = original()
        sample["lifetime_hwm_bytes"] = 2**50 if len(calls) == 1 else sample["lifetime_hwm_bytes"]
        return sample
    monkeypatch.setattr(memory, "_observe", observe)
    model = torch.nn.Linear(3, 2)
    trainer = Trainer(model, optimizer=torch.optim.AdamW(model.parameters()), loss_fn=torch.nn.CrossEntropyLoss())
    loader = DataLoader(TensorDataset(torch.ones(4, 3), torch.tensor([0, 1, 0, 1])), batch_size=2)
    result = memory.measure_training_memory(trainer, loader, epochs=1)
    basic = result.basic_metrics()
    assert basic["status"] == "partial"
    assert basic["metrics"]["parameter_bytes"]["status"] == "measured"
    assert basic["metrics"]["optimizer_state_bytes"]["value"] > 0
    assert basic["metrics"]["process_rss_bytes"]["value"] is None
    assert basic["metrics"]["process_rss_bytes"]["status"] == "unavailable"
    assert basic["metrics"]["process_rss_bytes"]["reason"]


@pytest.mark.skipif(os.name != "nt", reason="independent Windows native RSS readback")
def test_process_rss_matches_independent_native_working_set():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("peak_ws", "ws", "peak_paged", "paged",
                                               "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    api = ctypes.WinDLL("psapi", use_last_error=True)
    api.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    api.GetProcessMemoryInfo.restype = wintypes.BOOL
    def native():
        counters = Counters(cb=ctypes.sizeof(Counters))
        assert api.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters.ws, counters.peak_ws
    before = native()
    sample = memory._observe()
    after = native()
    # The three queries are consecutive; allow page changes from the observer.
    assert min(before[0], after[0])-1024**2 <= sample["rss_bytes"] <= max(before[0], after[0])+1024**2
    assert min(before[1], after[1]) <= sample["lifetime_hwm_bytes"] <= max(before[1], after[1])
