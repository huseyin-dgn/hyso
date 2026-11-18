"""CPU witness budget regression, with no CUDA initialization."""
import pytest
import torch

from hyso.performance import ProfilingProtocol
from hyso.performance._cpu_layer_memory import CPUAllocationRecorder


def make_recorder(*, event_budget=20, witness_budget=40):
    module = torch.nn.Module()
    inventory = [dict(module_id="m0", aliases=[""])]
    recorder = CPUAllocationRecorder(torch, torch.device("cpu"), event_budget,
                                     [("", module)], inventory, witness_budget=witness_budget)
    return recorder, module


def test_protocol_keeps_profiler_and_witness_budgets_separate():
    protocol = ProfilingProtocol(profile_memory=True, allocation_lifetimes=True,
                                 cpu_memory_ownership=True)
    assert protocol.max_events == 20000
    assert protocol.max_cpu_witness_items == 250000
    for invalid in (0, -1, False, 1.5, 500001):
        with pytest.raises((ValueError, TypeError)):
            ProfilingProtocol(max_cpu_witness_items=invalid)


def test_boundary_keeps_repeated_full_snapshots(monkeypatch):
    from hyso.performance import memory
    monkeypatch.setattr(memory, "_observe", lambda: dict(rss_bytes=1))
    recorder, module = make_recorder(event_budget=3, witness_budget=9)
    sample = torch.ones(2)
    try:
        recorder.begin(0, (sample,))
        recorder.end(0, sample, [("", module)])
        evidence = recorder.result()["cpu_context"]
        assert [p["phase"] for p in evidence["points"]] == ["entry", "exit"]
        assert len(evidence["points"][0]["storages"]) == 1
        assert len(evidence["points"][0]["tensors"]) == 1
        assert len(evidence["points"][0]["references"]) == 1
        assert len(evidence["points"][1]["references"]) == 2
        assert recorder.witness_items == 7
        assert recorder.witness_items > recorder.budget
    finally:
        recorder.close()


def test_overflow_reports_breakdown_and_keeps_boundary_unpublished(monkeypatch):
    from hyso.performance import memory
    observed = []
    monkeypatch.setattr(memory, "_observe", lambda: observed.append(True))
    recorder, _ = make_recorder(event_budget=10, witness_budget=2)
    sample = torch.ones(2)
    try:
        with pytest.raises(ValueError, match="CPU boundary witness evidence budget exceeded") as captured:
            recorder.begin(0, (sample,))
        failure = str(captured.value)
        for item in ("invocation_id=0", "phase=entry", "storages=1", "tensors=1",
                     "edges=1", "accumulated_before=0", "projected=3", "limit=2"):
            assert item in failure
        assert recorder.cpu_points == []
        assert observed == []
    finally:
        recorder.close()


def test_invalid_direct_witness_budget_rejected():
    for invalid in (0, False, "20", 500001):
        with pytest.raises(ValueError, match="invalid CPU witness evidence budget"):
            make_recorder(witness_budget=invalid)
