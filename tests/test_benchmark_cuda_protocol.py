"""CPU doubles prove protocol wiring only, never native CUDA performance."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hyso.benchmark import BenchmarkRunner, BenchmarkResult
from hyso.benchmark import runner as implementation


@pytest.fixture
def protocol(monkeypatch):
    calls = []
    state = dict(device=7, stream="caller", elapsed=2.5, held=False)
    selected = object()

    @contextmanager
    def device(value):
        previous = state["device"]
        state["device"] = value.index
        try:
            yield
        finally:
            state["device"] = previous
            calls.append("restore_device")

    @contextmanager
    def stream(value):
        previous = state["stream"]
        state["stream"] = value
        try:
            yield
        finally:
            state["stream"] = previous
            calls.append("restore_stream")

    class Event:
        count = 0
        def __init__(self, *, enable_timing):
            assert enable_timing is True
            self.name = "start" if Event.count % 2 == 0 else "end"
            Event.count += 1
            calls.append("create_" + self.name)
        def record(self, value):
            assert value is selected
            calls.append("record_" + self.name)
        def synchronize(self):
            calls.append("sync_" + self.name)
        def elapsed_time(self, other):
            assert self.name == "start" and other.name == "end"
            calls.append("elapsed")
            return state["elapsed"]

    def selected_device(value):
        assert value == torch.device("cuda:0")
    def synchronize(value):
        selected_device(value)
        calls.append("device_sync")
    def get_rng(value):
        selected_device(value)
        calls.append("get_selected_rng")
        return torch.tensor([1], dtype=torch.uint8)
    def set_rng(value, target):
        selected_device(target)
        assert torch.equal(value, torch.tensor([1], dtype=torch.uint8))
        calls.append("restore_selected_rng")

    attrs = dict(is_available=lambda: True, device_count=lambda: 1, current_device=lambda: state["device"],
                 device=device, stream=stream, current_stream=lambda d: selected if state["stream"] == "caller" else state["stream"],
                 Event=Event, synchronize=synchronize, get_rng_state=get_rng, set_rng_state=set_rng,
                 get_allocator_backend=lambda: "native", get_device_capability=lambda d: (8, 6),
                 get_device_properties=lambda d: SimpleNamespace(name="PROTOCOL DOUBLE", total_memory=100000))
    # A stream identity is plain metadata; the identity object itself is never serialized.
    selected = SimpleNamespace(cuda_stream=123)
    for name, value in attrs.items():
        monkeypatch.setattr(torch.cuda, name, value)
    for name, value in dict(memory_allocated=100, memory_reserved=200, max_memory_allocated=150,
                            max_memory_reserved=250).items():
        def read(d, name=name, value=value):
            selected_device(d)
            calls.append(name)
            return value
        monkeypatch.setattr(torch.cuda, name, read)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d: (selected_device(d), calls.append("reset")))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: pytest.fail("empty_cache forbidden"))
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("all-device RNG forbidden"))
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda *a: pytest.fail("all-device RNG forbidden"))
    def autocast(**kwargs):
        state["autocast"] = kwargs
        return nullcontext()
    monkeypatch.setattr(torch, "autocast", autocast)
    tensor_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw: self if a and isinstance(a[0], torch.device)
                        and a[0].type == "cuda" else tensor_to(self, *a, **kw))
    profile, tensor, output = implementation._profile, implementation._tensor, implementation._output
    monkeypatch.setattr(implementation, "_profile", lambda m, dt, d: profile(m, dt))
    monkeypatch.setattr(implementation, "_tensor", lambda v, dt, **kw: tensor(v, dt, **{k: v for k, v in kw.items() if k != "device"}))
    def fake_output(value, batch, expected=None, device=None):
        descriptor = output(value, batch)
        descriptor["device"] = "cuda:0"
        assert expected is None or descriptor == expected
        return descriptor
    monkeypatch.setattr(implementation, "_output", fake_output)
    monkeypatch.setattr(implementation.time, "perf_counter_ns", lambda: pytest.fail("host clock in CUDA path"))
    return calls, state, selected


def run_double(protocol, *, memory=False, fail_at=None, precision="fp32"):
    calls, state, selected = protocol
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.count = 0
        def forward(self, x):
            self.count += 1
            calls.append("forward")
            assert state["device"] == 0 and state["stream"] is selected
            assert not self.training and torch.is_inference_mode_enabled()
            if self.count == fail_at:
                raise LookupError("forward failure")
            return x + 1
    model = Model()
    runner = BenchmarkRunner(model, device="cuda:0", input_shape=(2, 1, 3, 3), precision=precision,
                             warmup_iterations=2, measure_iterations=2, repeats=2, measure_cuda_memory=memory)
    return runner, model


@pytest.mark.parametrize("memory", [False, True])
@pytest.mark.parametrize("precision", ["fp32", "amp_fp16"])
def test_selected_stream_event_order_counts_and_memory_separation(protocol, memory, precision):
    calls, state, _ = protocol
    runner, model = run_double(protocol, memory=memory, precision=precision)
    data = runner.run().to_dict()
    assert model.count == 1 + 2 * (2 + 2) + int(memory)
    event_sample = ["record_start", "forward", "record_end", "sync_end", "elapsed"]
    expected = ["get_selected_rng", "forward"] + ["forward"] * 2
    expected += ["create_start", "create_end", "record_start", "record_end", "sync_end", "device_sync"]
    expected += event_sample * 2 + ["forward"] * 2 + ["device_sync"] + event_sample * 2
    assert calls[:len(expected)] == expected
    assert data["timing"]["latency_samples_ms"] == [[2.5, 2.5], [2.5, 2.5]]
    assert data["throughput"]["images_per_sec"] == 800
    assert data["throughput"]["source_timing_method"] == "cuda_event_elapsed"
    assert calls.count("reset") == int(memory)
    assert data["configuration"]["memory_probe_count"] == int(memory)
    assert data["configuration"]["autocast_enabled"] == (precision == "amp_fp16")
    assert state["autocast"] == dict(device_type="cuda", dtype=torch.float16, enabled=precision == "amp_fp16")
    assert calls[-3:] == ["restore_selected_rng", "restore_stream", "restore_device"]
    assert state["device"] == 7 and state["stream"] == "caller" and model.training
    if memory:
        start = calls.index("reset")
        assert calls[start-5:start] == ["device_sync", "memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"]
        assert calls[start+1:start+7] == ["forward", "device_sync", "max_memory_allocated", "max_memory_reserved", "memory_allocated", "memory_reserved"]
        assert data["cuda_memory"]["incremental_peak_bytes"] == 50
        assert data["diagnostics"]["warnings"][0]["counter_reset_performed"] is True
    else:
        assert data["cuda_memory"]["status"] == "disabled"
        assert all(data["cuda_memory"][key] is None for key in implementation._MEMORY_FIELDS)


@pytest.mark.parametrize("fail_at", [1, 2, 4, 10])
def test_exception_cleanup_and_lock_reuse(protocol, fail_at):
    runner, model = run_double(protocol, memory=True, fail_at=fail_at)
    expected = RuntimeError if fail_at == 10 else LookupError
    with pytest.raises(expected):
        runner.run()
    calls, state, _ = protocol
    assert calls[-3:] == ["restore_selected_rng", "restore_stream", "restore_device"]
    assert state["device"] == 7 and state["stream"] == "caller" and model.training
    runner.run()


@pytest.mark.parametrize("elapsed", [0, -1, float("nan"), float("inf"), True])
def test_invalid_event_duration_rejected_without_clamping(protocol, elapsed):
    protocol[1]["elapsed"] = elapsed
    runner, _ = run_double(protocol)
    with pytest.raises(ValueError, match="finite and positive"):
        runner.run()
    assert protocol[0][-3:] == ["restore_selected_rng", "restore_stream", "restore_device"]


@pytest.mark.parametrize("field,value", [("incremental_peak_bytes", -1), ("peak_allocated_bytes", True),
                                       ("counter_reset_performed", False), ("scope", "timed_forward"),
                                       ("previous_peak_reserved_bytes", None)])
def test_measured_memory_schema_rejects_incoherence(protocol, field, value):
    runner, _ = run_double(protocol, memory=True)
    data = runner.run().to_dict()
    data["cuda_memory"][field] = value
    with pytest.raises(ValueError):
        BenchmarkResult(data)


def test_disabled_schema_and_timing_mismatch_rejected(protocol):
    runner, _ = run_double(protocol)
    data = runner.run().to_dict()
    data["cuda_memory"]["peak_allocated_bytes"] = 0
    with pytest.raises(ValueError):
        BenchmarkResult(data)
    data = runner.run().to_dict()
    data["timing"]["timing_method"] = "perf_counter_ns"
    with pytest.raises(ValueError):
        BenchmarkResult(data)


@pytest.mark.parametrize("kwargs", [dict(precision="fp64"), dict(device="cuda:1"), dict(device="mps"),
                                   dict(measure_cuda_memory=1), dict(precision="bf16")])
def test_invalid_cuda_config_on_cpu_double(protocol, kwargs):
    config = dict(device="cuda:0", input_shape=(1, 1, 2, 2))
    config.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        BenchmarkRunner(nn.Identity(), **config)


def test_cuda_unavailable_rejected_without_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        BenchmarkRunner(nn.Identity(), device="cuda", input_shape=(1, 1, 2, 2))


def test_probe_holds_then_releases_output_before_final_read(protocol):
    calls, _, _ = protocol
    class Output:
        def __del__(self):
            calls.append("release_output")
    def forward():
        calls.append("probe_forward")
        return Output()
    def validate(output):
        assert type(output) is Output and "release_output" not in calls
        calls.append("validate")
    implementation._cuda_memory_probe(torch.device("cuda:0"), forward, validate)
    assert calls == ["device_sync", "memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved",
                     "reset", "probe_forward", "device_sync", "max_memory_allocated", "max_memory_reserved",
                     "memory_allocated", "memory_reserved", "validate", "release_output", "device_sync",
                     "memory_allocated", "memory_reserved"]


def test_memory_reset_warning_required_by_schema(protocol):
    runner, _ = run_double(protocol, memory=True)
    data = runner.run().to_dict()
    data["diagnostics"]["warnings"] = []
    with pytest.raises(ValueError, match="diagnostics"):
        BenchmarkResult(data)


@pytest.mark.parametrize("group,key,value", [("input", "dtype", "torch.float16"), ("environment", "device", "cuda:1"),
                                           ("model", "output", {"device": "cpu"})])
def test_cuda_residency_schema_mismatch(protocol, group, key, value):
    runner, _ = run_double(protocol)
    data = runner.run().to_dict()
    data[group][key] = value
    with pytest.raises(ValueError):
        BenchmarkResult(data)


@pytest.mark.parametrize("group,key,value", [("environment", "cuda", None), ("model", "output", []),
                                           ("diagnostics", "warnings", [0])])
def test_malformed_cuda_metadata_rejected(protocol, group, key, value):
    runner, _ = run_double(protocol, memory=True)
    data = runner.run().to_dict()
    data[group][key] = value
    with pytest.raises(ValueError):
        BenchmarkResult(data)
