"""Real CPU eager forwards and controlled-clock protocol/statistics checks."""
import hashlib

import pytest
import torch
from torch import nn

from hyso.benchmark import BenchmarkRunner, BenchmarkResult
from hyso.benchmark import runner as benchmark
from hyso.models.modular import ModularCNN


def network(dtype=torch.float32):
    return nn.Sequential(nn.Conv2d(1, 3, 3, padding=1), nn.BatchNorm2d(3), nn.Dropout(.5),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2)).to(dtype=dtype)


@pytest.mark.parametrize("precision,dtype", [("fp32", torch.float32), ("fp64", torch.float64)])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("attention", ["none", "se", "cbam", "eca", "coordinate"])
def test_native_modular_attention_cpu(precision, dtype, batch, attention):
    model = ModularCNN(input_channels=1, num_classes=2, widths=[3], repeats=[1],
                       strides=[1], attention=attention).to(dtype=dtype)
    result = BenchmarkRunner(model, input_shape=(batch, 1, 5, 7), precision=precision,
                             warmup_iterations=1, measure_iterations=2, repeats=2).run()
    assert isinstance(result, BenchmarkResult)
    data = result.to_dict()
    assert data["configuration"]["precision"] == precision
    assert data["input"]["batch_size"] == batch
    assert len(data["timing"]["latency_samples_ms"]) == 2
    assert all(len(row) == 2 and all(value > 0 for value in row) for row in data["timing"]["latency_samples_ms"])
    assert data["stats"]["pooled"]["count"] == 4
    assert data["model"]["output"]["shape"] == [batch, 2]
    assert data["parameters"]["total_parameters"] == sum(p.numel() for p in model.parameters())
    assert data["cuda_memory"]["status"] == "not_applicable"
    assert data["environment"]["cuda"]["value"] is None
    assert not torch.cuda.is_initialized()


def test_correct_clock_scope_forward_counts_and_known_statistics(monkeypatch):
    events = []
    class Observed(nn.Module):
        def forward(self, inputs):
            events.append("forward")
            assert not self.training and torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
            return inputs + 1
    ns = iter([0, 1_000_000, 2_000_000, 4_000_000, 5_000_000, 8_000_000, 9_000_000, 13_000_000])
    def clock():
        events.append("clock")
        return next(ns)
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", clock)
    data = BenchmarkRunner(Observed(), input_shape=(2, 1, 3, 5), warmup_iterations=2,
                           measure_iterations=2, repeats=2).run().to_dict()
    assert events == ["forward"] + (["forward"]*2 + ["clock", "forward", "clock"]*2)*2
    assert data["timing"]["latency_samples_ms"] == [[1., 2.], [3., 4.]]
    stats = data["stats"]["pooled"]
    assert stats["mean_ms"] == stats["median_ms"] == stats["p50_ms"] == 2.5
    assert stats["p90_ms"] == pytest.approx(3.7)
    assert stats["p95_ms"] == pytest.approx(3.85)
    assert stats["std_ms"] == pytest.approx((5/3)**.5)
    assert data["stats"]["per_repeat"][0]["mean_ms"] == 1.5
    assert data["throughput"]["images_per_sec"] == 800.
    assert data["throughput"]["per_repeat_images_per_sec"] == [4000/3, 4000/7]
    assert data["throughput"]["method"] == "derived_serial_service_rate"


def test_custom_input_copy_seed_hash_and_repeatable_runner():
    model = network()
    inputs = torch.randn(2, 1, 5, 7)
    before = inputs.clone()
    runner = BenchmarkRunner(model, example_input=inputs, measure_iterations=1, repeats=1, warmup_iterations=0)
    first, second = runner.run().to_dict(), runner.run().to_dict()
    assert torch.equal(inputs, before)
    assert first["input"] == second["input"]
    assert first["input"]["content_sha256"] == hashlib.sha256(inputs.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert first["identity"]["run_id"] != second["identity"]["run_id"]
    assert first["stats"]["pooled"]["std_ms"] is None
    assert first["stats"]["pooled"]["std_status"] == "insufficient_samples"
    assert first["diagnostics"]["warnings"][0]["code"] == "no_warmup"


@pytest.mark.parametrize("precision,dtype", [("fp32", torch.float32), ("fp64", torch.float64)])
def test_flatten_linear_cnn_and_local_generated_input(precision, dtype, monkeypatch):
    model = nn.Sequential(nn.Flatten(), nn.Linear(15, 2)).to(dtype=dtype)
    def forbidden(*args, **kwargs):
        raise AssertionError("global manual_seed is forbidden")
    monkeypatch.setattr(torch, "manual_seed", forbidden)
    runner = BenchmarkRunner(model, input_shape=(2, 1, 3, 5), precision=precision, seed=91,
                             measure_iterations=2, repeats=1, warmup_iterations=0)
    a, b = runner.run().to_dict(), runner.run().to_dict()
    assert a["input"] == b["input"]
    assert a["model"]["weights_sha256"] == b["model"]["weights_sha256"]
    assert a["model"]["output"]["shape"] == [2, 2]


def test_explicit_cpu_ignores_external_default_meta_device():
    runner = BenchmarkRunner(nn.Identity(), input_shape=(1, 1, 3, 5), warmup_iterations=0,
                             measure_iterations=1, repeats=1)
    with torch.device("meta"):
        data = runner.run().to_dict()
    assert data["input"]["device"] == "cpu" and data["model"]["output"]["device"] == "cpu"


@pytest.mark.parametrize("source", ["generated", "ordinary", "inference_tensor"])
def test_outer_inference_context_and_no_grad_inference_example_are_preserved(source):
    model = network()
    inputs = torch.ones(2, 1, 5, 7)
    if source == "inference_tensor":
        with torch.inference_mode():
            inputs = inputs.clone()
    options = dict(input_shape=(2, 1, 5, 7)) if source == "generated" else dict(example_input=inputs)
    with torch.inference_mode():
        result = BenchmarkRunner(model, **options, warmup_iterations=1, measure_iterations=1, repeats=1).run()
        assert torch.is_inference_mode_enabled()
    assert model.training and result.to_dict()["stats"]["pooled"]["count"] == 1
    assert torch.equal(inputs, torch.ones_like(inputs))


@pytest.mark.parametrize("options", [
    {"input_shape": (True, 1, 3, 5)}, {"input_shape": (1, 0, 3, 5)}, {"input_shape": (1, 3, 5)},
    {"input_shape": (1, 1, 3., 5)}, {"input_shape": "1,1,3,5"},
    {"precision": "amp_fp16"}, {"precision": "bf16"}, {"precision": True},
    {"device": "cuda"}, {"device": "meta"}, {"device": "cpu:0"}, {"device": 0},
    {"measure_cuda_memory": True}, {"measure_cuda_memory": 1},
    {"warmup_iterations": -1}, {"warmup_iterations": True}, {"measure_iterations": 0},
    {"measure_iterations": 1.5}, {"repeats": 0}, {"repeats": False},
    {"seed": -1}, {"seed": True}, {"seed": 2**63}, {"model_name": 3},
    {"model_config": {"bad": torch.tensor(1)}}, {"model_config": {"bad": float("nan")}},
])
def test_invalid_constructor_options_rejected_before_forward(options):
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        BenchmarkRunner(network(), **dict(dict(input_shape=(1, 1, 5, 7)), **options))


@pytest.mark.parametrize("kind", ["none", "both", "requires_grad", "noncontiguous", "nonfinite", "dtype", "nonmodule", "integer"])
def test_invalid_input_contract(kind):
    model, options = network(), {}
    if kind == "both": options = dict(input_shape=(1, 1, 5, 7), example_input=torch.zeros(1, 1, 5, 7))
    elif kind == "requires_grad": options = dict(example_input=torch.zeros(1, 1, 5, 7, requires_grad=True))
    elif kind == "noncontiguous": options = dict(example_input=torch.zeros(1, 1, 5, 7).transpose(2, 3))
    elif kind == "nonfinite": options = dict(example_input=torch.full((1, 1, 5, 7), float("inf")))
    elif kind == "dtype": options = dict(example_input=torch.zeros(1, 1, 5, 7, dtype=torch.float64))
    elif kind == "integer": options = dict(example_input=torch.zeros(1, 1, 5, 7, dtype=torch.long))
    elif kind == "nonmodule": model, options = object(), dict(input_shape=(1, 1, 5, 7))
    with pytest.raises((ValueError, TypeError)):
        BenchmarkRunner(model, **options)


@pytest.mark.parametrize("kind", ["half", "double", "nan", "meta", "sparse", "lazy", "parallel", "hook", "train_override", "scripted", "quantized", "buffer_inf"])
def test_unsafe_model_profiles_rejected(kind):
    model = network()
    handle = None
    if kind == "half": model.half()
    elif kind == "double": model.double()
    elif kind == "nan":
        with torch.no_grad(): next(model.parameters()).fill_(float("nan"))
    elif kind == "meta": model.to("meta")
    elif kind == "sparse": model.register_parameter("sparse", nn.Parameter(torch.ones(2, 2).to_sparse()))
    elif kind == "lazy": model = nn.LazyLinear(2)
    elif kind == "parallel": model = nn.DataParallel(model)
    elif kind == "scripted": model = torch.jit.script(nn.Identity())
    elif kind == "quantized": model = torch.ao.nn.quantized.Linear(5, 2)
    elif kind == "buffer_inf": model[1].running_var.fill_(float("inf"))
    elif kind == "hook": handle = model.register_forward_hook(lambda *args: None)
    elif kind == "train_override":
        class SideEffect(nn.Identity):
            def train(self, mode=True):
                raise AssertionError("must reject without executing custom train")
        model = SideEffect()
    try:
        with pytest.raises((ValueError, TypeError)):
            BenchmarkRunner(model, input_shape=(1, 1, 5, 7))
    finally:
        if handle is not None: handle.remove()


def test_global_hooks_are_rejected_before_they_execute():
    from torch.nn.modules.module import register_module_forward_hook
    calls = []
    handle = register_module_forward_hook(lambda *args: calls.append(True))
    try:
        with pytest.raises(ValueError, match="global Module hooks"):
            BenchmarkRunner(nn.Identity(), input_shape=(1, 1, 3, 5))
        assert not calls
    finally:
        handle.remove()


@pytest.mark.parametrize("kind", ["tuple", "integer", "scalar", "batch", "nan", "dynamic"])
def test_invalid_forward_outputs(kind):
    calls = []
    class Invalid(nn.Module):
        def forward(self, x):
            calls.append(True)
            if kind == "tuple": return (x,)
            if kind == "integer": return x.long()
            if kind == "scalar": return x.sum()
            if kind == "batch": return x[:0]
            if kind == "nan": return x * float("nan")
            return x if len(calls) == 1 else x[:, :, :1]
    with pytest.raises(ValueError):
        BenchmarkRunner(Invalid(), input_shape=(2, 1, 3, 5), warmup_iterations=1, measure_iterations=1, repeats=1).run()


@pytest.mark.parametrize("times", [[0, 0], [2, 1], [0, float("nan")], [0, float("inf")]])
def test_invalid_clock_readings_are_not_fabricated(monkeypatch, times):
    iterator = iter(times)
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", lambda: next(iterator))
    with pytest.raises(FloatingPointError, match="positive latency"):
        BenchmarkRunner(nn.Identity(), input_shape=(1, 1, 3, 5), warmup_iterations=0, measure_iterations=1, repeats=1).run()
