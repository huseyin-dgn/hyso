"""Native state/RNG preservation and unsupported-mutation failure boundaries."""
import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from hyso.benchmark import BenchmarkRunner
from hyso.benchmark import runner as benchmark
from hyso.benchmark.runner import StateMutationError
from test_benchmark_cpu import network


def state(model):
    return dict(tensors={name: value.detach().clone() for name, value in model.state_dict().items()},
                parameters=[(id(p), p.data_ptr(), p._version) for p in model.parameters()],
                grads=[(id(p.grad), None if p.grad is None else p.grad.clone()) for p in model.parameters()],
                modes=[m.training for m in model.modules()], python=random.getstate(),
                numpy=copy.deepcopy(np.random.get_state()), torch=torch.get_rng_state())


def equal(a, b):
    if isinstance(b, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(b, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(b, dict):
        assert list(a) == list(b)
        for key in b: equal(a[key], b[key])
    elif isinstance(b, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else:
        assert a == b


def prepared():
    model = network()
    model.eval()
    model[1].train()
    model[-1].train()
    for p in model.parameters():
        p.grad = torch.full_like(p, .375)
    random.seed(92)
    np.random.seed(93)
    torch.manual_seed(94)
    return model


@pytest.mark.parametrize("outer_grad", [False, True])
def test_native_bn_dropout_gradients_modes_rng_and_threads_preserved(outer_grad, monkeypatch):
    model = prepared()
    before = state(model)
    settings = (torch.get_num_threads(), torch.get_num_interop_threads(), torch.backends.mkldnn.enabled,
                torch.are_deterministic_algorithms_enabled(), torch.get_float32_matmul_precision())
    monkeypatch.setattr(torch.Tensor, "backward", lambda *args, **kwargs: pytest.fail("no backward"))
    with torch.set_grad_enabled(outer_grad), torch.autocast("cpu", dtype=torch.bfloat16):
        BenchmarkRunner(model, input_shape=(2, 1, 5, 7), warmup_iterations=2, measure_iterations=3, repeats=2).run()
        assert torch.is_grad_enabled() is outer_grad
        assert torch.is_autocast_cpu_enabled()
    equal(state(model), before)
    assert settings == (torch.get_num_threads(), torch.get_num_interop_threads(), torch.backends.mkldnn.enabled,
                        torch.are_deterministic_algorithms_enabled(), torch.get_float32_matmul_precision())


@pytest.mark.parametrize("when", [1, 2, 4])
def test_preflight_warmup_measured_exceptions_restore_rng_gradients_and_modes(when):
    calls = []
    class Failure(nn.Sequential):
        def forward(self, inputs):
            calls.append(True)
            random.random(); np.random.random(); torch.rand(1)
            if len(calls) == when:
                raise LookupError("native forward failure")
            return super().forward(inputs)
    model = Failure(*list(prepared().children()))
    model.eval(); model[1].train()
    before = state(model)
    with pytest.raises(LookupError, match="native forward failure"):
        BenchmarkRunner(model, input_shape=(2, 1, 5, 7), warmup_iterations=2, measure_iterations=2, repeats=1).run()
    assert len(calls) == when
    equal(state(model), before)


@pytest.mark.parametrize("kind", ["parameter", "buffer", "gradient", "input", "reference", "key", "mode", "data_write"])
@pytest.mark.parametrize("when", [1, 4])
def test_mutation_is_rejected_and_best_effort_restores_caller_values(kind, when):
    calls = []
    class Mutating(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.weight.grad = torch.full_like(self.weight, .25)
            self.register_buffer("counter", torch.tensor(7))
        def forward(self, x):
            calls.append(True)
            if len(calls) == when:
                if kind == "parameter": self.weight.add_(1)
                elif kind == "buffer": self.counter.add_(1)
                elif kind == "gradient": self.weight.grad.add_(1)
                elif kind == "input": x.add_(1)
                elif kind == "reference": self.weight = nn.Parameter(torch.zeros(1))
                elif kind == "key": self.register_buffer("extra", torch.ones(1))
                elif kind == "mode": self.training = True
                elif kind == "data_write": self.weight.data.add_(1)
            return x * self.weight
    model = Mutating()
    original_weight, original_gradient = model.weight, model.weight.grad
    x = torch.ones(2, 1, 3, 5)
    rng = (random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state())
    with pytest.raises(StateMutationError, match="mutat|changed"):
        BenchmarkRunner(model, example_input=x, warmup_iterations=2, measure_iterations=2, repeats=1).run()
    assert model.weight is original_weight and model.weight.grad is original_gradient
    assert torch.equal(model.weight, torch.ones(1)) and torch.equal(model.weight.grad, torch.full((1,), .25))
    assert model.counter.item() == 7 and "extra" not in model._buffers and model.training
    assert torch.equal(x, torch.ones_like(x))
    equal((random.getstate(), np.random.get_state(), torch.get_rng_state()), rng)


def test_rollback_failure_is_fatal_and_lock_modes_rng_are_still_released(monkeypatch):
    class Bad(nn.Module):
        def __init__(self):
            super().__init__(); self.weight = nn.Parameter(torch.ones(1))
        def forward(self, x):
            self.weight.add_(1)
            return x
    model = Bad()
    rng = torch.get_rng_state()
    with monkeypatch.context() as patch:
        patch.setattr(benchmark._Snapshot, "restore_values", lambda *args: (_ for _ in ()).throw(RuntimeError("rollback fault")))
        with pytest.raises(RuntimeError, match="rollback failed.*runtime may be mutated"):
            BenchmarkRunner(model, input_shape=(1, 1, 3, 5)).run()
    assert model.training and torch.equal(torch.get_rng_state(), rng)
    BenchmarkRunner(nn.Identity(), input_shape=(1, 1, 3, 5), warmup_iterations=0, measure_iterations=1, repeats=1).run()


def test_reentrant_call_is_rejected_without_disturbing_outer_state():
    class Reentrant(nn.Module):
        def forward(self, x):
            BenchmarkRunner(nn.Identity(), example_input=x).run()
            return x
    model = Reentrant()
    before = state(model)
    with pytest.raises(RuntimeError, match="concurrent/reentrant"):
        BenchmarkRunner(model, input_shape=(1, 1, 3, 5)).run()
    equal(state(model), before)


def test_object_alias_and_storage_view_footprint_are_logical_not_rss():
    class Aliased(nn.Module):
        def __init__(self):
            super().__init__()
            self.first, self.second = nn.Linear(3, 3, bias=False), nn.Linear(3, 3, bias=False)
            self.second.weight = self.first.weight
            self.first.weight.requires_grad_(False)
            self.register_buffer("weight_buffer", self.first.weight)
            self.register_buffer("view", self.first.weight.detach().view(-1))
            self.register_buffer("counter", torch.tensor(0))
        def forward(self, x):
            return self.second(self.first(x.mean((2, 3))))
    model = Aliased()
    aliases = (model.first.weight, model.second.weight, model.weight_buffer, model.view.data_ptr())
    data = BenchmarkRunner(model, input_shape=(2, 3, 3, 5), warmup_iterations=0,
                           measure_iterations=1, repeats=1).run().to_dict()
    counts = data["parameters"]
    assert counts["total_parameters"] == counts["frozen_parameters"] == 9
    assert counts["trainable_parameters"] == 0
    assert counts["parameter_bytes"] == 36 and counts["buffer_bytes"] == 80
    assert counts["parameter_buffer_bytes"] == 80
    assert counts["cross_category_object_alias"] and counts["distinct_objects_share_storage"]
    assert model.first.weight is aliases[0] is model.second.weight is model.weight_buffer
    assert aliases[3] == model.view.data_ptr()
    assert all(data["optional_metrics"][key]["value"] is None for key in ("cpu_rss_bytes", "optimizer_memory_bytes", "activation_memory_bytes"))
