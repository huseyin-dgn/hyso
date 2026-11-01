"""Standalone native GPU gate. No CUDA is a failure, never a skip/fallback.

Run with the repository runtime: python -B tests/cuda_benchmark_validation.py
Optional --installed-target validates an isolated wheel outside the checkout.
"""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest
import weakref
from unittest.mock import patch

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path)
parser.add_argument("--installed-target", type=Path)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(args.installed_target.resolve() if args.installed_target else ROOT))

import numpy as np
import torch
from torch import nn
from hyso.benchmark import BenchmarkRunner, BenchmarkResult
from hyso.benchmark.runner import StateMutationError, _Snapshot, _profile
from hyso.models.modular import ModularCNN

RESULTS = []


def network():
    return nn.Sequential(nn.Conv2d(1, 3, 3, padding=1), nn.BatchNorm2d(3), nn.Dropout(.5),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2)).cuda()


def rng():
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state(0)


def assert_rng(before):
    after = rng()
    assert before[0] == after[0]
    assert before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1]) and before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2]) and torch.equal(before[3], after[3])


def state(model):
    tensors = list(model.parameters()) + list(model.buffers())
    return ([(t, t.data_ptr(), t._version, t.detach().cpu().clone()) for t in tensors],
            [(p, p.grad, None if p.grad is None else p.grad.cpu().clone()) for p in model.parameters()],
            [(m, m.training) for m in model.modules()])


def assert_state(before):
    for t, pointer, version, value in before[0]:
        assert t.data_ptr() == pointer and t._version == version and torch.equal(t.cpu(), value)
    for p, gradient, value in before[1]:
        assert p.grad is gradient
        if gradient is not None:
            assert torch.equal(gradient.cpu(), value)
    for module, mode in before[2]:
        assert module.training == mode


def run_case(model, *, precision="fp32", batch=2, memory=False, stream=None, outer=False):
    autocast_before, inference_before = torch.is_autocast_enabled(), torch.is_inference_mode_enabled()
    for i, module in enumerate(model.modules()):
        module.training = bool(i % 2)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    inputs = torch.randn(batch, 1, 7, 9, device="cuda:0")
    input_copy, before, random_before = inputs.cpu().clone(), state(model), rng()
    device_before, stream_before = torch.cuda.current_device(), torch.cuda.current_stream(0)
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        selected = torch.cuda.current_stream(0)
        # Spy delegates to the real CUDA allocator API, preserving native behavior.
        with patch.object(torch.cuda, "reset_peak_memory_stats", wraps=torch.cuda.reset_peak_memory_stats) as reset:
            with torch.inference_mode() if outer else nullcontext():
                result = BenchmarkRunner(model, device="cuda", example_input=inputs, precision=precision,
                                         warmup_iterations=1, measure_iterations=3, repeats=2,
                                         measure_cuda_memory=memory).run()
            assert reset.call_count == int(memory)
        assert torch.cuda.current_stream(0) == selected
    assert torch.cuda.current_device() == device_before and torch.cuda.current_stream(0) == stream_before
    assert_state(before)
    assert_rng(random_before)
    assert torch.equal(inputs.cpu(), input_copy)
    assert torch.is_autocast_enabled() == autocast_before and torch.is_inference_mode_enabled() == inference_before
    data = result.to_dict()
    assert BenchmarkResult(data).to_dict() == data
    assert data["environment"]["device"] == "cuda:0" and data["input"]["dtype"] == "torch.float32"
    assert data["timing"]["timing_method"] == "cuda_event_elapsed"
    assert data["timing"]["synchronization_policy"] == "repeat_device_drain_and_end_event_sync"
    assert data["environment"]["cuda"]["stream_id"] == selected.cuda_stream
    samples = data["timing"]["latency_samples_ms"]
    assert len(samples) == 2 and all(len(row) == 3 for row in samples)
    flat = sum(samples, [])
    assert all(math.isfinite(x) and x > 0 for x in flat)
    assert math.isclose(data["throughput"]["images_per_sec"], batch * 6 * 1000 / math.fsum(flat), rel_tol=1e-15)
    assert all(p.dtype == torch.float32 and p.device == torch.device("cuda:0") for p in model.parameters())
    if memory:
        mem = data["cuda_memory"]
        assert mem["status"] == "measured" and mem["counter_reset_performed"] is True
        assert mem["peak_allocated_bytes"] >= mem["baseline_allocated_bytes"]
        assert mem["incremental_peak_bytes"] == mem["peak_allocated_bytes"] - mem["baseline_allocated_bytes"] >= 0
        assert mem["current_allocated_after_probe_bytes"] > mem["current_allocated_after_output_release_bytes"]
        assert any(w["code"] == "cuda_peak_reset" for w in data["diagnostics"]["warnings"])
    else:
        assert data["cuda_memory"]["status"] == "disabled" and data["cuda_memory"]["peak_allocated_bytes"] is None
    RESULTS.append(data)
    return data


class NativeBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("FAIL: native CUDA GPU required; CPU fallback forbidden")
        torch.cuda.set_device(0)

    def test_fp32_conv_bn_memory_and_nondefault_stream(self):
        run_case(network(), memory=True, stream=torch.cuda.Stream(device=0))

    def test_amp_native_autocast_conv_output(self):
        data = run_case(network(), precision="amp_fp16", memory=True)
        assert data["model"]["output"]["dtype"] == "torch.float16"

    def test_outer_inference_context(self):
        run_case(network(), precision="amp_fp16", outer=True)

    def test_generated_input_repeatability_and_disabled_peak_history(self):
        model = network()
        before, before_rng = state(model), rng()
        runner = BenchmarkRunner(model, device=torch.device("cuda:0"), input_shape=(1, 1, 7, 9), seed=91,
                                 warmup_iterations=0, measure_iterations=2, repeats=1, precision="amp_fp16")
        sentinel = torch.ones(1024 * 1024, device="cuda:0")
        del sentinel
        old_peak = torch.cuda.max_memory_allocated(0)
        a, b = runner.run().to_dict(), runner.run().to_dict()
        assert a["input"] == b["input"] and a["input"]["source"] == "generated"
        assert a["cuda_memory"]["status"] == b["cuda_memory"]["status"] == "disabled"
        assert torch.cuda.max_memory_allocated(0) >= old_peak
        assert any(w["code"] == "no_warmup" for w in a["diagnostics"]["warnings"])
        assert_state(before)
        assert_rng(before_rng)
        RESULTS.extend((a, b))

    def test_exact_forward_count_stream_rng_and_cpu_snapshots(self):
        stream = torch.cuda.Stream(device=0)
        class Observed(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1, device="cuda:0"))
                self.count = 0
                self.previous = None
            def forward(self, x):
                assert self.previous is None or self.previous() is None
                self.count += 1
                assert torch.cuda.current_stream(0) == stream
                random.random()
                np.random.random()
                torch.rand(1)
                torch.rand(1, device="cuda:0")
                output = x * self.weight
                self.previous = weakref.ref(output)
                return output
        model = Observed()
        run_case(model, stream=stream, memory=True)
        assert model.count == 1 + 2 * (1 + 3) + 1
        modules, entries = _profile(model, torch.float32, torch.device("cuda:0"))
        snapshot = _Snapshot(modules, entries, torch.ones(1, 1, 3, 3, device="cuda:0"))
        assert all(saved.device.type == "cpu" for _, _, saved, _ in snapshot.tensors)
        assert all(saved is None or saved.device.type == "cpu" for _, _, saved, _ in snapshot.gradients)
        assert snapshot.input_copy.device.type == "cpu"

    def test_exception_cleanup(self):
        class Failing(nn.Module):
            def forward(self, x):
                torch.rand(1, device=x.device)
                raise LookupError("native forward failure")
        model = Failing()
        before = rng()
        stream = torch.cuda.Stream(device=0)
        previous = torch.cuda.current_stream(0)
        with torch.cuda.stream(stream):
            with self.assertRaisesRegex(LookupError, "native forward failure"):
                BenchmarkRunner(model, device="cuda:0", input_shape=(1, 1, 3, 3), precision="amp_fp16").run()
            assert torch.cuda.current_stream(0) == stream
        assert torch.cuda.current_stream(0) == previous and model.training
        assert_rng(before)
        assert not torch.is_autocast_enabled() and not torch.is_inference_mode_enabled()
        run_case(network())  # process lock was released

    def test_value_mutation_rejected_rolled_back(self):
        class Mutating(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(1, device="cuda:0"))
            def forward(self, x):
                self.buffer.add_(1)
                return x + self.buffer
        model = Mutating()
        before = rng()
        with self.assertRaises(StateMutationError):
            BenchmarkRunner(model, device="cuda:0", input_shape=(1, 1, 3, 3)).run()
        assert model.buffer.item() == 1 and model.training
        assert_rng(before)

    def test_stream_switch_rejected_and_restored(self):
        other = torch.cuda.Stream(device=0)
        class Switching(nn.Module):
            def forward(self, x):
                torch.cuda.set_stream(other)
                return x + 1
        before = torch.cuda.current_stream(0)
        with self.assertRaises(StateMutationError):
            BenchmarkRunner(Switching(), device="cuda:0", input_shape=(1, 1, 3, 3)).run()
        assert torch.cuda.current_stream(0) == before

    def test_parameter_topology_mutation_fatal(self):
        class Moving(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1, device="cuda:0"))
            def forward(self, x):
                self.weight.data = self.weight.data.double()
                return x
        with self.assertRaisesRegex(RuntimeError, "rollback failed"):
            BenchmarkRunner(Moving(), device="cuda:0", input_shape=(1, 1, 3, 3)).run()

    def test_alias_topology_and_outer_autocast_preserved(self):
        class Tied(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1, device="cuda"))
                self.tied = self.weight
                self.register_buffer("view", self.weight.detach().view(1))
            def forward(self, x):
                return x * self.weight
        model = Tied()
        with torch.autocast("cuda", dtype=torch.float16):
            run_case(model)
            assert torch.is_autocast_enabled()
        assert model.weight is model.tied and model.view.data_ptr() == model.weight.data_ptr()


def failure_case(fail_at):
    def test(self):
        class Failing(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = network()
                self.count = 0
            def forward(self, x):
                self.count += 1
                if self.count == fail_at:
                    random.random()
                    np.random.random()
                    torch.rand(1)
                    torch.rand(1, device=x.device)
                    raise LookupError("stage failure")
                return self.layer(x)
        model = Failing()
        model.layer[1].eval()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        before, before_rng = state(model), rng()
        stream = torch.cuda.Stream(device=0)
        with torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.float16):
            with self.assertRaisesRegex(LookupError, "stage failure"):
                BenchmarkRunner(model, device="cuda:0", input_shape=(2, 1, 7, 9), precision="amp_fp16",
                                warmup_iterations=1, measure_iterations=2, repeats=1).run()
            assert torch.cuda.current_stream(0) == stream and torch.is_autocast_enabled()
        assert_state(before)
        assert_rng(before_rng)
    return test


for fail_at in (1, 2, 3, 4):
    setattr(NativeBenchmark, f"test_exception_stage_{fail_at}", failure_case(fail_at))


def modular_case(attention, precision, batch):
    def test(self):
        model = ModularCNN(input_channels=1, num_classes=2, widths=[3], repeats=[1],
                           strides=[1], attention=attention).cuda()
        run_case(model, precision=precision, batch=batch)
    return test


for attention in ("none", "se", "cbam", "eca", "coordinate"):
    for precision in ("fp32", "amp_fp16"):
        for batch in (1, 3):
            setattr(NativeBenchmark, f"test_modular_{attention}_{precision}_batch{batch}", modular_case(attention, precision, batch))


def invalid_case(kind):
    def test(self):
        model = network()
        config = dict(device="cuda:0", input_shape=(1, 1, 3, 3))
        if kind == "half":
            model.half()
        elif kind == "cpu_model":
            model.cpu()
        elif kind == "mixed":
            model.register_buffer("cpu_buffer", torch.ones(1))
        elif kind == "fp64":
            config["precision"] = "fp64"
        elif kind == "device_index":
            config["device"] = "cuda:" + str(torch.cuda.device_count())
        elif kind == "cpu_amp":
            model.cpu()
            config.update(device="cpu", precision="amp_fp16")
        elif kind == "input_half":
            del config["input_shape"]
            config["example_input"] = torch.ones(1, 1, 3, 3, device="cuda", dtype=torch.float16)
        elif kind == "meta":
            model = nn.Linear(1, 1, device="meta")
        elif kind == "lazy":
            model = nn.LazyConv2d(2, 3).cuda()
        elif kind == "sparse":
            model.register_buffer("sparse", torch.ones(2, 2, device="cuda").to_sparse())
        elif kind == "quantized":
            model = nn.Module()
            model.register_buffer("quantized", torch.quantize_per_tensor(torch.ones(2), .1, 0, torch.qint8))
        with self.assertRaises((TypeError, ValueError)):
            BenchmarkRunner(model, **config)
    return test


for kind in ("half", "cpu_model", "mixed", "fp64", "device_index", "cpu_amp", "input_half", "meta", "lazy", "sparse", "quantized"):
    setattr(NativeBenchmark, "test_reject_" + kind, invalid_case(kind))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NativeBenchmark)
    identities = [test.id() for test in suite]
    outcome = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                capture_output=True, text=True, check=False).stdout.strip()
    except FileNotFoundError:
        driver = "nvidia-smi unavailable"
    report = dict(tests_run=outcome.testsRun, failures=len(outcome.failures), errors=len(outcome.errors),
                  passed=max(0, outcome.testsRun-len(outcome.failures)-len(outcome.errors)), success=outcome.wasSuccessful(),
                  identities=identities,
                  python=sys.version, torch=str(torch.__version__), cuda_build=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  driver=driver, benchmark_source=sys.modules[BenchmarkRunner.__module__].__file__,
                  results=RESULTS)
    if args.installed_target:
        assert Path(report["benchmark_source"]).resolve().is_relative_to(args.installed_target.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))
    sys.exit(0 if outcome.wasSuccessful() and outcome.testsRun else 1)
