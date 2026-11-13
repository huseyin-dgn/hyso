"""Standalone B3 native GPU exports/comparison. CUDA absence is a failure."""
import argparse
import csv
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--installed-target", type=Path)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(args.installed_target.resolve() if args.installed_target else ROOT))
import numpy as np
import torch
from torch import nn
from hyso.benchmark import BenchmarkRunner, BenchmarkResult, compare_results

ARTIFACTS = []


def run(*, precision="fp32", memory=False, batch=2, identity=False):
    model = nn.Identity() if identity else nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.BatchNorm2d(2), nn.AdaptiveAvgPool2d(1))
    model.cuda()
    return BenchmarkRunner(model, device="cuda:0", input_shape=(batch, 1, 7, 9), precision=precision,
                           warmup_iterations=1, measure_iterations=2, repeats=2, measure_cuda_memory=memory).run()


def export(result, directory):
    before = result.to_dict()
    directory.mkdir(parents=True)
    result.to_json(directory / "result.json")
    result.to_csv(directory / "summary.csv")
    result.to_samples_csv(directory / "samples.csv")
    loaded = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    assert loaded == before == result.to_dict()
    reconstructed = BenchmarkResult(loaded)
    assert compare_results(result, reconstructed)["status"] == "compatible"
    with (directory / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == 1 and summary[0]["cuda_memory_status"] == before["cuda_memory"]["status"]
    assert float(summary[0]["mean_ms"]) == before["stats"]["pooled"]["mean_ms"]
    with (directory / "samples.csv").open(encoding="utf-8", newline="") as handle:
        samples = list(csv.DictReader(handle))
    assert len(samples) == 4
    assert [float(x["latency_ms"]) for x in samples] == sum(before["timing"]["latency_samples_ms"], [])
    if before["cuda_memory"]["status"] == "measured":
        assert int(summary[0]["cuda_peak_allocated_bytes"]) == before["cuda_memory"]["peak_allocated_bytes"]
        assert any(w["code"] == "cuda_peak_reset" for w in loaded["diagnostics"]["warnings"])
    else:
        assert summary[0]["cuda_peak_allocated_bytes"] == ""
    ARTIFACTS.append(dict(directory=str(directory), result=before))


class NativeExportComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("native CUDA required; CPU fallback/skip forbidden")
        torch.cuda.set_device(0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cls.output = Path(tempfile.mkdtemp(prefix=args.output.stem + "-exports-", dir=args.output.parent))

    def test_fp32_measured_exports(self):
        export(run(memory=True), self.output / "fp32")

    def test_amp_measured_exports(self):
        export(run(precision="amp_fp16", memory=True), self.output / "amp")

    def test_disabled_exports(self):
        export(run(), self.output / "disabled")

    def test_different_native_architectures_compatible(self):
        report = compare_results(run(), run(identity=True))
        assert report["status"] == "compatible" and report["mismatched_fields"] == []
        assert not {"winner", "speedup", "ranking", "accuracy"} & set(report)

    def test_precision_mismatch_incompatible(self):
        report = compare_results(run(), run(precision="amp_fp16"))
        assert report["status"] == "incompatible"
        assert "configuration.precision" in [v["field"] for v in report["mismatched_fields"]]

    def test_batch_mismatch_incompatible(self):
        assert compare_results(run(batch=1), run(batch=3))["status"] == "incompatible"

    def test_memory_eligibility_separate(self):
        measured = run(memory=True)
        report = compare_results(run(), measured)
        assert report["status"] == "compatible" and report["memory_comparability"]["status"] == "not_comparable"
        report = compare_results(measured, measured)
        assert report["status"] == "compatible" and report["memory_comparability"]["status"] == "insufficient_metadata"
        assert not report["memory_comparability"]["absolute_peak_model_comparable"]

    def test_readonly_exports_no_runtime_calls_or_rng_change(self):
        result = run(memory=True)
        before = random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state(0)
        def forbidden(*a, **kw):
            raise AssertionError("export/comparison touched CUDA/model runtime")
        with patch.object(torch.cuda, "Event", forbidden), patch.object(torch.cuda, "is_available", forbidden), \
             patch.object(torch.cuda, "reset_peak_memory_stats", forbidden), patch.object(torch.cuda, "memory_allocated", forbidden), \
             patch.object(torch.cuda, "synchronize", forbidden), patch.object(nn.Sequential, "forward", forbidden):
            export(result, self.output / "readonly")
            assert compare_results(result, result)["status"] == "compatible"
        assert random.getstate() == before[0]
        assert np.array_equal(np.random.get_state()[1], before[1][1]) and np.random.get_state()[2:] == before[1][2:]
        assert torch.equal(torch.get_rng_state(), before[2]) and torch.equal(torch.cuda.get_rng_state(0), before[3])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NativeExportComparison)
    identities = [t.id() for t in suite]
    outcome = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    report = dict(tests_run=outcome.testsRun, passed=max(0,outcome.testsRun-len(outcome.failures)-len(outcome.errors)),
                  success=outcome.wasSuccessful(), failures=len(outcome.failures), errors=len(outcome.errors), identities=identities,
                  torch=str(torch.__version__), cuda_build=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  source=sys.modules[BenchmarkResult.__module__].__file__, artifacts=ARTIFACTS)
    if args.installed_target:
        assert Path(report["source"]).resolve().is_relative_to(args.installed_target.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("artifacts", "identities")}))
    sys.exit(0 if outcome.wasSuccessful() and outcome.testsRun else 1)
