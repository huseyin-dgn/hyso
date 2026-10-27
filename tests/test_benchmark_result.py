"""Immutable, plain finite schema-v1 snapshots with reproducible arithmetic."""
from dataclasses import FrozenInstanceError
import json

import pytest
import torch
from torch import nn

from hyso.benchmark import BenchmarkRunner, BenchmarkResult
from hyso.benchmark.result import _statistics


@pytest.fixture
def data():
    return BenchmarkRunner(nn.Identity(), input_shape=(1, 1, 3, 5), warmup_iterations=0,
                           measure_iterations=2, repeats=1, model_config={"label": ["fixture"]}).run().to_dict()


def test_result_is_immutable_defensive_plain_and_has_no_export_comparison(data):
    result = BenchmarkResult(data)
    data["timing"]["latency_samples_ms"][0][0] = -1
    original = result.to_dict()
    assert original["timing"]["latency_samples_ms"][0][0] > 0
    copy = result.to_dict()
    copy["model"]["model_config"]["label"].append("changed")
    assert result.to_dict()["model"]["model_config"]["label"] == ["fixture"]
    with pytest.raises(FrozenInstanceError): result._payload = "mutated"
    assert json.loads(json.dumps(original, allow_nan=False)) == original
    assert not hasattr(result, "model") and not hasattr(result, "compare_results")
    assert all(callable(getattr(result, method)) for method in ("to_json", "to_csv", "to_samples_csv"))
    def check(value):
        assert type(value) in (dict, list, str, bool, int, float, type(None))
        if type(value) is dict:
            assert all(type(key) is str for key in value)
            for item in value.values(): check(item)
        elif type(value) is list:
            for item in value: check(item)
    check(original)


@pytest.mark.parametrize("mutation", ["version", "type", "group", "unknown", "samples", "nan", "zero", "stats", "throughput", "batch", "params", "tensor", "object", "key", "tuple", "cuda", "empty_group", "bool_stats", "protocol", "missing_metric"])
def test_result_rejects_inconsistent_or_nonplain_schema(data, mutation):
    if mutation == "version": data["schema_version"] = True
    elif mutation == "type": data["benchmark_type"] = "training"
    elif mutation == "group": del data["environment"]
    elif mutation == "unknown": data["extra"] = None
    elif mutation == "samples": data["timing"]["latency_samples_ms"][0].pop()
    elif mutation == "nan": data["timing"]["latency_samples_ms"][0][0] = float("nan")
    elif mutation == "zero": data["timing"]["latency_samples_ms"][0][0] = 0
    elif mutation == "stats": data["stats"]["pooled"]["p95_ms"] += 1
    elif mutation == "throughput": data["throughput"]["method"] = "real_sustained_fps"
    elif mutation == "batch": data["input"]["batch_size"] = True
    elif mutation == "params": data["parameters"]["total_parameters"] = -1
    elif mutation == "tensor": data["model"]["model_config"] = {"tensor": torch.ones(1)}
    elif mutation == "object": data["model"]["model_config"] = {"model": nn.Identity()}
    elif mutation == "key": data["model"]["model_config"] = {1: "nonstring"}
    elif mutation == "tuple": data["model"]["model_config"] = {"tuple": (1, 2)}
    elif mutation == "cuda": data["cuda_memory"]["value"] = 0
    elif mutation == "empty_group": data["identity"] = {}
    elif mutation == "bool_stats": data["stats"]["std_ddof"] = True
    elif mutation == "protocol": data["configuration"]["memory_probe_count"] = 1
    elif mutation == "missing_metric": data["optional_metrics"]["cpu_rss_bytes"]["value"] = 0
    with pytest.raises((TypeError, ValueError)):
        BenchmarkResult(data)


@pytest.mark.parametrize("samples,expected", [([1.], (1., 1., 1., None)),
    ([1., 3.], (2., 2.8, 2.9, 2**.5)), ([2., 2., 2.], (2., 2., 2., 0.))])
def test_linear_quantiles_and_sample_stddev(samples, expected):
    stats = _statistics(samples)
    for key, value in zip(("p50_ms", "p90_ms", "p95_ms", "std_ms"), expected):
        if value is None: assert stats[key] is None and stats["std_status"] == "insufficient_samples"
        else: assert stats[key] == pytest.approx(value)
