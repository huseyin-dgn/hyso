"""Compatibility decisions, never hardware speed/ranking assertions."""
import copy
import json

import pytest
import torch
from torch import nn

from hyso.benchmark import BenchmarkRunner, BenchmarkResult, compare_results
from hyso.benchmark.result import _statistics, _throughput
from test_benchmark_export import cpu_result, cuda_data


def set_field(data, key, value):
    keys = key.split(".")
    for name in keys[:-1]:
        data = data[name]
    data[keys[-1]] = value


def cpu_data():
    data = cpu_result().to_dict()
    # Known test metadata; no claim that this fixture identifies real hardware.
    data["environment"]["cpu"].update(machine="fixture_arch", processor="fixture_CPU", logical_cpu_count=4)
    data["environment"]["hyso_version"] = "0.2.0"
    return data


def rebuild(data):
    data = copy.deepcopy(data)
    config = data["configuration"]
    raw = data["timing"]["latency_samples_ms"]
    if len(raw) != config["repeats"] or any(len(row) != config["measure_iterations"] for row in raw):
        raw = [[raw[0][0] + r + i for i in range(config["measure_iterations"])] for r in range(config["repeats"])]
        data["timing"]["latency_samples_ms"] = raw
    data["stats"] = dict(per_repeat=[_statistics(row) for row in raw], pooled=_statistics(sum(raw, [])),
                         quantile_method="linear_(n-1)*q", std_ddof=1)
    data["throughput"]["images_per_sec"] = _throughput(sum(raw, []), data["input"]["batch_size"])
    data["throughput"]["per_repeat_images_per_sec"] = [_throughput(row, data["input"]["batch_size"]) for row in raw]
    return BenchmarkResult(data)


@pytest.mark.parametrize("device", ["cpu", "cuda_disabled", "cuda_measured"])
def test_same_recorded_profile_compatible_with_separate_memory_status(device):
    data = cpu_data()
    if device != "cpu":
        data = cuda_data(data, measured=device == "cuda_measured")
    a, b = BenchmarkResult(data), BenchmarkResult(data)
    report = compare_results(a, b)
    assert report["status"] == report["timing_comparability"]["status"] == "compatible"
    assert report["mismatched_fields"] == report["missing_required_fields"] == []
    assert len(report["compared_fields"]) >= 39
    assert report["memory_comparability"]["status"] == dict(cpu="not_applicable", cuda_disabled="not_comparable",
                                                           cuda_measured="insufficient_metadata")[device]
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize("path,value", [
    ("configuration.warmup_iterations", 3), ("configuration.measure_iterations", 3), ("configuration.repeats", 3),
    ("input.shape", [2, 1, 5, 5]), ("input.stride", [99, 15, 5, 1]), ("input.content_sha256", "f"*64),
    ("environment.python_version", "3.10.99"), ("environment.torch_version", "2.2.2+cpu"),
    ("environment.hyso_version", "different_engine"), ("environment.operating_system", "DifferentOS"),
    ("environment.platform", "DifferentPlatform"), ("environment.cpu.machine", "different_arch"),
    ("environment.cpu.processor", "different_CPU"), ("environment.cpu.logical_cpu_count", 8),
    ("environment.cpu.threads", 3), ("environment.cpu.interop_threads", 999),
    ("environment.backend.mkldnn_enabled", False), ("environment.backend.deterministic_algorithms", True),
    ("environment.backend.deterministic_warn_only", True), ("environment.backend.float32_matmul_precision", "medium"),
])
def test_cpu_condition_mismatch_reports_exact_field(path, value):
    data = cpu_data()
    # Fix the counterpart even when caller defaults differ across runtimes.
    if type(value) is bool:
        set_field(data, path, not value)
    elif path == "environment.backend.float32_matmul_precision":
        set_field(data, path, "highest")
    changed = copy.deepcopy(data)
    set_field(changed, path, value)
    report = compare_results(rebuild(data), rebuild(changed))
    assert report["status"] == "incompatible"
    assert path in [item["field"] for item in report["mismatched_fields"]]


@pytest.mark.parametrize("key,value", [("name", "other_GPU"), ("capability", [9, 0]),
    ("total_device_memory_bytes", 999999), ("cuda_build", "12.2"), ("cudnn_version", 80000),
    ("cudnn_benchmark", True), ("cudnn_deterministic", True), ("cudnn_allow_tf32", False),
    ("matmul_allow_tf32", True), ("allocator_backend", "cudaMallocAsync")])
def test_cuda_hardware_runtime_backend_mismatch(key, value):
    data = cuda_data(cpu_data())
    changed = copy.deepcopy(data)
    changed["environment"]["cuda"][key] = value
    report = compare_results(BenchmarkResult(data), BenchmarkResult(changed))
    assert report["status"] == "incompatible"
    assert "environment.cuda."+key in [x["field"] for x in report["mismatched_fields"]]


def test_cpu_cuda_event_host_methods_are_incompatible():
    data = cpu_data()
    report = compare_results(BenchmarkResult(data), BenchmarkResult(cuda_data(data)))
    assert report["status"] == "incompatible"
    fields = {x["field"] for x in report["mismatched_fields"]}
    assert {"environment.device", "timing.timing_method", "timing.timing_scope", "timing.synchronization_policy",
            "throughput.source_timing_method"} <= fields


@pytest.mark.parametrize("kind", ["cpu_fp64", "cuda_amp", "batch"])
def test_precision_and_batch_changes_incompatible(kind):
    data = cpu_data()
    if kind == "cuda_amp":
        data = cuda_data(data)
        changed = cuda_data(data, precision="amp_fp16")
    else:
        changed = copy.deepcopy(data)
        if kind == "cpu_fp64":
            changed["configuration"]["precision"] = "fp64"
            changed["input"]["dtype"] = changed["model"]["output"]["dtype"] = "torch.float64"
        else:
            changed["input"]["shape"][0] = changed["input"]["batch_size"] = changed["model"]["output"]["shape"][0] = 3
    report = compare_results(rebuild(data), rebuild(changed))
    assert report["status"] == "incompatible"
    assert any(x["field"] == ("input.batch_size" if kind == "batch" else "configuration.precision") for x in report["mismatched_fields"])


@pytest.mark.parametrize("path,value", [("environment.cpu.processor", None), ("environment.cpu.processor", ""),
    ("environment.cpu.processor", "unknown"), ("environment.cpu.machine", ""), ("environment.cpu.threads", None),
    ("environment.cpu.processor", " Unknown "), ("input.content_sha256", "not_a_hash"),
    ("environment.cpu.processor", 123), ("environment.cpu.logical_cpu_count", True),
    ("environment.backend.mkldnn_enabled", "False"), ("input.content_sha256", ""),
    ("environment.python_version", None), ("environment.platform", None)])
def test_unknown_or_invalid_required_metadata_is_insufficient(path, value):
    data = cpu_data()
    set_field(data, path, value)
    result = BenchmarkResult(data)
    report = compare_results(result, result)
    assert report["status"] == "insufficient_metadata"
    assert report["mismatched_fields"] == []
    assert any(x["field"] == path and x["sides"] == ["left", "right"] for x in report["missing_required_fields"])


@pytest.mark.parametrize("key", ["name", "capability", "total_device_memory_bytes", "cudnn_version", "cudnn_benchmark", "allocator_backend"])
def test_unknown_cuda_identity_is_insufficient(key):
    data = cuda_data(cpu_data())
    data["environment"]["cuda"][key] = None
    report = compare_results(BenchmarkResult(data), BenchmarkResult(data))
    assert report["status"] == "insufficient_metadata"
    assert report["missing_required_fields"][0]["field"] == "environment.cuda." + key


def test_definite_mismatch_takes_precedence_over_missing_metadata():
    data = cpu_data()
    changed = copy.deepcopy(data)
    changed["environment"]["cpu"]["processor"] = None
    changed["configuration"]["warmup_iterations"] += 1
    report = compare_results(BenchmarkResult(data), BenchmarkResult(changed))
    assert report["status"] == "incompatible"
    assert report["mismatched_fields"][0]["field"] == "configuration.warmup_iterations"
    assert report["missing_required_fields"][0]["field"] == "environment.cpu.processor"


@pytest.mark.parametrize("key,value", [("model_name", "another_name"), ("qualified_class", "another.Model"),
    ("model_config", {"width": 99}), ("weights_sha256", "a"*64), ("source_sha256", "b"*64)])
def test_model_identity_is_an_informational_variable(key, value):
    data = cpu_data()
    changed = copy.deepcopy(data)
    changed["model"][key] = value
    report = compare_results(BenchmarkResult(data), BenchmarkResult(changed))
    assert report["status"] == "compatible"
    assert report["informational"]["right"]["model"][key] == value


def test_different_native_model_architectures_comparable_when_conditions_match():
    a = BenchmarkRunner(nn.Identity(), input_shape=(2, 1, 3, 5), warmup_iterations=0, repeats=1, measure_iterations=2).run()
    b = BenchmarkRunner(nn.Conv2d(1, 2, 1), input_shape=(2, 1, 3, 5), warmup_iterations=0, repeats=1, measure_iterations=2).run()
    left, right = a.to_dict(), b.to_dict()
    for data in (left, right):
        data["environment"]["cpu"].update(machine="fixture_arch", processor="fixture_CPU")
    report = compare_results(BenchmarkResult(left), BenchmarkResult(right))
    assert report["status"] == "compatible"
    assert left["model"]["qualified_class"] != right["model"]["qualified_class"]
    assert left["parameters"]["total_parameters"] != right["parameters"]["total_parameters"]
    assert not {"winner", "speedup", "accuracy", "ranking"} & set(report)


def test_optional_provenance_input_origin_and_source_hash_do_not_block():
    data = cpu_data()
    changed = copy.deepcopy(data)
    changed["provenance"] = dict(module_source="different/path.py", benchmark_source="installed/result.py",
                                  source_revision=None, revision_status="not_collected")
    changed["identity"]["run_id"] = "different_run"
    changed["input"]["source"] = "custom_clone"
    changed["input"]["seed"] = None
    assert compare_results(BenchmarkResult(data), BenchmarkResult(changed))["status"] == "compatible"


def test_memory_disabled_measured_does_not_break_timing_compatibility():
    data = cpu_data()
    report = compare_results(BenchmarkResult(cuda_data(data)), BenchmarkResult(cuda_data(data, measured=True)))
    assert report["status"] == "compatible" and report["memory_comparability"]["status"] == "not_comparable"
    assert report["memory_comparability"]["observations"]["left"]["status"] == "disabled"
    assert report["memory_comparability"]["observations"]["right"]["status"] == "measured"


def test_memory_baseline_warning_and_explicit_isolation_requirement():
    data = cuda_data(cpu_data(), measured=True)
    changed = copy.deepcopy(data)
    changed["cuda_memory"]["baseline_allocated_bytes"] = 90
    changed["cuda_memory"]["incremental_peak_bytes"] = 60
    report = compare_results(BenchmarkResult(data), BenchmarkResult(changed))
    assert report["status"] == "compatible"
    mem = report["memory_comparability"]
    assert mem["status"] == "insufficient_metadata" and not mem["absolute_peak_model_comparable"]
    assert {x["code"] for x in mem["warnings"]} >= {"memory_baseline_differs", "memory_isolation_unverified", "allocator_scope_only"}
    for row in (data, changed):
        row["provenance"]["memory_isolation"] = "exclusive_fresh_process"
    annotated = compare_results(BenchmarkResult(data), BenchmarkResult(changed))
    assert annotated["memory_comparability"]["status"] == "compatible"
    assert not annotated["memory_comparability"]["absolute_peak_model_comparable"]


def test_nonfresh_isolation_label_insufficient_with_explicit_missing_reason():
    data = cuda_data(cpu_data(), measured=True)
    data["provenance"]["memory_isolation"] = "exclusive_process"
    report = compare_results(BenchmarkResult(data), BenchmarkResult(data))
    assert report["status"] == "compatible"
    assert report["memory_comparability"]["status"] == "insufficient_metadata"
    assert report["memory_comparability"]["missing_required_fields"][0]["reason"] == "fresh_process_isolation_not_attested"


def test_mutating_inputs_and_report_cannot_change_results():
    data = cpu_data()
    a = BenchmarkResult(data)
    data["input"]["shape"][0] = 999
    before = a.to_dict()
    report = compare_results(a, a)
    report["compared_fields"][0]["left"] = "mutated"
    report["informational"]["left"]["model"]["model_config"] = {"changed": True}
    assert a.to_dict() == before and compare_results(a, a)["status"] == "compatible"


def test_valid_deep_model_annotation_survives_report_nesting():
    config = None
    for _ in range(30):
        config = {"nested": config}
    data = cpu_data()
    data["model"]["model_config"] = config
    result = BenchmarkResult(data)
    report = compare_results(result, result)
    assert report["status"] == "compatible"
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report["informational"]["left"]["model"]["model_config"] == config


@pytest.mark.parametrize("value", [None, {}, "file.json", nn.Identity(), torch.tensor(1)])
def test_only_benchmark_results_accepted(value):
    result = BenchmarkResult(cpu_data())
    with pytest.raises(TypeError):
        compare_results(result, value)
    with pytest.raises(TypeError):
        compare_results(value, result)
