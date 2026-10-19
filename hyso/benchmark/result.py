"""Immutable, finite, plain-data inference results. No file export in B1."""
from dataclasses import dataclass
import json
import math
import statistics


def _plain(value, depth=0):
    if depth > 32:
        raise ValueError("plain-data nesting exceeds 32 levels")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("plain data requires finite floats")
        return value
    if type(value) is list:
        return [_plain(item, depth + 1) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _plain(item, depth + 1) for key, item in value.items()}
    raise TypeError("plain data requires string-keyed dict/list and native finite scalars")


def _statistics(samples):
    if not samples or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("latencies must be finite and positive")
    ordered = sorted(samples)
    def quantile(q):
        position = (len(ordered) - 1) * q
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    try:
        result = dict(count=len(samples), mean_ms=statistics.fmean(samples),
                      median_ms=quantile(.5), p50_ms=quantile(.5), p90_ms=quantile(.9),
                      p95_ms=quantile(.95), min_ms=min(samples), max_ms=max(samples),
                      std_ms=statistics.stdev(samples) if len(samples) > 1 else None,
                      std_status="measured" if len(samples) > 1 else "insufficient_samples")
        return _plain(result)
    except (OverflowError, ValueError) as error:
        raise ValueError("latency statistics exceed finite numeric range") from error


def _throughput(samples, batch):
    try:
        rate = batch * len(samples) * 1000 / math.fsum(samples)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("invalid throughput")
        return rate
    except (OverflowError, ZeroDivisionError) as error:
        raise ValueError("throughput exceeds finite numeric range") from error


_GROUPS = {"identity", "model", "input", "configuration", "environment", "provenance", "timing",
           "stats", "throughput", "parameters", "cuda_memory", "optional_metrics", "diagnostics"}

_REQUIRED = {
    "identity": {"run_id", "started_at_utc", "finished_at_utc"},
    "model": {"model_name", "qualified_class", "model_config", "descriptor_version", "structural_descriptor",
              "weights_sha256", "source_sha256", "source_hash_status", "source_hash_reason", "output"},
    "input": {"shape", "stride", "dtype", "layout", "batch_size", "device", "source", "seed", "content_sha256"},
    "configuration": {"precision", "warmup_iterations", "measure_iterations", "repeats", "seed",
                      "preflight_forward_count", "memory_probe_count", "forced_model_mode", "inference_mode",
                      "autocast_enabled", "protocol_version", "input_policy"},
    "environment": {"python_version", "torch_version", "hyso_version", "torchvision_version", "torchvision_status",
                    "operating_system", "platform", "cpu", "device", "cuda", "backend"},
    "provenance": {"module_source", "benchmark_source", "source_revision", "revision_status"},
    "timing": {"timing_method", "timing_scope", "synchronization_policy", "clock_resolution_ns", "latency_samples_ms"},
    "diagnostics": {"warnings", "state_guard", "limitations"},
}


def _validate(data):
    if type(data) is not dict or set(data) != _GROUPS | {"schema_version", "benchmark_type"}:
        raise ValueError("invalid inference result schema fields")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1 or data["benchmark_type"] != "inference":
        raise ValueError("unsupported inference result schema")
    if any(type(data[key]) is not dict for key in _GROUPS):
        raise ValueError("result groups must be dictionaries")
    if any(not fields <= set(data[group]) for group, fields in _REQUIRED.items()):
        raise ValueError("missing required result fields")
    config, timing, inputs = data["configuration"], data["timing"], data["input"]
    for key, minimum in (("warmup_iterations", 0), ("measure_iterations", 1), ("repeats", 1)):
        if type(config.get(key)) is not int or config[key] < minimum:
            raise ValueError("invalid measurement counts")
    shape = inputs.get("shape")
    if (type(shape) is not list or len(shape) != 4 or any(type(n) is not int or n < 1 for n in shape)
            or type(inputs.get("batch_size")) is not int or inputs["batch_size"] != shape[0]):
        raise ValueError("invalid input/batch schema")
    if (timing.get("timing_method") != "perf_counter_ns" or timing.get("timing_scope") != "synchronous_host_forward"
            or config.get("precision") not in ("fp32", "fp64") or timing["synchronization_policy"] != "none"
            or inputs["device"] != "cpu" or data["environment"]["device"] != "cpu"):
        raise ValueError("unsupported B1 timing/precision schema")
    expected_config = dict(preflight_forward_count=1, memory_probe_count=0, forced_model_mode="eval",
                           inference_mode=True, autocast_enabled=False, protocol_version=1, input_policy="fixed_resident_input")
    if any(type(config[k]) is not type(v) or config[k] != v for k, v in expected_config.items()):
        raise ValueError("invalid inference protocol configuration")
    if type(config["seed"]) is not int or not 0 <= config["seed"] <= 2**63-1:
        raise ValueError("invalid result seed")
    nested = timing.get("latency_samples_ms")
    if type(nested) is not list or len(nested) != config["repeats"] or any(
            type(row) is not list or len(row) != config["measure_iterations"] for row in nested):
        raise ValueError("latency sample counts disagree with configuration")
    flat = [sample for row in nested for sample in row]
    expected_stats = dict(per_repeat=[_statistics(row) for row in nested], pooled=_statistics(flat),
                          quantile_method="linear_(n-1)*q", std_ddof=1)
    if json.dumps(data["stats"], sort_keys=True) != json.dumps(expected_stats, sort_keys=True):
        raise ValueError("statistics disagree with samples")
    expected_rate = dict(images_per_sec=_throughput(flat, shape[0]), method="derived_serial_service_rate",
                         source_timing_method="perf_counter_ns", per_repeat_images_per_sec=[_throughput(row, shape[0]) for row in nested],
                         window_wall_images_per_sec=None, window_status="unsupported", window_reason="separate_protocol_deferred")
    if json.dumps(data["throughput"], sort_keys=True) != json.dumps(expected_rate, sort_keys=True):
        raise ValueError("throughput disagrees with samples")
    if data["cuda_memory"] != {"status": "not_applicable", "value": None, "reason": "cpu_only_b1"}:
        raise ValueError("B1 has no CUDA memory measurement")
    params = data["parameters"]
    for key in ("total_parameters", "trainable_parameters", "frozen_parameters", "parameter_bytes", "buffer_bytes", "parameter_buffer_bytes"):
        if type(params.get(key)) is not int or params[key] < 0:
            raise ValueError("invalid parameter/byte count")
    if params["trainable_parameters"] + params["frozen_parameters"] != params["total_parameters"]:
        raise ValueError("incoherent trainable/frozen count")
    if not max(params["parameter_bytes"], params["buffer_bytes"]) <= params["parameter_buffer_bytes"] <= params["parameter_bytes"] + params["buffer_bytes"]:
        raise ValueError("incoherent logical byte accounting")
    for metric in data["optional_metrics"].values():
        if (type(metric) is not dict or metric.get("value", 0) is not None
                or metric.get("status") not in ("unsupported", "not_applicable", "not_collected")
                or type(metric.get("reason")) is not str):
            raise ValueError("optional metric must have an explicit missing status/reason")


@dataclass(frozen=True, init=False)
class BenchmarkResult:
    """Schema-v1 result; internally an immutable JSON string, never a model/Tensor.

    to_dict returns a new defensive copy. No JSON/CSV file export or model
    comparison is implemented here. Constructing a result validates plain data,
    protocol/sample consistency, statistics and the derived service rate.
    """
    __slots__ = ("_payload",)
    _payload: str

    def __init__(self, data):
        copied = _plain(data)
        _validate(copied)
        object.__setattr__(self, "_payload", json.dumps(copied, allow_nan=False, separators=(",", ":")))

    def to_dict(self):
        return json.loads(self._payload)
