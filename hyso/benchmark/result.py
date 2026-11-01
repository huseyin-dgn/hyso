"""Immutable inference results, explicit exports and condition compatibility."""
import csv
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


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
    device = inputs["device"]
    cuda = type(device) is str and device.startswith("cuda:") and device[5:].isdigit() and str(int(device[5:])) == device[5:]
    method = "cuda_event_elapsed" if cuda else "perf_counter_ns"
    if (timing.get("timing_method") != method
            or timing.get("timing_scope") != ("serial_current_stream_forward" if cuda else "synchronous_host_forward")
            or config.get("precision") not in (("fp32", "amp_fp16") if cuda else ("fp32", "fp64"))
            or timing["synchronization_policy"] != ("repeat_device_drain_and_end_event_sync" if cuda else "none")
            or (not cuda and device != "cpu") or data["environment"]["device"] != device):
        raise ValueError("unsupported timing/precision/device schema")
    measured_memory = cuda and data["cuda_memory"].get("status") == "measured"
    expected_config = dict(preflight_forward_count=1, memory_probe_count=int(measured_memory), forced_model_mode="eval",
                           inference_mode=True, autocast_enabled=config["precision"] == "amp_fp16", protocol_version=1, input_policy="fixed_resident_input")
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
                         source_timing_method=method, per_repeat_images_per_sec=[_throughput(row, shape[0]) for row in nested],
                         window_wall_images_per_sec=None, window_status="unsupported", window_reason="separate_protocol_deferred")
    if json.dumps(data["throughput"], sort_keys=True) != json.dumps(expected_rate, sort_keys=True):
        raise ValueError("throughput disagrees with samples")
    if not cuda and data["cuda_memory"] != {"status": "not_applicable", "value": None, "reason": "cpu_only_b1"}:
        raise ValueError("B1 has no CUDA memory measurement")
    if cuda:
        gpu = data["environment"]["cuda"]
        if (type(gpu) is not dict or type(data["model"]["output"]) is not dict
                or inputs["dtype"] != "torch.float32" or data["model"]["output"].get("device") != device
                or gpu.get("status") != "available" or type(gpu.get("device_index")) is not int
                or gpu["device_index"] != int(device[5:]) or type(gpu.get("stream_id")) is not int or gpu["stream_id"] < 0):
            raise ValueError("invalid CUDA resident input/output/environment")
        memory = data["cuda_memory"]
        fields = {"baseline_allocated_bytes", "baseline_reserved_bytes", "previous_peak_allocated_bytes",
                  "previous_peak_reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes", "incremental_peak_bytes",
                  "current_allocated_after_probe_bytes", "current_reserved_after_probe_bytes",
                  "current_allocated_after_output_release_bytes", "current_reserved_after_output_release_bytes"}
        if (set(memory) != fields | {"status", "value", "reason", "scope", "counter_reset_performed"}
                or memory["scope"] != "separate_single_forward" or type(memory["counter_reset_performed"]) is not bool
                or memory["counter_reset_performed"] != measured_memory or timing["clock_resolution_ns"] is not None):
            raise ValueError("invalid CUDA memory/timing protocol")
        if measured_memory:
            if any(type(memory[k]) is not int or memory[k] < 0 for k in fields):
                raise ValueError("invalid CUDA allocator bytes")
            if (type(memory["value"]) is not int or memory["value"] != memory["peak_allocated_bytes"]
                    or memory["reason"] is not None
                    or memory["incremental_peak_bytes"] != memory["peak_allocated_bytes"] - memory["baseline_allocated_bytes"]):
                raise ValueError("incoherent CUDA peak accounting")
            for allocated, reserved in (("baseline_allocated_bytes", "baseline_reserved_bytes"),
                                        ("previous_peak_allocated_bytes", "previous_peak_reserved_bytes"),
                                        ("peak_allocated_bytes", "peak_reserved_bytes"),
                                        ("current_allocated_after_probe_bytes", "current_reserved_after_probe_bytes"),
                                        ("current_allocated_after_output_release_bytes", "current_reserved_after_output_release_bytes")):
                if memory[allocated] > memory[reserved]:
                    raise ValueError("allocated bytes exceed reserved bytes")
            if (memory["peak_allocated_bytes"] < max(memory["baseline_allocated_bytes"], memory["current_allocated_after_probe_bytes"])
                    or memory["peak_reserved_bytes"] < max(memory["baseline_reserved_bytes"], memory["current_reserved_after_probe_bytes"])
                    or memory["previous_peak_allocated_bytes"] < memory["baseline_allocated_bytes"]
                    or memory["previous_peak_reserved_bytes"] < memory["baseline_reserved_bytes"]
                    or memory["current_allocated_after_output_release_bytes"] > memory["peak_allocated_bytes"]
                    or memory["current_reserved_after_output_release_bytes"] > memory["peak_reserved_bytes"]):
                raise ValueError("peak bytes below observed bytes")
            warnings = data["diagnostics"]["warnings"]
            if type(warnings) is not list or any(type(w) is not dict for w in warnings):
                raise ValueError("invalid CUDA peak reset diagnostics")
            reset_warnings = [w for w in warnings if w.get("code") == "cuda_peak_reset"]
            if not any(w.get("counter_reset_performed") is True and w.get("scope") == "separate_single_forward"
                       and all(w.get(k) == memory[k] for k in ("previous_peak_allocated_bytes", "previous_peak_reserved_bytes"))
                       and type(w.get("message")) is str for w in reset_warnings):
                raise ValueError("CUDA peak reset requires explicit diagnostics with prior peaks")
        elif (memory["status"] != "disabled" or memory["reason"] != "measure_cuda_memory_false"
              or any(memory[k] is not None for k in fields | {"value"})):
            raise ValueError("disabled CUDA memory values must be null")
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

    to_dict returns a new defensive copy. Explicit exports never run inference.
    Constructing a result validates plain data,
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

    def _export_data(self):
        data = _plain(self.to_dict())
        _validate(data)
        return data

    def to_json(self, path, *, overwrite=False):
        """Write full schema-v1 UTF-8 JSON, sorted keys and lossless samples.

        Existing destinations raise FileExistsError unless overwrite=True.
        A completed sibling temporary file is atomically published; parents
        must exist. I/O failures propagate and leave the result unchanged.
        """
        data = self._export_data()
        _write_export(path, json.dumps(data, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n", overwrite)

    def to_csv(self, path, *, overwrite=False):
        """Write one summary row; JSON remains the canonical complete record.

        Null cells are empty; status/reason distinguish missing measurements.
        Spreadsheet formula-like text is prefixed with an apostrophe. Column
        order is fixed by _SUMMARY_COLUMNS; floats use Python round-trip text.
        Destination/overwrite policy is identical to to_json.
        """
        data = self._export_data()
        row = {column: _field(data, key) for column, key in _SUMMARY_COLUMNS}
        row["input_shape"] = json.dumps(data["input"]["shape"], separators=(",", ":"))
        _write_export(path, _csv_text(list(row), [row]), overwrite)

    def to_samples_csv(self, path, *, overwrite=False):
        """Write R*N measured samples in repeat/iteration order (zero based).

        No warmup/preflight/memory samples are invented. Null/escaping and
        destination policy match to_csv; no append or automatic directory creation.
        """
        data = self._export_data()
        common = dict(schema_version=data["schema_version"], run_id=data["identity"]["run_id"],
                      timing_method=data["timing"]["timing_method"], timing_scope=data["timing"]["timing_scope"],
                      device=data["environment"]["device"], precision=data["configuration"]["precision"],
                      batch_size=data["input"]["batch_size"])
        rows = (dict(common, repeat_index=r, iteration_index=i, latency_ms=value)
                for r, repeat in enumerate(data["timing"]["latency_samples_ms"]) for i, value in enumerate(repeat))
        _write_export(path, _csv_text(_SAMPLE_COLUMNS, rows), overwrite)


def _field(data, path):
    for key in path.split("."):
        if type(data) is not dict or key not in data:
            return None
        data = data[key]
    return data


_SUMMARY_COLUMNS = (
    ("schema_version", "schema_version"), ("run_id", "identity.run_id"),
    ("started_at_utc", "identity.started_at_utc"), ("finished_at_utc", "identity.finished_at_utc"),
    ("model_name", "model.model_name"), ("model_class", "model.qualified_class"),
    ("device", "environment.device"), ("precision", "configuration.precision"),
    ("input_shape", "input.shape"), ("input_dtype", "input.dtype"), ("batch_size", "input.batch_size"),
    ("warmup_iterations", "configuration.warmup_iterations"),
    ("measure_iterations", "configuration.measure_iterations"), ("repeats", "configuration.repeats"),
    ("timing_method", "timing.timing_method"), ("timing_scope", "timing.timing_scope"),
    ("synchronization_policy", "timing.synchronization_policy"),
    ("sample_count", "stats.pooled.count"), ("mean_ms", "stats.pooled.mean_ms"),
    ("p50_ms", "stats.pooled.p50_ms"), ("p90_ms", "stats.pooled.p90_ms"), ("p95_ms", "stats.pooled.p95_ms"),
    ("min_ms", "stats.pooled.min_ms"), ("max_ms", "stats.pooled.max_ms"),
    ("std_ms", "stats.pooled.std_ms"), ("std_status", "stats.pooled.std_status"),
    ("throughput_images_per_second", "throughput.images_per_sec"), ("throughput_method", "throughput.method"),
    ("throughput_source_timing_method", "throughput.source_timing_method"),
    ("parameters_total", "parameters.total_parameters"), ("parameters_trainable", "parameters.trainable_parameters"),
    ("parameters_frozen", "parameters.frozen_parameters"), ("parameter_buffer_bytes", "parameters.parameter_buffer_bytes"),
    ("cuda_memory_status", "cuda_memory.status"), ("cuda_memory_reason", "cuda_memory.reason"),
    ("cuda_memory_scope", "cuda_memory.scope"), ("cuda_counter_reset_performed", "cuda_memory.counter_reset_performed"),
) + tuple(("cuda_" + key, "cuda_memory." + key) for key in (
    "baseline_allocated_bytes", "baseline_reserved_bytes", "previous_peak_allocated_bytes", "previous_peak_reserved_bytes",
    "peak_allocated_bytes", "peak_reserved_bytes", "incremental_peak_bytes", "current_allocated_after_probe_bytes",
    "current_reserved_after_probe_bytes", "current_allocated_after_output_release_bytes", "current_reserved_after_output_release_bytes"))

_SAMPLE_COLUMNS = ("schema_version", "run_id", "repeat_index", "iteration_index", "latency_ms",
                   "timing_method", "timing_scope", "device", "precision", "batch_size")


def _csv_text(columns, rows):
    def cell(value):
        if type(value) is str and value.lstrip(" \t\r\n").startswith(("=", "+", "-", "@")):
            return "'" + value
        return value
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: cell(value) for key, value in row.items()})
    return stream.getvalue()


def _write_export(path, text, overwrite):
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be bool")
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("destination must be a string or path-like object")
    if isinstance(path, str) and not path:
        raise ValueError("destination cannot be empty")
    destination = Path(path)
    if destination.is_symlink():
        raise ValueError("symlink destinations are unsupported")
    if destination.exists():
        if not destination.is_file():
            raise ValueError("destination must be a regular file")
        if not overwrite:
            raise FileExistsError(str(destination))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", delete=False,
                                         prefix=".hyso-export-", suffix=".tmp", dir=destination.parent) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Atomic publication without a check-then-overwrite race. Same-dir
            # hard links require filesystem support; unsupported I/O propagates.
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


_CONDITION_FIELDS = (
    "schema_version", "benchmark_type", "configuration.protocol_version",
    "configuration.precision", "configuration.autocast_enabled", "configuration.warmup_iterations",
    "configuration.measure_iterations", "configuration.repeats", "configuration.preflight_forward_count",
    "configuration.forced_model_mode", "configuration.inference_mode", "configuration.input_policy",
    "timing.timing_method", "timing.timing_scope", "timing.synchronization_policy",
    "throughput.method", "throughput.source_timing_method", "stats.quantile_method", "stats.std_ddof",
    "input.shape", "input.batch_size", "input.dtype", "input.layout", "input.stride", "input.content_sha256",
    "environment.device", "environment.python_version", "environment.torch_version", "environment.hyso_version",
    "environment.operating_system", "environment.platform", "environment.cpu.machine", "environment.cpu.processor",
    "environment.cpu.logical_cpu_count", "environment.cpu.threads", "environment.cpu.interop_threads",
    "environment.backend.deterministic_algorithms", "environment.backend.deterministic_warn_only",
    "environment.backend.float32_matmul_precision",
)
_CUDA_CONDITION_FIELDS = tuple("environment.cuda." + key for key in (
    "name", "capability", "total_device_memory_bytes", "cuda_build", "cudnn_version",
    "cudnn_benchmark", "cudnn_deterministic", "cudnn_allow_tf32", "matmul_allow_tf32", "allocator_backend"))


def _conditions(left, right, fields):
    compared, mismatched, missing = [], [], []
    def unknown(key, value):
        if value is None or value == "" or value == [] or (type(value) is str and value.strip().lower() in
                ("unknown", "not_collected", "unavailable", "n/a")):
            return True
        name = key.rsplit(".", 1)[-1]
        if name == "content_sha256":
            return type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        if name in ("autocast_enabled", "inference_mode", "deterministic_algorithms", "deterministic_warn_only",
                    "mkldnn_enabled", "cudnn_benchmark", "cudnn_deterministic", "cudnn_allow_tf32",
                    "matmul_allow_tf32", "counter_reset_performed"):
            return type(value) is not bool
        if name in ("schema_version", "protocol_version", "warmup_iterations", "measure_iterations", "repeats",
                    "preflight_forward_count", "std_ddof", "batch_size", "logical_cpu_count", "threads",
                    "interop_threads", "total_device_memory_bytes", "cudnn_version"):
            return type(value) is not int or value < (0 if name == "warmup_iterations" else 1)
        if name in ("shape", "stride", "capability"):
            return type(value) is not list or any(type(x) is not int or x < 0 for x in value) or (name == "capability" and len(value) != 2)
        return type(value) is not str or not value.strip()
    for key in fields:
        a, b = _field(left, key), _field(right, key)
        compared.append(dict(field=key, left=a, right=b))
        sides = [side for side, value in (("left", a), ("right", b)) if unknown(key, value)]
        if sides:
            missing.append(dict(field=key, sides=sides, reason="missing_unknown_or_invalid_metadata"))
        elif json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
            mismatched.append(dict(field=key, left=a, right=b))
    status = "incompatible" if mismatched else "insufficient_metadata" if missing else "compatible"
    return dict(status=status, mismatched_fields=mismatched, missing_required_fields=missing, compared_fields=compared)


def compare_results(a, b):
    """Return a plain compatibility report for two exact BenchmarkResult objects.

    Status describes timing conditions, never a winner or significance test.
    Model architecture/config/weights and optional source provenance are variables,
    not equality requirements. Memory eligibility is separate; unknown process
    isolation yields insufficient_metadata for measured GPU memory. Optional
    provenance.memory_isolation='exclusive_fresh_process' is a caller assertion,
    never inferred or verified here. No model/runtime/CUDA calls or I/O occur.
    """
    if type(a) is not BenchmarkResult or type(b) is not BenchmarkResult:
        raise TypeError("compare_results requires two BenchmarkResult objects")
    left, right = a.to_dict(), b.to_dict()
    cuda = [row["environment"]["device"].startswith("cuda:") for row in (left, right)]
    fields = list(_CONDITION_FIELDS)
    if all(cuda):
        fields.extend(_CUDA_CONDITION_FIELDS)
    elif not any(cuda):
        fields.append("environment.backend.mkldnn_enabled")
    timing = _conditions(left, right, fields)
    warnings = [dict(code="measurement_variability", message="Matching recorded conditions do not prove significance, equal clocks/load/cache/thermals/power or model quality."),
                dict(code="hardware_profile_identity", message="Recorded hardware/OS profiles do not uniquely identify a physical host/GPU; no unit UUID or driver/telemetry evidence is inferred.")]
    if any(row["model"].get("source_sha256") is None for row in (left, right)):
        warnings.append(dict(code="optional_source_hash_missing", message="Source hash is informational for model comparison; same-model regression needs an additional identity policy."))
    memory = dict(status="not_applicable", mismatched_fields=[], missing_required_fields=[], compared_fields=[],
                  absolute_peak_model_comparable=False, warnings=[], observations={"left": left["cuda_memory"], "right": right["cuda_memory"]})
    statuses = [row["cuda_memory"]["status"] for row in (left, right)]
    if any(cuda):
        if not all(cuda) or statuses != ["measured", "measured"]:
            memory["status"] = "not_comparable"
            memory["warnings"].append(dict(code="memory_probe_unavailable", message="Both results must contain measured CUDA probes; timing eligibility is independent."))
        else:
            memory.update(_conditions(left, right, (
                "environment.cuda.allocator_backend", "cuda_memory.scope", "cuda_memory.counter_reset_performed",
                "provenance.memory_isolation")))
            if timing["status"] != "compatible":
                memory["status"] = "incompatible" if timing["status"] == "incompatible" else "insufficient_metadata"
                memory["warnings"].append(dict(code="memory_conditions_unverified", message="Timing/hardware/runtime conditions must also be compatible."))
            isolation = [_field(row, "provenance.memory_isolation") for row in (left, right)]
            if isolation != ["exclusive_fresh_process", "exclusive_fresh_process"]:
                if memory["status"] == "compatible":
                    memory["status"] = "insufficient_metadata"
                    memory["missing_required_fields"].append(dict(field="provenance.memory_isolation",
                        sides=[side for side, value in zip(("left", "right"), isolation) if value != "exclusive_fresh_process"],
                        reason="fresh_process_isolation_not_attested"))
                memory["warnings"].append(dict(code="memory_isolation_unverified", message="B1/B2 do not attest fresh-process/caller exclusivity; supplied labels are caller assertions."))
            for key in ("baseline_allocated_bytes", "baseline_reserved_bytes"):
                if left["cuda_memory"][key] != right["cuda_memory"][key]:
                    memory["warnings"].append(dict(code="memory_baseline_differs", field="cuda_memory." + key,
                                                   left=left["cuda_memory"][key], right=right["cuda_memory"][key]))
            memory["warnings"].append(dict(code="allocator_scope_only", message="Absolute process/device peaks and cached baselines are not model-only physical VRAM; no peak ranking is produced."))
    # Payloads are already validated plain data and decoded defensively. Do not
    # reapply result-depth limits after nesting informational data in a report.
    return dict(comparison_schema_version=1, status=timing["status"],
                       mismatched_fields=timing["mismatched_fields"], missing_required_fields=timing["missing_required_fields"],
                       compared_fields=timing["compared_fields"], warnings=warnings, timing_comparability=timing,
                       memory_comparability=memory,
                       informational={side: dict(model=row["model"], parameters=row["parameters"], provenance=row["provenance"])
                                      for side, row in (("left", left), ("right", right))})
