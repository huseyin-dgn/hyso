"""Real CPU inference exports; CUDA-shaped fixtures test serialization only."""
import copy
import csv
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn

from hyso.benchmark import BenchmarkRunner, BenchmarkResult, compare_results
from hyso.benchmark import result as implementation


def cpu_result(**kwargs):
    config = dict(input_shape=(2, 1, 3, 5), warmup_iterations=0, measure_iterations=2, repeats=2,
                  model_name='Türkçe, "model"\nline')
    config.update(kwargs)
    return BenchmarkRunner(nn.Identity(), **config).run()


def cuda_data(data, *, measured=False, precision="fp32"):
    """Synthetic CUDA schema fixture, NOT a CUDA timing/performance result."""
    data = copy.deepcopy(data)
    data["configuration"].update(precision=precision, autocast_enabled=precision == "amp_fp16", memory_probe_count=int(measured))
    data["environment"]["device"] = data["input"]["device"] = data["model"]["output"]["device"] = "cuda:0"
    data["environment"]["cuda"] = dict(status="available", device_index=0, name="SERIALIZATION FIXTURE GPU",
        capability=[8, 6], total_device_memory_bytes=123456, cuda_build="12.1", cudnn_version=90100,
        allocator_backend="native", stream_id=0, cudnn_benchmark=False, cudnn_deterministic=False,
        cudnn_allow_tf32=True, matmul_allow_tf32=False)
    data["timing"].update(timing_method="cuda_event_elapsed", timing_scope="serial_current_stream_forward",
        synchronization_policy="repeat_device_drain_and_end_event_sync", clock_resolution_ns=None)
    data["throughput"]["source_timing_method"] = "cuda_event_elapsed"
    memory = dict(baseline_allocated_bytes=100, baseline_reserved_bytes=200, previous_peak_allocated_bytes=300,
        previous_peak_reserved_bytes=400, peak_allocated_bytes=150, peak_reserved_bytes=250, incremental_peak_bytes=50,
        current_allocated_after_probe_bytes=125, current_reserved_after_probe_bytes=250,
        current_allocated_after_output_release_bytes=100, current_reserved_after_output_release_bytes=250)
    data["cuda_memory"] = dict(status="measured" if measured else "disabled", value=150 if measured else None,
        reason=None if measured else "measure_cuda_memory_false", scope="separate_single_forward",
        counter_reset_performed=measured, **(memory if measured else dict.fromkeys(memory)))
    if measured:
        data["diagnostics"]["warnings"].append(dict(code="cuda_peak_reset", counter_reset_performed=True,
            previous_peak_allocated_bytes=300, previous_peak_reserved_bytes=400, scope="separate_single_forward",
            message="Fixture: irreversible process/device reset; caller/process exclusivity required."))
    return data


@pytest.fixture(params=["cpu", "cuda_disabled", "cuda_measured"])
def result(request):
    data = cpu_result().to_dict()
    if request.param != "cpu":
        data = cuda_data(data, measured=request.param == "cuda_measured")
    return BenchmarkResult(data)


def test_json_lossless_utf8_sorted_deterministic_and_immutable(result, tmp_path):
    before = result.to_dict()
    target, other = tmp_path / "result.json", tmp_path / "other.json"
    assert result.to_json(target) is None
    result.to_json(other)
    assert target.read_bytes() == other.read_bytes()
    text = target.read_text(encoding="utf-8")
    assert "Türkçe" in text and json.loads(text) == before
    assert text == json.dumps(before, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    assert BenchmarkResult(json.loads(text)).to_dict() == before
    assert len(before["timing"]["latency_samples_ms"]) == 2
    assert result.to_dict() == before and before["schema_version"] == 1
    assert all(v["value"] is None for v in before["optional_metrics"].values())
    if before["cuda_memory"]["status"] == "measured":
        assert json.loads(text)["diagnostics"]["warnings"][-1]["code"] == "cuda_peak_reset"
    with pytest.raises(FrozenInstanceError):
        result._payload = "changed"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def test_summary_single_row_header_stats_nulls_and_escaping(result, tmp_path):
    target = tmp_path / "summary.csv"
    assert result.to_csv(target) is None
    header, rows = read_csv(target)
    assert header == [name for name, _ in implementation._SUMMARY_COLUMNS]
    assert len(rows) == 1 and len(header) == len(set(header)) == 48
    row, data = rows[0], result.to_dict()
    assert row["model_name"] == data["model"]["model_name"]
    assert json.loads(row["input_shape"]) == data["input"]["shape"]
    assert row["run_id"] == data["identity"]["run_id"] and row["schema_version"] == "1"
    for key in ("mean_ms", "p50_ms", "p90_ms", "p95_ms", "std_ms", "min_ms", "max_ms"):
        assert float(row[key]) == data["stats"]["pooled"][key]
    assert float(row["throughput_images_per_second"]) == data["throughput"]["images_per_sec"]
    assert row["throughput_method"] == "derived_serial_service_rate"
    assert int(row["parameters_total"]) == data["parameters"]["total_parameters"]
    assert row["cuda_memory_status"] == data["cuda_memory"]["status"]
    if data["cuda_memory"]["status"] == "measured":
        assert row["cuda_peak_allocated_bytes"] == "150" and row["cuda_incremental_peak_bytes"] == "50"
        assert row["cuda_counter_reset_performed"] == "True"
    else:
        assert row["cuda_peak_allocated_bytes"] == row["cuda_incremental_peak_bytes"] == ""
    assert result.to_dict() == data


def test_long_csv_exact_indices_order_and_float_round_trip(result, tmp_path):
    target = tmp_path / "samples.csv"
    assert result.to_samples_csv(target) is None
    header, rows = read_csv(target)
    assert header == list(implementation._SAMPLE_COLUMNS)
    data = result.to_dict()
    expected = [(r, i, sample) for r, repeat in enumerate(data["timing"]["latency_samples_ms"]) for i, sample in enumerate(repeat)]
    assert len(rows) == len(expected) == 4
    assert [(int(x["repeat_index"]), int(x["iteration_index"]), float(x["latency_ms"])) for x in rows] == expected
    assert all(x["timing_method"] == data["timing"]["timing_method"] and x["run_id"] == data["identity"]["run_id"]
               and x["device"] == data["environment"]["device"] and x["precision"] == data["configuration"]["precision"] for x in rows)


@pytest.mark.parametrize("method", ["to_json", "to_csv", "to_samples_csv"])
def test_no_clobber_opt_in_replace_and_no_temp_leak(method, tmp_path):
    result = cpu_result()
    target = tmp_path / "existing"
    target.write_bytes(b"caller data")
    with pytest.raises(FileExistsError):
        getattr(result, method)(target)
    assert target.read_bytes() == b"caller data"
    getattr(result, method)(target, overwrite=True)
    assert target.read_bytes() != b"caller data"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["existing"]


@pytest.mark.parametrize("method", ["to_json", "to_csv", "to_samples_csv"])
@pytest.mark.parametrize("kind", ["missing_parent", "directory", "empty", "type", "overwrite_type"])
def test_invalid_destinations_and_missing_parent_errors(method, kind, tmp_path):
    result = cpu_result()
    before = result.to_dict()
    path = dict(missing_parent=tmp_path / "absent" / "file", directory=tmp_path, empty="", type=123,
                overwrite_type=tmp_path / "file")[kind]
    with pytest.raises((OSError, ValueError, TypeError)):
        getattr(result, method)(path, **({"overwrite": 1} if kind == "overwrite_type" else {}))
    assert result.to_dict() == before and not list(tmp_path.iterdir())


@pytest.mark.parametrize("operation", ["create", "write", "fsync", "link", "replace"])
@pytest.mark.parametrize("method", ["to_json", "to_csv", "to_samples_csv"])
def test_partial_write_permission_and_publish_failures_preserve_destination(monkeypatch, tmp_path, operation, method):
    result = cpu_result()
    before = result.to_dict()
    target = tmp_path / "caller"
    if operation == "replace":
        target.write_text("original", encoding="utf-8")
    def fail(*a, **kw):
        raise PermissionError("injected I/O failure")
    if operation == "create":
        monkeypatch.setattr(implementation.tempfile, "NamedTemporaryFile", fail)
    elif operation == "write":
        original = implementation.tempfile.NamedTemporaryFile
        class Partial:
            def __init__(self, *a, **kw):
                self.handle = original(*a, **kw)
                self.name = self.handle.name
            def __enter__(self):
                return self
            def __exit__(self, *a):
                self.handle.close()
            def write(self, text):
                self.handle.write(text[:17])
                fail()
        monkeypatch.setattr(implementation.tempfile, "NamedTemporaryFile", Partial)
    else:
        monkeypatch.setattr(implementation.os, operation, fail)
    with pytest.raises(PermissionError, match="injected"):
        getattr(result, method)(target, overwrite=operation == "replace")
    assert result.to_dict() == before
    assert list(tmp_path.iterdir()) == ([target] if operation == "replace" else [])
    if target.exists():
        assert target.read_text(encoding="utf-8") == "original"


def test_no_clobber_race_does_not_replace_late_destination(monkeypatch, tmp_path):
    result = cpu_result()
    target = tmp_path / "race.json"
    link = implementation.os.link
    def race(source, destination):
        Path(destination).write_bytes(b"concurrent caller")
        return link(source, destination)
    monkeypatch.setattr(implementation.os, "link", race)
    with pytest.raises(FileExistsError):
        result.to_json(target)
    assert target.read_bytes() == b"concurrent caller" and list(tmp_path.iterdir()) == [target]


def test_symlink_destination_validation_before_io(monkeypatch, tmp_path):
    target = tmp_path / "link"
    target.write_bytes(b"caller")
    is_symlink = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: True if p == target else is_symlink(p))
    with pytest.raises(ValueError, match="symlink"):
        cpu_result().to_json(target, overwrite=True)
    assert target.read_bytes() == b"caller" and list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("name", ["=SUM(1,2)", "+formula", "-text", "@formula", " \t=hidden", 'comma,"quote"\nline'])
def test_csv_text_escaping_formula_guard_and_json_full_fidelity(name, tmp_path):
    result = cpu_result(model_name=name)
    result.to_csv(tmp_path / "data.csv")
    result.to_json(tmp_path / "data.json")
    row = read_csv(tmp_path / "data.csv")[1][0]
    assert row["model_name"] == (name if name.startswith("comma") else "'" + name)
    assert json.loads((tmp_path / "data.json").read_text(encoding="utf-8"))["model"]["model_name"] == name


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), torch.tensor(1.), object()])
def test_nonfinite_nonplain_results_rejected_before_export(value):
    data = cpu_result().to_dict()
    data["model"]["model_config"] = {"invalid": value}
    with pytest.raises((ValueError, TypeError)):
        BenchmarkResult(data)


@pytest.mark.parametrize("method", ["to_json", "to_csv", "to_samples_csv"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), torch.tensor(1.)])
def test_export_boundary_rejects_invalid_snapshot_before_any_write(monkeypatch, method, value, tmp_path):
    result = cpu_result()
    invalid = result.to_dict()
    invalid["model"]["model_config"] = {"invalid": value}
    monkeypatch.setattr(BenchmarkResult, "to_dict", lambda self: invalid)
    with pytest.raises((ValueError, TypeError)):
        getattr(result, method)(tmp_path / "never_written")
    assert not list(tmp_path.iterdir())


def test_single_sample_std_remains_empty_with_status(tmp_path):
    result = cpu_result(measure_iterations=1, repeats=1)
    result.to_csv(tmp_path / "one.csv")
    row = read_csv(tmp_path / "one.csv")[1][0]
    assert row["std_ms"] == "" and row["std_status"] == "insufficient_samples"


def test_exports_and_compare_do_not_call_forward_runtime_or_change_rng(monkeypatch, tmp_path):
    result = cpu_result()
    before = result.to_dict()
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    def forbidden(*a, **kw):
        raise AssertionError("runtime/forward forbidden during export/comparison")
    monkeypatch.setattr(nn.Identity, "forward", forbidden)
    for key in ("Event", "is_available", "current_device", "get_device_properties", "synchronize", "memory_allocated", "reset_peak_memory_stats", "init"):
        monkeypatch.setattr(torch.cuda, key, forbidden)
    result.to_json(tmp_path / "result.json")
    result.to_csv(tmp_path / "summary.csv")
    result.to_samples_csv(tmp_path / "samples.csv")
    compare_results(result, result)
    assert result.to_dict() == before and random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], np_state[1]) and np.random.get_state()[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), cpu) and not torch.cuda.is_initialized()
