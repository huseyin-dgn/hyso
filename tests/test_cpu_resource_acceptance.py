"""Small CPU-only measurement and evidence boundary regression cases."""
import math

import pytest
import torch

from hyso.benchmark import BenchmarkRunner, BenchmarkResult, compare_results
from hyso.analysis import EvidenceRecord, EvidenceScope
from hyso.performance import analyze_latency
from hyso.reporting import create_report, ReportError
from examples.performance.cpu_resource_acceptance import builders, fresh, state_hash, summarize


@pytest.fixture
def result():
    return BenchmarkRunner(torch.nn.Identity(), input_shape=(1, 1, 8, 8),
        warmup_iterations=2, measure_iterations=3, repeats=2).run()


@pytest.mark.parametrize("field,value", [
    ("measure_iterations", 0), ("measure_iterations", -1), ("repeats", 0),
])
def test_invalid_repetition_count(field, value):
    with pytest.raises(ValueError):
        BenchmarkRunner(torch.nn.Identity(), input_shape=(1, 1, 8, 8), **{field: value})


@pytest.mark.parametrize("value", [-1, 0, math.nan, math.inf])
def test_invalid_duration(result, value):
    data = result.to_dict()
    data["timing"]["latency_samples_ms"][0][0] = value
    with pytest.raises(ValueError):
        BenchmarkResult(data)
    with pytest.raises(ValueError):
        summarize([value])


@pytest.mark.parametrize("mutation", ["warmup_in_samples", "batch", "empty", "partial", "sync", "provenance", "timer", "missing_memory"])
def test_incomplete_or_inconsistent_payload(result, mutation):
    data = result.to_dict()
    if mutation == "warmup_in_samples":
        data["timing"]["latency_samples_ms"][0].append(1.)
    elif mutation == "batch":
        data["input"]["batch_size"] = 8
    elif mutation == "empty":
        data["timing"]["latency_samples_ms"] = []
    elif mutation == "partial":
        data["timing"]["latency_samples_ms"].pop()
    elif mutation == "sync":
        del data["timing"]["synchronization_policy"]
    elif mutation == "provenance":
        del data["provenance"]
    elif mutation == "timer":
        data["timing"]["timing_method"] = "seconds"
    elif mutation == "missing_memory":
        del data["cuda_memory"]
    with pytest.raises((ValueError, KeyError)):
        BenchmarkResult(data)


@pytest.mark.parametrize("field,value", [("input_shape", (2, 1, 8, 8)), ("precision", "fp64")])
def test_different_conditions_are_incompatible(result, field, value):
    options = dict(input_shape=(1, 1, 8, 8), warmup_iterations=2, measure_iterations=3, repeats=2)
    options[field] = value
    other = BenchmarkRunner(torch.nn.Identity(), **options).run()
    assert compare_results(result, other)["status"] == "incompatible"


def test_different_device_rejected_by_native_scope(result):
    with pytest.raises(ValueError, match="native_scope_mismatch:device"):
        EvidenceRecord.from_native(result.to_dict(), feature_id="P01", scope=EvidenceScope(device="cuda:0"))


def test_same_state_identity_is_explicit_scope_constraint(result):
    with pytest.raises(ValueError, match="native_scope_mismatch:model_state_sha256"):
        EvidenceRecord.from_native(result.to_dict(), feature_id="P01", scope=EvidenceScope(model_state_sha256="0"*64))


def test_cuda_unavailable_has_no_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        BenchmarkRunner(torch.nn.Identity(), input_shape=(1, 1, 8, 8), device="cuda:0")


def test_missing_and_changed_artifact(result, tmp_path):
    path = tmp_path / "result.json"
    with pytest.raises((OSError, ValueError)):
        EvidenceRecord.from_json(path, feature_id="P01", expected_sha256="0"*64)
    result.to_json(path)
    with pytest.raises(ValueError):
        EvidenceRecord.from_json(path, feature_id="P01", expected_sha256="0"*64)


def test_untyped_crash_history_is_rejected():
    with pytest.raises((ValueError, TypeError)):
        BenchmarkResult(dict(status="native_crash", exit_code=0xC0000409, partial_samples=[1.]))
    with pytest.raises(ReportError):
        create_report(dict(status="PASS", history=True, authoritative_evidence=True))


@pytest.mark.parametrize("kind", ["sequential", "graph"])
def test_fixture_shape_gradient_and_restore(kind):
    torch.set_num_threads(1)
    config = builders()[kind].to_config()
    model = fresh(kind, config)
    inputs = torch.randn(8, 1, 8, 8)
    output = model(inputs)
    assert output.shape == (8, 10) and torch.isfinite(output).all()
    output.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    restored = fresh(kind, config)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert state_hash(model) == state_hash(restored)
    assert not torch.cuda.is_initialized()


def test_summary_arithmetic_and_insufficient_tail(result):
    stats = summarize([1., 2., 3., 4.])
    assert stats["mean"] == stats["p50"] == 2.5
    assert stats["p90"] == pytest.approx(3.7)
    assert stats["std"] == pytest.approx(math.sqrt(5/3))
    assert stats["p95"] is None
    assert all(row["status"] == "insufficient_samples" for row in analyze_latency(result, quantiles=(.9, .95)).to_dict()["measurements"])
    with pytest.raises(ValueError):
        summarize([])
