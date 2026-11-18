"""Result bytes and trace bytes have separate safety budgets."""
import pytest

from hyso.performance import ProfilingProtocol
from hyso.performance.profiling import _check_result_evidence_bytes


def test_cpu_result_budget_default_and_range():
    p = ProfilingProtocol(profile_memory=True, allocation_lifetimes=True,
                          cpu_memory_ownership=True)
    assert p.max_events == 20000
    assert p.max_trace_bytes == 16 * 1024 ** 2
    assert p.max_cpu_witness_items == 250000
    assert p.max_cpu_result_bytes == 128 * 1024 ** 2
    for bad in (0, -1, False, 1.5, 128 * 1024 ** 2 + 1):
        with pytest.raises((ValueError, TypeError)):
            ProfilingProtocol(max_cpu_result_bytes=bad)


def test_non_ownership_result_keeps_legacy_trace_bound():
    p = ProfilingProtocol(max_trace_bytes=1024)
    assert _check_result_evidence_bytes(2048, p) == 2048
    with pytest.raises(ValueError, match="actual_bytes=2049, limit_bytes=2048"):
        _check_result_evidence_bytes(2049, p)


def test_cpu_result_bytes_are_independent_of_trace_bytes():
    p = ProfilingProtocol(max_trace_bytes=1024, max_cpu_result_bytes=8192,
                          profile_memory=True, allocation_lifetimes=True,
                          cpu_memory_ownership=True)
    assert _check_result_evidence_bytes(8192, p) == 8192
    with pytest.raises(ValueError, match="scope=cpu_memory_ownership.*actual_bytes=8193, limit_bytes=8192"):
        _check_result_evidence_bytes(8193, p)


def test_limited_cpu_witness_results_cannot_silently_overflow():
    p = ProfilingProtocol(profile_memory=True, allocation_lifetimes=True,
                          cpu_memory_ownership=True, max_cpu_result_bytes=4096)
    with pytest.raises(ValueError, match="result exceeds evidence budget"):
        _check_result_evidence_bytes(4097, p)
