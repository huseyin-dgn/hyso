
"""P21 explicit dataset stage instrumentation."""

from __future__ import annotations
import typing as _typing

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, TypeVar

from ._loader_worker_trace import WorkerTimingRecorder

T = TypeVar("T")
R = TypeVar("R")

_active_recorder: ContextVar[
    WorkerTimingRecorder | None
] = ContextVar("hyso_loader_recorder", default=None)


@contextmanager
def activate_recorder(recorder: WorkerTimingRecorder):
    """Bind a recorder to the current execution context."""
    if not isinstance(recorder, WorkerTimingRecorder):
        raise TypeError("expected WorkerTimingRecorder")

    token = _active_recorder.set(recorder)
    try:
        yield
    finally:
        _active_recorder.reset(token)


def measure_stage(
    recorder: WorkerTimingRecorder | None,
    stage: str,
    operation: Callable[[T], R],
    value: T,
    *,
    sample_id: str | None = None,
) -> R:
    """Execute exactly once and record an explicit stage."""

    if recorder is None:
        recorder = _active_recorder.get()

    if not isinstance(recorder, WorkerTimingRecorder):
        raise RuntimeError("no active worker recorder")

    if stage not in ("load", "decode", "transform"):
        raise ValueError("unsupported dataset stage")

    if not callable(operation):
        raise TypeError("operation must be callable")

    with recorder.measure(stage, sample_id=sample_id):
        return operation(value)


def trace_data_stage(stage: _typing.Literal['load', 'decode', 'transform'], operation: '_typing.Any', value: '_typing.Any', *, sample_id: '_typing.Any'=None) -> '_typing.Any':
    """Run once; collect worker-local wall timing only in an active trace.

    A normal Dataset/DataLoader can call this without a profiler or custom
    loader. TracedDataset or the explicit task adapter activates collection.
    No I/O, RNG, result, exception or worker-lifecycle behavior is replaced.

    Args:
        stage (_typing.Any): Workflow stage or timing-stage identity for this operation. Required.
        operation (_typing.Any): Supported operation name. Required.
        value (_typing.Any): Observed metric value; unavailable values retain their native null/status
            policy. Required.
        sample_id (_typing.Any): Stable sample identity for correlation. Default: None.

    Returns:
        The retained trace data stage value, without recomputing measurements.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        trace_data_stage(stage, operation, value)

    HYSo Guide: analysis.p21
    Offline: hyso docs show analysis.p21 --lang en
    Open: hyso docs open analysis.p21 --lang en
    """
    if stage not in ("load", "decode", "transform"):
        raise ValueError("unsupported dataset stage")
    if not callable(operation):
        raise TypeError("operation must be callable")
    recorder = _active_recorder.get()
    if recorder is None:
        return operation(value)
    return measure_stage(recorder, stage, operation, value, sample_id=sample_id)
