
"""P21 explicit dataset stage instrumentation."""

from __future__ import annotations

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


def trace_data_stage(stage, operation, value, *, sample_id=None):
    """Run once; collect worker-local wall timing only in an active trace.

    A normal Dataset/DataLoader can call this without a profiler or custom
    loader. TracedDataset or the explicit task adapter activates collection.
    No I/O, RNG, result, exception or worker-lifecycle behavior is replaced.
    """
    if stage not in ("load", "decode", "transform"):
        raise ValueError("unsupported dataset stage")
    if not callable(operation):
        raise TypeError("operation must be callable")
    recorder = _active_recorder.get()
    if recorder is None:
        return operation(value)
    return measure_stage(recorder, stage, operation, value, sample_id=sample_id)
