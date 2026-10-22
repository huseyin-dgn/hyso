
"""P21 opt-in worker-local timing primitives.

No Torch dependency, no process spawning, no global hooks.
Cross-process clock subtraction is not supported here.
"""

from __future__ import annotations
from torch.utils.data import get_worker_info
from contextlib import contextmanager
from dataclasses import dataclass
import os
import time
from typing import Iterator


STAGES = frozenset({
    "dataset_fetch",
    "decode",
    "transform",
    "collate",
    "dispatch",
})


@dataclass(frozen=True)
class WorkerTimingEvent:
    worker_pid: int
    worker_id: int | None
    sample_id: str | None
    stage: str
    start_ns: int
    end_ns: int
    duration_ns: int
    outcome: str

    def to_dict(self) -> dict:
        return {
            "worker_pid": self.worker_pid,
            "worker_id": self.worker_id,
            "sample_id": self.sample_id,
            "stage": self.stage,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ns": self.duration_ns,
            "duration_ms": self.duration_ns / 1_000_000,
            "outcome": self.outcome,
            "clock": "perf_counter_ns",
            "clock_scope": "worker_local",
        }


class WorkerTimingRecorder:
    """Bounded local event recorder.

    Events are diagnostic only and never change model/data outputs.
    """

    def __init__(
        self,
        *,
        worker_id: int | None = None,
        max_events: int = 10000,
    ):
        if worker_id is not None and (
            type(worker_id) is not int or worker_id < 0
        ):
            raise ValueError("worker_id must be nonnegative")

        if type(max_events) is not int or not 1 <= max_events <= 100000:
            raise ValueError("invalid event budget")

        info = get_worker_info()

        self.worker_id = (
            worker_id
            if worker_id is not None
            else info.id if info is not None else None
        )
        self.max_events = max_events
        self.events: list[WorkerTimingEvent] = []
        self.dropped_events = 0

    @contextmanager
    def measure(
        self,
        stage: str,
        *,
        sample_id: str | None = None,
    ) -> Iterator[None]:

        if stage not in STAGES:
            raise ValueError(f"unsupported stage: {stage}")

        if sample_id is not None and type(sample_id) is not str:
            raise TypeError("sample_id must be a string or None")

        start_ns = time.perf_counter_ns()
        outcome = "completed"

        try:
            yield
        except BaseException:
            outcome = "failed"
            raise
        finally:
            end_ns = time.perf_counter_ns()

            event = WorkerTimingEvent(
                worker_pid=os.getpid(),
                worker_id=self.worker_id,
                sample_id=sample_id,
                stage=stage,
                start_ns=start_ns,
                end_ns=end_ns,
                duration_ns=end_ns - start_ns,
                outcome=outcome,
            )

            if len(self.events) < self.max_events:
                self.events.append(event)
            else:
                self.dropped_events += 1

    def snapshot(self) -> dict:
        return {
            "schema_version": 1,
            "events": [event.to_dict() for event in self.events],
            "recorded_events": len(self.events),
            "dropped_events": self.dropped_events,
            "complete": self.dropped_events == 0,
        }
