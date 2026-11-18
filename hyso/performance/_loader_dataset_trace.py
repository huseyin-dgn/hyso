
"""P21 opt-in map-style dataset tracing."""

from __future__ import annotations
import typing as _typing

import uuid

from torch.utils.data import Dataset, get_worker_info

from ._loader_worker_trace import WorkerTimingRecorder
from ._loader_stage_trace import activate_recorder


class TracedDataset(Dataset):
    """Wrap an existing Dataset with bounded worker-side timing evidence.

    Args:
        dataset (Dataset): Caller-owned finite Dataset used by the loader or trace wrapper. Required.
        recorder (WorkerTimingRecorder | None): Existing bounded evidence recorder. Default: None.
        channel (_typing.Any): Explicit worker-trace evidence channel. Default: None.
        max_events_per_sample (int): Upper bound on events per sample retained/accepted by this operation. Default: 64.
        trace_batches (bool): Observe the explicitly selected batch trace profile. Default: False.

    Returns:
        An instance of TracedDataset with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        TracedDataset(dataset)

    HYSo Guide: analysis.p21
    Offline: hyso docs show analysis.p21 --lang en
    Open: hyso docs open analysis.p21 --lang en
    """
    def __init__(
        self,
        dataset: Dataset,
        recorder: WorkerTimingRecorder | None = None,
        channel: '_typing.Any'=None,
        max_events_per_sample: int = 64,
        trace_batches: bool = False,
    ) -> 'None':
        """Initialize TracedDataset with the supplied settings.

        Args:
            dataset (Dataset): Caller-owned finite Dataset used by the loader or trace wrapper. Required.
            recorder (WorkerTimingRecorder | None): Existing bounded evidence recorder. Default: None.
            channel (_typing.Any): Explicit worker-trace evidence channel. Default: None.
            max_events_per_sample (int): Upper bound on events per sample retained/accepted by this operation. Default: 64.
            trace_batches (bool): Observe the explicitly selected batch trace profile. Default: False.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            TracedDataset(dataset)

        HYSo Guide: analysis.p21
        Offline: hyso docs show analysis.p21 --lang en
        Open: hyso docs open analysis.p21 --lang en
        """
        if not isinstance(dataset, Dataset):
            raise TypeError("expected torch Dataset")

        if recorder is not None and not isinstance(
            recorder, WorkerTimingRecorder
        ):
            raise TypeError("expected WorkerTimingRecorder")

        if recorder is None and channel is None:
            raise ValueError("recorder or channel required")

        if (
            type(max_events_per_sample) is not int
            or not 1 <= max_events_per_sample <= 1000
        ):
            raise ValueError("invalid per-sample event budget")

        if type(trace_batches) is not bool:
            raise TypeError("trace_batches must be bool")

        if trace_batches and channel is None:
            raise ValueError(
                "trace_batches requires an event channel"
            )

        self.dataset = dataset
        self.recorder = recorder
        self.channel = channel
        self.max_events_per_sample = max_events_per_sample
        self.trace_batches = trace_batches
        self.failed_sends = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        info = get_worker_info()

        recorder = self.recorder

        if recorder is None:
            recorder = WorkerTimingRecorder(
                worker_id=info.id if info else None,
                max_events=self.max_events_per_sample,
            )

        before = len(recorder.events)
        dropped_before = recorder.dropped_events

        # Unique identity for this __getitem__ execution.
        trace_id = uuid.uuid4().hex

        try:
            with activate_recorder(recorder):
                with recorder.measure(
                    "dataset_fetch",
                    sample_id=str(index),
                ):
                    return self.dataset[index]

        finally:
            if self.channel is not None:
                events = []

                for event in recorder.events[before:]:
                    payload = event.to_dict()
                    payload["trace_id"] = trace_id

                    if info is not None:
                        payload["worker_id"] = info.id

                    events.append(payload)

                if self.trace_batches:
                    envelope = {
                        "schema_version": 1,
                        "trace_id": trace_id,
                        "worker_id": (
                            info.id if info is not None else None
                        ),
                        "worker_pid": (
                            events[0]["worker_pid"]
                            if events else None
                        ),
                        "sample_id": str(index),
                        "event_count": len(events),
                        "dropped_events": (
                            recorder.dropped_events - dropped_before
                        ),
                        "events": events,
                    }

                    try:
                        if not self.channel.send_trace(envelope):
                            self.failed_sends += 1
                    except (OSError, ValueError, RuntimeError):
                        self.failed_sends += 1

                else:
                    for payload in events:
                        try:
                            if not self.channel.send(payload):
                                self.failed_sends += 1
                        except (OSError, ValueError, RuntimeError):
                            self.failed_sends += 1
