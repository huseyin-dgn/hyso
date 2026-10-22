
"""P21 opt-in map-style dataset tracing."""

from __future__ import annotations

import uuid

from torch.utils.data import Dataset, get_worker_info

from ._loader_worker_trace import WorkerTimingRecorder
from ._loader_stage_trace import activate_recorder


class TracedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        recorder: WorkerTimingRecorder | None = None,
        channel=None,
        max_events_per_sample: int = 64,
    ):
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

        self.dataset = dataset
        self.recorder = recorder
        self.channel = channel
        self.max_events_per_sample = max_events_per_sample
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

        # Unique ID for this specific __getitem__ execution.
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
                for event in recorder.events[before:]:
                    payload = event.to_dict()

                    payload["trace_id"] = trace_id

                    if info is not None:
                        payload["worker_id"] = info.id

                    try:
                        if not self.channel.send(payload):
                            self.failed_sends += 1
                    except (OSError, ValueError):
                        self.failed_sends += 1
