
"""P21 bounded, opt-in multiprocessing event channel."""

from __future__ import annotations

import multiprocessing as mp
from queue import Empty, Full


class WorkerEventChannel:
    """Non-blocking worker-to-parent event transfer.

    The owner must close the channel after workers have stopped.
    """

    def __init__(self, *, max_queue_size: int = 1024):
        if (
            type(max_queue_size) is not int
            or not 1 <= max_queue_size <= 100000
        ):
            raise ValueError("invalid queue capacity")

        context = mp.get_context("spawn")
        self.queue = context.Queue(maxsize=max_queue_size)
        self.max_queue_size = max_queue_size
        self.closed = False

    def send(self, event: dict) -> bool:
        if self.closed:
            raise RuntimeError("channel is closed")

        if type(event) is not dict:
            raise TypeError("event must be a dictionary")

        try:
            self.queue.put_nowait(event)
            return True
        except Full:
            return False

    def receive_available(self, *, max_items: int = 1024) -> list[dict]:
        if self.closed:
            raise RuntimeError("channel is closed")

        if type(max_items) is not int or max_items < 1:
            raise ValueError("invalid drain budget")

        received = []

        for _ in range(max_items):
            try:
                received.append(self.queue.get_nowait())
            except Empty:
                break

        return received

    def close(self):
        if not self.closed:
            self.closed = True
            self.queue.close()
            self.queue.join_thread()


def verify_event_coverage(events, expected_sample_ids):
    """Validate received sample-event coverage.

    Only valid for datasets where sample IDs are unique per run.
    """
    from collections import Counter

    expected = Counter(str(x) for x in expected_sample_ids)
    received = Counter(
        str(event["sample_id"])
        for event in events
        if event.get("stage") == "dataset_fetch"
    )

    missing = expected - received
    extra = received - expected

    return {
        "status": "complete" if not missing and not extra else "incomplete",
        "expected_count": sum(expected.values()),
        "received_count": sum(received.values()),
        "missing": dict(missing),
        "unexpected": dict(extra),
    }
