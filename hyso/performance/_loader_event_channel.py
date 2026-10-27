"""P21 bounded synchronous shared-memory IPC and delivery accounting.

No Queue feeder threads or pipe reads: a killed producer cannot leave a partial
frame blocking the parent. A dead owner's lock fails with TimeoutError. Such a
channel must be discarded, never repaired or reported as complete.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
import math
import multiprocessing as mp


class WorkerEventChannel:
    """Single-run bounded transport; drain and audit after producers stop.

    Counters cover send_trace only; legacy send events cannot share an audited
    trace stream. Reads remove immutable serialized snapshots synchronously.
    """

    def __init__(self, *, max_queue_size=1024, max_packet_bytes=65536,
                 lock_timeout=0.25):
        if type(max_queue_size) is not int or not 1 <= max_queue_size <= 10000:
            raise ValueError("invalid queue capacity")
        if type(max_packet_bytes) is not int or not 256 <= max_packet_bytes <= 1048576:
            raise ValueError("invalid packet capacity")
        if max_queue_size * max_packet_bytes > 128 * 1024 * 1024:
            raise ValueError("channel memory budget exceeds 128 MiB")
        if type(lock_timeout) not in (int, float) or not math.isfinite(lock_timeout) or not 0 < lock_timeout <= 5:
            raise ValueError("invalid lock deadline")
        context = mp.get_context("spawn")
        self._lock = context.Lock()
        self._buffer = context.RawArray("B", max_queue_size * max_packet_bytes)
        self._lengths = context.RawArray("q", max_queue_size)
        # head, queued, attempted, accepted, rejected
        self._state = context.RawArray("q", 5)
        self.max_queue_size = max_queue_size
        self.max_packet_bytes = max_packet_bytes
        self.lock_timeout = lock_timeout
        self.closed = False

    @contextmanager
    def _guard(self):
        if self.closed:
            raise RuntimeError("channel is closed")
        if not self._lock.acquire(timeout=self.lock_timeout):
            raise TimeoutError("worker event channel lock deadline exceeded")
        try:
            yield
        finally:
            self._lock.release()

    def _send(self, event, *, trace):
        if type(event) is not dict:
            raise TypeError("event must be a dictionary")
        # Serialize before acquiring the shared lock; no asynchronous pickle or
        # mutable caller references can turn an accepted packet into silent loss.
        try:
            payload = json.dumps(event, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError):
            payload = None
        with self._guard():
            if trace:
                self._state[2] += 1
            if payload is None or len(payload) > self.max_packet_bytes or self._state[1] == self.max_queue_size:
                if trace:
                    self._state[4] += 1
                return False
            slot = (self._state[0] + self._state[1]) % self.max_queue_size
            offset = slot * self.max_packet_bytes
            memoryview(self._buffer).cast("B")[offset:offset + len(payload)] = payload
            self._lengths[slot] = len(payload)
            self._state[1] += 1
            if trace:
                self._state[3] += 1
            return True

    def send(self, event):
        return self._send(event, trace=False)

    def send_trace(self, trace):
        # Malformed envelopes are still counted transport attempts. Semantic
        # validation belongs to the audit, never to a background feeder.
        return self._send(trace, trace=True)

    def trace_counters(self):
        with self._guard():
            return dict(zip(("attempted", "accepted", "rejected"), self._state[2:5]))

    def receive_available(self, *, max_items=1024):
        if type(max_items) is not int or max_items < 1:
            raise ValueError("invalid drain budget")
        payloads = []
        with self._guard():
            for _ in range(min(max_items, self._state[1])):
                slot = self._state[0]
                offset = slot * self.max_packet_bytes
                payloads.append(bytes(memoryview(self._buffer).cast("B")[offset:offset + self._lengths[slot]]))
                self._state[0] = (slot + 1) % self.max_queue_size
                self._state[1] -= 1
        return [json.loads(payload) for payload in payloads]

    def close(self):
        """Local, immediate close; owner stops producers before final audit.

        Shared memory is reclaimed when process references are released. No
        locks, feeder joins, or ownership of the caller's DataLoader workers.
        """
        self.closed = True


def verify_event_coverage(events, expected_sample_ids):
    """Verify received dataset-fetch coverage."""
    expected = Counter(str(x) for x in expected_sample_ids)

    received = Counter(
        str(event["sample_id"])
        for event in events
        if event.get("stage") == "dataset_fetch"
    )

    missing = expected - received
    extra = received - expected

    return {
        "status": (
            "complete" if not missing and not extra
            else "incomplete"
        ),
        "expected_count": sum(expected.values()),
        "received_count": sum(received.values()),
        "missing": dict(missing),
        "unexpected": dict(extra),
    }


def audit_trace_delivery(
    packets,
    *,
    expected_fetches,
    worker_exit_codes=None,
    accepted_count=None,
    trace_counters=None,
):
    """Fail-closed trace delivery audit.

    expected_fetches must be independently established.

    trace_counters should come from WorkerEventChannel
    after worker shutdown.
    """
    if type(expected_fetches) is not int or expected_fetches < 0:
        raise ValueError("invalid expected_fetches")

    if accepted_count is not None and (
        type(accepted_count) is not int
        or accepted_count < 0
    ):
        raise ValueError("invalid accepted_count")

    errors = []
    if trace_counters is None:
        errors.append("live trace counters missing")
    seen = set()
    worker_counts = Counter()
    received_events = 0
    recorder_dropped = 0

    for index, packet in enumerate(packets):
        if type(packet) is not dict:
            errors.append(f"packet_{index}: invalid type")
            continue

        if type(packet.get("schema_version")) is not int or packet["schema_version"] != 1:
            errors.append(f"packet_{index}: invalid schema")
        pid = packet.get("worker_pid")
        if type(pid) is not int or pid <= 0:
            errors.append(f"packet_{index}: invalid PID")
        trace_id = packet.get("trace_id")
        worker_id = packet.get("worker_id")
        sample_id = packet.get("sample_id")
        events = packet.get("events")
        count = packet.get("event_count")
        dropped = packet.get("dropped_events")

        if type(trace_id) is not str or not trace_id:
            errors.append(f"packet_{index}: invalid trace_id")
        elif trace_id in seen:
            errors.append(f"packet_{index}: duplicate trace_id")
        else:
            seen.add(trace_id)

        if type(worker_id) is not int or worker_id < 0:
            errors.append(f"packet_{index}: invalid worker_id")
        else:
            worker_counts[worker_id] += 1

        if type(sample_id) is not str:
            errors.append(f"packet_{index}: invalid sample_id")

        if type(events) is not list:
            errors.append(f"packet_{index}: invalid events")
            continue

        if type(count) is not int or count != len(events):
            errors.append(f"packet_{index}: event count mismatch")

        if type(dropped) is not int or dropped < 0:
            errors.append(f"packet_{index}: invalid dropped count")
        else:
            recorder_dropped += dropped

        fetch_count = sum(
            type(event) is dict
            and event.get("stage") == "dataset_fetch"
            for event in events
        )

        if fetch_count != 1:
            errors.append(f"packet_{index}: invalid fetch evidence")

        for event in events:
            if type(event) is not dict:
                errors.append(f"packet_{index}: invalid event")
                continue

            if (
                event.get("worker_pid") != pid
                or event.get("trace_id") != trace_id
                or event.get("worker_id") != worker_id
                or event.get("sample_id") != sample_id
            ):
                errors.append(
                    f"packet_{index}: event identity mismatch"
                )

        received_events += len(events)

    from ._loader_trace_groups import validate_trace_groups
    from ._loader_trace_validation import validate_timing_event
    for packet in packets:
        if type(packet) is dict and type(packet.get("events")) is list:
            rows = packet["events"]
            if any(not validate_timing_event(row) for row in rows):
                errors.append("invalid timing event")
            grouped = validate_trace_groups(rows)
            errors.extend(grouped["errors"])

    received = len(packets)

    if received != expected_fetches:
        errors.append("expected fetch count mismatch")

    if accepted_count is not None:
        if accepted_count < received:
            errors.append("received exceeds accepted")
        elif accepted_count != received:
            errors.append("accepted but not received")

    if recorder_dropped:
        errors.append("worker recorder dropped events")

    # Validate live multiprocessing trace counters.
    if trace_counters is not None:
        names = ("attempted", "accepted", "rejected")

        valid = (
            type(trace_counters) is dict
            and set(trace_counters) == set(names)
            and all(
                type(trace_counters.get(name)) is int
                and trace_counters[name] >= 0
                for name in names
            )
        )

        if not valid:
            errors.append("invalid shared trace counters")
        else:
            attempted = trace_counters["attempted"]
            accepted = trace_counters["accepted"]
            rejected = trace_counters["rejected"]

            if attempted != accepted + rejected:
                errors.append("unaccounted trace attempts")

            if attempted != expected_fetches:
                errors.append("trace attempt count mismatch")

            if accepted != received:
                errors.append("accepted but not received")

            if rejected:
                errors.append("queue rejected trace packets")

            if (
                accepted_count is not None
                and accepted_count != accepted
            ):
                errors.append("accepted counter mismatch")

    # Exit codes must come from actual worker Process objects
    # or an independently verified process lifecycle record.
    if worker_exit_codes is None:
        errors.append("worker termination not verified")

    elif (
        type(worker_exit_codes) is not dict
        or not worker_exit_codes
        or any(
            type(worker) is not int
            or type(code) is not int
            or code != 0
            for worker, code in worker_exit_codes.items()
        )
    ):
        errors.append("worker exit failure or invalid receipt")

    elif set(worker_counts) != set(worker_exit_codes):
        errors.append("worker receipt mismatch")

    return {
        "status": (
            "complete" if not errors else "incomplete"
        ),
        "expected_fetches": expected_fetches,
        "received_packets": received,
        "received_events": received_events,
        "accepted_count": accepted_count,
        "trace_counters": trace_counters,
        "recorder_dropped_events": recorder_dropped,
        "worker_packet_counts": {
            str(worker_id): count
            for worker_id, count in worker_counts.items()
        },
        "errors": errors,
    }
