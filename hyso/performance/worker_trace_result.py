
"""P21 additive worker evidence linked to an existing P21 result.

Hashes protect artifact integrity, not authenticity of caller-supplied evidence.
Execution, epoch and batch association remain explicitly unknown.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json

from hyso.benchmark.result import _plain, _write_export

from .dataloader import DataLoaderAnalysisResult
from ._loader_event_channel import audit_trace_delivery
from ._loader_trace_groups import validate_trace_groups
from ._loader_trace_validation import validate_timing_event


def _digest(data):
    raw = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _derive(parent, packets, counters, worker_states):
    """Recompute findings from raw P21 and worker evidence."""
    config = parent["provenance"]["loaders"]

    if config["validation"] is not None:
        raise ValueError("validation worker linking not supported yet")

    training = config["training"]

    if training["dataset_type"] != (
        "hyso.performance._loader_dataset_trace.TracedDataset"
    ):
        raise ValueError("P21 source did not use TracedDataset")

    worker_count = training["num_workers"]

    if worker_count < 1:
        raise ValueError("worker trace requires num_workers > 0")

    expected = sum(
        row["samples"]
        for row in parent["observations"]
        if row["kind"] == "batch"
        and row["phase"] == "training"
    )

    if expected < 1:
        raise ValueError("no completed training samples to link")

    if type(packets) is not list or len(packets) > 100000:
        raise ValueError("invalid packet budget")

    # P21.19: Require complete live IPC counters, not caller defaults.
    required_counters = {"attempted", "accepted", "rejected"}
    if (
        type(counters) is not dict
        or set(counters) != required_counters
        or any(
            type(counters[name]) is not int
            or counters[name] < 0
            for name in required_counters
        )
    ):
        raise ValueError("valid live IPC trace counters are required")

    if type(worker_states) is not list:
        raise ValueError("invalid worker states")

    state_map = {}

    for state in worker_states:
        if type(state) is not dict or set(state) != {
            "worker_id", "pid", "alive", "exitcode"
        }:
            raise ValueError("invalid worker state")

        wid = state["worker_id"]
        pid = state["pid"]
        alive = state["alive"]
        exitcode = state["exitcode"]

        if (
            type(wid) is not int
            or wid < 0
            or wid in state_map
            or type(pid) is not int
            or pid <= 0
            or type(alive) is not bool
            or (
                exitcode is not None
                and type(exitcode) is not int
            )
        ):
            raise ValueError("invalid worker process identity")

        state_map[wid] = state

    if set(state_map) != set(range(worker_count)):
        raise ValueError("worker set disagrees with DataLoader")

    exitcodes = {
        wid: state["exitcode"]
        for wid, state in state_map.items()
    }

    audit = audit_trace_delivery(
        packets,
        expected_fetches=expected,
        trace_counters=counters,
        worker_exit_codes=exitcodes,
    )

    errors = list(audit["errors"])
    events = []
    stage_durations = defaultdict(list)

    for wid, state in state_map.items():
        if state["alive"]:
            errors.append(f"worker_{wid}: still alive")

    for index, packet in enumerate(packets):
        if type(packet) is not dict:
            errors.append(f"packet_{index}: invalid type")
            continue

        wid = packet.get("worker_id")
        state = state_map.get(wid) if type(wid) is int else None

        if state is None:
            errors.append(f"packet_{index}: unknown worker")
        elif packet.get("worker_pid") != state["pid"]:
            errors.append(f"packet_{index}: worker PID mismatch")

        rows = packet.get("events")

        if type(rows) is not list or len(rows) > 1000:
            errors.append(f"packet_{index}: invalid event array")
            continue

        for event in rows:
            if type(event) is not dict:
                errors.append(f"packet_{index}: invalid event")
                continue

            events.append(event)
            milliseconds = event.get("duration_ms")
            stage = event.get("stage")
            valid = (validate_timing_event(event) and state is not None
                     and event.get("worker_pid") == state["pid"])

            if not valid:
                errors.append(f"packet_{index}: invalid timing evidence")
            else:
                stage_durations[stage].append(milliseconds)

    # Only compare clocks inside the same worker fetch.
    try:
        groups = validate_trace_groups(
            events,
            expected_traces=expected,
        )
    except (TypeError, ValueError, KeyError):
        groups = {
            "status": "incomplete",
            "trace_count": 0,
            "event_count": len(events),
            "errors": ["invalid trace grouping evidence"],
        }

    errors.extend(groups["errors"])

    status = "complete" if not errors else "incomplete"

    timings = {}

    for stage in sorted(stage_durations):
        values = stage_durations[stage]

        timings[stage] = {
            "count": len(values),
            "mean_ms": (
                sum(values) / len(values)
                if status == "complete"
                else None
            ),
            "status": (
                "measured"
                if status == "complete"
                else "incomplete"
            ),
        }

    return {
        "status": status,
        "association": {"execution": "unknown", "epoch": "unknown", "batch": "unknown"},
        "audit": audit,
        "groups": groups,
        "stage_timings": timings,
        "errors": errors,
    }


@dataclass(frozen=True, init=False)
class WorkerTraceAnalysisResult:
    """Immutable, independently auditable P21 companion artifact."""

    __slots__ = ("_payload",)
    _payload: str

    def __init__(self, data):
        data = _plain(data)
        self._check(data)

        object.__setattr__(
            self,
            "_payload",
            json.dumps(data, allow_nan=False, separators=(",", ":")),
        )

    @staticmethod
    def _check(data):
        fields = {
            "schema_version",
            "format",
            "feature_ids",
            "source_p21",
            "source_p21_sha256",
            "source_segment_id",
            "packets",
            "trace_counters",
            "worker_states",
            "evidence_sha256",
            "analysis",
            "limitations",
        }
        if type(data) is not dict or set(data) not in (fields, fields - {"evidence_sha256"}):
            raise ValueError("invalid worker trace result fields")
        legacy = "evidence_sha256" not in data

        if (
            type(data["schema_version"]) is not int
            or data["schema_version"] != 1
            or data["format"] != "hyso.performance.worker_trace"
            or data["feature_ids"] != ["P21"]
        ):
            raise ValueError("invalid worker trace schema")

        parent = DataLoaderAnalysisResult(
            data["source_p21"]
        ).to_dict()

        if (
            data["source_p21_sha256"] != _digest(parent)
            or data["source_segment_id"]
            != parent["source_segment_id"]
        ):
            raise ValueError("invalid parent P21 association")

        if not legacy and data["evidence_sha256"] != _digest({
            key: data[key] for key in ("packets", "trace_counters", "worker_states")
        }):
            raise ValueError("worker evidence hash mismatch")

        derived = _derive(
            parent,
            data["packets"],
            data["trace_counters"],
            data["worker_states"],
        )
        # Read the original prototype's v1 JSON without rewriting its payload.
        # Newly produced artifacts always include the explicit association and
        # raw evidence hash; source P19/P20/P21 contracts remain unchanged.
        if legacy:
            derived.pop("association")

        if data["analysis"] != derived:
            raise ValueError("worker trace evidence was modified")

        if data["limitations"] != [
            "Same-execution association is caller-supplied.",
            "Worker-local intervals are not parent wait times.",
            "Dataset fetch excludes uninstrumented collation.",
            "Queue occupancy and prefetch efficiency are unsupported.",
        ]:
            raise ValueError("invalid evidence limitations")

    def to_dict(self):
        return json.loads(self._payload)

    def to_json(self, path, *, overwrite=False):
        data = self.to_dict()
        self._check(data)

        text = json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"

        _write_export(path, text, overwrite)


def analyze_worker_traces(
    loader_result,
    packets,
    *,
    trace_counters,
    worker_states,
):
    """Attach caller-supplied same-execution evidence to P21.

    Does not rerun training or mutate the original P21 result.
    """
    if type(loader_result) is not DataLoaderAnalysisResult:
        raise TypeError("expected DataLoaderAnalysisResult")

    parent = loader_result.to_dict()

    data = {
        "schema_version": 1,
        "format": "hyso.performance.worker_trace",
        "feature_ids": ["P21"],
        "source_p21": parent,
        "source_p21_sha256": _digest(parent),
        "source_segment_id": parent["source_segment_id"],
        "packets": packets,
        "trace_counters": trace_counters,
        "worker_states": worker_states,
        "evidence_sha256": _digest(dict(packets=packets, trace_counters=trace_counters,
                                        worker_states=worker_states)),
        "analysis": _derive(
            parent,
            packets,
            trace_counters,
            worker_states,
        ),
        "limitations": [
            "Same-execution association is caller-supplied.",
            "Worker-local intervals are not parent wait times.",
            "Dataset fetch excludes uninstrumented collation.",
            "Queue occupancy and prefetch efficiency are unsupported.",
        ],
    }

    return WorkerTraceAnalysisResult(data)
