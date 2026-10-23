
"""P21 trace grouping and per-fetch evidence validation."""

from collections import Counter, defaultdict
from ._loader_trace_validation import validate_timing_event


def validate_trace_groups(
    events,
    *,
    expected_stages=None,
    expected_traces=None,
):
    """Validate actual worker-local trace evidence.

    expected_stages: optional stage -> count mapping per trace.
    expected_traces: optional number of expected fetch executions.
    """
    groups = defaultdict(list)
    errors = []

    if expected_stages is not None:
        if (
            type(expected_stages) is not dict
            or not expected_stages
            or any(
                type(stage) is not str
                or type(count) is not int
                or count < 0
                for stage, count in expected_stages.items()
            )
        ):
            raise ValueError("invalid expected stages")

        required = Counter({
            stage: count
            for stage, count in expected_stages.items()
            if count > 0
        })
    else:
        required = None

    if expected_traces is not None and (
        type(expected_traces) is not int
        or expected_traces < 0
    ):
        raise ValueError("invalid expected trace count")

    for index, event in enumerate(events):
        if not validate_timing_event(event):
            errors.append(f"event_{index}: invalid timing evidence")
            continue

        trace_id = event.get("trace_id")

        if type(trace_id) is not str or not trace_id:
            errors.append(f"event_{index}: missing trace_id")
            continue

        groups[trace_id].append(event)

    for trace_id, rows in groups.items():
        stages = Counter(row.get("stage") for row in rows)

        if stages["dataset_fetch"] != 1:
            errors.append(f"{trace_id}: invalid fetch count")

        if required is not None and stages != required:
            errors.append(f"{trace_id}: stage count mismatch")

        identities = {
            (
                row.get("worker_pid"),
                row.get("worker_id"),
                row.get("sample_id"),
            )
            for row in rows
        }

        if len(identities) != 1:
            errors.append(f"{trace_id}: identity mismatch")

        fetches = [
            row for row in rows
            if row.get("stage") == "dataset_fetch"
        ]

        if len(fetches) != 1:
            continue

        fetch = fetches[0]

        for row in rows:
            start = row.get("start_ns")
            end = row.get("end_ns")
            duration = row.get("duration_ns")

            if not all(
                type(value) is int
                for value in (start, end, duration)
            ):
                errors.append(f"{trace_id}: invalid clock")
                break

            if (
                start < 0
                or end < start
                or duration != end - start
            ):
                errors.append(f"{trace_id}: invalid duration")
                break

            if not (
                fetch["start_ns"] <= start
                <= end <= fetch["end_ns"]
            ):
                errors.append(f"{trace_id}: invalid nesting")
                break

    if (
        expected_traces is not None
        and len(groups) != expected_traces
    ):
        errors.append("trace count mismatch")

    return {
        "status": "complete" if not errors else "incomplete",
        "trace_count": len(groups),
        "event_count": len(events),
        "errors": errors,
    }
