
"""P21 per-fetch trace grouping and validation."""

from collections import Counter, defaultdict

_REQUIRED = Counter({
    "decode": 1,
    "transform": 1,
    "dataset_fetch": 1,
})


def validate_trace_groups(events):
    groups = defaultdict(list)
    errors = []

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"event_{index}: invalid type")
            continue

        trace_id = event.get("trace_id")

        if not isinstance(trace_id, str) or not trace_id:
            errors.append(f"event_{index}: missing trace_id")
            continue

        groups[trace_id].append(event)

    for trace_id, rows in groups.items():
        stages = Counter(row.get("stage") for row in rows)

        if stages != _REQUIRED:
            errors.append(f"{trace_id}: incomplete stages")

        identities = {
            (row.get("worker_pid"),
             row.get("worker_id"),
             row.get("sample_id"))
            for row in rows
        }

        if len(identities) != 1:
            errors.append(f"{trace_id}: identity mismatch")

        fetches = [
            row for row in rows
            if row.get("stage") == "dataset_fetch"
        ]

        if len(fetches) == 1:
            fetch = fetches[0]

            for row in rows:
                if not all(
                    type(row.get(key)) is int
                    for key in ("start_ns", "end_ns")
                ):
                    errors.append(f"{trace_id}: invalid clock")
                    break

                if not (
                    fetch["start_ns"] <= row["start_ns"]
                    <= row["end_ns"] <= fetch["end_ns"]
                ):
                    errors.append(f"{trace_id}: invalid nesting")
                    break

    return {
        "status": "complete" if not errors else "incomplete",
        "trace_count": len(groups),
        "event_count": len(events),
        "errors": errors,
    }
