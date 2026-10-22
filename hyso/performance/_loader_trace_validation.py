
"""P21 worker trace validation.

Validates received evidence only.
Does not claim guaranteed IPC delivery.
"""

from collections import Counter
import math


STAGES = ("decode", "transform", "dataset_fetch")


def validate_worker_trace(
    events,
    expected_sample_ids,
    *,
    expected_workers=None,
):
    """Validate an explicitly known sample/stage workload.

    Repeated sample IDs are supported via occurrence counts.
    """
    expected = Counter(str(x) for x in expected_sample_ids)
    observed = Counter()
    errors = []
    workers = set()

    for position, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"event_{position}: invalid type")
            continue

        stage = event.get("stage")
        sample = event.get("sample_id")
        worker = event.get("worker_id")
        pid = event.get("worker_pid")
        start = event.get("start_ns")
        end = event.get("end_ns")
        duration = event.get("duration_ns")

        if stage not in STAGES or not isinstance(sample, str):
            errors.append(f"event_{position}: invalid identity")
            continue

        observed[(stage, sample)] += 1

        if type(worker) is not int or worker < 0:
            errors.append(f"event_{position}: invalid worker")
        else:
            workers.add(worker)

        if type(pid) is not int or pid <= 0:
            errors.append(f"event_{position}: invalid PID")

        if not all(type(v) is int for v in (start, end, duration)):
            errors.append(f"event_{position}: invalid clock")
        elif start < 0 or end < start or duration != end - start:
            errors.append(f"event_{position}: clock mismatch")

        milliseconds = event.get("duration_ms")

        if (
            type(milliseconds) not in (int, float)
            or not math.isfinite(milliseconds)
            or type(duration) is not int
            or not math.isclose(
                milliseconds, duration / 1_000_000,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            errors.append(f"event_{position}: duration mismatch")

        if event.get("outcome") != "completed":
            errors.append(f"event_{position}: incomplete stage")

        if (
            event.get("clock") != "perf_counter_ns"
            or event.get("clock_scope") != "worker_local"
        ):
            errors.append(f"event_{position}: invalid clock scope")

    missing = {}
    unexpected = {}

    for stage in STAGES:
        for sample, count in expected.items():
            actual = observed[(stage, sample)]
            if actual < count:
                missing[(stage, sample)] = count - actual
            elif actual > count:
                unexpected[(stage, sample)] = actual - count

    for (stage, sample), count in observed.items():
        if sample not in expected:
            unexpected[(stage, sample)] = count

    if expected_workers is not None and workers != set(expected_workers):
        errors.append("worker set mismatch")

    complete = not errors and not missing and not unexpected

    return {
        "status": "complete" if complete else "incomplete",
        "expected_events": sum(expected.values()) * len(STAGES),
        "received_events": len(events),
        "workers": sorted(workers),
        "missing": [
            {"stage": stage, "sample_id": sample, "count": count}
            for (stage, sample), count in sorted(missing.items())
        ],
        "unexpected": [
            {"stage": stage, "sample_id": sample, "count": count}
            for (stage, sample), count in sorted(unexpected.items())
        ],
        "errors": errors,
    }
