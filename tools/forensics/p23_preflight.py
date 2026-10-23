"""Stdlib-only B4 diagnostic policy. No collection, process launch or Torch import.

Evidence is collected once by an operator, including an explicit review of every
GPU process. This decision is necessary, never sufficient, for native admission:
the sealed controller still requires separate approval, identity and Job checks.
This is a trusted local diagnostic protocol, not adversarial attestation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


POLICY_ID = "hyso.b4.diagnostic-preflight.v3"
GPU_UUID = "GPU-9eec2f80-4c4e-a5cd-7091-03f0d013cf11"
MAX_AGE_NS = 30 * 10**9
LIMITS = dict(child_timeout_s=60, outer_timeout_s=90,
              process_private_commit=4 * 1024**3,
              child_job_private_commit=8 * 1024**3, child_processes=4,
              outer_job_private_commit=12 * 1024**3, outer_processes=8,
              cuda_allocator_fraction=.15, log_bytes_per_channel=65536,
              result_bytes=2 * 1024**2, metadata_rows=2000,
              metadata_bytes_per_process=1024**2, kill_on_close=True)


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return value


def _text(value, name):
    if type(value) is not str or not value or len(value) > 1024:
        raise ValueError(name + " must be a bounded nonempty string")
    return value


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def evaluate(evidence, *, run_id, manifest_sha256, now_ns=None):
    """Validate one capture; malformed/missing evidence fails closed.

    gpu_processes contains every PID returned by the compute inventory. A human
    reviewer supplies category desktop/compute/unknown and a nonempty reason;
    names alone never decide. Unknown or compute entries block this diagnostic.
    The inventory digest binds the review to retained raw process/query rows.
    Neither digest nor reviewer text proves that an executable is trustworthy.
    """
    if type(evidence) is not dict:
        raise ValueError("evidence must be an object")
    now = time.time_ns() if now_ns is None else _integer(now_ns, "now_ns")
    blockers = []
    if (evidence["policy_id"] != POLICY_ID or evidence["run_id"] != run_id
            or evidence["manifest_sha256"] != manifest_sha256):
        blockers.append("policy_or_run_or_manifest_mismatch")
    start = _integer(evidence["collection_start_time_ns"], "collection_start_time_ns")
    end = _integer(evidence["checked_time_ns"], "checked_time_ns")
    if not 0 <= now - start <= MAX_AGE_NS or not start <= end <= now:
        blockers.append("stale_or_future_or_reversed_capture")
    if _integer(evidence["host_free_bytes"], "host_free_bytes") < 4 * 1024**3:
        blockers.append("host_free_below_4GiB")
    if _integer(evidence["disk_free_bytes"], "disk_free_bytes") < 12 * 1024**3:
        blockers.append("disk_free_below_12GiB")
    samples = evidence["gpu_samples"]
    if type(samples) is not list or len(samples) != 3:
        raise ValueError("exactly three GPU samples required; no resampling")
    previous = start - 1
    utilization = []
    for sample in samples:
        stamp = _integer(sample["time_ns"], "GPU sample time_ns")
        if not start <= stamp <= end or stamp <= previous:
            blockers.append("invalid_GPU_sample_order_or_capture_window")
        previous = stamp
        if sample["uuid"] != GPU_UUID:
            blockers.append("GPU_identity_mismatch")
        if _integer(sample["free_mib"], "free_mib") < 2048:
            blockers.append("GPU_free_below_2048MiB")
        busy = sample["utilization_percent"]
        if busy is not None and (type(busy) not in (float, int)
                                 or not math.isfinite(busy) or not 0 <= busy <= 100):
            raise ValueError("invalid GPU utilization telemetry")
        utilization.append(busy)
    inventory = evidence["inventory"]
    if type(inventory) is not dict or type(inventory["processes"]) is not list:
        raise ValueError("raw process inventory required")
    if inventory["process_query_succeeded"] is not True or inventory["compute_query_succeeded"] is not True:
        blockers.append("inventory_query_failed_or_unavailable")
    pids = inventory["compute_pids"]
    if type(pids) is not list or len(pids) > 4096 or len(inventory["processes"]) > 16384:
        raise ValueError("bounded compute/process inventory required")
    for pid in pids:
        if _integer(pid, "compute PID") == 0:
            raise ValueError("compute PID must be positive")
    if len(set(pids)) != len(pids):
        raise ValueError("duplicate compute PID")
    # raw_query strings retain permissions/N/A information, not fabricated zeros.
    if type(inventory["raw_query"]) is not list or any(type(x) is not str for x in inventory["raw_query"]):
        raise ValueError("raw compute query rows required")
    raw_pids = [int(row.split(",", 1)[0].strip()) for row in inventory["raw_query"]]
    if sorted(raw_pids) != sorted(pids):
        blockers.append("raw_compute_PID_mismatch")
    review = evidence["workload_review"]
    _text(review["reviewer"], "reviewer")
    if review["inventory_sha256"] != canonical_hash(inventory):
        blockers.append("inventory_review_digest_mismatch")
    if review["status"] != "complete":
        blockers.append("incomplete_workload_review")
    conflicts = review["conflicting_workloads"]
    if type(conflicts) is not list:
        raise ValueError("conflicting_workloads must be a list")
    if conflicts:
        blockers.append("conflicting_workload")
    rows = review["gpu_processes"]
    if type(rows) is not list or len(rows) > 4096:
        raise ValueError("bounded GPU process reviews required")
    reviewed = []
    process_ids = [_integer(p["pid"], "inventory PID") for p in inventory["processes"]]
    for process in inventory["processes"]:
        _text(process["name"], "process name")
    if len(set(process_ids)) != len(process_ids):
        raise ValueError("duplicate inventory PID")
    for row in rows:
        pid = _integer(row["pid"], "review PID")
        reviewed.append(pid)
        _text(row["reason"], "classification reason")
        if row["category"] not in ("desktop", "compute", "unknown"):
            raise ValueError("invalid GPU process category")
        if row["category"] != "desktop":
            blockers.append("compute_or_unknown_GPU_process")
        if pid not in process_ids:
            blockers.append("GPU_process_identity_missing")
    if sorted(reviewed) != sorted(pids):
        blockers.append("incomplete_or_duplicate_GPU_process_review")
    return dict(policy_id=POLICY_ID, run_id=run_id, eligible=not blockers,
                blockers=sorted(set(blockers)), utilization_percent_samples=utilization,
                utilization_role="telemetry_only_no_admission_threshold",
                performance_comparison=dict(value=None, status="unsupported",
                    reason="single_forensic_diagnostic_not_a_controlled_performance_comparison"),
                limitations=["Review is operator-supplied, not cryptographic process attestation.",
                             "Preflight cannot guarantee future memory or device availability.",
                             "Separate approval, source/runtime verification and Job admission remain mandatory."],
                cuda_workload_started=False)


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key: " + key)
        value[key] = item
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    try:
        with args.evidence.open("rb") as stream:
            raw = stream.read(1024**2 + 1)
        if len(raw) > 1024**2:
            raise ValueError("preflight evidence exceeds 1MiB")
        evidence = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique)
        result = evaluate(evidence, run_id=args.run_id, manifest_sha256=args.manifest_sha256)
    except (OSError, KeyError, TypeError, ValueError, UnicodeError) as error:
        print(json.dumps(dict(eligible=False, error=str(error), cuda_workload_started=False)))
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0 if result["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
