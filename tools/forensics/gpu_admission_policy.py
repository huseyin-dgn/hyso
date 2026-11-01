"""GPU context is measurement metadata; only owned execution integrity can block.

Stdlib-only, no CUDA imports, sensor queries, process launch or termination.
The historical r3 evaluator remains unchanged and is not used for decisions.
"""
import math
import time

import native_admission as legacy

policy = legacy.policy  # unchanged isolation budgets and bounded JSON utilities
load = legacy.load
POLICY_ID = 'hyso.gpu-context.warning.v1'
SAFETY_FIELDS = ('job_isolation', 'worker_ownership', 'limit_readback',
                 'source_runtime_identity', 'timeout_cleanup',
                 'execution_contract', 'no_retry', 'result_integrity')


def field(value=None, status='unavailable', reason='not_collected'):
    return dict(value=value, status=status, reason=reason)


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def evaluate(capture, review=None, *, run_id, manifest_sha256, now_ns=None, safety=None):
    """Allow shared/unknown GPU context, retaining null/status/reason explicitly.

    `eligible` is this policy's decision, not authorization or proof that pending
    Job/identity checks passed. The runner independently enforces all hard gates.
    Optional safety receipts fail closed when incomplete or not exactly True.
    The 30s threshold determines telemetry quality only, never execution access.
    """
    blockers, warnings = [], set()
    if (not isinstance(run_id, str) or not run_id or not isinstance(manifest_sha256, str)
            or len(manifest_sha256) != 64 or any(ch not in '0123456789abcdef' for ch in manifest_sha256)):
        blockers.append('invalid_execution_contract')
    if safety is not None:
        if not isinstance(safety, dict):
            blockers.append('invalid_safety_receipts')
        else:
            blockers.extend(name+'_failed_or_missing' for name in SAFETY_FIELDS
                            if safety.get(name) is not True)
    capture = capture if isinstance(capture, dict) else {}
    review = review if isinstance(review, dict) else {}
    now = time.time_ns() if now_ns is None else now_ns
    start, end = capture.get('collection_start_time_ns'), capture.get('checked_time_ns')
    fresh = (type(start) is int and type(end) is int and type(now) is int
             and 0 <= now-start <= policy.MAX_AGE_NS and start <= end <= now)
    if not fresh:
        warnings.add('gpu_telemetry_stale_or_unavailable')
    inventory = capture.get('inventory')
    inventory = inventory if isinstance(inventory, dict) else {}
    pids = inventory.get('compute_pids')
    if not isinstance(pids, list) or any(type(p) is not int or p <= 0 for p in pids):
        pids = []
        warnings.add('gpu_process_inventory_unavailable')
    elif pids:
        warnings.add('shared_gpu_processes')
    if inventory.get('process_query_succeeded') is not True or inventory.get('compute_query_succeeded') is not True:
        warnings.add('gpu_process_inventory_unavailable')
    rows = review.get('gpu_processes')
    rows = rows if isinstance(rows, list) else []
    if pids and (len(set(pids)) != len(pids) or any(
            sum(isinstance(row, dict) and row.get('pid') == pid for row in rows) != 1 for pid in pids)):
        warnings.add('gpu_process_identity_incomplete')
    if any(isinstance(row, dict) and row.get('category') in ('unknown', 'compute') for row in rows):
        warnings.add('unknown_or_compute_gpu_process')
    if review.get('conflicting_workloads'):
        warnings.add('hardware_resource_contention')
    raw = inventory.get('raw_query')
    raw = raw if isinstance(raw, list) else []
    if any('[N/A]' in str(row) or 'Insufficient Permissions' in str(row) for row in raw):
        warnings.add('wddm_identity_or_memory_unsupported')
    samples = capture.get('gpu_samples')
    samples = samples if isinstance(samples, list) else []
    normalized = []
    for sample in samples[:3]:
        sample = sample if isinstance(sample, dict) else {}
        stamp = sample.get('time_ns')
        valid_stamp = fresh and type(stamp) is int and start <= stamp <= end
        values = {}
        for name in ('utilization_percent', 'free_mib'):
            value = sample.get(name)
            if not valid_stamp:
                values[name] = field(reason='stale_or_missing_timestamp')
            elif value is None or value in ('N/A', '[N/A]'):
                values[name] = field(status='unsupported', reason='sensor_NA_or_unsupported')
                warnings.add('gpu_sensor_unsupported')
            elif not numeric(value) or (name == 'utilization_percent' and value > 100):
                values[name] = field(reason='invalid_sensor_value')
                warnings.add('gpu_telemetry_invalid')
            else:
                values[name] = field(value, 'measured', None)
                if name == 'utilization_percent' and value >= 80:
                    warnings.add('high_gpu_usage')  # warning label only
                if name == 'free_mib' and value < 2048:
                    warnings.add('low_free_vram')  # no minimum required
        normalized.append(dict(time_ns=stamp if type(stamp) is int else None,
                               uuid=sample.get('uuid'), **values))
    if not normalized:
        warnings.add('gpu_telemetry_unavailable')
    for name, threshold in (('host_free_bytes', 4*1024**3), ('disk_free_bytes', 12*1024**3)):
        value = capture.get(name)
        if not numeric(value):
            warnings.add(name+'_unavailable')
        elif value < threshold:
            warnings.add(name+'_low')
    return dict(policy_id=POLICY_ID, run_id=run_id, eligible=not blockers,
                execution_allowed=not blockers, blockers=blockers, warnings=sorted(warnings),
                gpu_samples=normalized,
                gpu_process_inventory=field(sorted(set(pids)), 'observed', None) if fresh and pids else
                    field(reason='stale_or_missing_inventory'),
                measurement_reliability=dict(status='unqualified' if warnings else 'observed',
                    reason='shared_or_incomplete_gpu_context' if warnings else None,
                    performance_comparison=field(status='unsupported', reason='no_controlled_comparison')),
                hard_safety_status='verified' if safety is not None and not blockers else
                    ('blocked' if blockers else 'pending_runner_checks'),
                cuda_workload_started=False)


def sensor_metadata(sample):
    """Preserve device-wide observations; missing sensors never become zeros."""
    if not isinstance(sample, dict):
        return field(reason='sensor_not_collected')
    status = sample.get('status')
    if status == 'measured' and isinstance(sample.get('values'), dict):
        values = {name: field(value, 'measured', None) if value is not None else
                  field(status='unsupported', reason='sensor_NA_or_unsupported')
                  for name, value in sample['values'].items()}
        return field(dict(raw_observation=sample, fields=values), 'measured', None)
    return field(status='unsupported' if status == 'unsupported' else 'unavailable',
                 reason=sample.get('reason') or 'sensor_unavailable')


def controlled_failure(reason, *, simulated=False):
    if reason not in ('cuda_oom', 'timeout'):
        raise ValueError('unrecognized controlled failure')
    return dict(status='controlled_failure', reason=reason, measurements=None,
                retries=0, simulated=simulated, cleanup_required=True)
