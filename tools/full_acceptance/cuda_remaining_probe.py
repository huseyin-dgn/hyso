"""Explicit, one-use, sequential diagnostic for the 21 untested CUDA tasks.

Never certifies the legacy 04.cuda stage. All GPU work is Windows Job owned.
"""
from pathlib import Path
from .core import verify, sha, publish, now
from .cuda_tasks import TASKS, validate_terminal
from .cuda_supervisor_adapter import WindowsJobCudaAdapter, check_admission

EXCLUDED = frozenset(('04.cuda.A.latency', '04.cuda.B.latency'))
REMAINING = tuple(task for task in TASKS if task.task_id not in EXCLUDED)
assert len(REMAINING) == 21


def run_remaining(manifest_path, approved_sha256, *, adapter_factory=WindowsJobCudaAdapter):
    if not isinstance(approved_sha256, str) or len(approved_sha256) != 64 or any(c not in '0123456789abcdef' for c in approved_sha256):
        raise ValueError('exact lowercase approved SHA256 required')
    manifest_path = Path(manifest_path).resolve()
    manifest = verify(manifest_path, expected=approved_sha256, live=True)
    root = Path(manifest['output']).resolve()
    if manifest_path != root / 'manifest.json' or sha(manifest_path) != approved_sha256:
        raise ValueError('noncanonical or changed manifest')
    if (root / 'STARTED.json').exists() or (root / 'stop-request.json').exists():
        raise PermissionError('existing execution or stop request')
    summary = root / 'cuda-remaining-probe-summary.json'
    if summary.exists() or (root / 'cuda-single-probe-result.json').exists() or (root / 'cuda-subtasks-summary.json').exists():
        raise FileExistsError('this manifest has already been used for CUDA diagnostics')
    for task in TASKS:
        if (root / 'cuda-subtasks' / task.task_id).exists():
            raise FileExistsError('existing task evidence; fresh manifest required: ' + task.task_id)
        check_admission(manifest, task, dispatch_installed=True)
    adapter = adapter_factory(manifest, approved_sha256, dispatch_installed=True, verify_native_evidence=True)
    results = {}
    stopped = None
    # Persist a single-use claim BEFORE native work; interruption never permits retry.
    publish(summary, dict(schema_version=1, state='CLAIMED', manifest_sha256=approved_sha256,
                          tasks_planned=[t.task_id for t in REMAINING], legacy_04_cuda_accepted=False,
                          suite_complete=False, recorded_at=now()), new=True)
    for task in REMAINING:
        if (root / 'stop-request.json').exists():
            stopped = 'stop_requested'
            break
        try:
            result = validate_terminal(task.task_id, adapter(task))
        except (Exception, KeyboardInterrupt) as exc:
            results[task.task_id] = dict(task_id=task.task_id, status='FAIL', reason='unverified_supervisor_exception:' + type(exc).__name__)
            stopped = 'unverified_failure'
            break
        results[task.task_id] = result
        # Conservative batch policy: do not proceed after ANY non-PASS.
        if result['status'] != 'PASS' or result.get('supervision_verified') is not True or result.get('cleanup_ok') is not True or result.get('native_crash') is True or result.get('resource_pressure') is True:
            stopped = 'non_pass_or_unverified_receipt'
            break
    output = dict(schema_version=1, state='DIAGNOSTIC_PASS' if not stopped and len(results) == 21 else 'DIAGNOSTIC_INCOMPLETE',
                  tasks_planned=[t.task_id for t in REMAINING], tasks_attempted=list(results),
                  results=results, stopped_reason=stopped, legacy_04_cuda_accepted=False,
                  suite_complete=False, other_cuda_subtasks_attempted=0, finished_at=now())
    # Claim remains immutable; completion is a separate, exclusive artifact.
    publish(root / 'cuda-remaining-probe-result.json', output, new=True)
    return output
