"""P0 step 3: fail-closed bridge to the existing Windows Job supervisor.

This module intentionally DOES NOT install a worker dispatch route. Step 4 must
register and review an explicitly subtask-scoped worker operation before enabling
native execution. Importing this module never starts processes or CUDA.
"""
from pathlib import Path
import os
from typing import Callable, Mapping

from .cuda_tasks import TASK_BY_ID, CudaTask, validate_terminal


class SubtaskDispatchNotInstalled(RuntimeError):
    """Do not run a legacy inference_cuda handler under a subtask identity."""


def check_admission(manifest: Mapping, task: CudaTask, *, dispatch_installed: bool = False):
    """Pure admission gate; no filesystem changes or native imports."""
    if task.task_id not in TASK_BY_ID or TASK_BY_ID[task.task_id] != task:
        raise ValueError('unrecognized or modified CUDA subtask')
    experiments = {e['id']: e for e in manifest['experiments']}
    parent = experiments.get('04.cuda')
    if parent is None or not parent['authorization']['approved']:
        raise PermissionError('04.cuda has no explicit authorization')
    if parent['handler'] != 'inference_cuda':
        raise ValueError('04.cuda handler contract changed')
    if not manifest.get('hardware', {}).get('gpu_uuid'):
        raise ValueError('manifest has no pinned physical GPU UUID')
    if not dispatch_installed:
        raise SubtaskDispatchNotInstalled('subtask worker dispatch is not installed; legacy handler forbidden')


def build_experiment(manifest: Mapping, task: CudaTask, stage_output: Path,
                     *, dispatch_installed: bool = False) -> dict:
    """Produce a dedicated supervisor experiment, never a legacy 04.cuda alias."""
    check_admission(manifest, task, dispatch_installed=dispatch_installed)
    if stage_output.exists():
        raise FileExistsError('CUDA subtask output must be new and exclusive')
    parent = next(e for e in manifest['experiments'] if e['id'] == '04.cuda')
    return dict(id=task.task_id, stage='04', handler='inference_cuda_subtask',
                output=str(stage_output), outer_seconds=parent['outer_seconds'],
                parent_id='04.cuda', arm=task.arm, measurement=task.measurement)


def receipt_from_supervisor(task: CudaTask, observed: Mapping, *, manifest_sha256: str) -> dict:
    """Validate existing safety.supervise receipt; no inferred PASS on empty evidence.

    This is a defensive *adapter-level* check, not cryptographic evidence
    validation. The existing supervisor performs Job admission and cleanup.
    """
    if observed.get('id') != task.task_id:
        raise ValueError('supervisor result belongs to another CUDA subtask')
    supervisor = observed.get('supervisor')
    if not isinstance(supervisor, Mapping):
        raise ValueError('missing Windows Job supervisor receipt')
    if supervisor.get('experiment') != task.task_id or supervisor.get('manifest_sha256') != manifest_sha256:
        raise ValueError('supervisor identity/hash mismatch')
    cleanup = supervisor.get('cleanup')
    clean = isinstance(cleanup, Mapping) and cleanup.get('active_processes') == 0
    native_crash = supervisor.get('native_crash') is True
    reason = observed.get('reason') or supervisor.get('reason')
    status = observed.get('status')
    good_exit = supervisor.get('exit_code') == 0 and supervisor.get('reason') is None
    if status == 'PASS' and (not good_exit or not clean or native_crash or
                             not observed.get('result_file') or not observed.get('evidence_directory') or
                             observed.get('attempted') is not True):
        raise ValueError('unverified PASS from supervised subtask')
    if status not in ('PASS', 'FAIL', 'PARTIAL', 'UNSUPPORTED', 'ABORTED'):
        raise ValueError('unexpected native subtask status')
    if status == 'PARTIAL':
        status = 'UNSUPPORTED'
        reason = reason or 'partial_native_subtask_not_a_pass'
    if status != 'PASS' and not reason:
        reason = 'native_subtask_failed_without_reason'
    isolated = (status == 'FAIL' and not native_crash and clean and
                supervisor.get('reason') is None and supervisor.get('exit_code') in (0, 1) and
                bool(observed.get('result_file')))
    record = dict(task_id=task.task_id, status=status, reason=reason,
                  cleanup_ok=clean, supervision_verified=(good_exit and clean and not native_crash),
                  native_crash=native_crash,
                  resource_pressure=supervisor.get('reason') in ('OS_low_memory_before_launch', 'Windows_OS_low_memory'),
                  isolated_failure=isolated, evidence_directory=observed.get('evidence_directory'))
    return validate_terminal(task.task_id, record)


class WindowsJobCudaAdapter:
    """Disabled by default until step 4 integrates manifest-bound worker dispatch.

    `supervise` is injectable only for offline testing. Native operation additionally
    requires the reviewed step-4 dispatch gate and a Windows host. Nothing in
    this class authorizes CUDA work or overrides the immutable source snapshot.
    """
    def __init__(self, manifest: Mapping, manifest_sha256: str, *,
                 dispatch_installed: bool = False, supervise: Callable = None,
                 verify_native_evidence: bool = False):
        self.manifest = manifest
        self.manifest_sha256 = manifest_sha256
        self.dispatch_installed = dispatch_installed
        self._supervise = supervise
        self.verify_native_evidence = verify_native_evidence

    def __call__(self, task: CudaTask) -> dict:
        check_admission(self.manifest, task, dispatch_installed=self.dispatch_installed)
        if os.name != 'nt' and self._supervise is None:
            raise OSError('Windows Job supervisor requires Windows')
        root = Path(self.manifest['output']) / 'cuda-subtasks' / task.task_id
        experiment = build_experiment(self.manifest, task, root,
                                      dispatch_installed=self.dispatch_installed)
        if self._supervise is None:
            from .safety import supervise
            supervise_fn = supervise
        else:
            supervise_fn = self._supervise
        observed = supervise_fn(self.manifest, experiment, operation='inference_cuda_subtask', output=root)
        terminal = receipt_from_supervisor(task, observed, manifest_sha256=self.manifest_sha256)
        if self.verify_native_evidence and terminal['status'] == 'PASS':
            from .cuda_evidence_integrity import verify_native_pass
            verify_native_pass(task.task_id, self.manifest_sha256, observed, root)
        return terminal
