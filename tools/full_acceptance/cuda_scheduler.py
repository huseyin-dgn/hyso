"""P0 step 2: fail-closed orchestration for CUDA subtask receipts.

No GPU imports, process creation or live execution in this module. A future
adapter must invoke existing safety.supervise with one owned Windows Job PER
subtask and return its verified terminal. This scheduler never treats injected
callbacks as proof of real execution.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional

from .cuda_tasks import TASKS, CudaTask, aggregate, validate_plan, validate_terminal


@dataclass(frozen=True)
class RunPolicy:
    # GPU work cannot overlap on the same physical device.
    max_active_gpu: int = 1
    stop_on_native_crash: bool = True
    stop_on_resource_pressure: bool = True

    def __post_init__(self):
        if self.max_active_gpu != 1:
            raise ValueError('CUDA measurement isolation requires exactly one GPU owner')


class UnsafeExecutionError(RuntimeError):
    pass


class CUDAOrchestrator:
    """Sequential GPU admission and independent diagnostic terminal collection.

    `invoke` is a trusted, externally provided per-task supervisor adapter.
    It MUST independently verify manifest binding, Windows Job ownership,
    cleanup, source/checkpoint/GPU identity and output hashes. No such adapter
    is installed by this step; the legacy 04.cuda path is unchanged.
    """

    def __init__(self, invoke: Callable[[CudaTask], Mapping], *, policy: RunPolicy = RunPolicy()):
        if not callable(invoke):
            raise TypeError('supervisor callback required')
        self.invoke = invoke
        self.policy = policy
        self._running = False

    def run(self, *, tasks=TASKS, approved: bool = False, adapter_verified: bool = False) -> Dict:
        if not approved or not adapter_verified:
            raise PermissionError('explicit approval and verified supervisor adapter required')
        if self._running:
            raise RuntimeError('orchestrator reentry forbidden')
        schedule = validate_plan(tasks)
        receipts = {}
        stopped_reason: Optional[str] = None
        self._running = True
        try:
            for task in schedule:
                if stopped_reason:
                    receipts[task.task_id] = dict(task_id=task.task_id, status='NOT_RUN', reason=stopped_reason)
                    continue
                # A task cannot overlap a predecessor: call is synchronous.
                try:
                    result = dict(self.invoke(task))
                    validated = validate_terminal(task.task_id, result)
                except (Exception, KeyboardInterrupt) as exc:
                    # Never call this PASS; do not trigger GPU retry.
                    validated = dict(task_id=task.task_id, status='FAIL',
                                     reason='supervisor_exception:'+type(exc).__name__)
                    stopped_reason = 'unverified_supervisor_failure'
                receipts[task.task_id] = validated
                if validated.get('native_crash') is True and self.policy.stop_on_native_crash:
                    stopped_reason = 'native_crash_stop'
                if validated.get('resource_pressure') is True and self.policy.stop_on_resource_pressure:
                    stopped_reason = 'resource_pressure_stop'
                if validated['status'] == 'ABORTED':
                    stopped_reason = 'aborted_stop'
                # A normal FAIL remains isolated; continue only if verified to be local.
                if validated['status'] == 'FAIL' and validated.get('isolated_failure') is not True:
                    stopped_reason = stopped_reason or 'unverified_failure_stop'
                if validated['status'] == 'PASS' and validated.get('supervision_verified') is not True:
                    receipts[task.task_id] = dict(task_id=task.task_id, status='FAIL',reason='missing_supervision_proof')
                    stopped_reason = 'unverified_supervisor_failure'
        finally:
            self._running = False
        outcome = aggregate(receipts, scheduled=schedule)
        outcome['stopped_reason'] = stopped_reason
        outcome['concurrency_policy'] = 'single_GPU_owner_no_parallel_measurements'
        outcome['supervisor_adapter_verified_by_runtime'] = False
        outcome['legacy_stage_pass_authorized'] = False
        return outcome
