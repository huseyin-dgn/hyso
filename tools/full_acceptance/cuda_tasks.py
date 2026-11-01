"""P0 step 1: declarative CUDA subtask contract (no execution or GPU imports).

The legacy 04.cuda experiment remains authoritative until the separate supervisor,
worker dispatch, and manifest migration are implemented and reviewed.
"""
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

SCHEMA_VERSION = 1
GPU_RESOURCE = 'exclusive_cuda_0'

@dataclass(frozen=True)
class CudaTask:
    task_id: str
    arm: str
    measurement: str
    resource: str = GPU_RESOURCE
    depends_on: Tuple[str, ...] = ()

    def __post_init__(self):
        if not self.task_id or self.arm not in ('A', 'B', 'shared'):
            raise ValueError('invalid CUDA task identity')
        if self.resource != GPU_RESOURCE:
            raise ValueError('unapproved GPU resource')

# Explicit order is stable for deterministic evidence and matched A/B comparisons.
# Precision-quality and precision sweep are different protocols.
_PER_ARM = (
    'latency', 'throughput', 'utilization', 'quality',
    'precision_quality', 'footprint', 'batch_sweep',
    'resolution_sweep', 'precision_sweep', 'pipeline',
)


def planned_tasks() -> Tuple[CudaTask, ...]:
    tasks = [CudaTask(f'04.cuda.{arm}.{measurement}', arm, measurement)
             for arm in ('A', 'B') for measurement in _PER_ARM]
    for label in ('pageable_blocking', 'pinned_blocking', 'pinned_nonblocking'):
        tasks.append(CudaTask(f'04.cuda.shared.transfer_{label}', 'shared', 'transfer'))
    return tuple(tasks)


TASKS = planned_tasks()
TASK_BY_ID = {task.task_id: task for task in TASKS}
STATUSES = frozenset({'PASS', 'FAIL', 'UNSUPPORTED', 'ABORTED', 'NOT_RUN'})


def validate_plan(tasks: Iterable[CudaTask] = TASKS) -> Tuple[CudaTask, ...]:
    tasks = tuple(tasks)
    lookup = {task.task_id: task for task in tasks}
    if len(lookup) != len(tasks):
        raise ValueError('duplicate CUDA task id')
    if set(lookup) != set(TASK_BY_ID):
        raise ValueError('missing or extra CUDA subtask')
    for task in tasks:
        if task != TASK_BY_ID[task.task_id]:
            raise ValueError('modified task contract')
        if task.task_id in task.depends_on or any(dep not in lookup for dep in task.depends_on):
            raise ValueError('invalid dependencies')
    return tasks


def validate_terminal(task_id: str, receipt: Mapping) -> Dict:
    """Accept only explicit, independently identified terminal receipts.

    This is *not* evidence of actual execution or a security proof. Worker
    identity, Job membership, manifest SHA, artifact hashes and GPU UUID must
    be checked by the future supervisor before accepting this receipt.
    """
    if task_id not in TASK_BY_ID or receipt.get('task_id') != task_id:
        raise ValueError('CUDA task/receipt identity mismatch')
    status = receipt.get('status')
    if status not in STATUSES:
        raise ValueError('invalid CUDA task status')
    if status == 'PASS' and (receipt.get('error') or receipt.get('cleanup_ok') is not True):
        raise ValueError('PASS without clean terminal/cleanup')
    if status in ('FAIL', 'ABORTED', 'UNSUPPORTED', 'NOT_RUN') and not receipt.get('reason'):
        raise ValueError('non-PASS result requires reason')
    return dict(receipt)


def aggregate(receipts: Mapping[str, Mapping], *, scheduled: Iterable[CudaTask] = TASKS) -> Dict:
    """Fail closed: never upgrade a partial execution to a legacy 04.cuda PASS."""
    tasks = validate_plan(scheduled)
    unexpected = set(receipts).difference(task.task_id for task in tasks)
    if unexpected:
        raise ValueError('unexpected CUDA subtask receipts')
    valid = {task.task_id: validate_terminal(task.task_id, receipts[task.task_id])
             for task in tasks if task.task_id in receipts}
    missing = [task.task_id for task in tasks if task.task_id not in valid]
    status = ('FAIL' if any(r['status'] == 'FAIL' for r in valid.values()) else
              'ABORTED' if any(r['status'] == 'ABORTED' for r in valid.values()) else
              'INCOMPLETE' if missing or any(r['status'] in ('NOT_RUN', 'UNSUPPORTED') for r in valid.values()) else
              'PASS' if all(r['status'] == 'PASS' for r in valid.values()) else 'INCOMPLETE')
    return {'schema_version': SCHEMA_VERSION, 'stage': '04.cuda', 'status': status,
            'subtask_total': len(tasks), 'terminal_count': len(valid),
            'missing_task_ids': missing,
            'results': valid,
            'legacy_stage_pass_authorized': False,
            'note': 'Aggregation is diagnostic only until supervised native execution and legacy criteria validation are integrated.'}
