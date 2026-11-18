"""04.cuda subtask dispatch admission, offline and fail-closed.

This module registers NO executable operation. The actual worker dispatch and
Windows Job admission must be integrated together in a subsequent reviewed step.
Never use the legacy ``inference_cuda`` handler for a subtask identity.
"""
from dataclasses import dataclass
from typing import Mapping
from .cuda_tasks import TASK_BY_ID

OPERATION = 'inference_cuda_subtask'


@dataclass(frozen=True)
class SubtaskRoute:
    task_id: str
    arm: str
    measurement: str
    operation: str = OPERATION


def resolve_subtask_route(manifest: Mapping, request: Mapping, *, admitted: bool = False) -> SubtaskRoute:
    """Validate a proposed, manifest-bound route without executing any code.

    ``admitted`` is only a prerequisite flag provided by the trusted Windows Job
    admission path; it is not cryptographic proof and cannot authorize a run.
    """
    if not admitted:
        raise PermissionError('exact Windows Job admission has not been verified')
    task_id = request.get('experiment_id')
    task = TASK_BY_ID.get(task_id)
    if task is None:
        raise ValueError('unknown CUDA subtask identity')
    if request.get('operation') != OPERATION:
        raise PermissionError('subtask requires exact isolated operation')
    if request.get('parent_id') != '04.cuda':
        raise ValueError('subtask parent mismatch')
    if request.get('arm') != task.arm or request.get('measurement') != task.measurement:
        raise ValueError('subtask identity fields mismatch')
    if not isinstance(manifest.get('experiments'), list):
        raise ValueError('missing manifest experiment registry')
    parents = [e for e in manifest['experiments'] if e.get('id') == '04.cuda']
    if len(parents) != 1 or parents[0].get('handler') != 'inference_cuda':
        raise ValueError('unexpected parent experiment')
    if parents[0].get('authorization', {}).get('approved') is not True:
        raise PermissionError('parent CUDA experiment not explicitly authorized')
    if not manifest.get('hardware', {}).get('gpu_uuid'):
        raise ValueError('physical GPU UUID is not pinned')
    if request.get('gpu_uuid') != manifest['hardware']['gpu_uuid']:
        raise ValueError('GPU UUID binding mismatch')
    if request.get('manifest_sha256') != manifest.get('_verified_manifest_sha256'):
        raise ValueError('missing verified manifest SHA binding')
    if request.get('source_tree_sha256') != manifest.get('source_snapshot', {}).get('tree_sha256'):
        raise ValueError('source snapshot binding mismatch')
    if not request.get('manifest_sha256') or not request.get('source_tree_sha256'):
        raise ValueError('empty immutable source identity')
    return SubtaskRoute(task.task_id, task.arm, task.measurement)
