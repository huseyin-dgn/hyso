"""Strict admission checks for a future CUDA subtask Windows Job route.

This module does not spawn a process, import torch, authorize execution,
modify an existing acceptance manifest, or enable a new CLI operation.
"""
from pathlib import Path
from typing import Mapping
from .cuda_tasks import TASK_BY_ID

OPERATION = 'inference_cuda_subtask'


class UninstalledSubtaskRoute(RuntimeError):
    pass


def validate_subtask_job_request(manifest: Mapping, experiment: Mapping,
                                 request: Mapping, *, route_installed: bool = False):
    """Validate a candidate job request against the sealed parent manifest.

    This is an additional check. The existing Job membership gate, manifest
    source-hash verification, and process cleanup remain mandatory.
    """
    ident = experiment.get('id')
    task = TASK_BY_ID.get(ident)
    if task is None:
        raise ValueError('unknown CUDA task identity')
    if experiment.get('parent_id') != '04.cuda':
        raise ValueError('subtask parent mismatch')
    parent = next((e for e in manifest.get('experiments', ())
                   if e.get('id') == '04.cuda'), None)
    if not parent or parent.get('handler') != 'inference_cuda':
        raise ValueError('missing or incompatible parent experiment')
    if parent.get('authorization', {}).get('approved') is not True:
        raise PermissionError('parent CUDA acceptance is not authorized')
    if experiment.get('handler') != OPERATION or request.get('operation') != OPERATION:
        raise ValueError('operation mismatch')
    if request.get('experiment_id') != ident:
        raise ValueError('request identity mismatch')
    if experiment.get('arm') != task.arm or experiment.get('measurement') != task.measurement:
        raise ValueError('arm or measurement mismatch')
    expected_gpu = manifest.get('hardware', {}).get('gpu_uuid')
    if not isinstance(expected_gpu, str) or not expected_gpu.startswith('GPU-'):
        raise ValueError('missing physical GPU UUID')
    if request.get('gpu_uuid') != expected_gpu:
        raise ValueError('GPU UUID binding mismatch')
    root = Path(manifest['output']).resolve()
    expected_dir = root / 'cuda-subtasks' / task.task_id
    stage_dir = Path(experiment['output']).resolve()
    if stage_dir != expected_dir or Path(request.get('output', '')).resolve() != expected_dir:
        raise ValueError('noncanonical CUDA subtask evidence directory')
    if request.get('manifest') != str(root / 'manifest.json'):
        raise ValueError('manifest location mismatch')
    if request.get('snapshot') != manifest['source_snapshot']['root']:
        raise ValueError('source snapshot location mismatch')
    if not isinstance(request.get('manifest_sha256'), str) or len(request['manifest_sha256']) != 64:
        raise ValueError('missing manifest hash')
    # The caller must still compare this hash with the actual manifest bytes.
    if not route_installed:
        raise UninstalledSubtaskRoute('CLI worker dispatch and GPU visibility route not installed')
    return dict(task_id=ident, device='cuda:0', gpu_uuid=expected_gpu,
                operation=OPERATION, scope='admission_only_not_execution_proof')
