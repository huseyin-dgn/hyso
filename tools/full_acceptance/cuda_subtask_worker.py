"""CUDA subtask execution placeholder — intentionally non-executable in 4B.

Step 4C will replace this function with true isolated measurements. Never call
legacy handlers.inference() here: it executes *every* metric for both arms.
"""
from .cuda_tasks import TASK_BY_ID


def run(manifest, stage_output, route):
    if route.task_id not in TASK_BY_ID or TASK_BY_ID[route.task_id].arm != route.arm:
        raise ValueError('subtask identity mismatch')
    return {
        'status': 'UNSUPPORTED',
        'reason': 'subtask_measurement_dispatch_not_installed_step4C',
        'metrics': [],
        'criteria': {'single_measurement_executed': False},
        'evidence': [],
    }
