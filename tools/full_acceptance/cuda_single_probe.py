"""One-shot, narrowly scoped CUDA native diagnostic; never accepts legacy stage."""
from pathlib import Path
from .core import verify, sha, publish, now
from .cuda_tasks import TASK_BY_ID
from .cuda_supervisor_adapter import WindowsJobCudaAdapter, check_admission

ONLY_TASK_ID = '04.cuda.A.latency'


def probe(manifest_path, approved_sha256, *, task_id=ONLY_TASK_ID, adapter_factory=WindowsJobCudaAdapter):
    if not isinstance(approved_sha256, str) or len(approved_sha256) != 64:
        raise ValueError('exact approved manifest SHA256 is required')
    manifest_path = Path(manifest_path).resolve()
    m = verify(manifest_path, expected=approved_sha256, live=True)
    root = Path(m['output']).resolve()
    if manifest_path != root / 'manifest.json':
        raise ValueError('noncanonical manifest path')
    if sha(manifest_path) != approved_sha256:
        raise ValueError('manifest hash changed')
    if (root / 'STARTED.json').exists():
        raise FileExistsError('existing execution started; use new manifest')
    if (root / 'stop-request.json').exists():
        raise PermissionError('stop already requested; use fresh manifest')
    if task_id not in ('04.cuda.A.latency', '04.cuda.B.latency'):
        raise ValueError('single probe only permits Model A/B latency')
    task = TASK_BY_ID[task_id]
    check_admission(m, task, dispatch_installed=True)
    evidence = root / 'cuda-subtasks' / task.task_id
    if evidence.exists():
        raise FileExistsError('single-use task evidence already exists; no native retry')
    if (root / 'cuda-single-probe-result.json').exists():
        raise FileExistsError('single-use probe already recorded')
    adapter = adapter_factory(m, approved_sha256, dispatch_installed=True)
    # Real WindowsJobCudaAdapter performs native evidence integrity verification
    # after the owned Windows Job supervisor has returned.
    result = adapter(task)
    if result.get('task_id') != task.task_id:
        raise ValueError('unexpected task result identity')
    if result.get('status') == 'PASS' and result.get('supervision_verified') is not True:
        raise ValueError('unverified subtask PASS')
    output = dict(schema_version=1, task_id=task.task_id,
                  status='DIAGNOSTIC_PASS' if result.get('status') == 'PASS' else 'DIAGNOSTIC_'+result.get('status', 'FAIL'),
                  native_subtask_status=result.get('status'),
                  subtask=result, legacy_04_cuda_accepted=False, suite_complete=False,
                  other_cuda_subtasks_attempted=0, recorded_at=now())
    publish(root / 'cuda-single-probe-result.json', output, new=True)
    return output
