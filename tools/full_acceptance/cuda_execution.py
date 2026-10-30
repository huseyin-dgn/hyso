"""Explicit opt-in diagnostic orchestration of isolated CUDA subtasks.

This does not certify legacy 04.cuda acceptance. The controller still owns the
existing Windows Job supervisor; no nested CUDA work runs concurrently.
"""
from pathlib import Path
from .core import publish, sha
from .cuda_tasks import TASKS
from .cuda_scheduler import CUDAOrchestrator
from .cuda_supervisor_adapter import WindowsJobCudaAdapter, check_admission


def run_diagnostic(manifest, *, manifest_path, stop_file, supervise_fn=None):
    """Return a top-level experiment result without ever claiming 04.cuda PASS.

    Must only be called after the legacy decision() says 04.cuda is READY.
    The caller must also have verified the immutable live manifest/snapshot.
    """
    manifest_path = Path(manifest_path).resolve()
    expected_manifest = Path(manifest['output']).resolve() / 'manifest.json'
    if manifest_path != expected_manifest:
        raise ValueError('diagnostic manifest must be the exact sealed run manifest')
    if not manifest_path.is_file():
        raise FileNotFoundError('missing sealed manifest')
    if Path(stop_file).resolve() != Path(manifest['output']).resolve() / 'stop-request.json':
        raise ValueError('stop file outside sealed output')
    for task in TASKS:
        check_admission(manifest, task, dispatch_installed=True)
    if Path(stop_file).exists():
        raise RuntimeError('stop requested before CUDA diagnostic')

    # This is an explicit execution route, not an adapter_verified proof of
    # empirical native correctness. The result never upgrades 04.cuda to PASS.
    from .safety import supervise as actual_supervise
    supervisor = supervise_fn if supervise_fn is not None else actual_supervise

    def owned_supervise(m, e, *, operation, output):
        if Path(stop_file).exists():
            raise RuntimeError('stop requested between CUDA subtasks')
        return supervisor(m, e, operation=operation, output=output, stop_file=stop_file)

    adapter = WindowsJobCudaAdapter(manifest, sha(manifest_path),
                                    dispatch_installed=True, supervise=owned_supervise,
                                    verify_native_evidence=supervise_fn is None)
    summary = CUDAOrchestrator(adapter).run(approved=True, adapter_verified=True)
    destination = Path(manifest['output']) / 'cuda-subtasks-summary.json'
    publish(destination, summary, new=True)
    # A subtask's PASS does NOT prove matched legacy quality, performance,
    # preservation, statistical acceptance, or independent GPU validation.
    result_status = 'FAIL' if summary['status'] in ('FAIL', 'ABORTED') else 'UNSUPPORTED'
    return dict(status=result_status,
                reason='isolated_cuda_diagnostic_only_legacy_acceptance_not_validated',
                metrics=[], criteria={'legacy_04_cuda_accepted': False},
                attempted=False,  # never mark full legacy stage attempted/complete
                diagnostic_subtasks_attempted=sum(v['status'] != 'NOT_RUN'
                                                 for v in summary['results'].values()),
                diagnostic_subtasks_total=len(TASKS),
                diagnostic_status=summary['status'],
                diagnostic_summary=str(destination),
                native_crash_stop=summary.get('stopped_reason') == 'native_crash_stop')
