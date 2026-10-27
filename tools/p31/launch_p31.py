"""Private, one-shot P31 launcher. Preflight never runs a factory or CUDA call."""
import argparse
import ast
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
PYTHON = Path(r'C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe')
COMMIT = '1ab52822c39c073eebca1d8c1d773e56c47cb728'
RUN = ROOT / 'run'
MARKER = ROOT / 'STARTED.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish_new(path, data):
    # Serialize before touching disk. Same-directory hard-link publication is
    # atomic and fails if the destination exists; never replace old receipts.
    payload = json.dumps(data, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                dir=path.parent, prefix='.p31-publish-', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # A leftover temp is preferable to hiding the write error.


def git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], check=True,
                          capture_output=True, text=True, timeout=30).stdout.strip()


def reserve_start(path, data):
    # Never delete or reuse this marker, even if startup subsequently fails.
    publish_new(path, data)


def fixed_protocol():
    return dict(duration_seconds=7200, interval_seconds=5, idle_seconds=0,
        startup_timeout_seconds=120, step_timeout_seconds=60, max_samples=2000,
        chunk_rows=12, max_journal_bytes=128*1024**2, seed=0,
        process_memory_bytes=None, job_memory_bytes=None, max_processes=None)


def source_inventory():
    # A no-commit integration is pinned by the complete tracked/untracked source
    # overlay as well as its base commit. Ignored runtime outputs are not source.
    names = git('ls-files', '--cached', '--others', '--exclude-standard').splitlines()
    return {name: sha(REPO/name) for name in sorted(set(names))}


def configure_output(directory):
    global ROOT, RUN, MARKER
    root = Path(directory).resolve()
    if root.is_relative_to(REPO) or not root.is_relative_to(Path(r'C:\Users\hdgn5\hyso-private-evidence')):
        raise RuntimeError('Use a fresh private evidence directory outside the repository')
    ROOT = root; RUN = root/'run'; MARKER = root/'STARTED.json'


def verify_source_snapshot(manifest):
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 2:
        raise RuntimeError('Integration launcher requires an explicit v2 source snapshot')
    expected_protocol = fixed_protocol()
    if manifest['commit'] != COMMIT or manifest['protocol'] != expected_protocol:
        raise RuntimeError('Fixed 7200-second protocol/commit changed; no run started')
    if git('rev-parse', 'HEAD') != COMMIT or git('status', '--porcelain=v1', '--untracked-files=all') != manifest['source_status']:
        raise RuntimeError('Pinned base/source status changed; no run started')
    if str(REPO) != manifest['repository'] or source_inventory() != manifest['source_sha256']:
        raise RuntimeError('Pinned integration source snapshot changed; no run started')


def preflight():
    manifest = read(ROOT / 'manifest.json')
    verify_source_snapshot(manifest)
    if os.name != 'nt' or Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError('Use the pinned Windows Python interpreter')
    if sys.version != manifest['python_version']:
        raise RuntimeError('Python runtime changed')
    if sha(__file__) != manifest['launcher_sha256']:
        raise RuntimeError('Launcher fingerprint changed')
    if sha(PYTHON) != manifest['python_executable_sha256']:
        raise RuntimeError('Python executable changed')
    for name, expected in manifest['packages'].items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError('Package version changed: ' + name)
    for name, expected in manifest['source_sha256'].items():
        path = REPO / name
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError('Pinned source changed: ' + name)
    reference_path = Path(manifest['reference_result'])
    if sha(reference_path) != manifest['reference_sha256']:
        raise RuntimeError('Recorded GPU reference changed')
    reference = read(reference_path)
    identity = reference['evidence']['header']['workload']
    if identity['device_id'] != manifest['gpu_uuid'] or identity['model_sha256'] != manifest['model_sha256']:
        raise RuntimeError('Reference GPU/model mismatch')
    factory_path = REPO / 'tests' / 'runtime_support.py'
    tree = ast.parse(factory_path.read_text(encoding='utf-8-sig'))
    factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'make_gpu')
    if [arg.arg for arg in factory.args.args] != ['gpu_uuid', 'cuda_memory_fraction']:
        raise RuntimeError('Unexpected factory signature')
    if not isinstance(factory.args.defaults[0], ast.Constant) or factory.args.defaults[0].value is not None:
        raise RuntimeError('Factory CUDA allocator quota is not disabled by default')
    if RUN.exists() or MARKER.exists():
        raise RuntimeError('This one-shot run is already reserved or has output; no retry/resume')
    if ROOT.is_relative_to(REPO) or not ROOT.is_relative_to(Path(r'C:\Users\hdgn5\hyso-private-evidence')):
        raise RuntimeError('Evidence must remain in the pinned private evidence directory')
    if not (REPO / 'hyso/performance/_runtime_worker.py').is_file():
        raise RuntimeError('Existing runtime worker is missing')

    # CPU-only import/protocol validation. No factory import, CUDA availability
    # probe, device query, allocation, model construction or owned worker.
    sys.path.insert(0, str(REPO))
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('Unexpected parent CUDA initialization before preflight')
    def forbidden_call(name):
        # Keep a distinct function per API: Torch Dynamo's import-time rule map
        # rejects one callable installed at multiple CUDA symbol locations.
        def forbidden(*args, **kwargs):
            raise AssertionError('Preflight attempted CUDA operation: ' + name)
        return forbidden
    names = ['_lazy_init', 'is_available', 'device_count', 'current_device',
             'get_device_properties', 'memory_stats', 'memory_allocated',
             'memory_reserved', 'synchronize', 'set_device', 'reset_peak_memory_stats',
             'set_per_process_memory_fraction']
    originals = {name: getattr(torch.cuda, name) for name in names}
    try:
        for name in names:
            setattr(torch.cuda, name, forbidden_call(name))
        from hyso.performance import StabilityProtocol, WorkloadRecipe
        from hyso.performance import stability
        if Path(stability.__file__).resolve() != (REPO / 'hyso/performance/stability.py').resolve():
            raise RuntimeError('Imported HYSo from the wrong checkout')
        protocol = StabilityProtocol(**manifest['protocol'])
        recipe = WorkloadRecipe('runtime_support:make_gpu', {'gpu_uuid': manifest['gpu_uuid']},
                                search_path=str(REPO / 'tests'), gpu_uuid=manifest['gpu_uuid'])
        if asdict(protocol) != manifest['protocol']:
            raise RuntimeError('Protocol defaults changed')
        if protocol.process_memory_bytes is not None or protocol.job_memory_bytes is not None or protocol.max_processes is not None:
            raise RuntimeError('Unexpected resource quota')
        if 'runtime_support' in sys.modules:
            raise RuntimeError('Factory module was imported during preflight')
        if torch.cuda.is_initialized():
            raise RuntimeError('Preflight initialized CUDA')
    finally:
        for name, value in originals.items():
            setattr(torch.cuda, name, value)
    report = dict(status='PREFLIGHT_PASS',time_utc=datetime.now(timezone.utc).isoformat(),
        worktree=str(REPO),commit=COMMIT,python=sys.executable,python_version=sys.version,
        packages=manifest['packages'],factory=recipe.factory,factory_path=str(factory_path),
        factory_sha256=sha(factory_path),gpu_uuid=manifest['gpu_uuid'],
        gpu_uuid_scope='Pinned from prior native P31 result; live device access deferred to explicitly started worker',
        reference_result=str(reference_path),reference_sha256=manifest['reference_sha256'],
        manifest_sha256=sha(ROOT/'manifest.json'),launcher_sha256=sha(__file__),
        journal=str(RUN/'journal'),output_directory=str(RUN),protocol=asdict(protocol),
        ready_to_exit_watchdog_seconds=7265,
        cleanup_deadlines=dict(worker_wait_seconds=5,fallback_kill_wait_seconds=5,job_observation_wait_seconds=5),
        worker_ownership='Existing Windows kill-on-close Job; active owned processes must be zero',
        requires_admin=False,occupancy_gate=False,automatic_retry=False,execution_resume=False,
        cuda_initialized=False,cuda_queries=0,factory_calls=0,owned_workers_started=0,long_run_started=False)
    report_path=ROOT/('preflight-'+uuid.uuid4().hex[:8]+'.json')
    publish_new(report_path,report)
    print(json.dumps(dict(status=report['status'],report=str(report_path),commit=COMMIT,
        gpu_uuid=manifest['gpu_uuid'],journal=report['journal'],long_run_started=False),indent=2),flush=True)
    return manifest,report


def export_result(result, stem='result'):
    result.to_json(RUN/(stem+'.json'))
    result.to_csv(RUN/(stem+'.csv'))
    diagnostics=result.stability_diagnostics()
    publish_new(RUN/(stem+'-diagnostics.json'),diagnostics)
    return diagnostics


def _error_detail(error):
    try:
        return type(error).__name__ + ': ' + str(error)
    except BaseException:
        return type(error).__name__ + ': <message unavailable>'


def _record_interruption(error, reopen):
    # Best effort only: disk failure or forced process termination cannot be
    # made durable by Python. Never replace the original exception.
    status = dict(status='INTERRUPTED_OR_FAILED', error=_error_detail(error),
        evidence_recovery=None, retry=False, P31='PARTIAL_PENDING_REVIEW',
        finalization_errors=[], controller=None,
        worker_exit_code_is_completion_evidence=False)
    try:
        status['controller'] = read(RUN/'journal/controller.json')
    except BaseException as secondary:
        status['finalization_errors'].append(dict(stage='controller_read', error=_error_detail(secondary)))
    try:
        publish_new(RUN/'launcher-interruption.json', dict(status, phase='export_pending'))
    except BaseException as secondary:
        status['finalization_errors'].append(dict(stage='interruption_receipt', error=_error_detail(secondary)))
    try:
        export_result(reopen(RUN/'journal'), 'interrupted-or-failed')
        status['evidence_recovery'] = 'verified evidence reopened; execution was not resumed'
    except BaseException as secondary:
        status['evidence_recovery'] = _error_detail(secondary)
        status['finalization_errors'].append(dict(stage='reopen_or_export', error=_error_detail(secondary)))
    try:
        publish_new(RUN/'launcher-status.json', dict(status, phase='finalization_finished'))
    except BaseException as secondary:
        status['finalization_errors'].append(dict(stage='launcher_status', error=_error_detail(secondary)))
        try:
            publish_new(RUN/'launcher-finalization-error.json', status)
        except BaseException:
            pass  # No writable destination is guaranteed; preserve original error.


def start_once(manifest,report):
    # Direct callers are test-only; the CLI always performs the pinned preflight.
    reserve_start(MARKER,dict(commit=COMMIT,manifest_sha256=report['manifest_sha256'],
        time_utc=datetime.now(timezone.utc).isoformat(),requested_seconds=7200,automatic_retry=False))
    RUN.mkdir(exist_ok=False)
    publish_new(RUN/'preflight.json',report)
    from hyso.performance import WorkloadRecipe,StabilityProtocol,measure_stability,reopen_stability
    recipe=WorkloadRecipe('runtime_support:make_gpu',{'gpu_uuid':manifest['gpu_uuid']},
        search_path=str(REPO/'tests'),gpu_uuid=manifest['gpu_uuid'])
    print('USER-STARTED P31: one 7200-second work window, 5-second observations; no retry.',flush=True)
    print('Private output: '+str(RUN),flush=True)
    try:
        result=measure_stability(recipe,RUN/'journal',protocol=StabilityProtocol(**manifest['protocol']))
        diagnostics=export_result(result)
        raw=result.to_dict()
        status=dict(status='RUN_FINISHED_PENDING_REVIEW',P31='PARTIAL_PENDING_REVIEW',
            native_analysis_status=raw['analysis']['status'],observed_work_seconds=diagnostics.get('observed_work_seconds'),
            controller_elapsed_seconds=raw['evidence']['controller']['elapsed_seconds'],
            gaps=diagnostics.get('sample_coverage',{}).get('gaps'),
            owned_active_processes=raw['evidence']['controller']['cleanup']['active_processes'],
            source_factory_matches=raw['evidence']['header']['source']['module_sha256']==report['factory_sha256']
                if raw['evidence']['header'] is not None else False,
            automatic_retry=False,execution_resume=False)
        publish_new(RUN/'launcher-status.json',status)
        print(json.dumps(status,indent=2),flush=True)
        # Analysis, not a forced worker exit code of zero, determines completion.
        return 0 if raw['analysis']['status']=='completed' else 1
    except BaseException as error:
        try:
            _record_interruption(error, reopen_stability)
        except BaseException as secondary:
            # Even an unexpected finalizer bug must not hide the triggering error.
            try:
                publish_new(RUN/'launcher-finalization-error.json', dict(
                    status='INTERRUPTED_OR_FAILED', P31='PARTIAL_PENDING_REVIEW',
                    error=_error_detail(error), finalization_error=_error_detail(secondary),
                    retry=False))
            except BaseException:
                pass
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight',action='store_true',help='CPU-only checks; no workload')
    group.add_argument('--start-two-hours',action='store_true',help='Explicitly start exactly one fixed 7200-second CUDA run')
    parser.add_argument('--evidence-root',type=Path,required=True,help='Fresh pinned manifest directory outside Git')
    args=parser.parse_args()
    configure_output(args.evidence_root)
    manifest,report=preflight()
    if args.preflight:
        return 0
    return start_once(manifest,report)


if __name__=='__main__':
    raise SystemExit(main())
