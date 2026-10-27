"""Explicit short integration harness, real Windows processes/Jobs, CPU only.

Invokes start_once with a test-only CPU recipe adapter and four-second protocol;
does not invoke the pinned GPU preflight or --start-two-hours entry point.
"""
import argparse
import csv
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]


def load_runtime():
    sys.path.insert(0, str(REPO))
    import hyso.performance as performance
    from hyso.performance import stability, _training_process
    return performance, stability, _training_process


def child(root):
    # Admission handshake ensures the outer watchdog owns us before imports.
    deadline = time.monotonic() + 10
    while not (root/'admitted').exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('outer Job admission missing')
        time.sleep(.01)
    from unittest.mock import patch
    import contextlib
    performance, stability, _ = load_runtime()
    import torch
    assert not torch.cuda.is_initialized()
    spec = importlib.util.spec_from_file_location('native_launcher', HERE/'launch_p31.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    recipe_class = performance.WorkloadRecipe
    def cpu_recipe(*args, **kwargs):
        return recipe_class('native_cpu_fixture:make_cpu', search_path=str(HERE))
    protocol = performance.StabilityProtocol(duration_seconds=4, interval_seconds=.05,
        startup_timeout_seconds=45, step_timeout_seconds=3, chunk_rows=1, max_samples=200)
    reports = []
    for scenario in ('interrupt_export_ok', 'json_export_failure', 'secondary_interrupt'):
        launcher.RUN = root/scenario
        launcher.MARKER = root/(scenario+'-STARTED.json')
        journal = launcher.RUN/'journal'
        original = KeyboardInterrupt('primary controlled native interruption: '+scenario)
        fired = []
        native_sleep = time.sleep
        def interrupt_after_committed_rows(seconds):
            if not fired and (journal/'chunk-000001.json').is_file():
                fired.append(True)
                raise original
            native_sleep(seconds)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(performance, 'WorkloadRecipe', cpu_recipe))
            stack.enter_context(patch.object(stability.time, 'sleep', interrupt_after_committed_rows))
            if scenario == 'json_export_failure':
                stack.enter_context(patch.object(performance.RuntimeResult, 'to_json',
                    side_effect=OSError('injected native JSON export failure')))
            elif scenario == 'secondary_interrupt':
                stack.enter_context(patch.object(performance.RuntimeResult, 'to_json',
                    side_effect=KeyboardInterrupt('secondary finalization interruption')))
            try:
                launcher.start_once(dict(gpu_uuid=None, protocol=asdict(protocol)),
                    dict(manifest_sha256='smoke-only-not-a-long-run-manifest',
                         factory_sha256=launcher.sha(HERE/'native_cpu_fixture.py')))
            except KeyboardInterrupt as error:
                assert error is original, 'original exception replaced'
            else:
                raise AssertionError('controlled interruption did not occur')
        receipt = launcher.read(launcher.RUN/'launcher-interruption.json')
        status = launcher.read(launcher.RUN/'launcher-status.json')
        controller = launcher.read(journal/'controller.json')
        result = performance.reopen_stability(journal)
        data = result.to_dict()
        assert fired and controller['reason'] == 'controller_interrupted'
        assert controller['cleanup']['active_processes'] == 0
        assert controller['exit_code'] == 0  # TerminateJobObject's forced code.
        assert receipt['controller'] == status['controller'] == controller
        assert receipt['phase'] == 'export_pending'
        assert status['status'] == 'INTERRUPTED_OR_FAILED'
        assert status['P31'] == 'PARTIAL_PENDING_REVIEW'
        assert not status['worker_exit_code_is_completion_evidence']
        assert data['analysis']['status'] == 'aborted'
        assert data['analysis']['duration_seconds'] is None
        assert data['evidence']['header']['workload']['device_id'] == 'cpu'
        assert len(data['evidence']['chunks']) >= 2
        if scenario == 'interrupt_export_ok':
            for suffix in ('.json', '.csv', '-diagnostics.json'):
                assert (launcher.RUN/('interrupted-or-failed'+suffix)).is_file()
            performance.RuntimeResult.from_json(launcher.RUN/'interrupted-or-failed.json')
            # P31 CSV is a flattened evidence export, not a from_csv-supported
            # P34/P35 envelope. Validate the actual legacy columns and values.
            with (launcher.RUN/'interrupted-or-failed.csv').open(newline='', encoding='utf-8') as handle:
                reader = csv.DictReader(handle)
                assert reader.fieldnames == ['feature_id', 'result_id', 'source_sha256', 'field', 'value']
                csv_rows = list(reader)
            assert csv_rows and all(row['feature_id'] == 'P31' for row in csv_rows)
            fields = {row['field']: row['value'] for row in csv_rows}
            assert fields['analysis.status'] == 'aborted'
            assert fields['evidence.controller.cleanup.active_processes'] == '0'
        else:
            expected = 'OSError' if scenario == 'json_export_failure' else 'KeyboardInterrupt'
            assert expected in status['evidence_recovery']
            assert status['finalization_errors']
        assert not torch.cuda.is_initialized()
        reports.append(dict(scenario=scenario, status='PASS', worker_pid=controller['launcher_pid'],
            worker_exit_code=controller['exit_code'], active_processes=0,
            original_interrupt_preserved=True, analysis_status=data['analysis']['status'],
            chunks=len(data['evidence']['chunks']), cuda_initialized=False,
            finalization_errors=status['finalization_errors']))
    (root/'child-summary.json').write_text(json.dumps(dict(controller_pid=os.getpid(),
        scenarios=reports, gpu_workload=False, injected_interrupt_not_console_signal=True), indent=2))
    return 0


def parent():
    if os.name != 'nt':
        raise RuntimeError('native Windows required')
    _, _, process_module = load_runtime()
    root = Path(tempfile.mkdtemp(prefix='hyso-p31-native-finalization-'))
    print('Native smoke receipts: '+str(root), flush=True)
    job = process_module._Job(None, None, None)
    process = None
    try:
        with (root/'controller-stdout.txt').open('wb') as out, (root/'controller-stderr.txt').open('wb') as err:
            process = subprocess.Popen([sys.executable, '-I', '-B', str(Path(__file__).resolve()),
                '--child', str(root)], stdout=out, stderr=err,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), creationflags=subprocess.CREATE_NO_WINDOW)
            job.assign(process)
            (root/'admitted').write_text(str(process.pid))
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                job.terminate()
                process.wait(timeout=5)
                raise TimeoutError('90-second native smoke watchdog')
        assert process.returncode == 0, (root/'controller-stderr.txt').read_text()
        summary = json.loads((root/'child-summary.json').read_text())
        deadline = time.monotonic()+5
        cleanup = job.observation()
        while cleanup['active_processes'] and time.monotonic()<deadline:
            time.sleep(.01)
            cleanup = job.observation()
        assert cleanup['active_processes'] == 0
        summary.update(status='PASS', controller_exit_code=process.returncode,
                       outer_job_cleanup=cleanup, evidence_directory=str(root))
        (root/'native-summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        try:
            job.terminate()
            if process is not None:
                process.wait(timeout=5)
            deadline = time.monotonic()+5
            cleanup = job.observation()
            while cleanup['active_processes'] and time.monotonic()<deadline:
                time.sleep(.01)
                cleanup = job.observation()
            (root/'outer-cleanup.json').write_text(json.dumps(dict(
                controller_pid=process.pid if process else None,
                controller_exit_code=process.returncode if process else None,
                cleanup=cleanup), indent=2))
            assert cleanup['active_processes'] == 0
        finally:
            job.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', type=Path)
    args = parser.parse_args()
    if args.child:
        raise SystemExit(child(args.child))
    parent()
