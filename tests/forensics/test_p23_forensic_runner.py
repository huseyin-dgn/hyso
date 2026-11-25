"""P23-only CPU isolation tests; stdlib supervisor, no repository conftest.

Only the explicitly named cpu-recipe case imports Torch, with devices hidden.
Artifacts are retained to review cleanup/terminal/dump evidence independently.
"""
import copy
import ctypes as c
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/forensics'))
import p23_forensic_runner as runner
import p23_forensic_windows as windows

REPO = Path(__file__).resolve().parents[2]
DESIGN = Path('C:/Users/hdgn5/AppData/Local/hyso-p30-p23-reconcile-20261110-9013b39c/p23-forensic-plan/manifest.json')


class Contracts(unittest.TestCase):
    def test_windows_layout(self):
        self.assertEqual(c.sizeof(windows.ExceptionRecord), 152)
        self.assertEqual(c.sizeof(windows.DebugEvent), 176)
        self.assertEqual(c.sizeof(windows.DumpExceptionInformation), 16)
        self.assertEqual(windows.DumpExceptionInformation.pointers.offset, 4)

    def test_transform_keeps_version_and_ownership(self):
        source = REPO/'hyso/performance/_saved_activation.py'
        text = source.read_text()
        candidate = runner.transform(source, 'saved')
        self.assertIn('payload.tensor._version != payload.version', candidate)
        self.assertIn('_Payload(tensor, self, saved_id, version)', candidate)
        # Removing just the known two guards must produce the exact candidate.
        self.assertEqual(text.count('gate unresolved'), 2)
        self.assertEqual(candidate.count('PRIVATE P23 diagnostic copy only'), 2)
        self.assertIn('saved-activation recorder is CPU-only', source.read_text())

    def test_wrong_design_hash_rejected(self):
        with self.assertRaises(ValueError):
            runner.prepare('must-never-create-p23-wrong-design', Path(__file__))
        self.assertFalse(Path('must-never-create-p23-wrong-design').exists())

    def test_collector_nonce_pid_hash_binding(self):
        request = dict(pid=71, nonce='nonce', manifest_sha256='sha', exception=dict(pid=71))
        worker = dict(pid=71, token='nonce')
        runner.validate_dump_request(request, worker, 'nonce', 'sha')
        for key, value in (('pid', 72), ('nonce', 'other'), ('manifest_sha256', 'other')):
            altered = copy.deepcopy(request); altered[key] = value
            with self.assertRaises(ValueError): runner.validate_dump_request(altered, worker, 'nonce', 'sha')
        altered = copy.deepcopy(request); altered['exception']['pid'] = 72
        with self.assertRaises(ValueError): runner.validate_dump_request(altered, worker, 'nonce', 'sha')

    def test_self_dump_forbidden(self):
        with self.assertRaisesRegex(ValueError, 'handle/PID'):
            windows.require_owned(windows.kernel().GetCurrentProcess(), 0, os.getpid())

    def test_process_outside_exact_job_forbidden(self):
        module = runner.admission.load(REPO/'hyso/performance/_training_process.py', 'p23_test_job')
        job = module._Job(4*1024**3, 8*1024**3, 4)
        process = subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c', 'import time;time.sleep(120)'],
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            with self.assertRaisesRegex(ValueError, 'outside exact owned Job'):
                windows.require_owned(int(process._handle), job.handle, process.pid)
        finally:
            process.kill(); process.wait(timeout=5); job.close()


class OwnedSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(os.environ['P23_TEST_OUTPUT'])
        cls.out.mkdir()
        cls.root = cls.out/'sealed'
        cls.manifest = runner.prepare(cls.root, DESIGN)
        cls.sha = runner.digest(cls.root/'manifest.json')

    def case(self, mode):
        work = self.root/('cpu-'+mode+'-'+uuid.uuid4().hex[:8])
        runner.controller(self.root, work, mode, self.sha)
        terminal = runner.read(work/'controller-terminal.json')
        outer = runner.read(work/'parent-receipt.json')
        self.assertFalse(terminal['active'])
        self.assertEqual(terminal['preservation'], 'PASS')
        self.assertEqual(outer['cleanup']['active_processes'], 0)
        self.assertFalse(terminal['cuda_workload_authorized'])
        self.assertEqual(terminal['p23_acceptance'], 'PARTIAL')
        self.assertFalse(runner.read(work/'controller-identity.json')['cuda_imported'])
        if mode != 'outer-timeout':
            child = runner.read(work/'worker-receipt.json')
            self.assertEqual(child['cleanup']['active_processes'], 0)
            self.assertEqual(child['nonce'], outer['nonce'])
            self.assertFalse(runner.read(work/'worker-identity.json')['cuda_imported'])
            rows = [json.loads(s) for s in (work/'worker-phases.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['pid'] == child['pid'] and r['nonce'] == child['nonce']
                                and r['manifest_sha256'] == self.sha for r in rows))
        return work, terminal

    def test_success(self):
        _, terminal = self.case('success')
        self.assertEqual(terminal['outcome'], 'DIAGNOSTIC_COMPLETE')

    def test_cpu_recipe_full_saved_payload(self):
        work, terminal = self.case('cpu-recipe')
        self.assertEqual(terminal['outcome'], 'DIAGNOSTIC_COMPLETE')
        result = runner.read(work/'result.json')
        self.assertFalse(result['cuda_initialized'])
        self.assertTrue(result['cpu_only'])
        summary = result['measurement']['activation']['summary']
        self.assertGreater(summary['pack_events'], 0)
        self.assertGreater(summary['unpack_events'], 0)
        self.assertEqual(summary['pack_events'], summary['release_events'])
        self.assertEqual(summary['live_payloads_at_end'], 0)

    def test_workload_exception_stops(self):
        work, terminal = self.case('exception')
        self.assertEqual(terminal['outcome'], 'STOP')
        self.assertNotEqual(runner.read(work/'worker-receipt.json')['exit_code'], 0)
        self.assertFalse((work/'result.json').exists())

    def test_actual_native_cpu_failfast_dump(self):
        work, terminal = self.case('failfast')
        self.assertEqual(terminal['outcome'], 'STOP')
        child = runner.read(work/'worker-receipt.json')
        self.assertEqual(child['fault']['code'], 0xc0000409)
        collector = runner.read(work/'collector-receipt.json')
        self.assertIsNone(collector['outcome'])
        self.assertEqual(collector['exit_code'], 0)
        checked = windows.validate_dump(work/'owned-worker.dmp')
        self.assertEqual(checked['exception_code'], 0xc0000409)
        self.assertEqual(checked['exception_tid'], child['fault']['tid'])
        self.assertEqual(runner.digest(work/'owned-worker.dmp'), collector['dump']['sha256'])

    def test_timeout_cleans_tree(self):
        work, terminal = self.case('timeout')
        self.assertEqual(terminal['outcome'], 'STOP')
        child = runner.read(work/'worker-receipt.json')
        self.assertEqual(child['monitor_outcome'], 'timeout')
        self.assertGreaterEqual(child['cleanup']['total_processes'], 3)

    def test_outer_watchdog_cleans_nested_tree(self):
        work, terminal = self.case('outer-timeout')
        self.assertEqual(terminal['outcome'], 'STOP')
        outer = runner.read(work/'parent-receipt.json')
        self.assertEqual(outer['monitor_outcome'], 'timeout')
        self.assertGreaterEqual(outer['cleanup']['total_processes'], 4)

    def test_real_private_commit_denial(self):
        work, terminal = self.case('memory')
        self.assertEqual(terminal['outcome'], 'DIAGNOSTIC_COMPLETE')
        self.assertIn('memory_limit.denied', (work/'worker-phases.jsonl').read_text())

    def test_descendant_cleanup_after_leader_exit(self):
        work, terminal = self.case('tree')
        self.assertEqual(terminal['outcome'], 'DIAGNOSTIC_COMPLETE')
        self.assertGreaterEqual(runner.read(work/'worker-receipt.json')['cleanup']['total_processes'], 3)

    def test_log_overflow_stops(self):
        work, terminal = self.case('logs')
        self.assertEqual(terminal['outcome'], 'STOP')
        self.assertEqual(runner.read(work/'worker-receipt.json')['monitor_outcome'], 'log_limit')

    def test_collector_timeout_stops(self):
        work, terminal = self.case('collector-timeout')
        self.assertEqual(terminal['outcome'], 'STOP')
        self.assertEqual(runner.read(work/'collector-receipt.json')['outcome'], 'collector_timeout')

    def test_collector_file_cap_stops(self):
        work, terminal = self.case('collector-limit')
        self.assertEqual(terminal['outcome'], 'STOP')
        self.assertEqual(runner.read(work/'collector-receipt.json')['outcome'], 'collector_dump_limit')

    def test_native_missing_consent_never_launches(self):
        work = self.root/'native-denied'
        work.mkdir()
        with self.assertRaises(FileNotFoundError): runner.controller(self.root, work, 'native', self.sha)
        self.assertFalse((self.root/'execution-consumed.json').exists())
        self.assertFalse((work/'controller-identity.json').exists())


if __name__ == '__main__': unittest.main(verbosity=2)
