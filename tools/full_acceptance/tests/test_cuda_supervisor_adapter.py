"""Offline fail-closed integration tests: no subprocesses, GPU, Torch or WinAPI."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from full_acceptance.cuda_tasks import TASKS
from full_acceptance.cuda_supervisor_adapter import (
    check_admission, build_experiment, receipt_from_supervisor,
    WindowsJobCudaAdapter, SubtaskDispatchNotInstalled,
)

TASK = TASKS[0]
SHA = 'abc123'


def fixture(root, approved=True):
    return dict(output=str(root), hardware=dict(gpu_uuid='GPU-pinned'), experiments=[
        dict(id='04.cuda', handler='inference_cuda', outer_seconds=900,
             authorization=dict(approved=approved))])


def observed(status='PASS', *, native_crash=False, exit_code=0, reason=None,
             cleanup=0, evidence=True):
    return dict(id=TASK.task_id, status=status, reason=reason,
        attempted=True, result_file='result.json' if evidence else None,
        evidence_directory='evidence' if evidence else None,
        supervisor=dict(experiment=TASK.task_id, manifest_sha256=SHA,
                        exit_code=exit_code, native_crash=native_crash,
                        reason=None, cleanup=dict(active_processes=cleanup)))


class SupervisorAdapterTests(unittest.TestCase):
    def test_dispatch_disabled_by_default(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(SubtaskDispatchNotInstalled):
                check_admission(fixture(tmp), TASK)

    def test_requires_explicit_parent_approval(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(PermissionError):
                check_admission(fixture(tmp, False), TASK, dispatch_installed=True)

    def test_missing_gpu_rejected(self):
        with TemporaryDirectory() as tmp:
            m=fixture(tmp);m['hardware']['gpu_uuid']=None
            with self.assertRaises(ValueError):
                check_admission(m,TASK,dispatch_installed=True)

    def test_distinct_experiment_and_exclusive_output(self):
        with TemporaryDirectory() as tmp:
            m=fixture(tmp);p=Path(tmp)/'new'
            e=build_experiment(m,TASK,p,dispatch_installed=True)
            self.assertEqual(e['id'],TASK.task_id)
            self.assertNotEqual(e['handler'],'inference_cuda')
            p.mkdir()
            with self.assertRaises(FileExistsError):
                build_experiment(m,TASK,p,dispatch_installed=True)

    def test_verified_pass(self):
        r=receipt_from_supervisor(TASK,observed(),manifest_sha256=SHA)
        self.assertEqual(r['status'],'PASS')
        self.assertTrue(r['supervision_verified'])

    def test_false_pass_rejected(self):
        for kw in (dict(native_crash=True),dict(exit_code=1),dict(cleanup=1),dict(evidence=False)):
            with self.subTest(kw=kw),self.assertRaises(ValueError):
                receipt_from_supervisor(TASK,observed(**kw),manifest_sha256=SHA)

    def test_hash_mismatch(self):
        with self.assertRaises(ValueError):
            receipt_from_supervisor(TASK,observed(),manifest_sha256='wrong')

    def test_native_crash_is_stop_signal(self):
        r=receipt_from_supervisor(TASK,observed('FAIL',native_crash=True,exit_code=-1,reason='crash'),manifest_sha256=SHA)
        self.assertTrue(r['native_crash'])
        self.assertFalse(r['isolated_failure'])

    def test_isolated_python_failure(self):
        r=receipt_from_supervisor(TASK,observed('FAIL',exit_code=1,reason='ValueError'),manifest_sha256=SHA)
        self.assertTrue(r['isolated_failure'])

    def test_adapter_never_calls_supervisor_without_dispatch(self):
        with TemporaryDirectory() as tmp:
            fn=Mock()
            with self.assertRaises(SubtaskDispatchNotInstalled):
                WindowsJobCudaAdapter(fixture(tmp),SHA,supervise=fn)(TASK)
            fn.assert_not_called()

    def test_adapter_uses_exact_owned_operation_when_enabled(self):
        with TemporaryDirectory() as tmp:
            fn=Mock(return_value=observed())
            adapter=WindowsJobCudaAdapter(fixture(tmp),SHA,dispatch_installed=True,supervise=fn)
            self.assertEqual(adapter(TASK)['status'],'PASS')
            args, kwargs=fn.call_args
            self.assertEqual(kwargs['operation'],'inference_cuda_subtask')
            self.assertEqual(args[1]['id'],TASK.task_id)
