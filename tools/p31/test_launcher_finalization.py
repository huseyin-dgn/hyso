"""Fault injection only: no subprocess, model factory or CUDA workload."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import hyso.performance as performance
from hyso.performance import stability, _training_process
spec = importlib.util.spec_from_file_location("p31_launcher_revision", Path(__file__).with_name("launch_p31.py"))
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

class FinalizationAbort(BaseException):
    pass

class Result:
    def __init__(self, state="completed", fail=None):
        self.state, self.fail = state, fail
    def to_json(self, path):
        if self.fail == "json": raise OSError("JSON export fault")
        Path(path).write_text(json.dumps(self.to_dict()))
    def to_csv(self, path):
        if self.fail == "csv": raise OSError("CSV export fault")
        Path(path).write_text("synthetic,csv\n")
    def stability_diagnostics(self):
        return dict(observed_work_seconds=7200 if self.state == "completed" else None,
                    sample_coverage=dict(gaps=[]))
    def to_dict(self):
        return dict(analysis=dict(status=self.state), evidence=dict(
            controller=dict(elapsed_seconds=7201, cleanup=dict(active_processes=0), exit_code=0),
            header=dict(source=dict(module_sha256="factory"))))

class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(launcher, "RUN", self.run))
        self.stack.enter_context(patch.object(launcher, "MARKER", self.root / "STARTED.json"))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(stability.subprocess, "Popen",
            side_effect=AssertionError("launcher tests must not spawn a process")))
        self.stack.enter_context(patch.object(torch.cuda, "_lazy_init",
            side_effect=AssertionError("tests must not initialize CUDA")))
        self.manifest = dict(gpu_uuid="GPU-11111111-1111-1111-1111-111111111111", protocol={})
        self.report = dict(manifest_sha256="synthetic", factory_sha256="factory")
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())
    def read(self, name):
        return json.loads((self.run / name).read_text())
    def receipt(self):
        journal = self.run / "journal"
        journal.mkdir()
        self.controller = dict(reason="controller_interrupted", exit_code=0,
                               cleanup=dict(active_processes=0))
        (journal / "controller.json").write_text(json.dumps(self.controller))
    def invoke(self, measure, reopen=None):
        with patch.object(performance, "measure_stability", measure), patch.object(
                performance, "reopen_stability", reopen or (lambda p: Result("aborted"))):
            return launcher.start_once(self.manifest, self.report)
    def interrupt(self, *args, **kwargs):
        self.receipt()
        raise self.original
    def assert_original(self, action):
        self.original = KeyboardInterrupt("original interruption")
        with self.assertRaises(KeyboardInterrupt) as caught:
            action()
        self.assertIs(caught.exception, self.original)
    def test_normal_completion_remains_pending_review(self):
        self.assertEqual(self.invoke(lambda *a, **k: Result()), 0)
        status = self.read("launcher-status.json")
        self.assertEqual(status["status"], "RUN_FINISHED_PENDING_REVIEW")
        self.assertEqual(status["P31"], "PARTIAL_PENDING_REVIEW")
        for name in ("result.json", "result.csv", "result-diagnostics.json"):
            self.assertTrue((self.run / name).is_file())
    def test_interrupt_receipt_precedes_reopen_and_preserves_cleanup(self):
        def reopen(path):
            receipt = self.read("launcher-interruption.json")
            self.assertEqual(receipt["phase"], "export_pending")
            self.assertEqual(receipt["controller"], self.controller)
            return Result("aborted")
        self.assert_original(lambda: self.invoke(self.interrupt, reopen))
        status = self.read("launcher-status.json")
        self.assertEqual(status["status"], "INTERRUPTED_OR_FAILED")
        self.assertFalse(status["worker_exit_code_is_completion_evidence"])
        self.assertEqual(status["controller"], self.controller)
        self.assertTrue((self.run / "interrupted-or-failed.csv").exists())
    def test_reopen_error_and_baseexception_preserve_original(self):
        for error in (ValueError("bad journal"), FinalizationAbort("unexpected abort")):
            with self.subTest(error=type(error).__name__):
                # Independent output directories; no retries into an existing run.
                launcher.RUN = self.root / type(error).__name__
                self.run = launcher.RUN
                launcher.MARKER = self.root / (type(error).__name__ + ".marker")
                def reopen(path): raise error
                self.assert_original(lambda: self.invoke(self.interrupt, reopen))
                self.assertIn(type(error).__name__, self.read("launcher-status.json")["evidence_recovery"])
    def test_json_export_error(self): self.export_failure("json")
    def test_csv_export_error(self): self.export_failure("csv")
    def export_failure(self, stage):
        self.assert_original(lambda: self.invoke(self.interrupt, lambda p: Result("aborted", stage)))
        self.assertIn("OSError", self.read("launcher-status.json")["evidence_recovery"])
        self.assertTrue((self.run / "launcher-interruption.json").exists())
    def test_export_unexpected_baseexception(self):
        with patch.object(launcher, "export_result", side_effect=SystemExit(7)):
            self.assert_original(lambda: self.invoke(self.interrupt))
        self.assertIn("SystemExit", self.read("launcher-status.json")["evidence_recovery"])
    def test_finalizer_bug_reports_secondary_without_masking_original(self):
        with patch.object(launcher, "_record_interruption", side_effect=RuntimeError("finalizer bug")):
            self.assert_original(lambda: self.invoke(self.interrupt))
        self.assertIn("finalizer bug", self.read("launcher-finalization-error.json")["finalization_error"])
    def test_receipt_write_failure_still_exports_and_records_status(self):
        write = launcher.publish_new
        def publish(path, data):
            if Path(path).name == "launcher-interruption.json": raise OSError("receipt fault")
            return write(path, data)
        with patch.object(launcher, "publish_new", publish):
            self.assert_original(lambda: self.invoke(self.interrupt))
        self.assertEqual(self.read("launcher-status.json")["finalization_errors"][0]["stage"], "interruption_receipt")
    def test_status_write_failure_retains_pending_receipt_and_fallback(self):
        write = launcher.publish_new
        def publish(path, data):
            if Path(path).name == "launcher-status.json": raise OSError("status fault")
            return write(path, data)
        with patch.object(launcher, "publish_new", publish):
            self.assert_original(lambda: self.invoke(self.interrupt))
        self.assertTrue((self.run / "launcher-interruption.json").exists())
        self.assertEqual(self.read("launcher-finalization-error.json")["finalization_errors"][-1]["stage"], "launcher_status")
    def test_all_finalization_writes_fail_original_survives(self):
        write = launcher.publish_new
        def publish(path, data):
            if Path(path).name.startswith("launcher-"): raise OSError("disk fault")
            return write(path, data)
        with patch.object(launcher, "publish_new", publish):
            self.assert_original(lambda: self.invoke(self.interrupt))
    def test_aborted_analysis_exit_zero_never_completed(self):
        self.assertEqual(self.invoke(lambda *a, **k: Result("aborted")), 1)
        status = self.read("launcher-status.json")
        self.assertEqual(status["native_analysis_status"], "aborted")
        self.assertEqual(status["P31"], "PARTIAL_PENDING_REVIEW")
    def test_success_summary_error_is_finalized(self):
        class Broken(Result):
            def to_dict(self): raise RuntimeError("summary fault")
        with patch.object(launcher, "export_result", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "summary fault"):
                self.invoke(lambda *a, **k: Broken())
        self.assertEqual(self.read("launcher-status.json")["status"], "INTERRUPTED_OR_FAILED")

class PublicationTests(unittest.TestCase):
    def test_serialization_failure_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'receipt.json'
            with self.assertRaises(ValueError):
                launcher.publish_new(path, dict(value=float('nan')))
            self.assertEqual(list(Path(temp).iterdir()), [])
    def test_fsync_failure_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(launcher.os, 'fsync', side_effect=OSError('flush fault')):
            path = Path(temp) / 'receipt.json'
            with self.assertRaises(OSError): launcher.publish_new(path, dict(ok=True))
            self.assertEqual(list(Path(temp).iterdir()), [])
    def test_link_failure_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(launcher.os, 'link', side_effect=OSError('link fault')):
            path = Path(temp) / 'receipt.json'
            with self.assertRaises(OSError): launcher.publish_new(path, dict(ok=True))
            self.assertEqual(list(Path(temp).iterdir()), [])
    def test_publication_exposes_complete_json_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'receipt.json'; link = launcher.os.link
            def check(source, destination):
                self.assertFalse(path.exists())
                self.assertEqual(json.loads(Path(source).read_text()), dict(ok=True))
                return link(source, destination)
            with patch.object(launcher.os, 'link', check):
                launcher.publish_new(path, dict(ok=True))
            original = path.read_bytes()
            with self.assertRaises(FileExistsError): launcher.publish_new(path, dict(ok=False))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(temp).iterdir()), [path])
    def test_second_keyboard_interrupt_in_reopen_preserves_first(self):
        case = LauncherTests('test_normal_completion_remains_pending_review')
        case.setUp()
        try:
            def reopen(path): raise KeyboardInterrupt('second interruption')
            case.assert_original(lambda: case.invoke(case.interrupt, reopen))
            status = case.read('launcher-status.json')
            self.assertIn('original interruption', status['error'])
            self.assertIn('second interruption', status['evidence_recovery'])
        finally:
            case.tearDown(); case.doCleanups()

class CleanupTests(unittest.TestCase):
    def test_controller_interrupt_waits_for_zero_and_preserves_exception(self):
        calls = []; observations = iter([dict(active_processes=1), dict(active_processes=0)])
        original = KeyboardInterrupt("controller fault")
        class Job:
            def __init__(self, *a): pass
            def assign(self, process): calls.append("assign")
            def terminate(self): calls.append("terminate")
            def observation(self): return next(observations)
            def close(self): calls.append("close")
        class Process:
            pid = 12345
            returncode = None
            def poll(self): raise original
            def wait(self, timeout): self.returncode = 0; calls.append("wait"); return 0
        with tempfile.TemporaryDirectory() as temp, patch.object(_training_process, "_Job", Job), patch.object(
                stability.subprocess, "Popen", return_value=Process()), patch.object(stability.time, "sleep"):
            root = Path(temp) / "journal"
            with self.assertRaises(KeyboardInterrupt) as caught:
                stability._run_owned(root, "stability", dict(recipe={}, protocol={}),
                    startup_timeout=1, heartbeat_timeout=1, max_runtime=2, max_bytes=65536)
            self.assertIs(caught.exception, original)
            controller = json.loads((root / "controller.json").read_text())
            self.assertEqual(controller["cleanup"]["active_processes"], 0)
            self.assertEqual(controller["reason"], "controller_interrupted")
            self.assertEqual(controller["exit_code"], 0)
            self.assertEqual(calls, ["assign", "terminate", "wait", "close"])
    def test_job_terminate_zero_is_explicit_forced_exit_code(self):
        calls = []
        class API:
            def TerminateJobObject(self, handle, code): calls.append((handle, code)); return True
        job = _training_process._Job.__new__(_training_process._Job)
        job.handle = 123; job.api = API()
        job.terminate()
        self.assertEqual(calls, [(123, 0)])

if __name__ == "__main__":
    unittest.main()
