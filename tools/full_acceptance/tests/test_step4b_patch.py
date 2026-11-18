"""Offline bridge tests; no WinAPI, subprocess, Torch or CUDA."""
import unittest
from pathlib import Path

class PatchChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[1]

    def test_parent_guards_subtask_before_launch(self):
        src=(self.root/'safety.py').read_text(encoding='utf-8')
        self.assertLess(src.index('validate_subtask_job_request(manifest,experiment,request'),
                        src.index('subprocess.Popen(argv'))

    def test_child_identity_before_dispatch(self):
        src=(self.root/'cli.py').read_text(encoding='utf-8')
        self.assertLess(src.index('route=resolve_subtask_route'),src.index('value=dispatch[operation]'))

    def test_does_not_fall_through_to_legacy(self):
        src=(self.root/'cli.py').read_text(encoding='utf-8')
        self.assertIn("if subtask:\n        # The parent admission",src)
        self.assertIn("'inference_cuda_subtask':lambda m,r:cuda_subtask_run(m,r,route)",src)

    def test_cuda_visibility_declared(self):
        src=(self.root/'safety.py').read_text(encoding='utf-8')
        self.assertIn("'inference_cuda','inference_cuda_subtask'",src)

    def test_worker_fails_closed(self):
        from types import SimpleNamespace
        from full_acceptance.cuda_tasks import TASKS
        from full_acceptance.cuda_subtask_worker import run

        task = TASKS[0]

        route = SimpleNamespace(
            task_id=task.task_id,
            arm=task.arm,
            measurement="invalid_measurement",
            operation="inference_cuda_subtask",
        )

        with self.assertRaises(ValueError):
            run({}, None, route)

if __name__=='__main__':unittest.main()
