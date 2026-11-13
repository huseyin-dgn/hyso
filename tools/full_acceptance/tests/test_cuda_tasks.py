"""Offline stdlib tests for the CUDA task declaration only; no torch import."""
import importlib.util
from pathlib import Path
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'cuda_tasks.py'
spec = importlib.util.spec_from_file_location('hyso_cuda_tasks_step01', MODULE)
mod = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class CudaTasksTests(unittest.TestCase):
    def test_plan_unique_and_complete(self):
        self.assertEqual(len(mod.validate_plan()), 23)
        self.assertEqual(len(set(t.task_id for t in mod.TASKS)), 23)
        self.assertTrue(all(t.resource == mod.GPU_RESOURCE for t in mod.TASKS))

    def test_module_does_not_import_torch(self):
        import ast
        tree = ast.parse(MODULE.read_text(encoding='utf-8'))
        imported = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imported += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any(name == 'torch' or name.startswith('torch.') for name in imported if name))

    def test_missing_receipts_are_incomplete(self):
        result = mod.aggregate({})
        self.assertEqual(result['status'], 'INCOMPLETE')
        self.assertEqual(result['terminal_count'], 0)
        self.assertFalse(result['legacy_stage_pass_authorized'])

    def test_partial_does_not_pass(self):
        task = mod.TASKS[0]
        result = mod.aggregate({task.task_id: self.ok(task)})
        self.assertEqual(result['status'], 'INCOMPLETE')

    def test_all_pass_diagnostic_only(self):
        result = mod.aggregate({t.task_id: self.ok(t) for t in mod.TASKS})
        self.assertEqual(result['status'], 'PASS')
        self.assertFalse(result['legacy_stage_pass_authorized'])

    def test_failure_dominates(self):
        receipts = {t.task_id: self.ok(t) for t in mod.TASKS}
        task = mod.TASKS[2]
        receipts[task.task_id] = dict(task_id=task.task_id, status='FAIL', reason='error')
        self.assertEqual(mod.aggregate(receipts)['status'], 'FAIL')

    def test_false_pass_rejected(self):
        task = mod.TASKS[0]
        with self.assertRaises(ValueError):
            mod.aggregate({task.task_id: dict(task_id=task.task_id, status='PASS', cleanup_ok=False)})

    def test_identity_mismatch_rejected(self):
        task = mod.TASKS[0]
        with self.assertRaises(ValueError):
            mod.aggregate({task.task_id: dict(task_id='other', status='FAIL', reason='oops')})

    def test_unknown_task_rejected(self):
        with self.assertRaises(ValueError):
            mod.aggregate({'04.cuda.unknown': {'task_id':'04.cuda.unknown','status':'PASS'}})

    def test_incomplete_nonpass_requires_reason(self):
        task = mod.TASKS[0]
        with self.assertRaises(ValueError):
            mod.aggregate({task.task_id: dict(task_id=task.task_id, status='UNSUPPORTED')})

    @staticmethod
    def ok(t):
        return dict(task_id=t.task_id, status='PASS', cleanup_ok=True)


if __name__ == '__main__':
    unittest.main()
