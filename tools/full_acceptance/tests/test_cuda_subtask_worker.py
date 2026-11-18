"""Pure routing tests. Import does not load torch; never starts a CUDA worker."""
import ast
from pathlib import Path
import unittest
from full_acceptance.cuda_tasks import TASKS
from full_acceptance.cuda_dispatch import SubtaskRoute
from full_acceptance import cuda_subtask_worker as w

class WorkerRoutingTests(unittest.TestCase):
    def test_all_23_identities(self):
        for task in TASKS:
            with self.subTest(task=task.task_id):
                self.assertEqual(w._check_route(SubtaskRoute(task.task_id,task.arm,task.measurement)),task)

    def test_wrong_arm_rejected(self):
        task=TASKS[0]
        with self.assertRaises(ValueError):w._check_route(SubtaskRoute(task.task_id,'B',task.measurement))

    def test_wrong_measurement_rejected(self):
        task=TASKS[0]
        with self.assertRaises(ValueError):w._check_route(SubtaskRoute(task.task_id,task.arm,'quality'))

    def test_wrong_operation_rejected(self):
        with self.assertRaises(PermissionError):
            w._check_route(type('R',(),dict(task_id=TASKS[0].task_id,arm='A',measurement='latency',operation='inference_cuda'))())

    def test_transfer_flags(self):
        self.assertEqual(w._transfer_flags('04.cuda.shared.transfer_pageable_blocking'),(False,False))
        self.assertEqual(w._transfer_flags('04.cuda.shared.transfer_pinned_blocking'),(True,False))
        self.assertEqual(w._transfer_flags('04.cuda.shared.transfer_pinned_nonblocking'),(True,True))
        with self.assertRaises(ValueError):w._transfer_flags('04.cuda.shared.transfer_bad')

    def test_no_legacy_full_inference_call(self):
        src=Path(w.__file__).read_text(encoding='utf-8')
        tree=ast.parse(src)
        calls=[n.func for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(f,ast.Attribute) and f.attr=='inference' for f in calls))

    def test_no_native_import_at_module_level(self):
        tree=ast.parse(Path(w.__file__).read_text(encoding='utf-8'))
        imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('torch' in ast.unparse(n) for n in imports))

if __name__=='__main__':unittest.main()
