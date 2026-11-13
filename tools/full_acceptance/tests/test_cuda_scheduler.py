"""Offline scheduling tests; no Torch, CUDA or native processes."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from full_acceptance.cuda_scheduler import CUDAOrchestrator, RunPolicy
from full_acceptance.cuda_tasks import TASKS


def ok(task):
    return {'task_id':task.task_id,'status':'PASS','cleanup_ok':True,'supervision_verified':True}


class SchedulerTests(unittest.TestCase):
    def test_requires_approval(self):
        with self.assertRaises(PermissionError):
            CUDAOrchestrator(ok).run(adapter_verified=True)
    def test_requires_verified_adapter(self):
        with self.assertRaises(PermissionError):
            CUDAOrchestrator(ok).run(approved=True)
    def test_disallows_parallel_gpu(self):
        with self.assertRaises(ValueError):RunPolicy(max_active_gpu=2)
    def test_order_all_receipts(self):
        order=[]
        def invoke(task):order.append(task.task_id);return ok(task)
        r=CUDAOrchestrator(invoke).run(approved=True,adapter_verified=True)
        self.assertEqual(order,[t.task_id for t in TASKS])
        self.assertEqual(r['status'],'PASS')
        self.assertFalse(r['legacy_stage_pass_authorized'])
    def test_isolated_failure_continues(self):
        def invoke(t):return dict(task_id=t.task_id,status='FAIL',reason='measurement_failure',isolated_failure=True) if t==TASKS[1] else ok(t)
        r=CUDAOrchestrator(invoke).run(approved=True,adapter_verified=True)
        self.assertEqual(r['terminal_count'],23)
        self.assertEqual(r['status'],'FAIL')
    def test_unverified_failure_stops(self):
        def invoke(t):return dict(task_id=t.task_id,status='FAIL',reason='unknown') if t==TASKS[1] else ok(t)
        r=CUDAOrchestrator(invoke).run(approved=True,adapter_verified=True)
        self.assertEqual(r['results'][TASKS[2].task_id]['status'],'NOT_RUN')
    def test_native_crash_stops(self):
        def invoke(t):return dict(task_id=t.task_id,status='FAIL',reason='crash',native_crash=True,isolated_failure=True) if t==TASKS[0] else ok(t)
        r=CUDAOrchestrator(invoke).run(approved=True,adapter_verified=True)
        self.assertEqual(r['results'][TASKS[1].task_id]['status'],'NOT_RUN')
    def test_exception_stops(self):
        def invoke(t):raise RuntimeError('bad')
        r=CUDAOrchestrator(invoke).run(approved=True,adapter_verified=True)
        self.assertEqual(r['results'][TASKS[0].task_id]['status'],'FAIL')
        self.assertEqual(r['results'][TASKS[1].task_id]['status'],'NOT_RUN')
    def test_false_supervision_pass_stops(self):
        def invoke(t):return dict(task_id=t.task_id,status='PASS',cleanup_ok=True)
        r=CUDAOrchestrator(invoke).run(approved=True,adapter_verified=True)
        self.assertEqual(r['results'][TASKS[0].task_id]['status'],'FAIL')
    def test_abort_stops(self):
        def invoke(t):return dict(task_id=t.task_id,status='ABORTED',reason='stop')
        r=CUDAOrchestrator(invoke).run(approved=True,adapter_verified=True)
        self.assertEqual(r['results'][TASKS[1].task_id]['status'],'NOT_RUN')

if __name__=='__main__':unittest.main()
