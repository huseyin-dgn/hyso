"""Pure dispatch admission tests; do not import torch or spawn a worker."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from full_acceptance.cuda_dispatch import OPERATION, resolve_subtask_route
from full_acceptance.cuda_tasks import TASKS

class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.t = TASKS[0]
        self.m = dict(experiments=[dict(id='04.cuda',handler='inference_cuda',authorization=dict(approved=True))],
                      hardware=dict(gpu_uuid='GPU-fixed'),source_snapshot=dict(tree_sha256='tree'),
                      _verified_manifest_sha256='manifest')
        self.r = dict(experiment_id=self.t.task_id, operation=OPERATION, parent_id='04.cuda',
                      arm=self.t.arm,measurement=self.t.measurement,gpu_uuid='GPU-fixed',
                      manifest_sha256='manifest',source_tree_sha256='tree')
    def test_valid(self):
        route=resolve_subtask_route(self.m,self.r,admitted=True)
        self.assertEqual((route.task_id,route.arm,route.measurement),(self.t.task_id,self.t.arm,self.t.measurement))
    def test_no_admission(self):
        with self.assertRaises(PermissionError):resolve_subtask_route(self.m,self.r)
    def test_no_parent_approval(self):
        self.m['experiments'][0]['authorization']['approved']=False
        with self.assertRaises(PermissionError):resolve_subtask_route(self.m,self.r,admitted=True)
    def test_legacy_handler_disallowed(self):
        self.r['operation']='inference_cuda'
        with self.assertRaises(PermissionError):resolve_subtask_route(self.m,self.r,admitted=True)
    def test_wrong_gpu(self):
        self.r['gpu_uuid']='GPU-other'
        with self.assertRaises(ValueError):resolve_subtask_route(self.m,self.r,admitted=True)
    def test_wrong_manifest(self):
        self.r['manifest_sha256']='other'
        with self.assertRaises(ValueError):resolve_subtask_route(self.m,self.r,admitted=True)
    def test_wrong_source(self):
        self.r['source_tree_sha256']='other'
        with self.assertRaises(ValueError):resolve_subtask_route(self.m,self.r,admitted=True)
    def test_unknown_task(self):
        self.r['experiment_id']='04.cuda.A.no_such_measurement'
        with self.assertRaises(ValueError):resolve_subtask_route(self.m,self.r,admitted=True)
    def test_wrong_arm(self):
        self.r['arm']='B'
        with self.assertRaises(ValueError):resolve_subtask_route(self.m,self.r,admitted=True)
    def test_wrong_measurement(self):
        self.r['measurement']='pipeline'
        with self.assertRaises(ValueError):resolve_subtask_route(self.m,self.r,admitted=True)
    def test_missing_parent(self):
        self.m['experiments']=[]
        with self.assertRaises(ValueError):resolve_subtask_route(self.m,self.r,admitted=True)
    def test_import_is_safe(self):
        self.assertNotIn('torch',sys.modules)

if __name__=='__main__': unittest.main()
