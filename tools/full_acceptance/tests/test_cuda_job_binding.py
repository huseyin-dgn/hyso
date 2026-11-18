"""No native imports or subprocesses. Tests the fail-closed job binding."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from full_acceptance.cuda_job_binding import (
    OPERATION, UninstalledSubtaskRoute, validate_subtask_job_request,
)
from full_acceptance.cuda_tasks import TASKS


class JobBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name).resolve()
        self.task = TASKS[0]
        self.manifest = dict(output=str(root), hardware={'gpu_uuid':'GPU-verified'},
            source_snapshot={'root':str(root/'source')}, experiments=[dict(
                id='04.cuda', handler='inference_cuda', authorization={'approved':True})])
        self.exp = dict(id=self.task.task_id, parent_id='04.cuda', handler=OPERATION,
            arm=self.task.arm, measurement=self.task.measurement,
            output=str(root/'cuda-subtasks'/self.task.task_id))
        self.req = dict(experiment_id=self.task.task_id, operation=OPERATION,
            gpu_uuid='GPU-verified', output=self.exp['output'],
            snapshot=str(root/'source'),manifest=str(root/'manifest.json'),
            manifest_sha256='f'*64)

    def validate(self, **kwargs):
        return validate_subtask_job_request(self.manifest,self.exp,self.req,**kwargs)

    def test_disabled_route(self):
        with self.assertRaises(UninstalledSubtaskRoute):self.validate()

    def test_valid_identity_only_with_installed_route(self):
        result = self.validate(route_installed=True)
        self.assertEqual(result['task_id'],self.task.task_id)
        self.assertIn('not_execution_proof',result['scope'])

    def test_no_approval(self):
        self.manifest['experiments'][0]['authorization']['approved']=False
        with self.assertRaises(PermissionError):self.validate(route_installed=True)

    def test_wrong_gpu(self):
        self.req['gpu_uuid']='GPU-other'
        with self.assertRaises(ValueError):self.validate(route_installed=True)

    def test_wrong_operation(self):
        self.req['operation']='inference_cuda'
        with self.assertRaises(ValueError):self.validate(route_installed=True)

    def test_wrong_task(self):
        self.exp['id']='04.cuda.unknown'
        with self.assertRaises(ValueError):self.validate(route_installed=True)

    def test_wrong_output(self):
        self.exp['output']=str(Path(self.tmp.name)/'other')
        with self.assertRaises(ValueError):self.validate(route_installed=True)

    def test_wrong_source(self):
        self.req['snapshot']=str(Path(self.tmp.name)/'other')
        with self.assertRaises(ValueError):self.validate(route_installed=True)

    def test_wrong_manifest(self):
        self.req['manifest']=str(Path(self.tmp.name)/'other.json')
        with self.assertRaises(ValueError):self.validate(route_installed=True)

    def test_wrong_arm(self):
        self.exp['arm']='B'
        with self.assertRaises(ValueError):self.validate(route_installed=True)

    def test_missing_hash(self):
        self.req['manifest_sha256']='short'
        with self.assertRaises(ValueError):self.validate(route_installed=True)
