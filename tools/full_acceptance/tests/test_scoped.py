"""No workers: isolated scope must keep all default safe stages closed too."""
import copy
from unittest.mock import patch
from test_infrastructure import Fixture
from full_acceptance import scoped


class ScopedTests(Fixture):
    def test_p13_plan_includes_strict_cpu_regression_and_no_other_native_work(self):
        from full_acceptance import reporting
        self.m['execution_policy']='continue_independent'
        self.m['hardware']['gpu_uuid']='GPU-offline-double'
        tool=self.base/'nsys.exe';tool.write_bytes(b'offline metadata double')
        self.m['optional']['p13_nsys_plan']=dict(configuration=dict(
            protocol='trained_B_3_forward_module_nvtx',tool=dict(path=str(tool))))
        for e in self.m['experiments']:
            e['authorization']['approved']=e['id'] in scoped.scope_ids('06.p13_capture')
        with patch.object(scoped.core,'verify',return_value=self.m):
            scoped.seal_fresh(self.m,'06.p13_capture',True)
        self.assertEqual(scoped.check_scope(self.m,'06.p13_capture',True),['02.regression','06.p13_capture'])
        rows=reporting.initial_results(self.m)
        capture=next(e for e in self.m['experiments'] if e['id']=='06.p13_capture')
        self.assertEqual(reporting.decision(self.m,capture,rows)[0],'BLOCKED_DEPENDENCY')
        rows['02.regression'].update(status='PASS',attempted=True)
        self.assertEqual(reporting.decision(self.m,capture,rows)[0],'READY')

    def test_camera_scope_closes_every_unrelated_stage(self):
        self.m['execution_policy']='continue_independent'
        with patch.object(scoped.core,'verify',return_value=self.m):
            scoped.seal_fresh(self.m,'08.camera',False)
        self.assertEqual(scoped.check_scope(self.m,'08.camera',False),[])
        self.assertFalse(any(e['authorization']['approved'] for e in self.m['experiments']))

    def test_unrelated_approval_and_scope_drift_are_rejected(self):
        self.m.update(execution_scope=['08.camera'],execution_policy='continue_independent')
        self.m['optional']['scoped_driver']=dict(configuration=dict(stage='08.camera'))
        with self.assertRaisesRegex(ValueError,'unrelated'):scoped.check_scope(self.m,'08.camera',False)
        with self.assertRaisesRegex(ValueError,'scope mismatch'):scoped.check_scope(self.m,'06.p13_capture',False)

    def test_consumed_manifest_cannot_be_resealed(self):
        (self.out/'STARTED.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'consumed'):scoped.seal_fresh(self.m,'08.camera',False)
