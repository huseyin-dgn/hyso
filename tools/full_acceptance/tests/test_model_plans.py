"""Per-model preparation/routing only. No CUDA/training/camera/long-run work."""
from pathlib import Path
from unittest.mock import patch

from test_infrastructure import Fixture
from full_acceptance import core,handlers,safety,reporting


class ModelPlans(Fixture):
    def select(self, arm='A'):
        self.m.update(model_run=dict(arm=arm,checkpoint_sha256=arm.lower()*64))
        self.m['assets']=dict(models={a:dict(sha256=a.lower()*64) for a in ('A','B')})
        self.m['protocols']=dict(training=dict(model_arm=arm,devices=['cpu','cuda:0']))
        self.m['hardware']['gpu_uuid']='GPU-fixture'

    def test_legacy_selection_and_visibility_are_preserved(self):
        self.assertEqual(core.selected_arm(self.m),'B')
        self.assertEqual(safety.cuda_visibility(self.m,'training'),'-1')
        with patch.object(handlers,'_training_device',return_value={'status':'PARTIAL'}) as run:
            self.assertEqual(handlers.training(self.m,self.out),{'status':'PARTIAL'})
        run.assert_called_once_with(self.m,self.out)

    def test_explicit_model_and_each_training_device_are_dispatched(self):
        for arm in ('A','B'):
            with self.subTest(arm=arm):
                self.select(arm)
                output=self.out/arm;output.mkdir()
                with patch.object(handlers,'_training_device',return_value=dict(status='PASS',criteria=dict(P23_basic=True))) as run:
                    result=handlers.training(self.m,output)
                self.assertEqual([call.kwargs['device'] for call in run.call_args_list],['cpu','cuda:0'])
                self.assertEqual(result['criteria']['model_arm'],arm)
                self.assertTrue(result['criteria']['P23_basic_each_device'])
                self.assertEqual(result['status'],'PASS')
                self.assertEqual(safety.cuda_visibility(self.m,'training'),'GPU-fixture')

    def test_device_failure_prevents_silent_continuation_or_pass(self):
        self.select()
        with patch.object(handlers,'_training_device',return_value=dict(status='FAIL')) as run:
            result=handlers.training(self.m,self.out)
        self.assertEqual(result['status'],'FAIL')
        self.assertEqual(run.call_count,1)

    def test_partial_device_does_not_become_pass(self):
        self.select()
        with patch.object(handlers,'_training_device',return_value=dict(status='PARTIAL',criteria=dict(P23_basic=False))):
            self.assertEqual(handlers.training(self.m,self.out)['status'],'PARTIAL')

    def test_manifest_rejects_wrong_training_arm_or_device_matrix(self):
        self.select();self.seal()
        self.assertEqual(core.verify(self.path,expected=self.hash,live=False,assets=False)['model_run']['arm'],'A')
        self.m['protocols']['training']['model_arm']='B';self.seal()
        with self.assertRaisesRegex(ValueError,'routing mismatch'):
            core.verify(self.path,expected=self.hash,live=False,assets=False)

    def test_selected_p13_protocol_is_not_silently_model_B(self):
        self.select()
        tool=self.base/'nsys.exe';tool.write_bytes(b'fixture')
        self.m['optional']['p13_nsys_plan']=dict(configuration=dict(tool=dict(path=str(tool)),protocol='trained_A_3_forward_module_nvtx'))
        experiment=next(e for e in self.m['experiments'] if e['id']=='06.p13_capture')
        self.assertEqual(reporting.prerequisites(self.m,experiment),[])
        self.m['optional']['p13_nsys_plan']['configuration']['protocol']='trained_B_3_forward_module_nvtx'
        self.assertTrue(reporting.prerequisites(self.m,experiment))
