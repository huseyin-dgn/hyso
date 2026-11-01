"""Offline tests for selectable A/B single CUDA latency probe."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock
from full_acceptance.cuda_single_probe import probe, ONLY_TASK_ID

class ModelBProbeTests(unittest.TestCase):
    def test_default_preserved(self):
        self.assertEqual(ONLY_TASK_ID,'04.cuda.A.latency')

    def test_model_b_success_without_gpu(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); manifest=root/'manifest.json';manifest.write_text('{}')
            m={'output':str(root)}
            adapter=Mock(return_value={'task_id':'04.cuda.B.latency','status':'PASS','supervision_verified':True})
            with patch('full_acceptance.cuda_single_probe.verify',return_value=m), \
                 patch('full_acceptance.cuda_single_probe.sha',return_value='a'*64), \
                 patch('full_acceptance.cuda_single_probe.check_admission'):
                outcome=probe(manifest,'a'*64,task_id='04.cuda.B.latency',adapter_factory=lambda *args,**kwargs:adapter)
            self.assertEqual(outcome['task_id'],'04.cuda.B.latency')
            self.assertEqual(outcome['status'],'DIAGNOSTIC_PASS')
            self.assertFalse(outcome['legacy_04_cuda_accepted'])
            self.assertFalse(outcome['suite_complete'])
            self.assertEqual(adapter.call_args.args[0].task_id,'04.cuda.B.latency')

    def test_other_tasks_denied_before_supervisor(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'manifest.json';manifest.write_text('{}')
            adapter=Mock()
            with patch('full_acceptance.cuda_single_probe.verify',return_value={'output':str(root)}), \
                 patch('full_acceptance.cuda_single_probe.sha',return_value='a'*64):
                with self.assertRaises(ValueError):
                    probe(manifest,'a'*64,task_id='04.cuda.A.throughput',adapter_factory=adapter)
            adapter.assert_not_called()

    def test_same_manifest_never_retries(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'manifest.json';manifest.write_text('{}')
            (root/'cuda-single-probe-result.json').write_text('{}')
            adapter=Mock()
            with patch('full_acceptance.cuda_single_probe.verify',return_value={'output':str(root)}), \
                 patch('full_acceptance.cuda_single_probe.sha',return_value='a'*64), \
                 patch('full_acceptance.cuda_single_probe.check_admission'):
                with self.assertRaises(FileExistsError):
                    probe(manifest,'a'*64,task_id='04.cuda.B.latency',adapter_factory=adapter)
            adapter.assert_not_called()
