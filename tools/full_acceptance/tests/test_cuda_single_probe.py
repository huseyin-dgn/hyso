import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from full_acceptance.cuda_single_probe import ONLY_TASK_ID, probe

class SingleProbeTests(unittest.TestCase):
    def test_exact_task(self):
        self.assertEqual(ONLY_TASK_ID,'04.cuda.A.latency')

    def test_missing_hash_rejected_without_verification(self):
        with patch('full_acceptance.cuda_single_probe.verify') as verify:
            with self.assertRaises(ValueError):probe('some.json',None)
            verify.assert_not_called()

    def test_noncanonical_manifest_rejected_before_adapter(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            path=root/'other.json';path.write_text('{}')
            m=dict(output=str(root))
            with patch('full_acceptance.cuda_single_probe.verify',return_value=m),patch('full_acceptance.cuda_single_probe.sha',return_value='a'*64),patch('full_acceptance.cuda_single_probe.WindowsJobCudaAdapter') as adapter:
                with self.assertRaises(ValueError):probe(path,'a'*64)
                adapter.assert_not_called()

    def test_started_run_rejected_before_adapter(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'manifest.json';path.write_text('{}')
            (root/'STARTED.json').write_text('{}')
            m=dict(output=str(root))
            with patch('full_acceptance.cuda_single_probe.verify',return_value=m),patch('full_acceptance.cuda_single_probe.sha',return_value='a'*64),patch('full_acceptance.cuda_single_probe.WindowsJobCudaAdapter') as adapter:
                with self.assertRaises(FileExistsError):probe(path,'a'*64)
                adapter.assert_not_called()

    def test_stop_request_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'manifest.json';path.write_text('{}')
            (root/'stop-request.json').write_text('{}')
            with patch('full_acceptance.cuda_single_probe.verify',return_value=dict(output=str(root))),patch('full_acceptance.cuda_single_probe.sha',return_value='a'*64):
                with self.assertRaises(PermissionError):probe(path,'a'*64)

if __name__=='__main__':unittest.main()
