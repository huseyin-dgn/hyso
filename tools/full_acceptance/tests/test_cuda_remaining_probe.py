import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from full_acceptance.cuda_remaining_probe import REMAINING, run_remaining


class RemainingCudaTests(unittest.TestCase):
    def test_exact_21_no_latency(self):
        self.assertEqual(len(REMAINING), 21)
        self.assertFalse(any(t.measurement == 'latency' for t in REMAINING))

    def test_missing_hash_no_verify(self):
        with patch('full_acceptance.cuda_remaining_probe.verify') as verify:
            with self.assertRaises(ValueError):
                run_remaining('manifest.json', None)
            verify.assert_not_called()

    def test_reject_started_before_adapter(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text('{}')
            (root / 'STARTED.json').write_text('{}')
            with patch('full_acceptance.cuda_remaining_probe.verify', return_value={'output': str(root)}), patch('full_acceptance.cuda_remaining_probe.sha', return_value='a'*64), patch('full_acceptance.cuda_remaining_probe.check_admission'), patch('full_acceptance.cuda_remaining_probe.WindowsJobCudaAdapter') as adapter:
                with self.assertRaises(PermissionError):
                    run_remaining(root / 'manifest.json', 'a'*64)
                adapter.assert_not_called()

    def test_refuses_existing_task_evidence(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text('{}')
            (root / 'cuda-subtasks' / REMAINING[0].task_id).mkdir(parents=True)
            with patch('full_acceptance.cuda_remaining_probe.verify', return_value={'output': str(root)}), patch('full_acceptance.cuda_remaining_probe.sha', return_value='a'*64), patch('full_acceptance.cuda_remaining_probe.check_admission'), patch('full_acceptance.cuda_remaining_probe.WindowsJobCudaAdapter') as adapter:
                with self.assertRaises(FileExistsError):
                    run_remaining(root / 'manifest.json', 'a'*64)
                adapter.assert_not_called()

    def test_stops_first_failure_and_never_legacy_pass(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text('{}')
            calls = []
            class Adapter:
                def __init__(self, *a, **k): pass
                def __call__(self, task):
                    calls.append(task.task_id)
                    return dict(task_id=task.task_id, status='FAIL', reason='isolated', cleanup_ok=True, supervision_verified=True)
            with patch('full_acceptance.cuda_remaining_probe.verify', return_value={'output': str(root)}), patch('full_acceptance.cuda_remaining_probe.sha', return_value='a'*64), patch('full_acceptance.cuda_remaining_probe.check_admission'):
                result = run_remaining(root / 'manifest.json', 'a'*64, adapter_factory=Adapter)
            self.assertEqual(calls, [REMAINING[0].task_id])
            self.assertFalse(result['legacy_04_cuda_accepted'])
            self.assertEqual(result['state'], 'DIAGNOSTIC_INCOMPLETE')

    def test_all_mocked_pass_remains_diagnostic_only(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text('{}')
            seen = []
            class Adapter:
                def __init__(self, *a, **k): pass
                def __call__(self, task):
                    seen.append(task.task_id)
                    return dict(task_id=task.task_id, status='PASS', cleanup_ok=True, supervision_verified=True, native_crash=False, resource_pressure=False)
            with patch('full_acceptance.cuda_remaining_probe.verify', return_value={'output': str(root)}), patch('full_acceptance.cuda_remaining_probe.sha', return_value='a'*64), patch('full_acceptance.cuda_remaining_probe.check_admission'):
                result = run_remaining(root / 'manifest.json', 'a'*64, adapter_factory=Adapter)
            self.assertEqual(len(seen), 21)
            self.assertEqual(result['state'], 'DIAGNOSTIC_PASS')
            self.assertFalse(result['legacy_04_cuda_accepted'])
            self.assertFalse(result['suite_complete'])


if __name__ == '__main__':
    unittest.main()
