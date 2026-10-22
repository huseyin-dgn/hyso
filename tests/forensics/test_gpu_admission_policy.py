"""Offline-only policy, integrity and controlled failure checks; no Torch."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/forensics'))
import gpu_admission_policy as policy
import gpu_scenario_runner as runner
import native_scenario_runner as previous
from test_p23_preflight import capture, NOW, RUN, SHA


def decide(value, review=None, **kwargs):
    return policy.evaluate(value, review, run_id=RUN, manifest_sha256=SHA, now_ns=NOW, **kwargs)


class WarningPolicyTests(unittest.TestCase):
    def test_unknown_pid_warns_and_allows_execution(self):
        value = capture()
        review = dict(gpu_processes=[dict(pid=42, category='unknown')])
        result = decide(value, review)
        self.assertTrue(result['execution_allowed'])
        self.assertIn('unknown_or_compute_gpu_process', result['warnings'])
        self.assertEqual(result['measurement_reliability']['status'], 'unqualified')

    def test_high_usage_warns_and_allows_execution(self):
        value = capture()
        for sample in value['gpu_samples']:
            sample['utilization_percent'] = 100
        result = decide(value)
        self.assertTrue(result['eligible'])
        self.assertIn('high_gpu_usage', result['warnings'])

    def test_zero_free_vram_warns_without_minimum_gate(self):
        value = capture()
        for sample in value['gpu_samples']:
            sample['free_mib'] = 0
        result = decide(value)
        self.assertTrue(result['execution_allowed'])
        self.assertIn('low_free_vram', result['warnings'])
        self.assertEqual(result['gpu_samples'][0]['free_mib']['value'], 0)

    def test_stale_telemetry_is_unavailable_not_zero_or_blocked(self):
        value = capture()
        value['collection_start_time_ns'] = NOW-31*10**9
        result = decide(value)
        self.assertTrue(result['eligible'])
        self.assertIsNone(result['gpu_samples'][0]['free_mib']['value'])
        self.assertEqual(result['gpu_samples'][0]['free_mib']['status'], 'unavailable')
        self.assertIn('gpu_telemetry_stale_or_unavailable', result['warnings'])

    def test_sensor_na_is_unsupported(self):
        value = capture()
        value['gpu_samples'][0].update(free_mib='[N/A]', utilization_percent=None)
        result = decide(value)
        self.assertTrue(result['eligible'])
        for name in ('free_mib', 'utilization_percent'):
            sensor = result['gpu_samples'][0][name]
            self.assertEqual(sensor['status'], 'unsupported')
            self.assertIsNone(sensor['value'])
            self.assertTrue(sensor['reason'])

    def test_missing_duplicate_and_unreviewed_gpu_pids_do_not_block(self):
        for value in (None, {}, dict(inventory={'compute_pids':[42,42]}), capture()):
            with self.subTest(value=value):
                self.assertTrue(decide(value)['execution_allowed'])

    def test_conflicting_gpu_workload_is_contention_warning(self):
        result = decide(capture(), dict(conflicting_workloads=['another training process']))
        self.assertTrue(result['eligible'])
        self.assertIn('hardware_resource_contention', result['warnings'])

    def test_host_and_disk_headroom_are_context_not_ownership_gates(self):
        value = capture()
        value.update(host_free_bytes=0, disk_free_bytes=0)
        self.assertTrue(decide(value)['eligible'])

    def test_partial_physical_sensor_fields_keep_null_status_reason(self):
        sample = dict(status='measured', values={'uuid':'GPU-test', 'gpu_busy_percent':None})
        result = policy.sensor_metadata(sample)
        self.assertEqual(result['value']['fields']['gpu_busy_percent'],
                         policy.field(status='unsupported', reason='sensor_NA_or_unsupported'))
        self.assertEqual(policy.sensor_metadata(dict(status='timed_out', reason='deadline')),
                         policy.field(reason='deadline'))

    def test_no_mutation_and_no_torch_import(self):
        value = capture()
        before = copy.deepcopy(value)
        decide(value)
        self.assertEqual(value, before)
        self.assertNotIn('torch', sys.modules)

    def test_each_hard_safety_receipt_fails_closed(self):
        good = dict.fromkeys(policy.SAFETY_FIELDS, True)
        self.assertTrue(decide(capture(), safety=good)['eligible'])
        for name in good:
            for wrong in (False, None, 1):
                with self.subTest(name=name, wrong=wrong):
                    bad = dict(good, **{name:wrong})
                    result = decide(capture(), safety=bad)
                    self.assertFalse(result['execution_allowed'])
                    self.assertIn(name+'_failed_or_missing', result['blockers'])
        self.assertFalse(decide(capture(), safety={})['eligible'])

    def test_ownership_and_limit_receipts_still_block(self):
        limits = runner.expected_limits('child')
        receipt = dict(pid=42, parent_pid=7, limits=limits, assigned=True, token='x', query_handle=1)
        for key, value in [('pid',1), ('parent_pid',8), ('assigned',False), ('limits',{})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_gate(dict(receipt, **{key:value}), pid=42, parent_pid=7, limits=limits, token='x')

    def test_manifest_runtime_source_and_retry_mismatches_block(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'source').mkdir()
            source = root/'source/probe.py'
            source.write_text('source')
            runtime = root/'runtime.bin'
            runtime.write_text('runtime')
            manifest = dict(source_files={'probe.py':runner.digest(source)},
                runtime_files={str(runtime):runner.digest(runtime)}, interpreter=sys.executable,
                scenario=runner.SCENARIO, limits=runner.LIMITS, executions=1, retries=0,
                baselines=0, admission_policy=policy.POLICY_ID)
            path = root/'manifest.json'
            runner.publish(path, manifest)
            sha = runner.digest(path)
            runner.verify(root, sha)
            for target in (source, runtime):
                original = target.read_bytes()
                target.write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    runner.verify(root, sha)
                target.write_bytes(original)
            with self.assertRaisesRegex(ValueError, 'manifest identity'):
                runner.verify(root, '0'*64)
            for key, wrong in [('retries',1), ('executions',2), ('executions',True), ('baselines',1)]:
                path.write_text(json.dumps(dict(manifest, **{key:wrong})))
                with self.assertRaisesRegex(ValueError, 'scenario/limit'):
                    runner.verify(root, runner.digest(path))

    def test_oom_handler_only_catches_typed_oom_and_never_retries(self):
        class FakeOOM(RuntimeError):
            pass
        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(OutOfMemoryError=FakeOOM))
        with patch.dict(sys.modules, torch=fake), patch.object(runner, 'run_native_worker', side_effect=FakeOOM()) as execute:
            result = runner.native_worker(None, None, None)
            self.assertEqual(result, policy.controlled_failure('cuda_oom'))
            self.assertEqual(execute.call_count, 1)
        with patch.object(runner, 'run_native_worker', side_effect=RuntimeError('native crash')):
            with self.assertRaisesRegex(RuntimeError, 'native crash'):
                runner.native_worker(None, None, None)
        self.assertNotIn('torch', sys.modules)

    def test_timeout_requires_cleanup_proof(self):
        receipt = dict(outcome='timeout', cleanup={'active_processes':0},
            owned_leader_exited=True, assignment={'assigned':True})
        self.assertEqual(runner.controlled_timeout(receipt), policy.controlled_failure('timeout'))
        bad = copy.deepcopy(receipt)
        bad['cleanup']['active_processes'] = 1
        with self.assertRaisesRegex(ValueError, 'cleanup'):
            runner.controlled_timeout(bad)

    def test_result_corruption_and_retry_results_block(self):
        result = policy.controlled_failure('cuda_oom')
        runner.validate_result(result)
        for key, wrong in [('reason','unexpected'), ('retries',1), ('measurements',{}), ('cleanup_required',False)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_result(dict(result, **{key:wrong}))
        with self.assertRaises(ValueError):
            runner.validate_result({'status':'invented'})

    def test_isolation_implementation_is_preserved(self):
        a, b = ast.parse(Path(previous.__file__).read_text()), ast.parse(Path(runner.__file__).read_text())
        for name in ('job_class', 'expected_limits', 'validate_gate', 'await_gate', 'supervise'):
            left = next(n for n in a.body if isinstance(n, ast.FunctionDef) and n.name==name)
            right = next(n for n in b.body if isinstance(n, ast.FunctionDef) and n.name==name)
            # The only supervisor change is the new sealed bootstrap filename.
            left = ast.parse(ast.unparse(left).replace('native_scenario_runner.py','gpu_scenario_runner.py'))
            self.assertEqual(ast.dump(left), ast.dump(ast.Module(body=[right], type_ignores=[])))
        self.assertEqual(runner.LIMITS, previous.LIMITS)


if __name__ == '__main__':
    unittest.main()
