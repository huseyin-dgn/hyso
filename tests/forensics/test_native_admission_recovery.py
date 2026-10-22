"""Offline rejection tests, without repository conftest, Torch or CUDA imports."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/forensics'))
import native_admission as admission
import native_scenario_runner as runner
from test_p23_preflight import capture, NOW, RUN, SHA


def fixture():
    value = capture()
    process = value['inventory']['processes'][0]
    process.update(parent_pid=1, path='C:/fixture.exe', command_line='fixture --ui')
    value['inventory']['processes'].append(dict(pid=1, name='parent', path=None,
        parent_pid=0, command_line=None))
    inventory = value['inventory']
    review = dict(reviewer='explicit offline fixture', conflicting_workloads=[],
        inventory_sha256=admission.policy.canonical_hash(inventory), gpu_processes=[dict(
            pid=42, category='desktop', reason='synthetic desktop evidence only',
            identity_sha256=admission.policy.canonical_hash(admission.facts(inventory,42)),
            restricted_fields=[])])
    return value, review


def decide(value, review, now=NOW):
    return admission.evaluate(value, review, run_id=RUN, manifest_sha256=SHA, now_ns=now)


class RecoveryRejections(unittest.TestCase):
    def test_bound_explicit_review_and_no_mutation(self):
        value, review = fixture()
        before = copy.deepcopy((value, review))
        self.assertTrue(decide(value, review)['eligible'])
        self.assertEqual((value, review), before)

    def test_stale_capture_rejected(self):
        value, review = fixture()
        self.assertIn('stale_or_future_or_reversed_capture', decide(value, review, NOW+31*10**9)['blockers'])

    def test_unknown_pid_rejected(self):
        value, review = fixture()
        review['gpu_processes'][0]['category'] = 'unknown'
        self.assertFalse(decide(value, review)['eligible'])

    def test_duplicate_classification_rejected(self):
        value, review = fixture()
        review['gpu_processes'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            decide(value, review)

    def test_changed_command_or_parent_rejected(self):
        for field in ('command_line', 'parent_pid'):
            value, review = fixture()
            value['inventory']['processes'][0][field] = 'changed'
            review['inventory_sha256'] = admission.policy.canonical_hash(value['inventory'])
            with self.assertRaisesRegex(ValueError, 'PID identity'):
                decide(value, review)

    def test_restricted_identity_must_be_recorded(self):
        value, review = fixture()
        value['inventory']['processes'][0]['path'] = None
        review['inventory_sha256'] = admission.policy.canonical_hash(value['inventory'])
        review['gpu_processes'][0]['identity_sha256'] = admission.policy.canonical_hash(admission.facts(value['inventory'],42))
        with self.assertRaisesRegex(ValueError, 'restricted'):
            decide(value, review)

    def test_unknown_new_pid_cannot_inherit_previous_review(self):
        value, review = fixture()
        value['inventory']['compute_pids'].append(43)
        value['inventory']['raw_query'].append('43, RiotClient.exe, [N/A]')
        review['inventory_sha256'] = admission.policy.canonical_hash(value['inventory'])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            decide(value, review)

    def test_missing_wrong_receipt_and_limit_readback_rejected(self):
        limits = runner.expected_limits('child')
        receipt = dict(pid=42, parent_pid=7, limits=limits, token='test', assigned=True, query_handle=99)
        runner.validate_gate(receipt, pid=42, parent_pid=7, limits=limits, token='test')
        for key in receipt:
            bad = copy.deepcopy(receipt)
            del bad[key]
            with self.subTest(missing=key), self.assertRaises((KeyError, ValueError)):
                runner.validate_gate(bad, pid=42, parent_pid=7, limits=limits, token='test')
        for key, wrong in [('pid',43), ('parent_pid',8), ('limits',{}), ('assigned',False), ('token','other')]:
            bad = dict(receipt, **{key:wrong})
            with self.subTest(wrong=key), self.assertRaises(ValueError):
                runner.validate_gate(bad, pid=42, parent_pid=7, limits=limits, token='test')

    def test_manifest_tamper_and_runtime_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'source').mkdir()
            source = root/'source/probe.py'
            source.write_text('original')
            runtime = root/'runtime.bin'
            runtime.write_bytes(b'original')
            manifest = dict(source_files={'probe.py':runner.digest(source)},
                runtime_files={str(runtime):runner.digest(runtime)}, interpreter=sys.executable,
                scenario=runner.SCENARIO, limits=runner.LIMITS)
            runner.publish(root/'manifest.json', manifest)
            sha = runner.digest(root/'manifest.json')
            runner.verify(root, sha)
            runtime.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'runtime'):
                runner.verify(root, sha)
            runtime.write_bytes(b'original')
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'source'):
                runner.verify(root, sha)
            with self.assertRaisesRegex(ValueError, 'manifest'):
                runner.verify(root, '0'*64)

    def test_bounded_artifacts_and_duplicate_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'result.json'
            with self.assertRaisesRegex(ValueError, 'budget'):
                runner.publish(path, {'data':'x'*(2*1024**2)})
            self.assertFalse(path.exists())
            path.write_text('{"pid":1,"pid":2}')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                runner.read(path)
        self.assertNotIn('torch', sys.modules)

    def test_executing_bootstrap_must_match_sealed_identity(self):
        modules = [('native_scenario_runner.py', runner),
                   ('native_admission.py', admission), ('p23_preflight.py', admission.policy)]
        manifest = dict(source_files={name:runner.digest(module.__file__) for name,module in modules})
        runner.verify_bootstrap(manifest)
        manifest['source_files']['native_admission.py'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'bootstrap identity'):
            runner.verify_bootstrap(manifest)


if __name__ == '__main__':
    unittest.main()
