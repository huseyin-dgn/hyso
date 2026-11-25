"""CPU/stdlib simulations of P23's final admission; no Torch or GPU sensors."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/forensics'))
import native_admission as admission
import p23_forensic_worker as worker
import p23_forensic_runner as runner
from test_native_admission_recovery import fixture
from test_p23_preflight import NOW, RUN, SHA
from offline_test_isolation import offline_test

LATE = NOW+40*10**9


def initial_pair():
    capture, review = fixture()
    capture['inventory'].update(gpu_table_query_succeeded=True, gpu_table_pids=[42],
        gpu_table_raw='| Processes: |\n| GPU GI CI PID Type Process name GPU Memory |\n'
                      '| ID ID Usage |\n|====|\n| 0 N/A N/A 42 C+G UI.exe N/A |\n+---+\n')
    capture['inventory']['processes'][0]['creation_date'] = 'original creation'
    review['inventory_sha256'] = admission.policy.canonical_hash(capture['inventory'])
    review['gpu_processes'][0]['identity_sha256'] = admission.policy.canonical_hash(
        admission.facts(capture['inventory'], 42))
    capture = admission.bind_capture(capture, run_id=RUN, manifest_sha256=SHA,
        stage='controller-preflight', nonce='current-initial-capture', worker_pid=0)
    review['capture_binding'] = copy.deepcopy(capture['binding'])
    return capture, review


def fresh_raw(initial):
    value = copy.deepcopy(initial)
    del value['binding']
    value['collection_start_time_ns'] += 40*10**9
    value['checked_time_ns'] += 40*10**9
    for sample in value['gpu_samples']:
        sample['time_ns'] += 40*10**9
    return value


def final_pair(initial, review, raw):
    fresh = admission.bind_capture(raw, run_id=RUN, manifest_sha256=SHA,
        stage='worker-final', nonce='owned-attempt-token', worker_pid=71)
    updated = admission.refreshed_review(initial, review, fresh, run_id=RUN,
        manifest_sha256=SHA, nonce='owned-attempt-token', worker_pid=71)
    return fresh, updated


def decide(capture, review, now=LATE):
    return admission.evaluate_bound(capture, review, run_id=RUN, manifest_sha256=SHA,
        stage='worker-final', nonce='owned-attempt-token', worker_pid=71, now_ns=now)


class FreshAdmission(unittest.TestCase):
    def test_long_import_does_not_reuse_expired_evidence(self):
        old, review = initial_pair()
        self.assertFalse(admission.evaluate_bound(old, review, run_id=RUN,
            manifest_sha256=SHA, stage='controller-preflight', nonce=old['binding']['nonce'],
            worker_pid=0, now_ns=LATE)['eligible'])
        fresh, updated = final_pair(old, review, fresh_raw(old))
        self.assertTrue(decide(fresh, updated)['eligible'])
        self.assertEqual(updated['initial_capture_sha256'], admission.policy.canonical_hash(old))

    def test_old_bound_capture_cannot_be_relabelled(self):
        old, _ = initial_pair()
        with self.assertRaisesRegex(ValueError, 'unbound'):
            admission.bind_capture(old, run_id=RUN, manifest_sha256=SHA,
                stage='worker-final', nonce='owned-attempt-token', worker_pid=71)

    def test_raw_wrong_run_or_manifest_rejected_before_binding(self):
        old, _ = initial_pair()
        for key in ('run_id', 'manifest_sha256'):
            raw = fresh_raw(old); raw[key] = 'other'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'identity mismatch'):
                admission.bind_capture(raw, run_id=RUN, manifest_sha256=SHA,
                    stage='worker-final', nonce='owned-attempt-token', worker_pid=71)

    def test_each_final_binding_field_is_checked(self):
        old, review = initial_pair(); fresh, updated = final_pair(old, review, fresh_raw(old))
        for target in ('binding', 'capture_binding'):
            for key in fresh['binding']:
                c, r = copy.deepcopy(fresh), copy.deepcopy(updated)
                (c['binding'] if target == 'binding' else r['capture_binding'])[key] = 'other'
                with self.subTest(target=target, key=key), self.assertRaisesRegex(ValueError, 'mismatch'):
                    decide(c, r)

    def test_top_level_identity_not_silently_overwritten(self):
        old, review = initial_pair(); fresh, updated = final_pair(old, review, fresh_raw(old))
        for key in ('run_id', 'manifest_sha256'):
            changed = copy.deepcopy(fresh); changed[key] = 'other'
            with self.subTest(key=key), self.assertRaises(ValueError): decide(changed, updated)

    def test_initial_review_from_another_manifest_not_inherited(self):
        old, review = initial_pair(); review['capture_binding']['manifest_sha256'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'mismatch'): final_pair(old, review, fresh_raw(old))

    def test_previous_authorization_does_not_supply_a_workload_review(self):
        old, _ = initial_pair()
        with self.assertRaises(ValueError):
            final_pair(old, dict(explicit_user_approval=True, manifest_sha256=SHA), fresh_raw(old))

    def test_new_or_disappeared_gpu_process_stops(self):
        old, review = initial_pair()
        for new in (True, False):
            raw = fresh_raw(old)
            raw['inventory']['compute_pids'] = [42, 43] if new else []
            with self.subTest(new=new), self.assertRaisesRegex(ValueError, 'process set changed'):
                final_pair(old, review, raw)

    def test_current_process_identity_changes_stop(self):
        old, review = initial_pair()
        for key in ('path', 'command_line', 'creation_date', 'parent_pid'):
            raw = fresh_raw(old); raw['inventory']['processes'][0][key] = 'changed'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'identity changed'):
                final_pair(old, review, raw)

    def test_current_parent_identity_changes_stop(self):
        old, review = initial_pair(); raw = fresh_raw(old)
        raw['inventory']['processes'][1]['command_line'] = 'different parent workload'
        with self.assertRaisesRegex(ValueError, 'identity changed'): final_pair(old, review, raw)

    def test_nvml_identity_changes_stop(self):
        old, review = initial_pair(); raw = fresh_raw(old)
        raw['inventory']['raw_query'] = ['42, different.exe, [N/A]']
        with self.assertRaisesRegex(ValueError, 'identity changed'): final_pair(old, review, raw)

    def test_all_resource_minima_still_apply_to_new_capture(self):
        old, review = initial_pair()
        for key in ('host_free_bytes', 'disk_free_bytes', 'gpu_free'):
            raw = fresh_raw(old)
            if key == 'gpu_free': raw['gpu_samples'][1]['free_mib'] -= 1
            else: raw[key] -= 1
            fresh, updated = final_pair(old, review, raw)
            with self.subTest(key=key): self.assertFalse(decide(fresh, updated)['eligible'])

    def test_new_capture_still_expires_at_30_seconds(self):
        old, review = initial_pair(); fresh, updated = final_pair(old, review, fresh_raw(old))
        start = fresh['collection_start_time_ns']
        self.assertTrue(decide(fresh, updated, start+30*10**9)['eligible'])
        self.assertIn('stale_or_future_or_reversed_capture',
            decide(fresh, updated, start+30*10**9+1)['blockers'])

    def test_invalid_gpu_sample_or_query_blocks(self):
        old, review = initial_pair()
        for field in ('uuid', 'timestamp', 'query_failed', 'missing_sample'):
            raw = fresh_raw(old)
            if field == 'uuid': raw['gpu_samples'][0]['uuid'] = 'wrong GPU'
            elif field == 'timestamp': raw['gpu_samples'][0]['time_ns'] = LATE+1
            elif field == 'query_failed': raw['inventory']['compute_query_succeeded'] = False
            else: raw['gpu_samples'].pop()
            fresh, updated = final_pair(old, review, raw)
            with self.subTest(field=field):
                try: self.assertFalse(decide(fresh, updated)['eligible'])
                except ValueError: pass

    def test_unknown_workload_and_bad_review_cannot_be_transferred(self):
        old, review = initial_pair()
        for field in ('unknown', 'conflict', 'digest'):
            altered = copy.deepcopy(review)
            if field == 'unknown': altered['gpu_processes'][0]['category'] = 'unknown'
            elif field == 'conflict': altered['conflicting_workloads'] = ['compute job']
            else: altered['inventory_sha256'] = 'bad'
            with self.subTest(field=field), self.assertRaises(ValueError):
                final_pair(old, altered, fresh_raw(old))

    def test_thresholds_and_limits_unchanged(self):
        self.assertEqual(admission.policy.MAX_AGE_NS, 30*10**9)
        self.assertEqual(runner.LIMITS['child_timeout_s'], 60)
        self.assertEqual(runner.LIMITS['outer_timeout_s'], 90)
        self.assertEqual(runner.LIMITS['process_private_commit'], 4*1024**3)
        self.assertEqual(runner.LIMITS['child_job_private_commit'], 8*1024**3)
        self.assertEqual(runner.LIMITS['outer_job_private_commit'], 12*1024**3)

    def test_graphics_only_pid_not_in_review_blocks(self):
        old, review = initial_pair(); raw = fresh_raw(old)
        raw['inventory']['gpu_table_pids'].append(99)
        fresh, updated = final_pair(old, review, raw)
        with self.assertRaisesRegex(ValueError, 'unreviewed PIDs'): decide(fresh, updated)

    def test_missing_or_failed_graphics_table_blocks(self):
        old, review = initial_pair()
        for value in (False, None):
            raw = fresh_raw(old); raw['inventory']['gpu_table_query_succeeded'] = value
            fresh, updated = final_pair(old, review, raw)
            with self.subTest(value=value), self.assertRaises(ValueError): decide(fresh, updated)

    def test_graphics_table_parser_checks_all_process_types_and_rejects_unknown_rows(self):
        raw = '| Processes: |\n| GPU GI CI PID Type Process name GPU Memory |\n| ID ID Usage |\n|====|\n'
        rows = '| 0 N/A N/A 42 C+G desktop.exe N/A |\n| 0 N/A N/A 43 G UI.exe N/A |\n| 0 N/A N/A 44 C compute.exe N/A |\n+---+\n'
        self.assertEqual(admission.gpu_table_pids(raw+rows), [42, 43, 44])
        for malformed in ('', raw, raw+'| malformed row |\n', raw+rows+rows):
            with self.subTest(raw=malformed), self.assertRaises(ValueError):
                admission.gpu_table_pids(malformed)
        self.assertEqual(admission.gpu_table_pids(raw+'| No running processes found |\n'), [])

    def test_previous_exact_hash_authorization_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            previous = dict(run_id='69c9d924-51bf-468b-b2bb-527992f28e8f',
                manifest_sha256='b026e525adad2a933c90c250837598c87f5dc314820e11dcb1cd2be12d1185d3',
                scenario=runner.SCENARIO, maximum_executions=1, explicit_user_approval=True)
            (work/'authorization.json').write_text(json.dumps(previous))
            with self.assertRaisesRegex(ValueError, 'exact-hash'):
                runner.approval(None, work, dict(run_id=RUN), SHA)


class WorkerBoundary(unittest.TestCase):
    def setUp(self):
        self.old, self.review = initial_pair()
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)
        self.journal = unittest.mock.Mock()
        self.journal_patch = patch.object(worker, '_journal', self.journal)
        self.journal_patch.start()
        self.addCleanup(self.journal_patch.stop)
        self.addCleanup(self.temp.cleanup)

    def run_final(self, raw=None, initialized=None):
        with patch.object(admission, 'collect_p23', return_value=raw or fresh_raw(self.old)) as collect:
            with patch.object(admission.policy.time, 'time_ns', return_value=LATE):
                worker.final_admission(self.work, self.old, self.review, 'owned-attempt-token',
                    SHA, dict(run_id=RUN), cuda_initialized=initialized or (lambda: False))
            return collect

    @offline_test
    def test_worker_collects_once_after_simulated_long_import_no_torch(self):
        collect = self.run_final()
        collect.assert_called_once_with()
        decision = json.loads((self.work/'admission-worker-final.json').read_bytes())
        self.assertTrue(decision['eligible'])
        fresh = json.loads((self.work/'final-capture.json').read_bytes())
        self.assertEqual(fresh['binding']['worker_pid'], os.getpid())
        self.assertEqual(fresh['binding']['nonce'], 'owned-attempt-token')
        self.assertNotIn('torch', sys.modules)

    def test_low_ram_worker_stops_and_retains_rejection(self):
        raw = fresh_raw(self.old); raw['host_free_bytes'] = 4*1024**3-1
        with self.assertRaisesRegex(ValueError, 'host_free_below_4GiB'): self.run_final(raw)
        self.assertFalse(json.loads((self.work/'admission-worker-final.json').read_bytes())['eligible'])
        self.assertFalse((self.work/'result.json').exists())

    def test_collector_timeout_is_not_retried(self):
        with patch.object(admission, 'collect_p23', side_effect=TimeoutError('sensor deadline')) as collect:
            with self.assertRaises(TimeoutError):
                worker.final_admission(self.work, self.old, self.review, 'owned-attempt-token',
                    SHA, dict(run_id=RUN), cuda_initialized=lambda: False)
        collect.assert_called_once_with()
        self.assertFalse((self.work/'admission-worker-final.json').exists())

    def test_cuda_already_initialized_prevents_collection(self):
        with patch.object(admission, 'collect_p23') as collect:
            with self.assertRaisesRegex(ValueError, 'before final capture'):
                worker.final_admission(self.work, self.old, self.review, 'owned-attempt-token',
                    SHA, dict(run_id=RUN), cuda_initialized=lambda: True)
            collect.assert_not_called()

    def test_cuda_initialization_during_capture_stops(self):
        initialized = unittest.mock.Mock(side_effect=[False, True])
        with self.assertRaisesRegex(ValueError, 'during stdlib'): self.run_final(initialized=initialized)

    def test_invalid_capture_retained_but_not_admitted(self):
        raw = fresh_raw(self.old); raw['manifest_sha256'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'identity mismatch'): self.run_final(raw)
        self.assertTrue((self.work/'final-capture-raw.json').exists())
        self.assertFalse((self.work/'admission-worker-final.json').exists())

    def test_worker_rejects_changed_process_and_retains_capture(self):
        raw = fresh_raw(self.old); raw['inventory']['processes'][0]['creation_date'] = 'reused PID'
        with self.assertRaisesRegex(ValueError, 'identity changed'): self.run_final(raw)
        self.assertTrue((self.work/'final-capture.json').exists())
        self.assertFalse((self.work/'final-review.json').exists())

    def test_publication_delay_cannot_bypass_final_freshness(self):
        original = runner.publish
        now = [LATE]
        def publish(path, value):
            original(path, value)
            if Path(path).name == 'admission-worker-final.json': now[0] += 31*10**9
        with patch.object(admission, 'collect_p23', return_value=fresh_raw(self.old)):
            with patch.object(runner, 'publish', side_effect=publish):
                with patch.object(admission.policy.time, 'time_ns', side_effect=lambda: now[0]):
                    with self.assertRaisesRegex(ValueError, 'expired before recipe'):
                        worker.final_admission(self.work, self.old, self.review, 'owned-attempt-token',
                            SHA, dict(run_id=RUN), cuda_initialized=lambda: False)


if __name__ == '__main__': unittest.main(verbosity=2)
