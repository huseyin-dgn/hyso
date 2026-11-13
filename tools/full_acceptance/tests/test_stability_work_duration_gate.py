"""Offline regression for P31/P33 acceptance's verified work-duration gate.

No model, GPU, sensor, native worker or long run starts in these tests.
"""
import sys
import types
from unittest.mock import patch

from test_infrastructure import Fixture
from full_acceptance import handlers


class VerifiedWorkDurationTests(Fixture):
    def _stage(self, *, energy, requested, work_seconds, completion_seconds,
               status='completed', exit_code=0, active_processes=0, closed=True):
        class Protocol:
            def __init__(self, **kwargs):
                self.duration_seconds = kwargs['duration_seconds']

        payload = {
            'analysis': {
                'status': status,
                'duration_seconds': work_seconds,
                'completed': 256,
            },
            'evidence': {
                'controller': {
                    'exit_code': exit_code,
                    'cleanup': {'active_processes': active_processes},
                },
                'terminal': {'closed': closed},
            },
        }
        diagnostics = {
            'completion_window': {'seconds': completion_seconds},
            'continuity': {
                'requested_duration_satisfied': status == 'completed',
                'no_failures_observed': status == 'completed',
                'no_sampling_gaps_detected': True,
            },
        }

        class Raw:
            def to_dict(self):
                return payload
            def stability_diagnostics(self):
                return diagnostics

        class Energy:
            def to_dict(self):
                return {'analysis': {
                    'gross_joules': 527.1697374,
                    'net_above_idle_joules': -276.8328634,
                    'source_complete': status == 'completed',
                    'reason': None,
                    'idle_baseline': {'reason': None},
                    'limitations': [],
                }}

        raw = Raw()
        fake = types.ModuleType('hyso.performance')
        fake.WorkloadRecipe = lambda *args, **kwargs: object()
        fake.StabilityProtocol = Protocol
        fake.measure_stability = lambda *args, **kwargs: raw
        fake.reopen_stability = lambda *args, **kwargs: raw
        fake.analyze_energy = lambda *args, **kwargs: Energy()
        fake.RuntimeResult = object
        self.m['protocols'] = {
            'energy': {
                'active_seconds': requested,
                'interval_seconds': .5,
                'idle_seconds': 5,
                'max_gap_seconds': 2,
                'idle_max_age_seconds': 60,
            },
            'stability': {
                'duration_seconds': requested,
                'interval_seconds': 5,
            },
        }
        self.m['hardware']['gpu_uuid'] = 'GPU-fixture-only'
        with patch.dict(sys.modules, {'hyso.performance': fake}), \
             patch.object(handlers, 'save', return_value={'sha256': 'fixture'}):
            return handlers.stability(self.m, self.out / 'no-native-work', energy=energy)

    def test_p33_verified_thirty_seconds_does_not_require_thirty_seconds_between_completions(self):
        outcome = self._stage(energy=True, requested=30.,
                              work_seconds=30.0053407, completion_seconds=29.9973609)
        self.assertEqual(outcome['status'], 'PASS')
        self.assertTrue(outcome['criteria']['trained_energy_full'])
        self.assertAlmostEqual(outcome['metrics'][0]['value'], 30.0053407)

    def test_p31_verified_7200_seconds_uses_work_window_not_completion_window(self):
        outcome = self._stage(energy=False, requested=7200.,
                              work_seconds=7200.005, completion_seconds=7199.997)
        self.assertEqual(outcome['status'], 'PASS')
        self.assertTrue(outcome['criteria']['P31_uninterrupted_7200'])
        self.assertAlmostEqual(outcome['metrics'][0]['value'], 7200.005)

    def test_incomplete_run_cannot_pass_with_full_energy_or_long_duration(self):
        for energy, requested in ((True, 30.), (False, 7200.)):
            with self.subTest(energy=energy):
                outcome = self._stage(energy=energy, requested=requested,
                    work_seconds=requested + 1, completion_seconds=requested + 1,
                    status='aborted')
                self.assertEqual(outcome['status'], 'PARTIAL' if energy else 'FAIL')

    def test_failed_cleanup_cannot_pass_with_sufficient_work(self):
        for energy, requested in ((True, 30.), (False, 7200.)):
            with self.subTest(energy=energy):
                with self.assertRaises(handlers.UnsafeCaptureError):
                    self._stage(energy=energy, requested=requested,
                        work_seconds=requested + 1, completion_seconds=requested + 1,
                        active_processes=1)

    def test_missing_terminal_duration_is_null_with_reason_not_a_metric_exception(self):
        outcome=self._stage(energy=False,requested=7200.,work_seconds=None,completion_seconds=0,status='aborted')
        self.assertEqual(outcome['status'],'FAIL')
        self.assertIsNone(outcome['metrics'][0]['value'])
        self.assertEqual(outcome['metrics'][0]['kind'],'missing')
        self.assertTrue(outcome['metrics'][0]['reason'])
