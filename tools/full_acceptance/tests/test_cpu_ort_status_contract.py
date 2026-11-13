"""Offline contract regression: valid mapped CPU execution vs incomplete/fallback."""
import unittest
from unittest.mock import Mock

from full_acceptance.handlers import _trained_cpu_ort_execution_confirmed


class CpuOrtStatusContractTests(unittest.TestCase):
    def setUp(self):
        self.diagnostics = dict(events=[{'provider': 'CPUExecutionProvider'}],
                                completed_runs=13, failure_kind=None,
                                observed_execution_providers=['CPUExecutionProvider'])

    @staticmethod
    def value(status, **changes):
        analysis = dict(status=status, reason=None, fallback_detected=False)
        analysis.update(changes)
        return Mock(to_dict=Mock(return_value={'analysis': analysis}))

    def test_real_expected_execution_is_accepted(self):
        self.assertTrue(_trained_cpu_ort_execution_confirmed(
            self.value('expected_execution_observed'), self.diagnostics))

    def test_ambiguous_fallback_or_legacy_completed_is_rejected(self):
        for status in ('unknown', 'unavailable', 'execution_observed',
                       'fallback_detected', 'completed'):
            with self.subTest(status=status):
                self.assertFalse(_trained_cpu_ort_execution_confirmed(
                    self.value(status), self.diagnostics))

    def test_missing_events_or_wrong_provider_is_rejected(self):
        for changes in (dict(events=[]), dict(completed_runs=None),
                        dict(failure_kind='ort_error'),
                        dict(observed_execution_providers=['CUDAExecutionProvider'])):
            diag = dict(self.diagnostics, **changes)
            self.assertFalse(_trained_cpu_ort_execution_confirmed(
                self.value('expected_execution_observed'), diag))

    def test_fallback_and_unexplained_status_are_rejected(self):
        for changes in (dict(fallback_detected=True), dict(reason='unknown scope')):
            self.assertFalse(_trained_cpu_ort_execution_confirmed(
                self.value('expected_execution_observed', **changes),
                self.diagnostics))
