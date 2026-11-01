"""Controller finalization fault injection; every worker is an offline double."""
import types
from unittest.mock import Mock, patch

from test_infrastructure import Fixture
from full_acceptance import cli, core, reporting, safety


class ControllerFinalizationTests(Fixture):
    def exercise(self, *, preserve_error=None, report_error=None,
                 diagnostic_error=False, terminal_error=False, progress_error=False):
        self.m['hardware']['gpu_uuid'] = 'GPU-offline-double'
        for entry in self.m['experiments']:
            entry['authorization']['approved'] = entry['id'] in ('02.regression', '03.cpu', '04.cuda')
        self.seal()
        args = types.SimpleNamespace(manifest=self.path, approved_sha256=self.hash, cuda_subtasks=False)
        self.progress = []
        self.publications = []
        original_publish = cli.publish

        def preserve(manifest):
            self.progress.append(core.read(self.out / 'run-status.json'))
            if preserve_error is not None:
                raise preserve_error
            return dict(all_readable_preserved=True)

        def reports(manifest, rows, *, state, preparation):
            if state != 'RUNNING':
                self.assertEqual([rows[eid]['status'] for eid in ('02.regression', '03.cpu', '04.cuda')],
                                 ['PASS', 'PASS', 'PASS'])
            if state != 'RUNNING' and report_error is not None:
                raise report_error

        def publish(path, value, **kwargs):
            self.publications.append((path.name, value))
            if diagnostic_error and path.name == 'preservation-error.json':
                raise PermissionError('diagnostic destination denied')
            if progress_error and path.name == 'run-status.json' and value.get('state') == 'FINALIZING':
                raise PermissionError('progress destination denied')
            if terminal_error and path.name == 'run-status.json' and value.get('controller_active') is False:
                raise PermissionError('terminal destination denied')
            return original_publish(path, value, **kwargs)

        worker = Mock(return_value=dict(status='PASS', attempted=True, metrics=[]))
        with patch.object(cli, 'verify', return_value=self.m), \
             patch.object(cli, 'preserve', side_effect=preserve), \
             patch.object(cli, 'write_reports', side_effect=reports), \
             patch.object(cli, 'publish', side_effect=publish), \
             patch.object(safety, 'supervise', worker):
            code = cli.execute(args)
        self.assertEqual([call.args[1]['id'] for call in worker.call_args_list],
                         ['02.regression', '03.cpu', '04.cuda'])
        return code, core.read(self.out / 'run-status.json')

    def assert_terminal(self, status):
        self.assertFalse(status['controller_active'])
        self.assertFalse(status['suite_complete'])
        self.assertEqual(status['run_id'], self.m['run_id'])
        self.assertEqual(status['manifest_sha256'], self.hash)

    def test_success_records_progress_then_terminal_incomplete_without_changing_stage_passes(self):
        code, status = self.exercise()
        self.assertEqual(code, 2)
        self.assert_terminal(status)
        self.assertEqual(status['state'], 'INCOMPLETE')
        self.assertEqual(status['preservation_status'], 'PASS')
        self.assertEqual(self.progress[0]['state'], 'FINALIZING')
        self.assertEqual(self.progress[0]['finalization_phase'], 'preservation')
        self.assertTrue(self.progress[0]['controller_active'])

    def test_unexpected_preserve_exception_still_records_terminal_failure(self):
        code, status = self.exercise(preserve_error=RuntimeError('unexpected scanner error'))
        self.assertEqual(code, 1)
        self.assert_terminal(status)
        self.assertEqual(status['state'], 'FAILED_PRESERVATION')
        self.assertEqual(status['preservation_status'], 'FAIL')
        self.assertEqual(status['finalization_errors'][0]['phase'], 'preservation')

    def test_expected_preserve_error_is_a_failed_exit_even_when_workers_pass(self):
        code, status = self.exercise(preserve_error=ValueError('protected input changed'))
        self.assertEqual(code, 1)
        self.assert_terminal(status)
        self.assertEqual(status['state'], 'FAILED_PRESERVATION')

    def test_report_error_still_records_terminal_failure(self):
        code, status = self.exercise(report_error=OSError('report disk error'))
        self.assertEqual(code, 1)
        self.assert_terminal(status)
        self.assertEqual(status['state'], 'FAILED_CONTROLLER')
        self.assertEqual(status['preservation_status'], 'PASS')
        self.assertEqual(status['finalization_errors'][0]['phase'], 'reports')

    def test_failed_diagnostic_write_cannot_skip_terminal_record(self):
        code, status = self.exercise(preserve_error=RuntimeError('scanner'), diagnostic_error=True)
        self.assertEqual(code, 1)
        self.assert_terminal(status)
        self.assertEqual(status['state'], 'FAILED_PRESERVATION')
        self.assertTrue(any(e['phase'] == 'preservation_error_record' for e in status['finalization_errors']))

    def test_preserve_interrupt_is_aborted_and_not_preservation_pass(self):
        code, status = self.exercise(preserve_error=KeyboardInterrupt())
        self.assertEqual(code, 1)
        self.assert_terminal(status)
        self.assertEqual(status['state'], 'ABORTED')
        self.assertEqual(status['preservation_status'], 'FAIL')

    def test_preserve_and_reports_can_both_fail_without_losing_terminal_record(self):
        code, status = self.exercise(preserve_error=RuntimeError('scanner'), report_error=RuntimeError('report'))
        self.assertEqual(code, 1)
        self.assert_terminal(status)
        self.assertEqual(status['state'], 'FAILED_PRESERVATION')
        self.assertEqual([e['phase'] for e in status['finalization_errors']], ['preservation', 'reports'])

    def test_unwritable_terminal_destination_fails_without_claiming_finalization(self):
        code, status = self.exercise(terminal_error=True)
        self.assertEqual(code, 1)
        self.assertTrue(status['controller_active'])
        self.assertEqual(status['state'], 'FINALIZING')
        self.assertTrue(any(name == 'run-status.json' and value.get('controller_active') is False
                            for name, value in self.publications))

    def test_failed_progress_publication_does_not_skip_preserve_or_terminal_record(self):
        code, status = self.exercise(progress_error=True)
        self.assertEqual(code, 1)
        self.assert_terminal(status)
        self.assertEqual(len(self.progress), 1)
        self.assertEqual(status['preservation_status'], 'PASS')
        self.assertEqual(status['state'], 'FAILED_CONTROLLER')

    def test_unexpected_report_exception_still_records_terminal_failure(self):
        code, status = self.exercise(report_error=RuntimeError('unexpected formatter error'))
        self.assertEqual(code, 1)
        self.assert_terminal(status)
        self.assertEqual(status['state'], 'FAILED_CONTROLLER')

    def test_partial_real_report_write_preserves_saved_stage_passes(self):
        rows = self.old_matrix()
        for eid in ('02.regression', '03.cpu', '04.cuda'):
            rows[eid].update(status='PASS', attempted=True)
        original_publish = reporting.publish

        def fail_late_report(path, value, **kwargs):
            if path.name == 'model-evidence.json':
                raise OSError('late report output denied')
            return original_publish(path, value, **kwargs)

        with patch.object(cli, 'preserve', return_value=dict(all_readable_preserved=True)), \
             patch.object(reporting, 'publish', side_effect=fail_late_report):
            state, complete, errors = cli._finalize_execution(
                self.m, rows, state='INCOMPLETE', complete=False,
                preparation={}, manifest_sha256=self.hash)
        self.assertEqual(state, 'FAILED_CONTROLLER')
        self.assertFalse(complete)
        self.assertEqual(errors[0]['phase'], 'reports')
        status = core.read(self.out / 'run-status.json')
        self.assert_terminal(status)
        saved = {r['id']: r for r in core.read(self.out / 'stage-results.json')['experiments']}
        self.assertEqual([saved[eid]['status'] for eid in ('02.regression', '03.cpu', '04.cuda')],
                         ['PASS', 'PASS', 'PASS'])

