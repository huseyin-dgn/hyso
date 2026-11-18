"""G3 offline contract tests. No subprocess, CUDA, factory import or worker run."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hyso.analysis.feature_plan import FeaturePlanRequest, ModelContext
from hyso.analysis.isolated_bridge import (
    TrustedFactoryBinding, IsolatedCPUExecutionPolicy, IsolatedBridgeError,
    execute_isolated_cpu, _read_worker_output, _source_receipt)


def request(models=('A',), feature='P11', *, declared=True):
    return FeaturePlanRequest('single', tuple(ModelContext(m, trusted_factory=True,
        input_contract=declared) for m in models), features=(feature,))


class G3Contract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'trusted.py'
        self.path.write_text('def build():\n    raise Exception("Never imported in parent")\n', encoding='utf-8')
        self.binding = TrustedFactoryBinding('A', str(self.path), _source_receipt(self.path))
        self.policy = IsolatedCPUExecutionPolicy(explicit_authorization=True)

    def test_no_authorization_no_work(self):
        with self.assertRaisesRegex(IsolatedBridgeError, 'explicit_worker_execution_authorization'):
            execute_isolated_cpu(request(), (self.binding,), authorized_features=('P11',))

    def test_source_hash_mismatch_fail_closed(self):
        b = TrustedFactoryBinding('A', str(self.path), 'f'*64)
        with self.assertRaisesRegex(IsolatedBridgeError, 'factory_sha256_mismatch'):
            execute_isolated_cpu(request(), (b,), authorized_features=('P11',), policy=self.policy)

    def test_no_windows_no_inprocess_fallback(self):
        from hyso.analysis import isolated_bridge as g3
        with patch.object(g3, '_windows_available', return_value=False):
            with self.assertRaisesRegex(IsolatedBridgeError, 'windows_job_backend_required'):
                execute_isolated_cpu(request(), (self.binding,), authorized_features=('P11',), policy=self.policy)

    def test_unauthorized_feature_and_wrong_model(self):
        for features in (('P31',), ('P11','P11'), ('P02',)):
            with self.subTest(features=features), self.assertRaises(IsolatedBridgeError):
                execute_isolated_cpu(request(), (self.binding,), authorized_features=features, policy=self.policy)
        b = TrustedFactoryBinding('B', str(self.path), _source_receipt(self.path))
        with self.assertRaisesRegex(IsolatedBridgeError, 'factory_model_identity_mismatch'):
            execute_isolated_cpu(request(), (b,), authorized_features=('P11',), policy=self.policy)

    def test_unmet_input_or_factory_declared_blocked(self):
        with self.assertRaisesRegex(IsolatedBridgeError, 'G1_prerequisites_not_satisfied'):
            execute_isolated_cpu(request(feature='P02', declared=False), (self.binding,),
                                 authorized_features=('P02',), policy=self.policy)

    def test_policy_types_bounds(self):
        for kwargs in ({'timeout_seconds':1},{'timeout_seconds':300},
                       {'job_memory_limit_bytes':100}, {'explicit_authorization':1},
                       {'batch_count':0}, {'max_input_bytes':1<<30}):
            with self.subTest(kwargs=kwargs), self.assertRaises(IsolatedBridgeError):
                IsolatedCPUExecutionPolicy(**kwargs)

    def test_factory_rejects_bad_path_hash_name_kwargs(self):
        for kwargs in (
            {'source_file':'relative.py'}, {'source_sha256':'z'*64},
            {'factory_name':'x.y'}, {'kwargs':{'__import__':'x'*100000}},
            {'kwargs':{'x':object()}}, {'kwargs':{'x':float('nan')}},
        ):
            with self.subTest(keys=list(kwargs)), self.assertRaises((IsolatedBridgeError, TypeError)):
                TrustedFactoryBinding(**({**dict(model_id='A',source_file=str(self.path),
                   source_sha256=_source_receipt(self.path)), **kwargs}))

    def test_changes_after_sha_prevent_execution(self):
        self.path.write_text('def build():\n    return 4\n', encoding='utf-8')
        with self.assertRaisesRegex(IsolatedBridgeError, 'factory_sha256_mismatch'):
            execute_isolated_cpu(request(), (self.binding,), authorized_features=('P11',), policy=self.policy)

    def test_failed_timeout_does_not_return_native_success(self):
        from hyso.analysis import isolated_bridge as g3
        def fake_job(command, directory, **kwargs):
            return dict(status='timeout', exit_code=1, assigned_before_resume=True,
                        active_owned_processes=0, hard_job_memory_limit_bytes=kwargs['job_memory_limit_bytes'])
        with patch.object(g3, '_windows_available', return_value=True), patch.object(
                __import__('hyso.analysis._windows_job', fromlist=['run_bounded_windows']),
                'run_bounded_windows', side_effect=fake_job):
            got = execute_isolated_cpu(request(), (self.binding,), authorized_features=('P11',), policy=self.policy)
        self.assertEqual(got.status, 'FAILED')
        self.assertEqual(got.reason, 'timeout')
        self.assertIsNone(got.native_bridge_report)
        self.assertEqual(got.acceptance, 'NOT_EVALUATED')

    def test_fake_zero_exit_without_terminal_is_not_success(self):
        from hyso.analysis import isolated_bridge as g3
        def fake_job(command, directory, **kwargs):
            return dict(status='process_exited', exit_code=0, assigned_before_resume=True,
                        active_owned_processes=0)
        with patch.object(g3, '_windows_available', return_value=True), patch.object(
                __import__('hyso.analysis._windows_job', fromlist=['run_bounded_windows']),
                'run_bounded_windows', side_effect=fake_job):
            got = execute_isolated_cpu(request(), (self.binding,), authorized_features=('P11',), policy=self.policy)
        self.assertEqual(got.status,'FAILED')
        self.assertIn('missing_or_oversize_worker_terminal',got.reason)

    def test_tampered_capture_rejected(self):
        terminal = Path(self.tmp.name)/'terminal.json'
        report = dict(acceptance='NOT_EVALUATED', authorization='EXPLICIT_CPU_P02_P11_ONLY',
                      scope='trusted_inprocess_cpu;not_isolated_no_hard_runtime_memory_limit',
                      requested_feature_ids=['P11'], executed_feature_ids=['P11'],
                      omitted_features=[], comparisons=[], captures=[dict(
                          model_id='A',feature_id='P11',status='MEASURED_NATIVE_SCOPE',
                          source_sha256='0'*64,native_result=dict(feature_ids=['P11']))])
        terminal.write_text(json.dumps(dict(status='COMPLETED',reason=None,report=report)),encoding='utf-8')
        with self.assertRaisesRegex(IsolatedBridgeError,'worker_native_source_hash_mismatch'):
            _read_worker_output(terminal,('P11',),('P11',),(self.binding,),{})

    def test_result_requires_correct_observed_diff(self):
        terminal = Path(self.tmp.name)/'terminal.json'
        payload = dict(feature_ids=['P11'],schema_version=1,protocol=dict(byte_accounting='unique_tensor_object_logical'),
                       provenance=dict(environment=dict(device='cpu')),
                       measurements=dict(parameter_buffer_bytes=dict(value=100,unit='byte',status='derived')))
        content=json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(',',':'))
        cap=lambda mid: dict(model_id=mid,feature_id='P11',status='MEASURED_NATIVE_SCOPE',
                             source_sha256=hashlib.sha256(content.encode()).hexdigest(),native_result=payload)
        report=dict(acceptance='NOT_EVALUATED', authorization='EXPLICIT_CPU_P02_P11_ONLY',
                    scope='trusted_inprocess_cpu;not_isolated_no_hard_runtime_memory_limit',
                    requested_feature_ids=['P11'], executed_feature_ids=['P11'],
                    omitted_features=[], comparisons=[dict(status='FAKE_WINNER')],
                    captures=[cap('A'),cap('B')])
        terminal.write_text(json.dumps(dict(status='COMPLETED',reason=None,report=report)),encoding='utf-8')
        b=TrustedFactoryBinding('B',str(self.path),_source_receipt(self.path))
        with self.assertRaisesRegex(IsolatedBridgeError,'worker_comparison_not_derived'):
            _read_worker_output(terminal,('P11',),('P11',),(self.binding,b),{})

if __name__=='__main__': unittest.main()
