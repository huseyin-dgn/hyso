"""Offline acceptance fault injection. Synthetic duration fields execute no work."""
import copy
import sys
import types
from unittest.mock import Mock, patch

from test_infrastructure import Fixture
from full_acceptance import cli, core, handlers, p31_evidence, reporting, safety, workloads


class P32SourceTests(Fixture):
    def setUp(self):
        super().setUp()
        self.stage=self.out/'stages/10.stability';self.stage.mkdir(parents=True)
        self.m['protocols']=dict(stability=dict(duration_seconds=7200.))
        self.m['assets']=dict(models={'B':dict(sha256='b'*64)})
        next(e for e in self.m['experiments'] if e['id']=='10.stability')['authorization']['approved']=True
        self.seal()
        self.terminal=dict(id='10.stability',status='PASS',manifest_sha256=self.hash,
            criteria=dict(P31_uninterrupted_7200=True))
        self.outer=dict(experiment='10.stability',manifest_sha256=self.hash,exit_code=0,reason=None,
            native_crash=False,owned_job_only=True,cleanup=dict(active_processes=0))
        self.data=dict(feature_id='P31',analysis=dict(status='completed',duration_seconds=7200.01),evidence=dict(
            header=dict(stability_version=2,workload=dict(kind='model_inference',device_id='cpu',trained=True,
                precision='fp32',input_shape=[256,1,28,28],checkpoint=dict(sha256='b'*64))),
            request=dict(protocol=self.m['protocols']['stability'],recipe=dict(factory='full_acceptance.workloads:inference_work',
                configuration=dict(manifest_path=str(self.out/'manifest.json'),device='cpu'),
                search_path=str(self.snapshot/'tools'),gpu_uuid=None)),
            terminal=dict(closed=True),controller=dict(exit_code=0,reason=None,cleanup=dict(active_processes=0)),
            rows=[dict(rss=dict(rss_bytes=1),process_commit=dict(value=1))]))
        self.continuity=dict(requested_duration_satisfied=True,no_failures_observed=True,no_sampling_gaps_detected=True)
        self.raw=Mock();self.raw.to_dict.side_effect=lambda:self.data
        self.raw.stability_diagnostics.side_effect=lambda:dict(continuity=self.continuity)

    def run_reader(self, journal=None):
        core.publish(self.stage/'result.json',self.terminal)
        core.publish(self.stage/'supervisor-receipt.json',self.outer)
        reopened=Mock();reopened.to_dict.return_value=journal or self.data
        with patch('hyso.performance.RuntimeResult.from_json',return_value=self.raw), \
             patch('hyso.performance.reopen_stability',return_value=reopened):
            return p31_evidence.validated_source(self.m)

    def test_complete_current_receipts_and_source_admitted_without_execution(self):
        self.assertIs(self.run_reader(),self.raw)

    def test_invalid_source_and_receipt_cases_rejected(self):
        for case in ('short','aborted','missing_terminal','gap','close_failure','other_model','cuda','untrained',
                     'other_manifest','missing_rss','missing_commit','outer_kill','wrong_outer_manifest','false_criterion',
                     'other_factory','header_missing'):
            with self.subTest(case=case):
                data=copy.deepcopy(self.data);outer=copy.deepcopy(self.outer);terminal=copy.deepcopy(self.terminal)
                continuity=self.continuity.copy();e=data['evidence']
                if case=='short':data['analysis']['duration_seconds']=7199.99
                elif case=='aborted':data['analysis']['status']='aborted'
                elif case=='missing_terminal':e['terminal']=None
                elif case=='gap':continuity['no_sampling_gaps_detected']=False
                elif case=='close_failure':e['terminal']['closed']=False
                elif case=='other_model':e['header']['workload']['checkpoint']['sha256']='a'*64
                elif case=='cuda':e['header']['workload']['device_id']='GPU-test'
                elif case=='untrained':e['header']['workload']['trained']=False
                elif case=='other_manifest':e['request']['recipe']['configuration']['manifest_path']='other.json'
                elif case=='missing_rss':e['rows'][0]['rss']=None
                elif case=='missing_commit':e['rows'][0]['process_commit']['value']=None
                elif case=='outer_kill':outer['exit_code']=1
                elif case=='wrong_outer_manifest':outer['manifest_sha256']='0'*64
                elif case=='false_criterion':terminal['criteria']['P31_uninterrupted_7200']=False
                elif case=='other_factory':e['request']['recipe']['factory']='other:factory'
                elif case=='header_missing':e['header']=None
                with patch.object(self,'data',data),patch.object(self,'outer',outer),patch.object(self,'terminal',terminal),patch.object(self,'continuity',continuity):
                    with self.assertRaises(ValueError):self.run_reader()

    def test_missing_receipts_and_journal_mismatch_rejected(self):
        with self.assertRaises(FileNotFoundError):p31_evidence.validated_source(self.m)
        different=copy.deepcopy(self.data);different['evidence']['rows']=[]
        with self.assertRaisesRegex(ValueError,'committed journal'):self.run_reader(different)

    def test_no_analysis_after_source_rejection(self):
        with patch.object(p31_evidence,'validated_source',side_effect=ValueError('incomplete')), \
             patch('hyso.performance.analyze_memory_trend') as analyze:
            with self.assertRaises(ValueError):handlers.memory(self.m,self.out)
            analyze.assert_not_called()


class ContinuationSafetyTests(Fixture):
    def test_counter_failure_or_timeout_is_fatal_without_retry(self):
        for outcome in (types.SimpleNamespace(returncode=1),types.SimpleNamespace(returncode=0xC0000409),
                        handlers.subprocess.TimeoutExpired(['fixture'],120)):
            with self.subTest(outcome=outcome):
                options=dict(side_effect=outcome) if isinstance(outcome,Exception) else dict(return_value=outcome)
                with patch.object(handlers.subprocess,'run',**options) as run:
                    with self.assertRaises(handlers.UnsafeCaptureError) as caught:
                        handlers._run_counter_capture(['never-launched'],None,None)
                    self.assertEqual(run.call_count,1)
                    self.assertTrue(cli._fatal_experiment_boundary(dict(status='FAIL',safety_stop=caught.exception.safety_stop)))

    def test_nested_crashes_cleanup_and_low_memory_remain_fatal(self):
        for data in (dict(outcome='native_crash'),dict(diagnostics=dict(status='crashed')),
                     dict(evidence=dict(controller=dict(exit_code=0xC0000409))),
                     dict(receipt=dict(cleanup=dict(active_processes=1))),
                     dict(evidence=dict(controller=dict(reason='Windows_OS_low_memory'))),
                     dict(evidence=dict(controller=dict(reason='os_low_memory')))):
            with self.subTest(data=data):
                boundary=safety.capture_stop(data);self.assertTrue(boundary)
                row=dict(status='FAIL',safety_stop=boundary,supervisor=dict(exit_code=0,native_crash=False,cleanup=dict(active_processes=0)))
                self.assertEqual(cli._fatal_experiment_boundary(row),boundary)
        self.assertIsNone(safety.capture_stop(dict(receipt=dict(exit_code=0,cleanup=dict(active_processes=0)))))

    def test_missing_worker_terminal_is_fatal_despite_clean_outer_exit(self):
        row=dict(status='FAIL',reason='missing_worker_terminal_result',supervisor=dict(exit_code=0,native_crash=False,cleanup=dict(active_processes=0)))
        self.assertEqual(cli._fatal_experiment_boundary(row),'missing_worker_terminal_result')

    def test_roofline_missing_locale_is_a_prerequisite_not_a_native_attempt(self):
        tool=self.base/'ncu.exe';tool.write_bytes(b'fixture')
        self.m['optional']['ncu_plan']=dict(configuration=dict(tool=dict(path=str(tool))))
        self.m['hardware']['gpu_uuid']='GPU-fixture'
        e=next(e for e in self.m['experiments'] if e['id']=='06.roofline')
        self.assertIn('explicit ncu_plan.number_format required before capture',reporting.prerequisites(self.m,e))
        self.m['optional']['ncu_plan']['configuration']['number_format']='decimal_comma'
        self.assertEqual(reporting.prerequisites(self.m,e),[])

    def test_isolated_memory_crash_stops_handler_before_further_fits(self):
        fake=types.ModuleType('hyso.performance')
        for name in ('TrainingMemoryRecipe','IsolatedTrainingMemoryProtocol','TrainingMemoryProtocol'):setattr(fake,name,lambda *a,**k:object())
        fake.IsolatedTrainingMemoryProtocol=Mock(return_value=object())
        fake.measure_training_with_loader=Mock(return_value=types.SimpleNamespace(timing=object(),stages=object(),loader=object(),throughput=object()))
        fake.measure_gradient_diagnostics=Mock(return_value=object())
        mem=Mock();mem.to_dict.return_value=dict(outcome='native_crash')
        fake.measure_training_memory_isolated=Mock(return_value=mem)
        self.m['protocols']=dict(training=dict(epochs=1,seed=1))
        build=Mock(return_value=dict(trainer=object(),train_loader=object(),val_loader=object()))
        with patch.dict(sys.modules,{'hyso.performance':fake}),patch.object(workloads,'training_recipe',build), \
             patch.object(handlers,'save',return_value={}):
            with self.assertRaises(handlers.UnsafeCaptureError):handlers.training(self.m,self.out)
        self.assertEqual(build.call_count,2)
        self.assertIs(fake.IsolatedTrainingMemoryProtocol.call_args.kwargs['saved_activation_lifetime'],False)

    def test_unauthorized_matrix_pass_does_not_promote_p31(self):
        rows=self.old_matrix();row=rows['10.stability']
        row.update(status='PASS',attempted=True,criteria=dict(P31_uninterrupted_7200=True),supervisor=dict(cleanup=dict(active_processes=0)))
        self.assertEqual(next(f for f in reporting.matrix(self.m,rows) if f['id']=='P31')['decision'],'PARTIAL')
        next(e for e in self.m['experiments'] if e['id']=='10.stability')['authorization']['approved']=True
        self.assertEqual(next(f for f in reporting.matrix(self.m,rows) if f['id']=='P31')['decision'],'PASS')
