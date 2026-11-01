"""Opt-in continuation contracts. Offline mocks only; NEVER launch native work."""
from unittest.mock import Mock, patch
import types

from test_infrastructure import Fixture
from full_acceptance import cli, core, reporting, safety


class IndependentContinuationTests(Fixture):
    def setUp(self):
        super().setUp()
        self.m['execution_policy']='continue_independent'
        self.seal()

    def exp(self, ident):
        return next(e for e in self.m['experiments'] if e['id']==ident)

    def test_advisory_failure_does_not_block_independent_gpu(self):
        self.m['hardware']['gpu_uuid']='GPU-local-double'
        rows=reporting.initial_results(self.m)
        rows['03.cpu']['status']='FAIL'
        self.exp('04.cuda')['authorization']['approved']=True
        status,reason,_=reporting.decision(self.m,self.exp('04.cuda'),rows)
        self.assertEqual(status,'READY')
        self.assertIsNone(reason)

    def test_stability_is_independent_of_cpu_benchmark_result(self):
        rows=reporting.initial_results(self.m)
        rows['03.cpu']['status']='FAIL'
        self.exp('10.stability')['authorization']['approved']=True
        self.assertEqual(reporting.decision(self.m,self.exp('10.stability'),rows)[0],'READY')

    def test_opt_in_runs_independent_diagnostics_before_p31(self):
        seq=[e['id'] for e in cli._execution_order(self.m)]
        self.assertLess(seq.index('03.cpu'),seq.index('10.stability'))
        self.assertLess(seq.index('04.cuda'),seq.index('10.stability'))
        self.assertLess(seq.index('06.layer'),seq.index('10.stability'))
        self.assertLess(seq.index('12.replay'),seq.index('10.stability'))
        self.assertEqual(seq[-2:],['10.stability','11.memory'])
        self.m['execution_policy']='fail_fast'
        self.assertLess([e['id'] for e in cli._execution_order(self.m)].index('04.cuda'),
                        [e['id'] for e in cli._execution_order(self.m)].index('10.stability'))

    def test_memory_analysis_requires_real_p31_pass(self):
        rows=reporting.initial_results(self.m)
        self.exp('11.memory')['authorization']['approved']=True
        for status in ('FAIL','PARTIAL','NOT_AUTHORIZED','BLOCKED_PREREQUISITE'):
            rows['10.stability']['status']=status
            self.assertEqual(reporting.decision(self.m,self.exp('11.memory'),rows)[0],
                             'BLOCKED_DEPENDENCY')
        rows['10.stability']['status']='PASS'
        self.assertEqual(reporting.decision(self.m,self.exp('11.memory'),rows)[0],'READY')

    def test_future_unspecified_edges_default_to_strict(self):
        rows=reporting.initial_results(self.m)
        e=self.exp('12.replay')
        e['depends']=['03.cpu']
        rows['03.cpu']['status']='FAIL'
        self.assertEqual(reporting.decision(self.m,e,rows)[0],'BLOCKED_DEPENDENCY')

    def test_disabled_mode_retains_legacy_dependence(self):
        self.m['execution_policy']='fail_fast'
        rows=reporting.initial_results(self.m)
        rows['02.regression']['status']='FAIL'
        self.assertEqual(reporting.decision(self.m,self.exp('03.cpu'),rows)[0],'BLOCKED_DEPENDENCY')

    def test_sealed_policy_rejects_unknown_value(self):
        self.m['execution_policy']='skip_everything'
        self.seal()
        with self.assertRaisesRegex(ValueError,'invalid sealed execution policy'):
            core.verify(self.path,expected=self.hash,assets=False)

    def test_only_clean_python_fail_can_continue(self):
        ordinary=dict(status='FAIL',supervisor=dict(native_crash=False,reason=None,
            exit_code=1,cleanup=dict(active_processes=0)))
        self.assertIsNone(cli._fatal_experiment_boundary(ordinary))
        for changes in (
            dict(native_crash=True),
            dict(reason='Windows_OS_low_memory'),
            dict(reason='heartbeat_watchdog'),
            dict(cleanup=dict(active_processes=1)),
            dict(cleanup=None),
        ):
            bad=dict(ordinary,supervisor=dict(ordinary['supervisor'],**changes))
            self.assertIsNotNone(cli._fatal_experiment_boundary(bad))
        self.assertIsNotNone(cli._fatal_experiment_boundary(dict(status='ABORTED')))
        self.assertIsNotNone(cli._fatal_experiment_boundary(dict(status='FAIL')))

    def exercise(self, fail_receipt):
        self.m['hardware']['gpu_uuid']='GPU-local-double'
        for e in self.m['experiments']:
            e['authorization']['approved']=e['id'] in ('02.regression','03.cpu','04.cuda','12.replay')
        self.seal()
        self.old_matrix()
        args=types.SimpleNamespace(manifest=self.path,approved_sha256=self.hash,cuda_subtasks=False)
        seen=[]
        def worker(manifest,experiment,**kwargs):
            seen.append(experiment['id'])
            if experiment['id']=='03.cpu':return dict(status='FAIL',reason='simulated failure',
                attempted=True,metrics=[],supervisor=fail_receipt)
            return dict(status='PASS',reason=None,attempted=True,metrics=[],
                supervisor=dict(native_crash=False,reason=None,cleanup=dict(active_processes=0)))
        with patch.object(cli,'verify',return_value=self.m), \
             patch.object(cli,'preserve'), \
             patch.object(cli,'write_reports'), \
             patch.object(safety,'supervise',side_effect=worker):
            code=cli.execute(args)
        return seen,code

    def test_controller_continues_normal_failure_without_fake_pass(self):
        seen,code=self.exercise(dict(native_crash=False,reason=None,exit_code=1,
            cleanup=dict(active_processes=0)))
        self.assertEqual(seen,['02.regression','03.cpu','04.cuda','12.replay'])
        self.assertEqual(code,1)  # one failed stage remains a failed overall exit

    def test_controller_stops_after_native_crash(self):
        seen,code=self.exercise(dict(native_crash=True,reason=None,exit_code=3221226505,
            cleanup=dict(active_processes=0)))
        self.assertEqual(seen,['02.regression','03.cpu'])
        self.assertEqual(code,1)
