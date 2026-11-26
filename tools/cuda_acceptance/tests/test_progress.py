"""Isolated stdlib regressions; no campaign, product tests, Torch, GPU or Job launch."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from tools.cuda_acceptance import runner,progress_recovery
from common import Journal,atomic_json,digest,file_hash,read_json


class LiveProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.folder=self.root/'attempt';self.folder.mkdir()
        self.cp=dict(results={},cpu_results={},attempts=[],history=[])
        self.plan=dict(cpu_inventory={n:{} for n in ('ok','failed','setup-failed','skip')})
        self.patch=patch.object(runner,'ROOT',self.root);self.patch.start();self.addCleanup(self.patch.stop)
        self.no_launch=patch.object(runner,'execute',side_effect=AssertionError('campaign launch forbidden'))
        self.no_launch.start();self.addCleanup(self.no_launch.stop)
        self.journal=Journal(self.folder/'events.jsonl',dict(attempt_id='a',group='CPU_REGRESSIONS',source_commit='c',environment_id='e'))
        self.addCleanup(self.journal.close)

    def emit_case(self,name,outcomes):
        self.journal.emit('test_start',nodeid=name)
        for stage,outcome in outcomes:
            self.journal.emit('phase',nodeid=name,when=stage,outcome=outcome,duration=.001)

    def progress(self):
        original=copy.deepcopy(self.cp)
        value=runner.progress(self.cp,self.plan,'RUNNING',folder=self.folder,group='CPU_REGRESSIONS')
        self.assertEqual(self.cp,original)
        self.assertEqual(read_json(self.root/'progress.json'),value)
        self.assertTrue(value['live_counts_provisional_until_cleanup'])
        return value

    def test_empty_journal(self):
        (self.folder/'events.jsonl').touch()
        value=self.progress()
        self.assertEqual(value['CPU'],dict(PASS=0,FAIL=0,BLOCKED=0))
        self.assertIsNone(value['active_feature']);self.assertEqual(value['journal_issues'],[])

    def test_partial_journal_and_incomplete_write(self):
        self.emit_case('ok',[('setup','passed'),('call','passed')])
        value=self.progress()
        self.assertEqual(value['CPU']['PASS'],0);self.assertEqual(value['active_feature'],'ok')
        with (self.folder/'events.jsonl').open('ab') as stream:stream.write(b'{"incomplete":')
        value=self.progress()
        self.assertEqual(value['CPU']['PASS'],0)
        self.assertEqual(value['journal_issues'][0]['reason'],'INTERRUPTED_LAST_WRITE')

    def test_completed_journal_counts_only_terminal_cases(self):
        self.emit_case('ok',[('setup','passed'),('call','passed'),('teardown','passed')])
        self.emit_case('failed',[('setup','passed'),('call','failed'),('teardown','passed')])
        self.emit_case('setup-failed',[('setup','failed'),('teardown','passed')])
        self.emit_case('skip',[('setup','skipped'),('teardown','passed')])
        value=self.progress()
        self.assertEqual(value['CPU'],dict(PASS=1,FAIL=2,BLOCKED=1))
        self.assertIsNone(value['active_feature']);self.assertEqual(value['journal_issues'],[])

    def test_corrupt_journal_cannot_show_pass(self):
        self.emit_case('ok',[('setup','passed'),('call','passed'),('teardown','passed')])
        path=self.folder/'events.jsonl';path.write_bytes(path.read_bytes().replace(b'"ok"',b'"changed"'))
        value=self.progress()
        self.assertEqual(value['CPU']['PASS'],0)
        self.assertEqual(value['journal_issues'][0]['reason'],'EVIDENCE_INVALID')

    def test_controller_error_precedes_secondary_runtime_error(self):
        receipt=dict(monitor_reason='CONTROLLER_ERROR',error=dict(type='AttributeError',message="'tuple' object has no attribute 'items'"))
        with patch.object(runner,'reconcile_attempt',side_effect=AssertionError('secondary reconciliation must not mask controller failure')):
            with self.assertRaisesRegex(RuntimeError,'CONTROLLER_ERROR: AttributeError') as error:
                runner.record_cpu(self.cp,self.folder,{},receipt,self.plan)
        self.assertNotIn('CPU_RUNTIME_UNVERIFIED',str(error.exception))
        self.assertEqual(self.cp['results'],{})


class ProgressRecoveryTests(unittest.TestCase):
    """Synthetic receipt/auditor fixtures test recovery decisions, never GPU proof."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.folder=self.root/'attempts/a';self.folder.mkdir(parents=True)
        self.old=dict(cpu_inventory={'case':{'path':'case.py'}},groups={'gpu':['ADMISSION']},runtime={'pinned':True},budgets={'bounded':True},
            source_sha256={'hyso/product.py':'unchanged','tools/cuda_acceptance/runner.py':'old'},preserved_evidence={},commit='c')
        self.new=copy.deepcopy(self.old);self.new['source_sha256']['tools/cuda_acceptance/runner.py']='fixed'
        self.cp=dict(plan_sha256=digest(self.old),results={},cpu_results={},history=[],attempts=[dict(
            directory=str(self.folder),group='CPU_REGRESSIONS',status='RUNNING',plan_sha256=digest(self.old))])
        self.receipt=dict(schema_version=3,attempt_id='a',monitor_reason='CONTROLLER_ERROR',
            error=dict(type='AttributeError',message="'tuple' object has no attribute 'items'",traceback='runner.py progress journal_results items'),
            source_integrity=True,cleanup_verified=True,cleanup=dict(active_processes=0),cleanup_error=None,
            assigned_before_imports=True,limit_readback_verified=True,launcher_wait_verified=True,
            launch_binding=dict(created_suspended=True,exact_job_membership_before_resume=True,no_breakaway=True,limits_readback=True))
        atomic_json(self.folder/'controller.json',self.receipt)
        atomic_json(self.folder/'admission-receipt.json',dict(no_breakaway=True,kill_on_close=True,limits_readback=True))
        self.audit=dict(global_errors=['CPU_RUNTIME_UNVERIFIED'],results={},stream=dict(scope='VALID'),events=[dict(kind='worker_start')],
            request=dict(attempt_id='a',group='CPU_REGRESSIONS',profile='cpu',mode='tests',inventory=self.old['cpu_inventory'],selected_nodes=['case']))

    def recover(self):
        with patch.object(progress_recovery,'reconcile_attempt',return_value=self.audit):
            progress_recovery.recover_progress_failure(self.cp,self.old,self.new,self.root)

    def test_verified_pretest_failure_keeps_evidence_and_cases_pending(self):
        original={p.name:file_hash(p) for p in self.folder.iterdir()}
        self.recover()
        self.assertEqual(original,{p.name:file_hash(p) for p in self.folder.iterdir()})
        self.assertEqual(self.cp['attempts'][0]['status'],'RECOVERED_CONTROLLER_ERROR')
        self.assertEqual(self.cp['results'],{});self.assertEqual(self.cp['cpu_results'],{})
        self.assertEqual(runner.pending_cases(['case'],self.cp['cpu_results'],set()),['case'])
        self.assertEqual(self.cp['last_controller_error']['secondary_evidence_errors'],['CPU_RUNTIME_UNVERIFIED'])
        self.assertEqual(self.cp['history'][0]['accepted_results'],0)

    def test_unverified_cleanup_refuses_recovery(self):
        for update in (dict(cleanup_verified=False),dict(cleanup=dict(active_processes=1)),dict(limit_readback_verified=False)):
            with self.subTest(update=update):
                receipt=dict(self.receipt,**update);atomic_json(self.folder/'controller.json',receipt)
                with self.assertRaisesRegex(RuntimeError,'RECOVERY_REFUSED'):self.recover()
                self.assertEqual(self.cp['attempts'][0]['status'],'RUNNING')

    def test_test_start_or_other_integrity_error_refuses_recovery(self):
        for events,errors in (([dict(kind='test_start')],['CPU_RUNTIME_UNVERIFIED']),
                              ([dict(kind='worker_start')],['CPU_RUNTIME_UNVERIFIED','REQUEST_HASH_MISMATCH'])):
            with self.subTest(errors=errors,events=events):
                self.audit.update(events=events,global_errors=errors)
                with self.assertRaisesRegex(RuntimeError,'RECOVERY_REFUSED'):self.recover()
                self.assertEqual(self.cp['attempts'][0]['status'],'RUNNING')

    def test_scope_or_product_source_change_refuses_recovery(self):
        for key in ('cpu_inventory','source_sha256'):
            with self.subTest(key=key):
                self.new=copy.deepcopy(self.old)
                self.new[key]['unexpected']='changed'
                with self.assertRaisesRegex(RuntimeError,'RECOVERY_REFUSED'):self.recover()
                self.assertEqual(self.cp['attempts'][0]['status'],'RUNNING')


if __name__=='__main__':
    unittest.main()
