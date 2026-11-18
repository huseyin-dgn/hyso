"""Completed evidence binding and safe containment; no native execution."""
import copy
from unittest.mock import patch
from test_infrastructure import Fixture
from full_acceptance import core, continuation, fresh


class Continuation(Fixture):
    def parent(self):
        self.m.update(assets=dict(models={'A':dict(sha256='a'*64)}),protocols={},model_run=dict(arm='A'))
        stage=self.out/'stages/03.cpu';stage.mkdir(parents=True,exist_ok=True)
        sup=dict(manifest_sha256=self.hash,native_crash=False,reason=None,error=None,
            cleanup=dict(active_processes=0),exit_code=0)
        terminal=dict(id='03.cpu',manifest_sha256=self.hash,status='PASS',criteria={'real':True},metrics=[],evidence=[])
        core.publish(stage/'supervisor-receipt.json',sup);core.publish(stage/'result.json',terminal)
        row=dict(terminal,attempted=True,supervisor=sup)
        state=dict(controller_active=False,preservation_status='PASS',finalization_errors=[],run_id=self.m['run_id'],manifest_sha256=self.hash)
        core.publish(self.out/'run-status.json',state)
        core.publish(self.out/'stage-results.json',dict(run_id=self.m['run_id'],manifest_sha256=self.hash,experiments=[row]))
        child=copy.deepcopy(self.m)
        child['continuation']=dict(parent_manifest=core.record(self.path),parent_terminal=core.record(self.out/'run-status.json'),
            parent_stage_results=core.record(self.out/'stage-results.json'),completed_ids=['03.cpu'])
        return child,row

    def test_completed_stage_is_carried_without_running_any_work(self):
        child,row=self.parent()
        with patch.object(continuation,'verify',return_value=self.m):rows=continuation.completed_rows(child)
        self.assertEqual(rows['03.cpu']['status'],'PASS')
        self.assertTrue(rows['03.cpu']['source_execution']['carried_without_execution'])

    def test_changed_parent_report_is_rejected(self):
        child,_=self.parent();core.publish(self.out/'stage-results.json',{})
        with patch.object(continuation,'verify',return_value=self.m):
            with self.assertRaisesRegex(ValueError,'artifact changed'):continuation.completed_rows(child)

    def test_different_assets_or_completed_set_is_rejected(self):
        child,_=self.parent();child['assets']={}
        with patch.object(continuation,'verify',return_value=self.m):
            with self.assertRaisesRegex(ValueError,'mismatch'):continuation.completed_rows(child)
        child,_=self.parent();child['continuation']['completed_ids']=[]
        with patch.object(continuation,'verify',return_value=self.m):
            with self.assertRaisesRegex(ValueError,'set changed'):continuation.completed_rows(child)

    def test_native_crash_cannot_be_continued(self):
        child,row=self.parent();row['supervisor']['native_crash']=True
        core.publish(self.out/'stages/03.cpu/supervisor-receipt.json',row['supervisor'])
        core.publish(self.out/'stage-results.json',dict(run_id=self.m['run_id'],manifest_sha256=self.hash,experiments=[row]))
        child['continuation']['parent_stage_results']=core.record(self.out/'stage-results.json')
        with patch.object(continuation,'verify',return_value=self.m):
            with self.assertRaisesRegex(ValueError,'unsafe'):continuation.completed_rows(child)

    def test_failed_stage_is_carried_as_failure(self):
        child,row=self.parent()
        row['status']='FAIL';row['supervisor']['exit_code']=1
        core.publish(self.out/'stages/03.cpu/result.json',{k:v for k,v in row.items() if k not in ('attempted','supervisor')})
        core.publish(self.out/'stages/03.cpu/supervisor-receipt.json',row['supervisor'])
        core.publish(self.out/'stage-results.json',dict(run_id=self.m['run_id'],manifest_sha256=self.hash,experiments=[row]))
        child['continuation']['parent_stage_results']=core.record(self.out/'stage-results.json')
        with patch.object(continuation,'verify',return_value=self.m):rows=continuation.completed_rows(child)
        self.assertEqual(rows['03.cpu']['status'],'FAIL')

    def test_passed_stage_cannot_be_selected_for_retry(self):
        child,_=self.parent();child['continuation']['retry_ids']=['03.cpu']
        with patch.object(continuation,'verify',return_value=self.m):
            with self.assertRaisesRegex(ValueError,'only explicit controlled P18'):continuation.completed_rows(child)

    def test_explicit_p18_retry_preserves_old_failure_and_excludes_only_that_stage(self):
        child,row=self.parent()
        stage=self.out/'stages/06.roofline';stage.mkdir(parents=True)
        next(e for e in self.m['experiments'] if e['id']=='06.roofline')['authorization']['approved']=True
        child['hardware']['gpu_uuid']=self.m['hardware']['gpu_uuid']='GPU-test'
        row.update(id='06.roofline',status='FAIL',safety_stop='nested_counter_capture_failed_exit_1')
        row['supervisor'].update(exit_code=1,resources=dict(low_memory_observed=False))
        core.publish(stage/'result.json',{k:v for k,v in row.items() if k not in ('attempted','supervisor')})
        core.publish(stage/'supervisor-receipt.json',row['supervisor'])
        core.publish(self.out/'stage-results.json',dict(run_id=self.m['run_id'],manifest_sha256=self.hash,experiments=[row]))
        denial=self.base/'denial.csv';denial.write_text('ERR_NVGPUCTRPERM')
        child['historical_counter_permission_block']=dict(gpu_uuid='GPU-test',error_artifact=core.record(denial))
        c=child['continuation'];c.update(parent_stage_results=core.record(self.out/'stage-results.json'),completed_ids=[],retry_ids=['06.roofline'])
        approval=self.base/'approval.json'
        core.publish(approval,dict(parent_manifest=c['parent_manifest'],stage='06.roofline',gpu_uuid='GPU-test',
            maximum_new_attempts=1,user_request='Explicit new controlled P18 attempt',automatic_retry=False))
        c['retry_authorization']=core.record(approval)
        with patch.object(continuation,'verify',return_value=self.m):self.assertEqual(continuation.completed_rows(child),{})
        self.assertEqual(core.read(stage/'result.json')['status'],'FAIL')
        core.publish(approval,{})
        with patch.object(continuation,'verify',return_value=self.m):
            with self.assertRaisesRegex(ValueError,'artifact changed'):continuation.completed_rows(child)

    def test_only_bound_counter_permission_error_blocks_this_device(self):
        p=self.base/'error.txt';p.write_text('ERR_NVGPUCTRPERM')
        self.m['hardware']['gpu_uuid']='GPU-test'
        self.m['known_counter_permission_block']=dict(gpu_uuid='GPU-test',error_artifact=core.record(p))
        self.assertIn('ERR_NVGPUCTRPERM',continuation.counter_block(self.m))
        self.m['known_counter_permission_block']['gpu_uuid']='GPU-other'
        with self.assertRaisesRegex(ValueError,'device mismatch'):continuation.counter_block(self.m)

    def test_calibration_failure_never_certifies_full_feature_acceptance(self):
        row=dict(status='FAIL',criteria=dict(real_profiler=True,ownership_estimate=False),
            supervisor=dict(native_crash=False,reason=None,exit_code=1))
        self.assertFalse(fresh.independent_layer_subset('P12','06.layer',row))
        self.assertFalse(fresh.independent_layer_subset('P14','06.layer',row))
        self.assertFalse(fresh.independent_layer_subset('P13','06.layer',row))
        row['safety_stop']='unknown'
        self.assertFalse(fresh.independent_layer_subset('P12','06.layer',row))
