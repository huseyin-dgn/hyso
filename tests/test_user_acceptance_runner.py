"""Offline runner regressions only: never import Torch or launch a payload."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.user_acceptance import runner as user
from tools.user_acceptance.evidence import verify_feature
from tools.cuda_acceptance.feature_evidence import SafetyViolation


class RunnerTests(unittest.TestCase):
    def test_preparation_imports_no_torch(self):
        import subprocess
        code="import sys;sys.path.insert(0,"+repr(str(user.REPO))+");from tools.user_acceptance import runner;assert 'torch' not in sys.modules;assert 'tools.user_acceptance.scenarios' not in sys.modules"
        p=subprocess.run([sys.executable,'-I','-B','-c',code],capture_output=True,text=True,timeout=20)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_scope_exact_and_bounded(self):
        cpu=user.cpu_selection()
        self.assertEqual(len(cpu),14)
        self.assertEqual({n.rsplit('::',1)[-1] for n in cpu},{'test_U'+str(i).zfill(2) for i in range(1,15)})
        gpu=[n for rows in user.GROUPS.values() for n in rows]
        self.assertEqual(len(gpu),12)
        self.assertEqual(len(set(gpu)),12)
        self.assertNotIn('U10',gpu);self.assertNotIn('U11',gpu);self.assertNotIn('U14',gpu)

    def test_preserves_all_prior_cuda_attempts(self):
        root=user.REPO/'outputs/cuda-functional-acceptance-20261126'
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'failed-attempt.json';p.write_text('failed',encoding='utf-8')
            with patch.object(user,'CUDA_ROOT',Path(directory)),patch.object(user,'BASE_PRESERVED',return_value={}):
                values=user.preserved()
            self.assertEqual(values,{str(p):user.engine.file_hash(p)})

    def test_pending_pass_skipped_fail_requires_explicit_retry(self):
        rows={'U01':dict(status='PASS'),'U02':dict(status='FAIL'),'U03':dict(status='BLOCKED')}
        self.assertEqual(user.engine.pending_cases(list(rows),rows,set()),[])
        self.assertEqual(user.engine.pending_cases(list(rows),rows,{'U02'}),['U02'])

    def test_cpu_assertion_can_continue_but_safety_cannot(self):
        plan={'cpu_inventory':{'a':{}}};cp={'cpu_results':{'a':{'status':'FAIL','source_valid':True}}}
        receipt=dict(exit_code=1,monitor_reason=None,source_integrity=True,cleanup_verified=True,cleanup={'active_processes':0})
        self.assertTrue(user.normal_cpu_completion('CPU_REGRESSIONS',receipt,cp,plan))
        for change in ({'monitor_reason':'TIMEOUT'},{'source_integrity':False},{'cleanup_verified':False},{'cleanup':{'active_processes':1}},{'exit_code':2}):
            self.assertFalse(user.normal_cpu_completion('CPU_REGRESSIONS',dict(receipt,**change),cp,plan))
        self.assertFalse(user.normal_cpu_completion('02-inference',receipt,cp,plan))
        self.assertFalse(user.cpu_group_ready({'cpu_results':{}},plan))

    def test_original_cuda_defaults_remain_strict(self):
        self.assertFalse(user.engine.normal_cpu_completion('CPU_REGRESSIONS',{}, {}, {}))
        self.assertFalse(user.engine.cpu_group_ready({'results':{}},{}))

    def test_request_uses_only_owned_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);work=folder/'work'
            with patch.object(user,'BASE_REQUEST',side_effect=lambda *a,**k:{'features':['U01']}):
                a=user.request_for(folder,work,'02-inference',{})
                b=user.request_for(folder,work,'CPU_REGRESSIONS',{'feature_tools':{'ort':{'site':'prepared'}}}, {'node':{}})
            self.assertEqual(Path(a['argv'][0]).name,'gpu_worker.py')
            self.assertEqual(Path(b['argv'][0]).name,'cpu_worker.py')
            self.assertEqual(b['cpu_ort_manifest'],{'site':'prepared'})

    def test_device_label_cannot_pass_without_measurements(self):
        row=dict(scenario='U01',device='cuda:0',status='PASS',public_api='BenchmarkRunner',checks={'ok':True},observations={'device':'cuda:0'})
        with self.assertRaises((KeyError,ValueError)):verify_feature('U01',row,{})

    def test_valid_functional_forward_and_corrupt_measurement(self):
        d=dict(finite=True,output_device='cuda:0',parameters=4,before={'state':'x'},after={'state':'x'},benchmark=dict(
            timing={'latency_samples_ms':[[.1]*5]},environment={'device':'cuda:0'},cuda_memory={'status':'measured','peak_allocated_bytes':64}))
        row=dict(scenario='U01',device='cuda:0',status='PASS',public_api='BenchmarkRunner',checks={'ok':True},observations=d)
        verify_feature('U01',row,{})
        for change in ({'after':{}},{'finite':False},{'output_device':'cpu'}):
            bad=copy.deepcopy(row);bad['observations'].update(change)
            with self.assertRaises(ValueError):verify_feature('U01',bad,{})
        bad=copy.deepcopy(row);bad['observations']['benchmark']['timing']['latency_samples_ms']=[[float('nan')]*5]
        with self.assertRaises(ValueError):verify_feature('U01',bad,{})

    def test_nested_cleanup_and_oom_stop(self):
        for value in ({'cleanup':{'active_processes':1}},{'failure_kind':'oom'},{'monitor_reason':'BUDGET'}):
            with self.assertRaises(SafetyViolation):verify_feature('U01',value,{})

    def test_unknown_scenario_rejected(self):
        with self.assertRaises(ValueError):verify_feature('U99',{}, {})

    def test_cpu_safety_stop_prevents_reconciliation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/'user-safety-stop.json').write_text('{}')
            with patch.object(user,'BASE_RECORD_CPU') as record:
                with self.assertRaisesRegex(RuntimeError,'USER_OWNED_SAFETY_STOP'):user.record_cpu({},folder,{}, {}, {})
                record.assert_not_called()

    def test_resume_rejects_tampered_raw_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);raw=folder/'raw.json';raw.write_text('original')
            evidence=folder/'cpu-results.json'
            user.engine.atomic_json(evidence,{'scenario_artifacts':{'node':{str(raw):user.engine.file_hash(raw)}}})
            cp={'cpu_results':{'node':{'evidence':str(evidence),'source_valid':True}}}
            with patch.object(user,'BASE_VERIFY_CACHED'):
                user.verify_cached(cp)
                raw.write_text('tampered')
                with self.assertRaisesRegex(RuntimeError,'USER_RAW_EVIDENCE_CHANGED'):user.verify_cached(cp)

    def test_cpu_and_gpu_ownership_precedes_scenario_import(self):
        tree=ast.parse((user.REPO/'tools/user_acceptance/gpu_worker.py').read_text(encoding='utf-8'))
        top=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any(getattr(n,'module','')=='tools.user_acceptance.scenarios' for n in top))

    def test_product_frozen_no_dangerous_calls(self):
        tree=ast.parse((user.REPO/'tools/user_acceptance/scenarios.py').read_text(encoding='utf-8'))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(c.func,ast.Attribute) and c.func.attr in ('measure_stability','empty_cache','set_per_process_memory_fraction') for c in calls))
        self.assertFalse(any(k.arg=='saved_activation_lifetime' for c in calls for k in c.keywords))

    def test_resume_retries_u13_and_starts_pending_gpu_only(self):
        failed='tests/test_user_acceptance_scenarios.py::test_U13'
        rows={'CPU-PASS':dict(status='PASS'),failed:dict(status='FAIL')}
        self.assertEqual(user.pending_cases(['CPU-PASS',failed],rows,{failed}),[failed])
        self.assertEqual(user.pending_cases(['ADMISSION','U01'],{}, {failed}),['ADMISSION','U01'])
        self.assertEqual(user.pending_cases([failed],rows,set()),[])

    def test_cleanup_requires_matching_campaign_marker(self):
        from campaign_operations import cleanup_owned
        with tempfile.TemporaryDirectory() as directory:
            scratch=Path(directory);work=scratch/'payload';work.mkdir()
            marker=work/'.hyso-owned-work.json'
            user.engine.atomic_json(marker,dict(campaign_id='final-user-acceptance',attempt_id='regression'))
            with self.assertRaisesRegex(ValueError,'owned marker'):cleanup_owned(work,scratch,'cuda-functional')
            self.assertTrue(marker.is_file())
            cleanup_owned(work,scratch,'final-user-acceptance')
            self.assertFalse(work.exists())

    def test_cleanup_missing_marker_still_refused(self):
        from campaign_operations import cleanup_owned
        with tempfile.TemporaryDirectory() as directory:
            scratch=Path(directory);work=scratch/'payload';work.mkdir()
            with self.assertRaisesRegex(ValueError,'owned marker'):cleanup_owned(work,scratch,'final-user-acceptance')
            self.assertTrue(work.is_dir())

    def test_shared_cleanup_uses_sealed_request_scope(self):
        tree=ast.parse((user.REPO/'tools/cuda_acceptance/runner.py').read_text(encoding='utf-8'))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='cleanup_owned']
        self.assertEqual(len(calls),1)
        self.assertEqual(ast.unparse(calls[0].args[2]),"request['campaign_id']")

    def test_cpu_ort_execution_requires_real_events_runtime_and_trace(self):
        from tools.user_acceptance.cpu_onnx import verify_cpu_execution,canonical_trace_hash
        from tools.cuda_acceptance.native_identity import canonical_path,DLLIntegrityError
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);trace=[{'name':'CPU node Ω','dur':2}]
            extension=root/'onnxruntime_pybind11_state.pyd';extension.write_bytes(b'offline fixture')
            expected=dict(site=str(root),version='test',allowed_binaries={canonical_path(extension):user.engine.file_hash(extension)})
            identity=dict(pid=42,admitted_pid='42',visible_uuid='-1',source=str(root/'onnxruntime/__init__.py'),
                version='test',loaded_binaries=expected['allowed_binaries'])
            result=dict(analysis=dict(status='expected_execution_observed',events=[{'provider':'CPUExecutionProvider'}],
                fallback_detected=False,uncovered_source_nodes=[],evidence_assessment={'node_coverage':dict(status='complete',
                    per_run_association='complete',events_per_source_node={'node':4},per_run_source_event_counts=[{'node':1} for _ in range(4)])}),
                evidence=dict(controller=dict(exit_code=0,reason=None,cleanup={'active_processes':0},kill_on_close=True,launcher_pid=42),
                    report=dict(completed_runs=4,session_providers=['CPUExecutionProvider'],trace=trace,
                        trace_sha256=canonical_trace_hash(trace),runtime={'version':'test'},capture_context={'input_unchanged':True,'run_intervals':[1]*4})))
            verify_cpu_execution(result,identity,expected)
            for changes in ({'events':[]},{'status':'unavailable'},{'fallback_detected':True},{'uncovered_source_nodes':['node']}):
                bad=copy.deepcopy(result);bad['analysis'].update(changes)
                with self.assertRaises(ValueError):verify_cpu_execution(bad,identity,expected)
            for changes in ({'version':'wrong'},{'pid':43},{'loaded_binaries':{}},{'visible_uuid':'GPU-fake'}):
                with self.assertRaises(DLLIntegrityError):verify_cpu_execution(result,dict(identity,**changes),expected)
            trace[0]['dur']=3
            with self.assertRaises(ValueError):verify_cpu_execution(result,identity,expected)

    def test_cpu_ort_uses_prepared_runtime_and_rejects_changed_binary(self):
        from tools.user_acceptance.cpu_onnx import runtime_options
        from tools.cuda_acceptance.native_identity import DLLIntegrityError
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'onnxruntime').mkdir();(root/'onnxruntime/__init__.py').write_text('')
            binary=root/'runtime.pyd';binary.write_bytes(b'original')
            expected=dict(site=str(root),version='test',allowed_binaries={str(binary):user.engine.file_hash(binary)})
            options,_=runtime_options({'cpu_ort_manifest':expected})
            self.assertEqual(options,dict(runtime_search_path=str(root)))
            binary.write_bytes(b'changed')
            with self.assertRaises(DLLIntegrityError):runtime_options({'cpu_ort_manifest':expected})

    def test_git_tracking_change_requires_identical_sealed_bytes(self):
        from tools.user_acceptance.recovery_review import tracking_only_changes
        old={'user_files':{'known.py':'hash'},'source_sha256':{'known.py':'hash'}}
        new={'user_files':{},'source_sha256':{'known.py':'hash'}}
        self.assertEqual(tracking_only_changes(old,new,{'known.py'},{'known.py'}),{'known.py'})
        self.assertEqual(tracking_only_changes(old,new,{'known.py'},set()),set())
        new['source_sha256']['known.py']='changed'
        self.assertEqual(tracking_only_changes(old,new,{'known.py'},{'known.py'}),set())


if __name__=='__main__':unittest.main()
