"""Acceptance-tool fault injection. No GPU, camera, training or long run."""
from pathlib import Path
import ast,copy,json,os,sys,tempfile,types,unittest,uuid,xml.etree.ElementTree as ET,zipfile
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from full_acceptance import core,registry,reporting,handlers,safety,cli

class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='hyso-acceptance-unit-')
        self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name).resolve();self.out=self.base/'out';self.out.mkdir()
        self.repo=self.base/'repo';self.snapshot=self.out/'source';self.repo.mkdir();(self.snapshot/'tools').mkdir(parents=True)
        for base in (self.repo,self.snapshot):
            (base/'tools').mkdir(exist_ok=True);(base/'tools/run_full_acceptance.py').write_text('pass\n')
        sources={'tools/run_full_acceptance.py':core.sha(self.snapshot/'tools/run_full_acceptance.py')}
        self.m=dict(format='hyso.full_acceptance.manifest',schema_version=1,run_id=str(uuid.uuid4()),output=str(self.out),repository=str(self.repo),
            source_snapshot=dict(root=str(self.snapshot),files=sources,tree_sha256=core.digest(sources)),
            experiments=registry.experiments(),optional={},hardware=dict(gpu_uuid=None,ort_cpu=None,ort_cuda=None),
            runtime=dict(python=dict(path=sys.executable),site_paths=[]),assets={},approval_history=[],forensic=dict(B4='OPEN',CPU65536='OPEN',sections=[]),prior_audit={},
            supervision=dict(startup_seconds=120,heartbeat_seconds=120,poll_seconds=.01,cleanup_seconds=.1,max_log_bytes=100000,max_stage_evidence_bytes=100000))
        for e in self.m['experiments']:
            e.update(output=str(self.out/'stages'/e['id']),outer_seconds=300,device='cpu',model='B',expected_work='fixed test work',deadline_basis='unit test only',stop_rules=[],cleanup='owned only')
            e['command']=[sys.executable,'-I','-S','-B',str(self.snapshot/'tools/run_full_acceptance.py'),'--worker',e['id'],
                '--manifest',str(self.out/'manifest.json'),'--stage-output',e['output']]
        self.path=self.out/'manifest.json';self.seal()
    def seal(self):
        core.publish(self.path,self.m);self.hash=core.sha(self.path)
        self.path.with_suffix('.sha256').write_text(self.hash)
    def old_matrix(self):
        rows=[]
        for i in range(1,36):
            ident=f'P{i:02}';rows.append(dict(id=ident,name='feature '+ident,decision='PARTIAL' if ident in ('P13','P18','P23','P30','P31','P33','P34') else 'PASS',
                decision_scope='historical explicit scope',decision_source=dict(path='historical.md',sha256='a'*64),unsupported_metrics='unmeasured domains'))
        p=self.base/'old-matrix.json';core.publish(p,dict(features=rows))
        self.m['prior_audit']['P01-P35-acceptance-matrix.json']=core.record(p)
        return reporting.initial_results(self.m)

class ManifestTests(Fixture):
    def test_manifest_roundtrip_and_exact_hash(self):self.assertEqual(core.verify(self.path,expected=self.hash,assets=False)['run_id'],self.m['run_id'])
    def test_wrong_manifest_hash_never_admits(self):
        with self.assertRaisesRegex(ValueError,'SHA256'):core.verify(self.path,expected='0'*64,assets=False)
    def test_live_source_drift_rejected(self):
        (self.repo/'tools/run_full_acceptance.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'live source drift'):core.verify(self.path,expected=self.hash,assets=False)
    def test_snapshot_drift_rejected_even_if_live_unchanged(self):
        (self.snapshot/'tools/run_full_acceptance.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'snapshot source changed'):core.verify(self.path,expected=self.hash,assets=False)
    def test_unpinned_snapshot_module_rejected(self):
        (self.snapshot/'injected.py').write_text('pass')
        with self.assertRaisesRegex(ValueError,'unpinned'):core.verify(self.path,expected=self.hash,assets=False)
    def test_new_live_module_requires_new_manifest(self):
        (self.repo/'tools/new_helper.py').write_text('pass')
        with self.assertRaisesRegex(ValueError,'file set changed'):core.verify(self.path,expected=self.hash,assets=False)
    def test_source_traversal_rejected(self):
        outside=self.base/'outside.py';outside.write_text('pass')
        files={'../../outside.py':core.sha(outside)};self.m['source_snapshot'].update(files=files,tree_sha256=core.digest(files));self.seal()
        with self.assertRaises(ValueError):core.verify(self.path,expected=self.hash,assets=False)
    def test_duplicate_json_keys_rejected(self):
        p=self.base/'duplicate.json';p.write_text('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError,'duplicate'):core.read(p)
    def test_nonfinite_json_rejected(self):
        for literal in ('NaN','Infinity','-Infinity'):
            p=self.base/'number.json';p.write_text('{"value":'+literal+'}')
            with self.assertRaises(ValueError):core.read(p)
    def test_boolean_authorization_not_integer(self):
        self.m['experiments'][0]['authorization']['approved']=1;self.seal()
        with self.assertRaisesRegex(ValueError,'boolean'):core.verify(self.path,expected=self.hash,assets=False)
    def test_handler_swap_cannot_inherit_safe_approval(self):
        self.m['experiments'][2]['handler']='roofline';self.seal()
        with self.assertRaisesRegex(ValueError,'routing'):core.verify(self.path,expected=self.hash,assets=False)
    def test_exact_command_cannot_be_arbitrary(self):
        self.m['experiments'][0]['command'][-1]='arbitrary';self.seal()
        with self.assertRaisesRegex(ValueError,'command binding'):core.verify(self.path,expected=self.hash,assets=False)
    def test_output_escape_rejected(self):
        self.m['experiments'][0]['output']=str(self.base/'outside');self.seal()
        with self.assertRaisesRegex(ValueError,'output escape'):core.verify(self.path,expected=self.hash,assets=False)
    def test_retry_flag_rejected(self):
        self.m['experiments'][0]['automatic_retry']=True;self.seal()
        with self.assertRaisesRegex(ValueError,'retry'):core.verify(self.path,expected=self.hash,assets=False)
    def test_authorize_creates_new_manifest_only(self):
        target=self.base/'authorized'
        # Fixture has no actual runtime/assets; exercise cloning, not native verification.
        with patch.object(core,'verify',return_value=copy.deepcopy(self.m)):
            child=core.authorize(self.path,target,['10.stability'],expected=self.hash,basis='explicit controlled CPU7200 approval')
        self.assertNotEqual(child['run_id'],self.m['run_id']);self.assertFalse((target/'STARTED.json').exists())
        self.assertTrue(next(x for x in child['experiments'] if x['id']=='10.stability')['authorization']['approved'])
        self.assertFalse(next(x for x in child['experiments'] if x['id']=='04.cuda')['authorization']['approved'])
        self.assertEqual(core.sha(self.path),self.hash)
    def test_child_does_not_inherit_previous_GPU_approval(self):
        next(e for e in self.m['experiments'] if e['id']=='04.cuda')['authorization']['approved']=True
        with patch.object(core,'verify',return_value=copy.deepcopy(self.m)):
            child=core.authorize(self.path,self.base/'new-child',['10.stability'],expected=self.hash,basis='only CPU long run')
        self.assertFalse(next(e for e in child['experiments'] if e['id']=='04.cuda')['authorization']['approved'])
    def test_unknown_forensic_reproducer_cannot_be_authorized(self):
        with patch.object(core,'verify',return_value=self.m),self.assertRaisesRegex(ValueError,'unknown'):
            core.authorize(self.path,self.base/'other',['CPU65536.reproducer'],expected=self.hash,basis='do not execute')

class DecisionTests(Fixture):
    def test_all_twelve_stages_and_stable_ids_present(self):
        self.assertEqual([x[0] for x in registry.STAGES],[f'{i:02}' for i in range(1,13)])
        self.assertEqual(set(registry.FEATURE_EXPERIMENTS),{f'P{i:02}' for i in range(1,36)})
    def test_native_stages_default_not_authorized(self):
        for ident in ('04.cuda','05.training','06.layer','06.roofline','07.energy','07.devices','08.camera','09.provider_cuda','10.stability'):
            e=next(x for x in self.m['experiments'] if x['id']==ident)
            self.assertEqual(reporting.decision(self.m,e,reporting.initial_results(self.m))[0],'NOT_AUTHORIZED')
    def test_missing_camera_not_replaced_by_MNIST(self):
        e=next(x for x in self.m['experiments'] if x['id']=='08.camera');e['authorization']['approved']=True
        status,reason,_=reporting.decision(self.m,e,reporting.initial_results(self.m))
        self.assertEqual(status,'BLOCKED_PREREQUISITE');self.assertIn('MNIST',reason)
    def test_GPU_approval_without_UUID_is_blocked(self):
        e=next(x for x in self.m['experiments'] if x['id']=='04.cuda');e['authorization']['approved']=True
        self.assertEqual(reporting.decision(self.m,e,reporting.initial_results(self.m))[0],'BLOCKED_PREREQUISITE')
    def test_failure_or_stop_blocks_successor_without_retry(self):
        e=next(x for x in self.m['experiments'] if x['id']=='03.cpu');rows=reporting.initial_results(self.m)
        rows['02.regression']['status']='FAIL'
        self.assertEqual(reporting.decision(self.m,e,rows)[0],'BLOCKED_DEPENDENCY')
        self.assertEqual(reporting.decision(self.m,e,rows,stopped=True)[0],'NOT_RUN_STOPPED')
    def test_approval_not_inherited_from_CPU(self):
        rows=reporting.initial_results(self.m);rows['03.cpu'].update(status='PASS',attempted=True)
        e=next(x for x in self.m['experiments'] if x['id']=='04.cuda')
        self.assertEqual(reporting.decision(self.m,e,rows)[0],'NOT_AUTHORIZED')
    def test_long_run_work_not_relaxed(self):
        source=(core.ROOT/'tools/full_acceptance/core.py').read_text(encoding='utf-8')
        self.assertIn('duration_seconds=7200.',source);self.assertIn('interval_seconds=5.',source)
        self.assertNotIn('set_per_process_memory_fraction(', (core.ROOT/'tools/full_acceptance/counter_worker.py').read_text())

class ReportingTests(Fixture):
    def test_preflight_cannot_overwrite_started_run(self):
        core.publish(self.out/'STARTED.json',dict(one_shot=True))
        with patch.object(cli,'verify') as verify,self.assertRaises(FileExistsError):
            cli.preflight(types.SimpleNamespace(manifest=self.path,approved_sha256=self.hash))
        verify.assert_not_called()
    def test_null_requires_reason_and_kind(self):
        with self.assertRaises(ValueError):handlers.metric('energy',None,'J','derived','CPU')
        m=handlers.metric('energy',None,'J','derived','CPU','no sensor')
        self.assertEqual(m['kind'],'missing');self.assertIsNone(m['value'])
        reporting.validate_metrics(dict(metrics=[m]))
        with self.assertRaises(ValueError):reporting.validate_metrics(dict(metrics=[dict(m,reason=None)]))
    def test_prepared_report_cannot_be_complete(self):
        rows=self.old_matrix();self.assertFalse(reporting.write_reports(self.m,rows))
        self.assertFalse(core.read(self.out/'stage-results.json')['suite_complete'])
        self.assertEqual(len(core.read(self.out/'P01-P35-final-acceptance-matrix.json')['features']),35)
        self.assertEqual(core.read(self.out/'energy-evidence.json')['status'],'MISSING')
        tree=ET.parse(self.out/'test-results.xml');types=[x.get('type') for x in tree.iter('skipped')]
        self.assertIn('NOT_AUTHORIZED',types)
    def test_current_unit_XML_is_included_without_old_run_inflation(self):
        rows=self.old_matrix()
        ET.ElementTree(ET.Element('testsuite',name='old-unit',tests='1')).write(self.out/'infrastructure-tests.xml')
        ET.ElementTree(ET.Element('testsuite',name='accepted-unit',tests='2')).write(self.out/'infrastructure-tests-accepted.xml')
        reporting.write_reports(self.m,rows)
        names=[s.get('name') for s in ET.parse(self.out/'test-results.xml').getroot().findall('testsuite')]
        self.assertIn('accepted-unit',names);self.assertNotIn('old-unit',names)
    def test_forged_unattempted_PASS_never_promotes_P31(self):
        rows=self.old_matrix();rows['10.stability'].update(status='PASS',criteria=dict(P31_uninterrupted_7200=True),supervisor=dict(cleanup=dict(active_processes=0)))
        f=next(x for x in reporting.matrix(self.m,rows) if x['id']=='P31');self.assertEqual(f['decision'],'PARTIAL')
    def test_failed_partial_prefix_not_long_PASS(self):
        rows=self.old_matrix();rows['10.stability'].update(status='FAIL',attempted=True,reason='short prefix',criteria=dict(P31_uninterrupted_7200=False))
        f=next(x for x in reporting.matrix(self.m,rows) if x['id']=='P31');self.assertEqual(f['decision'],'FAIL')
    def test_full_count_requires_actual_attempts_and_authorization(self):
        rows=self.old_matrix()
        for r in rows.values():r.update(status='PASS',attempted=True)
        self.assertFalse(reporting.write_reports(self.m,rows))
    def test_P13_CPU_does_not_promote_full_CUDA_scope(self):
        rows=self.old_matrix();rows['06.layer'].update(status='PASS',attempted=True,criteria=dict(ownership_estimate=True))
        self.assertEqual(next(f for f in reporting.matrix(self.m,rows) if f['id']=='P13')['decision'],'PARTIAL')
    def test_package_is_exclusive_and_not_in_evidence_tree(self):
        rows=self.old_matrix();reporting.write_reports(self.m,rows);core.publish(self.out/'run-status.json',dict(controller_active=False,suite_complete=False))
        target=self.base/'evidence.zip';args=types.SimpleNamespace(manifest=self.path,package=target)
        self.assertEqual(cli.package(args),0)
        with zipfile.ZipFile(target) as z:self.assertIn('package-inventory.json',z.namelist())
        with self.assertRaises(ValueError):cli.package(args)
        args.package=self.out/'bad.zip'
        with self.assertRaises(ValueError):cli.package(args)
    def test_active_run_cannot_be_packaged(self):
        core.publish(self.out/'run-status.json',dict(controller_active=True,suite_complete=False))
        with self.assertRaises(RuntimeError):cli.package(types.SimpleNamespace(manifest=self.path,package=self.base/'bad.zip'))

class APIBindingTests(unittest.TestCase):
    def test_public_keyword_bindings_match_current_source(self):
        # Static contract check catches invalid adapter keywords without launching
        # CUDA/ORT/camera/native producers during preparation.
        targets={'BenchmarkRunner':('hyso/benchmark/runner.py','BenchmarkRunner'),
            'StreamProtocol':('hyso/performance/streaming.py','StreamProtocol'),
            'DeviceProtocol':('hyso/performance/devices.py','DeviceProtocol'),
            'StabilityProtocol':('hyso/performance/stability.py','StabilityProtocol'),
            'ProviderProtocol':('hyso/performance/providers.py','ProviderProtocol'),
            'IsolatedTrainingMemoryProtocol':('hyso/performance/isolated_training_memory.py','IsolatedTrainingMemoryProtocol'),
            'ProfilingProtocol':('hyso/performance/profiling.py','ProfilingProtocol'),
            'evaluate_classification':('hyso/evaluation.py','evaluate_classification'),
            'measure_footprint':('hyso/performance/memory.py','measure_footprint')}
        allowed={}
        for name,(rel,ident) in targets.items():
            tree=ast.parse((core.ROOT/rel).read_text(encoding='utf-8-sig'));node=next(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==ident)
            if isinstance(node,ast.ClassDef):
                init=next((x for x in node.body if isinstance(x,ast.FunctionDef) and x.name=='__init__'),None)
                allowed[name]={x.target.id for x in node.body if isinstance(x,ast.AnnAssign) and isinstance(x.target,ast.Name)} if init is None else {a.arg for a in [*init.args.args,*init.args.kwonlyargs]}
            else:allowed[name]={a.arg for a in [*node.args.args,*node.args.kwonlyargs]}
        tree=ast.parse((core.ROOT/'tools/full_acceptance/handlers.py').read_text(encoding='utf-8'))
        for call in ast.walk(tree):
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id in allowed:
                self.assertLessEqual({k.arg for k in call.keywords if k.arg is not None},allowed[call.func.id],call.func.id)

class OwnershipTests(Fixture):
    def scenario(self,*,assign_error=False,exit_code=0,monitor=False,replace_during_scan=False):
        self.process=None;self.job=None
        self.raced=False
        outer=self
        class Process:
            pid=101;_handle=100
            def __init__(self,*a,**k):
                self.returncode=None if assign_error else exit_code;self.terminated=False;self.polls=0;outer.process=self
                path=Path(outer.m['experiments'][0]['output']);core.publish(path/'result.json',dict(id='01.preflight',manifest_sha256=core.sha(outer.path),status='PASS',metrics=[],criteria={}))
                if replace_during_scan:core.publish(path/'heartbeat.json.race.tmp',dict(unix_time=0),new=True)
            def poll(self):
                self.polls+=1
                return None if monitor and self.polls==1 else self.returncode
            def terminate(self):self.terminated=True;self.returncode=1
            def wait(self,timeout=None):return self.returncode
        class Job:
            def __init__(self,*args):
                self.args=args;self.terminated=False;self.closed=False;outer.job=self
            def assign(self,process):
                if assign_error:raise OSError('injected assign failure')
            def terminate(self):self.terminated=True
            def observation(self):return dict(active_processes=0)
            def close(self):self.closed=True
        class Resources:
            def __init__(self,*a):self.summary=dict(low_memory_observed=False)
            def sample(self,*a):return False
            def close(self):pass
        def load(path,name):return types.SimpleNamespace(_Job=Job) if 'job' in name else types.SimpleNamespace(_StreamResources=Resources)
        original_rglob=Path.rglob
        def racing_listing(root,pattern):
            entries=list(original_rglob(root,pattern))
            for path in entries:
                if replace_during_scan and path.name=='heartbeat.json.race.tmp':
                    os.replace(path,path.with_name('heartbeat.json'));outer.raced=True
                yield path
        with patch.object(safety,'load_file',side_effect=load),patch.object(safety,'duplicate_query',return_value=1),patch.object(safety.subprocess,'Popen',Process),patch.object(Path,'rglob',racing_listing):
            value=safety.supervise(self.m,self.m['experiments'][0])
        return value
    def test_success_owns_job_and_does_not_set_quotas(self):
        r=self.scenario();self.assertEqual(r['status'],'PASS');self.assertEqual(self.job.args,(None,None,None))
        self.assertTrue(self.job.terminated and self.job.closed);self.assertEqual(r['supervisor']['cleanup']['active_processes'],0)
    def test_assignment_failure_reaps_only_new_waiting_handle(self):
        r=self.scenario(assign_error=True);self.assertEqual(r['status'],'FAIL');self.assertTrue(self.process.terminated)
        self.assertTrue(self.job.closed)
    def test_native_exit_cannot_become_success_from_stale_result(self):
        r=self.scenario(exit_code=0xC0000409);self.assertEqual(r['status'],'FAIL');self.assertTrue(r['supervisor']['native_crash'])
        self.assertFalse(r['supervisor']['automatic_retry'])
    def test_exit_code_classification(self):
        self.assertTrue(safety.native_exit(0xC0000409));self.assertFalse(safety.native_exit(1));self.assertFalse(safety.native_exit(0))
    def test_atomic_heartbeat_move_does_not_fail_supervisor(self):
        r=self.scenario(monitor=True,replace_during_scan=True)
        self.assertTrue(self.raced);self.assertEqual(r['status'],'PASS')
        self.assertIsNone(r['supervisor']['error'])
        self.assertEqual(r['supervisor']['cleanup']['active_processes'],0)
        self.assertTrue(self.job.terminated and self.job.closed)
    def test_evidence_budget_remains_enforced(self):
        self.m['supervision']['max_stage_evidence_bytes']=0
        r=self.scenario(monitor=True)
        self.assertEqual(r['status'],'FAIL');self.assertEqual(r['reason'],'finite_evidence_budget')
        self.assertTrue(self.job.terminated and self.job.closed)

if __name__=='__main__':unittest.main()
