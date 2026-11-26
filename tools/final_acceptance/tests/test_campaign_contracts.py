"""Pure CPU/stdlib campaign checks, deferred until the user's main command."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import Journal, read_journal, read_json, atomic_json, ROOT, REPO
from campaign_checkpoint import aggregate, journal_results, Checkpoint
from campaign_preflight import check_config
from campaign_scheduler import Scheduler
from campaign_operations import cleanup_owned, expected_package_cases


def phases(*outcomes):
    return [dict(kind='phase',when=stage,outcome=outcome,duration=.1)
            for stage,outcome in zip(('setup','call','teardown'), outcomes)]


def test_call_pass_without_teardown_is_not_pass():
    assert aggregate('x',phases('passed','passed'))['status']=='NOT_RUN'


def test_teardown_error_overrides_call_pass():
    value=aggregate('x',phases('passed','passed','failed'))
    assert value['status']=='ERROR' and value['reason_code']=='TEARDOWN_ERROR'


def test_subtest_failure_overrides_main_call_pass():
    value=aggregate('x',phases('passed','passed','passed')+[
        dict(kind='phase',when='call',outcome='failed',is_subtest=True)])
    assert value['status']=='FAIL' and len(value['subtests'])==1


def test_cleanup_unverified_never_credits_pass():
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'journal.jsonl'
        journal=Journal(path,dict(attempt_id='a',source_commit='c',environment_id='e',group='g'))
        journal.emit('test_start',nodeid='x')
        for row in phases('passed','passed','passed'):
            journal.emit(row.pop('kind'),nodeid='x',**row)
        result,issues=journal_results(path,dict(cleanup_verified=False))
        assert not issues and result['x']['status']=='NOT_RUN'


def test_corrupt_hash_chain_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'journal.jsonl'
        Journal(path,{}).emit('test_start',nodeid='original')
        path.write_bytes(path.read_bytes().replace(b'original',b'tampered'))
        rows,issues=read_journal(path)
        assert not rows and issues[0]['reason']=='EVIDENCE_INVALID'


def test_power_loss_retains_complete_prefix():
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'journal.jsonl'
        Journal(path,{}).emit('test_start',nodeid='x')
        with path.open('ab') as stream:stream.write(b'{"partial":')
        rows,issues=read_journal(path)
        assert len(rows)==1 and issues[0]['reason']=='INTERRUPTED_LAST_WRITE'


def test_mandatory_limits_and_dangerous_flags_fail_closed():
    config=read_json(Path(__file__).resolve().parents[1]/'campaign_config.json')
    assert check_config(config)
    for key,value in [('allow_real_cuda',True),('process_memory_bytes',4*1024**3),('allow_p31_long_run',True)]:
        altered=dict(config,**{key:value})
        try:check_config(altered)
        except ValueError:pass
        else:raise AssertionError('unsafe configuration accepted: '+key)


def test_risk_results_remain_blocked_and_out_of_scheduler():
    with tempfile.TemporaryDirectory() as tmp:
        entries=[dict(nodeid='a.py::safe',path='a.py',profile='cpu'),
            dict(nodeid='b.py::risk',path='b.py',profile='cpu',risk_class='native')]
        cp=Checkpoint(tmp,entries,dict(config=dict(max_source_group_cases=100,max_source_group_files=10)))
        cp.data['results']['b.py::risk']=dict(status='PASS')
        scheduler=Scheduler(cp,{},tmp,lambda *a:None,lambda:False,lambda:True)
        assert cp.states()['b.py::risk']['status']=='BLOCKED'
        assert scheduler.source_groups()==[['a.py::safe']]


def test_cleanup_rejects_other_owner_and_outside_path():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
        root=Path(tmp);work=root/'owned';work.mkdir()
        atomic_json(work/'.hyso-owned-work.json',dict(campaign_id='different'))
        for path in (work,Path(other)):
            try:cleanup_owned(path,root,'current')
            except ValueError:pass
            else:raise AssertionError('unowned directory cleanup was allowed')
        assert work.is_dir() and Path(other).is_dir()


def test_package_denominator_is_separate_and_exact():
    cases=expected_package_cases()
    assert len(cases)==len(set(cases))==36
    assert sum(n.startswith('package:wheel::') for n in cases)==18
    assert sum(n.startswith('package:sdist::') for n in cases)==18


def test_source_integrity_failure_invalidates_complete_pass():
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'journal.jsonl';journal=Journal(path,{})
        for row in phases('passed','passed','passed'):journal.emit(row.pop('kind'),nodeid='x',**row)
        rows,_=journal_results(path,dict(source_integrity=False,cleanup_verified=True))
        assert rows['x']['status']=='NOT_RUN'


def test_unique_final_categories_reconcile_to_inventory():
    with tempfile.TemporaryDirectory() as tmp:
        cp=Checkpoint(tmp,[dict(nodeid='x'),dict(nodeid='y',risk_class='cuda')],{})
        cp.data['results']['x']=dict(nodeid='x',status='PASS')
        assert sum(cp.counts().values())==2 and cp.counts()['PASS']==cp.counts()['BLOCKED']==1


def test_live_resource_sample_timestamp_does_not_interrupt_owned_work():
    from unittest.mock import patch
    import campaign_scheduler
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);attempt=root/'attempt';attempt.mkdir()
        cp=Checkpoint(root,[],dict(config={}))
        scheduler=Scheduler(cp,{},root,lambda *args:None,lambda:False,lambda:True)
        scheduler.active=dict(group='mock',nodeid=None)
        with patch.object(campaign_scheduler,'ROOT',root):
            scheduler._update(attempt,dict(time_utc='prior',available_commit_bytes=5*1024**3),
                dict(active_processes=0,peak_job_commit_bytes=1024))
        sample=json.loads((root/'resource-samples.jsonl').read_text())
        assert sample['attempt']=='attempt' and sample['available_commit_bytes']==5*1024**3
        assert cp.data['resource_peaks']['peak_job_commit_bytes']==1024


def cleanup_binding():
    return dict(identity=dict(pid=71,creation_filetime=123,image='C:/Python/python.exe'),
        launcher_pid=71,created_suspended=True,exact_job_membership_before_resume=True,
        limits_readback=True,no_breakaway=True)


def test_windows_environment_case_preserves_systemroot_and_excludes_secrets():
    import campaign_controller as controller
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        for i,key in enumerate(('SYSTEMROOT','SystemRoot','systemroot')):
            work=Path(tmp)/str(i);work.mkdir()
            with patch.object(controller.os,'environ',{key:'C:/Windows','Path':'C:/Python',
                    'ProgramData':'C:/ProgramData','SECRET_TOKEN':'private','HYSO_ORT_CUDA_AVAILABLE':'1',
                    'PYTHONHASHSEED':'0'}):
                env=controller.safe_environment({},work)
            assert env['SYSTEMROOT']=='C:/Windows' and env['PROGRAMDATA']=='C:/ProgramData'
            assert env['CUDA_VISIBLE_DEVICES']=='-1'
            assert not {'SECRET_TOKEN','HYSO_ORT_CUDA_AVAILABLE','PYTHONHASHSEED'} & set(env)


def test_missing_systemroot_fails_before_any_launch():
    import campaign_controller as controller
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp, patch.object(controller.os,'environ',{}):
        try:controller.safe_environment({},Path(tmp))
        except RuntimeError as exc:assert 'SYSTEMROOT' in str(exc)
        else:raise AssertionError('missing mandatory Windows environment accepted')


def test_fixture_paths_cannot_override_windows_or_safety_environment():
    import campaign_controller as controller
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp, patch.object(controller.os,'environ',{'SYSTEMROOT':'C:/Windows'}):
        for key in ('SYSTEMROOT','SystemRoot','CUDA_VISIBLE_DEVICES'):
            try:controller.safe_environment({key:'unsafe'},Path(tmp))
            except ValueError:pass
            else:raise AssertionError('fixture attempted execution-environment override')


def test_preinit_exit_cleanup_requires_kernel_ownership_not_payload_admission():
    from campaign_controller import verified_cleanup
    assert verified_cleanup(dict(active_processes=0),cleanup_binding(),
        launcher_wait_verified=True,cleanup_error=None)


def test_cleanup_rejects_active_job_even_when_payload_was_admitted():
    from campaign_controller import verified_cleanup
    assert not verified_cleanup(dict(active_processes=1),cleanup_binding(),
        launcher_wait_verified=True,cleanup_error=None)


def test_cleanup_rejects_unbound_or_unsuspended_launcher():
    from campaign_controller import verified_cleanup
    binding=cleanup_binding()
    for key in ('created_suspended','exact_job_membership_before_resume','limits_readback','no_breakaway'):
        altered=dict(binding,**{key:False})
        assert not verified_cleanup(dict(active_processes=0),altered,launcher_wait_verified=True,cleanup_error=None)
    assert not verified_cleanup(dict(active_processes=0),None,launcher_wait_verified=True,cleanup_error=None)


def test_cleanup_rejects_pid_mismatch_unknown_exit_and_observation_errors():
    from campaign_controller import verified_cleanup
    binding=cleanup_binding();binding['launcher_pid']=72
    assert not verified_cleanup(dict(active_processes=0),binding,launcher_wait_verified=True,cleanup_error=None)
    for cleanup,wait,error in [(None,True,None),(dict(active_processes=0),False,None),
            (dict(active_processes=0),True,dict(message='QueryInformationJobObject failed'))]:
        assert not verified_cleanup(cleanup,cleanup_binding(),launcher_wait_verified=wait,cleanup_error=error)


def test_schema2_no_payload_admission_never_credits_test_pass():
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'journal.jsonl';journal=Journal(path,{})
        for row in phases('passed','passed','passed'):journal.emit(row.pop('kind'),nodeid='x',**row)
        rows,_=journal_results(path,dict(schema_version=2,source_integrity=True,
            cleanup_verified=True,payload_admitted=False))
        assert rows['x']['status']=='NOT_RUN' and rows['x']['reason_code']=='PAYLOAD_NOT_ADMITTED'


def test_resume_never_runs_a_launcher_outside_exact_job():
    import campaign_controller as controller
    from unittest.mock import patch
    with patch.object(controller,'exact_job_member',return_value=False), patch.object(controller.c,'WinDLL') as native:
        try:controller.resume_owned_launcher(object(),SimpleNamespace(_handle=71))
        except RuntimeError:pass
        else:raise AssertionError('unowned launcher was resumed')
        native.assert_not_called()


def mocked_early_exit(tmp,active):
    """Exercise controller receipt/ordering without launching any process or Job."""
    import campaign_controller as controller
    from contextlib import ExitStack
    from unittest.mock import patch
    import itertools
    calls=[]
    process=SimpleNamespace(pid=71,_handle=71,returncode=73,poll=lambda:73,
        wait=lambda **kw:73,kill=lambda:calls.append('kill'))
    job=SimpleNamespace(assign=lambda p:calls.append('assign'),terminate=lambda:calls.append('terminate'),
        close=lambda:calls.append('close'),observation=lambda:dict(active_processes=active))
    attempt=Path(tmp)/'attempt';attempt.mkdir()
    request=dict(work=str(Path(tmp)/'work'),attempt_id='mock',campaign_id='campaign',group='mock',
        token='nonce',source_commit='commit',started_utc='now',mode='startup-exit-before-latch')
    config=read_json(Path(__file__).resolve().parents[1]/'campaign_config.json')
    def popen(*args,**kwargs):
        assert kwargs['creationflags'] & 0x4
        calls.append('create_suspended')
        return process
    with ExitStack() as stack:
        stack.enter_context(patch.object(controller,'runtime_job',return_value=job))
        stack.enter_context(patch.object(controller.subprocess,'Popen',side_effect=popen))
        stack.enter_context(patch.object(controller,'process_identity',return_value=cleanup_binding()['identity']))
        stack.enter_context(patch.object(controller,'exact_job_member',return_value=True))
        stack.enter_context(patch.object(controller,'resume_owned_launcher',side_effect=lambda *args:calls.append('resume')))
        stack.enter_context(patch.object(controller,'resources',return_value={}))
        stack.enter_context(patch.object(controller.time,'monotonic',side_effect=itertools.count(0,30)))
        try:
            receipt=controller.execute(attempt,request,config,dict(python=sys.executable,runtime_paths=[]),
                update=lambda *args:None,stop_requested=lambda:False,integrity=lambda:True)
        except RuntimeError as exc:
            assert str(exc)=='OWNED_CLEANUP_UNVERIFIED; stop entire campaign'
            receipt=read_json(attempt/'controller.json')
        return receipt,calls


def test_preinit_failure_receipt_records_zero_job_without_claiming_payload_pass():
    with tempfile.TemporaryDirectory() as tmp:
        receipt,calls=mocked_early_exit(tmp,0)
        assert receipt['schema_version']==3 and receipt['cleanup_verified'] is True
        assert receipt['payload_admitted'] is False and receipt['identities'] is None
        assert receipt['monitor_reason']=='BOOTSTRAP_EXIT_BEFORE_ADMISSION' and receipt['exit_code']==73
        assert calls==['create_suspended','assign','resume','terminate','close']


def test_remaining_processes_still_stop_campaign_and_keep_failure_receipt():
    with tempfile.TemporaryDirectory() as tmp:
        receipt,calls=mocked_early_exit(tmp,1)
        assert receipt['cleanup_verified'] is False and receipt['cleanup']['active_processes']==1
        assert calls[-1]=='close'


def collection_fixture(tmp):
    from common import file_hash
    root=Path(tmp);path=root/'tests/reader.py';path.parent.mkdir();path.write_text('pass\n')
    rows=[dict(nodeid='tests/reader.py::'+name,path='tests/reader.py',profile='cpu',
        test_source_sha256=file_hash(path)) for name in ('safe','other')]
    cp=Checkpoint(root,rows,{})
    cp.data['infrastructure_inventory']={}
    return cp,rows


def test_parameter_ids_are_opaque_while_only_path_separators_normalize():
    from campaign_collection import canonical_nodeid
    suffix=r'test[x\ny\t\u0131-C:\\bad\x00path]'
    assert canonical_nodeid('tests\\reader.py::'+suffix)=='tests/reader.py::'+suffix
    assert canonical_nodeid('tests/reader.py::test[x/ny]')!=canonical_nodeid(r'tests/reader.py::test[x\ny]')


def test_exact_frozen_alias_migration_preserves_outcome_and_evidence():
    from campaign_collection import migrate_nodeids
    with tempfile.TemporaryDirectory() as tmp:
        cp,rows=collection_fixture(tmp)
        entry=dict(rows[0],nodeid=r'tests/reader.py::test[x\ny]')
        corrupted=entry['nodeid'].replace('\\','/')
        cp.data['inventory']={corrupted:dict(entry,nodeid=corrupted)}
        cp.data['results']={corrupted:dict(nodeid=corrupted,status='PASS',evidence_file='original.jsonl')}
        mapping=migrate_nodeids(cp,[entry])
        assert mapping=={corrupted:entry['nodeid']}
        assert cp.data['results'][entry['nodeid']]['status']=='PASS'
        assert cp.data['results'][entry['nodeid']]['evidence_file']=='original.jsonl'


def test_ambiguous_aliases_cannot_merge_two_tests():
    from campaign_collection import migrate_nodeids
    with tempfile.TemporaryDirectory() as tmp:
        cp,rows=collection_fixture(tmp)
        a=dict(rows[0],nodeid=r'tests/reader.py::test[x\ny]')
        b=dict(rows[0],nodeid='tests/reader.py::test[x/ny]')
        cp.data['inventory']={b['nodeid']:b}
        try:migrate_nodeids(cp,[a,b])
        except RuntimeError as exc:assert 'EVIDENCE_INVALID' in str(exc)
        else:raise AssertionError('ambiguous identity merge accepted')


def test_unknown_risky_node_is_quarantined_without_blocking_known_tests():
    import campaign_collection as module
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        cp,rows=collection_fixture(tmp)
        unknown=dict(rows[0],nodeid='tests/reader.py::actual_cuda',markers=['cuda'])
        with patch.object(module,'REPO',Path(tmp)),patch.object(module,'ROOT',Path(tmp)):
            result=module.reconcile_collection(cp,rows+[unknown],'cpu',evidence={})
        assert result['status']=='PARTIAL' and set(result['eligible'])==set(cp.data['inventory'])
        assert cp.data['quarantined_inventory']['cpu::'+unknown['nodeid']]['reason_code']=='BLOCKED_ADMISSION'
        assert unknown['nodeid'] not in cp.data['inventory']


def test_infrastructure_and_package_nodes_never_enter_cpu_source_inventory():
    import campaign_collection as module
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        cp,rows=collection_fixture(tmp)
        extra=[dict(nodeid='tools/final_acceptance/tests/test_x.py::test_x'),dict(nodeid='package:wheel::test_x')]
        with patch.object(module,'REPO',Path(tmp)),patch.object(module,'ROOT',Path(tmp)):
            result=module.reconcile_collection(cp,rows+extra,'cpu',evidence={})
        assert len(cp.data['inventory'])==2 and len(result['foreign_profiles'])==2
        assert {r['actual_profile'] for r in result['foreign_profiles']}=={'infrastructure','package'}


def test_duplicate_node_is_held_while_independent_node_remains_pending():
    import campaign_collection as module
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        cp,rows=collection_fixture(tmp)
        with patch.object(module,'REPO',Path(tmp)),patch.object(module,'ROOT',Path(tmp)):
            result=module.reconcile_collection(cp,rows+[rows[0]],'cpu',evidence={})
        assert result['duplicates']=={rows[0]['nodeid']:2}
        assert cp.states()[rows[0]['nodeid']]['status']=='BLOCKED'
        assert cp.pending()==[rows[1]['nodeid']]


def test_missing_current_node_preserves_original_pass_evidence():
    import campaign_collection as module
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        cp,rows=collection_fixture(tmp);node=rows[0]['nodeid']
        cp.data['results'][node]=dict(nodeid=node,status='PASS',evidence_file='original.jsonl')
        with patch.object(module,'REPO',Path(tmp)),patch.object(module,'ROOT',Path(tmp)):
            module.reconcile_collection(cp,rows[1:],'cpu',evidence={})
        assert cp.data['results'][node]['status']=='PASS'
        assert cp.states()[node]['status']=='BLOCKED' and cp.pending()==[rows[1]['nodeid']]


def test_known_but_unselected_cases_are_not_unknown_or_quarantined():
    import campaign_collection as module
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        cp,rows=collection_fixture(tmp)
        with patch.object(module,'REPO',Path(tmp)),patch.object(module,'ROOT',Path(tmp)):
            result=module.reconcile_collection(cp,rows,'cpu',expected_nodes=[rows[1]['nodeid']],evidence={})
        assert result['unknown']==[] and result['quarantined']==[] and result['status']=='PASS'


def test_changed_known_source_hash_stops_reconciliation():
    import campaign_collection as module
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        cp,rows=collection_fixture(tmp)
        (Path(tmp)/'tests/reader.py').write_text('changed\n')
        with patch.object(module,'REPO',Path(tmp)),patch.object(module,'ROOT',Path(tmp)):
            try:module.reconcile_collection(cp,rows,'cpu',evidence={})
            except RuntimeError as exc:assert 'EVIDENCE_INVALID' in str(exc)
            else:raise AssertionError('changed known source was accepted')


def test_unsafe_collection_paths_are_rejected_without_reading_external_files():
    from campaign_collection import canonical_nodeid
    for node in ('../outside.py::test_x','C:\\outside.py::test_x','/outside.py::test_x'):
        try:canonical_nodeid(node)
        except ValueError:pass
        else:raise AssertionError('unsafe collection path accepted')


def cached_collection_fixture(tmp):
    from common import file_hash
    from campaign_collection import collection_dependencies
    cp,rows=collection_fixture(tmp);folder=Path(tmp)/'collection';folder.mkdir()
    cp.data['baseline']=dict(commit='commit',source_sha256={'tests/reader.py':rows[0]['test_source_sha256']})
    request=dict(source_commit='commit',profile='cpu',mode='collect',inventory=cp.data['inventory'],
        collection_dependencies_sha256=collection_dependencies(cp.data['baseline']['source_sha256']))
    atomic_json(folder/'request.json',request);atomic_json(folder/'collected.json',rows)
    atomic_json(folder/'controller.json',dict(payload_admitted=True,cleanup_verified=True,
        source_integrity=True,monitor_reason=None,exit_code=5,request_sha256=file_hash(folder/'request.json')))
    journal=Journal(folder/'events.jsonl',{})
    journal.emit('runtime',torch_version='2.5.1+cpu',cuda_initialized=False)
    journal.emit('collection_finished',collected=len(rows))
    journal.emit('session_finish',torch_imported=True,cuda_initialized=False)
    cp.data['attempts']={'collection':dict(mode='collect',profile='cpu',status='FINISHED',directory=str(folder),
        collected_sha256=file_hash(folder/'collected.json'))}
    return cp,folder


def test_cached_collection_uses_valid_exit5_and_rejects_modified_artifact():
    import campaign_collection as module
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        cp,folder=cached_collection_fixture(tmp)
        with patch.object(module,'REPO',Path(tmp)):
            assert module.cached_collection(cp,'cpu') is not None
            atomic_json(folder/'collected.json',[])
            try:module.cached_collection(cp,'cpu')
            except RuntimeError as exc:assert 'EVIDENCE_INVALID' in str(exc)
            else:raise AssertionError('changed collected artifact accepted')


def test_cached_collection_is_not_reused_after_product_dependency_change():
    import campaign_collection as module
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        cp,folder=cached_collection_fixture(tmp)
        cp.data['baseline']['source_sha256']['hyso/changed.py']='new-digest'
        with patch.object(module,'REPO',Path(tmp)):
            assert module.cached_collection(cp,'cpu') is None


def recorded_source_fixture():
    from common import file_hash
    meta=read_json(Path(__file__).parent/'fixtures/recorded-source1.json')
    folder=REPO/meta['source_attempt']
    for name,sha in meta['files'].items():
        assert file_hash(folder/name)==sha, 'changed real replay fixture: '+name
    assert file_hash(REPO/meta['baseline_path'])==meta['baseline_sha256']
    return meta,folder,read_json(REPO/meta['baseline_path'])


def test_real_source_attempt_recovers_401_complete_passes_and_not_partial_call():
    from campaign_evidence import reconcile_attempt
    from collections import Counter
    meta,folder,baseline=recorded_source_fixture()
    audit=reconcile_attempt(folder,baseline=baseline)
    assert not audit['global_errors']
    assert dict(Counter(v['status'] for v in audit['results'].values()))==meta['expected_outcomes']
    assert audit['results'][meta['incomplete_node']]['status']=='NOT_RUN'
    assert all(audit['results'][n]['phases_complete'] for n in meta['examples'])


def test_real_attempt_hashes_independently_verify_prefix_and_quarantine_only_tail():
    import hashlib,json
    meta,folder,baseline=recorded_source_fixture()
    rows=[json.loads(line) for line in (folder/'events.jsonl').read_bytes().splitlines()]
    previous='0'*64
    for number,row in enumerate(rows[:meta['expected_prefix_events']],1):
        body={k:v for k,v in row.items() if k!='sha256'}
        raw=json.dumps(body,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
        assert hashlib.sha256(raw).hexdigest()==row['sha256'] and row['previous_sha256']==previous
        assert row['sequence']==number
        previous=row['sha256']
    from campaign_evidence import stream_audit
    audit=stream_audit(folder/'events.jsonl')
    assert audit['scope']=='LOCAL' and audit['quarantined_tail_events']==3
    assert all(row['kind'] not in ('test_start','phase') for row in rows[meta['expected_prefix_events']:])


def test_real_attempt_junit_corroborates_named_completed_examples():
    import xml.etree.ElementTree as ET
    meta,folder,baseline=recorded_source_fixture()
    cases=list(ET.parse(folder/'junit.xml').getroot().iter('testcase'))
    for node in meta['examples']:
        path,_,name=node.partition('::')
        matched=[c for c in cases if c.get('name')==name and c.get('classname')==path[:-3].replace('/','.')]
        assert len(matched)==1
        assert matched[0].find('failure') is None and matched[0].find('error') is None


def test_real_attempt_unexplained_prefix_tamper_is_global():
    from campaign_evidence import stream_audit
    meta,folder,baseline=recorded_source_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'events.jsonl'
        payload=(folder/'events.jsonl').read_bytes().replace(b'worker_start',b'worker_alter',1)
        path.write_bytes(payload)
        audit=stream_audit(path)
        assert audit['scope']=='GLOBAL' and audit['reason']=='UNEXPLAINED_JOURNAL_INTEGRITY_VIOLATION'


def test_real_attempt_identity_cleanup_and_source_mismatches_never_pass():
    import copy
    from campaign_evidence import reconcile_attempt
    meta,folder,baseline=recorded_source_fixture()
    original=read_json(folder/'controller.json')
    for fault in ('pid','cleanup','source'):
        receipt=copy.deepcopy(original)
        if fault=='pid':receipt['identities']['interpreter']['pid']+=1
        if fault=='cleanup':receipt['cleanup']['active_processes']=1
        if fault=='source':receipt['source_integrity']=False
        audit=reconcile_attempt(folder,baseline=baseline,receipt=receipt)
        assert audit['global_errors'] and all(v['status']!='PASS' for v in audit['results'].values())


def test_real_attempt_provisional_progress_changes_without_final_pass():
    from campaign_evidence import reconcile_attempt,provisional_progress
    meta,folder,baseline=recorded_source_fixture()
    request=read_json(folder/'request.json')
    with tempfile.TemporaryDirectory() as tmp:
        cp=Checkpoint(tmp,list(request['inventory'].values()),baseline)
        audit=reconcile_attempt(folder,baseline=baseline,provisional=True)
        cp.data['provisional_results']={n:dict(v,provisional_until_cleanup=True) for n,v in audit['results'].items()}
        progress=provisional_progress(cp,request['attempt_id'])
        assert cp.counts()['PASS']==0 and len(cp.pending())==len(request['selected_nodes'])
        assert progress['provisional_new_PASS']==401
        assert progress['provisional_remaining_tests']==len(request['selected_nodes'])-401


def test_real_attempt_checkpoint_recovery_is_idempotent():
    from unittest.mock import patch
    import campaign_evidence as module
    meta,folder,baseline=recorded_source_fixture();request=read_json(folder/'request.json')
    with tempfile.TemporaryDirectory() as tmp:
        cp=Checkpoint(tmp,list(request['inventory'].values()),baseline)
        cp.data['attempts']={request['attempt_id']:dict(mode='tests',profile='cpu',directory=str(folder))}
        with patch.object(module,'ROOT',Path(tmp)):
            assert module.recover_completed_attempts(cp)=={'PASS':401}
            assert module.recover_completed_attempts(cp)=={}
        assert cp.counts()['PASS']==401 and len(cp.pending())==len(request['selected_nodes'])-401


def test_product_monkeypatches_cannot_break_evidence_fsync_json_hash_or_clock():
    import os,json,hashlib,time
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'events.jsonl';journal=Journal(path,{})
        with patch.object(os,'fsync',side_effect=PermissionError('simulated product fault')), \
             patch.object(json,'dumps',side_effect=RuntimeError('product encoder fault')), \
             patch.object(hashlib,'sha256',side_effect=RuntimeError('product hash fault')), \
             patch.object(time,'monotonic',side_effect=RuntimeError('product clock fault')):
            journal.emit('test_start',nodeid='x')
            journal.emit('phase',nodeid='x',when='setup',outcome='passed')
        journal.close()
        records,issues=read_journal(path)
        assert len(records)==2 and issues==[]


def test_real_writer_io_error_poison_prevents_forked_append():
    import common
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'events.jsonl';journal=Journal(path,{})
        journal.emit('test_start',nodeid='x');head=journal.previous
        with patch.object(common,'_FSYNC',side_effect=OSError('genuine evidence persistence fault')):
            try:journal.emit('phase',nodeid='x',when='call',outcome='passed')
            except OSError:pass
            else:raise AssertionError('actual evidence fsync failure ignored')
        before=path.read_bytes()
        assert journal.failed and journal.sequence==1 and journal.previous==head
        try:journal.emit('framework_error',traceback='must not append using stale head')
        except RuntimeError as exc:assert 'JOURNAL_POISONED' in str(exc)
        else:raise AssertionError('poisoned writer appended a fork')
        journal.close();assert path.read_bytes()==before


def test_duplicate_or_reordered_main_phases_are_quarantined():
    for rows in (phases('passed','passed','passed')+[dict(kind='phase',when='call',outcome='passed')],
            list(reversed(phases('passed','passed','passed')))):
        result=aggregate('x',rows)
        assert result['status']=='ERROR' and result['evidence_quarantined']


def test_skipped_subtest_cannot_be_promoted_to_full_pass():
    rows=phases('passed','passed','passed')
    rows.insert(1,dict(kind='phase',when='call',outcome='skipped',is_subtest=True))
    assert aggregate('x',rows)['status']=='SKIP'


def test_real_forensic_subtest_sequence_is_reconciled_without_running_forensics():
    from common import file_hash
    meta,folder,baseline=recorded_source_fixture();path=REPO/meta['forensic_path']
    assert file_hash(path)==meta['forensic_sha256']
    rows=[json.loads(line) for line in path.read_bytes().splitlines()]
    rows=[r for r in rows if r['nodeid']==meta['forensic_subtest_node']]
    result=aggregate(meta['forensic_subtest_node'],rows)
    assert result['status']=='PASS' and len(result['subtests'])==meta['expected_forensic_subtests']
    subtest=dict(next(r for r in rows if r.get('is_subtest')),outcome='failed')
    assert aggregate(meta['forensic_subtest_node'],rows+[subtest])['status']=='FAIL'


def test_real_request_product_dependency_change_is_global():
    import copy
    from campaign_evidence import reconcile_attempt
    meta,folder,baseline=recorded_source_fixture();changed=copy.deepcopy(baseline)
    changed['source_sha256']['hyso/training/changed.py']='0'*64
    audit=reconcile_attempt(folder,baseline=changed)
    assert 'PRODUCT_DEPENDENCY_SNAPSHOT_CHANGED' in audit['global_errors']
    assert all(v['status']!='PASS' for v in audit['results'].values())


def test_recorded_phase_metadata_fault_is_local_to_one_node():
    import shutil
    from campaign_evidence import reconcile_attempt
    meta,folder,baseline=recorded_source_fixture()
    records,_=read_journal(folder/'events.jsonl')
    nodes=[r['nodeid'] for r in records if r['kind']=='test_start'][:2]
    with tempfile.TemporaryDirectory() as tmp:
        variant=Path(tmp)
        for name in meta['files']:
            if name!='events.jsonl':shutil.copyfile(folder/name,variant/name)
        journal=Journal(variant/'events.jsonl',{})
        for row in records:
            if row['kind'] not in ('worker_start','runtime') and row.get('nodeid') not in nodes:continue
            fields={k:v for k,v in row.items() if k not in ('kind','schema_version','sequence','sha256','previous_sha256','time_utc')}
            if row.get('nodeid')==nodes[0] and row.get('when')=='call':fields['test_source_sha256']='0'*64
            journal.emit(row['kind'],**fields)
        journal.close()
        audit=reconcile_attempt(variant,baseline=baseline)
        assert not audit['global_errors']
        assert audit['results'][nodes[0]]['evidence_quarantined']
        assert audit['results'][nodes[1]]['status']=='PASS'


def recorded_forensic_fixture():
    from common import file_hash
    meta=read_json(Path(__file__).parent/'fixtures/recorded-forensic.json')
    folder=REPO/meta['attempt']
    for name,sha in meta['files'].items():
        assert file_hash(folder/name)==sha, 'changed forensic replay fixture: '+name
    assert file_hash(REPO/meta['baseline_path'])==meta['baseline_sha256']
    return meta,folder,read_json(REPO/meta['baseline_path'])['baseline']


def test_direct_interpreter_forensic_receipt_recovers_pass():
    from campaign_evidence import reconcile_attempt
    meta,folder,baseline=recorded_forensic_fixture()
    audit=reconcile_attempt(folder,baseline=baseline)
    assert not audit['global_errors']
    assert audit['results'][meta['nodeid']]['status']=='PASS'
    identities=audit['receipt']['identities']
    assert identities['interpreter']['pid']==identities['launcher']['pid']
    finish=[r for r in audit['events'] if r['kind']=='session_finish']
    assert finish and all(r['torch_imported'] is False and r['cuda_initialized'] is False for r in finish)


def test_direct_interpreter_wrong_parent_is_rejected():
    from unittest.mock import patch
    import campaign_evidence as module
    meta,folder,baseline=recorded_forensic_fixture()
    original=module.read_json
    def altered(path,*args,**kwargs):
        value=original(path,*args,**kwargs)
        if Path(path)==folder/'interpreter-ready.json':
            value=dict(value,parent_pid=-1)
        return value
    with patch.object(module,'read_json',side_effect=altered):
        audit=module.reconcile_attempt(folder,baseline=baseline)
    assert 'PROCESS_IDENTITY_OR_ADMISSION_MISMATCH' in audit['global_errors']
    assert audit['results'][meta['nodeid']]['status']=='NOT_RUN'


def test_isolated_worker_bootstrap_uses_single_base_interpreter_without_site():
    import ast
    source=ast.parse((REPO/'hyso/analysis/isolated_bridge.py').read_text(encoding='utf-8'))
    helper=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='_worker_command')
    namespace={'sys':SimpleNamespace(_base_executable='base-python.exe',executable='venv-python.exe',sys_path=[]), 'Path':Path}
    namespace['sys'].path=[str(REPO)]
    exec(compile(ast.Module(body=[helper],type_ignores=[]),'<reviewed-worker-command>','exec'),namespace)
    command=namespace['_worker_command'](REPO/'worker.py',REPO/'config.json',REPO/'result.json')
    assert command[:5]==['base-python.exe','-I','-S','-B','-c']
    compile(command[-1],'<worker-bootstrap>','exec')
    assert 'runpy.run_path' in command[-1] and repr(str(REPO)) in command[-1]


def test_package_revalidation_preserves_34_passes_and_reopens_only_two_csv_cases():
    import copy,shutil
    from unittest.mock import patch
    import campaign_controller_main as module
    from common import file_hash
    before=ROOT/'remaining-defects-20261126/before-checkpoint.json'
    assert file_hash(before)=='4bd45237b14f4920224e450826adae05ae8381660feb52f53b49bbc620990375'
    cp=SimpleNamespace(data=copy.deepcopy(read_json(before)))
    expected=['package:'+kind+'::test_72_native_CSV_preserves_JSON_capture' for kind in ('wheel','sdist')]
    cp.data['package_case_revalidation']={n:dict(excluded_attempt_ids=list(cp.data['attempts'])) for n in expected}
    with tempfile.TemporaryDirectory() as tmp:
        work=Path(tmp)/'work';work.mkdir()
        for kind,attempt in [('wheel','80eeeb81ccdc45d89236c9ee8c06f6f6'),('sdist','6ad7bd2aeb1e47cdb36caad24f57c769')]:
            folder=work/(kind+'-external/E2E');folder.mkdir(parents=True)
            name='package-events-'+attempt+'.jsonl'
            shutil.copyfile(ROOT/'package-journals'/name,folder/name)
        with patch.object(module,'ROOT',Path(tmp)):
            result=module.package_state(cp,work,cp.data['package_operations'])
        assert result['test_counts']=={'PASS':34,'NOT_RUN':2}
        assert {v['nodeid'] for v in result['cases'] if v['status']=='NOT_RUN'}==set(expected)
        for kind in ('wheel','sdist'):
            terminal=read_json(work/(kind+'-resume.json'))['terminal_cases']
            assert len(terminal)==17 and all(s=='PASS' for s in terminal.values())
            assert 'test_72_native_CSV_preserves_JSON_capture' not in terminal


def test_reviewed_retirement_keeps_original_fail_and_all_passes_without_allowing_unknown_removals():
    import copy
    from unittest.mock import patch
    import campaign_collection as module
    from campaign_inventory import prepared_inventory
    from common import file_hash
    before=ROOT/'retirement-support-20261126/before-checkpoint.json'
    assert file_hash(before)=='27e816aae12e2a55f5446cd7c4da3ece9ffb6a987a23c916314bd9b9a541ab60'
    cp=SimpleNamespace(data=copy.deepcopy(read_json(before)))
    changes=read_json(Path(module.__file__).with_name('retired-tests.json'))
    target=changes['retirements'][0]['nodeid'];previous=copy.deepcopy(cp.data['results'][target])
    wanted=prepared_inventory()
    with tempfile.TemporaryDirectory() as tmp,patch.object(module,'ROOT',Path(tmp)):
        module.migrate_nodeids(cp,wanted,reviewed_changes=changes)
        assert target not in cp.data['inventory'] and target not in cp.data['results']
        archived=cp.data['retired_tests'][target]
        assert archived['status']=='RETIRED' and archived['previous_status']=='FAIL'
        assert archived['original_result']==previous
        assert sum(v['status']=='PASS' for v in cp.data['results'].values())==8535
        module.migrate_nodeids(cp,wanted,reviewed_changes=changes)
        assert len(list((Path(tmp)/'retired-history').glob('*.json')))==1
        removed=next(r['nodeid'] for r in wanted if cp.data['results'].get(r['nodeid'],{}).get('status')=='PASS')
        try:module.migrate_nodeids(cp,[r for r in wanted if r['nodeid']!=removed],reviewed_changes=changes)
        except RuntimeError:pass
        else:raise AssertionError('unreviewed active-test removal was allowed')
