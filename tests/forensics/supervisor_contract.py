"""Shared security invariants, proved with stdlib-only isolated Windows processes.

This checks the common safety contract, not equality of an entire supervisor AST.
The real kernel Job helper, admission bootstraps and both real supervisors run.
All created processes are owned by this test; no Torch or user GPU work executes.
"""
import ast
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT=Path(__file__).resolve().parents[2]


@lru_cache(maxsize=2)
def _live_contract(source_hashes):
    result=subprocess.run([sys.executable,'-I','-S','-B',__file__],capture_output=True,
        text=True,timeout=120,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1'),
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode!=0:raise AssertionError(result.stdout+result.stderr)
    return json.loads(result.stdout)


def assert_supervisor_contract(case, previous, current):
    case.assertEqual(current.LIMITS,previous.LIMITS)
    for kind,count in (('child',4),('outer',8)):
        limits=current.expected_limits(kind)
        case.assertEqual(limits,previous.expected_limits(kind))
        case.assertEqual(limits['flags'],0x2308)
        case.assertEqual(limits['processes'],count)
        case.assertEqual(limits['process_bytes'],4*1024**3)
    paths=[Path(previous.__file__),Path(current.__file__),ROOT/'hyso/performance/_training_process.py',
           *sorted((ROOT/'tools/forensics').glob('*admission*.py')),Path(__file__)]
    hashes=tuple(hashlib.sha256(p.read_bytes()).hexdigest() for p in paths)
    evidence=_live_contract(hashes)
    case.assertEqual(evidence['implementations'],['native_scenario_runner','gpu_scenario_runner'])
    case.assertEqual(evidence['regular_cases'],8)
    case.assertEqual(evidence['authorization_rejections'],4)
    case.assertEqual(evidence['unassigned_bootstrap_rejections'],2)
    case.assertEqual(evidence['nested_cases'],8)
    case.assertEqual(evidence['invalid_deadline_rejections'],14)
    case.assertEqual(evidence['process_limit_rejections'],2)
    case.assertTrue(evidence['outsider_survived'])
    case.assertTrue(evidence['all_jobs_empty'])
    case.assertFalse(evidence['torch_imported'])
    destination=os.environ.get('HYSO_SUPERVISOR_EVIDENCE')
    if destination:
        path=Path(destination)
        # Identical shared evidence may be requested by both original tests.
        if path.exists():case.assertEqual(json.loads(path.read_text()),evidence)
        else:path.write_text(json.dumps(evidence,indent=2),encoding='utf-8')


def _prepare(root, module):
    source=root/'source';source.mkdir(parents=True)
    files={}
    for name in ('native_scenario_runner.py','gpu_scenario_runner.py','native_admission.py',
                 'gpu_admission_policy.py','p23_preflight.py'):
        files[name]=(ROOT/'tools/forensics'/name).read_bytes()
    files['hyso/performance/_training_process.py']=(ROOT/'hyso/performance/_training_process.py').read_bytes()
    for name,data in files.items():
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    manifest=dict(run_id='isolated-supervision-contract',limits=module.LIMITS,scenario=module.SCENARIO,
        executions=1,retries=0,baselines=0,
        admission_policy=getattr(module.admission,'POLICY_ID',module.admission.policy.POLICY_ID),
        interpreter=sys.executable,runtime_files={sys.executable:module.digest(sys.executable)},
        source_files={name:hashlib.sha256(data).hexdigest() for name,data in files.items()})
    module.publish(root/'manifest.json',manifest)
    seal=dict(run_id=manifest['run_id'],manifest_sha256=module.digest(root/'manifest.json'))
    module.publish(root/'seal.json',seal)
    return manifest,seal


def _command(root,module,*arguments):
    return [sys.executable,'-I','-S','-B',str(root/'source'/(module.__name__+'.py')),
            '--root',str(root),*arguments]


def _process_limit(root,module):
    job=module.job_class(root)(module.LIMITS['child_job_private_commit'],module.LIMITS['child_processes'])
    processes=[]
    try:
        assert job.readback()==module.expected_limits('child')
        for index in range(module.LIMITS['child_processes']+1):
            p=subprocess.Popen([sys.executable,'-I','-S','-B','-c','import time;time.sleep(30)'],
                creationflags=subprocess.CREATE_NO_WINDOW)
            processes.append(p)
            if index<module.LIMITS['child_processes']:
                job.assign(p)
                assert job.receipt(p,module.expected_limits('child'))['assigned'] is True
            else:
                try:job.assign(p)
                except OSError as error:
                    assert error.errno==1816  # Kernel refuses the over-limit admission.
                    # The refused process is our dormant, unadmitted Popen. An
                    # assignment error does not promise that Windows killed it.
                    p.kill();p.wait(timeout=5)
                else:raise AssertionError('kernel accepted a process beyond the sealed Job limit')
        # A refused assignment can remain in Windows Job accounting until Job
        # termination. Check the limit readback and the live handles we own.
        assert job.readback()['processes']==module.LIMITS['child_processes']
        assert sum(p.poll() is None for p in processes)<=module.LIMITS['child_processes']
    finally:
        job.terminate()
        for p in processes:
            if p.poll() is None:p.kill()  # Includes our own rejected, unassigned Popen only.
            p.wait(timeout=5)
        assert job.observation()['active_processes']==0
        job.close()


def _main():
    sys.path[:0]=[str(ROOT/'tools/forensics'),str(Path(__file__).parent)]
    import native_scenario_runner as old
    import gpu_scenario_runner as new
    summary=dict(implementations=[old.__name__,new.__name__],regular_cases=0,
        authorization_rejections=0,unassigned_bootstrap_rejections=0,nested_cases=0,
        invalid_deadline_rejections=0,process_limit_rejections=0,
        outsider_survived=True,all_jobs_empty=True,torch_imported=False)
    outsider=subprocess.Popen([sys.executable,'-I','-S','-B','-c','import time;time.sleep(110)'],
        creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        with tempfile.TemporaryDirectory(prefix='hyso-supervisor-contract-') as temporary:
            for module in (old,new):
                root=Path(temporary)/module.__name__;manifest,seal=_prepare(root,module)
                for mode in ('success','timeout','log-overflow','outer-timeout'):
                    work=root/mode;work.mkdir()
                    role='parent' if mode=='outer-timeout' else 'worker'
                    receipt=module.supervise(root,work,role,mode,'contract-token',seal['manifest_sha256'],
                        timeout=1.5 if mode in ('timeout','outer-timeout') else 4)
                    assert receipt['assignment']['limits']==module.expected_limits('outer' if role=='parent' else 'child')
                    assert receipt['assignment']['assigned'] and receipt['kill_on_close']
                    assert receipt['owned_leader_exited'] and receipt['cleanup']['active_processes']==0
                    identity=module.read(work/(role+'-identity.json'))
                    assert identity['job_verified'] and not identity['cuda_imported']
                    assert identity['pid']==receipt['pid'] and identity['parent_pid']==os.getpid()
                    assert identity['token']=='contract-token' and identity['manifest_sha256']==seal['manifest_sha256']
                    if mode in ('timeout','outer-timeout'):assert receipt['outcome']=='timeout'
                    elif mode=='log-overflow':
                        assert receipt['outcome']=='bounded_log_exceeded'
                        assert receipt['log_bytes']['stdout']==70000
                        assert len(receipt['logs']['stdout'].encode())==65536
                    else:assert receipt['exit_code']==0 and receipt['outcome'] is None
                    assert outsider.poll() is None
                    summary['regular_cases']+=1
                for timeout in (float('nan'),float('inf'),-1,0,True,'1',61):
                    try:module.supervise(root,root/'never-created','worker','success','x',seal['manifest_sha256'],timeout=timeout)
                    except ValueError:pass
                    else:raise AssertionError('invalid deadline accepted')
                    assert not (root/'never-created').exists()
                    summary['invalid_deadline_rejections']+=1
                for kind in ('missing','wrong'):
                    work=root/('unauthorized-'+kind)
                    args=['--work',str(work),'--mode','native']
                    if kind=='wrong':
                        approval=root/'wrong-approval.json'
                        module.publish(approval,dict(run_id=manifest['run_id'],manifest_sha256=seal['manifest_sha256'],
                            scenario=manifest['scenario'],maximum_executions=1,explicit_user_approval=False))
                        args+=['--approval',str(approval)]
                    result=subprocess.run(_command(root,module,*args),capture_output=True,timeout=10)
                    assert result.returncode!=0 and b'authorization required' in result.stderr if kind=='missing' else (
                        result.returncode!=0 and b'approval identity mismatch' in result.stderr)
                    assert not work.exists() and not (root/'execution-consumed.json').exists()
                    summary['authorization_rejections']+=1
                work=root/'unassigned';work.mkdir()
                env=dict(os.environ,HYSO_CANDIDATE_SHA=seal['manifest_sha256'])
                p=subprocess.Popen(_command(root,module,'--work',str(work),'--internal','worker',
                    '--mode','success','--token','contract-token'),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    module.publish(work/'worker-admit.json',dict(pid=p.pid,parent_pid=os.getpid(),
                        limits=module.expected_limits('child'),assigned=False,query_handle=0,token='contract-token'))
                    out,err=p.communicate(timeout=10)
                    assert p.returncode!=0 and b'invalid Job assignment receipt' in err
                    assert not (work/'worker-identity.json').exists()
                finally:
                    if p.poll() is None:p.kill();p.wait(timeout=5)
                summary['unassigned_bootstrap_rejections']+=1
                _process_limit(root,module);summary['process_limit_rejections']+=1
            # Reuse existing native Job tests for real parent creation handles,
            # source/runtime/request tamper rejection and nested orphan cleanup.
            import unittest
            from test_nested_workers import NestedWorkerTests
            names=['test_preimport_admission_and_orphan_cleanup','test_timeout_and_tree_cleanup',
                'test_crash_is_not_oom','test_oom_transport_cleanup','test_wrong_parent_rejected_before_import',
                'test_modified_request_rejected_before_import','test_parent_creation_mismatch_rejected_before_import',
                'test_runtime_mismatch_rejected_before_import']
            result=unittest.TextTestRunner(stream=sys.stderr).run(unittest.TestSuite(NestedWorkerTests(name) for name in names))
            assert result.testsRun==8 and result.wasSuccessful()
            summary['nested_cases']=result.testsRun
            assert outsider.poll() is None and 'torch' not in sys.modules
    finally:
        if outsider.poll() is None:outsider.terminate()
        outsider.wait(timeout=5)
    print(json.dumps(summary))


if __name__=='__main__':_main()
