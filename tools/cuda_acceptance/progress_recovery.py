"""Close only the known pre-test progress callback failure, without accepting tests."""
from pathlib import Path
from common import atomic_json,digest,file_hash,read_json
from campaign_evidence import reconcile_attempt


def recover_progress_failure(cp,old_plan,new_plan,root):
    candidates=[]
    for attempt in cp['attempts']:
        if attempt['status']!='RUNNING' or attempt['group']!='CPU_REGRESSIONS':continue
        folder=Path(attempt['directory']);receipt=read_json(folder/'controller.json',{})
        error=receipt.get('error') or {}
        trace=error.get('traceback','')
        known=(receipt.get('monitor_reason')=='CONTROLLER_ERROR' and error.get('type')=='AttributeError'
            and error.get('message')=="'tuple' object has no attribute 'items'"
            and all(value in trace for value in ('runner.py','progress','journal_results','items')))
        if known:candidates.append((attempt,folder,receipt))
    if not candidates:return
    def require(value,reason):
        if not value:raise RuntimeError('PROGRESS_FAILURE_RECOVERY_REFUSED: '+reason)
    require(cp['plan_sha256']==digest(old_plan),'checkpoint plan binding')
    require(old_plan['cpu_inventory']==new_plan['cpu_inventory'] and old_plan['groups']==new_plan['groups'],'scope changed')
    require(old_plan['runtime']==new_plan['runtime'] and old_plan['budgets']==new_plan['budgets'],'runtime/budgets changed')
    allowed={'tools/cuda_acceptance/runner.py','tools/cuda_acceptance/progress_recovery.py',
        'tools/cuda_acceptance/tests/test_progress.py'}
    before=old_plan['source_sha256'];after=new_plan['source_sha256']
    changed={n for n in set(before)|set(after) if before.get(n)!=after.get(n)}
    require(changed<=allowed,'changes exceed this infrastructure fix')
    require(old_plan['preserved_evidence']==new_plan['preserved_evidence'],'historical evidence changed')
    for path,sha in old_plan['preserved_evidence'].items():require(file_hash(path)==sha,'historical evidence hash')
    for attempt,folder,receipt in candidates:
        require(folder.resolve().parent==(Path(root)/'attempts').resolve(),'attempt directory')
        require(attempt.get('plan_sha256')==digest(old_plan),'attempt plan binding')
        require(receipt.get('schema_version')==3 and receipt.get('source_integrity') is True,'source receipt')
        require(receipt.get('cleanup_verified') is True and receipt.get('cleanup',{}).get('active_processes')==0
            and receipt.get('cleanup_error') is None,'verified empty owned Job')
        require(receipt.get('assigned_before_imports') is True and receipt.get('limit_readback_verified') is True
            and receipt.get('launcher_wait_verified') is True,'admission/readback/reap')
        binding=receipt.get('launch_binding') or {};admission=read_json(folder/'admission-receipt.json',{})
        require(binding.get('created_suspended') is True and binding.get('exact_job_membership_before_resume') is True
            and binding.get('no_breakaway') is True and binding.get('limits_readback') is True
            and admission.get('no_breakaway') is True and admission.get('kill_on_close') is True
            and admission.get('limits_readback') is True,'no-breakaway ownership')
        audit=reconcile_attempt(folder,baseline=dict(commit=old_plan['commit'],source_sha256=before))
        # CPU_RUNTIME_UNVERIFIED stays an error, never a waived PASS condition.
        # It is allowed only as a secondary diagnostic for an entirely unstarted
        # test scope. A future worker must pass the unchanged runtime gate.
        require(set(audit['global_errors'])<={'CPU_RUNTIME_UNVERIFIED'},'bound evidence: '+repr(audit['global_errors']))
        require(audit['stream']['scope']=='VALID' and not audit['results'],'no test results to accept')
        require([r['kind'] for r in audit['events']] in ([],['worker_start']),'runtime or test execution observed')
        request=audit['request']
        require(request['attempt_id']==receipt['attempt_id']==folder.name and request['group']=='CPU_REGRESSIONS'
            and request['profile']=='cpu' and request['mode']=='tests','attempt identity')
        require(request['inventory']==old_plan['cpu_inventory'] and set(request['selected_nodes'])==set(old_plan['cpu_inventory']),
            'original CPU selection')
        require(not cp.get('cpu_results') and not cp['results'],'no accepted results may be replaced')
        original=dict(attempt)
        manifest={p.name:file_hash(p) for p in folder.iterdir() if p.is_file()}
        recovery=dict(reason='PRE_TEST_PROGRESS_CONTROLLER_ERROR',attempt=original,controller_error=receipt['error'],
            secondary_evidence_errors=audit['global_errors'],cleanup_verified=True,active_processes=0,
            accepted_results=0,tests_started=0,original_files_sha256=manifest,
            resume='Only never-started cases remain pending; runtime and ownership admission stay mandatory.')
        path=Path(root)/'recovery'/(folder.name+'-progress.json')
        atomic_json(path,recovery)
        attempt.update(status='RECOVERED_CONTROLLER_ERROR',receipt_sha256=manifest['controller.json'],
            recovery_evidence=str(path),recovery_sha256=file_hash(path),accepted_results=0)
        cp['history'].append(recovery)
        cp['last_controller_error']=dict(attempt_id=folder.name,**receipt['error'],
            secondary_evidence_errors=audit['global_errors'],recovery_evidence=str(path))
