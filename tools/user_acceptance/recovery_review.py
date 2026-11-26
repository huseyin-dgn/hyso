"""Narrow, hash-pinned source transition for the U13/runtime + cleanup defect."""
from pathlib import Path


def review_transition(old,new,cp):
    from tools.user_acceptance import runner as u
    e=u.engine
    prior=old.get('source_impact_review')
    if prior and old['source_sha256']==new['source_sha256'] and all(old[k]==new[k] for k in
            ('runtime','budgets','feature_tools','groups','cpu_inventory','preserved_evidence','user_files')):
        return prior
    precision=u07_precision_transition(old,new,cp)
    if precision:return precision
    memory=benchmark_memory_transition(old,new,cp)
    if memory:return memory
    four=four_issue_transition(old,new,cp)
    if four:return four
    # A prior fail-closed preparation may have held these same results before
    # the reviewed Git tracking transition was recognized. Recover only the
    # exact archived checkpoint: no new attempts or altered terminal evidence.
    spec=e.read_json(Path(__file__).with_name('four-failures-review.json'),{})
    for path in (u.ROOT/'history').glob('plan-*.json'):
        archived=e.read_json(path);baseline=archived.get('plan',{});proof=archived.get('checkpoint',{})
        if e.digest(baseline)!=spec.get('baseline_plan_sha256'):continue
        def terminal(value,key):
            return {n:{k:v for k,v in row.items() if k!='source_valid'} for n,row in value.get(key,{}).items() if n!='CPU_REGRESSIONS'}
        if cp['attempts']!=proof.get('attempts') or any(terminal(cp,key)!=terminal(proof,key) for key in ('results','cpu_results')):continue
        reviewed=four_issue_transition(baseline,new,proof)
        if reviewed:return dict(reviewed,recovered_unchanged_terminal_results=True)
    spec=e.read_json(Path(__file__).with_name('u13-cleanup-review.json'),{})
    if e.digest(old)!=spec.get('baseline_plan_sha256'):return None
    changed={n for n in set(old['source_sha256'])|set(new['source_sha256'])
             if old['source_sha256'].get(n)!=new['source_sha256'].get(n)}
    expected=set(spec.get('files',{}))|{'tools/user_acceptance/u13-cleanup-review.json'}
    if changed!=expected:return None
    for name,pins in spec['files'].items():
        if old['source_sha256'].get(name)!=pins['before'] or new['source_sha256'].get(name)!=pins['after']:return None
    for key in ('runtime','budgets','feature_tools','groups','cpu_inventory','preserved_evidence'):
        if old[key]!=new[key]:return None
    user_changes={n for n in set(old['user_files'])|set(new['user_files']) if old['user_files'].get(n)!=new['user_files'].get(n)}
    if not user_changes<=expected:return None
    u.verify_cached(cp)
    rows=cp.get('cpu_results',{});failed='tests/test_user_acceptance_scenarios.py::test_U13'
    if set(rows)!=set(old['cpu_inventory']) or rows[failed]['status']!='FAIL':return None
    retained=sorted(n for n,r in rows.items() if r['status']=='PASS' and r.get('source_valid',True))
    if len(retained)!=13 or any(n!='CPU_REGRESSIONS' for n in cp['results']):return None
    for attempt in cp['attempts']:
        folder=Path(attempt['directory']);receipt=e.read_json(folder/'controller.json')
        if not (attempt['status']=='FINISHED' and attempt['group']=='CPU_REGRESSIONS' and
                attempt['receipt_sha256']==e.file_hash(folder/'controller.json') and
                receipt['cleanup_verified'] is True and receipt['cleanup']['active_processes']==0 and
                receipt['cleanup_error'] is None and receipt['source_integrity'] is True):
            raise RuntimeError('REVIEW_REQUIRES_VERIFIED_OWNED_CLEANUP')
        admission=e.read_json(folder/'admission-receipt.json')
        if not all(admission.get(k) is True for k in ('no_breakaway','kill_on_close','limits_readback')):
            raise RuntimeError('REVIEW_OWNERSHIP_UNVERIFIED')
        audit=e.reconcile_attempt(folder,baseline=dict(commit=old['commit'],source_sha256=old['source_sha256']))
        if audit['global_errors'] or any(audit['results'].get(n,{}).get('status')!=rows[n]['status'] for n in rows):
            raise RuntimeError('REVIEW_SEALED_JOURNAL_MISMATCH')
    return dict(kind='U13_CPU_RUNTIME_AND_CLEANUP_SCOPE_ONLY',baseline_plan_sha256=e.digest(old),
        retained_cpu_pass=retained,revalidation_cpu=[failed],gpu_started=False,
        controller_seal_journal_artifact_ownership_verified=True,
        historical_scratch='Retained untouched; process cleanup verified; no retroactive marker or filesystem deletion.')


def four_issue_transition(old,new,cp):
    """Retain unchanged execution profiles only after their original seals verify."""
    from tools.user_acceptance import runner as u
    e=u.engine;spec=e.read_json(Path(__file__).with_name('four-failures-review.json'),{})
    if e.digest(old)!=spec.get('baseline_plan_sha256'):return None
    changed={n for n in set(old['source_sha256'])|set(new['source_sha256']) if old['source_sha256'].get(n)!=new['source_sha256'].get(n)}
    expected=set(spec.get('files',{}))|{'tools/user_acceptance/four-failures-review.json'}
    if changed!=expected:return None
    if any(old['source_sha256'].get(n)!=pins['before'] or new['source_sha256'].get(n)!=pins['after'] for n,pins in spec['files'].items()):return None
    if any(old[k]!=new[k] for k in ('runtime','budgets','feature_tools','groups','cpu_inventory','preserved_evidence')):return None
    user_changes={n for n in set(old['user_files'])|set(new['user_files']) if old['user_files'].get(n)!=new['user_files'].get(n)}
    tracked=set(e.git('ls-files','-z').split('\0'))
    if not user_changes-tracking_only_changes(old,new,user_changes,tracked)<=expected:return None
    u.verify_cached(cp)
    plans={e.digest(old):old}
    for path in (u.ROOT/'history').glob('plan-*.json'):
        historical=e.read_json(path).get('plan')
        if historical:plans[e.digest(historical)]=historical
    for attempt in cp['attempts']:
        folder=Path(attempt['directory']);r=e.read_json(folder/'controller.json')
        if not (attempt['status']=='FINISHED' and attempt['receipt_sha256']==e.file_hash(folder/'controller.json') and
                r['source_integrity'] is True and r['cleanup_verified'] is True and r['cleanup']['active_processes']==0 and
                r.get('cleanup_error') is None and not r.get('monitor_reason')):
            raise RuntimeError('FOUR_FIX_REVIEW_OWNERSHIP_OR_CLEANUP_UNVERIFIED')
        if attempt['group']=='CPU_REGRESSIONS':
            baseline=plans.get(attempt['plan_sha256'])
            if baseline is None:raise RuntimeError('FOUR_FIX_REVIEW_ORIGINAL_PLAN_MISSING')
            audit=e.reconcile_attempt(folder,baseline=dict(commit=baseline['commit'],source_sha256=baseline['source_sha256']))
            if audit['global_errors']:raise RuntimeError('FOUR_FIX_REVIEW_CPU_SEAL_INVALID: '+repr(audit['global_errors']))
        # GPU terminal files, admission, journal and feature hashes were checked
        # above by verify_cached, including every retained native PASS witness.
    cpu=sorted(n for n,r in cp['cpu_results'].items() if r['status']=='PASS' and r.get('source_valid',True))
    gpu=sorted(n for n,r in cp['results'].items() if n!='CPU_REGRESSIONS' and r['status']=='PASS' and r.get('source_valid',True))
    if len(cpu)!=13 or len(gpu)!=9:return None
    return dict(kind='FOUR_USER_FAILURES_ONLY',baseline_plan_sha256=e.digest(old),retained_cpu_pass=cpu,retained_gpu_pass=gpu,
        revalidation_cpu=['tests/test_user_acceptance_scenarios.py::test_U13'],revalidation_gpu=['U02','U05','U07'],
        impact='Active autocast Graph only; CUDA evaluation capture only; CPU trace checker only; CUDA architecture budget only.',
        evidence='Original plan, controller, journal, ownership, runtime and artifact hashes verified; all attempts retained.')


def tracking_only_changes(old,new,changed,tracked):
    """Git addition is not a content change; all three byte hashes must agree."""
    return {n for n in changed if n in tracked and n not in new['user_files'] and
        old['user_files'].get(n) is not None and
        old['user_files'][n]==old['source_sha256'].get(n)==new['source_sha256'].get(n)}


def verify_oom_cleanup(folder,plan):
    """An incomplete OOM session is never PASS; validate its closed ownership tree."""
    from tools.user_acceptance import runner as u
    from tools.cuda_acceptance.identity import parent_binding
    e=u.engine;folder=Path(folder);r=e.read_json(folder/'controller.json');q=e.read_json(folder/'request.json')
    def need(ok,reason):
        if not ok:raise RuntimeError('OOM_RECOVERY_UNVERIFIED: '+reason)
    need(r['cleanup_verified'] is True and r['cleanup']['active_processes']==0 and r['cleanup_error'] is None,'cleanup')
    need(q['attempt_id']==r['attempt_id']==folder.name and e.digest(q['gpu_plan'])==e.digest(plan),'original plan')
    need(r['source_integrity'] is True and r['assigned_before_imports'] is True and r['payload_admitted'] is True and
        r['limit_readback_verified'] is True and r['kill_on_close'] is True and not r['monitor_reason'] and r['exit_code']==1,'controller')
    b=plan['budgets'];need(r['limits']==dict(process_bytes=b['process_bytes'],job_bytes=b['job_bytes'],max_processes=b['max_processes']),'limits')
    for key,name in [('request_sha256','request.json'),('bootstrap_sha256','bootstrap.py'),('journal_sha256','events.jsonl'),
            ('worker_terminal_sha256','journal-terminal.json'),('terminal_evidence_sha256','terminal-evidence.json')]:
        need(r[key]==e.file_hash(folder/name),key)
    admission=e.read_json(folder/'admission-receipt.json');ids=r['identities'];ready=e.read_json(folder/'interpreter-ready.json');admit=e.read_json(folder/'admit.json')
    need(ids==admission['identities']==e.read_json(folder/'owner-identities.json') and ids['exact_job_membership'] and
        ids['nonce_sha256']==e.digest(q['token']) and parent_binding(ids,ready,r['launch_binding']) and
        admit['readback_verified'] and admit['pid']==ready['pid']==ids['interpreter']['pid'] and
        admit['token']==ready['token']==q['token'],'identities')
    need(all(admission.get(k) is True for k in ('no_breakaway','kill_on_close','limits_readback')),'Job inheritance')
    rows,issues=e.read_journal(folder/'events.jsonl');terminal=e.read_json(folder/'journal-terminal.json');seal=e.read_json(folder/'terminal-evidence.json')
    need(not issues and bool(rows) and terminal['closed'] and not terminal['writer_failed'] and
        terminal['committed_sequence']==len(rows)==seal['verified_prefix_events'] and
        terminal['committed_head_sha256']==rows[-1]['sha256']==seal['verified_prefix_head'] and
        seal['journal_sha256']==r['journal_sha256'],'journal seal')
    need(all(all(row[k]==q[k] for k in ('attempt_id','group','source_commit','environment_id')) for row in rows),'journal identity')
    need(rows[0]['kind']=='worker_start' and rows[0]['pid']==ids['interpreter']['pid'] and rows[0]['profile']=='cuda-functional','worker identity')
    need(q['features']==['U02'] and q['group']=='02-inference' and not any(row['kind'] in ('feature_end','gpu_session_finish') for row in rows) and
        any(row['kind']=='critical_error' and 'torch.OutOfMemoryError' in row['reason'] and 'benchmark' in row['reason'] for row in rows),'original OOM')
    runtime=e.read_json(folder/'gpu-runtime.json')
    need(runtime['uuid']==plan['gpu_uuid'] and runtime['loaded_binaries']==plan['runtime']['binary_expectations'] and
        runtime['process']==ids['interpreter'] and runtime['probe']==dict(completed=True,input=1,output=2,elements=1),'original runtime')
    need(any(row['kind']=='gpu_identity' and row['sha256_file']==e.file_hash(folder/'gpu-runtime.json') for row in rows),'runtime seal')


def benchmark_memory_transition(old,new,cp):
    from tools.user_acceptance import runner as u
    e=u.engine;spec=e.read_json(Path(__file__).with_name('benchmark-memory-review.json'),{})
    if e.digest(old)!=spec.get('baseline_plan_sha256'):return None
    changed={n for n in set(old['source_sha256'])|set(new['source_sha256']) if old['source_sha256'].get(n)!=new['source_sha256'].get(n)}
    expected=set(spec.get('files',{}))|{'tools/user_acceptance/benchmark-memory-review.json'}
    if changed!=expected:return None
    if any(old['source_sha256'].get(n)!=pins['before'] or new['source_sha256'].get(n)!=pins['after'] for n,pins in spec['files'].items()):return None
    if any(old[k]!=new[k] for k in ('runtime','budgets','feature_tools','groups','cpu_inventory','preserved_evidence')):return None
    user_changes={n for n in set(old['user_files'])|set(new['user_files']) if old['user_files'].get(n)!=new['user_files'].get(n)}
    if not user_changes-tracking_only_changes(old,new,user_changes,set(e.git('ls-files','-z').split('\0')))<=expected:return None
    if cp.get('safety_hold',{}).get('features')!=['U02']:return None
    u.verify_cached(cp)
    failed=u.ROOT/'attempts'/spec['oom_attempt']
    attempt=next((a for a in cp['attempts'] if Path(a['directory'])==failed),None)
    if not attempt or attempt['status']!='FINISHED' or attempt['receipt_sha256']!=e.file_hash(failed/'controller.json'):
        raise RuntimeError('OOM_RECOVERY_ATTEMPT_UNVERIFIED')
    if any(a['status']!='FINISHED' for a in cp['attempts']):raise RuntimeError('OOM_RECOVERY_UNFINISHED_OWNED_WORK')
    verify_oom_cleanup(failed,old)
    cpu=sorted(n for n,r in cp['cpu_results'].items() if r['status']=='PASS' and r.get('source_valid',True))
    gpu=sorted(n for n,r in cp['results'].items() if n!='CPU_REGRESSIONS' and r['status']=='PASS' and r.get('source_valid',True))
    if len(cpu)!=14 or len(gpu)!=9:return None
    return dict(kind='BOUNDED_BENCHMARK_FINITE_VALIDATION',baseline_plan_sha256=e.digest(old),
        retained_cpu_pass=cpu,retained_gpu_pass=gpu,revalidation_cpu=[],revalidation_gpu=['U02','U05','U07'],
        oom_attempt=spec['oom_attempt'],verified_oom_cleanup=True,fresh_worker_admission_required=True,
        recovery_order=['U07','U05','U02'],safety_hold='U02 remains explicit; any new OOM stops immediately',
        impact='Equivalent complete finite checks; bounded masks/transfers outside timing; original finite PASS witnesses retained.')


def u07_precision_transition(old,new,cp):
    """Only mixed-output P28 behavior changes; retain proven unaffected profiles."""
    from tools.user_acceptance import runner as u
    e=u.engine;spec=e.read_json(Path(__file__).with_name('u07-precision-review.json'),{})
    if e.digest(old)!=spec.get('baseline_plan_sha256'):return None
    changed={n for n in set(old['source_sha256'])|set(new['source_sha256']) if old['source_sha256'].get(n)!=new['source_sha256'].get(n)}
    expected=set(spec.get('files',{}))|{'tools/user_acceptance/u07-precision-review.json'}
    if changed!=expected:return None
    if any(old['source_sha256'].get(n)!=pin['before'] or new['source_sha256'].get(n)!=pin['after'] for n,pin in spec['files'].items()):return None
    if any(old[k]!=new[k] for k in ('runtime','budgets','feature_tools','groups','cpu_inventory','preserved_evidence')):return None
    user_changes={n for n in set(old['user_files'])|set(new['user_files']) if old['user_files'].get(n)!=new['user_files'].get(n)}
    if not user_changes-tracking_only_changes(old,new,user_changes,set(e.git('ls-files','-z').split('\0')))<=expected:return None
    if cp.get('safety_hold'):raise RuntimeError('U07_REVIEW_REQUIRES_NO_SAFETY_HOLD')
    u.verify_cached(cp)
    for attempt in cp['attempts']:
        folder=Path(attempt['directory']);r=e.read_json(folder/'controller.json')
        if not (attempt['status']=='FINISHED' and attempt['receipt_sha256']==e.file_hash(folder/'controller.json') and
                r['source_integrity'] is True and r['cleanup_verified'] is True and r['cleanup']['active_processes']==0 and
                r.get('cleanup_error') is None and not r.get('monitor_reason')):
            raise RuntimeError('U07_REVIEW_OWNERSHIP_OR_CLEANUP_UNVERIFIED')
    cpu=sorted(n for n,r in cp['cpu_results'].items() if r['status']=='PASS' and r.get('source_valid',True))
    gpu=sorted(n for n,r in cp['results'].items() if n!='CPU_REGRESSIONS' and r['status']=='PASS' and r.get('source_valid',True))
    if len(cpu)!=14 or len(gpu)!=11 or cp['results']['U07']['status']!='FAIL':return None
    return dict(kind='U07_MIXED_PRECISION_OUTPUT_ONLY',baseline_plan_sha256=e.digest(old),retained_cpu_pass=cpu,
        retained_gpu_pass=gpu,revalidation_cpu=[],revalidation_gpu=['U07'],
        impact='Uniform-output v2 and legacy evidence unchanged; v3 records actual FP32/low output while native low operator execution stays mandatory.',
        evidence='Existing controller/journal/artifact/runtime hashes and verified owned cleanup retained; no new GPU execution during preparation.')
