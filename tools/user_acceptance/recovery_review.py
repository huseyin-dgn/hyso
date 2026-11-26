"""Narrow, hash-pinned source transition for the U13/runtime + cleanup defect."""
from pathlib import Path


def review_transition(old,new,cp):
    from tools.user_acceptance import runner as u
    e=u.engine
    prior=old.get('source_impact_review')
    if prior and old['source_sha256']==new['source_sha256'] and all(old[k]==new[k] for k in
            ('runtime','budgets','feature_tools','groups','cpu_inventory','preserved_evidence','user_files')):
        return prior
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
