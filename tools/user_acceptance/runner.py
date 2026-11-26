"""User acceptance adapter for the existing serial, sealed campaign controller.

No product or CUDA acceptance evidence is modified. Preparation imports no Torch.
"""
from pathlib import Path
import ast
import json
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.cuda_acceptance import runner as engine

ROOT = REPO / 'outputs/final-user-acceptance-20261126'
CUDA_ROOT = REPO / 'outputs/cuda-functional-acceptance-20261126'
GROUPS = {
    '01-admission': ['ADMISSION'],
    '02-inference': ['U01', 'U02', 'U03', 'U04'],
    '03-sweeps': ['U07', 'U08'],
    '04-profiling': ['U09'],
    '05-training': ['U05', 'U06'],
    '06-new-features': ['U12'],
    '07-deployment': ['U13'],
}
BASE_SNAPSHOT = engine.snapshot
BASE_PRESERVED = engine.preserved
BASE_REQUEST = engine.request_for
BASE_PROGRESS = engine.progress
BASE_RECORD_CPU = engine.record_cpu
BASE_VERIFY_CACHED = engine.verify_cached


def pending_cases(nodes,results,retry):
    # Exact failures are retried only by request; unstarted independent journeys
    # must still run in that same resume. Valid historical PASS stays skipped.
    return [n for n in nodes if n in retry or n not in results]


def cpu_group_ready(cp, plan):
    rows=cp.get('cpu_results', {})
    return set(rows)==set(plan['cpu_inventory']) and all(r.get('source_valid',True) for r in rows.values())


def normal_cpu_completion(group, receipt, cp, plan):
    return (group=='CPU_REGRESSIONS' and receipt['exit_code']==1 and not receipt.get('monitor_reason')
        and receipt.get('source_integrity') is True and receipt.get('cleanup_verified') is True
        and receipt.get('cleanup',{}).get('active_processes')==0 and cpu_group_ready(cp,plan)
        and any(r['status']=='FAIL' for r in cp['cpu_results'].values()))


def progress(cp,plan,status,*args,**kwargs):
    value=BASE_PROGRESS(cp,plan,status,*args,**kwargs)
    value.update(campaign='final-user-acceptance',product_scope='frozen',scenario_counts=plan['scenario_counts'])
    engine.atomic_json(ROOT/'progress.json',value)
    rows=[]
    for a in cp['attempts']:
        for path in sorted(Path(a['directory']).glob('U*/result.json')):
            row=engine.read_json(path)
            rows.append({k:row.get(k) for k in ('scenario','public_api','model','device','expected','actual',
                'status','raw_evidence_path','product_defect','support_limitation')})
    engine.atomic_json(ROOT/'user-results.json',dict(attempt_rows=rows,full_cuda_acceptance=False,
        note='All attempts retained; authoritative latest statuses are in checkpoint.json.'))
    return value


def record_cpu(cp,folder,request,receipt,plan):
    if (folder/'user-safety-stop.json').exists():raise RuntimeError('USER_OWNED_SAFETY_STOP: '+str(folder/'user-safety-stop.json'))
    BASE_RECORD_CPU(cp,folder,request,receipt,plan)
    evidence=folder/'cpu-results.json';data=engine.read_json(evidence);data['scenario_artifacts']={}
    for node in request['selected_nodes']:
        name=node.rsplit('::',1)[-1].removeprefix('test_')
        artifacts={str(p):engine.file_hash(p) for p in (folder/name).rglob('*') if p.is_file()}
        if not artifacts:raise RuntimeError('USER_SCENARIO_ARTIFACTS_MISSING: '+name)
        data['scenario_artifacts'][node]=artifacts
    engine.atomic_json(evidence,data)
    for node in request['selected_nodes']:cp['cpu_results'][node]['sha256']=engine.file_hash(evidence)


def verify_cached(cp):
    BASE_VERIFY_CACHED(cp)
    for row in cp.get('cpu_results',{}).values():
        if not row.get('source_valid',True):continue
        for artifacts in engine.read_json(row['evidence'])['scenario_artifacts'].values():
            for path,sha in artifacts.items():
                if engine.file_hash(path)!=sha:raise RuntimeError('USER_RAW_EVIDENCE_CHANGED: '+path)


def snapshot():
    values = BASE_SNAPSHOT()
    paths = list((REPO / 'tools/user_acceptance').rglob('*.py'))
    paths += [REPO / 'tests/test_user_acceptance_scenarios.py',
              REPO / 'tests/test_user_acceptance_runner.py',
              REPO / 'docs/guides/final-user-acceptance.md']
    paths += [REPO/'tools/user_acceptance/u13-cleanup-review.json']
    values.update({p.relative_to(REPO).as_posix(): engine.file_hash(p) for p in paths if p.is_file()})
    return values


def preserved():
    values = BASE_PRESERVED()
    # Seal ALL prior CUDA attempts, including failures and raw measurements.
    values.update({str(p): engine.file_hash(p) for p in CUDA_ROOT.rglob('*') if p.is_file()})
    return values


def cpu_selection():
    path = REPO / 'tests/test_user_acceptance_scenarios.py'
    relative = path.relative_to(REPO).as_posix()
    return {relative + '::' + n.name: dict(nodeid=relative + '::' + n.name,
        path=relative, profile='cpu', test_source_sha256=engine.file_hash(path))
        for n in ast.parse(path.read_text(encoding='utf-8')).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith('test_U')}


def prepare():
    baseline = engine.read_json(CUDA_ROOT / 'plan.json')
    cp_cuda = engine.read_json(CUDA_ROOT / 'checkpoint.json')
    if not baseline or any(cp_cuda['results'].get(f, {}).get('status') != 'PASS'
            or not cp_cuda['results'][f].get('source_valid', True)
            for features in baseline['groups'].values() for f in features):
        raise RuntimeError('COMPLETED_CUDA_BASELINE_REQUIRED')
    if not all(r['status'] == 'PASS' and r.get('source_valid', True)
               for r in cp_cuda.get('cpu_results', {}).values()) or len(cp_cuda.get('cpu_results', {})) != 658:
        raise RuntimeError('658_CPU_PASS_BASELINE_REQUIRED')
    runtime = baseline['runtime']
    if not Path(runtime['python']).is_file(): raise RuntimeError('GPU_RUNTIME_MISSING')
    for item in runtime['binary_expectations'].values():
        if engine.file_hash(item['path']) != item['sha256']: raise RuntimeError('GPU_BINARY_CHANGED')
    from tools.cuda_acceptance.native_identity import ort_inventory
    if ort_inventory(runtime['ort_site'], runtime['site']) != baseline['feature_tools']['ort']:
        raise RuntimeError('ORT_PRE_ADMISSION_INVENTORY_CHANGED')
    source = snapshot()
    plan = dict(schema_version=1, campaign='final-user-acceptance',
        source_sha256=source, source_fingerprint=engine.digest(source), commit=engine.git('rev-parse', 'HEAD'),
        gpu_uuid=baseline['gpu_uuid'], runtime=runtime, budgets=baseline['budgets'],
        groups=GROUPS, cpu_inventory=cpu_selection(), preserved_evidence=preserved(),
        user_files=engine.user_snapshot(), feature_tools=baseline['feature_tools'],
        full_cuda_acceptance=False, deferred=baseline['deferred'], concurrency=1, automatic_retries=0,
        scenario_counts=dict(cpu=14, cuda=11, admission=1), product_scope='frozen',
        WDDM=baseline['WDDM'])
    old = engine.read_json(ROOT / 'plan.json')
    cp = engine.read_json(ROOT / 'checkpoint.json', dict(schema_version=1, results={}, attempts=[], history=[]))
    from tools.user_acceptance.recovery_review import review_transition
    review=review_transition(old,plan,cp) if old else None
    if review:plan['source_impact_review']=review
    if old and engine.digest(old) != engine.digest(plan):
        import uuid
        engine.atomic_json(ROOT / 'history' / ('plan-' + uuid.uuid4().hex + '.json'),
            dict(plan=old, checkpoint=cp,progress=engine.read_json(ROOT/'progress.json')))
        cp['history'].append(dict(reason='USER_SCOPE_SOURCE_TRANSITION', results=cp['results'], cpu_results=cp.get('cpu_results', {})))
        # No implicit rerun; exact Revalidate is required even after preparation.
        for table in ('results', 'cpu_results'):
            cp[table] = {n: dict(r, source_valid=False) if not review or n not in review['retained_cpu_pass'] else r
                         for n, r in cp.get(table, {}).items()}
        if cp.get('cpu_results'):cp['results']['CPU_REGRESSIONS']=engine.cpu_summary(cp,plan)
    cp['plan_sha256'] = engine.digest(plan)
    engine.atomic_json(ROOT / 'plan.json', plan)
    engine.atomic_json(ROOT / 'checkpoint.json', cp)
    engine.progress(cp, plan, 'PREPARED', tests_executed_by_preparation=0,
        preserved_counts=dict(source_PASS=8536, infrastructure_PASS=55, package_PASS=36,
                              support_scope_PASS=59, cuda_CPU_PASS=658, cuda_GPU_PASS=36))
    return plan, cp


def request_for(folder, work, group, plan, inventory=None):
    request = BASE_REQUEST(folder, work, group, plan, inventory)
    request['campaign_id'] = 'final-user-acceptance'
    if inventory:request['cpu_ort_manifest']=plan['feature_tools']['ort']
    request['argv'] = [str(Path(__file__).with_name('cpu_worker.py' if inventory else 'gpu_worker.py')),
                       str(folder / 'request.json')]
    return request


def configure():
    engine.ROOT = ROOT
    engine.FEATURES = GROUPS
    engine.snapshot = snapshot
    engine.prepare = prepare
    engine.request_for = request_for
    engine.progress = progress
    engine.record_cpu = record_cpu
    engine.verify_cached = verify_cached
    engine.cpu_group_ready = cpu_group_ready
    engine.normal_cpu_completion = normal_cpu_completion
    engine.pending_cases = pending_cases
    from tools.cuda_acceptance import validation
    from tools.user_acceptance.evidence import verify_feature
    validation.verify_feature = verify_feature
    # Keep the existing parent-side planned ORT binding in addition to U13 checks.
    original_validation = validation.validate_gpu_attempt
    def validate(folder, request, receipt):
        values, critical = original_validation(folder, request, receipt)
        if not critical and values.get('U13', {}).get('status') == 'PASS':
            data = engine.read_json(folder / 'U13.json')['data']
            if data['observations']['planned_runtime'] != request['gpu_plan']['feature_tools']['ort']:
                return {}, 'ORT_PLANNED_RUNTIME_BINDING'
        return values, critical
    validation.validate_gpu_attempt = validate


if __name__ == '__main__':
    configure()
    try: raise SystemExit(engine.main())
    except Exception as exc:
        if ROOT.is_dir(): engine.atomic_json(ROOT / 'progress.json', dict(
            engine.read_json(ROOT / 'progress.json', {}), status='BLOCKED', reason=str(exc), full_cuda_acceptance=False))
        raise
