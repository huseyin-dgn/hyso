"""Separate U15-U18 CPU/offline campaign using the existing owned controller."""
from pathlib import Path
import ast
import sys
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from tools.cuda_acceptance import runner as engine
from tools.user_acceptance import runner as user
from tools.product_acceptance.inventory import build_inventory,USER

ROOT=REPO/'outputs/product-experience-acceptance-20261126'
BASE_SNAPSHOT=user.snapshot
BASE_REQUEST=engine.request_for
BASE_PROGRESS=engine.progress
INVENTORY=REPO/'docs/acceptance/product-experience-inventory.json'


def snapshot():
    values=BASE_SNAPSHOT()
    paths=list((REPO/'tools/product_acceptance').glob('*.py'))
    paths += [INVENTORY,REPO/'docs/guides/product-experience-acceptance.md',
        REPO/'tests/test_product_experience.py',REPO/'tests/test_product_experience_scenarios.py',
        REPO/'tests/test_product_acceptance_runner.py']
    values.update({p.relative_to(REPO).as_posix():engine.file_hash(p) for p in paths if p.is_file()})
    return values


def preserved():
    # Existing completed CPU/CUDA/U01-U14 evidence stays byte-for-byte read-only.
    plan=engine.read_json(USER/'plan.json')
    values=dict(plan['preserved_evidence'])
    values.update({str(p):engine.file_hash(p) for p in USER.rglob('*') if p.is_file()})
    return values


def cpu_selection():
    p=REPO/'tests/test_product_experience_scenarios.py';name=p.relative_to(REPO).as_posix()
    return {name+'::'+n.name:dict(nodeid=name+'::'+n.name,path=name,profile='cpu',test_source_sha256=engine.file_hash(p))
        for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in ('test_U15','test_U16','test_U17','test_U18')}


def verify_baseline():
    p=engine.read_json(USER/'plan.json');cp=engine.read_json(USER/'checkpoint.json')
    if engine.digest(p)!=cp['plan_sha256']:raise RuntimeError('U01_U14_PLAN_SEAL_MISMATCH')
    expected={f for names in p['groups'].values() for f in names}
    if (len(cp.get('cpu_results',{}))!=14 or len(expected)!=12 or cp.get('safety_hold') or
        any(r['status']!='PASS' or not r.get('source_valid',True) for r in cp['cpu_results'].values()) or
        any(cp['results'].get(f,{}).get('status')!='PASS' or not cp['results'][f].get('source_valid',True) for f in expected)):
        raise RuntimeError('COMPLETED_26_CASE_USER_BASELINE_REQUIRED')
    from tools.cuda_acceptance import validation
    from tools.user_acceptance.evidence import verify_feature
    validation.verify_feature=verify_feature
    user.verify_cached(cp)
    # U13 uses the shared ORT checker plus this request-to-plan binding.
    record=cp['results']['U13'];folder=Path(record['evidence']).parent
    data=engine.read_json(record['evidence'])['data']['observations']
    if data['planned_runtime']!=engine.read_json(folder/'request.json')['gpu_plan']['feature_tools']['ort']:
        raise RuntimeError('U13_ORT_PLANNED_RUNTIME_BINDING')
    return p,cp


def prepare():
    baseline,_=verify_baseline();ROOT.mkdir(parents=True,exist_ok=True)
    inventory=build_inventory()
    if engine.read_json(INVENTORY)!=inventory:engine.atomic_json(INVENTORY,inventory)
    sources=snapshot()
    plan=dict(schema_version=1,campaign='product-experience',commit=engine.git('rev-parse','HEAD'),
        source_sha256=sources,source_fingerprint=engine.digest(sources),runtime={},budgets=baseline['budgets'],
        groups={},cpu_inventory=cpu_selection(),preserved_evidence=preserved(),user_files=engine.user_snapshot(),
        full_cuda_acceptance=False,deferred=baseline['deferred'],concurrency=1,automatic_retries=0,
        scenario_counts={'CPU':4,'GPU':0},real_IDE_tested=False,
        external_type_checker='not installed/not run; annotations, literals and runtime resolution checked separately')
    cp=engine.read_json(ROOT/'checkpoint.json',dict(schema_version=1,results={},attempts=[],history=[]))
    old=engine.read_json(ROOT/'plan.json')
    if old and engine.digest(old)!=engine.digest(plan):
        import uuid
        engine.atomic_json(ROOT/'history'/('plan-'+uuid.uuid4().hex+'.json'),dict(plan=old,checkpoint=cp))
        cp['history'].append(dict(reason='PRODUCT_SOURCE_TRANSITION',results=cp['results'],cpu_results=cp.get('cpu_results',{})))
        for table in ('results','cpu_results'):cp[table]={n:dict(r,source_valid=False) for n,r in cp.get(table,{}).items()}
    cp['plan_sha256']=engine.digest(plan)
    engine.atomic_json(ROOT/'plan.json',plan);engine.atomic_json(ROOT/'checkpoint.json',cp)
    progress(cp,plan,'PREPARED',tests_executed_by_preparation=0)
    return plan,cp


def request_for(folder,work,group,plan,inventory=None):
    if group!='CPU_REGRESSIONS' or not inventory:raise RuntimeError('PRODUCT_CAMPAIGN_IS_CPU_OFFLINE_ONLY')
    request=BASE_REQUEST(folder,work,group,plan,inventory)
    request['campaign_id']='product-experience'
    request['argv']=[str(REPO/'tools/user_acceptance/cpu_worker.py'),str(folder/'request.json')]
    return request


def progress(cp,plan,status,*args,**kwargs):
    value=BASE_PROGRESS(cp,plan,status,*args,**kwargs)
    value.update(campaign='product-experience',scenarios=['U15','U16','U17','U18'],
        scenario_counts=plan['scenario_counts'],admission='NOT_REQUIRED_CPU_OFFLINE_ONLY',
        prior_acceptance_untouched=True,real_IDE_tested=False,GPU_required=False)
    engine.atomic_json(ROOT/'progress.json',value)
    return value


def configure():
    engine.ROOT=ROOT;engine.FEATURES={};engine.snapshot=snapshot;engine.prepare=prepare
    engine.request_for=request_for;engine.progress=progress
    engine.record_cpu=user.record_cpu;engine.verify_cached=user.verify_cached
    engine.pending_cases=user.pending_cases;engine.normal_cpu_completion=user.normal_cpu_completion


if __name__=='__main__':
    configure()
    try:raise SystemExit(engine.main())
    except Exception as exc:
        if ROOT.exists():engine.atomic_json(ROOT/'progress.json',dict(engine.read_json(ROOT/'progress.json',{}),status='BLOCKED',reason=str(exc),full_cuda_acceptance=False))
        raise
