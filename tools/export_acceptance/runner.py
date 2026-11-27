"""ExportAcceptance manager: stdlib preparation, frozen evidence, no auto retry."""
from pathlib import Path
import ast
import importlib.metadata
import shutil
import sys
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from tools.cuda_acceptance import runner as engine
from tools.user_acceptance import runner as user
from tools.product_acceptance import runner as product

ROOT=REPO/'outputs/export-acceptance-20261127'
GROUPS={'01-admission':['ADMISSION'],'02-export-runtime':['E02_CUDA','E05_CUDA'],
        '03-native-engine':['E03_TRT','E04_TRT']}
BASE_REQUEST=engine.request_for;BASE_PROGRESS=engine.progress
BASE_RECORD_CPU=engine.record_cpu
NATIVE_PYTHON=None

def runtime_inventory(python):
    # Load the stdlib-only module by path: importing hyso would import Torch.
    import importlib.util
    spec=importlib.util.spec_from_file_location('export_runtime_inventory',REPO/'hyso/export/dependencies.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.runtime_inventory(python)

def snapshot():
    values=product.snapshot()
    paths=list((REPO/'tools/export_acceptance').glob('*.py'))+list((REPO/'hyso/export').glob('*.py'))
    paths+=list((REPO/'examples/export').rglob('*'))
    paths+=list((REPO/'tests').glob('test_export_*.py'))
    paths+=[REPO/'docs/guides/model-export.md',REPO/'hyso/_native_identity.py']
    values.update({p.relative_to(REPO).as_posix():engine.file_hash(p) for p in paths if p.is_file()})
    return values

def preserved():
    plan=engine.read_json(product.ROOT/'plan.json')
    values=dict(plan['preserved_evidence'])
    values.update({str(p):engine.file_hash(p) for p in product.ROOT.rglob('*') if p.is_file()})
    for path,sha in values.items():
        if engine.file_hash(path)!=sha:raise RuntimeError('PRIOR_ACCEPTANCE_EVIDENCE_CHANGED: '+path)
    return values

def cpu_selection():
    p=REPO/'tests/test_export_acceptance_scenarios.py';name=p.relative_to(REPO).as_posix()
    expected={'test_E01','test_E02_CPU','test_E05_CPU','test_E06_EP'}
    nodes={name+'::'+n.name:dict(nodeid=name+'::'+n.name,path=name,profile='cpu',test_source_sha256=engine.file_hash(p))
        for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in expected}
    if len(nodes)!=len(expected):raise RuntimeError('EXACT_FOUR_ACTIVE_CPU_EXPORT_CASES_REQUIRED')
    return nodes

def dependencies(site):
    dist={d.metadata.get('Name','').lower():d.version for d in importlib.metadata.distributions(path=[str(site)])}
    return {n:dist.get(n) for n in ('torch','onnx','onnxscript','tensorrt','tensorrt-cu12','torch-tensorrt')}

def native_inventory(site):
    """Installed TensorRT distribution files only, before admission/import/build."""
    python=Path(site).parent.parent/'python.exe'
    if python.is_file():return runtime_inventory(python)['tensorrt']
    rows={};versions={}
    for dist in importlib.metadata.distributions(path=[str(site)]):
        name=dist.metadata.get('Name','').lower()
        if not name.startswith('tensorrt'):continue
        versions[name]=dist.version
        for member in dist.files or []:
            p=Path(dist.locate_file(member))
            if p.suffix.lower() not in ('.dll','.pyd') or not p.is_file():continue
            import os
            rows[os.path.normcase(str(p.resolve()))]=engine.file_hash(p)
    return dict(distributions=versions,allowed_binaries=rows,runtime_verified=False)

def reviewed_dll_transition(old,plan,cp):
    """Exact reviewed source delta only; no general permission to reuse PASS."""
    review=engine.read_json(ROOT/'dll-manifest-source-review.json',{}) or engine.read_json(ROOT/'dll-diagnostic-source-review.json',{})
    if not review or review.get('old_plan_sha256')!=engine.digest(old):return False
    if review.get('new_sources')!=plan['source_sha256'] or review.get('new_user_files')!=plan['user_files']:return False
    excluded={'source_sha256','source_fingerprint','user_files'}
    if 'new_export_cuda' in review:
        if review['new_export_cuda']!=plan['feature_tools'].get('export_cuda'):return False
        if {k:v for k,v in old['feature_tools'].items() if k!='export_cuda'}!={k:v for k,v in plan['feature_tools'].items() if k!='export_cuda'}:return False
        excluded.add('feature_tools')
    if {k:v for k,v in old.items() if k not in excluded}!={k:v for k,v in plan.items() if k not in excluded}:return False
    allowed=set(review.get('reviewed_paths',[]))
    for table in ('source_sha256','user_files'):
        delta={p for p in set(old[table])|set(plan[table]) if old[table].get(p)!=plan[table].get(p)}
        if not delta<=allowed:return False
    if not review.get('frozen_evidence'):return False
    for path,sha in review['frozen_evidence'].items():
        if not Path(path).is_file() or engine.file_hash(path)!=sha:raise RuntimeError('DLL_REVIEW_HISTORICAL_EVIDENCE_CHANGED: '+path)
    user.verify_cached(cp)
    return True


def prepare():
    product.verify_baseline()
    old_product=engine.read_json(product.ROOT/'plan.json');done=engine.read_json(product.ROOT/'checkpoint.json')
    if (engine.digest(old_product)!=done['plan_sha256'] or done.get('safety_hold') or len(done.get('cpu_results',{}))!=4 or
        any(r['status']!='PASS' or not r.get('source_valid',True) for r in done['cpu_results'].values())):raise RuntimeError('COMPLETED_U15_U18_BASELINE_REQUIRED')
    user.verify_cached(done)
    baseline=engine.read_json(user.CUDA_ROOT/'plan.json')
    runtime=baseline['runtime']
    for row in runtime['binary_expectations'].values():
        if engine.file_hash(row['path'])!=row['sha256']:raise RuntimeError('CUDA_PREPARATION_BINARY_CHANGED')
    from tools.cuda_acceptance.native_identity import ort_inventory
    if ort_inventory(runtime['ort_site'],runtime['site'])!=baseline['feature_tools']['ort']:raise RuntimeError('ORT_PREPARATION_INVENTORY_CHANGED')
    from tools.cuda_acceptance.native_identity import _shared
    export_cuda=_shared.torch_compiler_inventory(runtime['site'],runtime['versions']['torch'],runtime['binary_expectations']['torch_cuda.dll'])
    sources=snapshot()
    plan=dict(schema_version=1,campaign='export-acceptance',commit=engine.git('rev-parse','HEAD'),source_sha256=sources,
        source_fingerprint=engine.digest(sources),runtime=runtime,gpu_uuid=baseline['gpu_uuid'],budgets=baseline['budgets'],
        feature_tools=dict(baseline['feature_tools'],tensorrt=native_inventory(runtime['site']),export_cuda=export_cuda),groups=GROUPS,cpu_inventory=cpu_selection(),preserved_evidence=preserved(),
        user_files=engine.user_snapshot(),full_cuda_acceptance=False,deferred=baseline['deferred'],WDDM=baseline['WDDM'],
        concurrency=1,automatic_retries=0,scenario_counts={'CPU':4,'GPU':4,'admission':1},
        export_deferred=['E05 native LibTorch C++ compilation/runtime','E06 AOTInductor native compilation/runtime'],
        dependencies=dependencies(runtime['site']),compiler_paths={n:shutil.which(n) for n in ('cmake','cl','g++','nvcc')},
        native_sdk_policy='No SDK download/install; missing native dependencies are BLOCKED; workspace is not total VRAM')
    if NATIVE_PYTHON is not None:
        from tools.cuda_acceptance.native_identity import ort_inventory
        selected=runtime_inventory(NATIVE_PYTHON);site=Path(selected['site'])
        binary={}
        for name,row in runtime['binary_expectations'].items():
            original=Path(row['path'])
            if original.is_relative_to(Path(runtime['site'])):
                target=site/original.relative_to(Path(runtime['site']))
            else:target=original
            if not target.is_file():raise RuntimeError('SELECTED_RUNTIME_BINARY_MISSING: '+name)
            import os
            binary[name]=dict(path=os.path.normcase(str(target.resolve())),sha256=engine.file_hash(target))
        native_runtime=dict(python=selected['python'],site=selected['site'],versions=selected['versions'],binary_expectations=binary,
            ort_site=runtime['ort_site'])
        native_ort=ort_inventory(runtime['ort_site'],site)
        compiler=_shared.torch_compiler_inventory(site,selected['versions']['torch'],binary['torch_cuda.dll'])
        plan['native_runtime']=dict(runtime=native_runtime,feature_tools=dict(plan['feature_tools'],
            tensorrt=selected['tensorrt'],ort=native_ort,export_cuda=compiler),issues=selected['issues'])
    ROOT.mkdir(parents=True,exist_ok=True)
    cp=engine.read_json(ROOT/'checkpoint.json',dict(schema_version=1,results={},attempts=[],history=[]))
    old=engine.read_json(ROOT/'plan.json')
    if old and engine.digest(old)!=engine.digest(plan):
        import uuid
        engine.atomic_json(ROOT/'history'/('plan-'+uuid.uuid4().hex+'.json'),dict(plan=old,checkpoint=cp))
        cp['history'].append(dict(reason='EXPORT_PREPARATION_TRANSITION',results=cp['results'],cpu_results=cp.get('cpu_results',{})))
        # Retain PASS across dependency-only preparation changes when source,
        # CPU/ORT runtime, budgets, ownership/user files and preserved seals agree.
        stable=all(old.get(k)==plan.get(k) for k in ('source_sha256','runtime','budgets','user_files','preserved_evidence')) and old.get('feature_tools',{}).get('ort')==plan['feature_tools']['ort']
        reviewed=not stable and reviewed_dll_transition(old,plan,cp)
        if reviewed:
            # Default P35 selection, CPU exporters, runtime, admission and budgets
            # are unchanged. GPU diagnostics do not promote failed cases to PASS.
            for name in ('E02_CUDA','E05_CUDA'):
                if name in cp['results']:cp['results'][name]=dict(cp['results'][name],source_valid=False)
        elif not stable:
            for key in ('cpu_results','results'):cp[key]={n:dict(r,source_valid=False) for n,r in cp.get(key,{}).items()}
        else:
            if old.get('feature_tools',{}).get('tensorrt')!=plan['feature_tools']['tensorrt']:
                for name in ('E03_TRT','E04_TRT'):
                    if name in cp['results']:cp['results'][name]=dict(cp['results'][name],source_valid=False)
            if old.get('compiler_paths')!=plan['compiler_paths']:
                node='tests/test_export_acceptance_scenarios.py::test_E05_CPP'
                if node in cp.get('cpu_results',{}):cp['cpu_results'][node]=dict(cp['cpu_results'][node],source_valid=False)
    cp['plan_sha256']=engine.digest(plan)
    engine.atomic_json(ROOT/'plan.json',plan);engine.atomic_json(ROOT/'checkpoint.json',cp)
    progress(cp,plan,'PREPARED',tests_executed_by_preparation=0)
    return plan,cp

def request_for(folder,work,group,plan,inventory=None):
    if group in ('03-native-engine','04-precision-engine','05-deployment-validation') and plan.get('native_runtime'):
        selected=plan['native_runtime']
        if selected['feature_tools']['tensorrt']['status']!='READY_FOR_ADMISSION':
            raise RuntimeError('TENSORRT_DEPENDENCIES_BLOCKED: '+repr(selected['issues']))
        plan=dict(plan,runtime=selected['runtime'],feature_tools=selected['feature_tools'])
    request=BASE_REQUEST(folder,work,group,plan,inventory)
    request['campaign_id']='export-acceptance'
    if inventory:request['cpu_ort_manifest']=plan['feature_tools']['ort']
    request['argv']=[str(Path(__file__).with_name('cpu_worker.py' if inventory else 'gpu_worker.py')),str(folder/'request.json')]
    return request

def progress(cp,plan,status,*args,**kwargs):
    value=BASE_PROGRESS(cp,plan,status,*args,**kwargs)
    value.update(campaign='export-acceptance',scenario_counts=plan['scenario_counts'],native_safety_hold=cp.get('safety_hold'),historical_acceptance='658 CPU / 36 CUDA / 30 user PASS preserved, not automatic current-source certification')
    engine.atomic_json(ROOT/'progress.json',value);return value

def record_cpu(cp,folder,request,receipt,plan):
    marker=engine.read_json(folder/'user-safety-stop.json')
    if not marker:
        user.record_cpu(cp,folder,request,receipt,plan)
        hold=cp.get('safety_hold',{})
        clean=(receipt.get('cleanup_verified') is True and receipt.get('source_integrity') is True and receipt.get('cleanup',{}).get('active_processes')==0 and not receipt.get('monitor_reason') and receipt.get('exit_code') in (0,1))
        if not clean:
            held=[n for n in request['selected_nodes'] if cp['cpu_results'][n]['status']!='PASS'] or list(request['selected_nodes'])
            for node in held:cp['cpu_results'][node].update(status='BLOCKED',reason='ABNORMAL_CPU_NATIVE_TERMINATION')
            cp['results']['CPU_REGRESSIONS']=engine.cpu_summary(cp,plan)
            cp['safety_hold']=dict(features=held,reason='ABNORMAL_CPU_NATIVE_TERMINATION',fresh_admission_required=True)
        elif hold and set(hold['features'])<=set(request['selected_nodes']) and all(n in plan['cpu_inventory'] for n in hold['features']):
            # Manager required exact explicit revalidation before this request.
            # The shared reader already verified the new sealed/clean receipt.
            cp.pop('safety_hold')
        return
    # Preserve sealed pre-crash PASS cases, but keep unstarted cases absent.
    # This path cannot pass through user.record_cpu's immediate stop guard.
    if not receipt.get('cleanup_verified') or receipt.get('source_integrity') is not True or receipt.get('cleanup',{}).get('active_processes')!=0:
        raise RuntimeError('OWNED_CPU_CLEANUP_OR_SOURCE_UNVERIFIED')
    BASE_RECORD_CPU(cp,folder,request,receipt,plan)
    evidence=folder/'cpu-results.json';data=engine.read_json(evidence);data['scenario_artifacts']={}
    target='tests/test_export_acceptance_scenarios.py::test_'+marker['scenario']
    for node in request['selected_nodes']:
        name=node.rsplit('::',1)[-1].removeprefix('test_')
        files={str(p):engine.file_hash(p) for p in (folder/name).rglob('*') if p.is_file()}
        if not files:
            cp['cpu_results'].pop(node,None);continue
        if node==target:files[str(folder/'user-safety-stop.json')]=engine.file_hash(folder/'user-safety-stop.json')
        data['scenario_artifacts'][node]=files
    engine.atomic_json(evidence,data)
    for node in data['scenario_artifacts']:cp['cpu_results'][node]['sha256']=engine.file_hash(evidence)
    if target not in cp['cpu_results']:raise RuntimeError('NATIVE_STOP_CASE_BINDING_MISSING')
    cp['cpu_results'][target].update(status='BLOCKED',reason=marker)
    cp['results']['CPU_REGRESSIONS']=engine.cpu_summary(cp,plan)
    cp['safety_hold']=dict(features=[target],reason=marker,fresh_admission_required=True)

def normal_cpu_completion(group,receipt,cp,plan):
    return not cp.get('safety_hold') and user.normal_cpu_completion(group,receipt,cp,plan)


def pending_cases(nodes,results,retry):
    pending=user.pending_cases(nodes,results,retry)
    return [n for n in pending if not retry or n in retry]

def configure():
    engine.ROOT=ROOT;engine.FEATURES=GROUPS;engine.snapshot=snapshot;engine.prepare=prepare
    engine.request_for=request_for;engine.progress=progress;engine.record_cpu=record_cpu
    engine.verify_cached=user.verify_cached;engine.pending_cases=pending_cases
    engine.cpu_group_ready=user.cpu_group_ready;engine.normal_cpu_completion=normal_cpu_completion
    from tools.cuda_acceptance import validation
    from tools.export_acceptance.evidence import verify_feature
    validation.verify_feature=verify_feature
    from tools.export_acceptance.validation import validate_gpu_attempt
    validation.validate_gpu_attempt=validate_gpu_attempt

if __name__=='__main__':
    if '--final' in sys.argv:
        from tools.export_acceptance.final_profile import main as final_main
        raise SystemExit(final_main([arg for arg in sys.argv[1:] if arg!='--final']))
    # Select a separate sealed campaign rather than rewriting the historical
    # baseline when using an independently installed native runtime.
    import argparse
    selector=argparse.ArgumentParser(add_help=False)
    selector.add_argument('--runtime-python',type=Path)
    selector.add_argument('--campaign-root',type=Path)
    selector.add_argument('--development-native',action='store_true')
    selected,remaining=selector.parse_known_args()
    if selected.runtime_python:
        if selected.campaign_root is None or selected.campaign_root.resolve()==ROOT.resolve():
            raise RuntimeError('SEPARATE_CAMPAIGN_ROOT_REQUIRED_FOR_NEW_INTERPRETER')
        NATIVE_PYTHON=selected.runtime_python;ROOT=selected.campaign_root.resolve()
    elif selected.campaign_root:ROOT=selected.campaign_root.resolve()
    if selected.development_native:
        if not selected.runtime_python:raise RuntimeError('EXPLICIT_SEPARATE_RUNTIME_REQUIRED_FOR_DEVELOPMENT_NATIVE')
        GROUPS=dict(GROUPS,**{'04-precision-engine':['E07_TRT_FP16','E08_TRT_QDQ'],
            '05-deployment-validation':['E09_TRT_COHORT','E10_TRT_BENCHMARK','E11_TRT_COMPARISON']})
    sys.argv=[sys.argv[0],*remaining]
    configure()
    try:raise SystemExit(engine.main())
    except Exception as exc:
        if ROOT.exists():engine.atomic_json(ROOT/'progress.json',dict(engine.read_json(ROOT/'progress.json',{}),status='BLOCKED',reason=str(exc),full_cuda_acceptance=False))
        raise
