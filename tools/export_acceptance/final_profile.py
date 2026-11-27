"""E13 profile for the existing owned acceptance engine, never a second runner.

Preparation is stdlib-only. Native imports occur only in the user-started owned
GPU worker. Existing untracked implementation/evidence files are read-only.
"""
from __future__ import annotations
import argparse
import ast
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'outputs/export-final-acceptance-20261127'
HISTORY=REPO/'outputs/export-acceptance-20261127'
NATIVE_PYTHON=None
CPU_ONLY=False
GROUPS={'01-admission':['ADMISSION'],
 '03-native-engine':['E03_TRT','E04_TRT'],
 '04-precision-engine':['E07_TRT_FP16','E08_TRT_QDQ'],
 '05-deployment-validation':['E09_TRT_COHORT','E10_TRT_BENCHMARK','E11_TRT_COMPARISON'],
 '06-profile-closure':['E03_TRT_DYNAMIC','E04_TRT_FP16']}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def canonical(path):return os.path.normcase(str(Path(path).resolve(strict=True)))

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def reference_runtime(site):
    """CPU equivalence still needs one unambiguous ORT distribution."""
    site=Path(site).resolve();module=site/'onnxruntime/__init__.py'
    versions={d.metadata.get('Name','').lower().replace('_','-'):d.version
              for d in importlib.metadata.distributions(path=[str(site)])
              if d.metadata.get('Name','').lower().replace('_','-') in ('onnxruntime','onnxruntime-gpu')}
    issues=[]
    if len(versions)!=1:issues.append('ONE_ORT_DISTRIBUTION_REQUIRED')
    if not module.is_file():issues.append('ORT_REFERENCE_MODULE_MISSING')
    extensions=list((site/'onnxruntime/capi').glob('onnxruntime_pybind11_state*.pyd'))
    if len(extensions)!=1:issues.append('ONE_ORT_REFERENCE_EXTENSION_REQUIRED')
    version=next(iter(versions.values()),None)
    # 1.20.2 is the already admitted shared ORT used by the preserved campaign;
    # 1.23.2 is the declared optional-dependency profile. Never force an upgrade
    # merely to reuse the former's exact, source/binary-bound reference evidence.
    if version not in ('1.20.2','1.23.2'):issues.append('UNSUPPORTED_ORT_REFERENCE_VERSION')
    return dict(status='BLOCKED' if issues else 'READY_FOR_ADMISSION',site=str(site),distributions=versions,
                version=version,module_sha256=sha(module) if module.is_file() else None,
                extension_identities=[dict(path=canonical(p),sha256=sha(p),file_id=p.stat().st_ino,bytes=p.stat().st_size) for p in extensions],
                issues=issues,runtime_verified=False)

def closure(python):
    """Collect every mismatch; no partial allowlist is usable for execution."""
    python=Path(python).resolve(strict=True)
    if python.name.lower()!='python.exe':raise ValueError('selected Windows python.exe required')
    site=python.parent/'Lib/site-packages'
    if not site.is_dir():raise ValueError('selected Conda site-packages missing')
    distributions=list(importlib.metadata.distributions(path=[str(site)]))
    versions={};issues=[];warnings=[];files=[]
    for dist in distributions:
        name=dist.metadata.get('Name','').lower().replace('_','-')
        versions.setdefault(name,[]).append(dist.version)
        if name not in ('torch','nvidia-cuda-runtime-cu12','nvidia-cuda-nvrtc-cu12','nvidia-nvjitlink-cu12') and not name.startswith('tensorrt'):continue
        for member in dist.files or []:
            if Path(str(member)).suffix.lower() not in ('.dll','.pyd'):continue
            if name=='torch' and not Path(str(member)).name.lower().startswith(('cudart','cublas','cudnn','nvrtc','nvjit','torch_cuda')):continue
            p=Path(dist.locate_file(member));problems=[]
            try:
                resolved=p.resolve(strict=True)
                if not resolved.is_relative_to(site.resolve()):raise ValueError('RECORD member escapes selected site')
                for part in (p,*p.parents):
                    if part==site:break
                    if part.is_symlink() or os.name=='nt' and part.lstat().st_file_attributes&0x400:raise ValueError('reparse RECORD member')
                before=resolved.stat();actual=sha(resolved);after=resolved.stat();entry=member.hash
                if member.size!=before.st_size:problems.append('RECORD_SIZE')
                if entry is None or entry.mode!='sha256' or base64.urlsafe_b64encode(bytes.fromhex(actual)).decode().rstrip('=')!=entry.value:problems.append('RECORD_SHA256')
                if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):problems.append('FILE_CHANGED')
                files.append(dict(distribution=name,version=dist.version,member=str(member),path=canonical(resolved),sha256=actual,
                    record_sha256=entry.value if entry else None,record_bytes=member.size,bytes=before.st_size,file_id=before.st_ino,
                    problems=problems))
            except (OSError,ValueError) as exc:problems.append(str(exc))
            issues.extend(name+': '+str(member)+': '+p for p in problems)
    for name in ('torch','onnx','tensorrt-cu12','tensorrt-cu12-libs','tensorrt-cu12-bindings','nvidia-cuda-runtime-cu12'):
        if len(versions.get(name,[]))!=1:issues.append('MISSING_OR_DUPLICATE_DISTRIBUTION: '+name)
    trt=[versions.get(n,[]) for n in ('tensorrt-cu12','tensorrt-cu12-libs','tensorrt-cu12-bindings')]
    if any(v!=['10.13.3.9'] for v in trt):issues.append('EXACT_TENSORRT_10_13_3_9_REQUIRED')
    remnants=[p.name for p in site.iterdir() if p.name.startswith(('~umpy','-umpy','~numpy'))]
    if len(versions.get('numpy',[]))!=1 or remnants:issues.append('NUMPY_METADATA_AMBIGUOUS')
    if 'onnxruntime' in versions and 'onnxruntime-gpu' in versions:issues.append('ORT_CPU_GPU_NAMESPACE_CONFLICT')
    cuda=[r for r in files if Path(r['path']).name.lower()=='cudart64_12.dll']
    chosen=next((r for r in cuda if r['distribution']=='torch' and not r['problems']),None)
    alternatives=[r for r in cuda if r is not chosen]
    if chosen is None:issues.append('TORCH_CUDART_NOT_VERIFIED')
    if chosen and any(r['sha256']!=chosen['sha256'] for r in alternatives):
        warnings.append('DIFFERENT_CUDART_SAME_BASENAME: only Torch cudart is admitted; TRT loader alternatives must fail before engine build')
    if not any(r['distribution']=='tensorrt-cu12-libs' for r in files):issues.append('TRT_NATIVE_LIBRARIES_MISSING')
    if not any(r['distribution']=='tensorrt-cu12-bindings' for r in files):issues.append('TRT_BINDINGS_MISSING')
    allowed={r['path']:r['sha256'] for r in files if not r['problems'] and r not in alternatives}
    ids={r['path']:dict(file_id=r['file_id'],bytes=r['bytes']) for r in files if r['path'] in allowed}
    return dict(status='BLOCKED' if issues else 'READY_FOR_ADMISSION',python=str(python),python_sha256=sha(python),site=str(site),
        versions={n:v[0] for n,v in versions.items() if len(v)==1},distribution_versions=versions,issues=issues,warnings=warnings,
        files=files,numpy_remnants=remnants,cuda_selection=dict(selected=chosen,alternatives=alternatives,
        policy='Torch RECORD-backed cudart only; no alternate-runtime allowlisting'),
        allowed_binaries={} if issues else allowed,binary_identities={} if issues else ids,native_imports=False,runtime_verified=False)

def reuse_cpu_evidence():
    """Reuse source-bound engineering proof, explicitly distinct from owned acceptance."""
    retained=REPO/'outputs/e13-prepare-verified-20261127/prepare.json'
    baseline=read(retained);selected={p:s for p,s in baseline['source_sha256'].items()
        if p.startswith('hyso/export/') or p.startswith('tests/test_export_development') or p in ('hyso/cli/export.py','hyso/cli/main.py')}
    changed=[p for p,s in selected.items() if not (REPO/p).is_file() or sha(REPO/p)!=s]
    files=[REPO/'outputs'/n for n in ('e13-development-tests-20261127.xml','e13-development-corrections-20261127.xml','e13-development-cli-final-20261127.xml')]
    cases={}
    for path in files:
        for case in ET.parse(path).iter('testcase'):
            key='tests/'+case.attrib['classname'].split('.')[-1]+'.py::'+case.attrib['name']
            passed=not any(case.find(tag) is not None for tag in ('failure','error','skipped'))
            cases[key]=dict(status='PASS' if passed else 'FAIL',evidence=str(path),evidence_sha256=sha(path),kind='CPU engineering JUnit',owned_acceptance=False)
    if changed:
        for value in cases.values():value.update(status='NOT_TESTED',reason='Prior proof source changed',affected_sources=changed)
    return dict(cases=cases,source_sha256=selected,changed_sources=changed,pass_count=sum(v['status']=='PASS' for v in cases.values()),
                files={str(p):sha(p) for p in [retained,*files]})

def matrix(reused,readiness):
    rows=[]
    coverage={'E01':'availability, overwrite and optional dependencies','E02':'real ONNX, CLI conversion, integrity',
      'E05':'TorchScript reload','E06':'ExportedProgram reload','E07':'mixed FP16 graph, dtype/provenance and drift',
      'E08':'dynamic INT8, static QDQ, calibration and negative inputs','E09':'numerical metrics and labelled classification quality',
      'E10':'small CPU deployment measurements','E11':'compatible comparison and speedup rejection',
      'E12':'API/CLI/Advisor, four languages, JSON/Markdown/HTML and saved records'}
    for feature,scope in coverage.items():rows.append(dict(id=feature+'_CPU',device='cpu',status='PASS' if reused['pass_count']==45 else 'NOT_TESTED',
        scope=scope,evidence_kind='reused CPU engineering; not a new owned/GPU receipt',evidence=list(reused['files'])))
    for name in [f for g,fs in GROUPS.items() for f in fs if g!='01-admission']:
        rows.append(dict(id=name,device='cuda',status='BLOCKED' if readiness['status']=='BLOCKED' else 'NOT_TESTED',
            scope='owned native conversion/load/inference/equivalence/preservation/integrity/cleanup',precision='int8_qdq' if 'QDQ' in name else 'fp16' if 'FP16' in name else 'fp32'))
    rows.append(dict(id='ADMISSION',device='cuda',status='BLOCKED' if readiness['status']=='BLOCKED' else 'NOT_TESTED',
        scope='fresh selected-interpreter UUID/driver/Job/resource admission; prior admission is historical'))
    rows.append(dict(id='FP16_CPU_P35_COVERAGE',device='cpu',status='BLOCKED',classification='unsupported_attestation_profile',
        product_numeric_defect=False,reason='ORT CPU may rewrite half operations into internal casts; complete original-node P35 mapping is not proven; numerical metrics remain separate'))
    return rows

def cpu_selection(cpu_only=False):
    p=REPO/('tests/test_export_blocker_resolution.py' if cpu_only else 'tests/test_export_final_readiness.py')
    return {p.relative_to(REPO).as_posix()+'::'+n.name:dict(nodeid=p.relative_to(REPO).as_posix()+'::'+n.name,
        path=p.relative_to(REPO).as_posix(),profile='cpu',test_source_sha256=sha(p))
        for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_cpu_scope_' if cpu_only else 'test_')}

def configure():
    from tools.export_acceptance import runner as previous
    engine=previous.engine
    baseline_request=previous.BASE_REQUEST;baseline_progress=previous.BASE_PROGRESS
    baseline_record=previous.BASE_RECORD_CPU
    def snapshot():return previous.snapshot()
    def prepare_cpu():
        history=read(HISTORY/'plan.json');reused=reuse_cpu_evidence();cpu=engine.runtime_plan()
        reference=reference_runtime(history['runtime']['ort_site'])
        runtime=dict(history['runtime'],python=cpu['cpu_python'],site=cpu['cpu_site'],
                     runtime_paths=list(dict.fromkeys([cpu['cpu_site'],history['runtime']['ort_site']])),binary_expectations={})
        cpu_python=Path(cpu['cpu_python'])
        identity=dict(python=cpu['cpu_python'],python_sha256=sha(cpu_python) if cpu_python.is_file() else None,
                      site=cpu['cpu_site'],torch=cpu['torch_cpu_distribution'],runtime_paths=cpu['runtime_paths'],missing=cpu['missing'])
        # Retain historical native manifests as evidence only; no native group is scheduled.
        tools=dict(history['feature_tools'])
        source=snapshot();inventory=cpu_selection(True)
        preserved=dict(reused['files'])
        preserved.update({str(p):sha(p) for p in HISTORY.iterdir() if p.is_file() and p.suffix=='.json'})
        descriptor=REPO/'outputs/release-acceptance-20261125/source-runtime.json'
        preserved[str(descriptor)]=sha(descriptor)
        issues=list(cpu['missing'])+reference['issues']
        readiness=dict(status='BLOCKED' if issues else 'READY_FOR_CPU_ACCEPTANCE',issues=issues,
                       native_imports=False,tensorrt_in_scope=False,tensorrt_status='BLOCKED_UNVERIFIED')
        rows=[r for r in matrix(reused,{'status':'BLOCKED'}) if r['device']=='cpu']
        plan=dict(history,campaign='export-final-acceptance',scope='cpu-only',source_sha256=source,
            source_fingerprint=engine.digest(source),user_files=engine.user_snapshot(),runtime=runtime,feature_tools=tools,
            groups={},cpu_inventory=inventory,preserved_evidence=preserved,dependencies={},dependency_readiness=readiness,
            reused_cpu=reused,acceptance_matrix=rows,cpu_runtime=identity,reference_runtime=reference,
            scenario_counts=dict(CPU=len(inventory),GPU=0,admission=0,reused_engineering_CPU=reused['pass_count']),
            historical_export_summary=dict(cpu=4,gpu=3,source='read-only prior campaign; no current-runtime promotion'),
            automatic_retries=0,concurrency=1,full_cuda_acceptance=False)
        return store_plan(plan)
    def prepare():
        if CPU_ONLY:return prepare_cpu()
        readiness=closure(NATIVE_PYTHON);history=read(HISTORY/'plan.json');checkpoint=read(HISTORY/'checkpoint.json')
        if checkpoint.get('safety_hold'):raise RuntimeError('HISTORICAL_SAFETY_HOLD_MUST_BE_RESOLVED')
        reused=reuse_cpu_evidence();runtime=dict(history['runtime']);runtime.update(python=readiness['python'],site=readiness['site'],versions=readiness['versions'])
        selected_ort=Path(readiness['site'])/'onnxruntime/__init__.py'
        runtime['ort_site']=readiness['site'] if selected_ort.is_file() else history['runtime']['ort_site']
        runtime['runtime_paths']=list(dict.fromkeys([readiness['site'],runtime['ort_site']]))
        reference=reference_runtime(runtime['ort_site'])
        native_files={r['path']:r for r in readiness['files']}
        binary={}
        for name,row in history['runtime']['binary_expectations'].items():
            p=Path(row['path']);base=Path(history['runtime']['site'])
            if p.is_relative_to(base):p=Path(readiness['site'])/p.relative_to(base)
            key=canonical(p)
            binary[name]=dict(path=key,sha256=native_files[key]['sha256'] if key in native_files else sha(p))
        runtime['binary_expectations']=binary
        source=snapshot();old_root=ROOT.resolve()
        if old_root==HISTORY.resolve() or old_root.is_relative_to(HISTORY.resolve()):raise RuntimeError('HISTORICAL_CAMPAIGN_OUTPUT_FORBIDDEN')
        preserved={str(p):sha(p) for p in HISTORY.iterdir() if p.is_file() and p.suffix=='.json'}
        preserved.update(reused['files'])
        cpu=engine.runtime_plan()
        cpu_descriptor=REPO/'outputs/release-acceptance-20261125/source-runtime.json'
        preserved[str(cpu_descriptor)]=sha(cpu_descriptor)
        cpu_python=Path(cpu['cpu_python'])
        cpu_identity=dict(python=cpu['cpu_python'],python_sha256=sha(cpu_python) if cpu_python.is_file() else None,
                          site=cpu['cpu_site'],torch=cpu['torch_cpu_distribution'],runtime_paths=cpu['runtime_paths'],missing=cpu['missing'])
        tools=dict(history['feature_tools']);tools['tensorrt']=dict(distributions={n:v for n,v in readiness['versions'].items() if n.startswith('tensorrt')},
            allowed_binaries=readiness['allowed_binaries'],binary_identities=readiness['binary_identities'],runtime_verified=False,status=readiness['status'])
        if readiness['status']=='READY_FOR_ADMISSION':
            from tools.cuda_acceptance.native_identity import ort_inventory,_shared
            tools['ort']=ort_inventory(runtime['ort_site'],runtime['site'])
            tools['ort']['version']=reference['version']
            banned={r['path'] for r in readiness['cuda_selection']['alternatives']}
            selected_cudart=readiness['cuda_selection']['selected']['path']
            tools['ort']['allowed_binaries']={p:s for p,s in tools['ort']['allowed_binaries'].items()
                if p not in banned and (Path(p).name.lower()!='cudart64_12.dll' or p==selected_cudart)}
            tools['export_cuda']=_shared.torch_compiler_inventory(runtime['site'],runtime['versions']['torch'],binary['torch_cuda.dll'])
            tools['export_cuda']['allowed_binaries'].update(readiness['allowed_binaries'])
            tools['export_cuda']['binary_identities'].update(readiness['binary_identities'])
        plan=dict(history,campaign='export-final-acceptance',source_sha256=source,source_fingerprint=engine.digest(source),
            user_files=engine.user_snapshot(),runtime=runtime,feature_tools=tools,groups=GROUPS,cpu_inventory=cpu_selection(),
            preserved_evidence=preserved,dependencies=readiness['versions'],dependency_readiness=readiness,reused_cpu=reused,
            acceptance_matrix=matrix(reused,readiness),scenario_counts=dict(CPU=len(cpu_selection()),GPU=9,admission=1,reused_engineering_CPU=reused['pass_count']),
            cpu_runtime=cpu_identity,
            reference_runtime=reference,
            historical_export_summary=dict(cpu=4,gpu=3,source='read-only prior campaign; no current-runtime promotion'),
            automatic_retries=0,concurrency=1,full_cuda_acceptance=False)
        return store_plan(plan)
    def store_plan(plan):
        source=plan['source_sha256'];runtime=plan['runtime'];tools=plan['feature_tools']
        if ROOT.resolve()==HISTORY.resolve() or ROOT.resolve().is_relative_to(HISTORY.resolve()):
            raise RuntimeError('HISTORICAL_CAMPAIGN_OUTPUT_FORBIDDEN')
        ROOT.mkdir(parents=True,exist_ok=True)
        cp=engine.read_json(ROOT/'checkpoint.json',dict(schema_version=1,results={},cpu_results={},attempts=[],history=[]))
        old=engine.read_json(ROOT/'plan.json')
        if old and old.get('scope','native')!=plan.get('scope','native'):raise RuntimeError('CAMPAIGN_SCOPE_CHANGED: use a separate acceptance root')
        if old and engine.digest(old)!=engine.digest(plan):
            if any(a['status']=='RUNNING' for a in cp['attempts']):raise RuntimeError('INTERRUPTED_ATTEMPT_MUST_BE_RESOLVED_BEFORE_REPREPARATION')
            engine.atomic_json(ROOT/'history'/('plan-'+engine.digest(old)+'.json'),dict(plan=old,checkpoint=cp))
            if old['source_sha256']!=source:
                for table in ('results','cpu_results'):
                    for row in cp.get(table,{}).values():row['source_valid']=False
            elif old['runtime']!=runtime or old['feature_tools']!=tools:
                for name,row in cp['results'].items():
                    if name!='CPU_REGRESSIONS':row['source_valid']=False
        cp['plan_sha256']=engine.digest(plan)
        engine.atomic_json(ROOT/'plan.json',plan);engine.atomic_json(ROOT/'checkpoint.json',cp)
        progress(cp,plan,'PREPARED',tests_executed_by_preparation=0)
        return plan,cp
    def request(folder,work,group,plan,inventory=None):
        req=baseline_request(folder,work,group,plan,inventory);req['campaign_id']='export-acceptance'
        if inventory:
            req['cpu_ort_manifest']=plan['feature_tools']['ort'];req['argv']=[str(Path(__file__).with_name('cpu_worker.py')),str(folder/'request.json')]
        else:req['argv']=[str(Path(__file__).resolve()),'--worker',str(folder/'request.json')]
        return req
    def progress(cp,plan,status,*args,**kwargs):
        value=baseline_progress(cp,plan,status,*args,**kwargs)
        rows=json.loads(json.dumps(plan['acceptance_matrix']))
        for item in rows:
            if item['id'] in cp['results']:
                actual=cp['results'][item['id']]
                item.update(status=actual['status'] if actual.get('source_valid',True) else 'NOT_TESTED',evidence=actual.get('evidence'))
        rows.extend(dict(id=node,device='cpu',status=cp.get('cpu_results',{}).get(node,{}).get('status','NOT_TESTED'),
                    scope='new E13 closure/dispatch/safety regression') for node in plan['cpu_inventory'])
        value.update(campaign='export-final-acceptance',dependency_readiness=plan['dependency_readiness']['status'],
            dependency_issues=plan['dependency_readiness']['issues'],reused_CPU_engineering=plan['reused_cpu']['pass_count'],
            reference_readiness=plan['reference_runtime']['status'],cpu_runtime_missing=plan['cpu_runtime']['missing'],
            historical_export=plan['historical_export_summary'],native_safety_hold=cp.get('safety_hold'),acceptance_matrix=rows)
        engine.atomic_json(ROOT/'progress.json',value);return value
    engine.ROOT=ROOT;engine.FEATURES={} if CPU_ONLY else GROUPS;engine.GPU_UUID=read(HISTORY/'plan.json')['gpu_uuid']
    engine.prepare=prepare;engine.snapshot=snapshot;engine.request_for=request;engine.progress=progress;engine.record_cpu=baseline_record
    engine.verify_cached=previous.user.BASE_VERIFY_CACHED;engine.pending_cases=previous.pending_cases
    engine.cpu_group_ready=previous.user.cpu_group_ready;engine.normal_cpu_completion=previous.user.normal_cpu_completion
    from tools.cuda_acceptance import validation
    from tools.export_acceptance.validation import validate_gpu_attempt
    validation.verify_feature=verify_feature;validation.validate_gpu_attempt=validate_gpu_attempt
    return engine

def verify_feature(feature,data,runtime):
    from tools.export_acceptance.evidence import verify_feature as base
    if feature=='E03_TRT_DYNAMIC':
        base('E03_TRT',data,runtime)
        ext=data.get('development_extension',{}).get('validation',{})
        if ext.get('numerical',{}).get('status')!='PASS' or len(ext.get('batches',[]))!=3:raise ValueError('all three admitted dynamic batch profiles required')
        shapes=[r['numerical']['outputs'][0]['reference_shape'][0] for r in ext['batches']]
        if shapes!=[1,2,4]:raise ValueError('dynamic execution must cover min/opt/max batch values')
    elif feature=='E04_TRT_FP16':
        base('E04_TRT',data,runtime)
        if data.get('precision',{}).get('requested')!='fp16' or data.get('conversion')!='pytorch_to_onnx_to_tensorrt':raise ValueError('explicit E04 FP16 chain required')
    else:base(feature,data,runtime)

def gpu_group(ctx):
    """Called only after the existing owned worker has admitted/probed its GPU."""
    from hyso import export as ex
    from hyso.export._worker import verify_loaded
    from tools.cuda_acceptance.feature_evidence import SafetyViolation
    from tools.export_acceptance.scenarios import gpu_group as base_group,model
    import torch
    plan=ctx.plan;allowed=dict(plan['feature_tools']['ort']['allowed_binaries'])
    allowed.update({r['path']:r['sha256'] for r in plan['runtime']['binary_expectations'].values()})
    allowed.update(plan['feature_tools']['tensorrt']['allowed_binaries']);deps=plan['feature_tools']['export_cuda'];allowed.update(deps['allowed_binaries'])
    # The third-party TRT loader may preload a second cudart by absolute path.
    # Inspect actual modules before any engine/Builder call and fail closed.
    import tensorrt
    ids={p:dict(file_id=Path(p).stat().st_ino,bytes=Path(p).stat().st_size) for p in allowed}
    native=ex.NativeConfig(ctx.uuid,allowed,runtime_search_path=plan['feature_tools']['ort']['site'],dependency_identities=deps['binary_identities'])
    try:verify_loaded(native,ids)
    except ex.ExportError as exc:raise SafetyViolation('PRE_ENGINE_DLL_SELECTION: '+str(exc)) from exc
    base_group(ctx)
    def run(name):
        folder=ctx.folder/name;folder.mkdir(exist_ok=False);x=torch.ones(2,3);m=model()
        config=ex.ExportConfig(atol=1e-3,rtol=1e-2) if name=='E04_TRT_FP16' else ex.ExportConfig()
        if name=='E04_TRT_FP16':result=ex.pytorch_tensorrt(m,x,folder/'model.engine',native=native,precision='fp16',config=config)
        else:
            middle=ex.onnx(m,x,folder/'source.onnx',dynamic_shapes={'input0':{0:dict(name='batch',min=1,max=4)}},config=ex.ExportConfig(validate_runtime=False))
            result=ex.tensorrt(middle,folder/'model.engine',native=native,example_input={'input0':x},profiles={'input0':dict(min=[1,3],opt=[2,3],max=[4,3])})
            measured=ex.validate(ex.Runtime(m),ex.Runtime(result,device='cuda',native=native,evidence_directory=folder),[torch.ones(n,3) for n in (1,2,4)])
            from hyso.export.contracts import seal
            data=result.to_dict();data['development_extension']=dict(validation=measured);result=seal(data)
        data=result.to_dict();(folder/'raw.json').write_text(json.dumps(data,allow_nan=False),encoding='utf-8');return data
    for name in ('E03_TRT_DYNAMIC','E04_TRT_FP16'):ctx.run(name,lambda name=name:run(name),scope='E13 bounded admitted profiles; no generic GPU certification')

def owned_worker(request_path):
    """Reuse the existing GPU worker and its Job latch, journal and terminal seal."""
    import runpy
    request=read(request_path);folder=Path(request_path).parent
    # Outer campaign admission uses the existing receipt, not the nested runtime
    # worker's HYSO_RUNTIME_ADMITTED_PID environment convention.
    from tools.cuda_acceptance.worker import process_identity,digest
    admission=read(folder/'admission-receipt.json');admit=read(folder/'admit.json');ids=admission['identities']
    if not (ids['exact_job_membership'] and ids['interpreter']==process_identity(os.getpid()) and
            ids['nonce_sha256']==digest(request['token']) and admit['pid']==os.getpid() and
            admit['token']==request['token'] and admit['readback_verified'] and
            admission.get('no_breakaway') and admission.get('kill_on_close') and admission.get('limits_readback')):
        raise RuntimeError('OWNED_WORKER_ADMISSION_REQUIRED')
    # Avoid recursion when the existing worker imports its dispatch functions.
    from tools.export_acceptance import scenarios,evidence
    original_group=scenarios.gpu_group;original_verify=evidence.verify_feature
    def dispatch(ctx):
        scenarios.gpu_group=original_group
        try:return gpu_group(ctx)
        finally:scenarios.gpu_group=dispatch
    def verified(feature,data,runtime):
        evidence.verify_feature=original_verify
        try:return verify_feature(feature,data,runtime)
        finally:evidence.verify_feature=verified
    scenarios.gpu_group=dispatch;evidence.verify_feature=verified
    sys.argv=[str(Path(__file__).with_name('gpu_worker.py')),str(request_path)]
    runpy.run_path(sys.argv[0],run_name='__main__')

def main(argv=None):
    global ROOT,NATIVE_PYTHON,CPU_ONLY
    sys.path.insert(0,str(REPO))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path);parser.add_argument('--runtime-python',type=Path)
    parser.add_argument('--conda-env',default='hyso_trt');parser.add_argument('--campaign-root',type=Path)
    parser.add_argument('--cpu-only',action='store_true')
    args,remaining=parser.parse_known_args(argv)
    if args.worker:return owned_worker(args.worker)
    CPU_ONLY=args.cpu_only
    ROOT=(args.campaign_root or (REPO/'outputs/export-final-cpu-acceptance-20261127' if CPU_ONLY else ROOT)).resolve()
    if '--status' in remaining:
        path=ROOT/'progress.json';print(json.dumps(read(path) if path.is_file() else dict(status='NOT_PREPARED'),indent=2));return 0
    NATIVE_PYTHON=args.runtime_python
    if NATIVE_PYTHON is None and not CPU_ONLY:
        command=['conda','env','list','--json']
        result=subprocess.run(command,capture_output=True,text=True,check=True,timeout=30,shell=os.name=='nt')
        matches=[Path(p)/'python.exe' for p in json.loads(result.stdout)['envs'] if Path(p).name==args.conda_env]
        if len(matches)!=1:raise RuntimeError('SELECTED_CONDA_ENVIRONMENT_NOT_UNIQUE')
        NATIVE_PYTHON=matches[0]
    engine=configure()
    if '--prepare-only' not in remaining:
        if not (ROOT/'plan.json').is_file():
            sys.argv=[sys.argv[0],'--prepare-only']
            prepared=engine.main()
            if prepared:return prepared
        plan=read(ROOT/'plan.json')
        if plan.get('scope','native')!=('cpu-only' if CPU_ONLY else 'native'):raise RuntimeError('CAMPAIGN_SCOPE_CHANGED')
        if not CPU_ONLY and canonical(NATIVE_PYTHON)!=canonical(plan['runtime']['python']):raise RuntimeError('PREPARED_INTERPRETER_CHANGED')
        current=plan['dependency_readiness'] if CPU_ONLY else closure(NATIVE_PYTHON)
        if not CPU_ONLY and current['status']!='READY_FOR_ADMISSION':
            engine.atomic_json(ROOT/'readiness-stop.json',current)
            state=engine.read_json(ROOT/'progress.json',{})
            state.update(status='BLOCKED',dependency_readiness='BLOCKED',dependency_issues=current['issues'],
                         GPU_executed=False,native_attempt_started=False)
            engine.atomic_json(ROOT/'progress.json',state)
            print(json.dumps(dict(status='BLOCKED',issues=current['issues'],GPU_executed=False),indent=2));return 1
        if current!=plan['dependency_readiness']:raise RuntimeError('DEPENDENCY_INVENTORY_CHANGED: PrepareOnly required')
        reference=reference_runtime(plan['runtime']['ort_site'])
        if reference!=plan['reference_runtime']:raise RuntimeError('ORT_REFERENCE_CHANGED: PrepareOnly required')
        if reference['status']!='READY_FOR_ADMISSION':raise RuntimeError('ORT_REFERENCE_BLOCKED: '+repr(reference['issues']))
        cpu=engine.runtime_plan();cpu_python=Path(cpu['cpu_python'])
        cpu_identity=dict(python=cpu['cpu_python'],python_sha256=sha(cpu_python) if cpu_python.is_file() else None,
                          site=cpu['cpu_site'],torch=cpu['torch_cpu_distribution'],runtime_paths=cpu['runtime_paths'],missing=cpu['missing'])
        if cpu_identity!=plan['cpu_runtime']:raise RuntimeError('CPU_RUNTIME_SELECTION_CHANGED: PrepareOnly required')
        if cpu_identity['missing']:raise RuntimeError('CPU_RUNTIME_DEPENDENCIES_BLOCKED: '+repr(cpu_identity['missing']))
    sys.argv=[sys.argv[0],*remaining]
    if '--prepare-only' in remaining:
        import io
        from contextlib import redirect_stdout
        captured=io.StringIO()
        with redirect_stdout(captured):code=engine.main()
        summary=json.loads(captured.getvalue());plan=read(ROOT/'plan.json')
        summary.update(dependency_readiness=plan['dependency_readiness']['status'],reused_CPU_engineering=plan['reused_cpu']['pass_count'],
                       gpu_native_cases=plan['scenario_counts']['GPU'],admission_cases=plan['scenario_counts']['admission'],automatic_retries=0)
        print(json.dumps(summary));return code
    return engine.main()

if __name__=='__main__':raise SystemExit(main())
