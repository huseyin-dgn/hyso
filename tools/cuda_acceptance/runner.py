"""Sealed, serial CUDA scope using the established owned-process controller.

Preparation and status are stdlib-only. The completed CPU campaigns are read-only.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import importlib.metadata
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(REPO/'tools/final_acceptance'))
from common import atomic_json,read_json,file_hash,digest,source_snapshot,user_snapshot,utc,git,tree_bytes
from campaign_controller import execute
from campaign_preflight import resources,runtime_plan,ExecutionState
from campaign_evidence import reconcile_attempt,terminal_phases
from campaign_operations import cleanup_owned
from common import read_journal

ROOT=REPO/'outputs/cuda-functional-acceptance-20261126'
CPU_ROOT=REPO/'outputs/release-acceptance-20261125/final-campaign'
GPU_UUID='GPU-9eec2f80-4c4e-a5cd-7091-03f0d013cf11'
FEATURES={
 '01-admission':['ADMISSION'],
 '02-inference':['Builder','GraphBuilder','P01','P02','P03','P04','P05','P09','P10','P11'],
 '03-sweeps':['P06','P07','P08','P26','P28'],
 '04-profiling':['P12','P13','P14','P15','P16','P17'],
 '05-training':['Trainer','P19','P20','P22','P23','P25'],
 '06-new-features':['P21','P24','P27','P29','P30'],
 '07-deployment':['P32','P34','P35'],
}
CPU_FILES=['tests/test_performance_dataloader.py','tests/test_quality_experiments.py',
 'tests/test_p29_memory.py','tests/test_p29_batched.py','tests/test_precision_quality_completion.py',
 'tests/test_trainer_classification.py','tests/test_trainer_updates.py','tests/test_native_execution.py',
 'tests/test_training_memory.py','tests/test_device_precision.py']
CPU_RUNTIME_CASES={'test_real_cpu_duration_chunks_reopen','test_cpu_model_state_modes_rng_preserved',
 'test_actual_video_capture_processing_conservation_drop_backpressure','test_stream_failure_bounded_cleanup',
 'test_stream_malformed_lifecycle_rejected','test_bad_capture_no_success'}
CPU_ISOLATED_CASES={'test_cuda_activation_crash_gate_precedes_allocator_and_hook_setup',
 'test_cuda_allocator_fraction_only_when_explicit'}


def snapshot():
    value=source_snapshot()
    paths=list((REPO/'tools/cuda_acceptance').rglob('*.py'))
    paths += [REPO/'tests/test_cuda_functional_completion.py',REPO/'docs/guides/cuda-functional-completion.md']
    value.update({p.relative_to(REPO).as_posix():file_hash(p) for p in paths if p.is_file()})
    return value


def preserved():
    names=['summary.json','checkpoint.json','package_acceptance.json','P01-P35-acceptance.json']
    files={str(CPU_ROOT/n):file_hash(CPU_ROOT/n) for n in names}
    for p in (CPU_ROOT/'support-scope-finalization-20261126').glob('attempt-*/targeted-result.json'):
        files[str(p)]=file_hash(p)
    return files


def gpu_info():
    p=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,memory.used,memory.free',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8,check=True)
    rows=[line.split(',') for line in p.stdout.splitlines() if line.strip()]
    row=next((list(map(str.strip,r)) for r in rows if r[1].strip()==GPU_UUID),None)
    if row is None:raise RuntimeError('GPU_IDENTITY_UNVERIFIED')
    return dict(name=row[0],uuid=row[1],driver=row[2],total_mib=int(row[3]),used_mib=int(row[4]),free_mib=int(row[5]))


def cpu_selection():
    inventory=read_json(CPU_ROOT/'checkpoint.json')['inventory']
    wanted={n:dict(r,test_source_sha256=file_hash(REPO/r['path'])) for n,r in inventory.items()
        if (r['path'] in CPU_FILES or (r['path']=='tests/test_runtime_analysis.py' and n.split('::')[-1].split('[')[0] in CPU_RUNTIME_CASES)
            or (r['path']=='tests/test_isolated_training_memory.py' and n.split('::')[-1].split('[')[0] in CPU_ISOLATED_CASES)) and not r.get('risk_class')}
    path=REPO/'tests/test_cuda_functional_completion.py'
    # New tests are a fixed reviewed CPU/offline module, not unknown inventory discovery.
    for n in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(n,ast.FunctionDef) and n.name.startswith('test_'):
            node=path.relative_to(REPO).as_posix()+'::'+n.name
            wanted[node]=dict(nodeid=node,path=path.relative_to(REPO).as_posix(),profile='cpu',test_source_sha256=file_hash(path))
    # Expanded parameter values in this existing test are a reviewed additive delta.
    base='tests/test_native_execution.py::test_CPU_binding_cannot_hide_CUDA_device_option'
    for feature in ('P24','P27','P29'):
        node=base+'['+feature+']';path='tests/test_native_execution.py'
        wanted[node]=dict(nodeid=node,path=path,profile='cpu',test_source_sha256=file_hash(REPO/path))
    if not wanted:raise RuntimeError('CPU_REGRESSION_SELECTION_EMPTY')
    return wanted


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    source=snapshot();python=Path('C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe')
    site=python.parent/'Lib/site-packages'
    versions={d.metadata.get('Name','').lower():d.version for d in importlib.metadata.distributions(path=[str(site)])}
    from tools.cuda_acceptance.native_identity import ort_inventory
    nsys=Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.5.1/target-windows-x64/nsys.exe')
    feature_tools=dict(nsys=dict(path=str(nsys),sha256=file_hash(nsys)) if nsys.is_file() else None,
        ort=ort_inventory('C:/Users/hdgn5/hyso-p35-runtime-env/Lib/site-packages',site))
    binaries=[site/'torch/lib/torch_cuda.dll',Path(os.environ['SYSTEMROOT'])/'System32/nvcuda.dll',*sorted((site/'torch/lib').glob('cudart64*.dll'))]
    if len(binaries)!=3 or any(not p.is_file() for p in binaries):raise RuntimeError('STATIC_CUDA_BINARY_INVENTORY_UNVERIFIED')
    binary_expectations={p.name:dict(path=os.path.normcase(str(p.resolve())),sha256=file_hash(p)) for p in binaries}
    old_sources=read_json(CPU_ROOT/'checkpoint.json')['baseline']['source_sha256']
    cold_dependencies=['hyso/__init__.py','hyso/quality.py','hyso/performance/__init__.py','hyso/performance/cold.py',
        'hyso/performance/_cold_worker.py','hyso/benchmark/runner.py','hyso/performance/stability.py','hyso/performance/streaming.py']
    case_tree=ast.parse((REPO/'tools/cuda_acceptance/cases.py').read_text(encoding='utf-8'))
    functions={n.name:ast.dump(n,include_attributes=False) for n in case_tree.body if isinstance(n,ast.FunctionDef)}
    group_functions=dict(zip(list(FEATURES)[1:],('inference','sweeps','profiling','training','new_features','deployment')))
    group_code={g:digest([functions[name],functions['cohort'],functions['require_cuda']]) for g,name in group_functions.items()}
    plan=dict(schema_version=1,source_sha256=source,source_fingerprint=digest(source),commit=git('rev-parse','HEAD'),
        gpu_uuid=GPU_UUID,runtime=dict(python=str(python),site=str(site),versions=versions,binary_expectations=binary_expectations,
            ort_site='C:/Users/hdgn5/hyso-p35-runtime-env/Lib/site-packages'),
        budgets=dict(process_bytes=3*1024**3,job_bytes=6*1024**3,max_processes=48,
            cuda_allocator_fraction=.15,ort_arena_bytes=256*1024**2,start_gpu_free_mib=4096,
            minimum_gpu_free_mib=1536,max_global_gpu_growth_mib=2048,group_timeout_seconds=900),
        groups=FEATURES,gpu_group_code_sha256=group_code,cpu_inventory=cpu_selection(),preserved_evidence=preserved(),user_files=user_snapshot(),feature_tools=feature_tools,
        deferred=['P18_hardware_counters','P23_CUDA_saved_activation','P31_CUDA_7200','P33_advanced_energy_thermal','native_failfast','allocation_stress'],
        p05_review=dict(historical_evidence_preserved=True,changed_import_dependencies=[p for p in cold_dependencies if old_sources.get(p)!=source.get(p)],
            decision='one_bounded_fresh_GPU_trial; original_evidence_not_current_source_acceptance'),full_cuda_acceptance=False,
        concurrency=1,automatic_retries=0,WDDM='Unknown per-process memory stays unknown. Torch allocator and global NVML observations are not physical per-process VRAM quotas.')
    old=read_json(ROOT/'plan.json');cp=read_json(ROOT/'checkpoint.json',dict(schema_version=1,results={},attempts=[],history=[]))
    from tools.cuda_acceptance.six_group_review import review_transition
    scope_review=review_transition(old,plan,cp,ROOT) if old else None
    if scope_review:plan['source_impact_review']=scope_review
    if old and digest(old)!=digest(plan):
        token=uuid.uuid4().hex;atomic_json(ROOT/'history'/('plan-'+token+'.json'),dict(plan=old,checkpoint=cp))
        from tools.cuda_acceptance.progress_recovery import recover_progress_failure
        recover_progress_failure(cp,old,plan,ROOT)
        changed={n for n in set(old['source_sha256'])|set(source) if old['source_sha256'].get(n)!=source.get(n)}
        local_cases={'tools/cuda_acceptance/cases.py','tools/cuda_acceptance/cold_probe.py','tools/cuda_acceptance/cold_worker.py'}
        all_gpu=any((n.startswith(('tools/cuda_acceptance/','tools/final_acceptance/')) and n not in local_cases) or n=='run_final_acceptance.ps1' for n in changed)
        affected=set()
        if changed & set(cold_dependencies):affected.add('02-inference')
        if 'tools/cuda_acceptance/cases.py' in changed:
            affected|={g for g,sha in group_code.items() if old.get('gpu_group_code_sha256',{}).get(g)!=sha}
        if changed & {'tools/cuda_acceptance/cold_probe.py','tools/cuda_acceptance/cold_worker.py'}:affected.add('02-inference')
        for n in changed:
            if n.startswith(('hyso/training/','hyso/performance/training','hyso/performance/dataloader','hyso/performance/gradients')):affected|={'05-training','06-new-features'}
            elif n.startswith(('hyso/quality','hyso/_quality_memory')):affected|={'03-sweeps','06-new-features'}
            elif n.startswith('hyso/performance/stream'):affected.add('06-new-features')
            elif n.startswith('hyso/'):all_gpu=True
        runtime_changed=old['runtime']!=plan['runtime']
        affected=set(FEATURES) if all_gpu or runtime_changed else affected
        invalid={f for g in affected for f in FEATURES[g]}
        if scope_review:invalid=set(scope_review['revalidation_features'])
        cp['history'].append(dict(reason='REVIEWED_SOURCE_TRANSITION',changed_files=sorted(changed),results=cp['results'],cpu_results=cp.get('cpu_results',{})))
        # Keep terminal evidence. A source-invalidated result is a visible hold,
        # never implicit permission to rerun training or promote historical PASS.
        cp['results']={n:dict(r,source_valid=False) if n in invalid else r for n,r in cp['results'].items()}
        cpu_dependencies_changed=any(n.startswith(('hyso/','tools/final_acceptance/')) for n in changed)
        if scope_review:cpu_dependencies_changed=False
        cp['cpu_results']={n:dict(r,source_valid=False) if cpu_dependencies_changed or
            old['cpu_inventory'].get(n)!=plan['cpu_inventory'].get(n) else r for n,r in cp.get('cpu_results',{}).items()}
        if cp.get('cpu_results'):cp['results']['CPU_REGRESSIONS']=cpu_summary(cp,plan)
    cp['plan_sha256']=digest(plan)
    atomic_json(ROOT/'plan.json',plan);atomic_json(ROOT/'checkpoint.json',cp)
    progress(cp,plan,'PREPARED',tests_executed_by_preparation=0,
        preserved_counts=dict(source_PASS=8536,infrastructure_PASS=55,package_PASS=36,support_scope_PASS=59))
    return plan,cp


def validate(plan):
    if snapshot()!=plan['source_sha256']:raise RuntimeError('SOURCE_CHANGED: run -CudaScope -PrepareOnly; prior evidence will be archived')
    if user_snapshot()!=plan['user_files']:raise RuntimeError('USER_FILES_CHANGED')
    for path,sha in plan['preserved_evidence'].items():
        if not Path(path).is_file() or file_hash(path)!=sha:raise RuntimeError('PRESERVED_EVIDENCE_CHANGED: '+path)


def request_for(folder,work,group,plan,inventory=None):
    from campaign_collection import collection_dependencies
    inventory=inventory or {}
    request=dict(schema_version=1,mode='tests',profile='cpu' if group=='CPU_REGRESSIONS' else 'cuda-functional',
        evidence_protocol='hyso.event-journal.v2',campaign_id='cuda-functional',attempt_id=folder.name,
        group=group,source_commit=plan['commit'],environment_id=digest(plan['runtime']),token=uuid.uuid4().hex,
        started_utc=utc(),work=str(work),scratch_root=str(work.parent),inventory=inventory,
        files=sorted({r['path'] for r in inventory.values()}),selected_nodes=list(inventory),known_nodes=list(inventory),
        blocked_nodes=[],timeout_seconds=1800 if inventory else plan['budgets']['group_timeout_seconds'],
        no_progress_timeout_seconds=300,initial_evidence_bytes=tree_bytes(ROOT)+tree_bytes(CPU_ROOT),
        collection_dependencies_sha256=collection_dependencies(plan['source_sha256']),
        conftest_sha256={n:s for n,s in plan['source_sha256'].items() if Path(n).name=='conftest.py'})
    if inventory:request['argv']=[str(REPO/'tools/final_acceptance/campaign_runner.py'),str(folder/'request.json')]
    else:
        request.update(cuda_authorized=True,gpu_plan=plan,features=FEATURES[group])
        request['argv']=[str(Path(__file__).with_name('worker.py')),str(folder/'request.json')]
    return request


def cpu_summary(cp,plan):
    rows={n:r for n,r in cp.get('cpu_results',{}).items() if n in plan['cpu_inventory']}
    complete=set(rows)==set(plan['cpu_inventory'])
    return dict(status='PASS' if complete and all(r['status']=='PASS' and r.get('source_valid',True) for r in rows.values()) else
        'FAIL' if any(r['status']=='FAIL' for r in rows.values()) else 'BLOCKED',
        recorded=len(rows),expected=len(plan['cpu_inventory']))


def retain_results(cp,key,values):
    target=cp.setdefault(key,{})
    for name,value in values.items():
        if name in target:
            if target[name]['status']=='PASS' and target[name].get('source_valid',True):raise RuntimeError('PASS_OVERWRITE_FORBIDDEN: '+name)
            cp['history'].append(dict(reason='EXPLICIT_DEFECT_REVALIDATION',case=name,previous=target[name]))
        target[name]=value


def pending_cases(nodes,results,retry):
    return [n for n in nodes if n in retry or (not retry and n not in results)]


def record_cpu(cp,folder,request,receipt,plan):
    # A controller callback may stop the child before it can emit its runtime.
    # Keep that original error primary; missing runtime is not its root cause.
    if receipt.get('monitor_reason')=='CONTROLLER_ERROR':
        error=receipt.get('error') or {}
        raise RuntimeError('CONTROLLER_ERROR: '+str(error.get('type'))+': '+str(error.get('message'))+
            '; original receipt: '+str(folder/'controller.json'))
    audit=reconcile_attempt(folder,baseline=dict(commit=request['source_commit'],source_sha256=plan['source_sha256']))
    if audit['global_errors']:raise RuntimeError('CPU_EVIDENCE_INVALID: '+repr(audit['global_errors']))
    selected=set(request['selected_nodes'])
    if not set(audit['results'])<=selected:raise RuntimeError('CPU_UNSELECTED_RESULT')
    evidence=folder/'cpu-results.json'
    atomic_json(evidence,dict(results=audit['results'],errors=audit['global_errors']))
    rows={}
    for name in selected:
        result=audit['results'].get(name)
        # Missing/aborted cases are terminal holds, never an automatic retry.
        status='PASS' if result and result['status']=='PASS' else 'FAIL' if result and result['status'] in ('FAIL','ERROR') else 'BLOCKED'
        rows[name]=dict(status=status,result=result,reason=None if result else 'NO_TERMINAL_CASE_EVIDENCE',
            evidence=str(evidence),sha256=file_hash(evidence),controller_sha256=file_hash(folder/'controller.json'),
            plan_sha256=digest(plan),source_valid=True)
    retain_results(cp,'cpu_results',rows)
    cp['results']['CPU_REGRESSIONS']=cpu_summary(cp,plan)


def verify_cached(cp):
    for name,row in {**cp.get('cpu_results',{}),**{n:r for n,r in cp['results'].items() if n!='CPU_REGRESSIONS'}}.items():
        if not row.get('source_valid',True):continue  # Historical only, never current acceptance.
        path=Path(row['evidence'])
        if not row.get('sha256') or file_hash(path)!=row['sha256']:raise RuntimeError('CACHED_EVIDENCE_HASH_MISMATCH: '+name)
        if name in cp.get('cpu_results',{}):
            if file_hash(path.parent/'controller.json')!=row['controller_sha256']:raise RuntimeError('CACHED_CPU_RECEIPT_CHANGED')
        elif row['status']=='PASS':
            from tools.cuda_acceptance.validation import validate_gpu_attempt
            values,critical=validate_gpu_attempt(path.parent,read_json(path.parent/'request.json'),read_json(path.parent/'controller.json'))
            if critical or values.get(name,{}).get('sha256')!=row['sha256']:raise RuntimeError('CACHED_GPU_ACCEPTANCE_INVALID: '+name)


def progress(cp,plan,status,started=None,folder=None,group=None,**extra):
    cpu={n:r for n,r in cp.get('cpu_results',{}).items() if n in plan['cpu_inventory']}
    gpu={n:r for n,r in cp['results'].items() if n!='CPU_REGRESSIONS'}
    active=None;journal_issues=[]
    if folder and (Path(folder)/'events.jsonl').is_file():
        rows,_=read_journal(Path(folder)/'events.jsonl')
        for row in rows:
            name=row.get('feature') or row.get('nodeid')
            if row['kind'] in ('feature_start','test_start'):active=name
            elif row['kind']=='phase' and row.get('when')=='teardown' and not row.get('is_subtest'):
                active=None
            elif row['kind']=='feature_end':
                active=None
                if name:
                    value=read_json(Path(folder)/(name+'.json'),{})
                    gpu[name]=dict(status=value.get('status','BLOCKED'))
        if group=='CPU_REGRESSIONS':
            from campaign_checkpoint import journal_results
            results,journal_issues=journal_results(Path(folder)/'events.jsonl')
            for name,result in results.items():
                if terminal_phases(result):
                    cpu[name]=dict(status='PASS' if result['status']=='PASS' else 'FAIL' if result['status'] in ('FAIL','ERROR') else 'BLOCKED')
    def counts(rows):return {s:sum(r['status']==s and r.get('source_valid',True) for r in rows.values()) for s in ('PASS','FAIL','BLOCKED')}
    value=dict(status=status,active_group=group,active_feature=active,CPU=counts(cpu),GPU=counts(gpu),
        heartbeat_utc=utc(),elapsed_seconds=round(time.monotonic()-started,3) if started else 0,
        live_counts_provisional_until_cleanup=bool(folder),
        journal_issues=journal_issues,last_controller_error=cp.get('last_controller_error'),
        revalidation_required=sorted(n for n,r in {**cpu,**gpu}.items() if r['status'] in ('FAIL','BLOCKED') or not r.get('source_valid',True)),
        source_invalidated=sorted(n for n,r in {**cpu,**gpu}.items() if not r.get('source_valid',True)),
        cpu_regressions=len(plan['cpu_inventory']),gpu_groups=len(FEATURES),GPU_executed=bool(gpu) or
            any(a['group']!='CPU_REGRESSIONS' for a in cp['attempts']),
        admission=cp['results'].get('ADMISSION',{}).get('status','UNVERIFIED_NO_GPU_STARTED'),
        full_cuda_acceptance=False,original_campaigns_unchanged=True,**extra)
    atomic_json(ROOT/'progress.json',value)
    return value


def cpu_group_ready(cp,plan):
    """Default CUDA scope requires its complete CPU regression gate."""
    return cp['results'].get('CPU_REGRESSIONS',{}).get('status')=='PASS' and cp['results']['CPU_REGRESSIONS'].get('source_valid',True)


def normal_cpu_completion(group,receipt,cp,plan):
    """Extension point for independent user journeys; default remains strict."""
    return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--prepare-only',action='store_true');mode.add_argument('--status',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--revalidate',action='append',default=[],metavar='FEATURE_OR_CPU_NODEID')
    args=parser.parse_args()
    if args.revalidate and (not args.resume or args.status or args.prepare_only):parser.error('--revalidate requires --resume execution')
    if args.status:
        print(json.dumps(read_json(ROOT/'progress.json',dict(status='NOT_PREPARED')),indent=2));return 0
    if args.prepare_only:
        lock=(CPU_ROOT/'manager.lock').open('a+b');lock.seek(0)
        if not lock.read(1):lock.write(b'0');lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:lock.close();raise RuntimeError('CUDA_MANAGER_ALREADY_OWNED')
        try:p,_=prepare()
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
        print(json.dumps(dict(status='PREPARED',cpu_cases=len(p['cpu_inventory']),groups=len(FEATURES),GPU_executed=False)));return 0
    plan=read_json(ROOT/'plan.json');cp=read_json(ROOT/'checkpoint.json')
    if not plan:raise RuntimeError('PREPARATION_REQUIRED: use -CudaScope -PrepareOnly')
    if cp['attempts'] and not args.resume:raise RuntimeError('RESUME_REQUIRED: use -CudaScope -Resume; no automatic retries')
    validate(plan)
    if digest(plan)!=cp['plan_sha256']:raise RuntimeError('PLAN_HASH_MISMATCH')
    # Share the existing host-wide campaign lock: never overlap CPU/native jobs.
    lock=(CPU_ROOT/'manager.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:lock.close();raise RuntimeError('CUDA_MANAGER_ALREADY_OWNED')
    try:
        config=read_json(REPO/'tools/final_acceptance/campaign_config.json')
        if any(config[k]!=plan['budgets'][p] for k,p in [('process_memory_bytes','process_bytes'),('job_memory_bytes','job_bytes'),('max_processes','max_processes')]):raise RuntimeError('JOB_CONFIG_PLAN_MISMATCH')
        # Poll this separate GPU profile more often without changing the CPU runner.
        config=dict(config,live_progress_interval_seconds=1,source_integrity_interval_seconds=5)
        gpu=plan['runtime'];violation={}
        for attempt in cp['attempts']:
            if attempt['status']!='RUNNING':continue
            if attempt.get('plan_sha256')!=digest(plan):
                raise RuntimeError('INTERRUPTED_ATTEMPT_PLAN_CHANGED: historical evidence cannot become current PASS')
            old_folder=Path(attempt['directory']);old_receipt=read_json(old_folder/'controller.json')
            # Never infer owner death from a PID. Unknown cleanup requires investigation.
            if not old_receipt or not old_receipt.get('cleanup_verified') or old_receipt.get('cleanup',{}).get('active_processes')!=0:
                raise RuntimeError('INTERRUPTED_ATTEMPT_CLEANUP_UNVERIFIED: '+str(old_folder))
            if attempt['group']=='CPU_REGRESSIONS':
                record_cpu(cp,old_folder,read_json(old_folder/'request.json'),old_receipt,plan)
            else:
                from tools.cuda_acceptance.validation import validate_gpu_attempt
                old_request=read_json(old_folder/'request.json')
                values,critical=validate_gpu_attempt(old_folder,old_request,old_receipt)
                if critical:
                    values={n:dict(status='BLOCKED',reason=critical,evidence=str(old_folder/'controller.json'),sha256=file_hash(old_folder/'controller.json')) for n in old_request['features']}
                    cp['safety_hold']=dict(features=old_request['features'],reason=critical)
                retain_results(cp,'results',values)
            attempt['status']='RECOVERED';atomic_json(ROOT/'checkpoint.json',cp)
        cpu=runtime_plan()
        if cpu['missing']:raise RuntimeError('CPU_RUNTIME_UNVERIFIED: '+repr(cpu['missing']))
        verify_cached(cp)
        retry=set(args.revalidate)
        for name in retry:
            value=cp.get('cpu_results',{}).get(name) or cp['results'].get(name)
            eligible=set(plan['cpu_inventory'])|{f for group in FEATURES.values() for f in group}
            if name not in eligible or not value or (value['status'] not in ('FAIL','BLOCKED') and value.get('source_valid',True)):
                raise RuntimeError('REVALIDATION_REQUIRES_EXACT_FAIL_OR_BLOCKED_CASE: '+name)
        if cp.get('safety_hold') and not set(cp['safety_hold']['features'])<=retry:
            raise RuntimeError('SAFETY_HOLD_REQUIRES_EXPLICIT_REVALIDATION: '+repr(cp['safety_hold']))
        if cp['results'].get('CPU_REGRESSIONS') and not cp.get('cpu_results'):
            raise RuntimeError('LEGACY_CPU_CASE_EVIDENCE_REQUIRED; refusing implicit full CPU retry')
        started=time.monotonic()
        progress(cp,plan,'RUNNING',started)
        for group in ['CPU_REGRESSIONS',*FEATURES]:
            nodes=list(plan['cpu_inventory']) if group=='CPU_REGRESSIONS' else FEATURES[group]
            results=cp.get('cpu_results',{}) if group=='CPU_REGRESSIONS' else cp['results']
            pending=pending_cases(nodes,results,retry)
            if not pending:continue
            if group!='CPU_REGRESSIONS' and not cpu_group_ready(cp,plan):break
            if group not in ('CPU_REGRESSIONS','01-admission') and (cp['results'].get('ADMISSION',{}).get('status')!='PASS' or not cp['results']['ADMISSION'].get('source_valid',True)):break
            live=resources()
            if live['available_commit_bytes']<config['start_available_commit_bytes'] or live['disk_free_bytes']<config['minimum_start_disk_bytes']:
                violation['reason']='BLOCKED_RESOURCE_BEFORE_LAUNCH';break
            before=None
            if group!='CPU_REGRESSIONS':
                before=gpu_info()
                if before['free_mib']<plan['budgets']['start_gpu_free_mib']:violation['reason']='GPU_FREE_BUDGET_NOT_ADMITTED';break
            folder=ROOT/'attempts'/uuid.uuid4().hex;folder.mkdir(parents=True)
            scratch=Path(tempfile.mkdtemp(prefix='hyso-cuda-scope-'));work=scratch/'payload'
            request=request_for(folder,work,group,plan,{n:plan['cpu_inventory'][n] for n in pending} if group=='CPU_REGRESSIONS' else None)
            if group!='CPU_REGRESSIONS':request.update(features=pending,gpu_admission=before)
            runtime=dict(cpu,python=cpu['cpu_python']) if group=='CPU_REGRESSIONS' else dict(python=gpu['python'],runtime_paths=[gpu['site']])
            last=[0.]
            def update(*unused):
                if time.monotonic()-last[0]<1:return
                last[0]=time.monotonic()
                progress(cp,plan,'RUNNING',started,folder,group)
                if before is None:return
                try:
                    now=gpu_info()
                    if now['driver']!=before['driver'] or now['uuid']!=before['uuid']:violation['reason']='GPU_IDENTITY_CHANGED'
                    if now['free_mib']<plan['budgets']['minimum_gpu_free_mib'] or now['used_mib']-before['used_mib']>plan['budgets']['max_global_gpu_growth_mib']:violation['reason']='GPU_RESOURCE_BUDGET_BREACH'
                except Exception as exc:violation['reason']='GPU_MONITOR_UNVERIFIED: '+str(exc)
            cp['attempts'].append(dict(directory=str(folder),group=group,status='RUNNING',plan_sha256=digest(plan)))
            atomic_json(ROOT/'checkpoint.json',cp)
            progress(cp,plan,'RUNNING',started,folder,group)
            with ExecutionState():
                receipt=execute(folder,request,config,runtime,update=update,stop_requested=lambda:bool(violation),
                    integrity=lambda:snapshot()==plan['source_sha256'] and user_snapshot()==plan['user_files'])
            cp['attempts'][-1].update(status='FINISHED',receipt_sha256=file_hash(folder/'controller.json'))
            if not receipt['source_integrity'] or not receipt['cleanup_verified'] or receipt['cleanup']['active_processes']!=0:raise RuntimeError('OWNED_SAFETY_OR_SOURCE_INTEGRITY_VIOLATION')
            if group=='CPU_REGRESSIONS':
                record_cpu(cp,folder,request,receipt,plan)
            else:
                from tools.cuda_acceptance.validation import validate_gpu_attempt
                values,critical=validate_gpu_attempt(folder,request,receipt)
                if critical:
                    values={n:dict(status='BLOCKED',reason=critical,evidence=str(folder/'controller.json'),sha256=file_hash(folder/'controller.json'),source_valid=True) for n in pending}
                    cp['safety_hold']=dict(features=pending,reason=critical)
                retain_results(cp,'results',values)
                if critical:violation['reason']=critical
                elif cp.get('safety_hold') and set(cp['safety_hold']['features'])<=set(pending):cp.pop('safety_hold')
            atomic_json(ROOT/'checkpoint.json',cp)
            cleanup_owned(work,scratch,request['campaign_id']);scratch.rmdir()
            if receipt['monitor_reason'] or (receipt['exit_code']!=0 and not normal_cpu_completion(group,receipt,cp,plan)):violation.setdefault('reason','ABNORMAL_WORKER_TERMINATION')
            if violation:break
            if group=='01-admission' and cp['results'].get('ADMISSION',{}).get('status')!='PASS':break
        validate(plan)
        counts=dict(Counter(v['status'] for v in cp['results'].values()))
        expected=1+sum(map(len,FEATURES.values()))
        complete=len(cp['results'])==expected and all(r.get('source_valid',True) for r in cp['results'].values())
        status='BLOCKED' if violation else 'FINISHED' if complete else 'PARTIAL'
        progress(cp,plan,status,started,counts=counts,recorded=len(cp['results']),expected=expected,
            safety_stop=violation or None,remaining=[n for n in ['CPU_REGRESSIONS',*[f for g in FEATURES.values() for f in g]] if n not in cp['results']])
        print(json.dumps(read_json(ROOT/'progress.json'),indent=2))
        return 0 if complete and not violation and counts.get('PASS')==expected else 1
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        if ROOT.is_dir():atomic_json(ROOT/'progress.json',dict(read_json(ROOT/'progress.json',{}),status='BLOCKED',reason=str(exc),full_cuda_acceptance=False,automatic_retries=0))
        raise
