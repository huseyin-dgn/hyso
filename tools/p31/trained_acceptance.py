"""P31 trained-CPU plan, short validation and hash-approved one-shot long run.

No command defaults to execution. `plan`/`prepare` use stdlib only. `start`
requires the exact manifest SHA256 supplied after separate user authorization.
"""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from launch_p31 import publish_new


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def sources():
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    return {name:sha(ROOT/name) for name in sorted(set(names)) if name and (ROOT/name).is_file()}


def plan(audit):
    prior=ROOT/'outputs/p29-tradeoff-completion-20261029/protocol.json'
    original=ROOT/'outputs/real-two-model-comparison-20261029/protocol.json'
    p=read(prior);m=read(original);dataset=read(m['dataset_manifest']);inventory={}
    references={}
    for arm in ('A','B'):
        candidate=p['models'][arm]
        if sha(candidate['checkpoint'])!=candidate['checkpoint_sha256']:raise ValueError('checkpoint hash mismatch')
        ref=ROOT/f'outputs/real-two-model-comparison-20261029/workers/{arm}-0/result.json'
        d=read(ref)
        if d['status']!='PASS' or d['checkpoint']['checkpoint_sha256']!=candidate['checkpoint_sha256']:
            raise ValueError('past inference reference mismatch')
        if d['checkpoint']['model_state_sha256']!=candidate['state_sha256']:raise ValueError('past state association mismatch')
        inventory[arm]=dict(candidate,config=m['models'][arm]['config'],reference=str(ref),reference_sha256=sha(ref))
        references[str(ref)]=sha(ref)
    if sha(dataset['source_path'])!=p['dataset_sha256']:raise ValueError('MNIST dataset hash mismatch')
    value=dict(feature='P31',model='trained_Model_B',device='cpu',precision='fp32',
        checkpoint=dict(path=p['models']['B']['checkpoint'],sha256=p['models']['B']['checkpoint_sha256'],
                        model_state_sha256=p['models']['B']['state_sha256']),
        model_config=m['models']['B']['config'],input_shape=[256,1,28,28],batch_size=256,
        preprocessing=p['preprocessing'],dataset=dict(path=dataset['source_path'],sha256=p['dataset_sha256']),
        cpu_threads=2,interop_threads=1,setup_warmup_calls=1,automatic_retry=False,
        scope='same_trained_B_CPU_FP32_batch256_resident_first_ordered_test_images_as_P29_no_reproduction_of_P29')
    publish_new(audit/'trained-workload.json',value)
    publish_new(audit/'model-reference-inventory.json',dict(models=inventory,references=references,
        previous_P17_P28_P29_reexecuted=False,source_protocol=str(prior),source_protocol_sha256=sha(prior)))
    print(json.dumps(dict(status='WORKLOAD_PLAN_ONLY',plan_sha256=sha(audit/'trained-workload.json'),long_run_started=False)))


def export(result,root):
    import csv
    data=result.to_dict();clock_id=data['evidence']['header']['clock_id'] if data['evidence']['header'] else None
    result.to_json(root/'result.json');result.to_csv(root/'result.csv')
    d=result.stability_diagnostics();publish_new(root/'diagnostics.json',d)
    rows=[]
    for w in d.get('latency_throughput_windows',[]):
        rows.append(dict(clock_id=clock_id,index=w['index'],start_ns=w['start_ns'],end_ns=w['end_ns'],phase=w['phase'],
            calls=w['completed_inference_calls'],images=w['completed_images'],images_per_second=w['completion_window_images_per_second'],
            mean_ms=w['latency']['mean_ms'],p50_ms=w['latency']['p50_ms'],p95_ms=w['latency']['p95_ms'],
            p99_ms=w['latency']['p99_ms'],p99_reason=w['latency']['p99_reason'],gap=w['gap']))
    if rows:
        with (root/'latency-throughput.csv').open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    memory=[]
    for row in data['evidence']['rows']:
        memory.append(dict(clock_id=clock_id,timestamp_ns=row['timestamp_ns'],
            rss_query_start_ns=row['rss']['query_start_ns'] if row['rss'] else None,
            rss_query_end_ns=row['rss']['query_end_ns'] if row['rss'] else None,
            rss_bytes=row['rss']['rss_bytes'] if row['rss'] else None,
            private_commit_query_start_ns=row['process_commit']['query_start_ns'],private_commit_query_end_ns=row['process_commit']['query_end_ns'],
            private_commit_bytes=row['process_commit']['value'],private_commit_reason=row['process_commit']['reason'],
            cuda_allocated_bytes=row['allocator']['allocated_bytes'] if row['allocator'] else None,
            cuda_reserved_bytes=row['allocator']['reserved_bytes'] if row['allocator'] else None,
            cuda_reason='not_requested_CPU_workload' if row['allocator'] is None else None))
    if memory:
        with (root/'memory.csv').open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(memory[0]));writer.writeheader();writer.writerows(memory)


def execute(audit,protocol,folder):
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    from hyso.performance import StabilityProtocol,WorkloadRecipe,measure_stability,reopen_stability,RuntimeResult
    root=audit/folder;root.mkdir(exist_ok=False)
    recipe=WorkloadRecipe('trained_workload:make_model_b',dict(plan_path=str(audit/'trained-workload.json'),
        plan_sha256=sha(audit/'trained-workload.json')),search_path=str(Path(__file__).parent))
    error=None;result=None;phase='measure';validation=None
    try:
        result=measure_stability(recipe,root/'journal',protocol=StabilityProtocol(**protocol))
        phase='export';export(result,root)
        if RuntimeResult.from_json(root/'result.json').to_dict()!=result.to_dict():raise ValueError('JSON reopen mismatch')
        if RuntimeResult.from_csv(root/'result.csv').to_dict()!=result.to_dict():raise ValueError('CSV reopen mismatch')
        if reopen_stability(root/'journal').to_dict()['evidence']!=result.to_dict()['evidence']:raise ValueError('journal reopen mismatch')
        a=result.to_dict()['analysis'];c=result.to_dict()['evidence']['controller']
        if a['status']!='completed' or c['cleanup']['active_processes']!=0:raise RuntimeError('run incomplete; no retry')
        validation=dict(status='COMPLETED_PENDING_SCOPE_REVIEW',actual_seconds=a['duration_seconds'],steps=a['steps'],completed_images=a['completed'],
            cleanup=c['cleanup'],exit_code=c['exit_code'],native_crash=False,JSON_CSV_journal_reopen=True)
    except BaseException as exc:
        error=exc
        # Reopen is evidence recovery only; never restart an interrupted worker.
        if phase=='measure' and (root/'journal/controller.json').exists():
            try:
                result=reopen_stability(root/'journal');export(result,root)
            except BaseException as secondary:
                publish_new(root/'evidence-recovery-error.json',dict(type=type(secondary).__name__,reason=str(secondary)))
    finally:
        publish_new(root/'launcher-final.json',dict(status='COMPLETED_PENDING_REVIEW' if error is None else 'INCOMPLETE',
            validation=validation,error=None if error is None else dict(type=type(error).__name__,reason=str(error)),
            phase=phase,automatic_retry=False,execution_resume=False,long_run_acceptance='NOT_GRANTED_BY_PROCESS_EXIT',
            finished_utc=datetime.now(timezone.utc).isoformat()))
    if error is not None:raise error
    print(json.dumps(validation))


def prepare(audit):
    p=read(audit/'trained-workload.json')
    if p['device']!='cpu' or p['precision']!='fp32':raise ValueError('only reviewed CPU plan supported')
    protocol=dict(duration_seconds=7200.,interval_seconds=5.,idle_seconds=0.,startup_timeout_seconds=120.,
        step_timeout_seconds=60.,max_samples=2000,chunk_rows=1,max_journal_bytes=128*1024**2,seed=0,
        process_memory_bytes=None,job_memory_bytes=None,max_processes=None)
    # Stdlib validation/pinning only. No factory, torch import, worker or CUDA query.
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            brand=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    except OSError:brand=platform.processor()
    value=dict(feature='P31',schema_version=1,status='PREPARED_NOT_STARTED',repository=str(ROOT),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        source_sha256=sources(),python_executable=sys.executable,python_sha256=sha(sys.executable),python_version=sys.version,
        packages={name:importlib.metadata.version(name) for name in ('torch','torchvision','numpy')},
        workload_plan=str(audit/'trained-workload.json'),workload_plan_sha256=sha(audit/'trained-workload.json'),
        workload=p,device_identity=dict(device='cpu',processor=brand,logical_cpus=os.cpu_count(),platform=platform.platform()),
        protocol=protocol,outer_watchdog_seconds=7440,expected_output_root=str(audit/'long-run'),
        expected_files=['result.json','result.csv','diagnostics.json','latency-throughput.csv','memory.csv',
                        'journal/header.json','journal/chunk-*.json','journal/terminal.json','journal/controller.json',
                        'journal/stdout.txt','journal/stderr.txt','launcher-final.json'],
        outer_files=['STARTED.json','launcher-stdout.txt','launcher-stderr.txt','parent-exit.json'],
        resource_policy='no_RAM_VRAM_or_allocator_quota;Windows_OS_low_memory_stop;owned_kill_on_close_Jobs;finite_duration_and_evidence_budget',
        stop_conditions=['OS_low_memory','startup_or_heartbeat_or_duration_timeout','native_crash',
            'workload_or_measurement_exception','checkpoint_or_source_or_input_mismatch','evidence_budget_exhausted','user_interrupt'],
        cleanup='inner_and_outer_Windows_Job_terminate_wait_observe_active_processes_zero;no_unowned_process_termination',
        approval_scope='one_7200_second_trained_Model_B_CPU_FP32_batch256_run_only;separate_explicit_user_approval_required',
        automatic_retry=False,execution_resume=False,new_GPU_workload=False,prior_P28_GPU_approval_used=False,
        legacy_GPU_7200_plan='preserved_not_executed_or_replaced;this_new_manifest_qualifies_only_trained_CPU_scope',
        P30='PARTIAL_preserved',P31='PARTIAL_until_actual_long_run_review',B4='OPEN',CPU65536='OPEN')
    publish_new(audit/'long-run-manifest.json',value)
    digest=sha(audit/'long-run-manifest.json')
    with (audit/'long-run-manifest.sha256').open('x',encoding='ascii') as f:f.write(digest+'\n')
    print(json.dumps(dict(status='MANIFEST_READY_NOT_STARTED',sha256=digest,long_run_started=False)))


def verify(audit,digest):
    path=audit/'long-run-manifest.json'
    if sha(path)!=digest:raise ValueError('explicit approved manifest SHA256 mismatch')
    m=read(path)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    if (head!=m['git_head'] or str(ROOT)!=m['repository'] or sources()!=m['source_sha256'] or sys.version!=m['python_version']
            or str(Path(sys.executable))!=m['python_executable'] or sha(sys.executable)!=m['python_sha256']):
        raise ValueError('source or interpreter changed; fresh review required')
    if any(importlib.metadata.version(name)!=version for name,version in m['packages'].items()):raise ValueError('runtime package changed')
    if sha(m['workload_plan'])!=m['workload_plan_sha256']:raise ValueError('workload changed')
    for item in ('checkpoint','dataset'):
        if sha(m['workload'][item]['path'])!=m['workload'][item]['sha256']:raise ValueError('input artifact changed')
    return m


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','smoke','prepare','preflight','start','inner'))
    parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--approved-sha256')
    args=parser.parse_args();audit=args.audit.resolve(strict=True)
    if args.action=='plan':return plan(audit)
    if args.action=='prepare':return prepare(audit)
    if args.action=='smoke':
        return execute(audit,dict(duration_seconds=3.,interval_seconds=.2,chunk_rows=1),'trained-smoke')
    if not args.approved_sha256:parser.error('exact manifest SHA256 required; no run started')
    manifest=verify(audit,args.approved_sha256)
    if args.action=='preflight':
        print(json.dumps(dict(status='PREFLIGHT_PASS',factory_calls=0,owned_workers_started=0,CUDA_queries=0,long_run_started=False)))
        return
    marker=audit/'STARTED.json'
    if args.action=='start':
        if marker.exists() or (audit/'long-run').exists():raise FileExistsError('one-shot run already reserved; no retry')
        publish_new(marker,dict(manifest_sha256=args.approved_sha256,started_utc=datetime.now(timezone.utc).isoformat(),automatic_retry=False))
        from start_approved import observe
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        return observe([sys.executable,'-B',str(Path(__file__).resolve()),'inner','--audit',str(audit),
            '--approved-sha256',args.approved_sha256],audit,timeout=manifest['outer_watchdog_seconds'])
    if not marker.exists() or read(marker)['manifest_sha256']!=args.approved_sha256:raise ValueError('missing one-shot approved start reservation')
    execute(audit,manifest['protocol'],'long-run')


if __name__=='__main__':raise SystemExit(main())
