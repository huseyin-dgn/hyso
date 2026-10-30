"""Prepare, approve, execute once, observe, stop cooperatively, package evidence."""
from pathlib import Path
import argparse,contextlib,hashlib,json,os,shutil,subprocess,sys,threading,time,uuid,zipfile
from .core import (ROOT,DEFAULT_PYTHON,prepare,authorize,verify,read,publish,record,sha,now)
from .registry import STAGES
from .reporting import initial_results,write_reports,decision,prerequisites

def preserve(m):
    out=Path(m['output']);baseline=out/'preservation-before.json'
    if not baseline.exists():
        # Generic subsequent runs protect the sealed live sources, asset files,
        # retained raw evidence and all explicitly bound runtime files.
        refs={str(Path(m['repository'])/rel):h for rel,h in m['source_snapshot']['files'].items()}
        refs.update(m['replay_sources'])
        refs.update({r['path']:r['sha256'] for r in [*m['assets']['models'].values(),m['assets']['dataset'],m['assets']['split_indices']]})
        changed=[dict(path=p,expected=h,actual=sha(p) if Path(p).is_file() else None) for p,h in refs.items() if not Path(p).is_file() or sha(p)!=h]
        result=dict(scope='sealed original sources/assets/raw evidence',checked=len(refs),unexpected_changes=changed,all_readable_preserved=not changed,
            checkpoint_changes=0,source_modifications_by_execution=0)
    else:
        before=read(baseline);changed=[];unreadable=[]
        for row in before['files']:
            if row.get('sha256') is None:unreadable.append(row);continue
            p=Path(m['repository'])/row['path']
            try:current=sha(p)
            except OSError:current=None
            if current!=row['sha256']:changed.append(dict(path=row['path'],before=row['sha256'],after=current))
        baseline_names={r['path'] for r in before['files']};additions=[]
        repo=Path(m['repository'])
        excluded={Path(p).resolve() for p in m.get('preservation_excluded_directories',[str(out)])}
        for base,dirs,files in os.walk(repo):
            dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','.pytest_cache') and Path(base,d).resolve() not in excluded]
            for name in files:
                rel=Path(base,name).relative_to(repo).as_posix()
                if rel not in baseline_names:additions.append(rel)
        unexpected_added=[x for x in additions if x!='tools/run_full_acceptance.py' and not x.startswith('tools/full_acceptance/')]
        result=dict(scope='pre-development full repository baseline',checked=len(before['files']),unexpected_changes=changed,
            all_readable_preserved=not changed,unreadable_baseline=unreadable,new_infrastructure_only=True,
            original_dirty_files_preserved=not changed,checkpoint_changes=0,new_infrastructure_files=additions,unexpected_added=unexpected_added)
        result['all_readable_preserved']&=not unexpected_added
        ext=out/'external-preservation-before.json'
        if ext.exists():
            rows=read(ext)['files'];bad=[]
            for r in rows:
                if r.get('sha256') and sha(r['path'])!=r['sha256']:bad.append(r['path'])
            result.update(external_checked=len(rows),external_changed=bad,all_readable_preserved=not changed and not bad and not unexpected_added)
    result['verified_at']=now();publish(out/'preservation-verification.json',result)
    if not result['all_readable_preserved']:raise ValueError('protected input changed; no subsequent native work')
    return result

def commands(m):
    out=Path(m['output']);manifest=out/'manifest.json';h=sha(manifest)
    entry=Path(m['repository'])/'tools/run_full_acceptance.py'
    text=f'''# Prepared only. Full native acceptance is NOT complete.
$py = '{m['runtime']['python']['path']}'
$entry = '{entry}'
$manifest = '{manifest}'
$sha = '{h}'

# 1. Read-only environment/source/checkpoint/dataset preflight. No GPU work.
& $py $entry --manifest $manifest --approved-sha256 $sha --preflight
# Optional bounded preparation smoke: one trained Model B CPU forward only.
& $py $entry --manifest $manifest --approved-sha256 $sha --smoke

# 2. One upper command; runs only the sealed approved experiments.
# Native stages lacking approval remain explicitly NOT_AUTHORIZED.
& $py $entry --manifest $manifest --approved-sha256 $sha --execute

# 3. Run in a second PowerShell terminal while execution is active.
& $py $entry --manifest $manifest --status
# Optional live monitor: while ($true) {{ & $py $entry --manifest $manifest --status; Start-Sleep -Seconds 2 }}

# 4. Cooperative stop; controller terminates only its owned Job tree.
& $py $entry --manifest $manifest --stop

# 5. Package an inactive completed/interrupted/prepared evidence directory.
& $py $entry --manifest $manifest --package '{out.parent/(out.name+'-evidence.zip')}'

# Separate explicit authorization produces a NEW manifest and run directory.
# Do not uncomment until these exact native experiments are actually approved.
# --authorize '04.cuda,05.training,06.layer,06.roofline,07.energy,07.devices,08.video,08.camera,09.provider_cpu,09.provider_cuda,10.stability'
# Camera/video still require a task-appropriate trained adapter. No crash/OOM reproducer is included.
# & $py $entry --manifest $manifest --approved-sha256 $sha --authorize '<EXPLICIT_IDS>' --approval-text '<ACTUAL_APPROVAL>' --output '<NEW_EXCLUSIVE_DIRECTORY>'
# Then use the NEW manifest/NEW SHA shown by that command with --execute.
'''
    (out/'commands.ps1').write_text(text,encoding='utf-8')
    lines=['# Full acceptance execution plan','',
        'Preparation is complete only after safe preflight/smoke; native acceptance remains incomplete. Commands below are concrete pinned child invocations; the single entrypoint manages ownership and does not run a native worker directly without admission.','',
        '| Experiment | Authorization | Prerequisites missing | Device | Expected work | Outer budget |','|---|---|---|---|---|---|']
    for e in m['experiments']:
        lines.append('| '+e['id']+' | '+('APPROVED' if e['authorization']['approved'] else 'NOT_AUTHORIZED')+' | '+('; '.join(prerequisites(m,e)) or 'none known; native validation still required')+
            ' | '+str(e['device'])+' | '+e['scope']+' | '+str(e['outer_seconds'])+'s: '+e['deadline_basis']+' |')
    for e in m['experiments']:
        lines+=['',f"## {e['id']}",'',
            '- Command: `'+subprocess.list2cmdline(e['command'])+'`',
            '- Manifest hash: `'+h+'`; source tree: `'+m['source_snapshot']['tree_sha256']+'`.',
            '- Model: '+e['model']+'; protocol/input/config: matching named `manifest.protocols` and exact assets/checkpoint/state hash.',
            '- Expected work: '+e['expected_work']+'; output: `'+e['output']+'`.',
            '- Stop: '+', '.join(e['stop_rules'])+'.',
            '- Cleanup: '+e['cleanup']+'.']
    lines+=['','## Specific criterion limits','',
        'P13:5 fresh alternating worker pairs,2 warmups,2 declared priming windows,5 analyzed windows; CV<=.25 and relative CI half-width<=.5 unchanged. CPU calibration does not prove CUDA attribution.',
        'P18:new target/compute/memory NCU captures,2 warmups+3 actual samples; independent output/FFMA counters and10% spread calculation. Old failed sessions remain unchanged. No allocation fraction; fixed reference buffers are workload parameters, not hardware maxima.',
        'P23:real Model B train/validation data and separate observer fits; registered logical/RSS/isolated CPU saved-payload evidence. Disabled CUDA saved activation remains disabled.',
        'P30:a task-appropriate trained camera processor/model/data/source contract is missing by default. Do not use MNIST as a real-world camera classifier by assumption.',
        'P31:new manifest only, Model B CPU FP32 batch256 actual work>=7200s; full terminal/close/accounting/continuity/gaps/inner+outer Job cleanup. Historical manifest is preserved and never executed.',
        'P33:trained same-UUID idle/active board-power trace with bracketed timestamps and completed work; gross/net values null if required sensor/coverage/baseline is absent.',
        'P34:each A/B model has an actual matched CPU/GPU pair; identical checkpoint/state/preprocessing/input/batch/precision/protocol. No independent P28/P29/P31 pooling.',
        'B4 and CPU65536:separate controlled forensic plans/approval required. Neither reproducer is part of executable ordinary suite. All native crashes stop continuation; no retry.',
        'Missing resources or approval are explicit BLOCKED_PREREQUISITE/NOT_AUTHORIZED. They cannot make suite_complete=true. Raw native evidence, measured/derived/missing metrics and previous decisions remain separate.']
    (out/'execution-plan.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def worker(args):
    from .safety import verify_child_gate
    root=args.stage_output.resolve();request=read(root/'request.json')
    end=time.monotonic()+120
    while not (root/'admit.json').exists():
        if time.monotonic()>end:raise TimeoutError('missing owned Job admission')
        time.sleep(.02)
    gate=verify_child_gate(read(root/'admit.json'),request)
    m=verify(args.manifest,expected=request['manifest_sha256'],live=True)
    if request['manifest']!=str(args.manifest.resolve()) or request['experiment_id']!=args.worker or Path(request['output']).resolve()!=root:
        raise ValueError('worker request binding mismatch')
    operation=request['operation'];internal={'__environment':'environment','__smoke_cpu':'smoke_cpu','__cpu_benchmark_check':'cpu_benchmark_check','__cuda_benchmark_check':'cuda_benchmark_check'}
    if args.worker in internal:
        if operation!=internal[args.worker]:raise PermissionError('invalid safe preparation operation')
        if operation=='cuda_benchmark_check' and not next(e for e in m['experiments'] if e['id']=='04.cuda')['authorization']['approved']:
            raise PermissionError('04.cuda NOT_AUTHORIZED')
    else:
        e=next(x for x in m['experiments'] if x['id']==args.worker)
        if not e['authorization']['approved'] or operation!=e['handler'] or root!=Path(e['output']):raise PermissionError('experiment NOT_AUTHORIZED or routing mismatch')
    os.environ.update(HYSO_FULL_ACCEPTANCE_ADMITTED=request['manifest_sha256'],PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    snapshot=m['source_snapshot']['root'];sys.path[:0]=[snapshot,*m['runtime']['site_paths']]
    os.environ['PYTHONPATH']=os.pathsep.join([snapshot,str(Path(snapshot)/'tools'),str(Path(snapshot)/'tests'),*m['runtime']['site_paths']])
    publish(root/'ready.json',dict(pid=os.getpid(),gate=gate,manifest_sha256=request['manifest_sha256']),new=True)
    stop=threading.Event()
    def pulse():
        while not stop.is_set():
            publish(root/'heartbeat.json',dict(pid=os.getpid(),unix_time=time.time(),at=now(),scope='controller_liveness_not_completed_model_work'))
            stop.wait(1)
    thread=threading.Thread(target=pulse,daemon=True);thread.start()
    status=None
    try:
        from . import handlers as h
        dispatch={'environment':h.environment,'smoke_cpu':h.smoke_cpu,'cpu_benchmark_check':h.cpu_benchmark_check,'cuda_benchmark_check':h.cuda_benchmark_check,'regression':h.regression,
            'inference_cpu':lambda m,r:h.inference(m,r,'cpu'),'inference_cuda':lambda m,r:h.inference(m,r,'cuda:0'),
            'training':h.training,'layer':h.layer,'roofline':h.roofline,'energy':lambda m,r:h.stability(m,r,energy=True),
            'devices':h.devices,'video':lambda m,r:h.camera(m,r,live=False),'camera':lambda m,r:h.camera(m,r,live=True),
            'provider_cpu':lambda m,r:h.provider(m,r,cuda=False),'provider_cuda':lambda m,r:h.provider(m,r,cuda=True),
            'stability':h.stability,'memory':h.memory,'replay':h.replay}
        value=dispatch[operation](m,root)
        from .reporting import validate_metrics
        validate_metrics(value);value.update(id=args.worker,finished_at=now(),manifest_sha256=request['manifest_sha256'],gate=gate)
        publish(root/'result.json',value,new=True);status=value['status']
    except BaseException as error:
        publish(root/'result.json',dict(id=args.worker,status='FAIL',reason=type(error).__name__+': '+str(error),metrics=[],
            criteria={},finished_at=now(),manifest_sha256=request['manifest_sha256']),new=True)
        import traceback;traceback.print_exc();status='FAIL'
    finally:stop.set();thread.join(3)
    return 1 if status=='FAIL' else 0

def preflight(args):
    from .safety import supervise
    if args.manifest:
        if (args.manifest.resolve().parent/'STARTED.json').exists():raise FileExistsError('preflight cannot reset an execution report after STARTED; use --status')
        m=verify(args.manifest,expected=args.approved_sha256)
    else:
        if args.output is None:
            from datetime import datetime,timezone,timedelta
            args.output=ROOT/'outputs'/('full-acceptance-'+datetime.now(timezone(timedelta(hours=3))).strftime('%Y%m%d-%H%M%S'))
        m=prepare(args.output,python=args.python,gpu_uuid=args.gpu_uuid,camera_config=args.camera_config,
            ncu_plan=args.ncu_plan,ort_cpu=args.ort_cpu,ort_cuda=args.ort_cuda)
    out=Path(m['output']);row=dict(id='__environment',handler='environment',output=None,outer_seconds=180)
    path=out/'preparation'/('preflight-'+uuid.uuid4().hex[:8]);p=supervise(m,row,operation='environment',output=path)
    payload=dict(status=p['status'],source_checkpoints_dataset_runtime='hash_verified_before_import',
        experiment_prerequisites={e['id']:prerequisites(m,e) for e in m['experiments']},result=p,
        model_loads=0,training_calls=0,GPU_native_work=False,live_camera_opened=False,long_run_started=False)
    publish(out/'preflight.json',payload)
    rows=initial_results(m);rows['01.preflight']=dict(p,id='01.preflight',stage='01')
    prep=read(out/'stage-results.json').get('preparation',{}) if (out/'stage-results.json').exists() else {}
    prep['preflight']=p['status'];write_reports(m,rows,state='PREFLIGHT_'+p['status'],preparation=prep)
    publish(out/'run-status.json',dict(state='PREPARED' if p['status']=='PASS' else 'PREFLIGHT_FAILED',run_id=m['run_id'],
        suite_complete=False,controller_active=False,manifest_sha256=sha(out/'manifest.json'),updated_at=now()))
    preserve(m);commands(m)
    print(json.dumps(dict(output=str(out),manifest_sha256=sha(out/'manifest.json'),preflight=p['status'],suite_complete=False)))
    return 0 if p['status']=='PASS' else 1

def smoke(args,*,benchmark_check=False):
    from .safety import supervise
    if benchmark_check and not args.approved_sha256:raise ValueError('--cpu-benchmark-check requires exact --approved-sha256')
    m=verify(args.manifest,expected=args.approved_sha256);out=Path(m['output'])
    if (out/'STARTED.json').exists():raise FileExistsError('smoke cannot alter an execution directory after STARTED')
    operation='cpu_benchmark_check' if benchmark_check else 'smoke_cpu'
    folder=out/'preparation'/ (('benchmark-check-' if benchmark_check else 'smoke-')+uuid.uuid4().hex[:8])
    p=supervise(m,dict(id='__'+operation,handler=operation,output=None,outer_seconds=180),operation=operation,output=folder)
    current=read(out/'stage-results.json');rows={r['id']:r for r in current['experiments']}
    prep=current.get('preparation',{});prep[operation]=p
    write_reports(m,rows,state='PREPARATION_VALIDATED' if p['status']=='PASS' else 'SMOKE_FAILED',preparation=prep)
    publish(out/('cpu-benchmark-check-results.json' if benchmark_check else 'smoke-results.json'),p)
    publish(out/'run-status.json',dict(state='PREPARATION_VALIDATED' if p['status']=='PASS' else 'SMOKE_FAILED',
        suite_complete=False,controller_active=False,run_id=m['run_id'],manifest_sha256=sha(out/'manifest.json'),updated_at=now()))
    preserve(m);print(json.dumps(dict(check=operation,status=p['status'],GPU_native_work=False,P31_long_run_started=False,suite_complete=False) if benchmark_check else
        dict(smoke=p['status'],GPU_native_work=False,P31_long_run_started=False,suite_complete=False)))
    return 0 if p['status']=='PASS' else 1

def cuda_check(args):
    from .safety import supervise
    if not args.approved_sha256:raise ValueError('--cuda-benchmark-check requires exact --approved-sha256')
    m=verify(args.manifest,expected=args.approved_sha256);out=Path(m['output'])
    if not next(e for e in m['experiments'] if e['id']=='04.cuda')['authorization']['approved']:raise PermissionError('04.cuda NOT_AUTHORIZED')
    if (out/'STARTED.json').exists():raise FileExistsError('boundary check cannot alter a started acceptance run')
    if (out/'stop-request.json').exists():raise ValueError('stop request exists; prepare a fresh manifest')
    folder=out/'preparation'/('cuda-benchmark-check-'+uuid.uuid4().hex[:8])
    value=supervise(m,dict(id='__cuda_benchmark_check',handler='cuda_benchmark_check',output=None,outer_seconds=180),
        operation='cuda_benchmark_check',output=folder,stop_file=out/'stop-request.json')
    publish(out/'cuda-benchmark-check-results.json',value);preserve(m)
    print(json.dumps(dict(check='cuda_benchmark_check',status=value['status'],suite_complete=False)))
    return 0 if value['status']=='PASS' else 1

def execute(args):
    from .safety import supervise
    if args.approved_sha256 is None:raise ValueError('--execute requires exact --approved-sha256')
    m=verify(args.manifest,expected=args.approved_sha256);out=Path(m['output']);stopfile=out/'stop-request.json'
    if stopfile.exists():raise ValueError('stop request exists; prepare a fresh manifest, do not clear history')
    publish(out/'STARTED.json',dict(pid=os.getpid(),at=now(),manifest_sha256=sha(args.manifest),one_shot=True),new=True)
    rows=initial_results(m);stopped=False;state='RUNNING';complete=False
    prep=read(out/'stage-results.json').get('preparation',{}) if (out/'stage-results.json').exists() else {}
    try:
        for e in m['experiments']:
            status,reason,missing=decision(m,e,rows,stopped=stopped)
            if stopfile.exists():status,reason='NOT_RUN_STOPPED','user stop before next experiment';stopped=True
            publish(out/'run-status.json',dict(state='RUNNING',run_id=m['run_id'],controller_pid=os.getpid(),controller_active=True,
                current_experiment=e['id'],manifest_sha256=sha(args.manifest),suite_complete=False,updated_at=now()))
            if status=='READY':
                verify(args.manifest,expected=args.approved_sha256,live=True)
                value=supervise(m,e,stop_file=stopfile);rows[e['id']]=dict(value,id=e['id'],stage=e['stage'])
                if value['status'] in ('FAIL','ABORTED'):
                    stopped=True
            else:rows[e['id']].update(status=status,reason=reason,readiness=missing)
            write_reports(m,rows,state='RUNNING',preparation=prep)
            print(e['id'],rows[e['id']]['status'],flush=True)
        complete=all(r.get('attempted') and r['status'] in ('PASS','PARTIAL','FAIL','UNSUPPORTED') for r in rows.values())
        state='EXECUTION_COMPLETE_WITH_RESULTS' if complete else 'INCOMPLETE'
    except BaseException as error:
        state='ABORTED' if isinstance(error,KeyboardInterrupt) else 'FAILED_CONTROLLER'
        publish(out/'controller-error.json',dict(type=type(error).__name__,reason=str(error),at=now()))
        for row in rows.values():
            if row['status']=='PENDING':row.update(status='NOT_RUN_STOPPED',reason='controller stopped; no retry')
    finally:
        try:preserve(m)
        except (OSError,ValueError) as error:
            state='FAILED_PRESERVATION';complete=False
            publish(out/'preservation-error.json',dict(reason=str(error),at=now()))
        write_reports(m,rows,state=state,preparation=prep)
        publish(out/'run-status.json',dict(state=state,suite_complete=complete,controller_active=False,controller_pid=os.getpid(),
            run_id=m['run_id'],manifest_sha256=sha(args.manifest),updated_at=now()))
    return 1 if any(r['status']=='FAIL' for r in rows.values()) else 0 if complete else 2

def package(args):
    m=read(args.manifest);out=Path(m['output']);state=read(out/'run-status.json')
    if state.get('controller_active'):raise RuntimeError('active run: request stop or wait for cleanup before packaging')
    target=args.package.resolve()
    if target.is_relative_to(out) or target.exists():raise ValueError('archive must be new and outside the evidence directory')
    entries=[]
    for p in sorted(out.rglob('*')):
        if p.is_symlink():raise ValueError('do not package linked evidence')
        if p.is_file() and '__pycache__' not in p.parts:entries.append(dict(path=p.relative_to(out).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    publish(out/'package-inventory.json',dict(created_at=now(),suite_complete=state['suite_complete'],files=entries))
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for r in entries:archive.write(out/r['path'],r['path'])
        archive.write(out/'package-inventory.json','package-inventory.json')
    print(json.dumps(dict(archive=str(target),sha256=sha(target),files=len(entries)+1,suite_complete=state['suite_complete'])))
    return 0

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    for name in ('preflight','execute','smoke','cpu-benchmark-check','cuda-benchmark-check','status','stop'):mode.add_argument('--'+name,action='store_true')
    mode.add_argument('--package',type=Path);mode.add_argument('--authorize',help='explicit comma-separated experiment IDs; creates new manifest only')
    mode.add_argument('--worker',help=argparse.SUPPRESS)
    p.add_argument('--manifest',type=Path);p.add_argument('--output',type=Path);p.add_argument('--approved-sha256')
    p.add_argument('--python',type=Path,default=DEFAULT_PYTHON);p.add_argument('--gpu-uuid');p.add_argument('--camera-config',type=Path)
    p.add_argument('--ncu-plan',type=Path);p.add_argument('--ort-cpu',type=Path);p.add_argument('--ort-cuda',type=Path)
    p.add_argument('--approval-text');p.add_argument('--stage-output',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if not a.preflight and a.manifest is None:p.error('--manifest is required')
    try:
        if a.worker:return worker(a)
        if a.preflight:return preflight(a)
        if a.smoke:return smoke(a)
        if a.cpu_benchmark_check:return smoke(a,benchmark_check=True)
        if a.cuda_benchmark_check:return cuda_check(a)
        if a.execute:return execute(a)
        if a.authorize:
            if a.output is None or not a.approved_sha256 or not a.approval_text:p.error('authorization requires --output, --approved-sha256 and --approval-text')
            m=authorize(a.manifest,a.output,a.authorize.split(','),expected=a.approved_sha256,basis=a.approval_text)
            rows=initial_results(m);write_reports(m,rows);commands(m);preserve(m)
            publish(Path(m['output'])/'run-status.json',dict(state='AUTHORIZED_PREPARED',suite_complete=False,controller_active=False,updated_at=now()))
            print(json.dumps(dict(output=m['output'],manifest_sha256=sha(Path(m['output'])/'manifest.json'),executed=False)))
            return 0
        if a.package:return package(a)
        if a.status:
            out=a.manifest.resolve().parent;status=read(out/'run-status.json');status['experiments']={r['id']:r['status'] for r in read(out/'stage-results.json')['experiments']}
            print(json.dumps(status,ensure_ascii=False,indent=2));return 0
        if a.stop:
            m=read(a.manifest);out=Path(m['output']);publish(out/'stop-request.json',dict(run_id=m['run_id'],manifest_sha256=sha(a.manifest),requested_at=now()),new=True)
            print(json.dumps(dict(status='STOP_REQUESTED',scope='controller cooperatively closes its exact owned Job; no external PID kill')));return 0
    except (OSError,ValueError,RuntimeError,StopIteration) as error:
        print(type(error).__name__+': '+str(error),file=sys.stderr);return 1
