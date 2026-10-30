"""Stdlib Windows Job supervisor. No memory/allocator quotas or PID-name kills."""
from pathlib import Path
import ctypes as c
from ctypes import wintypes as w
import importlib.util,json,os,stat,subprocess,sys,time,uuid
from .core import publish,read,now,sha

def load_file(path,name):
    """Execute checked source bytes without reading or writing a .pyc cache."""
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    # SourceFileLoader.exec_module can create __pycache__ in an immutable
    # snapshot when the coordinator was launched without -B. It can also read
    # existing bytecode. Neither behavior is valid for these pinned helpers.
    code=compile(Path(path).read_bytes(),str(path),'exec',dont_inherit=True)
    exec(code,module.__dict__);return module

def duplicate_query(job,process):
    api=job.api;api.GetCurrentProcess.restype=w.HANDLE
    api.DuplicateHandle.argtypes=[w.HANDLE,w.HANDLE,w.HANDLE,c.POINTER(w.HANDLE),w.DWORD,w.BOOL,w.DWORD]
    api.DuplicateHandle.restype=w.BOOL;remote=w.HANDLE()
    if not api.DuplicateHandle(api.GetCurrentProcess(),job.handle,w.HANDLE(int(process._handle)),c.byref(remote),4,False,0):
        raise OSError(c.get_last_error(),'DuplicateHandle Job query')
    return remote.value

def verify_child_gate(gate,request):
    if gate['pid']!=os.getpid() or gate['parent_pid']!=os.getppid() or gate['token']!=request['token']:
        raise ValueError('gate identity mismatch')
    api=c.WinDLL('kernel32',use_last_error=True)
    api.GetCurrentProcess.restype=w.HANDLE
    api.IsProcessInJob.argtypes=[w.HANDLE,w.HANDLE,c.POINTER(w.BOOL)];api.IsProcessInJob.restype=w.BOOL
    api.QueryInformationJobObject.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD,c.c_void_p];api.QueryInformationJobObject.restype=w.BOOL
    api.CloseHandle.argtypes=[w.HANDLE];api.CloseHandle.restype=w.BOOL;member=w.BOOL()
    if not api.IsProcessInJob(api.GetCurrentProcess(),w.HANDLE(gate['query_handle']),c.byref(member)) or not member.value:
        raise ValueError('child not assigned to exact owned Job')
    # Use the stdlib ctypes layout from the exact pinned production helper.
    module=load_file(Path(request['snapshot'])/'hyso/performance/_training_process.py','gate_layout')
    temporary=module._Job(None,None,None)
    try:
        info=temporary.Extended()
        if not api.QueryInformationJobObject(w.HANDLE(gate['query_handle']),9,c.byref(info),c.sizeof(info),None):
            raise OSError(c.get_last_error(),'child Job readback')
        if info.basic.flags!=0x2000 or info.process_memory or info.job_memory or info.basic.processes:
            raise ValueError('unexpected quota or missing kill-on-close')
    finally:temporary.close();api.CloseHandle(w.HANDLE(gate['query_handle']))
    return dict(assigned=True,exact_job_membership=True,kill_on_close=True,memory_quotas=None,process_limit=None)

def native_exit(code):return code is not None and (code<0 or (code&0xffffffff)>=0x80000000)

def stage_evidence_bytes(root):
    """Sample live evidence without racing atomic publishers' temporary paths."""
    total=0
    for path in root.rglob('*'):
        try:info=path.stat()
        except FileNotFoundError:
            # publish() can replace a listed UUID temporary file before stat.
            # Other I/O errors must still fail supervision.
            continue
        if stat.S_ISREG(info.st_mode):total+=info.st_size
    return total


def supervise(manifest,experiment,*,operation=None,output=None,stop_file=None):
    """Assign before native imports; retain crash/timeout/stop prefixes and receipts."""
    if os.name!='nt':raise NotImplementedError('owned native supervisor verified on Windows only')
    snapshot=Path(manifest['source_snapshot']['root']);root=Path(output or experiment['output'])
    root.mkdir(parents=True,exist_ok=False)
    policy=manifest['supervision'];token=uuid.uuid4().hex
    request=dict(token=token,snapshot=str(snapshot),manifest=str(Path(manifest['output'])/'manifest.json'),
        manifest_sha256=sha(Path(manifest['output'])/'manifest.json'),experiment_id=experiment['id'],
        operation=operation or experiment['handler'],output=str(root))
    if request['operation'] == 'inference_cuda_subtask':
        from .cuda_job_binding import validate_subtask_job_request
        request.update(parent_id=experiment.get('parent_id'),arm=experiment.get('arm'),
                       measurement=experiment.get('measurement'),
                       gpu_uuid=manifest['hardware'].get('gpu_uuid'),
                       source_tree_sha256=manifest['source_snapshot']['tree_sha256'])
        validate_subtask_job_request(manifest,experiment,request,route_installed=True)
        # The child requires all admission guard modules in its sealed snapshot.
        for item in ('cuda_tasks.py','cuda_dispatch.py','cuda_subtask_worker.py'):
            if not (snapshot/'tools/full_acceptance'/item).is_file():
                raise FileNotFoundError('missing pinned CUDA subtask module: '+item)
    publish(root/'request.json',request,new=True)
    argv=[manifest['runtime']['python']['path'],'-I','-S','-B',str(snapshot/'tools/run_full_acceptance.py'),
        '--worker',experiment['id'],'--manifest',request['manifest'],'--stage-output',str(root)]
    env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
    # Native descendants inherit only the explicit physical device for CUDA ops.
    env['CUDA_VISIBLE_DEVICES']=manifest['hardware']['gpu_uuid'] if request['operation'] in (
        'inference_cuda','inference_cuda_subtask','cuda_benchmark_check','energy','devices','provider_cuda','roofline') and manifest['hardware']['gpu_uuid'] else '-1'
    env.pop('HYSO_NATIVE_GPU_UUID',None)
    module=load_file(snapshot/'hyso/performance/_training_process.py','acceptance_owned_job')
    resource_module=load_file(snapshot/'hyso/performance/_stream_resources.py','acceptance_low_memory')
    job=module._Job(None,None,None);resources=None;process=None;assigned=False;reason=None;error=None;cleanup=None
    started=time.monotonic();started_at=now();stage_deadline=experiment.get('outer_seconds')
    try:
        resources=resource_module._StreamResources(token)
        if resources.sample(job):reason='OS_low_memory_before_launch'
        else:
            with (root/'stdout.log').open('xb') as stdout,(root/'stderr.log').open('xb') as stderr:
                process=subprocess.Popen(argv,cwd=snapshot,env=env,stdout=stdout,stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                job.assign(process)
                assigned=True
                remote=duplicate_query(job,process)
                publish(root/'admit.json',dict(pid=process.pid,parent_pid=os.getpid(),token=token,query_handle=remote),new=True)
                while process.poll() is None:
                    elapsed=time.monotonic()-started
                    if stop_file and Path(stop_file).exists():
                        stop=read(stop_file)
                        if stop.get('manifest_sha256')!=request['manifest_sha256']:raise ValueError('stop request bound to another manifest')
                        reason='user_stop';break
                    if resources.sample(job):reason='Windows_OS_low_memory';break
                    heartbeat=root/'heartbeat.json'
                    if not heartbeat.exists():
                        if elapsed>policy['startup_seconds']:reason='startup_watchdog';break
                    else:
                        try:age=time.time()-read(heartbeat)['unix_time']
                        except (OSError,ValueError,KeyError):age=0
                        if age>policy['heartbeat_seconds']:reason='heartbeat_watchdog';break
                    if stage_deadline is not None and elapsed>stage_deadline:reason='declared_outer_deadline';break
                    if sum(p.stat().st_size for p in (root/'stdout.log',root/'stderr.log'))>policy['max_log_bytes']:
                        reason='finite_log_budget';break
                    # Snapshot contains no evidence. Only this owned stage tree is bounded.
                    if stage_evidence_bytes(root)>policy['max_stage_evidence_bytes']:
                        reason='finite_evidence_budget';break
                    time.sleep(policy['poll_seconds'])
                if reason:job.terminate()
                process.wait(timeout=policy['cleanup_seconds'])
    except BaseException as exc:
        error=dict(type=type(exc).__name__,reason=str(exc));reason=reason or ('user_interrupt' if isinstance(exc,KeyboardInterrupt) else 'supervisor_error')
    finally:
        # Idempotent, owned Job only. No taskkill / external PID deletion.
        try:
            if process is not None and not assigned and process.poll() is None:
                # Assignment failed before the gate: terminate only the newly
                # created waiting Popen handle, which cannot yet import native code.
                process.terminate()
            job.terminate()
            if process is not None:process.wait(timeout=policy['cleanup_seconds'])
            end=time.monotonic()+policy['cleanup_seconds'];cleanup=job.observation()
            while cleanup['active_processes'] and time.monotonic()<end:
                time.sleep(.05);cleanup=job.observation()
        except BaseException as exc:
            cleanup=dict(active_processes=None,error=type(exc).__name__+': '+str(exc));reason=reason or 'cleanup_failure'
        observed=resources.summary if resources else None
        if resources:resources.close()
        job.close()
        receipt=dict(experiment=experiment['id'],operation=request['operation'],command=argv,manifest_sha256=request['manifest_sha256'],
            started_at=started_at,finished_at=now(),wall_seconds=time.monotonic()-started,pid=process.pid if process else None,
            exit_code=process.returncode if process else None,native_crash=native_exit(process.returncode) if process else False,
            reason=reason,error=error,cleanup=cleanup,resources=observed,automatic_retry=False,owned_job_only=True)
        publish(root/'supervisor-receipt.json',receipt,new=True)
    result_file=root/'result.json'
    result=read(result_file) if result_file.exists() else dict(status='FAIL',reason='missing_worker_terminal_result',metrics=[])
    if result_file.exists() and (result.get('id')!=experiment['id'] or result.get('manifest_sha256')!=request['manifest_sha256']):
        result.update(status='FAIL',reason='worker_terminal_identity_mismatch')
    if reason or receipt['exit_code']!=0 or cleanup.get('active_processes')!=0:
        result.update(status='ABORTED' if reason in ('user_stop','user_interrupt') else 'FAIL',reason=reason or 'worker_exit_or_cleanup_failure')
    result.update(attempted=process is not None,supervisor=receipt,result_file=str(result_file) if result_file.exists() else None,
                  evidence_directory=str(root))
    return result
