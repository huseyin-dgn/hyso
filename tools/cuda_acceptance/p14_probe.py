"""Owned adapter for the existing two-forward P14 Nsight producer and reader."""
from pathlib import Path
import ctypes as c
import importlib.util
import os
import runpy
import subprocess
import sys
import time
import uuid

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/final_acceptance')]
from common import read_json,new_json,file_hash,digest
from campaign_controller import runtime_job,exact_job_member,resume_owned_launcher,verified_cleanup
from campaign_preflight import process_identity
from tools.cuda_acceptance.feature_evidence import SafetyViolation


def commands(tool,python,bootstrap,request,folder):
    folder=Path(folder)
    profile=[str(tool),'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
        '--pytorch=autograd-nvtx','--capture-range=cudaProfilerApi','--capture-range-end=stop',
        '--force-overwrite=false','--output='+str(folder/'p14-capture'),str(python),'-I','-S','-B',str(bootstrap),str(request)]
    export=[str(tool),'export','--type=sqlite','--output='+str(folder/'p14-export.sqlite'),str(folder/'p14-capture.nsys-rep')]
    return profile,export


def _owned(argv,folder,label,config,seconds,request=None):
    # Reuse the controller's exact Job, suspended-launch, readback and cleanup
    # contracts. Nsight's descendants inherit this no-breakaway Job.
    job=runtime_job(config);process=None;binding=None;cleanup=None;error=None;reaped=False;target=None
    try:
        with (folder/(label+'.stdout.log')).open('xb') as out,(folder/(label+'.stderr.log')).open('xb') as err:
            process=subprocess.Popen(argv,stdout=out,stderr=err,cwd=REPO,env=dict(os.environ),
                creationflags=subprocess.CREATE_NO_WINDOW|0x4)
            job.assign(process)
            binding=dict(identity=process_identity(process.pid),launcher_pid=process.pid,created_suspended=True,
                exact_job_membership_before_resume=exact_job_member(job,process._handle),limits_readback=True,no_breakaway=True)
            resume_owned_launcher(job,process);deadline=time.monotonic()+seconds
            while process.poll() is None:
                ready=folder/'interpreter-ready.json'
                if request is not None and target is None and ready.is_file():
                    value=read_json(ready);api=job.api
                    api.OpenProcess.argtypes=[c.c_ulong,c.c_int,c.c_ulong];api.OpenProcess.restype=c.c_void_p
                    handle=api.OpenProcess(0x1000|0x400|0x100000,False,value['pid'])
                    if not handle:raise SafetyViolation('P14_TARGET_HANDLE_UNAVAILABLE')
                    try:
                        target=process_identity(value['pid'])
                        if (value['token']!=request['token'] or not exact_job_member(job,handle) or
                            target['creation_filetime']<binding['identity']['creation_filetime'] or
                            Path(target['image']).resolve()!=Path(request['python']).resolve()):
                            raise SafetyViolation('P14_TARGET_EXACT_JOB_OR_IDENTITY_MISMATCH')
                        new_json(folder/'target-admission.json',dict(identity=target,token_sha256=digest(request['token']),
                            exact_job_membership=True,no_breakaway=True,limits_readback=True,kill_on_close=True))
                        new_json(folder/'admit.json',dict(pid=target['pid'],token=request['token'],readback_verified=True))
                    finally:api.CloseHandle(handle)
                if time.monotonic()>deadline:raise SafetyViolation('P14_OWNED_TIMEOUT')
                if any(p.stat().st_size>256*1024**2 for p in folder.iterdir() if p.is_file()):raise SafetyViolation('P14_EVIDENCE_BUDGET')
                time.sleep(.05)
            code=process.returncode
    finally:
        try:
            job.terminate()
            if process is not None:process.wait(timeout=15);reaped=True
            deadline=time.monotonic()+15
            while True:
                cleanup=job.observation()
                if cleanup['active_processes']==0 or time.monotonic()>=deadline:break
                time.sleep(.05)
        except BaseException as exc:error=repr(exc)
        finally:job.close()
        good=verified_cleanup(cleanup,binding,launcher_wait_verified=reaped,cleanup_error=error)
        receipt=dict(launch_binding=binding,cleanup=cleanup,cleanup_verified=good,target=target,
            limits=dict(process_bytes=config['process_memory_bytes'],job_bytes=config['job_memory_bytes'],max_processes=config['max_processes']))
        new_json(folder/(label+'-controller.json'),receipt)
        if not good:raise SafetyViolation('P14_OWNED_CLEANUP_UNVERIFIED')
    if code<0 or code>=0x80000000:raise SafetyViolation('P14_NATIVE_PROCESS_FAILURE: '+str(code))
    if code!=0:raise RuntimeError('P14 '+label+' exit '+str(code)+'; inspect retained logs')
    if request is not None and target is None:raise SafetyViolation('P14_TARGET_NOT_ADMITTED')
    return receipt


def _existing_script(folder,sqlite=None):
    script=REPO/'tests/cuda_p14_nsys_acceptance.py';prior=sys.argv
    sys.argv=[str(script),'--output',str(folder)]+(['--sqlite',str(sqlite)] if sqlite is not None else [])
    try:runpy.run_path(str(script),run_name='__main__')
    finally:sys.argv=prior


def target(folder):
    request=read_json(folder/'request.json');admission=read_json(folder/'target-admission.json')
    actual=process_identity(os.getpid())
    if (admission['identity']!=actual or not admission['exact_job_membership'] or not admission['no_breakaway']
        or not admission['limits_readback'] or admission['token_sha256']!=digest(request['token'])):
        raise SafetyViolation('P14_PRE_IMPORT_ADMISSION_UNVERIFIED')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=request['gpu_uuid']:raise SafetyViolation('P14_VISIBLE_UUID_MISMATCH')
    if file_hash(REPO/'tests/cuda_p14_nsys_acceptance.py')!=request['producer_sha256']:raise SafetyViolation('P14_PRODUCER_CHANGED')
    import torch
    from tools.cuda_acceptance.identity import gpu_uuid
    from tools.cuda_acceptance.worker import binary_identity
    torch.cuda.set_per_process_memory_fraction(.15,0)
    if gpu_uuid(torch.cuda.get_device_properties(0).uuid)!=request['gpu_uuid']:raise SafetyViolation('P14_PHYSICAL_UUID_MISMATCH')
    _existing_script(folder)
    new_json(folder/'target-runtime.json',dict(uuid=request['gpu_uuid'],process=actual,device_index=0,
        torch_version=str(torch.__version__),cuda_build=torch.version.cuda,allocator_fraction=.15,
        loaded_binaries={name:binary_identity(name) for name in request['binary_expectations']}))


def capture(ctx):
    from .cases import MissingEvidence
    tool=ctx.plan['feature_tools']['nsys']
    if not tool or not Path(tool['path']).is_file():raise MissingEvidence('Pinned Nsight Systems unavailable')
    if file_hash(tool['path'])!=tool['sha256']:raise SafetyViolation('P14_NSYS_TOOL_CHANGED')
    folder=ctx.folder/'p14-nsys';folder.mkdir()
    bootstrap=folder/'bootstrap.py';bootstrap.write_bytes((REPO/'outputs/release-acceptance-20261125/bootstrap.py').read_bytes())
    request=dict(token=uuid.uuid4().hex,python=ctx.plan['runtime']['python'],gpu_uuid=ctx.uuid,
        binary_expectations=ctx.plan['runtime']['binary_expectations'],runtime_paths=[ctx.plan['runtime']['site']],
        producer_sha256=ctx.plan['source_sha256']['tests/cuda_p14_nsys_acceptance.py'],
        argv=[str(Path(__file__).resolve()),'--target',str(folder)])
    new_json(folder/'request.json',request)
    profile,export=commands(tool['path'],request['python'],bootstrap,folder/'request.json',folder)
    config=read_json(REPO/'tools/final_acceptance/campaign_config.json')
    profile_receipt=_owned(profile,folder,'profile',config,90,request)
    export_receipt=_owned(export,folder,'export',config,45)
    sqlite=folder/'p14-export.sqlite'
    if sqlite.stat().st_size>256*1024**2:raise SafetyViolation('P14_SQLITE_BUDGET')
    _existing_script(folder,sqlite)
    result=read_json(folder/'operators-nsys.json')
    value=dict(result=result,target_runtime=read_json(folder/'target-runtime.json'),
        profile_controller=profile_receipt,export_controller=export_receipt,
        tool=tool,report=dict(path=str(folder/'p14-capture.nsys-rep'),sha256=file_hash(folder/'p14-capture.nsys-rep')))
    verify_capture(value,ctx.runtime)
    return value


def verify_capture(value,runtime):
    result=value['result'];target=value['target_runtime']
    if target['uuid']!=runtime['uuid'] or target['device_index']!=0 or target['loaded_binaries']!=runtime['loaded_binaries']:
        raise ValueError('P14 native runtime binding')
    if target['torch_version']!=runtime['torch_version'] or target['cuda_build']!=runtime['cuda_build'] or target['allocator_fraction']!=.15:
        raise ValueError('P14 native runtime version/allocator binding')
    if value['profile_controller']['target']!=target['process']:raise ValueError('P14 target identity binding')
    for name in ('profile_controller','export_controller'):
        receipt=value[name]
        if not receipt['cleanup_verified'] or receipt['cleanup']['active_processes']!=0:raise ValueError('P14 Job cleanup')
        if receipt['limits']!=dict(process_bytes=3*1024**3,job_bytes=6*1024**3,max_processes=48):raise ValueError('P14 Job limit binding')
    if result['capture']['worker_pid']!=target['process']['pid']:raise ValueError('P14 capture PID binding')
    spec=importlib.util.spec_from_file_location('p14_offline_reader',REPO/'hyso/performance/_operator_activity.py')
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    replay=reader.analyze(result['source_sqlite'],result['source_capture'])
    if replay!=result:raise ValueError('P14 saved operator/launch/kernel replay mismatch')
    if result['status']!='complete' or result['mapped_kernel_count']!=result['kernel_count'] or result['kernel_count']!=6:
        raise ValueError('P14 needs six real mapped kernels from two completed forwards')
    if any(k['duration_us']<=0 or not k['operator_chain_nvtx_ids'] for k in result['kernels']):raise ValueError('P14 kernel timing/ownership')
    if file_hash(value['report']['path'])!=value['report']['sha256']:raise ValueError('P14 original native report hash')


if __name__=='__main__':
    # Nsight target arrives through the unchanged stdlib admission bootstrap.
    if sys.argv[1]!='--target':raise SystemExit('P14 target only')
    target(Path(sys.argv[2]))
