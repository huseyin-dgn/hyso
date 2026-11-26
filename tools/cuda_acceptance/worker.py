"""CUDA payload: imports Torch only after exact owned-Job admission."""
from pathlib import Path
import ctypes as c
import os
import sys
import traceback

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/final_acceptance')]
from common import Journal,read_json,new_json,file_hash,digest
from campaign_preflight import process_identity
from tools.cuda_acceptance.identity import gpu_uuid


def binary_identity(name):
    api=c.WinDLL('kernel32',use_last_error=True)
    api.GetModuleHandleW.argtypes=[c.c_wchar_p];api.GetModuleHandleW.restype=c.c_void_p
    api.GetModuleFileNameW.argtypes=[c.c_void_p,c.c_wchar_p,c.c_ulong]
    handle=api.GetModuleHandleW(name)
    if not handle:raise RuntimeError('LOADED_BINARY_UNVERIFIED: '+name)
    buf=c.create_unicode_buffer(32768)
    if not api.GetModuleFileNameW(handle,buf,len(buf)):raise OSError('module path unavailable')
    path=Path(buf.value)
    return dict(path=os.path.normcase(str(path.resolve())),sha256=file_hash(path))


def run(request,folder,journal):
    if request.get('profile')!='cuda-functional' or request.get('cuda_authorized') is not True:raise RuntimeError('CUDA_AUTHORIZATION_MISSING')
    admit=read_json(folder/'admit.json');admission=read_json(folder/'admission-receipt.json')
    ids=admission['identities'];actual=process_identity(os.getpid())
    if not (admit.get('readback_verified') and admit['pid']==os.getpid() and admit['token']==request['token'] and ids['exact_job_membership'] and ids['interpreter']==actual and ids['nonce_sha256']==digest(request['token'])):
        raise RuntimeError('PRE_IMPORT_OWNERSHIP_UNVERIFIED')
    plan=request['gpu_plan'];uuid=plan['gpu_uuid'];budget=plan['budgets']
    if uuid!=request['gpu_admission']['uuid'] or not (admission.get('limits_readback') and admission.get('no_breakaway') and admission.get('kill_on_close')):raise RuntimeError('PRE_IMPORT_LIMIT_OR_GPU_MISMATCH')
    journal.emit('worker_start',pid=os.getpid(),profile=request['profile'])
    # The CPU runner remains CUDA-invisible. Only this admitted payload sets visibility.
    os.environ['CUDA_VISIBLE_DEVICES']=uuid
    import torch
    expected=Path(plan['runtime']['site']).resolve()
    if not Path(torch.__file__).resolve().is_relative_to(expected) or not torch.version.cuda:raise RuntimeError('CUDA_RUNTIME_SOURCE_MISMATCH')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('CUDA_DEVICE_UNAVAILABLE')
    torch.cuda.set_per_process_memory_fraction(budget['cuda_allocator_fraction'],0)
    props=torch.cuda.get_device_properties(0)
    torch_uuid=getattr(props,'uuid',None)
    if gpu_uuid(torch_uuid)!=uuid:raise RuntimeError('CUDA_PHYSICAL_UUID_MISMATCH')
    probe=torch.ones(1,device='cuda:0');assert (probe+1).item()==2;del probe
    torch.cuda.synchronize()
    runtime=dict(uuid=uuid,device='cuda:0',driver=request['gpu_admission']['driver'],cuda_build=torch.version.cuda,
        torch_version=str(torch.__version__),torch_source=torch.__file__,python=sys.executable,
        base_python=getattr(sys,'_base_executable',sys.executable),process=actual,
        allocator_fraction=budget['cuda_allocator_fraction'],driver_binary=binary_identity('nvcuda.dll'),
        torch_cuda_binary=binary_identity('torch_cuda.dll'),total_device_bytes=props.total_memory,
        WDDM_process_VRAM_bytes=None,WDDM_reason='Not inferred: global NVML readings and Torch allocator are distinct scopes.',
        admission=request['gpu_admission'])
    runtime['torch_uuid_display']=str(torch_uuid)
    runtime['probe']=dict(completed=True,input=1,output=2,elements=1)
    runtime['loaded_binaries']={name:binary_identity(name) for name in plan['runtime']['binary_expectations']}
    if any(runtime['loaded_binaries'][name]!=expected for name,expected in plan['runtime']['binary_expectations'].items()):
        raise RuntimeError('LOADED_RUNTIME_BINARY_MISMATCH')
    if str(torch.__version__)!=plan['runtime']['versions']['torch']:raise RuntimeError('TORCH_VERSION_MISMATCH')
    new_json(folder/'gpu-runtime.json',runtime);runtime_sha=file_hash(folder/'gpu-runtime.json')
    journal.emit('gpu_identity',sha256_file=runtime_sha)
    from tools.cuda_acceptance.cases import GROUPS,MissingEvidence
    from tools.cuda_acceptance.feature_evidence import verify_feature,SafetyViolation
    class Context:
        def __init__(self):self.folder=folder;self.uuid=uuid;self.plan=plan;self.runtime=runtime;self.cache={};self.failures={}
        def shared(self,key,fn):
            if key in self.failures:raise MissingEvidence('shared capture failed: '+self.failures[key])
            if key not in self.cache:
                try:self.cache[key]=fn()
                except Exception as exc:self.failures[key]=str(exc);raise
            return self.cache[key]
        def run(self,feature,fn,scope='bounded_native_CUDA_measurement'):
            if feature not in request['features']:return
            journal.emit('feature_start',feature=feature)
            status='PASS';reason=None;data=None
            try:
                data=fn();torch.cuda.synchronize();verify_feature(feature,data,runtime)
            except (MissingEvidence,NotImplementedError) as exc:status='BLOCKED';reason=str(exc)
            except Exception as exc:
                reason=traceback.format_exc();status='FAIL'
                if isinstance(exc,(torch.cuda.OutOfMemoryError,SafetyViolation)) or any(x in str(exc).lower() for x in (
                        'device-side assert','illegal memory access','context is destroyed','launch failure',
                        'owned_cleanup_unverified','owned process tree','owned_safety','p05_nested_safety','owned pipe reader did not terminate')):
                    journal.emit('critical_error',feature=feature,reason=reason);raise
                # A normal assertion/API failure must still leave a healthy CUDA context.
                torch.cuda.synchronize()
            new_json(folder/(feature+'.json'),dict(feature=feature,status=status,reason=reason,scope=scope,
                runtime_sha256=runtime_sha,data=data,allocator_allocated_bytes=torch.cuda.memory_allocated(0),allocator_reserved_bytes=torch.cuda.memory_reserved(0)))
            journal.emit('feature_end',feature=feature,evidence_sha256=file_hash(folder/(feature+'.json')))
    ctx=Context()
    if request['group']=='01-admission':ctx.run('ADMISSION',lambda:runtime,scope='exact_Job_identity_runtime_binary_and_bounded_probe')
    else:GROUPS[request['group']](ctx)
    journal.emit('gpu_session_finish',cuda_initialized=torch.cuda.is_initialized())
    return 0


def main():
    path=Path(sys.argv[1]);request=read_json(path);identity={k:request[k] for k in ('attempt_id','group','source_commit','environment_id')}
    journal=Journal(path.parent/'events.jsonl',identity)
    try:return run(request,path.parent,journal)
    except BaseException:
        if not journal.failed:journal.emit('critical_error',reason=traceback.format_exc())
        raise
    finally:
        journal.close()
        new_json(path.parent/'journal-terminal.json',dict(schema_version=2,**identity,committed_sequence=journal.sequence,
            committed_head_sha256=journal.previous,writer_failed=journal.failed,closed=True,protocol='hyso.event-journal.v2'))


if __name__=='__main__':raise SystemExit(main())
