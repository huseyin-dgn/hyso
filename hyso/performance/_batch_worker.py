"""P17 stdlib bootstrap. Hard memory guard precedes Torch and factory import."""
import importlib
import json
import os
from pathlib import Path
import sys


def _ready(path, **fields):
    temporary=Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(dict(pid=os.getpid(),**fields)),encoding='utf-8')
    os.replace(temporary,path)


def _memory_guard(limit):
    if os.name!='nt': raise NotImplementedError('hard private-commit guard currently supports Windows only')
    import ctypes as c
    from ctypes import wintypes as w
    class IO(c.Structure):
        _fields_=[(name,c.c_ulonglong) for name in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
    class Basic(c.Structure):
        _fields_=[('process_time',c.c_longlong),('job_time',c.c_longlong),('flags',w.DWORD),
            ('min_ws',c.c_size_t),('max_ws',c.c_size_t),('process_count',w.DWORD),
            ('affinity',c.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
    class Extended(c.Structure):
        _fields_=[('basic',Basic),('io',IO),('process_memory',c.c_size_t),('job_memory',c.c_size_t),
            ('peak_process',c.c_size_t),('peak_job',c.c_size_t)]
    api=c.WinDLL('kernel32',use_last_error=True)
    api.CreateJobObjectW.argtypes=[c.c_void_p,w.LPCWSTR];api.CreateJobObjectW.restype=w.HANDLE
    api.SetInformationJobObject.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD];api.SetInformationJobObject.restype=w.BOOL
    api.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE];api.AssignProcessToJobObject.restype=w.BOOL
    api.GetCurrentProcess.argtypes=[];api.GetCurrentProcess.restype=w.HANDLE
    api.CloseHandle.argtypes=[w.HANDLE];api.CloseHandle.restype=w.BOOL
    handle=api.CreateJobObjectW(None,None)
    if not handle: raise OSError(c.get_last_error(),'CreateJobObjectW')
    info=Extended();info.basic.flags=0x100|0x200|0x8
    info.basic.process_count=1;info.process_memory=limit;info.job_memory=limit
    if not api.SetInformationJobObject(handle,9,c.byref(info),c.sizeof(info)) or not api.AssignProcessToJobObject(handle,api.GetCurrentProcess()):
        error=c.get_last_error();api.CloseHandle(handle);raise OSError(error,'worker memory guard unavailable')
    return api,handle


def main():
    request_path,output_path,ready_path,package_root=sys.argv[1:]
    _ready(ready_path,phase='bootstrap',runtime_ready=False)
    request=json.loads(Path(request_path).read_text(encoding='utf-8'))
    r=request['recipe'];p=request['protocol'];batch=request['batch'];guard=None
    result=dict(worker_pid=os.getpid(),status='failed',reason=None,actual_batch=None,
        forward_completed=False,latency_mean_ms=None,memory=None,sweep=None,provenance=None)
    try:
        guard=_memory_guard(p['cpu_memory_limit_bytes'])
        shape=list(r['input_shape']);shape[0]=batch
        import math
        input_bytes=math.prod(shape)*(8 if r['precision']=='fp64' else 4)
        if input_bytes>p['input_limit_bytes']:
            result['status']='resource_limit';raise ValueError('dense input exceeds explicit input byte budget')
        sys.path.insert(0,package_root)
        if r['factory_search_path']: sys.path.append(r['factory_search_path'])
        import torch
        from hyso.performance.sweeps import SweepProtocol,_point
        from hyso.performance.memory import _observe,_file_info
        if torch.cuda.is_initialized(): raise ValueError('CUDA initialized during module import')
        torch.set_num_threads(1);torch.manual_seed(r['seed'])
        module,qualified=r['factory'].split(':');factory=importlib.import_module(module)
        for part in qualified.split('.'): factory=getattr(factory,part)
        model=factory(**r['model_kwargs'])
        if not isinstance(model,torch.nn.Module): raise TypeError('factory must return nn.Module')
        if torch.cuda.is_initialized() or any(t.device.type!='cpu' for t in list(model.parameters())+list(model.buffers())):
            raise ValueError('factory must construct CPU model without CUDA initialization')
        if r['artifact_path']:
            info=_file_info(r['artifact_path'])
            if info['sha256']!=r['artifact_sha256']: raise ValueError('artifact SHA256 mismatch')
            state=torch.load(r['artifact_path'],map_location='cpu',weights_only=True)
            model.load_state_dict(state,strict=True);del state
        device=torch.device(r['device']);stream=None
        if device.type=='cuda':
            if not torch.cuda.is_available():
                result['status']='unsupported';raise ValueError('CUDA unavailable; no CPU fallback')
            index=0 if device.index is None else device.index
            device=torch.device('cuda',index);torch.cuda.set_device(device)
            torch.cuda.set_per_process_memory_fraction(p['cuda_memory_fraction'],device)
            stream=torch.cuda.current_stream(device)
        plan=SweepProtocol(device=str(device),precision=r['precision'],warmup_iterations=p['warmup_iterations'],
            measure_iterations=p['measure_iterations'],repeats=p['repeats'],measure_cuda_memory=device.type=='cuda')
        result['provenance']=dict(device=str(device),torch=str(torch.__version__),python=sys.version,
            cuda_build=torch.version.cuda,gpu=torch.cuda.get_device_name(device) if stream is not None else None,
            gpu_capability=list(torch.cuda.get_device_capability(device)) if stream is not None else None,
            stream_id=int(stream.cuda_stream) if stream is not None else None,
            package_source=importlib.import_module('hyso.performance').__file__,
            worker_source=__file__,factory=r['factory'],seed=r['seed'],input_shape=shape)
        _ready(ready_path,phase='runtime_prepared',runtime_ready=True,device=str(device))
        memory=dict(rss_before=_observe(),rss_after=None,budget_scope='windows_job_private_commit',
            private_commit_limit_bytes=p['cpu_memory_limit_bytes'],input_limit_bytes=p['input_limit_bytes'],
            cuda_allocator_fraction=p['cuda_memory_fraction'] if stream is not None else None,
            cuda_allocator_probe=None)
        result['memory']=memory
        point=_point(model,lambda s,d,v:torch.ones(s,dtype=d,device=v),shape,r['precision'],plan,device,stream).to_dict()
        memory['rss_after']=_observe()
        memory['cuda_allocator_probe']=point['raw_evidence']['benchmark']['cuda_memory']
        result.update(status='feasible',actual_batch=point['measurements']['batch_cardinality'],
            forward_completed=True,latency_mean_ms=point['measurements']['latency_mean_ms']['value'],sweep=point)
    except NotImplementedError as error:
        result.update(status='unsupported',reason=str(error))
    except MemoryError as error:
        result.update(status='infeasible_oom',reason='CPU MemoryError: '+str(error))
    except Exception as error:
        # Only the typed CUDA OOM class is CUDA OOM evidence. Never infer it from
        # an exit code, fail-fast, timeout, or an arbitrary error message.
        torch_module=sys.modules.get('torch')
        if torch_module is not None and isinstance(error,torch_module.cuda.OutOfMemoryError):
            result['status']='infeasible_oom'
        elif (torch_module is not None and isinstance(error,RuntimeError) and
              'DefaultCPUAllocator: not enough memory' in str(error)):
            result['status']='infeasible_oom'
        result['reason']=type(error).__name__+': '+str(error)
    Path(output_path).write_text(json.dumps(result,allow_nan=False),encoding='utf-8')
    # No work is admitted again in this process, including after an OOM.
    if guard is not None: guard[0].CloseHandle(guard[1])


if __name__=='__main__': main()
