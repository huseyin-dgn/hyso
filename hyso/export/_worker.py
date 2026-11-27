"""Owned TensorRT build/runtime; imports occur after the existing Job latch."""
import ctypes
import json
import os
from pathlib import Path
import platform
import sys
from uuid import UUID
from .artifacts import file_hash
from .contracts import NativeConfig,ExportError

def loaded_binaries():
    """Same Windows module enumeration contract used by CUDA acceptance."""
    from .._native_identity import loaded_binaries as enumerate_loaded,export_selected
    from ._dll_identity import file_identity
    try:
        hashes=enumerate_loaded(selected=export_selected)
    except Exception as exc:
        raise ExportError('DLL_INTEGRITY','loaded module enumeration failed: '+str(exc)) from exc
    result={}
    for p,sha in hashes.items():
        try:identity=file_identity(p)
        except OSError as exc:raise ExportError('DLL_INTEGRITY','loaded DLL identity unavailable: '+p) from exc
        result[p]=dict(sha256=sha,file_id=identity['file_id'],bytes=identity['size'])
    return result


def verify_loaded(native,identities):
    from ._dll_identity import compare
    actual=loaded_binaries()
    diagnostic=compare(actual,native.allowed_binaries,identities)
    if diagnostic['rejected']:
        raise ExportError('DLL_INTEGRITY','loaded native DLL hash/path/file identity differs from pre-admission manifest',result=diagnostic)
    return actual


def canonical_uuid(value):
    if isinstance(value,bytes):parsed=UUID(bytes=value) if len(value)==16 else UUID(value.decode('ascii').removeprefix('GPU-'))
    else:parsed=UUID(str(value).removeprefix('GPU-'))
    return 'GPU-'+str(parsed)

def _gpu(native,identities):
    import torch
    if os.environ.get('HYSO_RUNTIME_ADMITTED_PID')!=str(os.getpid()) or os.environ.get('CUDA_VISIBLE_DEVICES')!=native.gpu_uuid:raise ExportError('NATIVE_ADMISSION','owned process/GPU binding missing')
    if not torch.version.cuda or not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ExportError('CUDA_RUNTIME','one visible admitted CUDA device required')
    props=torch.cuda.get_device_properties(0);value=getattr(props,'uuid',None)
    if canonical_uuid(value)!=native.gpu_uuid:raise ExportError('GPU_IDENTITY','physical UUID mismatch')
    verify_loaded(native,identities)
    free,total=torch.cuda.mem_get_info(0)
    if free<native.max_gpu_bytes+256*1024**2:raise ExportError('RESOURCE_ADMISSION','insufficient physical free memory')
    torch.cuda.set_per_process_memory_fraction(native.allocator_fraction,0)
    return torch,dict(gpu_uuid=native.gpu_uuid,name=props.name,compute_capability=[props.major,props.minor],total_bytes=total,
        admitted_free_bytes=free,platform=platform.platform(),torch=str(torch.__version__),cuda_build=torch.version.cuda)

def gpu_allocator(trt,torch,limit):
    """TRT-owned requests capped separately from workspace and context memory."""
    class Allocator(trt.IGpuAllocator):
        def __init__(self):
            import threading
            super().__init__();self.values={};self.used=0;self.peak=0;self.denied=False;self.fatal_oom=False;self.lock=threading.RLock()
        def allocate(self,size,alignment,flags):
            with self.lock:
                if size==0:return 0
                if alignment<0 or alignment and alignment&(alignment-1) or alignment>256 or size<0 or self.used+size>limit:
                    self.denied=True;return 0
                try:t=torch.empty((size,),dtype=torch.uint8,device='cuda:0')
                except (MemoryError,torch.OutOfMemoryError):self.denied=True;self.fatal_oom=True;return 0
                pointer=t.data_ptr()
                if alignment and pointer%alignment:
                    self.denied=True;return 0
                self.values[pointer]=t;self.used+=size;self.peak=max(self.peak,self.used);return pointer
        def deallocate(self,memory):
            with self.lock:
                if memory in (None,0):return True
                value=self.values.pop(memory,None)
                if value is not None:self.used-=value.numel()
                return value is not None
        def reallocate(self,address,alignment,new_size):
            # Refuse resize while retaining the old allocation, per TRT contract.
            return 0
        def allocate_async(self,size,alignment,flags,stream):
            # This is a synchronous allocator. Make arbitrary TRT stream use safe
            # before publishing a Torch-owned pointer to another CUDA stream.
            torch.cuda.synchronize(0)
            return self.allocate(size,alignment,flags)
        def deallocate_async(self,memory,stream):
            torch.cuda.synchronize(0)
            return self.deallocate(memory)
    return Allocator()


def execute_engine(trt,torch,engine,request,allocator,native):
    import time
    from .runtime import decode_arrays,encode_arrays
    result={}
    context=engine.create_execution_context()
    if context is None:raise ExportError('ENGINE_LOAD','execution context unavailable')
    cpu_values=decode_arrays(request['inputs']['arrays'])
    import numpy as np
    if not cpu_values or any(not np.isfinite(t.numpy()).all() for t in cpu_values):raise ExportError('INPUT_SPEC','finite nonempty engine feeds required')
    if allocator.used+sum(v.numel()*v.element_size() for v in cpu_values)>native.max_gpu_bytes:raise ExportError('RESOURCE_BUDGET','engine input allocation would exceed admitted GPU budget')
    transfer_start=time.perf_counter_ns()
    inputs={n:t.to('cuda:0').contiguous() for n,t in zip(request['inputs']['names'],cpu_values)}
    torch.cuda.synchronize(0)
    h2d_ms=(time.perf_counter_ns()-transfer_start)/1e6
    names=[engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    expected={n for n in names if engine.get_tensor_mode(n)==trt.TensorIOMode.INPUT}
    if expected!=set(inputs):raise ExportError('INPUT_SPEC','engine input names differ')
    buffers=dict(inputs);outnames=[];total=sum(t.numel()*t.element_size() for t in inputs.values());output_bytes=0
    for n,t in inputs.items():
        if str(t.detach().cpu().numpy().dtype)!=str(__import__('numpy').dtype(trt.nptype(engine.get_tensor_dtype(n)))):raise ExportError('INPUT_SPEC','engine input dtype mismatch')
        if not context.set_input_shape(n,tuple(t.shape)):raise ExportError('INPUT_SPEC','shape outside engine profile')
    for n in names:
        if n in inputs:continue
        shape=tuple(context.get_tensor_shape(n))
        if any(v<0 for v in shape):raise ExportError('OUTPUT_SPEC','data-dependent unresolved output shape unsupported')
        import numpy as np,math
        dtype=np.dtype(trt.nptype(engine.get_tensor_dtype(n)));size=math.prod(shape)*dtype.itemsize;total+=size;output_bytes+=size
        if output_bytes>32*1024**2:raise ExportError('RESOURCE_BUDGET','native output exceeds validation/transfer budget before allocation')
        if total+allocator.used>native.max_gpu_bytes:raise ExportError('RESOURCE_BUDGET','engine I/O exceeds admitted GPU budget')
        buffers[n]=torch.empty(shape,dtype=torch.from_numpy(np.empty((),dtype=dtype)).dtype,device='cuda:0');outnames.append(n)
    for n,t in buffers.items():
        if not context.set_tensor_address(n,t.data_ptr()):raise ExportError('ENGINE_BINDING','binding rejected')
    stream=torch.cuda.current_stream(0)
    samples=[]
    warmup=request.get('warmup',0);iterations=request.get('iterations',1)
    if type(warmup) is not int or type(iterations) is not int or not 0<=warmup<=100 or not 1<=iterations<=1000:
        raise ExportError('RESOURCE_BUDGET','bounded native measurement iterations required')
    for i in range(warmup+iterations):
        begin=time.perf_counter_ns()
        if not context.execute_async_v3(stream.cuda_stream):raise ExportError('ENGINE_EXECUTION','execution rejected')
        stream.synchronize()
        if i>=warmup:samples.append((time.perf_counter_ns()-begin)/1e6)
    if set(outnames)!=set(request['output_names']):raise ExportError('OUTPUT_SPEC','engine output names differ from ONNX provenance')
    outnames=request['output_names']
    transfer_start=time.perf_counter_ns()
    host=[buffers[n].detach().cpu().numpy() for n in outnames]
    if any(a.size<1 or not np.isfinite(a).all() for a in host):raise ExportError('NONFINITE','engine output is empty or NaN/Inf')
    torch.cuda.synchronize(0)
    d2h_ms=(time.perf_counter_ns()-transfer_start)/1e6
    result['outputs']=encode_arrays(host);result['completed']=True
    result['measurement']=dict(samples_ms=samples,host_to_device_ms=h2d_ms,device_to_host_ms=d2h_ms,
        boundary='resident_enqueue_and_stream_synchronize_host_wall_time',gpu_kernel_only=False,
        gpu_vram=None,gpu_vram_reason='TRT requested allocations and Torch reserved memory are not physical per-process VRAM')
    result['io']=dict(inputs=list(inputs),outputs=outnames,stream=stream.cuda_stream,device='cuda:0')
    return result

def deployment(root,request,native):
    """Reload a verified engine within the same owned Job/DLL/budget contract."""
    import time
    import tensorrt as trt
    if not trt.__version__.startswith('10.') or trt.__version__!=request['runtime_version']:
        raise ExportError('BACKEND_VERSION','saved engine requires exact recorded TensorRT version')
    torch,environment=_gpu(native,request['_binary_file_ids'])
    source=Path(request['source_path'])
    if source.stat().st_size>256*1024**2 or file_hash(source)!=request['source_sha256']:
        raise ExportError('ARTIFACT_INTEGRITY','engine source changed or exceeds artifact budget')
    verify_loaded(native,request['_binary_file_ids'])
    allocator=gpu_allocator(trt,torch,native.max_gpu_bytes)
    logger=trt.Logger(trt.Logger.WARNING)
    start=time.perf_counter_ns();runtime=trt.Runtime(logger);runtime.gpu_allocator=allocator
    engine=runtime.deserialize_cuda_engine(source.read_bytes())
    if engine is None:raise ExportError('ENGINE_LOAD','saved engine deserialization failed')
    loading=(time.perf_counter_ns()-start)/1e6
    result=execute_engine(trt,torch,engine,request,allocator,native)
    if allocator.fatal_oom:raise MemoryError('native allocator OOM; no retry')
    if allocator.peak<=0:raise ExportError('NATIVE_ADMISSION','no allocator activity observed')
    if file_hash(source)!=request['source_sha256']:raise ExportError('ARTIFACT_INTEGRITY','engine changed during execution')
    result.update(status='PASS',environment=environment,runtime_version=trt.__version__,runtime_loading_ms=loading,
        allocator=dict(limit_bytes=native.max_gpu_bytes,peak_requested_bytes=allocator.peak,denied_request_observed=allocator.denied),
        loaded_binaries=verify_loaded(native,request['_binary_file_ids']))
    return result

def build(root,request,native):
    import time
    import tensorrt as trt
    if not trt.__version__.startswith('10.'):raise ExportError('BACKEND_VERSION','only explicit TensorRT 10.x Python API adapter is implemented')
    torch,environment=_gpu(native,request['_binary_file_ids'])
    verify_loaded(native,request['_binary_file_ids'])
    source=Path(request['source_path'])
    if file_hash(source)!=request['source_sha256']:raise ExportError('ARTIFACT_INTEGRITY','source ONNX changed after admission')
    for row in request['external_source_files']:
        if file_hash(row['path'])!=row['sha256']:raise ExportError('ARTIFACT_INTEGRITY','ONNX external data changed')
    logger=trt.Logger(trt.Logger.WARNING)
    allocator=gpu_allocator(trt,torch,native.max_gpu_bytes);builder=trt.Builder(logger);builder.gpu_allocator=allocator
    network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser=trt.OnnxParser(network,logger)
    if not parser.parse_from_file(str(source)):raise ExportError('ONNX_PARSER','; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config();config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,native.workspace_bytes)
    if hasattr(trt.BuilderFlag,'TF32'):config.clear_flag(trt.BuilderFlag.TF32)
    if request['precision']=='fp16':
        if not builder.platform_has_fast_fp16:raise ExportError('PRECISION','hardware/runtime does not expose fast FP16')
        config.set_flag(trt.BuilderFlag.FP16)
    if request['profiles']:
        profile=builder.create_optimization_profile()
        for name,row in request['profiles'].items():
            if not profile.set_shape(name,row['min'],row['opt'],row['max']):raise ExportError('OPTIMIZATION_PROFILE','TRT rejected shape bounds')
        if config.add_optimization_profile(profile)<0:raise ExportError('OPTIMIZATION_PROFILE','profile rejected')
    started=time.perf_counter_ns()
    serialized=builder.build_serialized_network(network,config)
    build_ms=(time.perf_counter_ns()-started)/1e6
    if allocator.fatal_oom:raise MemoryError('native allocator observed real OOM; no recovery claim or retry')
    if serialized is None:
        if allocator.denied:raise MemoryError('capped native allocator refused TensorRT build; no retry')
        raise ExportError('ENGINE_BUILD','TensorRT build returned no engine')
    path=root/'model.engine';path.write_bytes(bytes(serialized))
    result=dict(status='PASS',artifact_path=str(path),artifact_sha256=file_hash(path),runtime_version=trt.__version__,environment=environment,
        workspace_bytes=native.workspace_bytes,engine_build_ms=build_ms,allocator=dict(limit_bytes=native.max_gpu_bytes,peak_requested_bytes=allocator.peak),outputs=None,
        compatibility='exact recorded TRT/GPU/platform; no portable-engine claim')
    if request['inputs'] and request['validate_runtime']:
        from .runtime import decode_arrays,encode_arrays
        runtime=trt.Runtime(logger);runtime.gpu_allocator=allocator;engine=runtime.deserialize_cuda_engine(serialized)
        if engine is None:raise ExportError('ENGINE_LOAD','engine deserialization failed')
        result.update(execute_engine(trt,torch,engine,request,allocator,native))
    if allocator.peak<=0:raise ExportError('NATIVE_ADMISSION','TensorRT allocator activity was not observed; resource contract unverified')
    result['allocator'].update(denied_request_observed=allocator.denied,scope='requested TRT allocations; driver/context and allocator rounding are separate')
    result['loaded_binaries']=verify_loaded(native,request['_binary_file_ids'])
    return result

def worker(root,request):
    from ..performance.stability import _atomic
    _atomic(root/'ready.json',dict(pid=os.getpid(),token=request['token']))
    native=NativeConfig(**request['native'])
    try:
        if request['operation']=='tensorrt':report=build(root,request,native)
        elif request['operation']=='deployment':report=deployment(root,request,native)
        elif request['operation']=='dll_diagnostic':
            # No model/artifact, session, inference, conversion or tensor allocation.
            torch,environment=_gpu(native,request['_binary_file_ids'])
            if native.runtime_search_path:sys.path.insert(0,native.runtime_search_path)
            import onnxruntime as ort
            expected=Path(native.runtime_search_path)/'onnxruntime/__init__.py'
            if Path(ort.__file__).resolve()!=expected.resolve():raise ExportError('DLL_INTEGRITY','diagnostic ORT source differs from admitted runtime')
            # Explicitly load only the pre-admitted provider binary, without a session.
            provider=expected.parent/'capi/onnxruntime_providers_cuda.dll'
            from ._dll_identity import canonical_path
            if native.allowed_binaries.get(canonical_path(provider))!=file_hash(provider):raise ExportError('DLL_INTEGRITY','diagnostic provider is not pre-admitted')
            with os.add_dll_directory(str(provider.parent)):
                provider_handle=ctypes.WinDLL(str(provider))
            report=dict(status='PASS',diagnostic_only=True,inference_executed=False,conversion_executed=False,
                environment=environment,ort_version=ort.__version__,loaded_binaries=verify_loaded(native,request['_binary_file_ids']))
        elif request['operation'] in ('ort','jit'):
            torch,environment=_gpu(native,request['_binary_file_ids'])
            path=Path(request['source_path'])
            if file_hash(path)!=request['source_sha256']:raise ExportError('ARTIFACT_INTEGRITY','validation artifact changed')
            from .runtime import decode_arrays,encode_arrays,_ort_capture
            values=decode_arrays(request['inputs']['arrays'])
            if request['operation']=='ort':
                if native.runtime_search_path:sys.path.insert(0,native.runtime_search_path)
                runtime,actual=_ort_capture(path,values,request['inputs']['names'],root,request['token'],
                    provider='CUDAExecutionProvider',gpu_uuid=native.gpu_uuid,gpu_bytes=native.max_gpu_bytes)
            else:
                module=torch.jit.load(str(path),map_location='cuda:0').eval()
                with torch.no_grad():actual=module(*(v.to('cuda:0') for v in values))
                torch.cuda.synchronize(0)
                from ._model import outputs
                actual=outputs(actual);runtime=dict(runtime='torch.jit.load',completed=True,device='cuda:0')
            report=dict(status='PASS',environment=environment,runtime=runtime,source_sha256=request['source_sha256'],
                outputs=encode_arrays([v.detach().cpu().numpy() for v in actual]),loaded_binaries=verify_loaded(native,request['_binary_file_ids']))
        else:raise ExportError('UNSUPPORTED_BACKEND','unknown native export backend')
    except (MemoryError,SystemExit,KeyboardInterrupt):raise
    except Exception as exc:
        import torch
        if isinstance(exc,torch.OutOfMemoryError):raise
        code=getattr(exc,'code','NATIVE_CONVERSION_FAILED')
        if code in ('DLL_INTEGRITY','GPU_IDENTITY','NATIVE_ADMISSION','ARTIFACT_INTEGRITY'):
            import importlib.metadata
            versions={}
            for name in ('torch','onnxruntime-gpu'):
                try:versions[name]=importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:versions[name]=None
            _atomic(root/'identity-failure.json',dict(schema_version=1,code=code,reason=str(exc),
                details=getattr(exc,'result',None),pid=os.getpid(),token=request['token'],
                attempt_identity=request.get('attempt_identity'),operation=request['operation'],
                request_sha256=file_hash(root/'request.json'),runtime_versions=versions,
                loaded_runtime=dict(torch_version=str(torch.__version__),torch_source=str(Path(torch.__file__).resolve()),
                    ort_source=getattr(sys.modules.get('onnxruntime'),'__file__',None),
                    ort_version=getattr(sys.modules.get('onnxruntime'),'__version__',None)),
                expected_runtime_search_path=native.runtime_search_path,cuda_build=torch.version.cuda))
            raise
        report=dict(status='BLOCKED' if isinstance(exc,ImportError) or code in ('BACKEND_VERSION','RESOURCE_ADMISSION') else 'FAIL',code=code,reason=str(exc))
    import torch
    report.update(pid=os.getpid(),token=request['token'],request_sha256=file_hash(root/'request.json'),
        torch_version=str(torch.__version__),cuda_initialized=torch.cuda.is_initialized())
    _atomic(root/'report.json',report)
