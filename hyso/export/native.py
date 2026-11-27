"""TensorRT/AOT adapters behind the existing owned Job and explicit budgets."""
from __future__ import annotations
from dataclasses import asdict
import importlib.util
import json
import platform
from pathlib import Path
from typing import Any,Literal
from contextlib import contextmanager
from .contracts import ExportConfig,NativeConfig,ExportError,ExportResult,STAGES,seal,availability
from .artifacts import Transaction,file_hash,safe_file

def blocked(code,reason,backend,stage='dependencies'):
    body=dict(schema_version=1,format='hyso.export',backend=backend,stages={s:dict(status='NOT_RUN') for s in STAGES},deployment_verified=False)
    body['stages'][stage]=dict(status='BLOCKED',reason=reason)
    raise ExportError(code,reason,result=body)

def validate_profiles(profiles,shape_map):
    if type(profiles) is not dict or set(profiles)-set(shape_map):raise ExportError('OPTIMIZATION_PROFILE','unknown profile input')
    for name,shape in shape_map.items():
        row=profiles.get(name)
        if row is None:
            if any(s<0 for s in shape):raise ExportError('OPTIMIZATION_PROFILE','every dynamic input needs min/opt/max')
            continue
        if type(row) is not dict or set(row)!={'min','opt','max'}:raise ExportError('OPTIMIZATION_PROFILE','min/opt/max required')
        if any(type(v) not in (tuple,list) or len(v)!=len(shape) for v in row.values()):raise ExportError('OPTIMIZATION_PROFILE','rank mismatch')
        for index,static in enumerate(shape):
            low,opt,high=(row[k][index] for k in ('min','opt','max'))
            if any(type(v) is not int or v<1 for v in (low,opt,high)) or not low<=opt<=high or static>0 and (low!=static or high!=static):raise ExportError('OPTIMIZATION_PROFILE','invalid static/dynamic bounds')
    return profiles

@contextmanager
def gpu_lock(uuid):
    """Nonblocking per-user/GPU export lock; never queues a risky native retry."""
    import os,msvcrt
    base=Path(os.environ['LOCALAPPDATA'])/'HYSo/export-locks';base.mkdir(parents=True,exist_ok=True)
    path=base/(uuid+'.lock');stream=path.open('a+b');acquired=False
    try:
        try:
            if os.fstat(stream.fileno()).st_size==0:stream.write(b'0');stream.flush()
            stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1);acquired=True
        except OSError:blocked('GPU_BUSY','another export owns this GPU; no automatic retry','native','admission')
        yield str(path)
    finally:
        if acquired:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        stream.close()

def run_owned(mode,root,request,native,*,failure_directory=None):
    if type(native) is not NativeConfig:blocked('NATIVE_ADMISSION','NativeConfig required',mode,'admission')
    import os
    if os.name!='nt':blocked('PLATFORM','owned native adapter is Windows-only in this release',mode,'admission')
    identities={}
    for path,sha in native.allowed_binaries.items():
        if file_hash(path)!=sha:raise ExportError('DLL_INTEGRITY','pre-admission binary differs: '+path)
        stat=Path(path).stat();identities[os.path.normcase(str(Path(path).resolve()))]=dict(file_id=stat.st_ino,bytes=stat.st_size)
        expected=native.dependency_identities.get(path)
        if expected is not None and expected!=dict(file_id=stat.st_ino,bytes=stat.st_size):
            raise ExportError('DLL_INTEGRITY','pre-admission dependency file identity differs: '+path)
    from ..performance.stability import _run_owned
    protocol=dict(process_memory_bytes=native.process_memory_bytes,job_memory_bytes=native.job_memory_bytes,max_processes=native.max_processes)
    try:
        with gpu_lock(native.gpu_uuid) as lock:
            folder=_run_owned(root,'export',dict(operation=mode,recipe={'gpu_uuid':native.gpu_uuid},native=asdict(native),protocol=protocol,_binary_file_ids=identities,host_gpu_lock=lock,**request),
                startup_timeout=45,heartbeat_timeout=native.timeout_seconds,max_runtime=native.timeout_seconds,max_bytes=512*1024**2)
    except ExportError:raise
    except Exception as exc:
        raise ExportError('NATIVE_SAFETY_STOP','owned native launch/admission/cleanup could not be verified: '+str(exc)) from exc
    controller=json.loads((folder/'controller.json').read_text())
    if controller['exit_code']!=0 or controller['reason'] is not None or controller['cleanup']['active_processes']!=0:
        from ._dll_identity import failure_bundle,retain
        bundle=failure_bundle(folder,controller)
        evidence=dict(controller=controller,failure_bundle=bundle)
        diagnostic=bundle['records'].get('identity-failure.json')
        if diagnostic and json.loads(diagnostic['text']).get('code')=='DLL_INTEGRITY':
            evidence.update(failure_stage='admission',deployment_verified=False,
                stages={s:dict(status='FAIL',reason='DLL_INTEGRITY') if s=='admission' else dict(status='NOT_RUN') for s in STAGES})
        if failure_directory is not None:evidence['retained_failure']=retain(failure_directory,bundle)
        raise ExportError('NATIVE_SAFETY_STOP','native worker/cleanup failed; no automatic retry',result=evidence)
    report=json.loads((folder/'report.json').read_text())
    readback=controller.get('limit_readback') or {}
    if not (readback.get('verified') and readback.get('no_breakaway') and readback.get('kill_on_close') and
            readback.get('process_memory_bytes')==native.process_memory_bytes and readback.get('job_memory_bytes')==native.job_memory_bytes and
            readback.get('max_processes')==native.max_processes and report.get('token')==controller['token'] and
            report.get('pid')==controller['launcher_pid'] and report.get('request_sha256')==file_hash(folder/'request.json')):
        raise ExportError('NATIVE_SAFETY_STOP','native receipt/process/request/readback identity mismatch',result={'controller':controller,'report':report})
    if report['status']!='PASS':
        code=report.get('code','NATIVE_FAILED')
        if report['status']=='BLOCKED':blocked(code,report['reason'],mode,'admission')
        raise ExportError(code,report['reason'],result={'controller':controller,'report':report})
    return report,controller

def validate_native(kind,path,example,input_names,tx,native):
    """Bind actual same-input CUDA outputs to an owned worker receipt."""
    from .runtime import encode_arrays,decode_arrays
    from ._model import tensors
    payload=dict(source_path=str(path),source_sha256=file_hash(path),
        inputs=dict(names=input_names,arrays=encode_arrays([t.detach().cpu().numpy() for t in tensors(example)])))
    report,controller=run_owned(kind,tx.root/('native-'+tx.token),dict(payload,attempt_identity=str(tx.output.parent)),native,failure_directory=tx.output.parent)
    return dict(report=report,controller=controller),decode_arrays(report['outputs'])

def tensorrt(model_path: str | Path | ExportResult, output_path: str | Path, *,
             precision: Literal['fp32','fp16','int8']='fp32',profiles: dict[str,dict[str,list[int]]] | None=None,
             example_input: dict[str,Any] | None=None, native: NativeConfig | None=None,
             config: ExportConfig=ExportConfig()) -> ExportResult:
    """Build an engine from checked ONNX in an owned worker; never silent fallback.

    TensorRT 10.x Python name-based I/O and explicit workspace are supported.
    NativeConfig requires predeclared runtime DLL hashes, physical UUID and budgets.
    Fixed shapes or complete min/opt/max profiles are required. INT8 requires
    explicit static signed symmetric QDQ; dynamic integer operators are blocked.
    Supplying example_input enables ONNX CPU vs actual TensorRT output equivalence;
    without it conversion can succeed but deployment remains unverified. Workspace
    is separate from the capped GPU allocator budget. Engine portability is limited
    to the recorded GPU/platform/TRT version. No runtime/SDK is installed.

    Args:
        model_path (str | Path | ExportResult): Existing inference model artifact path; source
            verification precedes runtime use. Required.
        output_path (str | Path): Destination for a separate inference artifact and sidecars; existing
            outputs are refused unless config.overwrite permits replacement. Required.
        precision (Literal['fp32', 'fp16', 'int8']): Requested fp32/fp16/int8 numerical policy;
            precision labels do not prove hardware instruction selection. Default: 'fp32'.
        profiles (dict[str, dict[str, list[int]]] | None): Named dynamic input min/opt/max shape bounds;
            calibration feeds must lie inside the declared range. Default: None.
        example_input (dict[str, Any] | None): Representative input tensor or ordered tensor collection
            for this explicit operation; source model/device must match. Default: None.
        native (NativeConfig | None): Explicit NativeConfig for owned GPU execution. Omission grants no
            accelerator admission; experimental support limits remain in force. Default: None.
        config (ExportConfig): Configuration object or versioned mapping accepted by this API; loading
            settings alone does not execute a workload. Default: ExportConfig().

    HYSo Guide: export.overview
    """
    if precision not in ('fp32','fp16','int8'):raise ExportError('PRECISION','fp32/fp16 or explicit INT8 QDQ required')
    if type(config) is not ExportConfig:raise TypeError('ExportConfig required')
    if importlib.util.find_spec('tensorrt') is None:blocked('MISSING_DEPENDENCY','TensorRT is not installed; no engine build started','tensorrt')
    if type(native) is not NativeConfig:blocked('NATIVE_ADMISSION','explicit NativeConfig required','tensorrt','admission')
    import onnx
    if isinstance(model_path,ExportResult):
        model_path.verify();source=model_path.to_dict();path=Path(source['artifact_path'])
        if source['artifact_format']!='onnx':raise ExportError('ARTIFACT_FORMAT','ONNX source required')
    else:
        path=Path(model_path).resolve(strict=True);source=dict(artifact_path=str(path),sha256=file_hash(path))
    # Validate external references before ONNX is permitted to read them.
    if path.stat().st_size>config.max_model_bytes:blocked('RESOURCE_BUDGET','ONNX source exceeds bounded parser budget','tensorrt','admission')
    proto=onnx.load(str(path),load_external_data=False);external=[];external_seen=set()
    from .artifacts import onnx_tensors
    for tensor in onnx_tensors(proto):
        for item in tensor.external_data:
            if item.key=='location':
                p=safe_file(path.parent,item.value)
                if str(p) not in external_seen:external.append(dict(path=str(p),sha256=file_hash(p)));external_seen.add(str(p))
    onnx.checker.check_model(str(path),full_check=True)
    if precision=='int8':
        import numpy as np
        nodes=list(proto.graph.node);ops={n.op_type for n in nodes}
        initializers={t.name:t for t in proto.graph.initializer}
        if not {'QuantizeLinear','DequantizeLinear'}<=ops or ops & {'DynamicQuantizeLinear','MatMulInteger'}:
            blocked('QUANTIZATION','TensorRT INT8 requires static explicit QDQ; dynamic integer operators are unsupported','tensorrt')
        for node in nodes:
            if node.op_type not in ('QuantizeLinear','DequantizeLinear'):continue
            if len(node.input)<3 or node.input[2] not in initializers:
                blocked('QUANTIZATION','explicit signed symmetric INT8 zero points required','tensorrt')
            zero=onnx.numpy_helper.to_array(initializers[node.input[2]])
            if zero.dtype!=np.int8 or np.any(zero!=0):blocked('QUANTIZATION','TensorRT QDQ requires signed symmetric INT8','tensorrt')
    if any(v.type.tensor_type.elem_type not in (onnx.TensorProto.FLOAT,onnx.TensorProto.FLOAT16) for v in proto.graph.input):blocked('BACKEND_DTYPE','bounded TensorRT adapter accepts FP32/FP16 tensor inputs only','tensorrt','admission')
    shape_map={v.name:[d.dim_value if d.dim_value>0 else -1 for d in v.type.tensor_type.shape.dim] for v in proto.graph.input}
    validate_profiles(profiles or {},shape_map)
    if path.stat().st_size+sum(Path(r['path']).stat().st_size for r in external)>config.max_model_bytes:blocked('RESOURCE_BUDGET','ONNX source exceeds bounded native build profile','tensorrt','admission')
    body=dict(schema_version=1,format='hyso.export',conversion='onnx_to_tensorrt',backend='tensorrt',source_artifact=source,external_source_files=external,
        config=asdict(config),options=dict(precision=precision,profiles=profiles or {}),stages={s:dict(status='NOT_RUN') for s in STAGES},warnings=[],limitations=['Engine is specific to recorded platform/GPU/TRT; workspace is not total physical VRAM'])
    with Transaction(output_path,config.overwrite,config.max_artifact_bytes) as tx:
        from .runtime import encode_arrays,decode_arrays
        import numpy as np
        inputs=None
        if example_input is not None:
            if set(example_input)!=set(shape_map):raise ExportError('INPUT_SPEC','named inputs must cover engine inputs exactly')
            if any(hasattr(v,'detach') and v.device.type!='cpu' for v in example_input.values()):raise ExportError('INPUT_SPEC','CPU feeds required; CUDA input transfer before admission is forbidden')
            arrays=[v.detach().numpy() if hasattr(v,'detach') else np.asarray(v) for v in example_input.values()]
            if sum(a.nbytes for a in arrays)>config.max_input_bytes:raise ExportError('RESOURCE_BUDGET','native inputs exceed configured CPU staging budget')
            specifications={v.name:v.type.tensor_type for v in proto.graph.input}
            for name,a in zip(example_input,arrays):
                shape=shape_map[name];expected=np.dtype(onnx.helper.tensor_dtype_to_np_dtype(specifications[name].elem_type))
                if a.dtype!=expected or a.dtype.hasobject or a.ndim!=len(shape) or any(s<=0 or fixed>0 and s!=fixed for s,fixed in zip(a.shape,shape)) or not np.isfinite(a).all():raise ExportError('INPUT_SPEC','finite ONNX-matching shape/dtype required')
                if name in (profiles or {}) and any(not low<=s<=high for s,low,high in zip(a.shape,profiles[name]['min'],profiles[name]['max'])):raise ExportError('INPUT_SPEC','input outside TensorRT optimization profile')
            inputs=dict(names=list(example_input),arrays=encode_arrays(arrays))
        estimate=native.workspace_bytes+path.stat().st_size+sum(Path(r['path']).stat().st_size for r in external)+32*1024**2+(sum(a.nbytes for a in arrays)*3 if inputs else 0)
        if estimate>native.max_gpu_bytes:blocked('RESOURCE_BUDGET','workspace + source + I/O estimate exceeds admitted allocator budget','tensorrt','admission')
        report,controller=run_owned('tensorrt',tx.root/('native-'+tx.token),dict(source_path=str(path),source_sha256=file_hash(path),external_source_files=external,
            precision=precision,profiles=profiles or {},output_names=[v.name for v in proto.graph.output],inputs=inputs,validate_runtime=config.validate_runtime),native)
        if file_hash(path)!=source.get('sha256',next((r['sha256'] for r in source.get('artifact_files',[]) if r['path']==path.name),None)) or any(file_hash(r['path'])!=r['sha256'] for r in external):raise ExportError('ARTIFACT_INTEGRITY','ONNX source changed during native conversion')
        generated=Path(report['artifact_path'])
        if generated.resolve().parent!=(tx.root/('native-'+tx.token)).resolve() or file_hash(generated)!=report['artifact_sha256']:raise ExportError('ARTIFACT_INTEGRITY','native artifact identity mismatch')
        generated.rename(tx.root/tx.output.name)
        body.update(artifact_path=str(tx.output),artifact_root=str(tx.parent),artifact_format='tensorrt_engine',artifact_files=tx.inventory(),runtime=report,controller=controller,
            backend_version=report['runtime_version'],environment=report['environment'],
            precision=dict(requested=precision,scope='explicit_qdq' if precision=='int8' else 'builder_flag' if precision=='fp16' else 'fp32_tf32_disabled',all_operations_in_requested_dtype=False))
        main=next(r for r in body['artifact_files'] if r['path']==tx.output.name)
        body.update(artifact_sha256=main['sha256'],artifact_bytes=main['bytes'],resource_estimate_bytes=estimate)
        body['inputs']=[dict(name=v.name,shape=shape_map[v.name],dtype=str(onnx.helper.tensor_dtype_to_np_dtype(v.type.tensor_type.elem_type))) for v in proto.graph.input]
        body['outputs']=[dict(name=v.name,shape=[d.dim_value if d.dim_value>0 else -1 for d in v.type.tensor_type.shape.dim],dtype=str(onnx.helper.tensor_dtype_to_np_dtype(v.type.tensor_type.elem_type))) for v in proto.graph.output]
        for stage in ('dependencies','admission','conversion','artifact','preservation'):body['stages'][stage]=dict(status='PASS')
        if report.get('outputs') is not None:
            from .runtime import ort_cpu
            from ._model import equivalence
            import torch
            examples=tuple(torch.from_numpy(np.frombuffer(__import__('base64').b64decode(r['data']),dtype=np.dtype(r['dtype'])).reshape(r['shape']).copy()) for r in inputs['arrays'])
            baseline_runtime,baseline=ort_cpu(path,examples,inputs['names'],tx.root,tx.token)
            body['onnx_reference_runtime']=baseline_runtime
            body['stages']['runtime_loading']=dict(status='PASS');body['stages']['runtime_execution']=dict(status='PASS')
            body['stages']['equivalence']=dict(status='PASS',comparisons=equivalence(baseline,decode_arrays(report['outputs']),config))
            body['reference_outputs']=encode_arrays([v.detach().cpu().numpy() for v in baseline]);body['actual_outputs']=report['outputs']
        else:body['stages']['runtime_execution']=dict(status='BLOCKED',reason='No explicit runtime input/equivalence request')
        # Filesystem cleanup is required before the final verified record.
        body['stages']['cleanup']=dict(status='NOT_RUN');tx.publish(seal(body))
    body['stages']['cleanup']=dict(status='PASS',active_processes=0,owned_staging_removed=True)
    result=seal(body);result.to_json(tx.manifest,overwrite=True);return result

def pytorch_tensorrt(model: Any, example_input: Any, output_path: str | Path, *,
                   backend: Literal['onnx','torch_tensorrt']='onnx',onnx_artifact: ExportResult | None=None,
                   retain_intermediate: bool=True,native: NativeConfig | None=None,
                   config: ExportConfig=ExportConfig(),precision: Literal['fp32','fp16','int8']='fp32',
                   profiles: dict[str,dict[str,list[int]]] | None=None) -> ExportResult:
    """Explicit PyTorch→checked ONNX→TensorRT chain with retained provenance.

    Existing ONNX may be reused only with matching source weights and input content.
    Direct Torch-TensorRT compilation is a distinct unavailable backend in this
    profile. Intermediate retention is mandatory in E04; unverified cleanup is never
    attempted. An absent TensorRT dependency is detected before any repeated export.

    Args:
        model (Any): Caller-owned torch.nn.Module; must meet the operation device, dtype and input
            contract. Required.
        example_input (Any): Representative input tensor or ordered tensor collection for this explicit
            operation; source model/device must match. Required.
        output_path (str | Path): Destination for a separate inference artifact and sidecars; existing
            outputs are refused unless config.overwrite permits replacement. Required.
        backend (Literal['onnx', 'torch_tensorrt']): onnx selects the retained ONNX chain;
            torch_tensorrt remains an unavailable distinct backend, with no fallback. Default: 'onnx'.
        onnx_artifact (ExportResult | None): Existing verified ONNX ExportResult used as the conversion
            intermediate. Default: None.
        retain_intermediate (bool): Keep the explicitly generated ONNX intermediate when True; ownership
            and cleanup rules remain in force. Default: True.
        native (NativeConfig | None): Explicit NativeConfig for owned GPU execution. Omission grants no
            accelerator admission; experimental support limits remain in force. Default: None.
        config (ExportConfig): Configuration object or versioned mapping accepted by this API; loading
            settings alone does not execute a workload. Default: ExportConfig().
        precision (Literal['fp32', 'fp16', 'int8']): Requested fp32/fp16/int8 numerical policy;
            precision labels do not prove hardware instruction selection. Default: 'fp32'.
        profiles (dict[str, dict[str, list[int]]] | None): Named dynamic input min/opt/max shape bounds;
            calibration feeds must lie inside the declared range. Default: None.

    HYSo Guide: export.overview
    """
    if backend!='onnx':blocked('UNSUPPORTED_BACKEND','direct Torch-TensorRT compilation is not admitted in E01–E06','torch_tensorrt')
    if not retain_intermediate:raise ExportError('RETENTION','E04 retains its provenance-bound intermediate explicitly')
    if importlib.util.find_spec('tensorrt') is None:blocked('MISSING_DEPENDENCY','TensorRT absent; no intermediate export started','onnx_to_tensorrt')
    from .converters import onnx
    from ._model import isolated,fingerprint,tensors
    if onnx_artifact is None:
        middle=Path(output_path).with_name(Path(output_path).name+'.intermediate.onnx')
        onnx_artifact=onnx(model,example_input,middle,config=config)
    else:
        onnx_artifact.verify()
        with isolated(model,example_input,config) as (_,__,identity):
            row=onnx_artifact.to_dict()
            if row['source_model']!=identity or row['input_fingerprints']!=[fingerprint(t) for t in tensors(example_input)]:raise ExportError('PROVENANCE','reused ONNX does not match model/input')
    row=onnx_artifact.to_dict()
    feed={r['name']:t for r,t in zip(row['inputs'],__import__('hyso.export._model',fromlist=['tensors']).tensors(example_input))}
    result=tensorrt(onnx_artifact,output_path,precision=precision,profiles=profiles,example_input=feed,native=native,config=config)
    body=result.to_dict();body['conversion']='pytorch_to_onnx_to_tensorrt';body['intermediate_manifest']=str(Path(row['artifact_path']).with_name(Path(row['artifact_path']).name+'.export.json'))
    body['intermediate_sha256']=file_hash(row['artifact_path']);body['source_model']=row['source_model']
    final=seal(body);final.to_json(str(output_path)+'.export.json',overwrite=True);return final

def aotinductor(exported_program: ExportResult, output_path: str | Path, *,config: ExportConfig=ExportConfig(),native: NativeConfig | None=None) -> ExportResult:
    """Separate experimental AOTInductor compilation, never a LibTorch artifact.

    Windows/Torch 2.5 native AOT deployment is unsupported. Other platforms are
    capability-detected but not admitted by this Windows-owned release adapter.
    No compiler/SDK download or in-process compilation is attempted.

    Args:
        exported_program (ExportResult): Existing ExportedProgram artifact to compile explicitly;
            unsupported platforms fail closed. Required.
        output_path (str | Path): Destination for a separate inference artifact and sidecars; existing
            outputs are refused unless config.overwrite permits replacement. Required.
        config (ExportConfig): Configuration object or versioned mapping accepted by this API; loading
            settings alone does not execute a workload. Default: ExportConfig().
        native (NativeConfig | None): Explicit NativeConfig for owned GPU execution. Omission grants no
            accelerator admission; experimental support limits remain in force. Default: None.

    HYSo Guide: export.overview
    """
    exported_program.verify()
    if exported_program.to_dict()['artifact_format']!='exported_program':raise ExportError('ARTIFACT_FORMAT','ExportedProgram required')
    if platform.system()=='Windows':blocked('AOT_PLATFORM','Torch 2.5 Windows AOTInductor deployment is unsupported; ExportedProgram remains usable','aotinductor','admission')
    blocked('AOT_ADMISSION','Experimental Linux AOT API requires a Linux-owned compiler adapter; this release has Windows Job admission only','aotinductor','admission')
