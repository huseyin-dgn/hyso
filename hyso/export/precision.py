"""Transactional ONNX precision transformations with bounded calibration evidence."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Literal
import hashlib
import importlib.metadata
import uuid
from .contracts import ExportConfig,ExportError,ExportResult,STAGES,seal
from .artifacts import Transaction,file_hash,onnx_tensors,safe_file


def _source(source: ExportResult, config: ExportConfig):
    import onnx
    if type(source) is not ExportResult:raise TypeError('verified ONNX ExportResult required')
    source.verify();row=source.to_dict()
    if row['artifact_format']!='onnx':raise ExportError('ARTIFACT_FORMAT','ONNX artifact required')
    if sum(r['bytes'] for r in row['artifact_files'])>config.max_artifact_bytes:raise ExportError('RESOURCE_BUDGET','source manifest exceeds transform budget')
    path=Path(row['artifact_path'])
    proto=onnx.load(str(path),load_external_data=False)
    total=path.stat().st_size
    for tensor in onnx_tensors(proto):
        for entry in tensor.external_data:
            if entry.key=='location':total+=safe_file(path.parent,entry.value).stat().st_size
    if total>config.max_model_bytes:raise ExportError('RESOURCE_BUDGET','ONNX source exceeds transform model budget')
    onnx.checker.check_model(str(path),full_check=True)
    return row,path,onnx.load(str(path))


def _feeds(proto, feed, profiles=None):
    import numpy as np
    import onnx
    if not isinstance(feed,Mapping) or set(feed)!={v.name for v in proto.graph.input}:
        raise ExportError('CALIBRATION_INPUT','feed names must cover all graph inputs exactly')
    result={}
    for spec in proto.graph.input:
        value=feed[spec.name]
        if hasattr(value,'detach'):
            if value.device.type!='cpu':raise ExportError('CUDA_ADMISSION_REQUIRED','CPU calibration inputs required')
            value=value.detach().numpy()
        a=np.asarray(value);tt=spec.type.tensor_type
        shape=[d.dim_value if d.dim_value>0 else -1 for d in tt.shape.dim]
        if (a.dtype!=np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)) or a.ndim!=len(shape)
                or not a.size or a.dtype.kind not in 'fiu' or not np.isfinite(a).all()
                or any(n<1 or expected>0 and n!=expected for n,expected in zip(a.shape,shape))):
            raise ExportError('CALIBRATION_INPUT','finite graph-matching shape and dtype required')
        if profiles and spec.name in profiles:
            bounds=profiles[spec.name]
            if any(not low<=n<=high for n,low,high in zip(a.shape,bounds['min'],bounds['max'])):
                raise ExportError('CALIBRATION_INPUT','input outside declared dynamic profile')
        result[spec.name]=a.copy()
    if sum(a.nbytes for a in result.values())>16*1024**2:raise ExportError('RESOURCE_BUDGET','feed exceeds input budget')
    return result


def _transform(source,output_path,config,requested,options,convert,example_input):
    if type(config) is not ExportConfig:raise TypeError('ExportConfig required')
    import onnx
    from .runtime import ort_cpu,encode_arrays
    from ._model import equivalence
    row,path,proto=_source(source,config)
    if example_input is not None:feed=_feeds(proto,example_input)
    else:feed=None
    try:version=importlib.metadata.version('onnxruntime')
    except importlib.metadata.PackageNotFoundError:
        try:version=importlib.metadata.version('onnxruntime-gpu')
        except importlib.metadata.PackageNotFoundError:version=None
    body=dict(schema_version=1,format='hyso.export',conversion='onnx',backend='onnxruntime_precision',
        backend_version=version,
        config=asdict(config),options=options,precision=dict(requested=requested,scope='mixed_operator_precision',
            all_operations_in_requested_dtype=False),source_artifact_sha256=row['artifact_sha256'],
        source_model=row.get('source_model'),input_fingerprints=row.get('input_fingerprints'),
        stages={s:dict(status='NOT_RUN') for s in STAGES},warnings=[],
        limitations=['No performance or whole-cohort quality claim','Precision describes the graph; it does not prove hardware instruction selection'])
    try:
        with Transaction(output_path,config.overwrite,config.max_artifact_bytes) as tx:
            target=tx.root/tx.output.name
            evidence=convert(proto,path,target)
            body['precision'].update(evidence)
            onnx.checker.check_model(str(target),full_check=True)
            out=onnx.load(str(target),load_external_data=False)
            body['inputs']=[dict(name=v.name,shape=[d.dim_value if d.dim_value>0 else -1 for d in v.type.tensor_type.shape.dim],
                                dtype=str(onnx.helper.tensor_dtype_to_np_dtype(v.type.tensor_type.elem_type))) for v in out.graph.input]
            body['outputs']=[dict(name=v.name,shape=[d.dim_value if d.dim_value>0 else -1 for d in v.type.tensor_type.shape.dim],
                                 dtype=str(onnx.helper.tensor_dtype_to_np_dtype(v.type.tensor_type.elem_type))) for v in out.graph.output]
            if config.validate_runtime and feed is not None:
                import torch,numpy as np
                names=list(feed);inputs=tuple(torch.from_numpy(feed[n]) for n in names)
                _,reference=ort_cpu(path,inputs,names,tx.root,uuid.uuid4().hex)
                cast={s.name:feed[s.name].astype(onnx.helper.tensor_dtype_to_np_dtype(s.type.tensor_type.elem_type)) for s in out.graph.input}
                coverage=True
                try:runtime,actual=ort_cpu(target,tuple(torch.from_numpy(cast[n]) for n in names),names,tx.root,uuid.uuid4().hex)
                except ExportError as exc:
                    if exc.code!='PROVIDER_EXECUTION' or 'export_outputs' not in exc.result.get('report',{}):raise
                    # CPU ORT can rewrite half operations into internal casts that
                    # P35 cannot map completely. Preserve the failure and actual
                    # outputs without converting it into deployment PASS.
                    from .runtime import decode_arrays
                    coverage=False;actual=decode_arrays(exc.result['report']['export_outputs'])
                    runtime=dict(artifact_binding=dict(main_sha256=file_hash(target),external_files={}),
                                 report=exc.result['report'],analysis=exc.result['analysis'],coverage_status='BLOCKED')
                from .validation import output_metrics,Tolerance
                checks=output_metrics(reference,actual,precision=requested,tolerance=Tolerance(config.atol,config.rtol,True))
                body['precision_validation']=checks
                if checks['numerical']['status']!='PASS':raise ExportError('OUTPUT_MISMATCH','precision drift exceeds configured tolerances',result=checks)
                # Existing deployment seal requires exact output dtype. Explicit
                # cross-dtype metrics are retained separately instead of coercing.
                if all(a.dtype==b.dtype for a,b in zip(reference,actual)):
                    body['stages']['equivalence']=dict(status='PASS',comparisons=equivalence(reference,actual,config))
                    body['reference_outputs']=encode_arrays([v.numpy() for v in reference]);body['actual_outputs']=encode_arrays([v.numpy() for v in actual])
                else:body['stages']['equivalence']=dict(status='BLOCKED',reason='Explicit output dtype change; see precision_validation and cohort validation')
                body['runtime']=runtime
                body['stages']['runtime_loading']=dict(status='PASS')
                body['stages']['runtime_execution']=dict(status='PASS' if coverage else 'BLOCKED',
                    reason=None if coverage else 'P35 cannot completely map ORT FP16 CPU cast rewrites; actual output comparison is separate')
            else:body['stages']['runtime_execution']=dict(status='BLOCKED',reason='No explicit input/runtime validation request')
            source.verify()
            body.update(artifact_format='onnx',artifact_path=str(tx.output),artifact_root=str(tx.parent),artifact_files=tx.inventory())
            main=next(v for v in body['artifact_files'] if v['path']==tx.output.name)
            body.update(artifact_sha256=main['sha256'],artifact_bytes=main['bytes'])
            for s in ('dependencies','admission','conversion','artifact','preservation'):body['stages'][s]=dict(status='PASS')
            tx.publish(seal(body))
        body['stages']['cleanup']=dict(status='PASS',owned_staging_removed=True)
        result=seal(body);result.to_json(tx.manifest,overwrite=True);return result
    except ImportError as exc:
        body['stages']['dependencies']=dict(status='BLOCKED',reason=str(exc));body['deployment_verified']=False
        raise ExportError('MISSING_DEPENDENCY',str(exc),result=body) from exc


def fp16(source: ExportResult, output_path: str | Path, *, keep_io_types: bool=True,
         op_block_list: list[str] | None=None, example_input: Mapping[str,Any] | None=None,
         config: ExportConfig=ExportConfig(atol=1e-3,rtol=1e-2)) -> ExportResult:
    """Convert ONNX floats using ORT's converter, retaining its safe FP32 op list.

    keep_io_types preserves caller FP32 feeds/outputs. Blocked operations remain
    FP32 with casts; this is mixed precision, not a claim of universal FP16.

    Args:
        source (ExportResult): Verified ONNX ExportResult; its manifest and referenced artifact files
            are checked before transformation. Required.
        output_path (str | Path): Destination for a separate inference artifact and sidecars; existing
            outputs are refused unless config.overwrite permits replacement. Required.
        keep_io_types (bool): Keep exported graph input/output dtypes unchanged when True; internal
            conversion is still mixed precision. Default: True.
        op_block_list (list[str] | None): ONNX operator types retained in FP32 during FP16 conversion;
            None uses converter defaults. Default: None.
        example_input (Mapping[str, Any] | None): Optional complete graph-name to CPU tensor/array feed
            for runtime comparison. Without it, transformation alone cannot yield deployment
            verification. Default: None.
        config (ExportConfig): Configuration object or versioned mapping accepted by this API; loading
            settings alone does not execute a workload. Default: ExportConfig(atol=0.001, rtol=0.01).

    Returns:
        ExportResult describing mixed operator precision. Read deployment_verified separately from
        artifact/checker success.

    Raises:
        ExportError: Unsupported input, dependency, integrity, resource or evidence requirements fail.
        OSError: Explicit artifact or report file access fails.
    """
    if type(keep_io_types) is not bool or op_block_list is not None and (type(op_block_list) is not list or any(type(n) is not str for n in op_block_list)):
        raise ExportError('PRECISION','boolean I/O policy and optional operation names required')
    def convert(proto,path,target):
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16,DEFAULT_OP_BLOCK_LIST
        blocked=sorted(set(DEFAULT_OP_BLOCK_LIST)|set(op_block_list or []))
        changed=convert_float_to_float16(proto,keep_io_types=keep_io_types,op_block_list=blocked)
        # ORT's converter appends I/O casts. Sort dependencies before the strict
        # ONNX checker; do not weaken checker validation for generated graphs.
        from onnxruntime.transformers.onnx_model import OnnxModel
        OnnxModel(changed).topological_sort()
        has_half=any(t.data_type==onnx.TensorProto.FLOAT16 for t in onnx_tensors(changed)) or any(
            v.type.tensor_type.elem_type==onnx.TensorProto.FLOAT16 for v in list(changed.graph.value_info)+list(changed.graph.input)+list(changed.graph.output))
        if not has_half:raise ExportError('UNSUPPORTED_OPERATORS','converter produced no FP16 tensors; operation profile remains FP32')
        onnx.save(changed,str(target))
        return dict(converter='onnxruntime.transformers.float16',keep_io_types=keep_io_types,fp32_op_block_list=blocked,
                    fp16_initializers=sum(t.data_type==onnx.TensorProto.FLOAT16 for t in changed.graph.initializer),
                    fp32_initializers=sum(t.data_type==onnx.TensorProto.FLOAT for t in changed.graph.initializer))
    return _transform(source,output_path,config,'fp16',dict(keep_io_types=keep_io_types),convert,example_input)


def quantize(source: ExportResult, output_path: str | Path, *, mode: Literal['dynamic','static']='dynamic',
             calibration: Iterable[Mapping[str,Any]] | None=None, dataset_id: str | None=None,
             representative: bool=False, min_calibration_batches: int=2, max_calibration_batches: int=64,
             op_types: list[str] | None=None, profiles: dict | None=None,
             example_input: Mapping[str,Any] | None=None,
             config: ExportConfig=ExportConfig(atol=.1,rtol=.1)) -> ExportResult:
    """Dynamic MatMul/Gemm or static symmetric INT8 QDQ Conv/MatMul/Gemm.

    Calibration is user supplied, finite, copied and content-hashed. The caller's
    representativeness declaration is recorded, never inferred from batch count.
    Dynamic quantization does not promise INT8 Conv or TensorRT compatibility.

    Args:
        source (ExportResult): Verified ONNX ExportResult; its manifest and referenced artifact files
            are checked before transformation. Required.
        output_path (str | Path): Destination for a separate inference artifact and sidecars; existing
            outputs are refused unless config.overwrite permits replacement. Required.
        mode (Literal['dynamic', 'static']): dynamic uses integer MatMul/Gemm without calibration;
            static uses symmetric QDQ and requires calibration plus dataset_id. Default: 'dynamic'.
        calibration (Iterable[Mapping[str, Any]] | None): Caller-supplied iterable of complete named CPU
            input feeds. Static INT8 consumes copied, finite, shape/dtype-matching batches; dynamic mode
            rejects calibration. Default: None.
        dataset_id (str | None): Explicit bounded cohort identifier; static calibration requires a
            nonempty string of at most 256 characters. Default: None.
        representative (bool): Caller declaration that calibration data represents deployment inputs;
            batch count never proves representativeness. Default: False.
        min_calibration_batches (int): Minimum static calibration batches; must satisfy
            2<=min_calibration_batches<=max_calibration_batches<=256. Default: 2.
        max_calibration_batches (int): Maximum consumed calibration batches; exceeding the declared
            bound raises rather than returning partial PASS. Default: 64.
        op_types (list[str] | None): Quantized operator selection: dynamic permits MatMul/Gemm; static
            permits Conv/MatMul/Gemm. None selects all permitted types. Default: None.
        profiles (dict | None): Named dynamic input min/opt/max shape bounds; calibration feeds must lie
            inside the declared range. Default: None.
        example_input (Mapping[str, Any] | None): Optional complete graph-name to CPU tensor/array feed
            for runtime comparison. Without it, transformation alone cannot yield deployment
            verification. Default: None.
        config (ExportConfig): Configuration object or versioned mapping accepted by this API; loading
            settings alone does not execute a workload. Default: ExportConfig(atol=0.1, rtol=0.1).

    Returns:
        ExportResult with the actual quantized operators, calibration provenance and independent
        validation statuses.

    Raises:
        ExportError: Unsupported input, dependency, integrity, resource or evidence requirements fail.
        OSError: Explicit artifact or report file access fails.
    """
    if mode not in ('dynamic','static'):raise ExportError('QUANTIZATION','dynamic/static required')
    allowed={'MatMul','Gemm'} if mode=='dynamic' else {'Conv','MatMul','Gemm'}
    selected=sorted(allowed) if op_types is None else op_types
    if type(selected) is not list or not selected or any(type(n) is not str or n not in allowed for n in selected):
        raise ExportError('QUANTIZATION','unsupported operation selection')
    if type(representative) is not bool:raise ExportError('CALIBRATION_INPUT','representative declaration must be boolean')
    if (type(min_calibration_batches) is not int or type(max_calibration_batches) is not int
            or not 2<=min_calibration_batches<=max_calibration_batches<=256):raise ExportError('CALIBRATION_INPUT','calibration batch bounds must be 2..256')
    def convert(proto,path,target):
        import onnx
        from onnxruntime.quantization import quantize_dynamic,quantize_static,QuantType,QuantFormat,CalibrationDataReader,CalibrationMethod
        entries=[];hashes=[];total=0
        if mode=='static':
            if calibration is None or not isinstance(dataset_id,str) or not dataset_id.strip() or len(dataset_id)>256:
                raise ExportError('CALIBRATION_INPUT','nonempty calibration and explicit dataset_id required')
            if profiles:
                from .native import validate_profiles
                validate_profiles(profiles,{v.name:[d.dim_value if d.dim_value>0 else -1 for d in v.type.tensor_type.shape.dim] for v in proto.graph.input})
            for index,entry in enumerate(calibration):
                if index>=max_calibration_batches:raise ExportError('RESOURCE_BUDGET','calibration exceeds declared batch limit')
                feed=_feeds(proto,entry,profiles);total+=sum(a.nbytes for a in feed.values())
                if total>32*1024**2:raise ExportError('RESOURCE_BUDGET','calibration exceeds 32 MiB')
                entries.append(feed);hashes.append({n:dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest()) for n,a in feed.items()})
            if len(entries)<min_calibration_batches:raise ExportError('CALIBRATION_INPUT','insufficient calibration batches')
            class Reader(CalibrationDataReader):
                def __init__(self):self.rewind()
                def get_next(self):return next(self.iterator,None)
                def rewind(self):self.iterator=iter(entries)
            quantize_static(str(path),str(target),Reader(),quant_format=QuantFormat.QDQ,activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,op_types_to_quantize=selected,calibrate_method=CalibrationMethod.MinMax,
                extra_options={'ActivationSymmetric':True,'WeightSymmetric':True})
        else:
            if calibration is not None:raise ExportError('CALIBRATION_INPUT','dynamic quantization does not consume calibration')
            quantize_dynamic(str(path),str(target),weight_type=QuantType.QInt8,op_types_to_quantize=selected)
        result=onnx.load(str(target));nodes=[n.op_type for n in result.graph.node]
        if not any(n in ('QuantizeLinear','DequantizeLinear','MatMulInteger','DynamicQuantizeLinear','QLinearConv','QLinearMatMul') for n in nodes):
            raise ExportError('UNSUPPORTED_OPERATORS','no selected operation was quantized')
        from ..analysis._codec import digest
        return dict(mode=mode,format='QDQ' if mode=='static' else 'dynamic_integer_operators',
            selected_op_types=selected,observed_op_types=sorted(set(nodes)),
            calibration=dict(status='provided' if entries else 'not_required',dataset_id=dataset_id,
                representative_declared=representative,batches=len(entries),input_bytes=total,
                content_sha256=digest(hashes) if entries else None,method='MinMax' if entries else None),
            native_validation='NOT_TESTED',symmetric=mode=='static')
    return _transform(source,output_path,config,'int8',dict(mode=mode,op_types=selected),convert,example_input)
