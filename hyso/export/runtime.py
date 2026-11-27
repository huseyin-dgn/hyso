"""Output equivalence bound to the existing P35 completed provider capture."""
from __future__ import annotations

from typing import Any, Literal, ContextManager, Callable
import base64
import hashlib
import json
import os
import sys
from pathlib import Path
import torch
from .contracts import ExportError, NativeConfig, ExportResult

class Runtime:
    """Describe an eager model or verified export artifact for explicit runtime calls.

    ``source`` is an eager Module or verified ExportResult. CPU sessions are explicit. CUDA handles
    require NativeConfig and an evidence directory; run() delegates to the owned native worker. Resident
    GPU execution is forbidden.

    Args:
        source: Existing source evidence/artifact; verification is separate from execution. Required.
        precision: Requested fp32/fp16/int8 numerical policy; precision labels do not prove hardware
            instruction selection. Default: 'fp32'.
        device: Explicit execution device selector; passing a CUDA selector does not authorize or
            validate native execution. Default: 'cpu'.
        native: Explicit NativeConfig for owned GPU execution. Omission grants no accelerator admission;
            experimental support limits remain in force. Default: None.
        evidence_directory: Existing or explicitly owned directory for runtime evidence; output
            ownership checks still apply. Default: None.
    """
    def __init__(self, source: torch.nn.Module | ExportResult, *, precision: Literal['fp32','fp16','int8']='fp32', device: Literal['cpu','cuda']='cpu',native: NativeConfig | None=None,evidence_directory: str | Path | None=None) -> None:
        """Initialize Runtime from the supplied settings or evidence.

        Args:
            source: Existing source evidence/artifact; verification is separate from execution. Required.
            precision: Requested fp32/fp16/int8 numerical policy; precision labels do not prove hardware
                instruction selection. Default: 'fp32'.
            device: Explicit execution device selector; passing a CUDA selector does not authorize or
                validate native execution. Default: 'cpu'.
            native: Explicit NativeConfig for owned GPU execution. Omission grants no accelerator admission;
                experimental support limits remain in force. Default: None.
            evidence_directory: Existing or explicitly owned directory for runtime evidence; output
                ownership checks still apply. Default: None.
        """
        from .contracts import ExportResult
        if precision not in ('fp32','fp16','int8') or device not in ('cpu','cuda'):
            raise ExportError('RUNTIME_SPEC','explicit precision and cpu/cuda required')
        self.source=source;self.device=device;self.precision=precision;self.native=native;self.evidence_directory=evidence_directory;self.receipts=[]
        if isinstance(source,ExportResult):
            source.verify();row=source.to_dict()
            precision=row.get('precision',{}).get('requested',precision)
            self.precision=precision
            self.info=dict(backend=row['artifact_format'],precision=precision,device=device,
                artifact_sha256=row['artifact_sha256'],source_model=row.get('source_model'),
                artifact_bytes=sum(r['bytes'] for r in row['artifact_files']),
                portability=row.get('limitations',[]))
        else:
            import torch
            if not isinstance(source,torch.nn.Module):raise ExportError('RUNTIME_SPEC','Module or ExportResult required')
            self.info=dict(backend='pytorch',precision=precision,device=device,artifact_sha256=None,artifact_bytes=None)

    @staticmethod
    def input_identity(inputs: Any) -> list[dict[str, Any]]:
        """Input identity using the explicitly supplied evidence.

        Args:
            inputs: Tensor inputs matching the selected runtime input names, shapes and dtypes. Required.
        """
        from ._model import tensors,fingerprint
        return [fingerprint(t) for t in tensors(inputs)]

    def session(self, inputs: Any) -> ContextManager[tuple[Callable[[], Any], float]]:
        """Context-managed load/isolated model and independent working inputs.

        Args:
            inputs: Tensor inputs matching the selected runtime input names, shapes and dtypes. Required.
        """
        from contextlib import contextmanager
        @contextmanager
        def opened():
            import copy,time,torch
            from contextlib import ExitStack
            from .contracts import ExportResult,ExportConfig
            from ._model import isolated,tensors,invoke
            if self.device!='cpu' or self.info['backend']=='tensorrt_engine':
                raise ExportError('CUDA_ADMISSION_REQUIRED','use the owned native execution adapter; resident GPU runtime is forbidden')
            values=tensors(inputs)
            if any(t.device.type!='cpu' for t in values):raise ExportError('CUDA_ADMISSION_REQUIRED','CPU inputs required')
            if sum(t.numel()*t.element_size() for t in values)>16*1024**2:raise ExportError('RESOURCE_BUDGET','runtime input exceeds 16 MiB')
            before=self.input_identity(inputs)
            started=time.perf_counter_ns()
            with ExitStack() as stack:
                from ._model import _rng_state,_restore_rng
                rng=_rng_state();stack.callback(_restore_rng,rng)
                if not isinstance(self.source,ExportResult):
                    model,working,identity=stack.enter_context(isolated(self.source,inputs,ExportConfig()))
                    self.info['source_model']=identity
                    forward=lambda:invoke(model,working)
                    version=str(torch.__version__)
                else:
                    self.source.verify();row=self.source.to_dict();path=row['artifact_path'];working=copy.deepcopy(inputs)
                    backend=self.info['backend']
                    if backend=='onnx':
                        import onnxruntime as ort
                        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
                        session=ort.InferenceSession(path,sess_options=options,providers=['CPUExecutionProvider'])
                        names=[v.name for v in session.get_inputs()]
                        if isinstance(working,dict):
                            if set(working)!=set(names):raise ExportError('INPUT_SPEC','named runtime inputs must match ONNX names')
                            feed={n:working[n].detach().numpy() for n in names}
                        else:
                            if len(values)!=len(names):raise ExportError('INPUT_SPEC','ONNX input arity differs')
                            feed={n:t.detach().numpy() for n,t in zip(names,tensors(working))}
                        forward=lambda:tuple(torch.from_numpy(v) for v in session.run(None,feed))
                        version=ort.__version__
                    elif backend=='torchscript':
                        if isinstance(working,dict):working=tuple(working[r['name']] for r in row['inputs'])
                        model=torch.jit.load(path,map_location='cpu').eval();forward=lambda:invoke(model,working);version=str(torch.__version__)
                    elif backend=='exported_program':
                        if isinstance(working,dict):working=tuple(working[r['name']] for r in row['inputs'])
                        model=torch.export.load(path).module();forward=lambda:invoke(model,working);version=str(torch.__version__)
                    else:raise ExportError('UNSUPPORTED_BACKEND','unsupported saved runtime artifact')
                working_before=self.input_identity(working)
                self.info.update(runtime_version=version,provider='CPUExecutionProvider' if self.info['backend']=='onnx' else 'pytorch_cpu')
                load_ms=(time.perf_counter_ns()-started)/1e6
                def call():
                    with torch.no_grad():return forward()
                yield call,load_ms
                if self.input_identity(inputs)!=before or self.input_identity(working)!=working_before:
                    raise ExportError('MODEL_PRESERVATION','runtime modified source or working inputs')
                if isinstance(self.source,ExportResult):self.source.verify()
        return opened()

    def run(self, inputs: Any) -> list[torch.Tensor]:
        """Run a small explicit CPU request; returns validated flat tensors.

        Args:
            inputs: Tensor inputs matching the selected runtime input names, shapes and dtypes. Required.
        """
        from ._model import outputs
        if self.device=='cuda' and self.info['backend']=='tensorrt_engine':
            report,_=self.measure_native(inputs,warmup=0,iterations=1)
            return outputs(decode_arrays(report['outputs']))
        with self.session(inputs) as (call,_):return outputs(call())

    def measure_native(self,inputs: Any,*,warmup: int=2,iterations: int=10) -> tuple[dict[str, Any], dict[str, Any]]:
        """Explicit native request. Evidence is retained; no automatic retry.

        Args:
            inputs: Tensor inputs matching the selected runtime input names, shapes and dtypes. Required.
            warmup: Untimed inference calls before sampling; integer in 0..100. Default: 2.
            iterations: Timed inference calls; integer in 1..1000 for export.benchmark. Default: 10.
        """
        from .contracts import NativeConfig,ExportResult
        from ._model import tensors
        from .native import run_owned
        import uuid
        if self.device!='cuda' or self.info['backend']!='tensorrt_engine' or type(self.native) is not NativeConfig:
            raise ExportError('CUDA_ADMISSION_REQUIRED','TensorRT engine and explicit NativeConfig required')
        if self.evidence_directory is None or not Path(self.evidence_directory).is_dir():
            raise ExportError('OWNERSHIP','existing evidence directory required for retained native receipts')
        values=tensors(inputs)
        if any(t.device.type!='cpu' for t in values) or sum(t.numel()*t.element_size() for t in values)>16*1024**2:
            raise ExportError('INPUT_SPEC','bounded CPU feeds required before admission')
        self.source.verify();row=self.source.to_dict()
        if row['environment']['gpu_uuid']!=self.native.gpu_uuid:raise ExportError('GPU_IDENTITY','engine GPU differs from selected admission')
        names=[r['name'] for r in row['inputs']]
        if isinstance(inputs,dict):
            if set(inputs)!=set(names):raise ExportError('INPUT_SPEC','named engine feeds differ')
            values=[inputs[n] for n in names]
        if len(values)!=len(names):raise ExportError('INPUT_SPEC','engine input arity differs')
        payload=dict(source_path=row['artifact_path'],source_sha256=row['artifact_sha256'],runtime_version=row['backend_version'],
            inputs=dict(names=names,arrays=encode_arrays([t.detach().numpy() for t in values])),
            output_names=[r['name'] for r in row['outputs']],warmup=warmup,iterations=iterations)
        root=Path(self.evidence_directory)/('deployment-'+uuid.uuid4().hex)
        report,controller=run_owned('deployment',root,payload,self.native,
                                    failure_directory=self.evidence_directory)
        self.receipts.append(dict(evidence_root=str(root),report=report,controller=controller))
        self.source.verify()
        self.info.update(provider='TensorRT',runtime_version=report['runtime_version'],environment=report['environment'])
        return report,controller

def encode_arrays(values):
    if sum(v.nbytes for v in values)>32*1024**2:raise ExportError('RESOURCE_BUDGET','runtime outputs exceed retained evidence budget')
    rows=[]
    for v in values:
        raw=v.tobytes(order='C')
        rows.append(dict(shape=list(v.shape),dtype=v.dtype.str,data=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest()))
    return rows

def decode_arrays(rows):
    import numpy as np
    import torch
    result=[];total=0
    for r in rows:
        raw=base64.b64decode(r['data'],validate=True);total+=len(raw)
        if total>32*1024**2 or hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ExportError('OUTPUT_INTEGRITY','runtime output bytes modified/oversized')
        dtype=np.dtype(r['dtype'])
        if dtype.hasobject:raise ExportError('OUTPUT_INTEGRITY','object array forbidden')
        result.append(torch.from_numpy(np.frombuffer(raw,dtype=dtype).reshape(r['shape']).copy()))
    return tuple(result)

def onnx_factory(model_path,feed_path):
    """Trusted installed factory for the existing P35 provider worker."""
    import numpy as np
    import onnx
    # External data is validated by conversion; produce self-contained P35 bytes.
    model=onnx.load(model_path);onnx.external_data_helper.convert_model_from_external_data(model)
    with np.load(feed_path,allow_pickle=False) as archive:feed={k:archive[k].copy() for k in archive.files}
    return model.SerializeToString(),feed

def ort_cpu(model_path,example,input_names,staging,token):
    return _ort_capture(model_path,example,input_names,staging,token)

def _ort_capture(model_path,example,input_names,staging,token,*,provider='CPUExecutionProvider',gpu_uuid=None,gpu_bytes=None):
    from dataclasses import asdict
    import numpy as np
    import importlib.util
    spec=importlib.util.find_spec('onnxruntime')
    if spec is None:raise ImportError('onnxruntime is unavailable')
    from .artifacts import file_hash
    from .artifacts import onnx_tensors,safe_file
    import onnx
    source_path=Path(model_path)
    source_sha=file_hash(source_path);external={}
    for tensor in onnx_tensors(onnx.load(str(source_path),load_external_data=False)):
        for item in tensor.external_data:
            if item.key=='location':external[item.value]=file_hash(safe_file(source_path.parent,item.value))
    expected_root=Path(spec.origin).parent.resolve()
    expected={os.path.normcase(str(p.resolve())):file_hash(p) for p in (expected_root/'capi').iterdir() if p.is_file() and p.suffix.lower() in ('.pyd','.dll','.so')}
    import onnxruntime as ort
    if Path(ort.__file__).resolve().parent!=expected_root:raise ExportError('RUNTIME_IDENTITY','loaded ORT source differs from discovered runtime')
    from ..performance.providers import ProviderProtocol,_collect_profile,_analyze_provider_report,_options
    from ..performance.stability import WorkloadRecipe
    from ..analysis._codec import digest
    from ._model import tensors
    root=staging/(Path(model_path).name+'.evidence.'+token);root.mkdir()
    feeds=root/'inputs.npz';np.savez(feeds,**{n:t.detach().numpy() for n,t in zip(input_names,tensors(example))})
    recipe=WorkloadRecipe('hyso.export.runtime:onnx_factory',dict(model_path=str(model_path),feed_path=str(feeds)))
    providers=[provider]+(['CPUExecutionProvider'] if provider!='CPUExecutionProvider' else [])
    options={provider:{'device_id':0,'gpu_mem_limit':gpu_bytes}} if gpu_uuid else None
    recipe=WorkloadRecipe('hyso.export.runtime:onnx_factory',dict(model_path=str(model_path),feed_path=str(feeds)),gpu_uuid=gpu_uuid)
    request=dict(token=token,recipe=asdict(recipe),protocol=asdict(ProviderProtocol(runs=1)),
        providers=providers,expected={'*':provider},optimization='disabled',
        runtime_search_path=None,provider_options=options,_export_capture_outputs=True)
    report=_collect_profile(root,request,{})
    if file_hash(source_path)!=source_sha or any(file_hash(safe_file(source_path.parent,p))!=sha for p,sha in external.items()):raise ExportError('ARTIFACT_INTEGRITY','ONNX source/external data changed during actual runtime capture')
    evidence=dict(report=report,expected=request['expected'],optimization='disabled',providers=request['providers'],
        recipe=asdict(recipe),protocol=request['protocol'],token=token,
        provider_options=_options(options,request['providers'],gpu_uuid),graph_analysis_version=2)
    # This is a synchronous small CPU call, not an owned subprocess. Do not
    # fabricate a Job/controller receipt to reuse graph/trace validation.
    analysis=_analyze_provider_report(evidence,report,os.getpid())
    if analysis['status']!='expected_execution_observed' or not analysis['events'] or analysis['fallback_detected'] is not False or analysis['uncovered_source_nodes']:
        raise ExportError('PROVIDER_EXECUTION','actual complete CPU provider node execution required',result={'report':report,'analysis':analysis})
    if any(e['provider']!=provider for e in analysis['events']):raise ExportError('CPU_FALLBACK','unexpected provider event')
    # Retain real runtime source/extension identities, not package-list heuristics.
    native=Path(ort.__file__).parent/'capi'
    identities={str(p.resolve()):file_hash(p) for p in native.iterdir() if p.is_file() and p.suffix.lower() in ('.pyd','.dll','.so')}
    actual={}
    if os.name=='nt':
        from ._worker import loaded_binaries
        actual={p:r for p,r in loaded_binaries().items() if Path(p).is_relative_to(expected_root)}
        if not actual or not any('pybind11' in Path(p).name for p in actual) or any(expected.get(p)!=r['sha256'] for p,r in actual.items()):raise ExportError('RUNTIME_IDENTITY','actual loaded ORT extension hash/path mismatch')
    result=dict(report=report,analysis=analysis,source=str(Path(ort.__file__).resolve()),version=ort.__version__,
        python=sys.executable,binary_files=identities,loaded_binaries=actual,identity_scope='discovered ORT source/extension hashes bound to actual loaded CPU runtime; GPU DLL admission is separate')
    result['artifact_binding']=dict(main_sha256=source_sha,external_files=external,self_contained_runtime_model_sha256=report['model_sha256'])
    result['output_trace_binding']=digest(dict(trace=report['trace_sha256'],inputs=report['input_sha256'],outputs=report['export_outputs'],session_id=report['session_id']))
    (root/'provider-evidence.json').write_text(json.dumps(result,allow_nan=False),encoding='utf-8')
    return result,decode_arrays(report['export_outputs'])
