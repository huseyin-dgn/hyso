"""Explicit CPU inference exporters; no optional backend import at module import."""
from __future__ import annotations
import importlib
import inspect
import uuid
import warnings
from contextlib import ExitStack,redirect_stdout,redirect_stderr
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
import torch
from .contracts import ExportConfig,NativeConfig,ExportError,ExportResult
from ._common import convert
from ._model import tensors,invoke,names

def dynamic_spec(dynamic_shapes, inputs, input_names):
    if dynamic_shapes is None:return {}
    if type(dynamic_shapes) is not dict or set(dynamic_shapes)-set(input_names):raise ExportError('DYNAMIC_SHAPE','unknown input names')
    result={}
    symbols={}
    for name,axes in dynamic_shapes.items():
        tensor=inputs[input_names.index(name)]
        if type(axes) is not dict:raise ExportError('DYNAMIC_SHAPE','axis mapping required')
        result[name]={}
        for axis,bound in axes.items():
            if type(axis) is not int or not 0<=axis<tensor.ndim or type(bound) is not dict or set(bound)!={'name','min','max'}:raise ExportError('DYNAMIC_SHAPE','axis/name/min/max required')
            low,high=bound['min'],bound['max'];symbol=bound['name']
            if not isinstance(symbol,str) or not symbol.isidentifier() or type(low) is not int or type(high) is not int or not 1<=low<high or not low<=tensor.shape[axis]<=high:raise ExportError('DYNAMIC_SHAPE','example must satisfy positive finite bounds')
            if symbol in symbols and symbols[symbol]!=(low,high,tensor.shape[axis]):raise ExportError('DYNAMIC_SHAPE','shared symbol bounds/example disagree')
            symbols[symbol]=(low,high,tensor.shape[axis]);result[name][axis]=dict(bound)
    return result

def _args(example):return (example,) if type(example) in (torch.Tensor,dict) else tuple(example)

class _Named(torch.nn.Module):
    def __init__(self,model,keys,output_keys=None):super().__init__();self.inner=model;self.keys=keys;self.output_keys=output_keys
    def forward(self,*values):
        result=self.inner(dict(zip(self.keys,values))) if self.keys is not None else self.inner(*values)
        return tuple(result[k] for k in self.output_keys) if self.output_keys is not None else result

def onnx(model: torch.nn.Module, example_input: Any, output_path: str | Path, *,
         config: ExportConfig=ExportConfig(), backend: Literal['legacy','dynamo']='legacy',opset_version: int=17,
         input_names: Sequence[str] | None=None,output_names: Sequence[str] | None=None,
         dynamic_shapes: dict[str,dict[int,dict[str,Any]]] | None=None,
         runtime_device: Literal['cpu','cuda']='cpu',native: NativeConfig | None=None) -> ExportResult:
    """Export an inference-only isolated copy to ONNX, with checker and optional ORT CPU execution.

    Tensor, flat positional tensors and named GraphBuilder input mappings are supported
    in the bounded dense CPU profile. legacy uses Torch 2.5 dynamo=False explicitly;
    dynamo is opt-in, fallback=False, and requires onnxscript. No backend fallback is
    hidden. dynamic_shapes bounds are metadata for ONNX symbolic axes (ONNX does not
    enforce these ranges); ExportedProgram enforces its native constraints separately.
    Existing output/sidecar paths require config.overwrite=True. Missing ORT can leave
    a checked artifact with deployment_verified=False. No packages are installed.

    Raises ExportError for unsupported input/operators, budgets, integrity or mismatch.

    Args:
        model (torch.nn.Module): Caller-owned torch.nn.Module; must meet the operation device, dtype and
            input contract. Required.
        example_input (Any): Representative input tensor or ordered tensor collection for this explicit
            operation; source model/device must match. Required.
        output_path (str | Path): Destination for a separate inference artifact and sidecars; existing
            outputs are refused unless config.overwrite permits replacement. Required.
        config (ExportConfig): Configuration object or versioned mapping accepted by this API; loading
            settings alone does not execute a workload. Default: ExportConfig().
        backend (Literal['legacy', 'dynamo']): Explicit exporter backend; ONNX accepts legacy or dynamo
            with no hidden fallback. Default: 'legacy'.
        opset_version (int): Integer ONNX operator-set version in 13..20; actual emitted opsets are
            recorded. Default: 17.
        input_names (Sequence[str] | None): Ordered names for exported graph inputs; count must match
            flattened example tensors. Default: None.
        output_names (Sequence[str] | None): Ordered graph output names; count must match flattened
            source outputs. Default: None.
        dynamic_shapes (dict[str, dict[int, dict[str, Any]]] | None): Explicit per-input symbolic
            dimensions with min/max bounds; symbolic ONNX axes alone do not enforce runtime bounds.
            Default: None.
        runtime_device (Literal['cpu', 'cuda']): cpu or cuda runtime validation target; CUDA requires a
            validated native configuration and owned worker. Default: 'cpu'.
        native (NativeConfig | None): Explicit NativeConfig for owned GPU execution. Omission grants no
            accelerator admission; experimental support limits remain in force. Default: None.

    Returns:
        ExportResult with separate conversion, artifact, runtime, equivalence and preservation statuses.

    HYSo Guide: export.overview

    Raises:
        ExportError: Unsupported input, dependency, integrity, resource or evidence requirements fail.
        OSError: Explicit artifact or report file access fails.

    Notes:
        Dynamo may fail strict provider-node coverage for graphs whose shape/constant nodes are folded
        by the runtime. A converted artifact is not deployment PASS when full coverage is unavailable.
    """
    if backend not in ('legacy','dynamo') or type(opset_version) is not int or not 13<=opset_version<=20:raise ExportError('BACKEND','explicit legacy/dynamo and opset 13..20 required')
    if runtime_device not in ('cpu','cuda'):raise ExportError('DEVICE','cpu/cuda runtime device required')
    ins=tensors(example_input);inn=names(input_names,len(ins),'input');dyn=dynamic_spec(dynamic_shapes,ins,inn)
    def produce(tx,owned,inputs,reference,body,mark):
        mark('dependencies');ox=importlib.import_module('onnx')
        if backend=='dynamo':importlib.import_module('onnxscript')
        body['backend_version']=str(torch.__version__);body['backend_api']='torch.onnx.export';body['onnx_checker_version']=ox.__version__
        body['runtime_scope']='ONNX Runtime '+runtime_device
        outn=names(output_names,len(tensors(reference)),'output')
        target=tx.root/tx.output.name
        inner=_Named(owned,list(inputs) if type(inputs) is dict else None,list(reference) if type(reference) is dict else None) if type(inputs) is dict or type(reference) is dict else owned
        args=tuple(tensors(inputs)) if type(inputs) is dict else _args(inputs)
        kwargs=dict(input_names=inn,output_names=outn,opset_version=opset_version,dynamo=backend=='dynamo',training=torch.onnx.TrainingMode.EVAL)
        if dyn:kwargs['dynamic_axes']={n:{axis:v['name'] for axis,v in axes.items()} for n,axes in dyn.items()}
        if backend=='dynamo':
            if 'fallback' not in inspect.signature(torch.onnx.export).parameters:raise ExportError('BACKEND_VERSION','this Torch exporter cannot disable fallback')
            kwargs['fallback']=False
            reports=tx.root/(tx.output.name+'.capture.'+tx.token);reports.mkdir()
            kwargs.update(report=True,artifacts_dir=str(reports))
            if dyn:
                symbols={};shapes=[]
                for name in inn:
                    axes={}
                    for axis,bound in dyn.get(name,{}).items():
                        if bound['name'] not in symbols:symbols[bound['name']]=torch.export.Dim(bound['name'],min=bound['min'],max=bound['max'])
                        axes[axis]=symbols[bound['name']]
                    shapes.append(axes)
                kwargs.pop('dynamic_axes',None)
                kwargs['dynamic_shapes']=(tuple(shapes),) if inner is not owned else tuple(shapes)
        mark('conversion',exporter=backend,inference_only=True,fallback=False)
        with warnings.catch_warnings(record=True) as captured, ExitStack() as logs:
            if backend=='dynamo':
                stream=logs.enter_context((reports/'export.log').open('w',encoding='utf-8'))
                logs.enter_context(redirect_stdout(stream));logs.enter_context(redirect_stderr(stream))
            warnings.simplefilter('always');torch.onnx.export(inner,args,str(target),**kwargs)
        body['warnings'].extend(type(w.message).__name__+': '+str(w.message) for w in captured)
        if backend=='dynamo':
            logs=list(reports.glob('*.md'))
            if len(logs)!=1:raise ExportError('BACKEND_EVIDENCE','one real capture strategy report required')
            text=logs[0].read_text(encoding='utf-8')
            successes=[line for line in text.splitlines() if line.startswith('\u2705 Obtain model graph')]
            if len(successes)!=1 or 'torch.jit.trace' in successes[0]:raise ExportError('BACKEND_FALLBACK','unapproved/unknown modern exporter capture fallback')
            body['options']['actual_capture_strategy']='torch.export.export(strict=False)' if 'strict=False' in successes[0] else 'torch.export.export(strict=True)'
            body['options']['legacy_fallback_allowed']=False
        mark('artifact');checked=ox.load(str(target),load_external_data=False)
        # All external references must remain within owned staging. Unique names
        # ensure replacing a generation cannot overwrite prior/user data files.
        moved={}
        from .artifacts import onnx_tensors
        for initializer in onnx_tensors(checked):
            for entry in initializer.external_data:
                if entry.key=='location':
                    from .artifacts import safe_file
                    old=safe_file(tx.root,entry.value);newname=moved.get(entry.value,tx.output.name+'.data.'+tx.token+'.'+str(len(moved)))
                    new=tx.root/newname
                    if old!=new and entry.value not in moved:
                        if new.exists():raise ExportError('ARTIFACT_PATH','external data naming collision')
                        old.rename(new);moved[entry.value]=newname
                    entry.value=newname
        for i,node in enumerate(checked.graph.node):
            if not node.name:node.name='hyso_node_'+str(i)
        ox.save_model(checked,str(target))
        ox.checker.check_model(str(target),full_check=True)
        inferred=ox.shape_inference.infer_shapes(ox.load(str(target)),check_type=True,strict_mode=True)
        body['options']['actual_opsets']={v.domain:v.version for v in checked.opset_import}
        if [v.name for v in inferred.graph.input]!=inn or [v.name for v in inferred.graph.output]!=outn:raise ExportError('IO_CONTRACT','ONNX input/output names differ')
        for source_values,declared in ((tensors(inputs),inferred.graph.input),(tensors(reference),inferred.graph.output)):
            for value,spec in zip(source_values,declared):
                typed=spec.type.tensor_type;expected=ox.helper.np_dtype_to_tensor_dtype(value.detach().numpy().dtype)
                if typed.elem_type!=expected or len(typed.shape.dim)!=value.ndim:raise ExportError('IO_CONTRACT','ONNX dtype/rank differs from source sample')
                for axis,dim in enumerate(typed.shape.dim):
                    if dim.HasField('dim_value') and dim.dim_value!=value.shape[axis]:raise ExportError('IO_CONTRACT','ONNX static shape differs from source sample')
                    if spec.name in inn and axis not in dyn.get(spec.name,{}) and not dim.HasField('dim_value'):raise ExportError('IO_CONTRACT','unexpected dynamic input axis')
        body['stages']['artifact'].update(onnx_checker=True,shape_inference=True,io_contract=True)
        body['options']['actual_input_names']=inn;body['options']['actual_output_names']=outn
        if not config.validate_runtime:return None
        if runtime_device=='cuda':
            mark('runtime_loading')
            from .native import validate_native
            runtime,actual=validate_native('ort',target,inputs,inn,tx,native)
            body['runtime']=runtime;mark('runtime_execution',provider='CUDAExecutionProvider',completed=True)
            return actual
        if importlib.util.find_spec('onnxruntime') is None:return None
        mark('runtime_loading')
        from .runtime import ort_cpu
        runtime,actual=ort_cpu(target,inputs,inn,tx.root,tx.token)
        body['runtime']=runtime;mark('runtime_execution',provider='CPUExecutionProvider',completed=True)
        return actual
    return convert(model,example_input,output_path,'onnx',backend,config,produce,dict(opset_version=opset_version,input_names=inn,output_names=list(output_names) if output_names else None,dynamic_shapes=dyn,runtime_device=runtime_device))

def libtorch(model: torch.nn.Module, example_input: Any, output_path: str | Path, *,
             method: Literal['trace','script']='trace',config: ExportConfig=ExportConfig(),
             runtime_device: Literal['cpu','cuda']='cpu',native: NativeConfig | None=None) -> ExportResult:
    """Save/load TorchScript and compare real Python runtime outputs on CPU.

    Trace observes example branches only; data-dependent control flow is not certified.
    Script compiles supported source syntax. TorchScript is the legacy LibTorch path.
    Python torch.jit.load success never establishes C++ runtime acceptance. The supplied
    CMake example must be compiled/executed separately in an admitted campaign.

    Args:
        model (torch.nn.Module): Caller-owned torch.nn.Module; must meet the operation device, dtype and
            input contract. Required.
        example_input (Any): Representative input tensor or ordered tensor collection for this explicit
            operation; source model/device must match. Required.
        output_path (str | Path): Destination for a separate inference artifact and sidecars; existing
            outputs are refused unless config.overwrite permits replacement. Required.
        method (Literal['trace', 'script']): TorchScript capture strategy: trace records the example
            path; script compiles supported source. Default: 'trace'.
        config (ExportConfig): Configuration object or versioned mapping accepted by this API; loading
            settings alone does not execute a workload. Default: ExportConfig().
        runtime_device (Literal['cpu', 'cuda']): cpu or cuda runtime validation target; CUDA requires a
            validated native configuration and owned worker. Default: 'cpu'.
        native (NativeConfig | None): Explicit NativeConfig for owned GPU execution. Omission grants no
            accelerator admission; experimental support limits remain in force. Default: None.

    Returns:
        ExportResult describing the TorchScript file and its independently verified runtime stages.

    HYSo Guide: export.overview

    Raises:
        ExportError: Unsupported input, dependency, integrity, resource or evidence requirements fail.
        OSError: Explicit artifact or report file access fails.
    """
    if method not in ('trace','script'):raise ExportError('METHOD','trace or script required')
    if runtime_device not in ('cpu','cuda'):raise ExportError('DEVICE','cpu/cuda runtime device required')
    def produce(tx,owned,inputs,reference,body,mark):
        mark('dependencies');body['backend_version']=str(torch.__version__)
        body['runtime_scope']='TorchScript Python '+runtime_device
        mark('conversion',method=method)
        inner=_Named(owned,list(inputs) if type(inputs) is dict else None,list(reference) if type(reference) is dict else None) if method=='trace' and (type(inputs) is dict or type(reference) is dict) else owned
        args=tuple(tensors(inputs)) if inner is not owned else _args(inputs)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            compiled=torch.jit.trace(inner,args,check_trace=True,strict=True) if method=='trace' else torch.jit.script(inner)
        body['warnings'].extend(type(w.message).__name__+': '+str(w.message) for w in captured)
        if method=='trace':body['warnings'].append('Trace validates example branches only; no general data-dependent control-flow guarantee')
        body['cpp_runtime']=dict(status='NOT_RUN',reason='Python TorchScript loading is separate from LibTorch C++ execution')
        target=tx.root/tx.output.name;torch.jit.save(compiled,str(target));mark('artifact')
        if not config.validate_runtime:return None
        if runtime_device=='cuda':
            from .native import validate_native
            mark('runtime_loading');body['runtime'],actual=validate_native('jit',target,tuple(tensors(inputs)),names(None,len(tensors(inputs)),'input'),tx,native)
            mark('runtime_execution',runtime='torch.jit.load',device='cuda:0');return actual
        mark('runtime_loading');loaded=torch.jit.load(str(target),map_location='cpu');loaded.eval()
        mark('runtime_execution',device='cpu',runtime='torch.jit.load')
        return loaded(*args)
    return convert(model,example_input,output_path,'torchscript','torch.jit.'+method,config,produce,dict(method=method))

def torch_export(model: torch.nn.Module, example_input: Any, output_path: str | Path, *,
                 config: ExportConfig=ExportConfig(),dynamic_shapes: dict[str,dict[int,dict[str,Any]]] | None=None,
                 input_names: Sequence[str] | None=None) -> ExportResult:
    """Serialize ExportedProgram, reload and compare its real Python execution.

    Uses installed torch.export.export/save/load, strict=True. This artifact is not
    directly loadable by LibTorch C++; AOTInductor is a separate compilation request.
    Dynamic shape names/min/max are translated to this version's native Dim API.

    Args:
        model (torch.nn.Module): Caller-owned torch.nn.Module; must meet the operation device, dtype and
            input contract. Required.
        example_input (Any): Representative input tensor or ordered tensor collection for this explicit
            operation; source model/device must match. Required.
        output_path (str | Path): Destination for a separate inference artifact and sidecars; existing
            outputs are refused unless config.overwrite permits replacement. Required.
        config (ExportConfig): Configuration object or versioned mapping accepted by this API; loading
            settings alone does not execute a workload. Default: ExportConfig().
        dynamic_shapes (dict[str, dict[int, dict[str, Any]]] | None): Explicit per-input symbolic
            dimensions with min/max bounds; symbolic ONNX axes alone do not enforce runtime bounds.
            Default: None.
        input_names (Sequence[str] | None): Ordered names for exported graph inputs; count must match
            flattened example tensors. Default: None.

    Returns:
        ExportResult for the serialized ExportedProgram and native shape constraints.

    HYSo Guide: export.overview

    Raises:
        ExportError: Unsupported input, dependency, integrity, resource or evidence requirements fail.
        OSError: Explicit artifact or report file access fails.
    """
    ins=tensors(example_input);inn=names(input_names,len(ins),'input');dyn=dynamic_spec(dynamic_shapes,ins,inn)
    def produce(tx,owned,inputs,reference,body,mark):
        mark('dependencies');body['backend_version']=str(torch.__version__)
        body['runtime_scope']='ExportedProgram Python CPU; not LibTorch C++'
        dims={}
        translated=[]
        for n in inn:
            axes={}
            for axis,v in dyn.get(n,{}).items():
                dims.setdefault(v['name'],torch.export.Dim(v['name'],min=v['min'],max=v['max']))
                axes[axis]=dims[v['name']]
            translated.append(axes)
        shapes=({k:v for k,v in zip(inputs,translated)},) if type(inputs) is dict else tuple(translated)
        mark('conversion',strict=True)
        ep=torch.export.export(owned,_args(inputs),dynamic_shapes=shapes if dyn else None,strict=True)
        body['export_constraints']={str(k):str(v) for k,v in ep.range_constraints.items()}
        target=tx.root/tx.output.name;torch.export.save(ep,str(target));mark('artifact')
        if not config.validate_runtime:return None
        mark('runtime_loading');loaded=torch.export.load(str(target))
        mark('runtime_execution',runtime='ExportedProgram.module',device='cpu')
        return invoke(loaded.module(),inputs)
    return convert(model,example_input,output_path,'exported_program','torch.export',config,produce,dict(input_names=inn,dynamic_shapes=dyn))
