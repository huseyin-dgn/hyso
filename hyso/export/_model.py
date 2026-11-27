"""Bounded existing benchmark tensor guards plus evaluation's isolated deepcopy."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
from contextlib import contextmanager
import torch
from ..benchmark.runner import _profile, _signature, _host_copy, _same_bits, _all_finite, _descriptor
from ..checkpoint import _rng_state, _restore_rng
from .contracts import ExportError

def tensors(example):
    if type(example) is torch.Tensor:return [example]
    if type(example) in (tuple,list) and example and all(type(x) is torch.Tensor for x in example):return list(example)
    if type(example) is dict and example and all(type(k) is str and type(v) is torch.Tensor for k,v in example.items()):return list(example.values())
    raise ExportError('INPUT_SPEC','a Tensor, nonempty flat Tensor tuple, or named Tensor mapping is required')

def invoke(model,example):
    return model(example) if type(example) in (torch.Tensor,dict) else model(*example)

def outputs(value):
    values=tensors(value)
    if any(v.numel()<1 or v.layout!=torch.strided or v.is_quantized or v.is_complex() for v in values):raise ExportError('OUTPUT_SPEC','nonempty dense real tensor outputs required')
    if sum(v.numel()*v.element_size() for v in values)>32*1024**2:raise ExportError('RESOURCE_BUDGET','logical output exceeds bounded validation/transfer budget')
    if any(not _all_finite(v,host=True) for v in values):raise ExportError('NONFINITE','output contains NaN/Inf')
    return values

def specs(values,names):
    return [dict(name=n,shape=list(t.shape),dtype=str(t.dtype),device=str(t.device)) for n,t in zip(names,values)]

def names(given,count,prefix):
    values=list(given) if given is not None else [prefix+str(i) for i in range(count)]
    if len(values)!=count or len(set(values))!=count or any(type(n) is not str or not n or len(n)>128 for n in values):raise ExportError('IO_NAMES','unique nonempty names must match tensor arity')
    return values

def fingerprint(tensor):
    from ..benchmark.runner import _finite_chunks,_bits
    h=hashlib.sha256()
    h.update(str((str(tensor.dtype),tuple(tensor.shape),tuple(tensor.stride()))).encode())
    for chunk in _finite_chunks(tensor):h.update(_bits(chunk).numpy().tobytes())
    return h.hexdigest()

def python_identity(modules):
    """Bound plain per-instance configuration and defining files; no pickle/repr."""
    from .artifacts import file_hash
    def plain(value,depth=0):
        if depth>16:raise ExportError('MODEL_IDENTITY','Python configuration exceeds supported nesting')
        if value is None or type(value) in (str,int,float,bool):return value
        if isinstance(value,(torch.dtype,torch.device)):return str(value)
        if isinstance(value,dict):
            if any(type(k) is not str for k in value):raise ExportError('MODEL_IDENTITY','plain string-keyed model configuration required')
            return {k:plain(v,depth+1) for k,v in value.items()}
        if type(value) in (list,tuple):return [plain(v,depth+1) for v in value]
        if isinstance(value,set) and all(type(v) is str for v in value):return sorted(value)
        raise ExportError('MODEL_IDENTITY','opaque nonregistered model configuration is unsupported')
    rows=[];files={}
    for name,module in modules:
        row={k:plain(v) for k,v in vars(module).items() if k not in ('_parameters','_buffers','_modules')}
        rows.append(dict(name=name,attributes=row))
        path=inspect.getsourcefile(type(module))
        if path and Path(path).is_file():files[str(Path(path).resolve())]=file_hash(path)
    raw=json.dumps(rows,sort_keys=True,allow_nan=False).encode()
    if len(raw)>1024**2:raise ExportError('MODEL_IDENTITY','Python configuration exceeds byte limit')
    return dict(python_configuration_sha256=hashlib.sha256(raw).hexdigest(),source_files=files)

@contextmanager
def isolated(model,example,config):
    ins=tensors(example)
    if not isinstance(model,torch.nn.Module):raise ExportError('MODEL_SPEC','nn.Module required')
    if any(x.device.type!='cpu' for x in ins):raise ExportError('CUDA_ADMISSION_REQUIRED','resident CUDA export must run through the admitted ExportAcceptance worker')
    if sum(t.numel()*t.element_size() for t in model.parameters())+sum(t.numel()*t.element_size() for t in model.buffers())>config.max_model_bytes:raise ExportError('RESOURCE_BUDGET','resident model exceeds bounded copy profile')
    if sum(t.numel()*t.element_size() for t in ins)>config.max_input_bytes:raise ExportError('RESOURCE_BUDGET','input exceeds bounded profile')
    modules,entries=_profile(model,ins[0].dtype)
    for _,module in modules:
        if any('__deepcopy__' in cls.__dict__ for cls in type(module).__mro__):raise ExportError('MODEL_COPY','custom model deepcopy protocol is outside the preservation profile')
    for t in ins:
        if t.numel()<1 or t.layout!=torch.strided or t.requires_grad or t.dtype!=ins[0].dtype or t.dtype not in (torch.float16,torch.float32,torch.float64) or not _all_finite(t):raise ExportError('INPUT_SPEC','nonempty finite dense no-grad FP16/FP32/FP64 same-dtype inputs required')
    rng=_rng_state()
    unique={id(t):t for _,_,t in entries}
    values=[(t,_signature(t),_host_copy(t),t._version) for t in unique.values()]
    gradients=[(p,p.grad,None if p.grad is None else _host_copy(p.grad)) for p in model.parameters()]
    modes=[(m,m.training,dict(m._parameters),dict(m._buffers),dict(m._modules)) for _,m in modules]
    input_values=[(x,_signature(x),_host_copy(x),x._version) for x in ins]
    descriptor,weights=_descriptor(modules,entries)
    source_python=python_identity(modules)
    try:
        owned=copy.deepcopy(model)
        if {id(m) for m in owned.modules()} & {id(m) for _,m in modules}:raise ExportError('MODEL_COPY','deepcopy retained source modules')
        source_storages={t.untyped_storage()._cdata for t in unique.values()}
        if any(t.untyped_storage()._cdata in source_storages for t in list(owned.parameters())+list(owned.buffers())):raise ExportError('MODEL_COPY','deepcopy retained source tensor storage')
        owned.eval();example_copy=copy.deepcopy(example)
        identity=dict(model_type=type(model).__module__+'.'+type(model).__qualname__,weights_sha256=weights,descriptor=descriptor,**source_python,
            preservation_scope='registered state, alias topology, mode, gradients, example inputs and RNG; trusted Python, not a sandbox')
        yield owned,example_copy,identity
    finally:
        _restore_rng(rng)
        unchanged=all(_signature(t)==sig and t._version==ver and _same_bits(t,saved) for t,sig,saved,ver in values+input_values)
        unchanged=unchanged and all(p.grad is grad and (grad is None or _same_bits(grad,saved)) for p,grad,saved in gradients)
        unchanged=unchanged and all(m.training==mode and m._parameters==ps and m._buffers==bs and m._modules==ms for m,mode,ps,bs,ms in modes)
        unchanged=unchanged and python_identity(modules)==source_python
        if not unchanged:raise ExportError('MODEL_PRESERVATION','source model or input changed; export cannot PASS')

def equivalence(reference,actual,config):
    refs=outputs(reference);vals=outputs(actual)
    if len(refs)!=len(vals):raise ExportError('OUTPUT_MISMATCH','output arity differs')
    measurements=[]
    for a,b in zip(refs,vals):
        if a.shape!=b.shape or a.dtype!=b.dtype:raise ExportError('OUTPUT_MISMATCH','shape/dtype mismatch')
        a=a.detach().cpu();b=b.detach().cpu()
        # No dtype coercion disguises mismatches. Accumulate errors in bounded slabs.
        from ..benchmark.runner import _finite_chunks
        absolute=relative=0.;matched=True
        for aa,bb in _finite_chunks(a,b):
            error=(aa.double()-bb.double()).abs()
            if error.numel():
                absolute=max(absolute,error.max().item());relative=max(relative,(error/aa.double().abs().clamp_min(1e-30)).max().item())
            matched=matched and (bool((error<=config.atol+config.rtol*aa.double().abs()).all()) if aa.is_floating_point() else torch.equal(aa,bb))
        measurements.append(dict(shape=list(a.shape),dtype=str(a.dtype),finite=True,max_absolute_error=absolute,max_relative_error=relative,relative_denominator_floor=1e-30,atol=config.atol,rtol=config.rtol,matched=matched))
    if not all(r['matched'] for r in measurements):raise ExportError('OUTPUT_MISMATCH','configured error tolerance exceeded',result={'comparisons':measurements})
    return measurements
