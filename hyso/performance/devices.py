"""P34 same-recipe inference comparisons in bounded, owned Windows workers.

Factories return (CPU torch module, CPU NCHW tensor). No caller model or optimizer
is passed to a worker. Factories are trusted executable code, not a sandbox.
"""
from dataclasses import dataclass, asdict
import hashlib
import math
import os
from pathlib import Path
import platform
import time

from .runtime_result import _result, _hash, _int, _number, _digest
from .stability import WorkloadRecipe, _run_owned, _load, _atomic, _factory


@dataclass(frozen=True)
class DeviceProtocol:
    warmup: int = 3
    samples: int = 20
    precision: str = 'float32'
    seed: int = 0
    startup_timeout_seconds: float = 45.
    step_timeout_seconds: float = 10.
    max_runtime_seconds: float = 60.

    def __post_init__(self):
        _int(self.warmup,0,100);_int(self.samples,3,10000);_int(self.seed,0,2**32-1)
        if self.precision not in ('float32','float64'):raise ValueError('only explicit FP32/FP64 supported')
        for value,limit in ((self.startup_timeout_seconds,120),(self.step_timeout_seconds,60),
                            (self.max_runtime_seconds,300)):
            _number(value,.05)
            if value>limit:raise ValueError('worker time budget exceeded')


def _missing(reason):
    return dict(value=None,status='unavailable',reason=reason)


def _launch(recipe, directory, mode, extra, protocol):
    p=protocol
    root=_run_owned(directory,mode,dict(recipe=asdict(recipe),protocol=asdict(p),**extra),
        startup_timeout=p.startup_timeout_seconds,heartbeat_timeout=p.step_timeout_seconds,
        max_runtime=p.max_runtime_seconds,max_bytes=64*1024**2)
    request=_load(root/'request.json');controller=_load(root/'controller.json')
    report=_load(root/'report.json',32*1024**2) if (root/'report.json').exists() else None
    return dict(token=request['token'],recipe=asdict(recipe),protocol=asdict(p),
                controller=controller,report=report,**extra)


def compare_devices(recipe, devices, directory, *, protocol=DeviceProtocol()):
    """Explicit 'cpu' or physical GPU UUIDs; serial resident eager inference.

    Energy and quality remain null: this adapter has no labeled evaluation or
    sufficiently sampled power interval. It never estimates these from latency.
    Evidence includes the explicit physical GPU UUID and local recipe paths for
    identity validation. Keep raw evidence private; a shared summary should omit
    those fields without being presented as the original verifiable artifact.
    """
    if type(recipe) is not WorkloadRecipe or type(protocol) is not DeviceProtocol:
        raise TypeError('typed recipe and protocol required')
    if recipe.gpu_uuid is not None:raise ValueError('select devices explicitly, outside recipe')
    if type(devices) not in (list,tuple) or not 1<=len(devices)<=8 or len(set(devices))!=len(devices):
        raise ValueError('one to eight distinct device identities required')
    for device in devices:
        if device!='cpu':WorkloadRecipe(recipe.factory,recipe.configuration,recipe.search_path,device)
    root=Path(directory);root.mkdir(parents=False,exist_ok=False)
    runs=[]
    for index,device in enumerate(devices):
        selected=WorkloadRecipe(recipe.factory,recipe.configuration,recipe.search_path,
                                device if device!='cpu' else None)
        runs.append(_launch(selected,root/str(index),'devices',dict(device=device),protocol))
    return _result('P34',dict(devices=list(devices),protocol=asdict(protocol),runs=runs))


def _valid_run(run):
    c=run['controller']
    if c['token']!=run['token'] or c['cleanup']['active_processes']!=0 or not c['kill_on_close']:
        raise ValueError('worker ownership/cleanup mismatch')
    report=run['report']
    if report is not None and (report['token']!=run['token'] or report['request_sha256']!=_hash(
            dict(recipe=run['recipe'],protocol=run['protocol'],
                 **{k:run[k] for k in ('device','providers','expected','optimization','runtime_search_path','provider_options') if k in run}))):
        raise ValueError('report belongs to another request/session')
    if c['reason'] or c['exit_code']!=0 or report is None:
        return None,c['reason'] or 'worker_failed_or_missing_report'
    if report['status']!='completed':return None,report['reason']
    return report,None


def _analyze_devices(e):
    p=DeviceProtocol(**e['protocol']);runs=e['runs'];devices=e['devices']
    if not 1<=len(devices)<=8 or len(set(devices))!=len(devices) or len(runs)!=len(devices):
        raise ValueError('invalid comparison devices')
    if len({run['token'] for run in runs})!=len(runs):raise ValueError('independent worker runs required')
    rows=[];identities=[]
    for device,run in zip(devices,runs):
        if run['device']!=device or run['protocol']!=e['protocol']:raise ValueError('protocol/device mismatch')
        recipe=WorkloadRecipe(**run['recipe'])
        if recipe.gpu_uuid!=(device if device!='cpu' else None):raise ValueError('device admission mismatch')
        report,reason=_valid_run(run)
        if report is None:
            rows.append(dict(device=device,status='failed',reason=reason,latency=_missing(reason),
                throughput=_missing(reason),memory=_missing(reason),energy=_missing(reason),quality=_missing(reason)))
            continue
        ident=report['identity'];samples=report['latency_ns'];window=report['window_ns']
        if ident['device_id']!=device or ident['precision']!=p.precision or ident['backend']!='pytorch_eager':
            raise ValueError('backend/precision/device identity mismatch')
        for key in ('model_sha256','input_sha256','factory_sha256'):_digest(ident[key])
        if (ident['warmup']!=p.warmup or ident['measured_samples']!=p.samples or
                ident['scope']!='resident forward + terminal device synchronization' or
                ident['clock']!='perf_counter_ns' or ident['shape'][0]!=ident['batch']):
            raise ValueError('measurement scope/protocol mismatch')
        if len(samples)!=p.samples or report['completed']!=p.samples*ident['batch']:
            raise ValueError('incomplete completed sample ledger')
        _int(ident['batch'],1)
        for sample in samples:_int(sample,1)
        _int(window,sum(samples))
        if report['model_unchanged'] is not True:raise ValueError('inference mutated model state')
        memory=report['memory']
        for key in ('rss_before_bytes','rss_after_bytes'):_int(memory[key],1)
        if memory['cuda_peak_allocated_bytes'] is not None:_int(memory['cuda_peak_allocated_bytes'],1)
        identities.append({k:ident[k] for k in ('model_sha256','input_sha256','factory_sha256',
                                              'backend','precision','batch','scope','runtime')})
        ordered=sorted(samples);mean=sum(samples)/len(samples);variance=sum((x-mean)**2 for x in samples)/(len(samples)-1)
        rows.append(dict(device=device,status='measured',reason=None,identity=ident,
            latency=dict(status='measured',reason=None,value=dict(mean_ms=mean/1e6,
                median_ms=(ordered[(len(ordered)-1)//2]+ordered[len(ordered)//2])/2e6,
                min_ms=ordered[0]/1e6,max_ms=ordered[-1]/1e6,
                conditional_standard_error_ms=math.sqrt(variance/len(samples))/1e6),
                uncertainty='Descriptive samples; independence, thermal stability and confidence bounds unproven'),
            throughput=dict(status='measured',reason=None,value=report['completed']/(window/1e9),
                scope='completed images / instrumented host wall including validation and heartbeat overhead'),
            memory=dict(status='measured',reason=None,value=memory,
                scope='process RSS snapshots; CUDA owned-worker allocator peak, not model-only/physical VRAM'),
            energy=_missing('no power integration adapter in comparison window'),
            quality=_missing('no labeled evaluation protocol')))
    comparable=len(identities)>=2 and len(identities)==len(devices) and all(x==identities[0] for x in identities)
    return dict(status='comparable' if comparable else 'limited',devices=rows,comparable=comparable,
        reason=None if comparable else 'requires two successful devices with identical model/input/runtime/backend/precision/scope',
        ranking=None,confidence='observed serial runs; cross-run environmental variation unquantified')


def _tensor_hash(tensor):
    value=tensor.detach().cpu().contiguous()
    raw=value.reshape(-1).view(__import__('torch').uint8).numpy().tobytes()
    return _hash(dict(dtype=str(value.dtype),shape=list(value.shape),bytes_sha256=hashlib.sha256(raw).hexdigest()))


def _model_hash(model):
    import inspect
    definitions={}
    for module in model.modules():
        cls=type(module);key=cls.__module__+'.'+cls.__qualname__
        definitions[key]=hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()
    return _hash(dict(structure=repr(model),definitions=definitions,
                      state={k:_tensor_hash(v) for k,v in model.state_dict().items()}))


def _worker_devices(root, request):
    import random
    import numpy as np
    import torch
    from .memory import _observe
    p=DeviceProtocol(**request['protocol']);torch.set_num_threads(1)
    random.seed(p.seed);np.random.seed(p.seed);torch.manual_seed(p.seed)
    (model,inputs),source=_factory(request['recipe'])
    if not isinstance(model,torch.nn.Module) or not isinstance(inputs,torch.Tensor):
        raise TypeError('factory must return module and tensor')
    if inputs.device.type!='cpu' or inputs.ndim!=4 or not inputs.is_floating_point():
        raise ValueError('resident floating CPU NCHW tensor required')
    if any(t.device.type!='cpu' for t in list(model.parameters())+list(model.buffers())):
        raise ValueError('factory must construct CPU model')
    device=request['device'];cuda=device!='cpu'
    if cuda:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=device or not torch.cuda.is_available():
            raise RuntimeError('requested CUDA device unavailable')
        torch.cuda.set_per_process_memory_fraction(.1,0)
    target=torch.device('cuda:0' if cuda else 'cpu');dtype=getattr(torch,p.precision)
    model=model.to(dtype=dtype).eval();inputs=inputs.detach().clone().to(dtype=dtype).contiguous()
    before=_model_hash(model);input_hash=_tensor_hash(inputs)
    identity=dict(model_sha256=before,input_sha256=input_hash,factory_sha256=source['module_sha256'],
        device_id=device,device_name=torch.cuda.get_device_name(0) if cuda else platform.processor(),
        runtime=dict(torch=str(torch.__version__),python=platform.python_version()),backend='pytorch_eager',
        precision=p.precision,batch=len(inputs),shape=list(inputs.shape),warmup=p.warmup,
        measured_samples=p.samples,clock='perf_counter_ns',scope='resident forward + terminal device synchronization')
    model=model.to(target);inputs=inputs.to(target)
    def step():
        output=model(inputs)
        if cuda:torch.cuda.synchronize(0)
        return output
    def validate(output):
        if not isinstance(output,torch.Tensor) or output.shape[0]!=len(inputs) or not torch.isfinite(output).all().item():
            raise ValueError('finite batched tensor output required')
    with torch.inference_mode():
        for _ in range(p.warmup):validate(step());_atomic(root/'heartbeat.json',dict(warmup=True))
        rss_before=_observe()['rss_bytes']
        if cuda:torch.cuda.reset_peak_memory_stats(0)
        samples=[];start=time.perf_counter_ns()
        for index in range(p.samples):
            a=time.perf_counter_ns();output=step();b=time.perf_counter_ns();validate(output)
            samples.append(b-a);_atomic(root/'heartbeat.json',dict(index=index))
        end=time.perf_counter_ns()
    return dict(identity=identity,latency_ns=samples,window_ns=end-start,completed=p.samples*len(inputs),
        model_unchanged=_model_hash(model)==before,
        memory=dict(rss_before_bytes=rss_before,rss_after_bytes=_observe()['rss_bytes'],
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0) if cuda else None))


def _worker_dispatch(root, request):
    _atomic(root/'ready.json',dict(token=request['token'],pid=os.getpid()))
    report=dict(token=request['token'],request_sha256=_hash({k:request[k] for k in
        ('recipe','protocol','device','providers','expected','optimization','runtime_search_path','provider_options') if k in request}))
    try:
        if request['mode']=='devices':data=_worker_devices(root,request)
        else:
            from .providers import _worker_providers
            data=_worker_providers(root,request)
        report.update(data,status='completed',reason=None)
    except Exception as exc:
        if request['mode']=='providers':
            from .providers import _ProviderFailure
            if isinstance(exc,_ProviderFailure):report.update(exc.details)
        report.update(status='failed',reason=type(exc).__name__+': '+str(exc)[:1024])
    _atomic(root/'report.json',report)
