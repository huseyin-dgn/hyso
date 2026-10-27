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
import uuid

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
    cuda_memory_fraction: float | None = None
    process_memory_bytes: int | None = None
    job_memory_bytes: int | None = None
    max_processes: int | None = None

    def __post_init__(self):
        _int(self.warmup,0,100);_int(self.samples,3,10000);_int(self.seed,0,2**32-1)
        if self.precision not in ('float32','float64'):raise ValueError('only explicit FP32/FP64 supported')
        for value,limit in ((self.startup_timeout_seconds,120),(self.step_timeout_seconds,60),
                            (self.max_runtime_seconds,300)):
            _number(value,.05)
            if value>limit:raise ValueError('worker time budget exceeded')
        if self.cuda_memory_fraction is not None:
            _number(self.cuda_memory_fraction,.000001)
            if self.cuda_memory_fraction>1:raise ValueError('invalid CUDA allocator fraction')
        for limit in (self.process_memory_bytes,self.job_memory_bytes):
            if limit is not None:_int(limit,1)
        if self.max_processes is not None:_int(self.max_processes,1,2**32-1)
        if (self.process_memory_bytes is not None and self.job_memory_bytes is not None
                and self.process_memory_bytes>self.job_memory_bytes):raise ValueError('job memory must cover one process')


def _device_id(value):
    if value=='cpu':return value
    if type(value) is not str or not value.startswith('GPU-'):raise ValueError('full physical GPU UUID required')
    try:physical=uuid.UUID(value[4:])
    except (ValueError,AttributeError) as exc:raise ValueError('full physical GPU UUID required') from exc
    if value[4:].lower()!=str(physical):raise ValueError('full hyphenated physical GPU UUID required')
    return 'GPU-'+str(physical)


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
    if type(devices) not in (list,tuple) or not 1<=len(devices)<=8:
        raise ValueError('one to eight distinct device identities required')
    devices=[_device_id(d) for d in devices]
    if len(set(devices))!=len(devices):raise ValueError('distinct physical devices required')
    for device in devices:
        if device!='cpu':WorkloadRecipe(recipe.factory,recipe.configuration,recipe.search_path,device)
    root=Path(directory);root.mkdir(parents=False,exist_ok=False)
    runs=[]
    for index,device in enumerate(devices):
        selected=WorkloadRecipe(recipe.factory,recipe.configuration,recipe.search_path,
                                device if device!='cpu' else None)
        runs.append(_launch(selected,root/str(index),'devices',dict(device=device),protocol))
    return _result('P34',dict(comparison_version=2,devices=list(devices),protocol=asdict(protocol),runs=runs))


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


def _analyze_devices_legacy(e):
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


_MEMORY_DOMAINS = {
    'rss_before_bytes':'process_RSS_snapshot', 'rss_after_bytes':'process_RSS_snapshot',
    'cuda_peak_allocated_bytes':'owned_worker_native_CUDA_allocator_window_peak_allocated',
    'cuda_peak_reserved_bytes':'owned_worker_native_CUDA_allocator_window_peak_reserved',
    'logical_model_bytes':'registered_parameter_and_buffer_logical_bytes',
}


def _analyze_devices(e):
    version=e.get('comparison_version',1)
    if type(version) is not int or version not in (1,2):raise ValueError('unsupported comparison version')
    if version==1:return _analyze_devices_legacy(e)
    p=DeviceProtocol(**e['protocol']);devices=e['devices'];runs=e['runs']
    if (not 1<=len(devices)<=8 or devices!=[_device_id(d) for d in devices]
            or len(set(devices))!=len(devices) or len(runs)!=len(devices)):
        raise ValueError('distinct canonical physical devices required')
    if len({r['token'] for r in runs})!=len(runs):raise ValueError('independent worker runs required')
    rows=[];contracts=[];physical=[]
    for device,run in zip(devices,runs):
        if run['device']!=device or run['protocol']!=e['protocol']:raise ValueError('protocol/device mismatch')
        recipe=WorkloadRecipe(**run['recipe'])
        if recipe.gpu_uuid!=(None if device=='cpu' else device):raise ValueError('device admission mismatch')
        c=run['controller']
        if (c['memory_limits']!=dict(process_memory_bytes=p.process_memory_bytes,job_memory_bytes=p.job_memory_bytes)
                or c['process_limit']!=p.max_processes):raise ValueError('controller resource policy mismatch')
        report,reason=_valid_run(run)
        if report is None:
            rows.append(dict(device=device,status='failed',reason=reason,**{k:_missing(reason)
                for k in ('latency','throughput','memory','energy','quality')}));continue
        ident=report['identity'];cuda=device!='cpu'
        if (ident['device_id']!=device or ident['precision']!=p.precision or ident['backend']!='pytorch_eager'
                or ident['worker_pid']!=c['launcher_pid']):raise ValueError('backend/precision/device/worker identity mismatch')
        hardware=ident['hardware']
        if (hardware['type']!=('cuda' if cuda else 'cpu') or hardware['physical_id']!=device
                or hardware['identity_source']!=('torch.cuda.get_device_properties.uuid' if cuda else 'host_CPU')):
            raise ValueError('observed physical device identity mismatch')
        physical.append(hardware['physical_id'])
        if len(set(physical))!=len(physical):raise ValueError('duplicate physical device')
        if cuda:
            if (hardware['logical_index']!=0 or hardware['visible_device_count']!=1
                    or hardware['allocator_backend']!='native'):raise ValueError('unverified CUDA mapping/allocator')
            _int(hardware['total_memory_bytes'],1)
        for key in ('model_sha256','input_sha256','factory_sha256'):_digest(ident[key])
        shape=ident['shape'];_int(ident['batch'],1)
        if len(shape)!=4:raise ValueError('NCHW input required')
        for dim in shape:_int(dim,1)
        execution=ident['execution']
        if (ident['warmup']!=p.warmup or ident['measured_samples']!=p.samples or shape[0]!=ident['batch']
                or ident['clock']!='perf_counter_ns'
                or ident['scope']!='resident forward + terminal device synchronization'
                or ident['synchronization']!=('device_drain_before_and_after_each_forward' if cuda else 'synchronous_host_forward')
                or execution['input_device']!=('cuda:0' if cuda else 'cpu')
                or execution['output_device']!=execution['input_device']
                or execution['output_dtype']!='torch.'+p.precision
                or report['input_unchanged'] is not True or report['model_unchanged'] is not True):
            raise ValueError('measurement scope/protocol mismatch or mutated model/input')
        settings=execution['settings']
        for key in ('inference_mode','autocast_enabled','cudnn_benchmark','cudnn_deterministic',
                    'cudnn_allow_tf32','matmul_allow_tf32','deterministic_algorithms','mkldnn_enabled'):
            if type(settings[key]) is not bool:raise ValueError('explicit precision settings required')
        for key in ('num_threads','num_interop_threads'):_int(settings[key],1)
        if settings['float32_matmul_precision'] not in ('highest','high','medium'):raise ValueError('unknown matmul precision')
        samples=report['latency_ns'];window=report['window_ns']
        if len(samples)!=p.samples or report['completed']!=p.samples*ident['batch']:raise ValueError('incomplete completed sample ledger')
        for sample in samples:_int(sample,1)
        _int(window,sum(samples))
        if report['cuda_memory_fraction']!=(p.cuda_memory_fraction if cuda else None):raise ValueError('allocator resource policy mismatch')
        memory=report['memory'];domains=report['memory_domains']
        for key in _MEMORY_DOMAINS:
            value=memory.get(key)
            if value is not None:_int(value)
            if not cuda and key.startswith('cuda_') and value is not None:raise ValueError('CPU cannot attest CUDA memory')
        allocated=memory.get('cuda_peak_allocated_bytes');reserved=memory.get('cuda_peak_reserved_bytes')
        if allocated is not None and reserved is not None and allocated>reserved:raise ValueError('inconsistent allocator memory')
        mean=sum(samples)/len(samples);ordered=sorted(samples)
        variance=sum((x-mean)**2 for x in samples)/(len(samples)-1)
        contracts.append(dict(model=ident['model_sha256'],input=ident['input_sha256'],factory=ident['factory_sha256'],
            recipe=dict(factory=recipe.factory,configuration=recipe.configuration),shape=shape,
            precision=ident['precision'],backend=ident['backend'],runtime=ident['runtime'],settings=settings,
            scope=ident['scope'],clock=ident['clock']))
        rows.append(dict(device=device,status='measured',reason=None,identity=ident,
            latency=dict(status='measured',reason=None,value=dict(mean_ms=mean/1e6,
                median_ms=(ordered[(len(ordered)-1)//2]+ordered[len(ordered)//2])/2e6,
                min_ms=ordered[0]/1e6,max_ms=ordered[-1]/1e6,
                conditional_standard_error_ms=math.sqrt(variance/len(samples))/1e6),
                uncertainty='Descriptive samples; independence, thermal stability and confidence bounds unproven'),
            throughput=dict(status='measured',reason=None,value=report['completed']/(window/1e9),
                scope='completed images / instrumented host wall including validation and heartbeat overhead'),
            memory=dict(status='measured' if any(v is not None for v in memory.values()) else 'unavailable',
                reason=None if any(v is not None for v in memory.values()) else 'no memory observations',value=memory,
                domains=domains,scope='separate RSS, native CUDA allocator and logical model domains; no memory ranking'),
            energy=_missing('no power integration adapter in comparison window'),quality=_missing('no labeled evaluation protocol')))
    comparable=len(contracts)==len(devices) and len(contracts)>=2 and all(x==contracts[0] for x in contracts)
    differences=sorted({k for contract in contracts[1:] for k in contract if contract[k]!=contracts[0][k]})
    axes={k:dict(comparable=comparable,reason=None if comparable else 'missing device or incompatible execution contract')
          for k in ('latency','throughput')}
    for key,scope in _MEMORY_DOMAINS.items():
        available=(len(rows)>=2 and all(row['status']=='measured' and row['memory']['value'].get(key) is not None
            and row['memory']['domains'].get(key)==dict(unit='byte',scope=scope) for row in rows))
        eligible=comparable and available and key=='logical_model_bytes'
        axes[key]=dict(comparable=eligible,reason=None if eligible else
            'missing/incompatible metric or process/allocator domains do not establish model memory superiority')
    for key in ('quality','energy'):axes[key]=dict(comparable=False,reason='metric unavailable')
    return dict(status='comparable' if comparable else 'limited',devices=rows,comparable=comparable,
        reason=None if comparable else 'requires two successful distinct devices with matching execution contracts',
        differences=differences,axes=axes,ranking=None,
        confidence='observed serial runs; cross-run environmental variation unquantified')


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


def _timed_forward(model, inputs, synchronize=None):
    # Drain resident transfers/previous work outside the latency interval. The
    # terminal drain is inside it, so asynchronous launches cannot appear complete.
    if synchronize is not None:synchronize()
    start=time.perf_counter_ns();output=model(inputs)
    if synchronize is not None:synchronize()
    return output,time.perf_counter_ns()-start


def _cpu_name():
    if os.name=='nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            return winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    return platform.processor()


def _cuda_hardware(torch, device):
    props=torch.cuda.get_device_properties(0)
    # The request/environment is selection intent; only the runtime observation
    # independently establishes which physical device executes this worker.
    observed=_device_id('GPU-'+str(props.uuid).removeprefix('GPU-'))
    if observed!=device or torch.cuda.device_count()!=1:raise ValueError('observed physical GPU UUID mismatch')
    import subprocess
    try:
        driver=subprocess.check_output(['nvidia-smi','--id='+observed,'--query-gpu=driver_version',
            '--format=csv,noheader'],timeout=5,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).decode().strip()
    except (OSError,subprocess.SubprocessError):driver=None
    return dict(type='cuda',physical_id=observed,identity_source='torch.cuda.get_device_properties.uuid',
        logical_index=0,visible_device_count=torch.cuda.device_count(),name=props.name,
        capability=[props.major,props.minor],total_memory_bytes=props.total_memory,driver_version=driver,
        allocator_backend=torch.cuda.get_allocator_backend())


def _worker_devices(root, request):
    import random
    import numpy as np
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from .memory import _observe
    p=DeviceProtocol(**request['protocol']);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
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
        if p.cuda_memory_fraction is not None:torch.cuda.set_per_process_memory_fraction(p.cuda_memory_fraction,0)
        hardware=_cuda_hardware(torch,device)
        if hardware['allocator_backend']!='native':raise NotImplementedError('native CUDA allocator required')
    else:hardware=dict(type='cpu',physical_id='cpu',identity_source='host_CPU',name=_cpu_name(),logical_processors=os.cpu_count())
    target=torch.device('cuda:0' if cuda else 'cpu');dtype=getattr(torch,p.precision)
    model=model.to(dtype=dtype).eval();inputs=inputs.detach().clone().to(dtype=dtype).contiguous()
    before=_model_hash(model);input_hash=_tensor_hash(inputs)
    identity=dict(model_sha256=before,input_sha256=input_hash,factory_sha256=source['module_sha256'],
        device_id=device,device_name=hardware['name'],hardware=hardware,worker_pid=os.getpid(),
        runtime=dict(torch=str(torch.__version__),python=platform.python_version(),cuda_build=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),platform=platform.platform()),backend='pytorch_eager',
        precision=p.precision,batch=len(inputs),shape=list(inputs.shape),warmup=p.warmup,
        measured_samples=p.samples,clock='perf_counter_ns',scope='resident forward + terminal device synchronization',
        synchronization='device_drain_before_and_after_each_forward' if cuda else 'synchronous_host_forward')
    logical=sum(t.numel()*t.element_size() for t in list(model.parameters())+list(model.buffers()))
    model=model.to(target);inputs=inputs.to(target)
    synchronize=(lambda:torch.cuda.synchronize(0)) if cuda else None
    def validate(output):
        if (not isinstance(output,torch.Tensor) or output.ndim<1 or output.shape[0]!=len(inputs)
                or output.device!=inputs.device or output.dtype!=dtype or not torch.isfinite(output).all().item()):
            raise ValueError('finite batched tensor output required')
        if any(t.device!=inputs.device or (t.is_floating_point() and t.dtype!=dtype)
               for t in list(model.parameters())+list(model.buffers())):raise ValueError('model device/precision mismatch')
    with torch.inference_mode():
        for _ in range(p.warmup):
            output,_=_timed_forward(model,inputs,synchronize);validate(output);del output
            _atomic(root/'heartbeat.json',dict(warmup=True))
        rss_before=_observe()['rss_bytes']
        if cuda:synchronize()
        if cuda:torch.cuda.reset_peak_memory_stats(0)
        samples=[];start=time.perf_counter_ns()
        for index in range(p.samples):
            output,elapsed=_timed_forward(model,inputs,synchronize);validate(output)
            samples.append(elapsed)
            if index==p.samples-1:
                output_signature=dict(device=str(output.device),dtype=str(output.dtype),shape=list(output.shape))
            del output
            _atomic(root/'heartbeat.json',dict(index=index))
        end=time.perf_counter_ns()
        settings=dict(num_threads=torch.get_num_threads(),num_interop_threads=torch.get_num_interop_threads(),
            inference_mode=torch.is_inference_mode_enabled(),autocast_enabled=torch.is_autocast_enabled(),
            cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
            float32_matmul_precision=torch.get_float32_matmul_precision(),cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            mkldnn_enabled=torch.backends.mkldnn.enabled)
    identity['execution']=dict(input_device=str(inputs.device),output_device=output_signature['device'],
        output_dtype=output_signature['dtype'],output_shape=output_signature['shape'],settings=settings)
    memory=dict(rss_before_bytes=rss_before,rss_after_bytes=_observe()['rss_bytes'],logical_model_bytes=logical,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0) if cuda else None,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0) if cuda else None)
    return dict(identity=identity,latency_ns=samples,window_ns=end-start,completed=p.samples*len(inputs),
        model_unchanged=_model_hash(model)==before,input_unchanged=_tensor_hash(inputs)==input_hash,
        cuda_memory_fraction=p.cuda_memory_fraction if cuda else None,memory=memory,
        memory_domains={k:dict(unit='byte',scope=v) for k,v in _MEMORY_DOMAINS.items()})


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
