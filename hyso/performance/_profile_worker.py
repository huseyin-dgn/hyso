"""Fresh profiler process. Bootstrap is stdlib and memory-limited before Torch."""
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid


def _ready(path,phase):
    temporary=Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(dict(pid=os.getpid(),phase=phase)),encoding='utf-8')
    os.replace(temporary,path)


def _duration(value):
    return float(value) if type(value) in (int,float) and math.isfinite(value) and value>=0 else None


def _meta(value,depth=0):
    import torch
    if depth>8:return dict(kind='unrecorded',reason='metadata_depth_limit')
    if isinstance(value,torch.Tensor):
        return dict(kind='tensor',shape=list(value.shape),dtype=str(value.dtype),device=str(value.device))
    if type(value) in (tuple,list):return [_meta(v,depth+1) for v in value]
    if type(value) is dict:return {str(k):_meta(v,depth+1) for k,v in value.items()}
    if type(value) is float and not math.isfinite(value):return dict(kind='unrecorded',reason='nonfinite_scalar_metadata')
    if value is None or type(value) in (int,float,bool,str):return value
    return dict(kind='unrecorded',type=type(value).__name__)


def _inspect(model,dtype,device):
    import torch
    from hyso.benchmark.runner import _tensor
    from torch.nn.modules import module as module_api
    if not isinstance(model,torch.nn.Module):raise TypeError('factory must return nn.Module')
    if any(n.startswith('_global_') and n.endswith('_hooks') and bool(v) for n,v in vars(module_api).items() if isinstance(v,dict)):
        raise ValueError('worker global Module hooks unsupported')
    modules=[];entries=[];inventory=[];seen={};paths=0
    def visit(m,path,ancestors):
        nonlocal paths
        paths+=1
        if paths>10000:raise ValueError('module path budget exceeded')
        if id(m) in ancestors:raise ValueError('cyclic modules unsupported')
        if id(m) in seen:inventory[seen[id(m)]]['aliases'].append(path)
        else:
            if isinstance(m,(torch.nn.DataParallel,torch.nn.parallel.DistributedDataParallel,torch.jit.ScriptModule)) or '_orig_mod' in vars(m):
                raise ValueError('parallel/compiled/scripted model unsupported')
            for name in ('train','eval','children','named_children'):
                if name in vars(m) or getattr(type(m),name) is not getattr(torch.nn.Module,name):
                    raise ValueError('custom mode/traversal unsupported')
            i=len(modules);seen[id(m)]=i;modules.append((path,m))
            inventory.append(dict(module_id='m'+str(i),path=path,aliases=[path],
                qualified_class=type(m).__module__+'.'+type(m).__qualname__,parent_path=path.rpartition('.')[0] if path else None))
            for kind,mapping in (('parameter',m._parameters),('buffer',m._buffers)):
                for name,t in mapping.items():
                    if t is None:continue
                    _tensor(t,dtype,parameter=kind=='parameter',device=device)
                    entries.append((kind,(path+'.' if path else '')+name,t))
                    if kind=='parameter' and t.grad is not None:
                        if t.grad.device!=device or t.grad.layout!=torch.strided:raise ValueError('unsupported gradient')
        for name,child in m._modules.items():
            if child is not None:visit(child,(path+'.' if path else '')+name,ancestors|{id(m)})
    try:visit(model,'',set())
    finally:visit=None
    return modules,entries,inventory


def _hooks(modules):
    return [(m,{n:(v,dict(v)) for n,v in vars(m).items() if n.startswith('_forward') and '_hooks' in n and isinstance(v,dict)}) for _,m in modules]


def _hooks_equal(saved):
    return all(getattr(m,n) is original and list(original)==list(values) and
        all(original[k] is v for k,v in values.items()) for m,maps in saved for n,(original,values) in maps.items())


def main():
    request_path,output_path,trace_path,ready_path,package_root=sys.argv[1:]
    _ready(ready_path,'bootstrap')
    request=json.loads(Path(request_path).read_text(encoding='utf-8'));r=request['recipe'];p=request['protocol']
    body=dict(worker_pid=os.getpid(),status='failed',reason=None,cleanup={},result=None)
    guard=None;snapshot=None;saved_hooks=None;rng=None;cuda_rng=None;device=None;stream=None;handles=[];active=[]
    try:
        # This existing Windows guard owns only this fresh process and disallows children.
        spec=importlib.util.spec_from_file_location('hyso_profile_memory_guard',Path(__file__).with_name('_batch_worker.py'))
        bootstrap=importlib.util.module_from_spec(spec);spec.loader.exec_module(bootstrap)
        guard=bootstrap._memory_guard(p['cpu_memory_limit_bytes'])
        if math.prod(r['input_shape'])*(8 if r['precision']=='fp64' else 4)>p['input_limit_bytes']:
            raise ValueError('input exceeds explicit resource budget')
        sys.path.insert(0,package_root)
        if r['factory_search_path']:sys.path.append(r['factory_search_path'])
        import random
        import numpy as np
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        from hyso.benchmark.runner import _Snapshot,_descriptor,_bits,_output,StateMutationError
        from hyso.performance.inference import _base,_context_check
        from hyso.performance.profiling import _operator_metrics,_cuda_metrics,_profile_status
        from hyso.performance._profile_evidence import module_timings,normalize_trace,correlate
        from hyso.performance.memory import _file_info
        torch.set_num_threads(1);torch.manual_seed(r['seed']);random.seed(r['seed']);np.random.seed(r['seed']%(2**32))
        module,name=r['factory'].split(':');factory=importlib.import_module(module)
        for part in name.split('.'):factory=getattr(factory,part)
        model=factory(**r['model_kwargs'])
        if not isinstance(model,torch.nn.Module):raise TypeError('factory must return nn.Module')
        if torch.cuda.is_initialized() or any(t.device.type!='cpu' for t in list(model.parameters())+list(model.buffers())):
            raise ValueError('factory must construct CPU model without CUDA initialization')
        if r['artifact_path']:
            if _file_info(r['artifact_path'])['sha256']!=r['artifact_sha256']:raise ValueError('artifact SHA mismatch')
            state=torch.load(r['artifact_path'],map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True);del state
        device=torch.device(r['device'])
        if device.type=='cuda':
            if not torch.cuda.is_available():raise NotImplementedError('CUDA unavailable; no CPU fallback')
            device=torch.device('cuda',0 if device.index is None else device.index);torch.cuda.set_device(device)
            torch.cuda.set_per_process_memory_fraction(p['cuda_memory_fraction'],device);stream=torch.cuda.current_stream(device)
        dtype=torch.float64 if r['precision']=='fp64' else torch.float32
        model=model.to(device=device,dtype=dtype).eval();inputs=torch.ones(r['input_shape'],device=device,dtype=dtype)
        modules,entries,inventory=_inspect(model,dtype,device)
        snapshot=_Snapshot(modules,entries,inputs);saved_hooks=_hooks(modules)
        descriptor,weights=_descriptor(modules,entries);digest=hashlib.sha256(_bits(inputs).numpy().tobytes()).hexdigest()
        rng=random.getstate(),np.random.get_state(),torch.get_rng_state()
        if stream is not None:cuda_rng=torch.cuda.get_rng_state(device)
        def forward():
            with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=r['precision']=='amp_fp16'):
                return model(inputs)
        for _ in range(p['warmup_iterations']):
            out=forward();_output(out,inputs.shape[0],device=device);del out
        snapshot.check(full=True);_context_check(device,stream)
        if stream is not None:stream.synchronize()
        calls=[];invocations=[];measurement_index=0
        class Observer(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                kwargs=kwargs or {};out=func(*args,**kwargs)
                if str(func).startswith('profiler.'):return out
                if len(calls)>=p['max_events']:raise RuntimeError('dispatch evidence budget exceeded')
                bound={}
                for index,arg in enumerate(func._schema.arguments):
                    if index<len(args):bound[arg.name]=_meta(args[index])
                    elif arg.name in kwargs:bound[arg.name]=_meta(kwargs[arg.name])
                    elif arg.has_default_value():bound[arg.name]=_meta(arg.default_value)
                calls.append(dict(call_index=len(calls),measurement_index=measurement_index,
                    operator=str(func),arguments=bound,output=_meta(out)))
                return out
        ids={id(m):inventory[i]['module_id'] for i,(_,m) in enumerate(modules)}
        by_module={};prefix='hyso.profile.'+str(uuid.uuid4())+'.module.'
        def pre(m,args,kwargs):
            if len(invocations)>=p['max_events']:raise RuntimeError('module evidence budget exceeded')
            i=len(invocations);row=dict(invocation_id=i,module_id=ids[id(m)],
                parent_invocation_id=active[-1][0] if active else None,measurement_index=measurement_index,
                annotation=prefix+str(i),input_shapes=_meta([args,kwargs]),output_shapes=None,
                cpu_total_us=None,cpu_exclusive_us=None,cuda_activity_us=None)
            invocations.append(row);context=torch.profiler.record_function(row['annotation']);context.__enter__()
            active.append((i,context));by_module.setdefault(id(m),[]).append(i)
        def post(m,args,kwargs,out):
            stack=by_module.get(id(m),[])
            if not stack:return
            i=stack.pop();invocations[i]['output_shapes']=_meta(out)
            if not active or active[-1][0]!=i:raise RuntimeError('module call nesting unsupported')
            _,ctx=active.pop();ctx.__exit__(None,None,None)
        for _,m in modules:
            handles.append(m.register_forward_pre_hook(pre,with_kwargs=True))
            handles.append(m.register_forward_hook(post,with_kwargs=True,always_call=True))
        activities=[torch.profiler.ProfilerActivity.CPU]
        if stream is not None:activities.append(torch.profiler.ProfilerActivity.CUDA)
        supported=[str(a).split('.')[-1] for a in torch.profiler.supported_activities()]
        capture_start=time.perf_counter_ns();host_windows=[]
        with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=False,
                                    with_stack=False,with_flops=False) as prof:
            _ready(ready_path,'profiler_active')
            with Observer():
                for measurement_index in range(p['measure_iterations']):
                    t=time.perf_counter_ns();out=forward();host_windows.append((time.perf_counter_ns()-t)/1e6)
                    # Validation/copies remain outside profiler capture.
                    if measurement_index+1<p['measure_iterations']:del out
            if stream is not None:stream.synchronize()
        lifecycle_ms=(time.perf_counter_ns()-capture_start)/1e6
        boundary=_output(out,inputs.shape[0],device=device);del out
        _context_check(device,stream);snapshot.check(full=True)
        if device.type=='cpu' and torch.cuda.is_initialized():raise ValueError('CPU recipe initialized CUDA during forward')
        for handle in reversed(handles):handle.remove()
        handles.clear()
        if not _hooks_equal(saved_hooks):raise StateMutationError('factory hooks changed during forward')
        events=[]
        native_events=prof.events()
        if len(native_events)>p['max_events']:raise ValueError('profiler event budget exceeded')
        for event in native_events:
            events.append(dict(event_id=event.id,name=event.name,
                device_type=str(event.device_type).split('.')[-1],scope=int(event.scope),
                parent_id=event.cpu_parent.id if event.cpu_parent else None,
                start_us=_duration(event.time_range.start),end_us=_duration(event.time_range.end),thread_id=event.thread,
                input_shapes=_meta(event.input_shapes),cpu_total_us=_duration(event.cpu_time_total),
                self_cpu_us=_duration(event.self_cpu_time_total),
                timing_status='measured' if _duration(event.cpu_time_total) is not None and _duration(event.self_cpu_time_total) is not None else 'invalid_or_missing'))
        invocations=module_timings(inventory,invocations,events,p['measure_iterations'])
        prof.export_chrome_trace(str(trace_path))
        trace_bytes=Path(trace_path).read_bytes()
        if len(trace_bytes)>p['max_trace_bytes']:raise ValueError('trace evidence budget exceeded')
        from hyso.benchmark.result import _plain
        trace=_plain(json.loads(trace_bytes.decode('utf-8')))
        trace_events,kernels=normalize_trace(trace,p['max_events'])
        attribution=correlate(trace_events,invocations)
        cuda_metrics=_cuda_metrics(kernels,stream is not None)
        status,reason=_profile_status(events,cuda_metrics)
        data=_base('operator_profile','P14',dict(id='isolated_torch_profiler',version=1,
            scope='owned_worker_eval_inference_profile',configuration=p,recipe=r,
            instrumentation=dict(record_shapes=True,profile_memory=False,with_stack=False,
                with_flops=False,module_hooks='worker_only',dispatch_observer='TorchDispatchMode_version_verified',evidence_version=2)),
            dict(model=dict(structural_descriptor=descriptor,weights_sha256=weights,execution_mode='owned_worker_eval'),
                input=dict(shape=list(inputs.shape),dtype=str(inputs.dtype),content_sha256=digest,recipe='ones'),
                environment=dict(device=str(device),torch=str(torch.__version__),python=sys.version,
                    gpu=torch.cuda.get_device_name(device) if stream is not None else None,
                    cuda_build=torch.version.cuda,package_source=importlib.import_module('hyso.performance').__file__)))
        ledger=dict(measured_forwards=p['measure_iterations'],warmup_forwards=p['warmup_iterations'])
        data.update(requested=ledger,actual=ledger,measurements=dict(
            status=status,reason=reason,
            operators=_operator_metrics(events),cuda_kernels=kernels,
            cuda_attribution=attribution,
            cuda_activity=cuda_metrics,
            overhead=dict(collector_lifecycle_wall_ms=lifecycle_ms,instrumented_host_forward_windows_ms=host_windows,
                unit='ms',clock='worker_perf_counter_ns',
                perturbation_ms=None,perturbation_status='not_measured',
                reason='no_matched_uninstrumented_baseline; host_windows_are_not_GPU_kernel_time')),
            raw_evidence=dict(events=events,aten_calls=calls,module_inventory=inventory,invocations=invocations,
                output=boundary,supported_activities=supported,kernel_events=kernels,trace_events=trace_events,
                trace=dict(sha256=hashlib.sha256(trace_bytes).hexdigest(),size_bytes=len(trace_bytes),path=None,
                    retention='generated_not_retained',event_count=len(trace.get('traceEvents',[])),
                    timestamp_unit='us',base_time_nanoseconds=trace.get('baseTimeNanoseconds'),
                    display_time_unit=trace.get('displayTimeUnit'),
                    timestamp_domain='single_exported_chrome_trace; native_FunctionEvent_clock_is_separate'),
                safety={},worker={}))
        data['diagnostics']['limitations']=['Analytic dispatch counts and instrumented timings are not hardware FLOP counters or benchmark latency.',
            'Only observed recipe/seed/input/control flow; no full NVIDIA kernel coverage claim.',
            'Legacy event.kernels labels are excluded: they can be CPU operator names.',
            'Hooks/shape capture/dispatch tracing change allocation and scheduling; perturbation is not quantified.',
            'Owned inference worker only; parent model/hook/device/RNG/allocator handles never transferred.',
            'No verified kernel-to-module attribution; CUDA layer durations remain null.',
            'B4 native failure UNRESOLVED; risk OPEN.']
        body.update(status='completed',result=data)
    except Exception as error:
        kind=type(error).__name__
        body.update(status='state_mutation' if kind=='StateMutationError' else 'unsupported' if isinstance(error,NotImplementedError) else 'failed',
                    reason=kind+': '+str(error))
    finally:
        cleanup=dict(hooks_preserved=True,rng_restored=True,state_preserved=True)
        try:
            while active:
                _,ctx=active.pop();ctx.__exit__(None,None,None)
            for handle in reversed(handles):handle.remove()
            if saved_hooks is not None:cleanup['hooks_preserved']=_hooks_equal(saved_hooks)
            if snapshot is not None:
                try:snapshot.check(full=True)
                except Exception:
                    snapshot.restore_values();cleanup['state_preserved']=False
            if rng is not None:
                random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
                if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng,device)
        except Exception as error:
            cleanup['rng_restored']=False;body.update(status='failed',reason='cleanup: '+str(error))
        body['cleanup']=cleanup
        if body['status']=='completed':
            if not all(cleanup.values()):body.update(status='failed',reason='profile safety validation failed')
            else:body['result']['raw_evidence']['safety']=dict(cleanup,allocator_reset_not_called=True)
    Path(output_path).write_text(json.dumps(body,allow_nan=False),encoding='utf-8')
    if guard is not None:guard[0].CloseHandle(guard[1])


if __name__=='__main__':main()
