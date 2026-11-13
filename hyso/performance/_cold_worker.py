"""Minimal stdlib bootstrap; heavy imports are timed only inside the fresh worker."""
import importlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    request_path,output_path,package_root=sys.argv[1:]
    entry=time.perf_counter_ns()
    print('@@HYSO_READY '+json.dumps(dict(pid=os.getpid(),entry_ns=entry)),flush=True)
    request=json.loads(Path(request_path).read_text(encoding='utf-8'))
    recipe=request['recipe']; protocol=request['protocol']
    phases={name:dict(start_ns=None,end_ns=None,elapsed_ms=None,unit='ms',clock='worker_perf_counter_ns',
        status='not_observed',reason='prior_failure',device='cpu' if name in ('module_import','model_construction','artifact_load') else recipe['device'],
        backend='pytorch_eager',included_operations=[]) for name in
        ('module_import','model_construction','artifact_load','runtime_prepare','first_inference','warmup','warm_reference')}
    result=dict(worker_pid=os.getpid(),status='failed',phases=phases,first_output=None,warm_samples_ms=[],artifact=None,
                provenance=dict(device=recipe['device'],worker_source=__file__,entry_ns=entry),error=None)
    current=None
    operations=dict(module_import=['torch_hyso_and_factory_import'],model_construction=['CPU_seed_model_and_input_construction'],
        artifact_load=['file_read_stat_streamed_sha256','weights_only_cpu_deserialization','strict_state_dict_load'],
        runtime_prepare=['context_initialization_if_CUDA','model_input_dtype_device_movement','profile_validation','device_drain_if_CUDA'],
        first_inference=['inference_autocast_context','forward','terminal_completion_if_CUDA'],
        warmup=['warmup_forward_completion_and_output_validation'],warm_reference=['repeated_completion_wall_samples','output_validation'])
    def begin(name):
        nonlocal current
        current=name
        phases[name].update(start_ns=time.perf_counter_ns(),status='failed',reason='incomplete',included_operations=operations[name])
    def finish():
        end=time.perf_counter_ns(); phase=phases[current]
        phase.update(end_ns=end,elapsed_ms=(end-phase['start_ns'])/1e6,status='measured',reason=None)
    try:
        begin('module_import')
        if recipe['factory_search_path']: sys.path.insert(0,recipe['factory_search_path'])
        sys.path.insert(0,package_root)
        import torch
        from hyso.benchmark.runner import _profile,_output
        from hyso.performance.memory import _file_info
        module,qualified=recipe['factory'].split(':')
        factory=importlib.import_module(module)
        for part in qualified.split('.'): factory=getattr(factory,part)
        finish()
        result['provenance']['cuda_initialized_after_import']=torch.cuda.is_initialized()
        if torch.cuda.is_initialized(): raise ValueError('factory module initialized CUDA before preparation')
        begin('model_construction')
        torch.set_num_threads(1); torch.manual_seed(recipe['seed'])
        model=factory(**recipe['model_kwargs'])
        if not isinstance(model,torch.nn.Module): raise TypeError('factory must return nn.Module')
        if torch.cuda.is_initialized() or any(t.device.type!='cpu' for t in list(model.parameters())+list(model.buffers())):
            raise ValueError('factory must construct CPU model without CUDA initialization')
        dtype=torch.float64 if recipe['precision']=='fp64' else torch.float32
        inputs=torch.ones(recipe['input_shape'],dtype=dtype)
        finish()
        result['provenance']['cuda_initialized_after_construction']=torch.cuda.is_initialized()
        if recipe['artifact_path']:
            begin('artifact_load')
            info=_file_info(recipe['artifact_path'])
            if info['sha256']!=recipe['artifact_sha256']: raise ValueError('artifact SHA256 mismatch')
            state=torch.load(recipe['artifact_path'],map_location='cpu',weights_only=True)
            model.load_state_dict(state,strict=True); result['artifact']=info
            del state
            finish()
        else: phases['artifact_load'].update(status='disabled',reason='no_artifact')
        begin('runtime_prepare')
        if recipe['device']!='cpu':
            if not torch.cuda.is_available():
                result['status']='unsupported'; raise ValueError('CUDA unavailable; no CPU fallback')
            requested_index=None if recipe['device']=='cuda' else int(recipe['device'].split(':')[1])
            if requested_index is not None and requested_index>=torch.cuda.device_count():
                result['status']='unsupported'; raise ValueError('CUDA device index unavailable')
            torch.cuda.init()
            device=torch.device('cuda',torch.cuda.current_device() if requested_index is None else requested_index)
            torch.cuda.set_device(device)
            fraction=request.get('cuda_memory_fraction')
            if fraction is not None:
                if not 0 < fraction <= .15: raise ValueError('invalid forensic allocator budget')
                torch.cuda.set_per_process_memory_fraction(fraction,device)
        else: device=torch.device('cpu')
        model=model.to(device=device,dtype=dtype).eval(); inputs=inputs.to(device)
        _profile(model,dtype,device)
        if device.type=='cuda': torch.cuda.synchronize(device)
        result['provenance'].update(python=sys.version,torch=str(torch.__version__),device=str(device),
            cuda_build=torch.version.cuda,gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            stream_id=int(torch.cuda.current_stream(device).cuda_stream) if device.type=='cuda' else None,
            package_source=importlib.import_module('hyso.performance').__file__,
            cuda_initialized=torch.cuda.is_initialized(),load_policy='weights_only_cpu_verified_sha',
            cuda_allocator_fraction=request.get('cuda_memory_fraction'),
            input_shape=list(inputs.shape),input_dtype=str(inputs.dtype),model_dtype=str(dtype))
        finish()
        def inference():
            with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=recipe['precision']=='amp_fp16'):
                output=model(inputs)
                if device.type=='cuda':
                    event=torch.cuda.Event(); event.record(torch.cuda.current_stream(device)); event.synchronize()
                return output
        begin('first_inference'); output=inference(); finish()
        result['first_output']=_output(output,inputs.shape[0],device=device); del output
        begin('warmup')
        for _ in range(protocol['warmup_iterations']):
            output=inference(); _output(output,inputs.shape[0],device=device); del output
        finish()
        begin('warm_reference')
        for _ in range(protocol['warm_samples']):
            start=time.perf_counter_ns(); output=inference(); end=time.perf_counter_ns()
            result['warm_samples_ms'].append((end-start)/1e6)
            _output(output,inputs.shape[0],device=device); del output
        finish()
        result['provenance']['forward_count']=1+protocol['warmup_iterations']+protocol['warm_samples']
        del model,inputs
        result['status']='measured'
    except BaseException as error:
        phase=phases.get(current)
        if phase is not None and phase['status']=='failed' and phase['start_ns'] is not None:
            end=time.perf_counter_ns(); phase.update(end_ns=end,elapsed_ms=(end-phase['start_ns'])/1e6,reason=str(error))
        result['error']=type(error).__name__+': '+str(error)
    Path(output_path).write_text(json.dumps(result,allow_nan=False),encoding='utf-8')
    return 0 if result['status']=='measured' else 1


if __name__=='__main__': raise SystemExit(main())
