"""Deployment measurements/comparisons using the existing benchmark statistics."""
from __future__ import annotations
from typing import Any
import math
import platform
import time
from .contracts import ExportError
from .runtime import Runtime


def benchmark(runtime: Runtime, inputs: Any, *, warmup: int=2, iterations: int=10) -> dict[str,Any]:
    """Bounded CPU resident inference. Load and validation are outside samples.

    Disk and process RSS observations have distinct scopes. GPU timing, transfers,
    peaks, and engine build time are unavailable, never inferred from host time.
    """
    from ..benchmark.result import _statistics,_throughput
    from ._model import tensors,outputs
    from ..analysis._codec import digest
    if type(runtime) is not Runtime:raise TypeError('Runtime required')
    if type(warmup) is not int or type(iterations) is not int or not 0<=warmup<=100 or not 1<=iterations<=1000:
        raise ExportError('RESOURCE_BUDGET','warmup 0..100; iterations 1..1000 required')
    values=tensors(inputs)
    if not values or any(v.ndim<1 or v.shape[0]!=values[0].shape[0] for v in values):raise ExportError('INPUT_SPEC','common explicit batch dimension required')
    batch=int(values[0].shape[0]);signature=runtime.input_identity(inputs)
    if runtime.device=='cuda':
        report,controller=runtime.measure_native(inputs,warmup=warmup,iterations=iterations)
        measurement=report['measurement'];samples=measurement['samples_ms']
        stats=_statistics(samples);ordered=sorted(samples);p=(len(ordered)-1)*.99;lo=int(p);hi=min(lo+1,len(ordered)-1)
        stats['p99_ms']=ordered[lo]+(ordered[hi]-ordered[lo])*(p-lo)
        conditions=dict(device='cuda',platform=report['environment']['platform'],machine=platform.machine(),
            processor=report['environment']['name'],threads=None,batch_size=batch,input_signature=signature,
            boundary=measurement['boundary'],warmup=warmup,iterations=iterations,timer='perf_counter_ns',
            synchronization='CUDA stream synchronize after each enqueue',thread_policy='owned native serial',
            gpu_uuid=report['environment']['gpu_uuid'],compute_capability=report['environment']['compute_capability'])
        conditions['driver_binaries']={p:r['sha256'] for p,r in report['loaded_binaries'].items() if __import__('pathlib').Path(p).name.lower().startswith('nvcuda')}
        return dict(format='hyso.export.benchmark',schema_version=1,status='PASS',runtime=dict(runtime.info),
            conditions=conditions,conditions_sha256=digest(conditions),latency=stats,samples_ms=samples,
            throughput_per_second=_throughput(samples,batch),runtime_loading_ms=report['runtime_loading_ms'],
            artifact_bytes=runtime.info['artifact_bytes'],cpu_ram=dict(value=None,status='unavailable'),
            gpu_vram=dict(value=None,status='unavailable',reason=measurement['gpu_vram_reason']),
            host_to_device_ms=measurement['host_to_device_ms'],device_to_host_ms=measurement['device_to_host_ms'],
            end_to_end_ms=None,engine_build_ms=runtime.source.to_dict().get('runtime',{}).get('engine_build_ms'),
            native_receipt=runtime.receipts[-1],limitations=['Synchronized host wall time is not GPU kernel-only timing'])
    samples=[];rss=[]
    try:
        import psutil
        process=psutil.Process()
    except ImportError:process=None
    with runtime.session(inputs) as (call,load_ms):
        for _ in range(warmup):outputs(call())
        for _ in range(iterations):
            start=time.perf_counter_ns();actual=call();elapsed=(time.perf_counter_ns()-start)/1e6
            outputs(actual)
            if not math.isfinite(elapsed) or elapsed<=0:raise ExportError('TIMING','positive finite clock readings required')
            samples.append(elapsed)
            if process:
                try:rss.append(process.memory_info().rss)
                except (OSError,psutil.Error):pass
    stats=_statistics(samples)
    ordered=sorted(samples);p=(len(ordered)-1)*.99;lo=int(p);hi=min(lo+1,len(ordered)-1)
    stats['p99_ms']=ordered[lo]+(ordered[hi]-ordered[lo])*(p-lo)
    import torch
    conditions=dict(device='cpu',platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),
                    threads=torch.get_num_threads(),batch_size=batch,input_signature=signature,
                    boundary='resident_runtime_call_including_python_dispatch_and_output_wrapping',
                    warmup=warmup,iterations=iterations,timer='perf_counter_ns',synchronization='CPU synchronous',
                    thread_policy='ORT intra/inter=1; Torch recorded threads')
    return dict(format='hyso.export.benchmark',schema_version=1,status='PASS',runtime=dict(runtime.info),
                conditions=conditions,conditions_sha256=digest(conditions),latency=stats,samples_ms=samples,
                throughput_per_second=_throughput(samples,batch),runtime_loading_ms=load_ms,
                artifact_bytes=runtime.info['artifact_bytes'],cpu_ram=dict(value=max(rss) if rss else None,
                    status='measured' if rss else 'unavailable',scope='sampled whole-process RSS; not peak or artifact incremental memory'),
                gpu_vram=dict(value=None,status='unavailable',reason='CPU measurement'),
                host_to_device_ms=None,device_to_host_ms=None,end_to_end_ms=None,engine_build_ms=None,
                unavailable_reason='No owned GPU/transfer/build measurement was requested',
                limitations=['Small serial samples are not sustained service throughput','Validation outside timing can influence caches'])


def compare(baseline: dict, candidate: dict, *, validation: dict | None=None) -> dict[str,Any]:
    """Compare saved measurements only when workload, scope and hardware match."""
    from ..analysis._codec import digest
    for row in (baseline,candidate):
        if row.get('format')!='hyso.export.benchmark' or row.get('status')!='PASS':raise ExportError('COMPARISON_SPEC','completed deployment benchmarks required')
        if digest(row['conditions'])!=row.get('conditions_sha256'):raise ExportError('RESULT_INTEGRITY','measurement conditions changed')
        if not row.get('samples_ms') or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in row['samples_ms']):raise ExportError('COMPARISON_SPEC','positive real samples required')
        from ..benchmark.result import _statistics,_throughput
        measured=_statistics(row['samples_ms'])
        if any(row['latency'].get(k)!=v for k,v in measured.items()) or row['throughput_per_second']!=_throughput(row['samples_ms'],row['conditions']['batch_size']):
            raise ExportError('RESULT_INTEGRITY','summary differs from samples')
    fields=('device','platform','machine','processor','threads','batch_size','input_signature','boundary','warmup','iterations','timer','synchronization','thread_policy','gpu_uuid','compute_capability','driver_binaries')
    reasons=[key for key in fields if baseline['conditions'].get(key)!=candidate['conditions'].get(key)]
    source_a=baseline['runtime'].get('source_model');source_b=candidate['runtime'].get('source_model')
    if not source_a or source_a!=source_b:reasons.append('source_model_identity')
    valid=not reasons
    def delta(a,b):return b-a if a is not None and b is not None else None
    return dict(format='hyso.export.comparison',schema_version=1,status='PASS' if valid else 'BLOCKED',
                comparable=valid,incompatibilities=reasons,baseline=baseline,candidate=candidate,
                latency_delta_ms=delta(baseline['latency']['mean_ms'],candidate['latency']['mean_ms']) if valid else None,
                throughput_delta=delta(baseline['throughput_per_second'],candidate['throughput_per_second']) if valid else None,
                speedup=baseline['latency']['mean_ms']/candidate['latency']['mean_ms'] if valid else None,
                artifact_bytes_delta=delta(baseline['artifact_bytes'],candidate['artifact_bytes']),
                ram_delta=None,vram_delta=None,memory_reason='whole-process RSS is not comparable incremental runtime memory',
                numerical=validation.get('numerical') if validation else dict(status='NOT_TESTED'),
                quality=validation.get('quality') if validation else dict(status='NOT_TESTED'))
