"""P15 stdlib bootstrap: own process, Job Object, tensors, stream and events."""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid


def _ready(path, phase):
    tmp=Path(str(path)+'.tmp'); tmp.write_text(json.dumps(dict(pid=os.getpid(),phase=phase)),encoding='utf-8')
    os.replace(tmp,path)


def _resources(torch):
    """Outside all timing windows. Memory occupancy never gates execution."""
    from hyso.performance.memory import _observe, _BackendUnavailable
    try: cpu=dict(status='measured',observation=_observe(),reason=None)
    except _BackendUnavailable as error: cpu=dict(status='unsupported',observation=None,reason=str(error))
    # CUDA errors propagate; an advisory policy must not disguise a runtime failure.
    free,total=torch.cuda.mem_get_info(0)
    return dict(cpu=cpu,cuda_allocated_bytes=torch.cuda.memory_allocated(0),
        cuda_reserved_bytes=torch.cuda.memory_reserved(0),cuda_free_bytes=free,cuda_total_bytes=total)


def _overlap_probe(torch,stream,host,gpu,reference,returned,p):
    """Common event epoch, distinct streams, independent preallocated workloads.

    These windows include idle/staging/launch gaps and cannot prove simultaneous
    copy-engine/SM activity. Preparation and validation are outside the probe.
    """
    compute=torch.cuda.Stream(device=0)
    with torch.cuda.stream(compute):
        a=torch.ones((256,256),device='cuda:0'); b=torch.ones_like(a); c=torch.empty_like(a)
        torch.mm(a,b,out=c)
    compute.synchronize()
    host.copy_(reference)
    with torch.cuda.stream(stream): gpu.fill_(255)
    stream.synchronize()
    epoch,cs,ce,ms,me=[torch.cuda.Event(enable_timing=True) for _ in range(5)]
    # CUDA Events initialize lazily; initialize every event outside its window.
    for event in (epoch,cs,ce): event.record(stream)
    for event in (ms,me): event.record(compute)
    ce.synchronize(); me.synchronize()
    epoch.record(stream); compute.wait_event(epoch)
    with torch.cuda.stream(compute):
        ms.record(compute)
        for _ in range(16): torch.mm(a,b,out=c)
        me.record(compute)
    with torch.cuda.stream(stream):
        cs.record(stream); gpu.copy_(host,non_blocking=p['non_blocking']); ce.record(stream)
    ce.synchronize(); me.synchronize()
    times=[float(epoch.elapsed_time(e)) for e in (cs,ce,ms,me)]
    with torch.cuda.stream(stream): returned.copy_(gpu)
    stream.synchronize()
    equal=bool(torch.equal(returned,reference)); compute_equal=bool(torch.equal(c,torch.full_like(c,256.)))
    if not equal or not compute_equal: raise ValueError('copy/compute probe validation failed')
    result=dict(direction='H2D',transfer_bytes=host.numel()*host.element_size(),
        copy_stream_id=int(stream.cuda_stream),compute_stream_id=int(compute.cuda_stream),
        epoch_stream_id=int(stream.cuda_stream),compute_waited_on_epoch=True,
        copy_start_ms=times[0],copy_end_ms=times[1],compute_start_ms=times[2],compute_end_ms=times[3],
        compute_shape=[256,256],compute_iterations=16,terminal_completed=bool(ce.query() and me.query()),
        round_trip_equal=equal,compute_equal=compute_equal,host_pinned=host.is_pinned(),
        non_blocking_requested=p['non_blocking'],events_preinitialized=True)
    if p['compare_execution']:
        result['comparison']=_compare_execution(torch,stream,compute,host,gpu,reference,returned,a,b,c,p)
    return result


def _compare_execution(torch,stream,compute,host,gpu,reference,returned,a,b,c,p):
    """Same owned copy and matmuls; serial stream vs independent streams.

    The wall window includes Python submission and BOTH terminal waits, never
    sums event durations. No validation, allocation or poisoning inside it.
    """
    copy_end=torch.cuda.Event(); compute_end=torch.cuda.Event()
    copy_end.record(stream); compute_end.record(compute)
    copy_end.synchronize(); compute_end.synchronize()
    expected=torch.full_like(c,256.)
    torch.cuda.synchronize(0)
    trace_token=str(uuid.uuid4()) if p['profile_capture'] else None

    def run(mode,label):
        target=stream if mode=='serial' else compute
        host.copy_(reference)
        with torch.cuda.stream(stream): gpu.fill_(255)
        with torch.cuda.stream(target): c.fill_(-1)
        stream.synchronize(); compute.synchronize()
        if trace_token: torch.cuda.nvtx.range_push(f'hyso.p15/{trace_token}/{label}/{mode}')
        start=time.perf_counter_ns()
        with torch.cuda.stream(stream):
            gpu.copy_(host,non_blocking=True); copy_end.record(stream)
        with torch.cuda.stream(target):
            for _ in range(16): torch.mm(a,b,out=c)
            compute_end.record(target)
        copy_end.synchronize(); compute_end.synchronize()
        end=time.perf_counter_ns()
        if trace_token: torch.cuda.nvtx.range_pop()
        complete=bool(copy_end.query() and compute_end.query())
        with torch.cuda.stream(stream): returned.copy_(gpu)
        stream.synchronize()
        equal=bool(torch.equal(returned,reference)); compute_equal=bool(torch.equal(c,expected))
        if not (complete and equal and compute_equal): raise ValueError('execution comparison validation failed')
        return dict(host_start_ns=start,terminal_end_ns=end,completion_wall_ms=(end-start)/1e6,
            terminal_completed=complete,round_trip_equal=equal,compute_equal=compute_equal,
            copy_stream_id=int(stream.cuda_stream),compute_stream_id=int(target.cuda_stream),
            events_preinitialized=True,host_pinned=host.is_pinned(),non_blocking_requested=True,
            transfer_bytes=host.numel()*host.element_size(),compute_shape=[256,256],compute_iterations=16)

    pairs=[]
    with torch.inference_mode():
        for i in range(p['warmup_iterations']):
            for mode in (('serial','concurrent') if i%2==0 else ('concurrent','serial')): run(mode,f'warmup/{i}')
        for i in range(p['repeats']*p['measure_iterations']):
            order=['serial','concurrent'] if i%2==0 else ['concurrent','serial']
            pair=dict(index=i,order=order)
            for mode in order: pair[mode]=run(mode,f'measure/{i}')
            pairs.append(pair)
    result=dict(warmup_pairs=p['warmup_iterations'],pairs=pairs)
    if trace_token: result['trace_token']=trace_token
    return result


def main():
    request,output,ready,package_root=sys.argv[1:]
    _ready(ready,'bootstrap'); guard=None
    body=dict(worker_pid=os.getpid(),status='failed',reason=None,result=None)
    try:
        p=json.loads(Path(request).read_text(encoding='utf-8'))
        spec=importlib.util.spec_from_file_location('hyso_transfer_guard',Path(__file__).with_name('_batch_worker.py'))
        bootstrap=importlib.util.module_from_spec(spec); spec.loader.exec_module(bootstrap)
        guard=bootstrap._memory_guard(p['cpu_memory_limit_bytes'])
        sizes={'float16':2,'float32':4,'float64':8,'int32':4,'int64':8,'uint8':1}
        if p['layout']!='contiguous_dense' or p['device'] not in ('cuda','cuda:0') or p['dtype'] not in sizes:
            raise NotImplementedError('only contiguous dense supported dtype and single cuda:0; sparse/quantized/strided/cross-device unsupported')
        if p['input_limit_bytes'] is not None and math.prod(p['shape'])*sizes[p['dtype']]>p['input_limit_bytes']:
            raise ValueError('input byte budget exceeded')
        sys.path.insert(0,package_root)
        import torch
        from hyso.performance.inference import _base
        from hyso.performance.transfers import TransferProtocol, _transfer_metrics
        TransferProtocol(**p)
        if not torch.cuda.is_available(): raise NotImplementedError('CUDA unavailable; no CPU fallback')
        torch.set_num_threads(1); torch.cuda.set_device(0)
        if p['cuda_memory_fraction'] is not None: torch.cuda.set_per_process_memory_fraction(p['cuda_memory_fraction'],0)
        stream=torch.cuda.Stream(device=0)
        resources_before=_resources(torch)
        dtype=getattr(torch,p['dtype'])
        # Distinct non-random pattern; no RNG use. Buffers are always owned.
        reference=(torch.arange(math.prod(p['shape']),dtype=torch.int64)%251).to(dtype).reshape(p['shape'])
        try:
            host=torch.empty(p['shape'],dtype=dtype,pin_memory=p['pinned'])
            returned=torch.empty_like(reference)
        except RuntimeError:
            if p['pinned']:
                body['status']='pin_allocation_failed'
            raise
        host.copy_(reference)
        gpu=torch.empty(p['shape'],dtype=dtype,device='cuda:0')
        tensor=dict(shape=list(host.shape),numel=host.numel(),element_size=host.element_size(),
                    dtype=p['dtype'],layout='contiguous_dense',transfer_bytes=host.numel()*host.element_size())
        assert host.is_pinned()==p['pinned'] and host.is_contiguous() and gpu.is_contiguous()
        rows=[]; _ready(ready,'runtime_prepared')
        if p['profile_capture']:
            error=torch.cuda.cudart().cudaProfilerStart()
            if int(error)!=0: raise RuntimeError(f'cudaProfilerStart failed: {error}')
        with torch.inference_mode(),torch.cuda.stream(stream):
            for direction in p['directions']:
                for repeat in range(p['repeats']):
                    for index in range(p['warmup_iterations']+p['measure_iterations']):
                        if direction=='H2D': host.copy_(reference); gpu.fill_(255)
                        else: gpu.copy_(reference); host.fill_(255)
                        stream.synchronize()
                        first=torch.cuda.Event(enable_timing=True); last=torch.cuda.Event(enable_timing=True)
                        first.record(stream); last.record(stream); last.synchronize()
                        first.record(stream); start=time.perf_counter_ns()
                        if direction=='H2D': gpu.copy_(host,non_blocking=p['non_blocking'])
                        else: host.copy_(gpu,non_blocking=p['non_blocking'])
                        enqueued=time.perf_counter_ns(); last.record(stream)
                        start_complete=bool(first.query()); end_complete=bool(last.query())
                        observed=time.perf_counter_ns()
                        last.synchronize(); terminal=time.perf_counter_ns()
                        elapsed=first.elapsed_time(last)
                        # No CPU read of D2H output until terminal completion.
                        if direction=='H2D': returned.copy_(gpu); stream.synchronize(); equal=torch.equal(returned,reference)
                        else: returned.copy_(host); equal=torch.equal(returned,reference)
                        if not equal: raise ValueError('round-trip equality failed')
                        if index>=p['warmup_iterations']:
                            event_index=2*len(rows)
                            rows.append(dict(direction=direction,repeat=repeat,iteration=index-p['warmup_iterations'],
                                transfer_bytes=tensor['transfer_bytes'],host_start_ns=start,host_end_ns=enqueued,
                                terminal_end_ns=terminal,host_enqueue_ms=(enqueued-start)/1e6,
                                device_stream_ms=elapsed,terminal_complete=bool(last.query()),equal=equal,
                                host_pinned=host.is_pinned(),non_blocking_requested=p['non_blocking'],
                                start_stream_id=int(stream.cuda_stream),end_stream_id=int(stream.cuda_stream),
                                start_event_index=event_index,end_event_index=event_index+1,
                                start_complete_after_enqueue=start_complete,end_complete_after_enqueue=end_complete,
                                host_observation_ns=observed,events_preinitialized=True))
            stream.synchronize()
        probe=_overlap_probe(torch,stream,host,gpu,reference,returned,p) if p['probe_overlap'] else None
        if p['profile_capture']:
            error=torch.cuda.cudart().cudaProfilerStop()
            if int(error)!=0: raise RuntimeError(f'cudaProfilerStop failed: {error}')
        resources_after=_resources(torch)
        count=len(rows)
        evidence_version=3 if p['compare_execution'] else 2
        data=_base('transfer','P15',dict(id='owned_serial_copy',version=1,evidence_version=evidence_version,
            scope='fresh_worker_preallocated_buffers',configuration=p),
            dict(model={'kind':'not_applicable'},input=tensor,environment=dict(device='cuda:0',torch=str(torch.__version__),
                python=sys.version,gpu=torch.cuda.get_device_name(),cuda_build=torch.version.cuda,
                package_source=str(Path(package_root)/'hyso/performance/__init__.py'),worker_source=__file__)))
        data.update(requested=dict(measured_copies=count),actual=dict(completed_copies=count,
            warmup_copies=len(p['directions'])*p['repeats']*p['warmup_iterations'],validation_copies=count,pending_copies=0),
            measurements=_transfer_metrics(rows,evidence_version=evidence_version,overlap_probe=probe),
            raw_evidence=dict(copies=rows,tensor=tensor,worker={},overlap_probe=probe,
                resources=dict(policy='advisory_only; occupancy_is_not_an_execution_gate',before=resources_before,after=resources_after),
                stream_id=int(stream.cuda_stream),allocator_fraction=p['cuda_memory_fraction'],
                budget_scope='optional_windows_job_private_commit',cpu_memory_limit_bytes=p['cpu_memory_limit_bytes'],
                validation='every_copy_poisoned_destination_and_completed_round_trip; reverse_validation_unmeasured',
                cleanup=dict(terminal_completed=True,process_owned_resources=True)))
        data['diagnostics']['limitations']=['CUDA event interval includes stream/host staging effects; not pure PCIe hardware peak.',
            'Host enqueue is copy_ call duration and may block; non_blocking is a request.',
            'Host completion wall includes event submission, observation queries and terminal wait; it is not enqueue or device time.',
            'Preparation, allocation, poisoning, warmup and round-trip validation are excluded from measured copy intervals.',
            'Optional copy/compute probe is separate from copy samples and bandwidth accounting.',
            'A pending start event proves only host return before that stream position, not intrinsic DMA asynchrony or overlap.',
            'No direct copy-engine/SM activities; hardware overlap remains unsupported.']
        body.update(status='completed',result=data)
    except NotImplementedError as error: body.update(status='unsupported',reason=str(error))
    except Exception as error:
        torch=sys.modules.get('torch')
        if torch is not None and isinstance(error,torch.cuda.OutOfMemoryError): body['status']='cuda_oom'
        body['reason']=type(error).__name__+': '+str(error)
    Path(output).write_text(json.dumps(body,allow_nan=False),encoding='utf-8')
    if guard is not None: guard[0].CloseHandle(guard[1])


if __name__=='__main__': main()
