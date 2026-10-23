"""P15 stdlib bootstrap: own process, Job Object, tensors, stream and events."""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


def _ready(path, phase):
    tmp=Path(str(path)+'.tmp'); tmp.write_text(json.dumps(dict(pid=os.getpid(),phase=phase)),encoding='utf-8')
    os.replace(tmp,path)


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
        if math.prod(p['shape'])*sizes[p['dtype']]>p['input_limit_bytes']: raise ValueError('input byte budget exceeded')
        sys.path.insert(0,package_root)
        import torch
        from hyso.performance.inference import _base
        from hyso.performance.transfers import TransferProtocol, _transfer_metrics
        TransferProtocol(**p)
        if not torch.cuda.is_available(): raise NotImplementedError('CUDA unavailable; no CPU fallback')
        torch.set_num_threads(1); torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(p['cuda_memory_fraction'],0)
        stream=torch.cuda.Stream(device=0)
        dtype=getattr(torch,p['dtype'])
        # Distinct non-random pattern; no RNG use. Buffers are always owned.
        reference=(torch.arange(math.prod(p['shape']),dtype=torch.int64)%251).to(dtype).reshape(p['shape'])
        try:
            host=torch.empty(p['shape'],dtype=dtype,pin_memory=p['pinned'])
            returned=torch.empty_like(reference)
        except RuntimeError as error:
            if p['pinned']:
                body['status']='pin_allocation_failed'
                raise RuntimeError('pinned host allocation failed: '+str(error)) from error
            raise
        host.copy_(reference)
        gpu=torch.empty(p['shape'],dtype=dtype,device='cuda:0')
        tensor=dict(shape=list(host.shape),numel=host.numel(),element_size=host.element_size(),
                    dtype=p['dtype'],layout='contiguous_dense',transfer_bytes=host.numel()*host.element_size())
        assert host.is_pinned()==p['pinned'] and host.is_contiguous() and gpu.is_contiguous()
        rows=[]; _ready(ready,'runtime_prepared')
        with torch.inference_mode(),torch.cuda.stream(stream):
            for direction in p['directions']:
                for repeat in range(p['repeats']):
                    for index in range(p['warmup_iterations']+p['measure_iterations']):
                        if direction=='H2D': host.copy_(reference); gpu.fill_(255)
                        else: gpu.copy_(reference); host.fill_(255)
                        stream.synchronize()
                        first=torch.cuda.Event(enable_timing=True); last=torch.cuda.Event(enable_timing=True)
                        first.record(stream); start=time.perf_counter_ns()
                        if direction=='H2D': gpu.copy_(host,non_blocking=p['non_blocking'])
                        else: host.copy_(gpu,non_blocking=p['non_blocking'])
                        enqueued=time.perf_counter_ns(); last.record(stream)
                        last.synchronize(); terminal=time.perf_counter_ns()
                        elapsed=first.elapsed_time(last)
                        # No CPU read of D2H output until terminal completion.
                        if direction=='H2D': returned.copy_(gpu); stream.synchronize(); equal=torch.equal(returned,reference)
                        else: returned.copy_(host); equal=torch.equal(returned,reference)
                        if not equal: raise ValueError('round-trip equality failed')
                        if index>=p['warmup_iterations']:
                            rows.append(dict(direction=direction,repeat=repeat,iteration=index-p['warmup_iterations'],
                                transfer_bytes=tensor['transfer_bytes'],host_start_ns=start,host_end_ns=enqueued,
                                terminal_end_ns=terminal,host_enqueue_ms=(enqueued-start)/1e6,
                                device_stream_ms=elapsed,terminal_complete=bool(last.query()),equal=equal,
                                host_pinned=host.is_pinned(),non_blocking_requested=p['non_blocking']))
            stream.synchronize()
        count=len(rows)
        data=_base('transfer','P15',dict(id='owned_serial_copy',version=1,scope='fresh_worker_preallocated_buffers',configuration=p),
            dict(model={'kind':'not_applicable'},input=tensor,environment=dict(device='cuda:0',torch=str(torch.__version__),
                python=sys.version,gpu=torch.cuda.get_device_name(),cuda_build=torch.version.cuda,
                package_source=str(Path(package_root)/'hyso/performance/__init__.py'),worker_source=__file__)))
        data.update(requested=dict(measured_copies=count),actual=dict(completed_copies=count,
            warmup_copies=len(p['directions'])*p['repeats']*p['warmup_iterations'],validation_copies=count,pending_copies=0),
            measurements=_transfer_metrics(rows),raw_evidence=dict(copies=rows,tensor=tensor,worker={},
                stream_id=int(stream.cuda_stream),allocator_fraction=p['cuda_memory_fraction'],
                budget_scope='windows_job_private_commit',cpu_memory_limit_bytes=p['cpu_memory_limit_bytes'],
                validation='every_copy_poisoned_destination_and_completed_round_trip; reverse_validation_unmeasured',
                cleanup=dict(terminal_completed=True,process_owned_resources=True)))
        data['diagnostics']['limitations']=['CUDA event interval includes stream/host staging effects; not pure PCIe hardware peak.',
            'Host enqueue is copy_ call duration and may block; non_blocking is a request.',
            'No CUPTI copy activity/kernel correlation or verified compute overlap.',
            'B4 crash risk OPEN/UNRESOLVED; B6 CUDA kernel profiling UNSUPPORTED.']
        body.update(status='completed',result=data)
    except NotImplementedError as error: body.update(status='unsupported',reason=str(error))
    except Exception as error:
        torch=sys.modules.get('torch')
        if torch is not None and isinstance(error,torch.cuda.OutOfMemoryError): body['status']='cuda_oom'
        body['reason']=type(error).__name__+': '+str(error)
    Path(output).write_text(json.dumps(body,allow_nan=False),encoding='utf-8')
    if guard is not None: guard[0].CloseHandle(guard[1])


if __name__=='__main__': main()
