"""Explicit finite point sweeps; isolated copies, existing latency and rate scopes."""
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import copy
import hashlib
import math
import platform
import random
import threading
import time
import uuid

import numpy as np
import torch

from hyso.benchmark import BenchmarkRunner, BenchmarkResult
from hyso.benchmark.runner import (_profile, _Snapshot, _descriptor, _shape, _bits,
                                   _output, StateMutationError, _RUN_LOCK)
from .inference import (_base, _window, _context_check, ThroughputProtocol,
                        measure_throughput)
from .result import PrecisionResult, SweepResult, _integer, _positive

_SWEEP_LOCK = threading.Lock()
_PRECISIONS = ('fp32', 'fp64', 'amp_fp16', 'bf16', 'int8_dynamic')


class _Unsupported(ValueError):
    pass


@dataclass(frozen=True)
class SweepProtocol:
    device: str = 'cpu'
    precision: str = 'fp32'
    warmup_iterations: int = 2
    measure_iterations: int = 10
    repeats: int = 1
    sustained_batches: int | None = None
    queue_depth: int = 4
    continue_on_error: bool = True
    measure_cuda_memory: bool = False

    def __post_init__(self):
        ThroughputProtocol(device=self.device, precision='fp32', batch_count=1,
                           queue_depth=self.queue_depth)
        if self.precision not in _PRECISIONS: raise ValueError('unknown precision')
        _integer(self.warmup_iterations); _integer(self.measure_iterations, 1); _integer(self.repeats, 1)
        if self.sustained_batches is not None: _integer(self.sustained_batches, 1)
        if type(self.continue_on_error) is not bool or type(self.measure_cuda_memory) is not bool:
            raise TypeError('expected bool policy')


def _name(value):
    return (getattr(value, '__module__', None) or '<unknown>')+'.'+(getattr(value, '__qualname__', None) or type(value).__name__)


def _prepare(model, precision, device):
    """Only owned copies are converted. No artifact deserialize or calibration."""
    if device.type == 'cpu' and precision == 'amp_fp16': raise _Unsupported('AMP FP16 requires CUDA')
    if device.type == 'cuda' and precision in ('fp64', 'int8_dynamic'):
        raise _Unsupported('selected precision has no supported CUDA recipe')
    if precision == 'bf16' and device.type == 'cuda':
        if torch.cuda.get_device_capability(device)[0] < 8 or not torch.cuda.is_bf16_supported():
            raise _Unsupported('native CUDA BF16 capability unavailable')
    proof = dict(backend='pytorch_eager', quantization_recipe=None, calibration=None,
                 execution_proof=None, artifact=None, load_cost_ms=None,
                 load_cost_status='not_measured', accuracy_status='not_measured_P28')
    start = time.perf_counter_ns()
    owned = copy.deepcopy(model)
    # Reject malicious/custom deepcopy that aliases caller tensors or modules.
    if set(map(id, owned.modules())) & set(map(id, model.modules())):
        raise ValueError('model copy aliases caller modules')
    source_tensors=list(model.parameters())+list(model.buffers())
    copied_tensors=list(owned.parameters())+list(owned.buffers())
    if any(a.untyped_storage().data_ptr()==b.untyped_storage().data_ptr() and a.numel() and b.numel()
           for a in source_tensors for b in copied_tensors): raise ValueError('model copy aliases caller storage')
    owned.eval()
    if precision == 'int8_dynamic':
        allowed=(torch.nn.Sequential, torch.nn.Flatten, torch.nn.Linear, torch.nn.ReLU, torch.nn.Identity)
        if type(owned) is not torch.nn.Sequential or any(type(m) not in allowed for m in owned.modules()):
            raise _Unsupported('INT8 recipe requires Sequential of Flatten/Linear/ReLU/Identity')
        engine=torch.backends.quantized.engine
        if engine not in ('x86', 'fbgemm', 'qnnpack', 'onednn') or engine not in torch.backends.quantized.supported_engines:
            raise _Unsupported('active CPU quantization engine unavailable')
        owned=torch.ao.quantization.quantize_dynamic(owned, {torch.nn.Linear}, dtype=torch.qint8, inplace=False)
        layers=[m for m in owned.modules() if isinstance(m, torch.ao.nn.quantized.dynamic.Linear)]
        if not layers or any(m.weight().dtype != torch.qint8 for m in layers):
            raise _Unsupported('no actual qint8 Linear conversion')
        proof.update(backend='pytorch_quantized_'+engine, quantization_recipe='dynamic_qint8_Linear',
            calibration=dict(status='not_applicable',reason='dynamic_activation_quantization_no_calibration_dataset'),
            execution_proof=dict(quantized_linear_count=len(layers), weight_dtype='torch.qint8', operators=[]))
    else:
        owned.to(device=device, dtype=torch.float64 if precision == 'fp64' else torch.float32)
    if device.type == 'cuda': torch.cuda.synchronize(device)
    proof['copy_conversion_ms']=(time.perf_counter_ns()-start)/1e6
    proof['conversion_scope']='owned_copy_eval_dtype_device_or_dynamic_quantization_and_completion'
    return owned, proof


def _special(model, inputs, precision, plan, device, stream, proof):
    """BF16/INT8 only. Instrumented proof preflight is outside timing windows."""
    if not _RUN_LOCK.acquire(blocking=False): raise RuntimeError('concurrent benchmark unsupported')
    snapshot=None
    try:
        if precision == 'bf16':
            modules,entries=_profile(model,torch.float32,device)
            snapshot=_Snapshot(modules,entries,inputs)
        else:
            snapshot=_Snapshot([],[],inputs)
        with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=precision=='bf16'):
            if precision == 'int8_dynamic':
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                    output=model(inputs)
                operators=sorted({event.key for event in trace.key_averages() if event.key=='quantized::linear_dynamic'})
                if not operators: raise _Unsupported('quantized::linear_dynamic execution not observed')
                proof['execution_proof']['operators']=operators
                del trace
            else: output=model(inputs)
            expected=_output(output,inputs.shape[0],device=device)
            if precision=='bf16' and output.dtype!=torch.bfloat16:
                raise _Unsupported('BF16 output not observed; no FP32 fallback claim')
            del output
            samples=[]; windows=[]
            start_event=end_event=None
            if stream is not None:
                start_event=torch.cuda.Event(enable_timing=True); end_event=torch.cuda.Event(enable_timing=True)
                start_event.record(stream); end_event.record(stream); end_event.synchronize()
            def forward():
                _context_check(device,stream)
                value=model(inputs)
                _context_check(device,stream)
                return value
            for _ in range(plan.repeats):
                for __ in range(plan.warmup_iterations):
                    output=forward(); _output(output,inputs.shape[0],expected,device); del output
                if stream is not None: torch.cuda.synchronize(device)
                row=[]
                for __ in range(plan.measure_iterations):
                    if stream is not None: start_event.record(stream)
                    begin=time.perf_counter_ns(); output=forward()
                    if stream is not None: end_event.record(stream); end_event.synchronize()
                    end=time.perf_counter_ns()
                    elapsed=start_event.elapsed_time(end_event) if stream is not None else (end-begin)/1e6
                    _positive(elapsed); row.append(elapsed)
                    _output(output,inputs.shape[0],expected,device); del output
                    snapshot.check(full=True)
                samples.append(row)
            if plan.sustained_batches is not None:
                rate_plan=ThroughputProtocol(device=str(device), batch_count=plan.sustained_batches,
                    warmup_iterations=plan.warmup_iterations,repeats=plan.repeats,queue_depth=plan.queue_depth)
                for repeat in range(plan.repeats):
                    for __ in range(plan.warmup_iterations):
                        output=forward(); _output(output,inputs.shape[0],expected,device); del output
                    if stream is not None: torch.cuda.synchronize(device)
                    row=_window(model,inputs,rate_plan,device,stream,expected); row['repeat_index']=repeat
                    windows.append(row); snapshot.check(full=True)
        return dict(result_id=str(uuid.uuid4()),latency_samples_ms=samples,output=expected,windows=windows,
                    warmup_iterations=plan.warmup_iterations,preflight_forwards=1,
                    forward_count=1+plan.repeats*(plan.warmup_iterations+plan.measure_iterations)+(
                        plan.repeats*(plan.warmup_iterations+plan.sustained_batches) if plan.sustained_batches is not None else 0),
                    timing_scope='serial_current_stream_forward' if stream is not None else 'synchronous_host_forward',
                    synchronization='repeat_device_drain_and_end_event_sync' if stream is not None else 'synchronous_host',
                    clock='cuda_event_elapsed' if stream is not None else 'perf_counter_ns',
                    memory=dict(status='unsupported' if plan.measure_cuda_memory else 'disabled',value=None,
                                reason='special_precision_allocator_probe_not_implemented' if plan.measure_cuda_memory else 'not_requested'))
    finally:
        try:
            if stream is not None: torch.cuda.synchronize(device)
        finally: _RUN_LOCK.release()


def _point(model, factory, shape, precision, plan, device, stream):
    owned=None; inputs=None
    try:
        owned,proof=_prepare(model,precision,device)
        # Factory receives exact NCHW shape, dtype and device. No resize/cast fallback.
        dtype=torch.float64 if precision=='fp64' else torch.float32
        produced=factory(tuple(shape),dtype,device)
        if (type(produced) is not torch.Tensor or produced.device!=device or produced.dtype!=dtype or
                list(produced.shape)!=list(shape) or produced.layout!=torch.strided or not produced.is_contiguous()):
            raise ValueError('input factory must return exact contiguous NCHW shape/dtype/device')
        with torch.inference_mode(False),torch.no_grad(): inputs=produced.detach().clone()
        del produced
        digest=hashlib.sha256(_bits(inputs).numpy().tobytes()).hexdigest()
        if precision in ('fp32','fp64','amp_fp16'):
            benchmark=BenchmarkRunner(owned,device=str(device),precision=precision,example_input=inputs,
                warmup_iterations=plan.warmup_iterations,measure_iterations=plan.measure_iterations,
                repeats=plan.repeats,measure_cuda_memory=plan.measure_cuda_memory).run().to_dict()
            throughput=None
            if plan.sustained_batches is not None:
                throughput=measure_throughput(owned,inputs,protocol=ThroughputProtocol(device=str(device),precision=precision,
                    batch_count=plan.sustained_batches,warmup_iterations=plan.warmup_iterations,
                    repeats=plan.repeats,queue_depth=plan.queue_depth)).to_dict()
            raw=dict(benchmark=benchmark,throughput=throughput,special=None)
        else:
            special=_special(owned,inputs,precision,plan,device,stream,proof)
            raw=dict(benchmark=None,throughput=None,special=special)
        output_boundary=benchmark['model']['output'] if raw['benchmark'] is not None else special['output']
        source_modules,source_entries=_profile(model,torch.float32,torch.device('cpu'))
        source_descriptor,source_hash=_descriptor(source_modules,source_entries)
        data=_base('precision','P08',dict(id='isolated_precision_point',version=1,scope='resident_forward',
            configuration=asdict(plan),precision=precision),dict(model=dict(source='isolated_master_copy'),
                input=dict(shape=list(shape),dtype=str(dtype),device=str(device),content_sha256=digest),
                environment=dict(device=str(device),torch=str(torch.__version__),python=platform.python_version(),
                    cpu_capability=torch.backends.cpu.get_cpu_capability(),cuda_build=torch.version.cuda,
                    gpu_name=torch.cuda.get_device_name(device) if stream is not None else None,
                    gpu_capability=list(torch.cuda.get_device_capability(device)) if stream is not None else None,
                    stream_id=int(stream.cuda_stream) if stream is not None else None),
                precision=dict(policy=precision,master_dtype='torch.float64' if precision=='fp64' else 'torch.float32',
                    input_dtype=str(dtype),output_dtype=output_boundary['dtype'],
                    source_master_dtype='torch.float32',autocast_dtype={'amp_fp16':'torch.float16','bf16':'torch.bfloat16'}.get(precision),**proof)))
        data['provenance']['model'].update(source_weights_sha256=source_hash,source_descriptor=source_descriptor)
        data.update(requested=dict(samples=plan.measure_iterations*plan.repeats),
            actual=dict(samples=plan.measure_iterations*plan.repeats),raw_evidence=raw,measurements=_point_metrics(raw,shape[0]))
        data['diagnostics']['limitations']=['Performance is not P28 accuracy preservation evidence.',
            'CPU BF16 proves dtype execution, not native instruction acceleration; backend may emulate.',
            'INT8 is dynamic Linear only; remaining allowed operators stay floating point.',
            'Conversion is a separate owned-copy scope, not cold model loading.']
        return PrecisionResult(data)
    finally:
        del owned,inputs


def _point_metrics(raw,batch):
    b=raw['benchmark']; s=raw['special']; t=raw['throughput']
    samples=b['timing']['latency_samples_ms'] if b is not None else s['latency_samples_ms']
    windows=t['raw_evidence']['windows'] if t is not None else s['windows'] if s is not None else []
    flat=[v for row in samples for v in row]
    rate=sum(w['completed_images'] for w in windows)/math.fsum(w['elapsed_wall_seconds'] for w in windows) if windows else None
    return dict(latency_mean_ms=dict(value=math.fsum(flat)/len(flat),unit='ms/batch',status='derived',reason=None),
                sustained_images_per_second=dict(value=rate,unit='images/s',status='measured' if windows else 'disabled',
                    reason=None if windows else 'not_requested'),
                batch_cardinality=batch)


def run_sweep(model, input_factory, *, axis, points, input_shape=(1,1,5,7), protocol=SweepProtocol()):
    """Finite ordered batch/resolution/precision points; CPU FP32 master is copied.

    input_factory(shape, dtype, device) must return exact resident NCHW input.
    Value/ordinary runtime failures may continue. OOM always stops admission;
    remaining points are not_run. State/stream mutation and runtime drain failure
    are fatal exceptions. There is no interpolation or automatic compatibility.
    """
    if type(protocol) is not SweepProtocol: raise TypeError('expected SweepProtocol')
    if axis not in ('batch','resolution','precision'): raise ValueError('unknown sweep axis')
    if not callable(input_factory): raise TypeError('input_factory must be callable')
    shape=_shape(input_shape)
    if type(points) not in (tuple,list) or not points: raise ValueError('explicit finite nonempty points required')
    normalized=[]
    for value in points:
        if axis=='batch': _integer(value,1); point=value
        elif axis=='resolution':
            if type(value) not in (tuple,list) or len(value)!=2: raise ValueError('resolution must be H,W')
            for item in value: _integer(item,1)
            point=list(value)
        else:
            if type(value) is not str or value not in _PRECISIONS: raise ValueError('unknown precision point')
            point=value
        if point in normalized: raise ValueError('duplicate point')
        normalized.append(point)
    if not _SWEEP_LOCK.acquire(blocking=False): raise RuntimeError('concurrent/reentrant sweeps unsupported')
    contexts=ExitStack(); snapshot=None; rng=None; cuda_rng=None; device=stream=None
    try:
        rng=random.getstate(),np.random.get_state(),torch.get_rng_state()
        modules,entries=_profile(model,torch.float32,torch.device('cpu'))
        snapshot=_Snapshot(modules,entries,torch.empty(0))
        descriptor,weights_hash=_descriptor(modules,entries)
        model.eval()
        device=torch.device(protocol.device)
        if device.type=='cuda':
            if not torch.cuda.is_available(): raise ValueError('CUDA unavailable; no CPU fallback')
            index=torch.cuda.current_device() if device.index is None else device.index
            if index>=torch.cuda.device_count(): raise ValueError('CUDA device unavailable')
            device=torch.device('cuda',index); contexts.enter_context(torch.cuda.device(device))
            stream=torch.cuda.current_stream(device); contexts.enter_context(torch.cuda.stream(stream))
            cuda_rng=torch.cuda.get_rng_state(device)
        rows=[]; stopped=False
        for index,value in enumerate(normalized):
            point_shape=list(shape)
            if axis=='batch': point_shape[0]=value
            elif axis=='resolution': point_shape[2:]=value
            precision=value if axis=='precision' else protocol.precision
            row=dict(point_id=str(uuid.uuid4()),observation_index=index,value=value,shape=point_shape,
                aspect_ratio=point_shape[3]/point_shape[2],precision=precision,status='not_run',reason='previous_point_stopped',
                result_id=None,measurement_refs=[],result=None)
            if not stopped:
                # Every point starts from the same caller RNG, independent of prior workload/failure.
                random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
                if cuda_rng is not None: torch.cuda.set_rng_state(cuda_rng,device)
                try:
                    measured=_point(model,input_factory,point_shape,precision,protocol,device,stream).to_dict()
                    row.update(status='measured',reason=None,result_id=measured['result_id'],result=measured,
                        measurement_refs=_refs(measured))
                except StateMutationError: raise
                except (torch.cuda.OutOfMemoryError, MemoryError) as error:
                    row.update(status='failed',reason='out_of_memory: '+str(error)); stopped=True
                except (_Unsupported,NotImplementedError) as error:
                    row.update(status='unsupported',reason=str(error)); stopped=not protocol.continue_on_error
                except (ValueError,RuntimeError) as error:
                    # CPU tensor/input allocation can report an ordinary RuntimeError
                    # rather than the typed CUDA OOM. Do not admit another point
                    # after this known allocator failure, even with continuation.
                    message=str(error).lower()
                    cpu_oom=(isinstance(error,RuntimeError) and 'defaultcpuallocator' in message
                             and any(text in message for text in ('not enough memory', "can't allocate memory", 'cannot allocate memory')))
                    row.update(status='failed',reason=('out_of_memory: ' if cpu_oom else type(error).__name__+': ')+str(error))
                    stopped=cpu_oom or not protocol.continue_on_error
                _context_check(device,stream)
                if stream is not None: torch.cuda.synchronize(device)
                snapshot.check(full=True)
            rows.append(row)
        feature={'batch':'P06','resolution':'P07','precision':'P08'}[axis]
        data=_base('sweep',feature,dict(id='finite_isolated_points',version=1,scope='ordered_resident_point_sweep',
            axis=axis,input_shape=list(shape),points=normalized,configuration=asdict(protocol),
            continuation='ordinary_errors_follow_flag_oom_stop_fatal_state_or_drain_raise',
            cross_point_compatibility='not_asserted'),dict(model=dict(structural_descriptor=descriptor,weights_sha256=weights_hash),
                input=dict(factory=_name(input_factory),requested_shape=list(shape)),
                environment=dict(device=str(device),torch=str(torch.__version__),python=platform.python_version())))
        data.update(requested=dict(points=len(rows)),actual=_counts(rows),measurements=dict(point_results='raw_evidence.points'),
            raw_evidence=dict(points=rows))
        data['diagnostics']['limitations']=['No interpolation, automatic comparison compatibility or P28 accuracy claim.',
            'Caller-exclusive ordinary dense CPU FP32 master; owned copies may move/convert.',
            'OOM stops admission; no allocator reset/cache flush/recovery claim.',
            'Factory and model Python side effects outside registered state are caller responsibility.']
        return SweepResult(data)
    except StateMutationError:
        if snapshot is not None: snapshot.restore_values()
        raise
    finally:
        try:
            if snapshot is not None: snapshot.restore_modes()
        finally:
            try:
                if rng is not None: random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
                if cuda_rng is not None: torch.cuda.set_rng_state(cuda_rng,device)
            finally:
                try: contexts.close()
                finally: _SWEEP_LOCK.release()


def _refs(data):
    raw=data['raw_evidence']
    return ([raw['benchmark']['identity']['run_id']] if raw['benchmark'] is not None else [raw['special']['result_id']])+(
        [raw['throughput']['result_id']] if raw['throughput'] is not None else [])


def _counts(rows):
    return dict(points=len(rows),**{status:sum(r['status']==status for r in rows) for status in ('measured','failed','unsupported','not_run')})


def _validate_sweep(data,kind):
    p=SweepProtocol(**data['protocol']['configuration'])
    if kind=='precision':
        if data['protocol']['id']!='isolated_precision_point' or data['protocol']['scope']!='resident_forward': raise ValueError('invalid precision scope')
        raw=data['raw_evidence']; shape=data['provenance']['input']['shape']; _shape(tuple(shape))
        precision=data['protocol']['precision']
        if precision not in _PRECISIONS: raise ValueError('unknown precision')
        b=raw['benchmark']; s=raw['special']; t=raw['throughput']
        if (b is None)==(s is None): raise ValueError('exactly one timing source required')
        if b is not None:
            b=BenchmarkResult(b).to_dict()
            if b['configuration']['precision']!=precision or b['input']['shape']!=shape: raise ValueError('mismatched benchmark identity')
            if any(b['configuration'][key]!=getattr(p,key) for key in ('warmup_iterations','measure_iterations','repeats')):
                raise ValueError('mismatched timing configuration')
            samples=b['timing']['latency_samples_ms']
            if t is not None:
                from .result import ThroughputResult
                ThroughputResult(t)
                if (t['provenance']['input']['shape']!=shape or t['protocol']['configuration']['precision']!=precision or
                        t['requested']['batch_count']!=p.sustained_batches): raise ValueError('mismatched throughput')
            if (t is not None)!=(p.sustained_batches is not None): raise ValueError('missing requested throughput')
        else:
            if precision not in ('bf16','int8_dynamic'): raise ValueError('invalid special precision')
            samples=s['latency_samples_ms']; uuid.UUID(s['result_id'])
            if (s['warmup_iterations']!=p.warmup_iterations or s['preflight_forwards']!=1 or
                    s['forward_count']!=1+p.repeats*(p.warmup_iterations+p.measure_iterations)+(
                        p.repeats*(p.warmup_iterations+p.sustained_batches) if p.sustained_batches is not None else 0)):
                raise ValueError('invalid special forward ledger')
            if len(s['windows'])!=(p.repeats if p.sustained_batches is not None else 0): raise ValueError('missing sustained windows')
            for index,w in enumerate(s['windows']):
                if (w['repeat_index']!=index or w['admitted_batches']!=p.sustained_batches or
                        w['completed_batches']!=p.sustained_batches or w['pending_batches']!=0 or
                        w['completed_images']!=shape[0]*p.sustained_batches or
                        not w['start_ns']<=w['admission_end_ns']<=w['end_ns'] or
                        w['elapsed_wall_seconds']!=(w['end_ns']-w['start_ns'])/1e9): raise ValueError('invalid special completion ledger')
                _positive(w['elapsed_wall_seconds'])
            if precision=='bf16' and s['output']['dtype']!='torch.bfloat16': raise ValueError('BF16 fallback forbidden')
            if precision=='int8_dynamic':
                proof=data['provenance']['precision']['execution_proof']
                if (proof['quantized_linear_count']<1 or proof['weight_dtype']!='torch.qint8' or
                        proof['operators']!=['quantized::linear_dynamic']): raise ValueError('missing INT8 execution proof')
        if len(samples)!=p.repeats or any(len(row)!=p.measure_iterations for row in samples): raise ValueError('invalid sample count')
        for row in samples:
            for v in row: _positive(v)
        if data['actual']!=dict(samples=p.repeats*p.measure_iterations) or data['requested']!=data['actual']: raise ValueError('inconsistent requested/actual')
        if data['measurements']!=_point_metrics(raw,shape[0]): raise ValueError('invalid measurements')
        return
    protocol=data['protocol']; axis=protocol['axis']; rows=data['raw_evidence']['points']
    if protocol['id']!='finite_isolated_points' or protocol['scope']!='ordered_resident_point_sweep' or axis not in ('batch','resolution','precision'): raise ValueError('invalid sweep scope')
    if data['feature_ids']!=[{'batch':'P06','resolution':'P07','precision':'P08'}[axis]]: raise ValueError('wrong feature axis')
    if not rows or len(rows)!=len(protocol['points']) or len({r['point_id'] for r in rows})!=len(rows): raise ValueError('invalid point counts/IDs')
    ids=[r['result_id'] for r in rows if r['status']=='measured']
    if len(set(ids))!=len(ids): raise ValueError('reused point result ID')
    for index,row in enumerate(rows):
        uuid.UUID(row['point_id']); _shape(tuple(row['shape']))
        if row['observation_index']!=index or row['value']!=protocol['points'][index]: raise ValueError('invalid point association')
        expected=list(protocol['input_shape'])
        if axis=='batch': expected[0]=row['value']
        elif axis=='resolution': expected[2:]=row['value']
        if row['shape']!=expected or row['aspect_ratio']!=expected[3]/expected[2]: raise ValueError('invalid shape/aspect ratio')
        if row['precision']!=(row['value'] if axis=='precision' else p.precision): raise ValueError('invalid point precision')
        if row['status']=='measured':
            measured=PrecisionResult(row['result']).to_dict()
            if measured['result_id']!=row['result_id'] or _refs(measured)!=row['measurement_refs'] or measured['provenance']['input']['shape']!=row['shape']: raise ValueError('invalid result association')
            if measured['protocol']['precision']!=row['precision']: raise ValueError('precision association mismatch')
            if measured['provenance']['model']['source_weights_sha256']!=data['provenance']['model']['weights_sha256']:
                raise ValueError('source model identity mismatch')
            if row['reason'] is not None: raise ValueError('measured point reason')
        elif (row['status'] not in ('failed','unsupported','not_run') or not row['reason'] or
              row['result'] is not None or row['result_id'] is not None or row['measurement_refs']): raise ValueError('invalid failed point')
    if data['actual']!=_counts(rows) or data['requested']!=dict(points=len(rows)): raise ValueError('inconsistent sweep counts')
