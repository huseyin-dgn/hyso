"""Serial instrumented pipeline: explicit frame IDs, bounded admission and stage clocks."""
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import copy
import inspect
import math
import platform
import random

import numpy as np
import torch

from hyso.benchmark.runner import (_profile, _Snapshot, _descriptor, _tensor, _shape, _output,
                                   _RUN_LOCK, StateMutationError)
from .inference import ThroughputProtocol, TailPolicy, _base, _clock, _context_check
from .result import PipelineResult, StageTimingResult, _integer, _latency_metrics


@dataclass(frozen=True)
class PipelineSample:
    sample_id: str
    payload: torch.Tensor


@dataclass(frozen=True)
class PipelineStage:
    name: str
    function: object
    device: str = 'cpu'

    def __post_init__(self):
        if self.name not in ('preprocess','transfer','postprocess','deliver'):
            raise ValueError('unsupported stage identity/order')
        _callable(self.function)
        if self.device not in ('cpu','cuda'):
            raise ValueError('stage device must be cpu or selected cuda')


def _callable(fn):
    if (not callable(fn) or inspect.iscoroutinefunction(fn) or
            inspect.iscoroutinefunction(getattr(fn,'__call__',None))):
        raise TypeError('stage/source must be a synchronous callable')


@dataclass(frozen=True)
class PipelineProtocol:
    device: str = 'cpu'
    precision: str = 'fp32'
    samples: int = 10
    repeats: int = 1
    warmup_samples: int = 1
    queue_depth: int = 2

    def __post_init__(self):
        ThroughputProtocol(device=self.device,precision=self.precision,batch_count=self.samples,
                           repeats=self.repeats,warmup_iterations=self.warmup_samples)
        _integer(self.queue_depth,1)


def _boundary(sample):
    if (type(sample) is not PipelineSample or type(sample.sample_id) is not str or not sample.sample_id.strip()
            or type(sample.payload) is not torch.Tensor or sample.payload.layout != torch.strided
            or sample.payload.device.type not in ('cpu','cuda') or sample.payload.is_quantized):
        raise ValueError('stage must return PipelineSample(nonempty ID, ordinary dense Tensor)')
    return dict(shape=list(sample.payload.shape),dtype=str(sample.payload.dtype),device=str(sample.payload.device))


def _stage(name, fn, argument, stage_device, device, stream, events):
    """Host launch and host completion clocks are distinct from stream elapsed."""
    before = None if argument is None else _boundary(argument)
    _context_check(device,stream)
    cuda = stage_device == 'cuda'
    if cuda and stream is None: raise ValueError('CUDA stage in CPU pipeline unsupported')
    start = _clock()
    if cuda: events[0].record(stream)
    returned = fn() if argument is None else fn(argument)
    launch_end = _clock()
    _context_check(device,stream)
    after = _boundary(returned)
    if cuda and returned.payload.device != device: raise ValueError('CUDA stage must return selected-device tensor')
    if argument is not None and returned.sample_id != argument.sample_id:
        raise ValueError('stage changed sample ID; E2E association invalid')
    event_ms = None
    if cuda:
        events[1].record(stream); events[1].synchronize()
        event_ms = events[0].elapsed_time(events[1])
        if not math.isfinite(event_ms) or event_ms <= 0: raise ValueError('invalid CUDA stage elapsed')
    end = _clock()
    if not start <= launch_end <= end or end <= start: raise ValueError('invalid stage host clock')
    return returned, dict(name=name, sample_id=returned.sample_id, start_ns=start, launch_end_ns=launch_end,
        completion_ns=end, host_launch_ms=(launch_end-start)/1e6, host_elapsed_ms=(end-start)/1e6,
        device_event_ms=event_ms, device_event_status='measured' if cuda else 'not_applicable',
        device_event_reason=None if cuda else 'synchronous_host_stage',device=stage_device,
        stream_id=int(stream.cuda_stream) if cuda else None,
        sync_policy='end_event_sync' if cuda else 'synchronous_callable',
        input_boundary=before,output_boundary=after,status='measured',reason=None,unit='ms')


def _metrics(rows):
    latencies=[row['e2e_ms'] for row in rows]
    # P50/P95 are descriptive for any nonempty sample; P99 uses B1 tail policy.
    from .result import _quantile
    tail=_latency_metrics([latencies],[.99],asdict(TailPolicy()))[0]
    return dict(p50_ms=dict(value=_quantile(latencies,.5),unit='ms',status='derived',reason=None),
                p95_ms=dict(value=_quantile(latencies,.95),unit='ms',status='derived',reason=None),p99_ms=tail)


def measure_pipeline(model, source, *, preprocess=None, transfer=None, postprocess=None, deliver=None,
                     protocol=PipelineProtocol()):
    """Measured source/read-start to postprocess completion, plus optional delivery.

    source() returns unique PipelineSample IDs and CPU tensors; stages accept and
    return the same-ID wrapper. Queue policy is bounded block, no dropping/overlap.
    CUDA stages are isolated with selected-stream end-event sync. This is separate
    instrumented analysis, never a BenchmarkRunner latency or sustained FPS run.
    """
    if type(protocol) is not PipelineProtocol: raise TypeError('protocol must be PipelineProtocol')
    _callable(source)
    declared=[]
    for name,value in (('preprocess',preprocess),('transfer',transfer),('postprocess',postprocess),('deliver',deliver)):
        if value is not None:
            if type(value) is not PipelineStage or value.name != name: raise ValueError('incorrect stage name/order')
            if protocol.device=='cpu' and value.device=='cuda': raise ValueError('CUDA stage in CPU pipeline')
            declared.append(value)
    if protocol.device!='cpu' and transfer is None: raise ValueError('CUDA pipeline requires explicit transfer stage')
    if protocol.device!='cpu' and transfer.device!='cuda': raise ValueError('CUDA transfer requires CUDA event scope')
    if not _RUN_LOCK.acquire(blocking=False): raise RuntimeError('concurrent/reentrant performance runs unsupported')
    contexts=ExitStack(); rng=cuda_rng=snapshot=device=stream=None
    queue=deque(); ids=set(); rows=[]; max_pending=0; activated=False
    try:
        rng=random.getstate(),np.random.get_state(),torch.get_rng_state()
        device=torch.device(protocol.device)
        if device.type=='cuda':
            if not torch.cuda.is_available(): raise ValueError('CUDA unavailable; no CPU fallback')
            device=torch.device('cuda',torch.cuda.current_device() if device.index is None else device.index)
            contexts.enter_context(torch.cuda.device(device)); stream=torch.cuda.current_stream(device)
            contexts.enter_context(torch.cuda.stream(stream)); cuda_rng=torch.cuda.get_rng_state(device)
        else: stream=None
        dtype=torch.float64 if protocol.precision=='fp64' else torch.float32
        modules,entries=_profile(model,dtype,device)
        with torch.inference_mode(False),torch.no_grad(): placeholder=torch.empty(0,device=device,dtype=dtype)
        snapshot=_Snapshot(modules,entries,placeholder)
        descriptor,weights_hash=_descriptor(modules,entries)
        events=None
        if stream is not None:
            events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            events[0].record(stream); events[1].record(stream); events[1].synchronize()
        stage_specs=[('source','cpu')]+[(s.name,s.device) for s in declared if s.name in ('preprocess','transfer')]
        stage_specs+=[('inference','cuda' if stream is not None else 'cpu')]
        stage_specs += [(s.name,s.device) for s in declared if s.name in ('postprocess','deliver')]
        model.eval()
        activated=True
        with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,
                                                   enabled=protocol.precision=='amp_fp16'):
            for repeat in range(protocol.repeats):
                if stream is not None: torch.cuda.synchronize(device)
                for measured,count in ((False,protocol.warmup_samples),(True,protocol.samples)):
                    admitted=0
                    while admitted<count or queue:
                        while admitted<count and len(queue)<protocol.queue_depth:
                            item,record=_stage('source',source,None,'cpu',device,stream,events)
                            if item.sample_id in ids: raise ValueError('duplicate source sample ID')
                            if item.payload.device.type!='cpu': raise ValueError('source must provide CPU tensors')
                            ids.add(item.sample_id)
                            # Protect caller-owned source payload against in-place adapters.
                            with torch.inference_mode(False),torch.no_grad(): owned=item.payload.detach().clone()
                            queue.append((PipelineSample(item.sample_id,owned),record)); admitted+=1
                            if measured: max_pending=max(max_pending,len(queue))
                        sample,source_record=queue.popleft(); sample_id=sample.sample_id
                        stages=[source_record]; begin=_clock(); queue_wait=begin-source_record['completion_ns']
                        for spec in (preprocess,transfer):
                            if spec is not None:
                                sample,record=_stage(spec.name,spec.function,sample,spec.device,device,stream,events)
                                stages.append(record)
                        with torch.inference_mode(False),torch.no_grad(): actual_input=sample.payload.detach().clone()
                        sample=PipelineSample(sample.sample_id,actual_input)
                        _tensor(sample.payload,dtype,input_tensor=True,device=device); _shape(tuple(sample.payload.shape))
                        input_guard=_Snapshot([],[],sample.payload)
                        def inference(frame): return PipelineSample(frame.sample_id,model(frame.payload))
                        sample,record=_stage('inference',inference,sample,'cuda' if stream is not None else 'cpu',device,stream,events)
                        stages.append(record); _output(sample.payload,input_guard.inputs.shape[0],device=device)
                        input_guard.check(full=True); del input_guard
                        if postprocess is not None:
                            sample,record=_stage('postprocess',postprocess.function,sample,postprocess.device,device,stream,events)
                            stages.append(record)
                        # Always drain the declared stream before E2E post completion.
                        if stream is not None: events[1].record(stream); events[1].synchronize()
                        post_end=_clock()
                        if not torch.isfinite(sample.payload).all(): raise ValueError('nonfinite postprocess output')
                        if deliver is not None:
                            sample,record=_stage('deliver',deliver.function,sample,deliver.device,device,stream,events)
                            stages.append(record)
                        delivered=_clock()
                        if measured:
                            rows.append(dict(sample_id=sample_id,repeat_index=repeat,source_start_ns=source_record['start_ns'],
                                source_completion_ns=source_record['completion_ns'],postprocess_completion_ns=post_end,
                                delivery_completion_ns=delivered,queue_wait_ns=queue_wait,
                                e2e_ms=(post_end-source_record['start_ns'])/1e6,status='completed',reason=None,unit='ms',stages=stages))
                        del sample, actual_input
                        snapshot.check(full=True)
        data=_base('pipeline','P03',dict(id='serial_bounded_pipeline',version=1,scope='read_start_to_post_completion',
            configuration=asdict(protocol),clock='perf_counter_ns',queue_policy='block',stage_order=[n for n,d in stage_specs],
            stage_devices=[d for n,d in stage_specs],p99_tail_policy=asdict(TailPolicy()),
            included=['source_read','source_clone','queue_wait','declared_stages','guards_and_stage_instrumentation','terminal_completion'],
            excluded=['optional_delivery_from_e2e','model_snapshot_setup','exports','tracking']),
            dict(model=dict(structural_descriptor=descriptor,weights_sha256=weights_hash),
                 input=dict(source='caller_sample_source',source_callable=(getattr(source,'__module__',None) or '<unknown_module>')+'.'+(getattr(source,'__qualname__',None) or type(source).__name__),
                            content_hash_status='not_collected_varying_source',unit_of_work='one_ID_may_contain_a_batch'),
                 environment=dict(device=str(device),python=platform.python_version(),torch=str(torch.__version__),os=platform.platform(),
                    stream_id=int(stream.cuda_stream) if stream is not None else None,cuda_build=torch.version.cuda,
                    gpu_name=torch.cuda.get_device_name(device) if stream is not None else None,
                    gpu_capability=list(torch.cuda.get_device_capability(device)) if stream is not None else None,
                    backend='pytorch_eager',driver_version=None,driver_status='not_collected'),
                 stages=[dict(name=s.name,callable=getattr(s.function,'__qualname__',type(s.function).__name__)) for s in declared]))
        data.update(requested=dict(samples=protocol.samples,repeats=protocol.repeats,warmup_samples=protocol.warmup_samples),
            actual=dict(completed=len(rows),dropped=0,pending=0,warmup_samples=protocol.warmup_samples*protocol.repeats,
                        forward_count=(protocol.samples+protocol.warmup_samples)*protocol.repeats,max_pending=max_pending),
            measurements=_metrics(rows),raw_evidence=dict(samples=rows))
        data['diagnostics']['limitations']=['Instrumented serial blocking pipeline, not overlapped scheduling/camera FPS.',
            'Stage CUDA event and host launch/completion clocks are distinct; stage sum is not E2E.',
            'Queue wait includes other reads and prior instrumentation/guards; no frame dropping supported.',
            'Hidden cross-stream/device work, async CPU callbacks, concurrent callers/external side effects unsupported.',
            'Pooled samples need not be independent; P99 eligibility does not establish a confidence interval.']
        return PipelineResult(data)
    except BaseException as error:
        if stream is not None: torch.cuda.synchronize(device)
        if snapshot is not None and activated:
            try: snapshot.check(full=True)
            except StateMutationError as mutation:
                snapshot.restore_values()
                raise StateMutationError(str(mutation)+'; best-effort registered-value rollback') from error
        raise
    finally:
        queue.clear()
        try:
            if snapshot is not None: snapshot.restore_modes()
        finally:
            try:
                if rng is not None:
                    random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
            finally:
                try:
                    if cuda_rng is not None: torch.cuda.set_rng_state(cuda_rng,device)
                finally:
                    try: contexts.close()
                    finally: _RUN_LOCK.release()


def analyze_stages(pipeline_result):
    """Build separate P04 result from measured stage evidence without rerunning."""
    if not isinstance(pipeline_result,PipelineResult): raise TypeError('source must be PipelineResult')
    source=PipelineResult(pipeline_result.to_dict()).to_dict()
    data=copy.deepcopy(source)
    import uuid
    data.update(format='hyso.performance.stage_timing',feature_ids=['P04'],result_id=str(uuid.uuid4()))
    data['identity']['source_result_id']=source['result_id']
    data['protocol']['scope']='instrumented_stage_host_and_device'
    data['measurements']=_stage_metrics(data['raw_evidence']['samples'],data['protocol']['stage_order'])
    return StageTimingResult(data)


def measure_stages(model, source, **kwargs):
    return analyze_stages(measure_pipeline(model,source,**kwargs))


def _stage_metrics(rows,names):
    result={}
    for name in names:
        records=[stage for row in rows for stage in row['stages'] if stage['name']==name]
        device=[r['device_event_ms'] for r in records if r['device_event_ms'] is not None]
        result[name]=dict(count=len(records),unit='ms',host_mean_ms=math.fsum(r['host_elapsed_ms'] for r in records)/len(records),
            device_mean_ms=math.fsum(device)/len(device) if device else None,
            device_status='measured' if device else 'not_applicable',device_reason=None if device else 'synchronous_host_stage')
    return result


def _validate_pipeline(data,kind):
    p=PipelineProtocol(**data['protocol']['configuration']); protocol=data['protocol']
    scope='read_start_to_post_completion' if kind=='pipeline' else 'instrumented_stage_host_and_device'
    if protocol['id']!='serial_bounded_pipeline' or protocol['scope']!=scope or protocol['clock']!='perf_counter_ns' or protocol['queue_policy']!='block':
        raise ValueError('unsupported pipeline clock/scope/queue')
    rows=data['raw_evidence']['samples']; names=protocol['stage_order']; devices=protocol['stage_devices']
    if (not rows or len(rows)!=p.samples*p.repeats or len({r['sample_id'] for r in rows})!=len(rows)
            or names[0]!='source' or names.count('inference')!=1 or len(names)!=len(devices)):
        raise ValueError('invalid sample/stage counts or duplicate IDs')
    allowed=['source','preprocess','transfer','inference','postprocess','deliver']
    if names!=[name for name in allowed if name in names]: raise ValueError('incorrect stage order')
    if any(d not in ('cpu','cuda') for d in devices) or devices[0]!='cpu': raise ValueError('invalid stage device declarations')
    for index,row in enumerate(rows):
        if type(row['sample_id']) is not str or not row['sample_id'].strip() or row['repeat_index']!=index//p.samples or row['status']!='completed':
            raise ValueError('invalid sample ID/outcome/repeat')
        for name in ('source_start_ns','source_completion_ns','postprocess_completion_ns','delivery_completion_ns','queue_wait_ns'): _integer(row[name])
        if not row['source_start_ns']<=row['source_completion_ns']<=row['postprocess_completion_ns']<=row['delivery_completion_ns']:
            raise ValueError('invalid E2E timestamps')
        if row['e2e_ms']!=(row['postprocess_completion_ns']-row['source_start_ns'])/1e6 or row['e2e_ms']<=0:
            raise ValueError('E2E latency disagrees with timestamps')
        if [s['name'] for s in row['stages']]!=names: raise ValueError('missing stage')
        previous=row['source_start_ns']
        for stage,stage_device in zip(row['stages'],devices):
            if stage['sample_id']!=row['sample_id'] or stage['device']!=stage_device: raise ValueError('wrong frame/stage device')
            if stage['status']!='measured' or stage['reason'] is not None or stage['unit']!='ms': raise ValueError('invalid stage status/unit')
            for key in ('start_ns','launch_end_ns','completion_ns'): _integer(stage[key])
            if not previous<=stage['start_ns']<=stage['launch_end_ns']<=stage['completion_ns'] or stage['completion_ns']<=stage['start_ns']:
                raise ValueError('invalid stage timestamp order')
            if (stage['host_elapsed_ms']!=(stage['completion_ns']-stage['start_ns'])/1e6 or
                    stage['host_launch_ms']!=(stage['launch_end_ns']-stage['start_ns'])/1e6): raise ValueError('invalid host stage duration')
            cuda=stage_device=='cuda'
            if cuda:
                if p.device=='cpu' or stage['device_event_status']!='measured' or stage['device_event_ms'] is None or stage['device_event_ms']<=0 or stage['sync_policy']!='end_event_sync':
                    raise ValueError('missing actual CUDA stage completion')
            elif stage['device_event_ms'] is not None or stage['device_event_status']!='not_applicable': raise ValueError('CPU stage cannot claim CUDA timing')
            previous=stage['completion_ns']
        stages=row['stages']
        if stages[0]['start_ns']!=row['source_start_ns'] or stages[0]['completion_ns']!=row['source_completion_ns']:
            raise ValueError('source evidence mismatch')
        if stages[1]['start_ns']-row['source_completion_ns']<row['queue_wait_ns']: raise ValueError('inconsistent queue wait')
        if max(s['completion_ns'] for s in stages if s['name']!='deliver')>row['postprocess_completion_ns']:
            raise ValueError('E2E ends before stage completion')
    actual=data['actual']
    if (actual['completed']!=len(rows) or actual['dropped']!=0 or actual['pending']!=0 or
            actual['warmup_samples']!=p.warmup_samples*p.repeats or actual['forward_count']!=(p.samples+p.warmup_samples)*p.repeats or
            not 1<=actual['max_pending']<=p.queue_depth): raise ValueError('inconsistent pipeline completion accounting')
    for value in actual.values(): _integer(value)
    expected=_metrics(rows) if kind=='pipeline' else _stage_metrics(rows,names)
    if data['measurements']!=expected: raise ValueError('pipeline statistics disagree with evidence')
    if data['requested']!=dict(samples=p.samples,repeats=p.repeats,warmup_samples=p.warmup_samples): raise ValueError('invalid requested counts')
