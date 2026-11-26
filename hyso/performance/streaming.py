"""P30 real OpenCV capture/replay with explicit application-queue lifecycle evidence."""
import typing as _typing
from collections import deque
from dataclasses import dataclass, asdict
import hashlib
import math
import os
from pathlib import Path
import threading
import time

from .runtime_result import _result, _hash, _int, _number, _name, _digest
from .stability import WorkloadRecipe, _run_owned, _read_journal, _factory, _Journal, _atomic


def _physical_uuid(value):
    from uuid import UUID
    return 'GPU-'+str(UUID(str(value).removeprefix('GPU-')))


@dataclass(frozen=True)
class StreamSource:
    """Describe the explicit bounded streaming input and its identity.

    Args:
        kind (str): Existing run or artifact kind accepted by this operation. Required.
        location (str | int): Explicit source locator. Required.
        provenance (str): Plain origin/environment metadata; declarations are not producer attestation. Required.
        camera_binding (dict | None): Explicit permitted camera/source binding; no device discovery is inferred. Default: None.

    Returns:
        An instance of StreamSource with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        StreamSource(kind=kind, location=location, provenance=provenance)

    HYSo Guide: analysis.p30
    Offline: hyso docs show analysis.p30 --lang en
    Open: hyso docs open analysis.p30 --lang en
    """
    kind: _typing.Literal['video_file', 'camera']
    location: str | int
    provenance: str
    camera_binding: dict | None = None

    def __post_init__(self):
        _name(self.provenance)
        if self.kind=='video_file':
            if type(self.location) is not str or not Path(self.location).is_absolute() or not Path(self.location).is_file():
                raise ValueError('existing absolute video path required')
        elif self.kind=='camera':
            _int(self.location,0,32)
            if self.camera_binding is not None:
                from ._windows_camera import validate_binding
                validate_binding(self.camera_binding,self.location)
        else:raise NotImplementedError('local recorded video or camera only; no synthetic camera adapter')
        if self.kind!='camera' and self.camera_binding is not None:
            raise ValueError('camera identity cannot be attached to a video file')


@dataclass(frozen=True)
class StreamProtocol:
    """Finite experiment plan. Defaults describe a smoke run, not hardware limits.

    Args:
        max_frames (int): Upper bound on frames retained/accepted by this operation. Default: 16.
        queue_capacity (int): Maximum queued streaming frames. Default: 2.
        queue_policy (str): Explicit supported queue overflow policy. Default: 'block'.
        late_after_seconds (float): Late after in seconds under the native protocol. Default: 0.1.
        pace_video (bool): Pace video delivery according to source timestamps. Default: False.
        startup_timeout_seconds (float): Startup timeout in seconds under the native protocol. Default: 45.0.
        heartbeat_timeout_seconds (float): Heartbeat timeout in seconds under the native protocol. Default: 5.0.
        max_runtime_seconds (float): Max runtime in seconds under the native protocol. Default: 30.0.
        max_journal_bytes (int): Max journal in bytes; this does not imply a GPU hardware quota. Default: 128 * 1024 ** 2.

    Returns:
        An instance of StreamProtocol with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        StreamProtocol()

    HYSo Guide: analysis.p30
    Offline: hyso docs show analysis.p30 --lang en
    Open: hyso docs open analysis.p30 --lang en
    """
    max_frames: int = 16
    queue_capacity: int = 2
    queue_policy: _typing.Literal['block', 'drop_newest'] = 'block'
    late_after_seconds: float = .1
    pace_video: bool = False
    startup_timeout_seconds: float = 45.
    heartbeat_timeout_seconds: float = 5.
    max_runtime_seconds: float = 30.
    max_journal_bytes: int = 128*1024**2
    process_memory_bytes: int | None = None
    job_memory_bytes: int | None = None
    max_processes: int | None = None

    def __post_init__(self):
        _int(self.max_frames,1);_int(self.queue_capacity,1);_int(self.max_journal_bytes,65536)
        for v in (self.late_after_seconds,self.startup_timeout_seconds,self.heartbeat_timeout_seconds,self.max_runtime_seconds):_number(v,.001)
        if self.queue_policy not in ('block','drop_newest') or type(self.pace_video) is not bool:raise ValueError('unknown stream queue/pacing policy')
        for limit in (self.process_memory_bytes,self.job_memory_bytes,self.max_processes):
            if limit is not None:_int(limit,1)
        if self.process_memory_bytes is not None and self.job_memory_bytes is not None and self.job_memory_bytes<self.process_memory_bytes:raise ValueError('job budget below process budget')


def measure_streaming(source: '_typing.Any', processor_recipe: '_typing.Any', directory: '_typing.Any', *, protocol: '_typing.Any'=StreamProtocol()) -> '_typing.Any':
    """Run an explicitly requested bounded streaming workload in its owned worker.

    Args:
        source (_typing.Any): Typed native/retained source required by this API, not a declaration of producer authority. Required.
        processor_recipe (_typing.Any): Trusted processor construction recipe for the owned streaming worker. Required.
        directory (_typing.Any): Dedicated output/owned-run directory; existing overwrite and ownership rules apply. Required.
        protocol (_typing.Any): Typed native measurement protocol; scope, budgets and device guards remain unchanged. Default: StreamProtocol().

    Returns:
        A validated RuntimeResult retaining actual evidence, coverage and unavailable-status fields.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        measure_streaming(source, processor_recipe, directory)

    HYSo Guide: analysis.p30
    Offline: hyso docs show analysis.p30 --lang en
    Open: hyso docs open analysis.p30 --lang en
    """
    if type(source) is not StreamSource or type(processor_recipe) is not WorkloadRecipe or type(protocol) is not StreamProtocol:
        raise TypeError('typed stream source/recipe/protocol required')
    if processor_recipe.gpu_uuid is not None and any(v is None for v in (protocol.process_memory_bytes,protocol.job_memory_bytes,protocol.max_processes)):
        raise ValueError('CUDA streaming requires explicit owned process/Job budgets')
    p=protocol
    protocol_data={k:v for k,v in asdict(p).items() if v is not None}
    root=_run_owned(directory,'streaming',dict(source=asdict(source),recipe=asdict(processor_recipe),protocol=protocol_data),
        startup_timeout=p.startup_timeout_seconds,heartbeat_timeout=p.heartbeat_timeout_seconds,
        max_runtime=p.max_runtime_seconds,max_bytes=p.max_journal_bytes)
    return _result('P30',_read_journal(root))


def _analyze_stream(e):
    p=StreamProtocol(**e['request']['protocol']);c=e['controller'];header=e['header'];rows=e['rows'];terminal=e['terminal']
    _int(c['cleanup']['active_processes'])
    if type(c['exit_code']) is not int:raise ValueError('invalid worker exit code')
    if c['token']!=e['request']['token'] or c['cleanup']['active_processes']!=0 or c['kill_on_close'] is not True:
        raise ValueError('stream controller association/cleanup mismatch')
    if header is None:
        if rows or e['chunks'] or terminal:raise ValueError('stream rows without capture header')
        analysis=dict(status='aborted',reason=c['reason'] or 'capture_or_processor_setup_failed',received=0,processed=0,dropped=0,pending=0)
        if 'max_journal_bytes' in e['request']['protocol']:
            from ._stream_diagnostics import _diagnostics
            analysis.update(processed_per_second=None,late_processed=0,terminal_accounting_complete=False)
            analysis.update(_diagnostics(e,analysis))
        return analysis
    if header['run_id']!=c['token'] or header['pid']!=c['launcher_pid']:raise ValueError('stream run identity mismatch')
    if header['clock_id']!=c['token']+':perf_counter_ns':raise ValueError('stream clock domain mismatch')
    if 'lifecycle_version' in header and header['lifecycle_version']!=2:raise ValueError('unsupported lifecycle version')
    if any(header['capture'][k]!=e['request']['source'][k] for k in ('kind','location','provenance')):
        raise ValueError('capture source association mismatch')
    if header['processor'].get('processing_version')==2 and (not e['request']['recipe'].get('gpu_uuid') or header['processor']['device_id']!=e['request']['recipe']['gpu_uuid']):raise ValueError('GPU header/recipe identity mismatch')
    source=e['request']['source']
    if source.get('camera_binding') is not None:
        if (header['capture'].get('camera_binding')!=source['camera_binding']
                or header['capture'].get('backend')!='DSHOW'):
            raise ValueError('physical camera binding/backend mismatch')
    expected_target=p.max_frames
    if source['kind']=='video_file':
        _digest(header['capture']['file_sha256']);_int(header['capture']['declared_frames'],1)
        expected_target=min(p.max_frames,header['capture']['declared_frames'])
    _int(header['target_frames'],1)
    if header['target_frames']!=expected_target:raise ValueError('stream target association mismatch')
    source_recipe=header['processor_source'];requested_recipe=e['request']['recipe']
    if any(source_recipe[k]!=requested_recipe[k] for k in ('factory','configuration')):
        raise ValueError('stream processor recipe association mismatch')
    if source_recipe['module_sha256'] is not None:_digest(source_recipe['module_sha256'])
    previous=_hash(header);cursor=0
    for i,chunk in enumerate(e['chunks']):
        n=chunk['rows'];_int(n,1,64)
        body=dict(schema_version=1,run_id=header['run_id'],index=i,previous_sha256=previous,rows=rows[cursor:cursor+n])
        previous=_hash(body);cursor+=n
        if chunk['name']!=f'chunk-{i:06d}.json' or chunk['sha256']!=previous:raise ValueError('stream journal hash mismatch')
    if cursor!=len(rows):raise ValueError('stream chunk accounting mismatch')
    states={};pending_ids=set();queue=deque();inflight=None;received=processed=dropped=late=0;last_ns=0;max_queue=0;blocks={};blocked_ns=0;errors=[];latencies=[];read_end=0
    unadmitted=None;producer_ready_ns=0
    for i,r in enumerate(rows):
        _int(r['index']);_int(r['queue_depth'])
        if r['index']!=i:raise ValueError('stream sequence mismatch')
        stamp=r['timestamp_ns'];_int(stamp,last_ns);last_ns=stamp;kind=r['event'];frame=r.get('frame_id')
        if r.get('clock_id',header['clock_id'])!=header['clock_id']:raise ValueError('mixed stream clock domains')
        if kind=='received':
            if unadmitted is not None:raise ValueError('next capture before prior frame admission/drop')
            if frame!=str(received) or frame in states:raise ValueError('duplicate/reordered frame ID')
            _digest(r['content_sha256']);_int(r['read_start_ns'],max(read_end,producer_ready_ns));_int(r['read_end_ns'],r['read_start_ns'])
            if r['read_end_ns']>stamp:raise ValueError('read completion after receive')
            read_end=r['read_end_ns']
            assigned=r['frame_assigned_ns'] if header.get('lifecycle_version')==2 else r.get('frame_assigned_ns',stamp)
            _int(assigned,read_end)
            if assigned>stamp:raise ValueError('frame assignment after receive')
            if 'frame_shape' in r:
                shape=r['frame_shape']
                if type(shape) is not list or len(shape)!=3 or shape[2]!=3:raise ValueError('invalid BGR frame shape')
                for dimension in shape:_int(dimension,1)
                _int(r['frame_nbytes'],1)
                if r['frame_nbytes']!=math.prod(shape):raise ValueError('frame byte evidence mismatch')
            pts=r['media_pts_ms']
            if pts is not None:_number(pts)
            received+=1;pending_ids.add(frame);unadmitted=frame
            states[frame]=dict(state='received',receive_ns=stamp,read_start_ns=r['read_start_ns'])
        elif kind=='enqueued':
            if frame not in states or states[frame]['state']!='received' or len(queue)>=p.queue_capacity:raise ValueError('invalid queue admission')
            if frame in blocks:raise ValueError('queue admission before backpressure end')
            queue.append(frame);states[frame]['state']='queued';states[frame]['enqueued_ns']=stamp
            unadmitted=None;producer_ready_ns=stamp
        elif kind=='dequeued':
            if not queue or queue.popleft()!=frame or inflight is not None:raise ValueError('invalid queue dispatch')
            states[frame]['state']='processing';states[frame]['dequeued_ns']=stamp;inflight=frame
        elif kind=='processed':
            if frame!=inflight or states[frame]['state']!='processing':raise ValueError('invalid/duplicate processing completion')
            _int(r['process_start_ns'],states[frame]['dequeued_ns']);_int(r['process_end_ns'],r['process_start_ns'])
            if r['process_end_ns']>stamp:raise ValueError('processing after completion event')
            stages=r['stages'];last=r['process_start_ns'];stage_names=set()
            for stage in stages:
                _name(stage['name']);_int(stage['start_ns'],last);_int(stage['end_ns'],stage['start_ns'])
                if stage['name'] in stage_names:raise ValueError('duplicate processing stage')
                stage_names.add(stage['name'])
                if stage['end_ns']>r['process_end_ns']:raise ValueError('stage outside processing window')
                last=stage['end_ns']
            if header['processor'].get('kind')=='frame_model_inference':
                if not {'preprocess','inference'}<=stage_names:raise ValueError('missing model inference stages')
                if [s['name'] for s in stages if s['name'] in ('preprocess','inference','postprocess')] not in (
                        ['preprocess','inference'],['preprocess','inference','postprocess']):
                    raise ValueError('invalid model stage order')
            if header['processor'].get('processing_version')==2:
                if not e['request']['recipe'].get('gpu_uuid') or header['processor']['device_id']!=e['request']['recipe']['gpu_uuid']:raise ValueError('GPU frame identity mismatch')
                infer=next(s for s in stages if s['name']=='inference')
                if infer.get('completed') is not True or type(infer.get('stream_id')) is not int or type(infer.get('cuda_event_ms')) not in (int,float) or not math.isfinite(infer['cuda_event_ms']) or infer['cuda_event_ms']<0:raise ValueError('unverified CUDA frame completion')
            elapsed=stamp-states[frame]['receive_ns'];late+=elapsed>p.late_after_seconds*1e9
            latencies.append(elapsed/1e6);processed+=1;states[frame]['state']='processed';pending_ids.remove(frame);inflight=None
        elif kind=='dropped':
            if frame not in states or states[frame]['state'] in ('processed','dropped'):raise ValueError('invalid/duplicate drop')
            if frame in blocks:raise ValueError('drop before backpressure end')
            if r['reason']=='queue_full' and (states[frame]['state']!='received' or p.queue_policy!='drop_newest' or len(queue)!=p.queue_capacity):
                raise ValueError('unsupported queue-full drop evidence')
            if r['reason']=='processing_error' and frame!=inflight:raise ValueError('processing error without dispatch')
            if states[frame]['state']=='queued':
                if not queue or queue.popleft()!=frame:raise ValueError('drop queue order mismatch')
            if frame==inflight:inflight=None
            if r['reason'] not in ('queue_full','aborted','processing_error'):raise ValueError('unknown drop disposition')
            if r['reason']!='queue_full':errors.append('drop: '+r['reason'])
            if frame==unadmitted:unadmitted=None;producer_ready_ns=stamp
            states[frame]['state']='dropped';pending_ids.remove(frame);dropped+=1
        elif kind=='backpressure_start':
            if blocks or frame not in states or states[frame]['state']!='received' or len(queue)!=p.queue_capacity or p.queue_policy!='block':raise ValueError('invalid backpressure boundary')
            blocks[frame]=stamp
        elif kind=='backpressure_end':
            if frame not in blocks:raise ValueError('unpaired backpressure boundary')
            if len(queue)>=p.queue_capacity:raise ValueError('backpressure ended while FIFO remains full')
            blocked_ns+=stamp-blocks.pop(frame)
        elif kind=='failure':_name(r['reason']);errors.append(r['reason'])
        else:raise ValueError('unknown stream event')
        if r['queue_depth']!=len(queue):raise ValueError('queue depth differs from lifecycle evidence')
        max_queue=max(max_queue,len(queue))
        if received!=processed+dropped+len(pending_ids):raise ValueError('received conservation failed')
    if received>p.max_frames:raise ValueError('capture budget exceeded')
    complete=False
    if terminal is not None:
        for key in ('received','processed','dropped','rows'):_int(terminal[key])
        if terminal['status'] not in ('completed','failed') or type(terminal['closed']) is not bool:raise ValueError('invalid stream terminal')
        if (terminal['chunks']!=e['chunks'] or terminal['last_chunk_sha256']!=previous or terminal['rows']!=len(rows)
                or terminal['received']!=received or terminal['processed']!=processed or terminal['dropped']!=dropped):
            raise ValueError('stream terminal ledger mismatch')
        if terminal['status']=='completed' and (pending_ids or blocks or inflight is not None):
            raise ValueError('completed terminal with incomplete frame lifecycle')
        complete=(terminal['status']=='completed' and terminal['closed'] is True and received==header['target_frames']
            and not errors and not blocks and received==processed+dropped and c['reason'] is None and c['exit_code']==0 and not e['incomplete_tail'])
    elapsed=(last_ns-rows[0]['read_start_ns'])/1e9 if rows and rows[0]['event']=='received' else None
    analysis=dict(status='completed' if complete else 'aborted',reason=None if complete else c['reason'] or
        ((terminal['reason'] or 'incomplete_frame_coverage') if terminal else 'missing_terminal'),received=received,processed=processed,dropped=dropped,
        pending=received-processed-dropped,late_processed=late,max_queue_depth=max_queue,backpressure_seconds=blocked_ns/1e9,
        elapsed_seconds=elapsed,processed_per_second=processed/elapsed if elapsed and complete else None,
        mean_receive_to_completion_ms=sum(latencies)/len(latencies) if latencies else None,failures=errors,
        source_kind=header['capture']['kind'],queue_scope='owned_application_queue_not_driver_or_camera_buffer',
        terminal_accounting_complete=complete,accounting_scope='full_run' if complete else 'committed_prefix_uncommitted_tail_unknown',
        capture_clock='host_read_brackets_not_sensor_exposure',media_clock='separate_no_cross_clock_duration',
        conservation='received=processed+dropped+pending')
    # Preserve the derived schema of previously saved P30 envelopes.
    if header.get('lifecycle_version')==2:
        from ._stream_diagnostics import _diagnostics
        analysis.update(_diagnostics(e,analysis))
    return analysis


class FrameInferenceProcessor:
    """BGR uint8 -> CPU preprocessing -> explicitly selected CPU/CUDA owned inference.

    Args:
        model (_typing.Any): Caller-owned model; device, dtype and task must match the native operation. Required.
        size (_typing.Sequence[int]): Explicit byte count or shape extent required by this API. Default: (32, 32).

    Returns:
        An instance of FrameInferenceProcessor with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        FrameInferenceProcessor(model)

    HYSo Guide: analysis.p30
    Offline: hyso docs show analysis.p30 --lang en
    Open: hyso docs open analysis.p30 --lang en
    """
    def __init__(self,model: '_typing.Any',*,size: '_typing.Sequence[int]'=(32,32),device: 'str'='cpu',gpu_uuid: '_typing.Any'=None,cuda_memory_fraction: '_typing.Any'=None) -> 'None':
        """Initialize FrameInferenceProcessor with the supplied settings.

        Args:
            model (_typing.Any): Caller-owned model; device, dtype and task must match the native operation. Required.
            size (_typing.Sequence[int]): Explicit byte count or shape extent required by this API. Default: (32, 32).

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            FrameInferenceProcessor(model)

        HYSo Guide: analysis.p30
        Offline: hyso docs show analysis.p30 --lang en
        Open: hyso docs open analysis.p30 --lang en
        """
        import copy
        import torch
        from hyso.benchmark.runner import _profile,_descriptor,_Snapshot,_output
        if len(size)!=2:raise ValueError('explicit H,W required')
        for v in size:_int(v,1)
        self.device=torch.device(device)
        if self.device.type not in ('cpu','cuda'):raise NotImplementedError('CPU/CUDA frame processing only')
        if self.device.type=='cuda':
            if not gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu_uuid or self.device.index not in (None,0):raise ValueError('explicit UUID-visible CUDA device 0 required')
            if type(cuda_memory_fraction) not in (int,float) or not math.isfinite(cuda_memory_fraction) or not 0<cuda_memory_fraction<=1:raise ValueError('explicit CUDA allocator fraction required')
            if not torch.cuda.is_available():raise NotImplementedError('CUDA unavailable; no CPU fallback')
            self.device=torch.device('cuda:0');props=torch.cuda.get_device_properties(self.device)
            if _physical_uuid(getattr(props,'uuid',None))!=gpu_uuid:raise ValueError('physical GPU UUID mismatch')
            torch.cuda.set_per_process_memory_fraction(cuda_memory_fraction,self.device)
        elif gpu_uuid is not None or cuda_memory_fraction is not None:raise ValueError('GPU options require CUDA')
        self.torch=torch;self.size=tuple(size);self.model=copy.deepcopy(model).to(self.device).float().eval()
        placeholder=torch.empty(0,device=self.device);modules,entries=_profile(self.model,torch.float32,self.device)
        descriptor,digest=_descriptor(modules,entries);self.guard=_Snapshot(modules,entries,placeholder);self.output_check=_output
        self.identity=dict(kind='frame_model_inference',device_id='cpu',model_sha256=digest,structural_descriptor=descriptor,
            preprocessing=dict(color='BGR_to_RGB',scale='divide_255',size=list(size),resize='bilinear_align_corners_false'))
        if self.device.type=='cuda':self.identity.update(device_id=gpu_uuid,processing_version=2,cuda=dict(device='cuda:0',physical_gpu_uuid=gpu_uuid,cuda_build=torch.version.cuda,allocator_fraction=cuda_memory_fraction,completion='selected_stream_end_event_sync'))

    def process(self,frame: '_typing.Any') -> '_typing.Any':
        """Process one caller-supplied frame in this processor implementation.

        Args:
            frame (_typing.Any): Caller-supplied frame payload for one processing step. Required.

        Returns:
            The implementation's per-step output under this explicit worker protocol.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.process(frame)

        HYSo Guide: analysis.p30
        Offline: hyso docs show analysis.p30 --lang en
        Open: hyso docs open analysis.p30 --lang en
        """
        torch=self.torch;a=time.perf_counter_ns()
        x=torch.from_numpy(frame[:,:,::-1].copy()).permute(2,0,1).unsqueeze(0).float()/255
        x=torch.nn.functional.interpolate(x,size=self.size,mode='bilinear',align_corners=False)
        b=time.perf_counter_ns()
        if self.device.type=='cuda':
            stream=torch.cuda.current_stream(self.device)
            x=x.to(self.device);transfer_end=time.perf_counter_ns()
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record(stream)
            with torch.inference_mode():out=self.model(x)
            end.record(stream);end.synchronize()
            c=time.perf_counter_ns();elapsed=float(start.elapsed_time(end))
            if not end.query() or not math.isfinite(elapsed) or elapsed<0:raise RuntimeError('CUDA frame completion unverified')
            self.output_check(out,1,device=self.device);self.guard.check(full=True)
            return [dict(name='preprocess',start_ns=a,end_ns=b),dict(name='transfer',start_ns=b,end_ns=transfer_end),
                dict(name='inference',start_ns=transfer_end,end_ns=c,cuda_event_ms=elapsed,stream_id=int(stream.cuda_stream),completed=True)]
        with torch.inference_mode():out=self.model(x)
        c=time.perf_counter_ns();self.output_check(out,1,device=torch.device('cpu'));self.guard.check(full=True)
        return [dict(name='preprocess',start_ns=a,end_ns=b),dict(name='inference',start_ns=b,end_ns=c)]

    def close(self) -> 'None':
        """Close this object's owned resources according to its lifecycle.

        Returns:
            None. The owned store/sampler resources are closed according to the existing lifecycle.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.close()

        HYSo Guide: analysis.p30
        Offline: hyso docs show analysis.p30 --lang en
        Open: hyso docs open analysis.p30 --lang en
        """
        self.guard.check(full=True)


def _file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def _prepare_capture(root, request, p, source, processor, processor_source):
    """Close both resources on any startup failure, before producer ownership."""
    import cv2
    capture = None
    try:
        binding=None
        if source.camera_binding is not None:
            from ._windows_camera import verify_binding
            binding=verify_binding(source.camera_binding,source.location)
            capture=cv2.VideoCapture(source.location,cv2.CAP_DSHOW)
            verify_binding(source.camera_binding,source.location)
        else:
            capture=cv2.VideoCapture(source.location)
        if not capture.isOpened():raise RuntimeError('capture open failed')
        if binding is not None and capture.getBackendName()!='DSHOW':
            raise ValueError('physical camera backend differs from pinned DSHOW')
        fps=capture.get(cv2.CAP_PROP_FPS);count=capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if source.kind=='video_file' and (not math.isfinite(count) or count<1):
            raise ValueError('finite recorded video frame count required')
        if p.pace_video and (source.kind!='video_file' or not math.isfinite(fps) or fps<=0):
            raise ValueError('video pacing requires declared positive FPS')
        target=min(p.max_frames,int(count)) if source.kind=='video_file' else p.max_frames
        digest=_file_hash(source.location) if source.kind=='video_file' else None
        header=dict(run_id=request['token'],pid=os.getpid(),clock_id=request['token']+':perf_counter_ns',lifecycle_version=2,
            target_frames=target,capture=dict(kind=source.kind,location=source.location,provenance=source.provenance,
                file_sha256=digest,backend=capture.getBackendName(),opencv_version=cv2.__version__,declared_fps=fps if math.isfinite(fps) and fps>0 else None,
                declared_frames=int(count) if source.kind=='video_file' else None),processor=processor.identity,processor_source=processor_source)
        if binding is not None:header['capture']['camera_binding']=binding
        journal=_Journal(root,header,64,p.max_frames*8+16)
        _atomic(root/'ready.json',dict(pid=os.getpid(),token=request['token']))
        return capture, fps, target, digest, journal
    except BaseException:
        try:
            if capture is not None:capture.release()
        finally:processor.close()
        raise


def _worker_stream(root,request):
    import cv2
    import torch
    torch.set_num_threads(1)
    p=StreamProtocol(**request['protocol']);source=StreamSource(**request['source'])
    processor,processor_source=_factory(request['recipe'])
    gpu=request['recipe'].get('gpu_uuid')
    try:
        if gpu:
            if processor.identity.get('device_id')!=gpu or _physical_uuid(getattr(torch.cuda.get_device_properties(0),'uuid',None))!=gpu:
                raise ValueError('CUDA frame processor/recipe physical identity mismatch')
        elif processor.identity.get('device_id')!='cpu' or torch.cuda.is_initialized():raise ValueError('CPU stream factory initialized CUDA')
    except BaseException:
        processor.close();raise
    capture,fps,target,digest,journal=_prepare_capture(root,request,p,source,processor,processor_source)
    condition=threading.Condition();queue=deque()
    state=dict(done=False,stop=False,failure=None,received=0,processed=0,dropped=0,closed=False,capture_closed=False)
    def event(kind,frame_id=None,**extra):
        journal.append(dict(event=kind,frame_id=frame_id,timestamp_ns=time.perf_counter_ns(),queue_depth=len(queue),**extra))
        _atomic(root/'heartbeat.json',dict(sequence=journal.count))
    def produce():
        started=time.perf_counter()
        try:
            for index in range(target):
                if p.pace_video:
                    remaining=started+index/fps-time.perf_counter()
                    if remaining>0:time.sleep(remaining)
                with condition:
                    if state['stop']:break
                a=time.perf_counter_ns();ok,frame=capture.read();b=time.perf_counter_ns()
                if not ok:raise RuntimeError('capture ended before target; decode/device failure')
                if frame.dtype.name!='uint8' or frame.ndim!=3 or frame.shape[2]!=3 or not frame.size:
                    raise ValueError('nonempty uint8 BGR frame required')
                pts=capture.get(cv2.CAP_PROP_POS_MSEC);frame_id=str(index)
                assigned=time.perf_counter_ns()
                with condition:
                    state['received']+=1
                    event('received',frame_id,read_start_ns=a,read_end_ns=b,frame_assigned_ns=assigned,
                        frame_shape=list(frame.shape),frame_nbytes=int(frame.nbytes),content_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                        media_pts_ms=pts if math.isfinite(pts) and pts>=0 else None)
                    if p.queue_policy=='drop_newest' and len(queue)==p.queue_capacity:
                        state['dropped']+=1;event('dropped',frame_id,reason='queue_full');continue
                    blocked=False
                    while len(queue)==p.queue_capacity and not state['stop']:
                        if not blocked:event('backpressure_start',frame_id);blocked=True
                        condition.wait(.1)
                    if blocked:event('backpressure_end',frame_id)
                    if state['stop']:
                        state['dropped']+=1;event('dropped',frame_id,reason='aborted');break
                    queue.append((frame_id,frame.copy()));event('enqueued',frame_id);condition.notify_all()
        except Exception as exc:
            with condition:
                state['failure']=type(exc).__name__+': '+str(exc)[:1024];event('failure',reason=state['failure'])
        finally:
            try:
                capture.release();state['capture_closed']=True
            except Exception as exc:
                with condition:state['failure']=state['failure'] or 'capture release: '+str(exc)[:1024]
            finally:
                with condition:state['done']=True;condition.notify_all()
    producer=threading.Thread(target=produce,name='hyso-real-capture',daemon=True)
    try:producer.start()
    except BaseException:
        try:capture.release()
        finally:processor.close()
        raise
    try:
        while True:
            with condition:
                while not queue and not state['done']:condition.wait(.1)
                if not queue and state['done']:break
                frame_id,frame=queue.popleft();event('dequeued',frame_id);condition.notify_all()
            a=time.perf_counter_ns()
            try:
                stages=processor.process(frame);b=time.perf_counter_ns()
                with condition:
                    state['processed']+=1;event('processed',frame_id,process_start_ns=a,process_end_ns=b,stages=stages)
            except Exception as exc:
                with condition:
                    state['stop']=True;state['failure']=type(exc).__name__+': '+str(exc)[:1024]
                    state['dropped']+=1;event('dropped',frame_id,reason='processing_error');event('failure',reason=state['failure'])
                    while queue:
                        pending_id,_=queue.popleft();state['dropped']+=1;event('dropped',pending_id,reason='aborted')
                    condition.notify_all()
                break
    finally:
        with condition:state['stop']=True;condition.notify_all()
        producer.join()  # Parent heartbeat watchdog bounds blocked native read/join.
        try:
            processor.close()
            if digest is not None and _file_hash(source.location)!=digest:raise ValueError('video content changed during run')
            state['closed']=state['capture_closed']
        except Exception as exc:state['failure']=state['failure'] or 'close: '+str(exc)[:1024]
        journal.flush()
        _atomic(root/'terminal.json',dict(status='completed' if state['failure'] is None else 'failed',reason=state['failure'],
            closed=state['closed'],received=state['received'],processed=state['processed'],dropped=state['dropped'],
            rows=journal.count,chunks=journal.chunks,last_chunk_sha256=journal.previous))
