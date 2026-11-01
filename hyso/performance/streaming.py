"""P30 real OpenCV capture/replay with explicit application-queue lifecycle evidence."""
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


@dataclass(frozen=True)
class StreamSource:
    kind: str
    location: str | int
    provenance: str

    def __post_init__(self):
        _name(self.provenance)
        if self.kind=='video_file':
            if type(self.location) is not str or not Path(self.location).is_absolute() or not Path(self.location).is_file():
                raise ValueError('existing absolute video path required')
        elif self.kind=='camera':_int(self.location,0,32)
        else:raise NotImplementedError('local recorded video or camera only; no synthetic camera adapter')


@dataclass(frozen=True)
class StreamProtocol:
    """Finite experiment plan. Defaults describe a smoke run, not hardware limits."""
    max_frames: int = 16
    queue_capacity: int = 2
    queue_policy: str = 'block'
    late_after_seconds: float = .1
    pace_video: bool = False
    startup_timeout_seconds: float = 45.
    heartbeat_timeout_seconds: float = 5.
    max_runtime_seconds: float = 30.
    max_journal_bytes: int = 128*1024**2

    def __post_init__(self):
        _int(self.max_frames,1);_int(self.queue_capacity,1);_int(self.max_journal_bytes,65536)
        for v in (self.late_after_seconds,self.startup_timeout_seconds,self.heartbeat_timeout_seconds,self.max_runtime_seconds):_number(v,.001)
        if self.queue_policy not in ('block','drop_newest') or type(self.pace_video) is not bool:raise ValueError('unknown stream queue/pacing policy')


def measure_streaming(source, processor_recipe, directory, *, protocol=StreamProtocol()):
    if type(source) is not StreamSource or type(processor_recipe) is not WorkloadRecipe or type(protocol) is not StreamProtocol:
        raise TypeError('typed stream source/recipe/protocol required')
    if processor_recipe.gpu_uuid is not None:raise NotImplementedError('P30 processing adapter CPU-only')
    p=protocol
    root=_run_owned(directory,'streaming',dict(source=asdict(source),recipe=asdict(processor_recipe),protocol=asdict(p)),
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
    source=e['request']['source']
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
    """CPU BGR uint8 -> explicit RGB resize/normalize -> actual owned model inference."""
    def __init__(self,model,*,size=(32,32)):
        import copy
        import torch
        from hyso.benchmark.runner import _profile,_descriptor,_Snapshot,_output
        if len(size)!=2:raise ValueError('explicit H,W required')
        for v in size:_int(v,1)
        self.torch=torch;self.size=tuple(size);self.model=copy.deepcopy(model).cpu().float().eval()
        placeholder=torch.empty(0);modules,entries=_profile(self.model,torch.float32,torch.device('cpu'))
        descriptor,digest=_descriptor(modules,entries);self.guard=_Snapshot(modules,entries,placeholder);self.output_check=_output
        self.identity=dict(kind='frame_model_inference',device_id='cpu',model_sha256=digest,structural_descriptor=descriptor,
            preprocessing=dict(color='BGR_to_RGB',scale='divide_255',size=list(size),resize='bilinear_align_corners_false'))

    def process(self,frame):
        torch=self.torch;a=time.perf_counter_ns()
        x=torch.from_numpy(frame[:,:,::-1].copy()).permute(2,0,1).unsqueeze(0).float()/255
        x=torch.nn.functional.interpolate(x,size=self.size,mode='bilinear',align_corners=False)
        b=time.perf_counter_ns()
        with torch.inference_mode():out=self.model(x)
        c=time.perf_counter_ns();self.output_check(out,1,device=torch.device('cpu'));self.guard.check(full=True)
        return [dict(name='preprocess',start_ns=a,end_ns=b),dict(name='inference',start_ns=b,end_ns=c)]

    def close(self):self.guard.check(full=True)


def _file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def _worker_stream(root,request):
    import cv2
    import torch
    torch.set_num_threads(1)
    p=StreamProtocol(**request['protocol']);source=StreamSource(**request['source'])
    processor,processor_source=_factory(request['recipe']);capture=cv2.VideoCapture(source.location)
    if not capture.isOpened():capture.release();processor.close();raise RuntimeError('capture open failed')
    fps=capture.get(cv2.CAP_PROP_FPS);count=capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if source.kind=='video_file' and (not math.isfinite(count) or count<1):
        capture.release();processor.close();raise ValueError('finite recorded video frame count required')
    if p.pace_video and (source.kind!='video_file' or not math.isfinite(fps) or fps<=0):
        capture.release();processor.close();raise ValueError('video pacing requires declared positive FPS')
    target=min(p.max_frames,int(count)) if source.kind=='video_file' else p.max_frames
    digest=_file_hash(source.location) if source.kind=='video_file' else None
    header=dict(run_id=request['token'],pid=os.getpid(),clock_id=request['token']+':perf_counter_ns',lifecycle_version=2,
        target_frames=target,capture=dict(kind=source.kind,location=source.location,provenance=source.provenance,
            file_sha256=digest,backend=capture.getBackendName(),opencv_version=cv2.__version__,declared_fps=fps if math.isfinite(fps) and fps>0 else None,
            declared_frames=int(count) if source.kind=='video_file' else None),processor=processor.identity,processor_source=processor_source)
    journal=_Journal(root,header,64,p.max_frames*8+16);condition=threading.Condition();queue=deque()
    state=dict(done=False,stop=False,failure=None,received=0,processed=0,dropped=0,closed=False)
    def event(kind,frame_id=None,**extra):
        journal.append(dict(event=kind,frame_id=frame_id,timestamp_ns=time.perf_counter_ns(),queue_depth=len(queue),**extra))
        _atomic(root/'heartbeat.json',dict(sequence=journal.count))
    _atomic(root/'ready.json',dict(pid=os.getpid(),token=request['token']))
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
            capture.release()
            with condition:state['done']=True;condition.notify_all()
    producer=threading.Thread(target=produce,name='hyso-real-capture',daemon=True);producer.start()
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
            state['closed']=True
        except Exception as exc:state['failure']=state['failure'] or 'close: '+str(exc)[:1024]
        journal.flush()
        _atomic(root/'terminal.json',dict(status='completed' if state['failure'] is None else 'failed',reason=state['failure'],
            closed=state['closed'],received=state['received'],processed=state['processed'],dropped=state['dropped'],
            rows=journal.count,chunks=journal.chunks,last_chunk_sha256=journal.previous))
