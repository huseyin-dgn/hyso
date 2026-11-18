"""P30 CPU-only analytical references and owned OpenCV fixture regressions.

Constructed event ledgers test validation/math only; generated AVI tests prove
decode/forward/cleanup, never real-camera acceptance.
"""
import copy
import csv
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from hyso.performance import RuntimeResult, StreamProtocol, StreamSource, WorkloadRecipe, measure_streaming
from hyso.performance.runtime_result import _hash, _result
from hyso.performance.streaming import _analyze_stream
from hyso.performance.stability import _read_journal


HERE=str(Path(__file__).parent.resolve())


def rechain(e):
    previous=_hash(e['header']);chunks=[]
    for i,start in enumerate(range(0,len(e['rows']),64)):
        rows=e['rows'][start:start+64]
        for j,row in enumerate(rows,start):row['index']=j
        previous=_hash(dict(schema_version=1,run_id='run',index=i,previous_sha256=previous,rows=rows))
        chunks.append(dict(name=f'chunk-{i:06d}.json',sha256=previous,rows=len(rows)))
    e['chunks']=chunks
    if e['terminal'] is not None:
        e['terminal'].update(chunks=chunks,last_chunk_sha256=previous,rows=len(e['rows']))
    return e


@pytest.fixture
def ledger():
    p=StreamProtocol(max_frames=4,queue_capacity=2,late_after_seconds=.018)
    recipe=WorkloadRecipe('runtime_support:make_frames')
    e=dict(request=dict(token='run',protocol=asdict(p),recipe=asdict(recipe),
        source=dict(kind='video_file',location='C:/offline/reference.avi',provenance='analytical_not_capture')),
        header=dict(run_id='run',pid=123,clock_id='run:perf_counter_ns',lifecycle_version=2,target_frames=4,
            capture=dict(kind='video_file',location='C:/offline/reference.avi',provenance='analytical_not_capture',
                file_sha256='a'*64,declared_frames=4,declared_fps=30),
            processor=dict(kind='frame_model_inference'),processor_source=dict(factory=recipe.factory,
                configuration={},module_sha256='b'*64)),
        controller=dict(token='run',launcher_pid=123,exit_code=0,reason=None,kill_on_close=True,
            cleanup=dict(active_processes=0)),rows=[],chunks=[],incomplete_tail=False,
        terminal=dict(status='completed',closed=True,reason=None,received=4,processed=4,dropped=0))
    def event(kind,frame,stamp,depth,**kw):
        e['rows'].append(dict(event=kind,frame_id=str(frame),timestamp_ns=int(stamp*1e6),queue_depth=depth,**kw))
    def receive(frame,stamp,depth):
        event('received',frame,stamp,depth,read_start_ns=int((stamp-2)*1e6),read_end_ns=int((stamp-1)*1e6),
            frame_assigned_ns=int((stamp-.5)*1e6),content_sha256='c'*64,media_pts_ms=frame*1000/30)
    def processed(frame,stamp,depth,start,inf_end):
        event('processed',frame,stamp,depth,process_start_ns=start*10**6,process_end_ns=(stamp-1)*10**6,
            stages=[dict(name='preprocess',start_ns=start*10**6,end_ns=(start+1)*10**6),
                    dict(name='inference',start_ns=(start+1)*10**6,end_ns=inf_end*10**6),
                    dict(name='postprocess',start_ns=inf_end*10**6,end_ns=(stamp-1)*10**6)])
    receive(0,2,0);event('enqueued',0,3,1);event('dequeued',0,4,0);processed(0,10,0,5,8)
    receive(1,12,0);event('enqueued',1,13,1);receive(2,15,1);event('enqueued',2,16,2)
    receive(3,18,2);event('backpressure_start',3,19,2);event('dequeued',1,20,1)
    event('backpressure_end',3,21,1);event('enqueued',3,22,2);processed(1,30,2,21,28)
    event('dequeued',2,31,1);processed(2,40,1,32,38);event('dequeued',3,41,0);processed(3,50,0,42,48)
    return rechain(e)


def test_independent_percentiles_rates_queue_and_late_boundary(ledger):
    a=_analyze_stream(ledger)
    assert a['status']=='completed' and a['received']==a['processed']==4 and a['pending']==a['dropped']==0
    assert a['capture_read_completion_rate_per_second']==pytest.approx(4/.017)
    assert a['completed_processing_fps']==80 and a['video_metadata_fps']==30
    assert a['inference_calls_per_active_second']==pytest.approx(4/.018)
    assert a['latency_ms']['queue_wait']['p50']==11
    assert a['latency_ms']['queue_wait']['p95']==pytest.approx(18.4)
    assert a['latency_ms']['receive_to_completion']['p50']==21.5
    assert a['latency_ms']['receive_to_completion']['p95']==pytest.approx(30.95)
    assert a['latency_ms']['inference']['p50']==5
    assert a['latency_ms']['inference']['p95']==pytest.approx(5.85)
    assert a['latency_ms']['preprocess']['mean']==1 and a['latency_ms']['postprocess']['count']==4
    assert a['backpressure_seconds']==.002
    assert a['queue_occupancy']['time_weighted_mean_depth']==pytest.approx(.84)
    assert a['late_processed']==2 and a['late_rate']==.5 and a['drop_rate']==0
    assert a['display_fps'] is None and a['display_fps_reason']
    assert a['sensor_fps'] is None and a['upstream_dropped_frames'] is None


@pytest.mark.parametrize('cut',range(1,19))
def test_every_committed_intermediate_prefix_conserves_frames(ledger,cut):
    ledger['rows']=ledger['rows'][:cut];ledger['terminal']=None;rechain(ledger)
    a=_analyze_stream(ledger)
    assert a['status']=='aborted' and a['completed_processing_fps'] is None
    assert a['received']==a['processed']+a['dropped']+a['pending']
    assert a['measurement_scope']=='committed_event_prefix_only_tail_unknown'


@pytest.mark.parametrize('damage',[
    'duplicate_id','missing_id','read_order','assignment','process_before_dispatch','stage_order',
    'missing_inference','duplicate_stage','missing_completion','queue_depth','fifo','capacity',
    'foreign_backpressure','duplicate_backpressure','unpaired_backpressure','early_backpressure_end','enqueue_while_blocked',
    'inflight_queue_full_drop','clock','recipe','target','terminal_counts'])
def test_rehashed_semantic_corruption_rejected(ledger,damage):
    rows=ledger['rows'];find=lambda event,frame:next(r for r in rows if r['event']==event and r['frame_id']==str(frame))
    if damage=='duplicate_id':find('received',1)['frame_id']='0'
    elif damage=='missing_id':find('received',1)['frame_id']='2'
    elif damage=='read_order':find('received',1)['read_start_ns']=0
    elif damage=='assignment':find('received',0)['frame_assigned_ns']=0
    elif damage=='process_before_dispatch':find('processed',0)['process_start_ns']=0
    elif damage=='stage_order':find('processed',0)['stages'][1]['start_ns']=0
    elif damage=='missing_inference':find('processed',0)['stages'].pop(1)
    elif damage=='duplicate_stage':find('processed',0)['stages'][1]['name']='preprocess'
    elif damage=='missing_completion':rows.pop()
    elif damage=='queue_depth':find('enqueued',0)['queue_depth']=0
    elif damage=='fifo':find('dequeued',1)['frame_id']='2'
    elif damage=='capacity':ledger['request']['protocol']['queue_capacity']=1
    elif damage=='foreign_backpressure':find('backpressure_start',3)['frame_id']='foreign'
    elif damage=='duplicate_backpressure':rows.insert(10,copy.deepcopy(find('backpressure_start',3)))
    elif damage=='unpaired_backpressure':rows.remove(find('backpressure_start',3))
    elif damage=='early_backpressure_end':
        rows[10],rows[11]=dict(rows[11],timestamp_ns=20*10**6),dict(rows[10],timestamp_ns=21*10**6)
    elif damage=='enqueue_while_blocked':rows.remove(find('backpressure_end',3))
    elif damage=='inflight_queue_full_drop':
        ledger['request']['protocol']['queue_policy']='drop_newest'
        rows[9]=dict(rows[9],event='dropped',frame_id='0',reason='queue_full')
    elif damage=='clock':ledger['header']['clock_id']='foreign'
    elif damage=='recipe':ledger['header']['processor_source']['configuration']={'delay':1}
    elif damage=='target':ledger['header']['target_frames']=3
    elif damage=='terminal_counts':ledger['terminal']['processed']=3
    with pytest.raises(ValueError):_analyze_stream(rechain(ledger))


def test_drop_newest_only_new_frame_and_no_completion_is_invented(ledger):
    ledger['request']['protocol']['queue_policy']='drop_newest'
    rows=ledger['rows'];rows[9]=dict(rows[9],event='dropped',reason='queue_full')
    # Remove block end, enqueue(3), dequeue(3), completion(3); adjust actual FIFO evidence.
    ledger['rows']=[r for i,r in enumerate(rows) if i not in (11,12,16,17)]
    for r in ledger['rows']:
        if r['timestamp_ns']>=30*10**6:r['queue_depth']-=1
    ledger['terminal'].update(processed=3,dropped=1)
    a=_analyze_stream(rechain(ledger))
    assert (a['received'],a['processed'],a['dropped'],a['pending'])==(4,3,1,0)
    assert a['drop_rate']==.25 and a['backpressure_seconds']==0


def test_limits_are_finite_plan_fields_without_hardware_caps():
    p=StreamProtocol(max_frames=100001,queue_capacity=100,max_runtime_seconds=7200,
        startup_timeout_seconds=180,heartbeat_timeout_seconds=90)
    assert p.max_frames==100001 and p.queue_capacity==100
    assert StreamProtocol().max_frames==16  # Existing explicit smoke default stays compatible.
    for kwargs in ({'max_frames':0},{'queue_capacity':0},{'max_frames':True},{'max_runtime_seconds':float('inf')},
                   {'max_runtime_seconds':float('nan')},{'max_journal_bytes':0}):
        with pytest.raises(ValueError):StreamProtocol(**kwargs)


def test_lossless_json_csv_and_legacy_analysis_reopen(ledger,tmp_path):
    result=_result('P30',ledger)
    for suffix in ('json','csv'):
        path=tmp_path/('new.'+suffix);getattr(result,'to_'+suffix)(path)
        assert getattr(RuntimeResult,'from_'+suffix)(path).to_dict()==result.to_dict()
        with pytest.raises(FileExistsError):getattr(result,'to_'+suffix)(path)
    old=copy.deepcopy(ledger);old['header'].pop('lifecycle_version');old['request']['protocol'].pop('max_journal_bytes')
    legacy=_result('P30',rechain(old));assert 'latency_ms' not in legacy.to_dict()['analysis']
    assert RuntimeResult(legacy.to_dict()).to_dict()==legacy.to_dict()
    result.to_csv(tmp_path/'tamper.csv')
    with (tmp_path/'tamper.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    rows[0]['source_sha256']='0'*64
    with (tmp_path/'tamper.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError):RuntimeResult.from_csv(tmp_path/'tamper.csv')


@pytest.fixture(scope='module')
def synthetic_video(tmp_path_factory):
    import cv2
    import numpy as np
    path=tmp_path_factory.mktemp('p30-synthetic')/'fixture.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),25,(64,48))
    assert writer.isOpened()
    for i in range(20):writer.write(np.full((48,64,3),i*10,dtype=np.uint8))
    writer.release()
    return StreamSource('video_file',str(path),'generated_regression_fixture_not_real_video_acceptance')


@pytest.mark.parametrize('policy',['block','drop_newest'])
def test_actual_decode_forward_reopen_and_resource_ownership(policy,synthetic_video,tmp_path):
    recipe=WorkloadRecipe('runtime_support:make_frames',dict(delay=.02),search_path=HERE)
    root=tmp_path/'journal';r=measure_streaming(synthetic_video,recipe,root,
        protocol=StreamProtocol(max_frames=20,queue_capacity=1,queue_policy=policy,late_after_seconds=.001))
    d=r.to_dict();a=d['analysis'];c=d['evidence']['controller']
    assert a['status']=='completed' and a['received']==20 and a['pending']==0
    assert a['latency_ms']['inference']['count']==a['processed']>0
    assert a['latency_ms']['postprocess']['status']=='unsupported'
    assert a['latency_ms']['postprocess']['reason']=='postprocessing_not_instrumented'
    assert a['dropped']>0 if policy=='drop_newest' else a['dropped']==0
    assert c['memory_limits']==dict(process_memory_bytes=None,job_memory_bytes=None)
    assert c['resource_monitor']['samples']>0 and c['cleanup']['active_processes']==0
    assert _result('P30',_read_journal(root)).to_dict()['analysis']==a
    r.to_json(tmp_path/'r.json');r.to_csv(tmp_path/'r.csv')
    assert RuntimeResult.from_json(tmp_path/'r.json').to_dict()==d
    assert RuntimeResult.from_csv(tmp_path/'r.csv').to_dict()==d
    (root/'terminal.json').rename(root/'terminal.backup')
    assert _analyze_stream(_read_journal(root))['status']=='aborted'


@pytest.mark.parametrize('mode',['crash','child_hang','close_exception','close_hang'])
def test_controlled_worker_failure_and_process_tree_cleanup(mode,synthetic_video,tmp_path):
    r=measure_streaming(synthetic_video,WorkloadRecipe('runtime_support:make_frames',dict(mode=mode),search_path=HERE),
        tmp_path/'journal',protocol=StreamProtocol(max_frames=4,queue_capacity=1,heartbeat_timeout_seconds=.6))
    d=r.to_dict();a=d['analysis'];c=d['evidence']['controller']
    assert a['status']=='aborted' and a['processed_per_second'] is None
    assert c['cleanup']['active_processes']==0
    if mode=='crash':assert c['exit_code']==17
    if mode in ('child_hang','close_hang'):assert c['reason']=='heartbeat_timeout'
    if mode=='child_hang':assert c['cleanup']['total_processes']>=2


def test_read_failure_after_actual_decode_is_aborted(synthetic_video,tmp_path):
    r=measure_streaming(synthetic_video,WorkloadRecipe('runtime_support:make_frames',dict(mode='capture_failure'),search_path=HERE),
        tmp_path/'journal',protocol=StreamProtocol(max_frames=4))
    a=r.to_dict()['analysis']
    assert a['status']=='aborted' and a['received']==2 and a['pending']==0
    assert a['processed_per_second'] is None and any('capture ended before target' in s for s in a['failures'])


def test_os_low_memory_stops_job_without_allocation_stress(monkeypatch,synthetic_video,tmp_path):
    from hyso.performance._stream_resources import _StreamResources
    def pressure(self,job):
        self.summary['low_memory_observed']=True
        return True
    monkeypatch.setattr(_StreamResources,'sample',pressure)
    r=measure_streaming(synthetic_video,WorkloadRecipe('runtime_support:make_frames',search_path=HERE),tmp_path/'journal')
    c=r.to_dict()['evidence']['controller']
    assert c['reason']=='os_low_memory' and c['cleanup']['active_processes']==0
    a=r.to_dict()['analysis']
    assert a['display_fps'] is None and a['display_fps_reason']
    r.to_csv(tmp_path/'pressure.csv')
    assert RuntimeResult.from_csv(tmp_path/'pressure.csv').to_dict()==r.to_dict()


def test_duration_timeout_is_distinct_from_heartbeat(synthetic_video,tmp_path):
    r=measure_streaming(synthetic_video,WorkloadRecipe('runtime_support:make_frames',dict(mode='hang'),search_path=HERE),
        tmp_path/'journal',protocol=StreamProtocol(max_frames=4,max_runtime_seconds=.2,heartbeat_timeout_seconds=2))
    d=r.to_dict()
    assert d['analysis']['status']=='aborted' and d['evidence']['controller']['reason']=='duration_timeout'
    assert d['evidence']['controller']['cleanup']['active_processes']==0


def test_frame_larger_than_old_byte_ceiling_is_processed(tmp_path):
    import cv2
    import numpy as np
    path=tmp_path/'large-fixture.avi'
    frame=np.zeros((2304,2560,3),dtype=np.uint8)
    assert frame.nbytes>16*1024**2
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),25,(2560,2304))
    assert writer.isOpened();writer.write(frame);writer.release()
    r=measure_streaming(StreamSource('video_file',str(path),'large_generated_regression_frame_not_camera'),
        WorkloadRecipe('runtime_support:make_frames',dict(delay=0),search_path=HERE),tmp_path/'journal',
        protocol=StreamProtocol(max_frames=1,queue_capacity=1))
    d=r.to_dict();assert d['analysis']['status']=='completed' and d['analysis']['processed']==1
    received=next(row for row in d['evidence']['rows'] if row['event']=='received')
    assert received['frame_nbytes']==frame.nbytes and received['frame_shape']==list(frame.shape)
