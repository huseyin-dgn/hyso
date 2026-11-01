"""P30--P33 actual owned Windows workers and independent analytical references."""
import copy
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import time

import pytest
import torch
from torch import nn

from hyso.performance import (RuntimeResult,WorkloadRecipe,StabilityProtocol,InferenceWorkload,measure_stability,
    reopen_stability,StreamSource,StreamProtocol,measure_streaming,RetentionTracker,analyze_memory_trend,analyze_energy)
from hyso.performance.runtime_result import _derive,_hash
from hyso.performance.memory_trend import _fit
from hyso.performance.energy import _integral,_parse_power,observe_nvidia_power
from hyso.checkpoint import _rng_state
from trainer_replay_support import equal


HERE=str(Path(__file__).parent.resolve())


def bounded_stability(**kwargs):
    return StabilityProtocol(process_memory_bytes=4*1024**3,job_memory_bytes=6*1024**3,max_processes=4,**kwargs)


def recipe(mode='normal',delay=.005):
    return WorkloadRecipe('runtime_support:make_cpu',dict(mode=mode,delay=delay),search_path=HERE)


def save(result,name):
    directory=os.environ.get('HYSO_RUNTIME_EVIDENCE')
    if directory:
        root=Path(directory);root.mkdir(exist_ok=True,parents=True)
        result.to_json(root/(name+'.json'));result.to_csv(root/(name+'.csv'))
    return result


@pytest.fixture(scope='module')
def normal(tmp_path_factory):
    root=tmp_path_factory.mktemp('runtime-normal')/'journal'
    result=measure_stability(recipe(),root,protocol=bounded_stability(duration_seconds=.25,interval_seconds=.02,chunk_rows=2))
    return save(result,'p31-cpu-smoke'),root


@pytest.fixture(scope='module')
def retained(tmp_path_factory):
    return save(measure_stability(recipe('retain'),tmp_path_factory.mktemp('runtime-retained')/'journal',
        protocol=bounded_stability(duration_seconds=.3,interval_seconds=.02,chunk_rows=2)),'p31-retained')


@pytest.fixture(scope='module')
def released(tmp_path_factory):
    return save(measure_stability(recipe('release'),tmp_path_factory.mktemp('runtime-release')/'journal',
        protocol=bounded_stability(duration_seconds=.2,interval_seconds=.02,chunk_rows=2)),'p31-released')


def test_real_cpu_duration_chunks_reopen(normal):
    result,root=normal;d=result.to_dict();a=d['analysis']
    assert a['status']=='completed',d
    assert a['duration_seconds']>=.25 and a['completed']>0 and a['steps']>0
    assert a['health']['rss_samples']>=3 and a['health']['thermal_samples']==0
    assert a['health']['cpu_samples']>=3 and a['cpu_timeline'] and a['performance_timeline']
    assert len(d['evidence']['chunks'])>=2
    assert reopen_stability(root).to_dict()['evidence']==d['evidence']
    assert d['evidence']['controller']['cleanup']['active_processes']==0


@pytest.mark.parametrize('mode',['exception','hang','crash','close_exception','close_hang'])
def test_failure_timeout_and_bounded_process_cleanup(mode,tmp_path):
    started=time.monotonic()
    r=measure_stability(recipe(mode),tmp_path/'journal',protocol=bounded_stability(
        duration_seconds=.12,interval_seconds=.02,step_timeout_seconds=.4,chunk_rows=1))
    a=r.to_dict()['analysis'];assert a['status']=='aborted' and a['completed_per_second'] is None
    assert time.monotonic()-started<50
    assert r.to_dict()['evidence']['controller']['cleanup']['active_processes']==0
    if 'hang' in mode:assert a['reason']=='heartbeat_timeout'


def test_real_weakref_retention_release_and_warmup(retained,released):
    r=save(analyze_memory_trend(retained,warmup_seconds=.03),'p32-retained').to_dict()
    assert r['analysis']['classification']=='tracked_live_reference_growth_observed'
    assert r['analysis']['retention']['live_reference_growth'] and not r['analysis']['leak_proven']
    r=save(analyze_memory_trend(released),'p32-released').to_dict()
    assert r['analysis']['retention']['released_object_ids']
    assert not r['analysis']['leak_proven']
    r=analyze_memory_trend(released,warmup_seconds=999).to_dict()['analysis']
    assert r['status']=='insufficient_samples' and r['rss']['slope_bytes_per_second'] is None


def test_independent_ols_slope_uncertainty_reference():
    # x=0,1,2,3; y=2,3,5,8: exact slope2, residual SSE1, Sxx5.
    r=_fit([(i*10**9,y) for i,y in enumerate([2,3,5,8])])
    assert r['slope_bytes_per_second']==2
    assert r['standard_error']==pytest.approx(math.sqrt(1/2/5))
    assert r['confidence_interval'] is None
    plateau=_fit([(i*10**9,42) for i in range(4)])
    assert plateau['slope_bytes_per_second']==0 and plateau['standard_error']==0
    assert _fit([(0,0),(1,1)])['status']=='insufficient_samples'
    with pytest.raises(ValueError):_fit([(0,0),(1,1),(1,2)])


def test_tracker_never_retains_and_rejects_duplicates():
    tracker=RetentionTracker();t=torch.ones(12);tracker.watch(t,label='owned')
    with pytest.raises(ValueError):tracker.watch(t,label='alias')
    assert tracker.snapshot()['logical_live_bytes']==48
    del t;gc.collect()
    assert tracker.snapshot()['alive_objects']==0


def test_cpu_model_state_modes_rng_preserved(monkeypatch):
    m=nn.Sequential(nn.Flatten(),nn.Linear(12,2));m.train();m[1].eval()
    before=copy.deepcopy(m.state_dict());modes=[x.training for x in m.modules()];rng=_rng_state()
    work=InferenceWorkload(m,torch.ones(2,3,2,2));assert work.step()==2;work.close()
    equal(m.state_dict(),before);equal(_rng_state(),rng)
    assert [x.training for x in m.modules()]==modes
    monkeypatch.delenv('HYSO_RUNTIME_ADMITTED_PID',raising=False)
    initialized=torch.cuda.is_initialized()
    with pytest.raises(ValueError,match='admitted owned'):
        InferenceWorkload(m,torch.ones(2,3,2,2),device='cuda:0',gpu_uuid='GPU-test')
    assert torch.cuda.is_initialized()==initialized


@pytest.fixture(scope='module')
def video_source(tmp_path_factory):
    real=os.environ.get('HYSO_REAL_VIDEO')
    if real:return StreamSource('video_file',str(Path(real).resolve()),'OpenCV-4.10.0-vtest-recorded-real-video')
    import cv2
    import numpy as np
    path=tmp_path_factory.mktemp('synthetic-video')/'accounting.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,48))
    assert writer.isOpened()
    for i in range(16):writer.write(np.full((48,64,3),i*10,dtype=np.uint8))
    writer.release()
    return StreamSource('video_file',str(path),'synthetic-test-fixture-not-native-camera-acceptance')


@pytest.fixture(scope='module',params=['block','drop_newest'])
def stream(request,video_source,tmp_path_factory):
    policy=request.param
    r=measure_streaming(video_source,WorkloadRecipe('runtime_support:make_frames',dict(delay=.05),search_path=HERE),
        tmp_path_factory.mktemp('stream-'+policy)/'journal',protocol=StreamProtocol(max_frames=12,queue_capacity=1,
        queue_policy=policy,late_after_seconds=.01))
    return save(r,'p30-'+policy)


def test_actual_video_capture_processing_conservation_drop_backpressure(stream):
    d=stream.to_dict();a=d['analysis'];assert a['status']=='completed',d
    assert a['received']==12 and a['pending']==0 and a['processed']>0
    assert a['received']==a['processed']+a['dropped']+a['pending']
    assert a['max_queue_depth']==1 and a['late_processed']>0
    if d['evidence']['request']['protocol']['queue_policy']=='block':
        assert a['dropped']==0 and a['backpressure_seconds']>0
    else:assert a['dropped']>0


def test_stream_malformed_lifecycle_rejected(stream):
    e=stream.to_dict()['evidence'];e['rows'][0]['frame_id']='foreign'
    with pytest.raises(ValueError):_derive('P30',e)


@pytest.mark.parametrize('mode',['exception','hang'])
def test_stream_failure_bounded_cleanup(mode,video_source,tmp_path):
    r=measure_streaming(video_source,WorkloadRecipe('runtime_support:make_frames',dict(mode=mode),search_path=HERE),
        tmp_path/'failed-stream',protocol=StreamProtocol(max_frames=4,queue_capacity=1,heartbeat_timeout_seconds=.5))
    d=r.to_dict();assert d['analysis']['status']=='aborted'
    assert not d['analysis']['terminal_accounting_complete']
    assert d['evidence']['controller']['cleanup']['active_processes']==0


def test_bad_capture_no_success(tmp_path):
    bad=tmp_path/'bad.avi';bad.write_bytes(b'not video')
    r=measure_streaming(StreamSource('video_file',str(bad),'invalid-negative-test'),
        WorkloadRecipe('runtime_support:make_frames',search_path=HERE),tmp_path/'journal')
    assert r.to_dict()['analysis']['status']=='aborted'


def test_reopen_gap_partial_and_actual_memory_classification(normal,tmp_path):
    import shutil
    root=tmp_path/'partial';shutil.copytree(normal[1],root)
    # Loss of terminal commit is an incomplete prefix, never a complete run.
    (root/'terminal.json').rename(root/'interrupted-terminal.backup')
    assert reopen_stability(root).to_dict()['analysis']['status']=='aborted'
    # A missing middle chunk is rejected rather than silently joined.
    chunk=sorted(root.glob('chunk-*.json'))[0];chunk.rename(root/'missing-chunk.backup')
    with pytest.raises(ValueError):reopen_stability(root)
    plateau=analyze_memory_trend(normal[0],plateau_range_bytes=10**9,slope_tolerance_bytes_per_second=10**12)
    assert plateau.to_dict()['analysis']['classification']=='sampled_plateau'


@pytest.mark.parametrize('field,value',[('unit','mW'),('clock_id','other'),('device_id','GPU-other'),('timestamp_ns',99)])
def test_energy_clock_unit_device_rejection(field,value):
    from hyso.performance.energy import _validate_power_sample
    p=_parse_power('GPU-test, PCI, GPU, 10, 50, Not Active',device_id='GPU-test',clock_id='clock',start_ns=1,end_ns=2)
    p[field]=value
    with pytest.raises(ValueError):_validate_power_sample(p,'GPU-test','clock')


def test_power_n_a_and_timeout_are_unsupported(monkeypatch):
    from hyso.performance import energy
    p=_parse_power('GPU-test, PCI, GPU, N/A, 50, Not Active',device_id='GPU-test',clock_id='clock',start_ns=1,end_ns=2)
    assert p['status']=='unsupported' and p['watts'] is None and p['temperature_c']==50
    monkeypatch.setattr(energy.shutil,'which',lambda x:'tool')
    def timeout(*args,**kwargs):raise subprocess.TimeoutExpired('sensor',.1)
    monkeypatch.setattr(energy.subprocess,'run',timeout)
    assert observe_nvidia_power('GPU-test',clock_id='clock')['status']=='unsupported'


def test_independent_energy_trapezoid_clipping_gaps_and_boundaries():
    points=[dict(timestamp_ns=t*10**9,watts=w) for t,w in [(0,10),(1,20),(3,40)]]
    energy,reason=_integral(points,0,3*10**9,2*10**9)
    assert reason is None and energy==75 # 15*1 + 30*2
    assert _integral(points,500000000,2*10**9,2*10**9)[0]==pytest.approx(33.75)
    assert _integral(points,0,4*10**9,2*10**9)[1]=='unbracketed_window'
    assert _integral(points,0,3*10**9,10**9)[1]=='sensor_sample_gap'
    with pytest.raises(ValueError):_integral([points[0],points[0]],0,1,10)


@pytest.mark.parametrize('field,value',[('power','nan'),('power','-1'),('power','inf'),('device','GPU-other')])
def test_power_sensor_units_identity_invalid(field,value):
    device=value if field=='device' else 'GPU-test';power=value if field=='power' else '10'
    with pytest.raises(ValueError):_parse_power(f'{device}, PCI, GPU, {power}, 50, Not Active',
        device_id='GPU-test',clock_id='clock',start_ns=1,end_ns=2)


def test_sensor_unavailable_not_fake_energy(monkeypatch,normal):
    from hyso.performance import energy
    monkeypatch.setattr(energy.shutil,'which',lambda x:None)
    assert observe_nvidia_power('GPU-test',clock_id='clock')['status']=='unsupported'
    r=save(analyze_energy(normal[0]),'p33-cpu-unsupported').to_dict()['analysis']
    assert r['status']=='unsupported' and r['gross_joules'] is None


@pytest.mark.parametrize('damage',['chunks','source','terminal','duration','cleanup'])
def test_stability_evidence_rejects_tamper(normal,damage):
    e=normal[0].to_dict()['evidence']
    if damage=='chunks':e['chunks'][0]['sha256']='0'*64
    elif damage=='source':e['header']['source']['factory']='foreign:factory'
    elif damage=='terminal':e['terminal']['completed']+=1
    elif damage=='duration':e['request']['protocol']['duration_seconds']=100
    else:e['controller']['cleanup']['active_processes']=1
    with pytest.raises(ValueError):_derive('P31',e)


def test_reopen_deleted_modified_and_partial_chunk(normal,tmp_path):
    import shutil
    root=tmp_path/'copy';shutil.copytree(normal[1],root)
    chunk=next(root.glob('chunk-*.json'));chunk.write_text('{}')
    with pytest.raises((ValueError,KeyError)):reopen_stability(root)


@pytest.mark.parametrize('feature',['P30','P31','P32','P33'])
def test_immutable_roundtrip_csv_json_no_clobber_tamper(feature,stream,normal,retained,tmp_path):
    r={'P30':stream,'P31':normal[0],'P32':analyze_memory_trend(retained),'P33':analyze_energy(normal[0])}[feature]
    d=r.to_dict();r.to_json(tmp_path/'r.json');r.to_csv(tmp_path/'r.csv')
    assert RuntimeResult(json.loads((tmp_path/'r.json').read_text())).to_dict()==d
    for method,path in ((r.to_json,tmp_path/'r.json'),(r.to_csv,tmp_path/'r.csv')):
        with pytest.raises(FileExistsError):method(path)
    with pytest.raises((AttributeError,TypeError)):r._payload='changed'
    detached=r.to_dict();detached['evidence'].clear();assert r.to_dict()==d
    for key in ('analysis','evidence'):
        bad=copy.deepcopy(d);bad[key]={}
        with pytest.raises(ValueError):RuntimeResult(bad)
    bad=copy.deepcopy(d);bad['analysis']['status']='invented'
    bad['source_sha256']=_hash(dict(feature_id=bad['feature_id'],evidence=bad['evidence']))
    with pytest.raises(ValueError):RuntimeResult(bad)


def test_protocol_budgets_and_no_clobber_run(normal):
    for kwargs in (dict(duration_seconds=0),dict(interval_seconds=0),dict(duration_seconds=86401),dict(max_samples=3)):
        with pytest.raises(ValueError):StabilityProtocol(**kwargs)
    with pytest.raises(FileExistsError):measure_stability(recipe(),normal[1],protocol=bounded_stability(duration_seconds=.1,interval_seconds=.02))
    with pytest.raises(ValueError):StreamProtocol(queue_capacity=0)
    with pytest.raises(NotImplementedError):StreamSource('synthetic','anything','not-camera')


@pytest.mark.skipif(not os.environ.get('HYSO_NATIVE_GPU_UUID'),reason='NOT_RUN: explicit native physical GPU budget not enabled')
def test_native_gpu_power_thermal_energy(tmp_path):
    gpu=os.environ['HYSO_NATIVE_GPU_UUID']
    r=save(measure_stability(WorkloadRecipe('runtime_support:make_gpu',dict(gpu_uuid=gpu,cuda_memory_fraction=.1),search_path=HERE,gpu_uuid=gpu),
        tmp_path/'native-gpu',protocol=bounded_stability(duration_seconds=2.,idle_seconds=.9,interval_seconds=.2,chunk_rows=2)),
        'p31-native-gpu-smoke')
    assert r.to_dict()['analysis']['status']=='completed',r.to_dict()
    result=save(analyze_energy(r,max_gap_seconds=2.),'p33-native-gpu-energy').to_dict()['analysis']
    assert result['status']=='derived_from_measured_power_samples',result
    assert result['gross_joules']>0 and result['joules_per_completed_image']>0
    assert result['device_id']==gpu and result['idle_sample_count']>=2
    assert r.to_dict()['analysis']['health']['thermal_samples']>=3


@pytest.mark.skipif(not os.environ.get('HYSO_NATIVE_GPU_UUID'),reason='NOT_RUN: explicit native physical GPU budget not enabled')
def test_native_allocator_cache_growth_is_not_leak(tmp_path):
    gpu=os.environ['HYSO_NATIVE_GPU_UUID']
    r=save(measure_stability(WorkloadRecipe('runtime_support:make_gpu_cache',dict(gpu_uuid=gpu),search_path=HERE,gpu_uuid=gpu),
        tmp_path/'cache',protocol=bounded_stability(duration_seconds=.8,interval_seconds=.1,chunk_rows=2)),
        'p31-native-cache-probe')
    assert r.to_dict()['analysis']['status']=='completed',r.to_dict()['analysis']
    result=save(analyze_memory_trend(r,plateau_range_bytes=1024**2),'p32-native-cache-growth').to_dict()['analysis']
    assert result['classification']=='allocator_cache_growth_observed',result
    assert result['allocated']['delta_bytes']==0 and result['reserved']['delta_bytes']>1024**2
    assert not result['leak_proven']


def test_atomic_publication_transient_and_bounded_failure(monkeypatch,tmp_path):
    from hyso.performance import stability
    original=stability.os.replace;calls=[]
    def transient(a,b):
        calls.append(1)
        if len(calls)==1:raise PermissionError('transient Windows sharing lock')
        return original(a,b)
    monkeypatch.setattr(stability.os,'replace',transient)
    stability._atomic(tmp_path/'receipt.json',dict(sequence=1))
    assert len(calls)==2 and json.loads((tmp_path/'receipt.json').read_text())==dict(sequence=1)
    def denied(*args):raise PermissionError('persistent denial')
    monkeypatch.setattr(stability.os,'replace',denied);started=time.monotonic()
    with pytest.raises(PermissionError):stability._atomic(tmp_path/'denied.json',dict(sequence=2))
    assert time.monotonic()-started<2


def test_real_sampling_gap_is_reported_not_interpolated(tmp_path):
    r=save(measure_stability(recipe(delay=.07),tmp_path/'gap',protocol=bounded_stability(
        duration_seconds=.21,interval_seconds=.01,chunk_rows=1)),'p31-real-sampling-gap')
    a=r.to_dict()['analysis'];assert a['status']=='completed'
    assert a['gap_count']>=2 and all(g['gap_ns']>20000000 for g in a['gaps'])
