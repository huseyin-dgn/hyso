"""Permanent P21 unit contracts and isolated Windows spawn acceptance."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from hyso.performance._loader_event_channel import WorkerEventChannel, audit_trace_delivery
from hyso.performance._loader_trace_groups import validate_trace_groups
from hyso.performance.worker_trace_result import WorkerTraceAnalysisResult, analyze_worker_traces
from hyso.performance import DataLoaderAnalysisResult


def packet():
    row = dict(trace_id='one', worker_id=0, worker_pid=123, sample_id='0', stage='dataset_fetch',
               start_ns=10, end_ns=20, duration_ns=10, duration_ms=.00001,
               outcome='completed', clock='perf_counter_ns', clock_scope='worker_local')
    return dict(schema_version=1, trace_id='one', worker_id=0, worker_pid=123, sample_id='0',
                event_count=1, dropped_events=0, events=[row])


def audit(packets, **kw):
    options = dict(expected_fetches=1, worker_exit_codes={0:0},
                   trace_counters=dict(attempted=1, accepted=1, rejected=0))
    options.update(kw)
    return audit_trace_delivery(packets, **options)


def test_channel_snapshot_saturation_oversize_and_close():
    channel = WorkerEventChannel(max_queue_size=1, max_packet_bytes=1024)
    original = packet()
    assert channel.send_trace(original)
    original['events'].clear()
    assert not channel.send_trace(packet())
    assert channel.receive_available() == [packet()]
    assert not channel.send_trace(dict(large='x'*2048))
    assert not channel.send_trace(dict(invalid=float('nan')))
    assert channel.trace_counters() == dict(attempted=4, accepted=1, rejected=3)
    channel.close()
    channel.close()
    with pytest.raises(RuntimeError):
        channel.receive_available()


@pytest.mark.parametrize('damage', ['missing','duplicate','schema','pid','identity','duration','clock',
    'failed','dropped','event_count','empty','unhashable_stage','unhashable_identity','bad_fetch_clock'])
def test_delivery_rejects_corrupt_evidence(damage):
    p = packet()
    packets = [p]
    if damage == 'missing': packets=[]
    elif damage == 'duplicate': packets.append(copy.deepcopy(p))
    elif damage == 'schema': p['schema_version']=True
    elif damage == 'pid': p['worker_pid']=124
    elif damage == 'identity': p['events'][0]['sample_id']='foreign'
    elif damage == 'duration': p['events'][0]['duration_ns']+=1
    elif damage == 'clock': p['events'][0]['clock_scope']='parent'
    elif damage == 'failed': p['events'][0]['outcome']='failed'
    elif damage == 'dropped': p['dropped_events']=1
    elif damage == 'event_count': p['event_count']=2
    elif damage == 'empty': p['events']=[]
    elif damage == 'unhashable_stage': p['events'][0]['stage']=[]
    elif damage == 'unhashable_identity': p['events'][0]['sample_id']=[]
    elif damage == 'bad_fetch_clock': p['events'][0]['start_ns']='bad'
    assert audit(packets)['status'] == 'incomplete'


@pytest.mark.parametrize('counters', [None, {}, {'attempted':True,'accepted':1,'rejected':0},
    {'attempted':2,'accepted':1,'rejected':0}, {'attempted':1,'accepted':0,'rejected':1}])
def test_missing_or_inconsistent_counters_fail_closed(counters):
    assert audit([packet()], trace_counters=counters)['status'] == 'incomplete'


def test_valid_audit_and_variable_stages():
    assert audit([packet()])['status'] == 'complete'
    p=packet()
    for i in range(3):
        row=copy.deepcopy(p['events'][0]);row['stage']='decode';p['events'].append(row)
    p['event_count']=4
    assert audit([p])['status']=='complete'
    assert validate_trace_groups(p['events'], expected_stages={'dataset_fetch':1,'decode':3})['status']=='complete'


def test_recorder_per_trace_overflow_preserves_data():
    from torch.utils.data import TensorDataset
    import torch
    from hyso.performance._loader_dataset_trace import TracedDataset
    from hyso.performance._loader_worker_trace import WorkerTimingRecorder
    channel=WorkerEventChannel(max_queue_size=4)
    recorder=WorkerTimingRecorder(max_events=1)
    source=TensorDataset(torch.arange(3))
    traced=TracedDataset(source, recorder=recorder, channel=channel, trace_batches=True)
    for i in range(3):
        assert torch.equal(traced[i][0], source[i][0])
    packets=channel.receive_available()
    assert [p['dropped_events'] for p in packets]==[0,1,1]
    assert recorder.dropped_events==2
    assert channel.trace_counters()==dict(attempted=3,accepted=3,rejected=0)
    assert len({p['trace_id'] for p in packets})==3
    channel.close()


@pytest.fixture(scope='module')
def native_case(tmp_path_factory):
    if os.name != 'nt': pytest.skip('Windows Job acceptance requires Windows')
    root=Path(__file__).resolve().parent
    folder=tmp_path_factory.mktemp('p21-native')
    cache={}
    def execute(case):
        if case not in cache:
            output=folder/(case+'-'+str(time.time_ns())+'.json')
            command=[sys.executable, str(root/'p21_native_supervisor.py'), '--script',
                     str(root/'p21_worker_acceptance.py'), '--case',case,'--output',str(output),'--timeout','180']
            result=subprocess.run(command, capture_output=True, text=True, timeout=195)
            logs='\n'.join(p.read_text(encoding='utf-8', errors='replace') for p in folder.glob(output.stem+'.*.log'))
            assert result.returncode==0, result.stdout+result.stderr+logs
            receipt=json.loads(output.with_suffix('.receipt.json').read_text())
            assert receipt['exit_code']==0 and not receipt['timed_out']
            cache[case]=json.loads(output.read_text(encoding='utf-8'))
        return copy.deepcopy(cache[case])
    return execute


@pytest.mark.parametrize('case', ['training','saturation','overflow','exception','crash','lock_crash','shutdown_timeout','hang'])
def test_native_windows_spawn_acceptance(native_case, case):
    native_case(case)


def test_artifact_roundtrip_immutability_and_no_clobber(native_case, tmp_path):
    data=native_case('training')
    result=WorkerTraceAnalysisResult(data)
    path=tmp_path/'workers.json'
    result.to_json(path)
    assert WorkerTraceAnalysisResult(json.loads(path.read_text(encoding='utf-8'))).to_dict()==data
    legacy=copy.deepcopy(data)
    legacy.pop('evidence_sha256');legacy['analysis'].pop('association')
    assert WorkerTraceAnalysisResult(legacy).to_dict()==legacy
    with pytest.raises(FileExistsError):result.to_json(path)
    detached=result.to_dict();detached['packets'].clear()
    assert result.to_dict()==data
    with pytest.raises((AttributeError,TypeError)):result._payload='bad'
    for field in ('source_p21_sha256','source_segment_id','evidence_sha256','analysis','packets'):
        damaged=copy.deepcopy(data)
        if field=='packets': damaged[field][0]['events'][0]['duration_ns']+=1
        elif field=='analysis': damaged[field]['status']='fabricated'
        else: damaged[field]='bad'
        with pytest.raises(ValueError):WorkerTraceAnalysisResult(damaged)
    parent=DataLoaderAnalysisResult(data['source_p21'])
    with pytest.raises(ValueError):
        analyze_worker_traces(parent,data['packets'],trace_counters=None,worker_states=data['worker_states'])
    damaged=copy.deepcopy(data['packets']);damaged[0]['worker_pid']+=1
    result=analyze_worker_traces(parent,damaged,trace_counters=data['trace_counters'],worker_states=data['worker_states'])
    assert result.to_dict()['analysis']['status']=='incomplete'
