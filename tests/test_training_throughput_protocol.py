"""Deterministic P22 arithmetic and historical P19 wire compatibility."""
from fixture_support import load_fixture
import copy
import json
from pathlib import Path
import pytest
from hyso.performance import (TrainingPerformanceResult, TrainingStageProtocol,
                              TrainingThroughputResult, analyze_training_throughput)


def source():
    data = load_fixture('training-p19-schema1')
    # Deterministic clocks: epoch 1 train=1 second, epoch 2 train=3 seconds.
    for i, e in enumerate(data['epochs']):
        start = e['start_ns']
        scale = (i * 2 + 1) * 1e9 / (e['train_end_ns'] - start)
        def clock(value):
            return None if value is None else int((value-start)*scale) + i*100000000000
        for row in data['observations']:
            if row['epoch_id'] == e['epoch_id']:
                row['start_ns'], row['end_ns'] = clock(row['start_ns']), clock(row['end_ns'])
                row['host_ms'] = (row['end_ns']-row['start_ns'])/1e6
        for key in ('start_ns','train_start_ns','train_end_ns','end_ns','validation_start_ns','validation_end_ns'):
            e[key] = clock(e[key])
        e['training_s'] = (e['train_end_ns']-e['train_start_ns'])/1e9
        e['total_s'] = (e['end_ns']-e['start_ns'])/1e9
    return data


def test_old_p19_deserialize_and_exact_ratio_of_sums():
    d = source()
    result = analyze_training_throughput(TrainingPerformanceResult(d)).to_dict()
    assert result['segment']['completed'] == dict(samples=10,microbatches=6,windows=4,attempts=4,applied=4,skipped=0)
    assert result['segment']['training_wall_seconds'] == 4
    assert result['segment']['rates']['samples_per_second']['value'] == 2.5
    assert result['segment']['rates']['applied_updates_per_second']['value'] == 1
    assert result['segment']['rates']['samples_per_second']['value'] != sum(e['rates']['samples_per_second']['value'] for e in result['epochs'])/2
    assert TrainingThroughputResult(result).to_dict() == result


@pytest.mark.parametrize('values',[dict(max_records=True),dict(max_records=0),dict(max_records=1000001)])
def test_invalid_stage_protocol(values):
    with pytest.raises(ValueError): TrainingStageProtocol(**values)


def test_pure_analysis_does_not_train_or_read_clock(monkeypatch):
    from hyso.training import Trainer
    import hyso.performance.training as module
    def forbidden(*args,**kwargs): raise AssertionError('analysis executed work')
    monkeypatch.setattr(Trainer,'fit',forbidden)
    monkeypatch.setattr(module.time,'perf_counter_ns',forbidden)
    analyze_training_throughput(TrainingPerformanceResult(source()))


def test_zero_duration_is_null():
    d=source()
    for e in d['epochs']:
        for r in d['observations']:
            if r['epoch_id']==e['epoch_id']:
                r['start_ns']=r['end_ns']=e['start_ns']; r['host_ms']=0.
        e['train_end_ns']=e['train_start_ns']; e['training_s']=0.
    r=analyze_training_throughput(TrainingPerformanceResult(d)).to_dict()
    assert r['segment']['rates']['samples_per_second']==dict(value=None,unit='samples/s',status='unavailable',reason='nonpositive_training_duration')


@pytest.mark.parametrize('field',['rate','hash','source_count','protocol','association'])
def test_tampered_throughput_rejected(field):
    d=analyze_training_throughput(TrainingPerformanceResult(source())).to_dict()
    if field=='rate': d['segment']['rates']['samples_per_second']['value']+=1
    elif field=='hash': d['source_sha256']='0'*64
    elif field=='source_count': d['source_p19']['epochs'][0]['samples']+=1
    elif field=='protocol': d['protocol']['validation']='included'
    else: d['source_segment_id']='wrong'
    with pytest.raises(ValueError): TrainingThroughputResult(d)
