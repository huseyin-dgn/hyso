"""P03/P04 IDs, clock evidence, bounded queue and P05 actual fresh interpreters."""
from dataclasses import FrozenInstanceError
import csv
import gc
import hashlib
import json
import os
import random
from pathlib import Path
import re
import threading
import weakref

import pytest
import numpy as np
import torch
from torch import nn

from hyso.performance import (PipelineSample, PipelineStage, PipelineProtocol, PipelineResult, StageTimingResult,
    ColdStartRecipe, ColdStartProtocol, ColdStartResult, measure_pipeline, analyze_stages, measure_stages,
    measure_cold_start, measure_footprint)
from hyso.performance import pipeline as engine
from hyso.benchmark.runner import StateMutationError
from test_benchmark_state_safety import prepared, state, equal


def source():
    ids=iter(range(10000))
    def read():
        index=next(ids)
        return PipelineSample(str(index),torch.full((2,1,5,7),float(index)))
    return read


@pytest.fixture
def pipeline_result():
    return measure_pipeline(nn.Identity(),source(),protocol=PipelineProtocol(samples=3,warmup_samples=1,queue_depth=2))


def test_real_preprocess_postprocess_delivery_same_frame():
    delivered=[]; reads=[]; calls=[]
    read=source()
    def src():
        item=read(); reads.append(item.sample_id); return item
    def pre(frame):
        calls.append(('pre',frame.sample_id))
        return PipelineSample(frame.sample_id,torch.nn.functional.interpolate(frame.payload,size=(3,4))+1)
    def post(frame):
        calls.append(('post',frame.sample_id))
        return PipelineSample(frame.sample_id,frame.payload.mean((2,3),keepdim=True))
    def deliver(frame):
        assert torch.equal(frame.payload,torch.full_like(frame.payload,float(frame.sample_id)+1))
        delivered.append(frame.sample_id); return frame
    result=measure_pipeline(nn.Identity(),src,preprocess=PipelineStage('preprocess',pre),
        postprocess=PipelineStage('postprocess',post),deliver=PipelineStage('deliver',deliver),
        protocol=PipelineProtocol(samples=3,repeats=2,warmup_samples=1,queue_depth=2))
    data=result.to_dict(); rows=data['raw_evidence']['samples']
    assert len(reads)==len(delivered)==8 and len(calls)==16
    assert [r['sample_id'] for r in rows]==['1','2','3','5','6','7']
    assert data['actual']==dict(completed=6,dropped=0,pending=0,warmup_samples=2,forward_count=8,max_pending=2)
    assert all(r['delivery_completion_ns']>=r['postprocess_completion_ns'] for r in rows)
    stage=analyze_stages(result).to_dict()
    assert stage['measurements']['preprocess']['count']==6
    assert stage['measurements']['postprocess']['device_mean_ms'] is None
    assert data['measurements']['p99_ms']['value'] is None


def test_known_host_timestamps_are_not_stage_sum(monkeypatch):
    ticks=iter(range(100,190,10)); monkeypatch.setattr(engine,'_clock',lambda:next(ticks))
    data=measure_pipeline(nn.Identity(),source(),protocol=PipelineProtocol(samples=1,warmup_samples=0,queue_depth=1)).to_dict()
    row=data['raw_evidence']['samples'][0]
    assert row['source_start_ns']==100 and row['postprocess_completion_ns']==170
    assert row['queue_wait_ns']==10 and row['e2e_ms']==70/1e6
    assert sum(s['host_elapsed_ms'] for s in row['stages'])==40/1e6
    assert data['measurements']['p50_ms']['value']==70/1e6


def test_p99_tail_uses_real_1000_samples():
    result=measure_pipeline(nn.Identity(),source(),protocol=PipelineProtocol(samples=1000,warmup_samples=0,queue_depth=4))
    data=result.to_dict()
    assert data['measurements']['p99_ms']['status']=='derived'
    assert data['actual']['completed']==1000


@pytest.mark.parametrize('options',[dict(samples=0),dict(repeats=0),dict(warmup_samples=-1),dict(queue_depth=0),
    dict(queue_depth=True),dict(device='mps'),dict(precision='bf16')])
def test_invalid_pipeline_protocol(options):
    with pytest.raises((TypeError,ValueError)): PipelineProtocol(**options)


@pytest.mark.parametrize('case',['none','missing_id','empty_id','duplicate','wrong_id','eof','wrong_stage','wrong_clock'])
def test_invalid_source_stage_association(case):
    read=source(); post=None; protocol=PipelineProtocol(samples=2,warmup_samples=0)
    if case=='none': read=lambda:None
    elif case=='missing_id': read=lambda:PipelineSample(None,torch.ones(1,1,2,2))
    elif case=='empty_id': read=lambda:PipelineSample('',torch.ones(1,1,2,2))
    elif case=='duplicate': read=lambda:PipelineSample('same',torch.ones(1,1,2,2))
    elif case=='wrong_id': post=PipelineStage('postprocess',lambda s:PipelineSample('other',s.payload))
    elif case=='eof': read=iter([]).__next__
    elif case=='wrong_stage': post=PipelineStage('preprocess',lambda s:s)
    else: post=PipelineStage('postprocess',lambda s:s,device='cuda')
    with pytest.raises((ValueError,TypeError,StopIteration)):
        measure_pipeline(nn.Identity(),read,postprocess=post,protocol=protocol)
    assert not engine._RUN_LOCK.locked()


def test_cuda_unavailable_is_not_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(ValueError,match='CUDA unavailable'):
        measure_pipeline(nn.Identity(),source(),transfer=PipelineStage('transfer',lambda s:s,device='cuda'),
                         protocol=PipelineProtocol(device='cuda'))


def test_model_state_rng_modes_input_owned_and_stage_exception():
    model=prepared(); before=state(model); original=torch.ones(2,1,5,7)
    counter=iter(range(20))
    def read(): return PipelineSample(str(next(counter)),original)
    def inplace(frame):
        random.random(); np.random.random(); torch.rand(1)
        frame.payload.add_(1)
        return frame
    measure_pipeline(model,read,preprocess=PipelineStage('preprocess',inplace),protocol=PipelineProtocol(samples=2))
    equal(state(model),before); assert torch.equal(original,torch.ones_like(original))
    def fail(frame): raise LookupError('stage failed')
    with pytest.raises(LookupError):
        measure_pipeline(model,source(),postprocess=PipelineStage('postprocess',fail))
    equal(state(model),before)


@pytest.mark.parametrize('kind',['parameter','input','mode','gradient'])
def test_model_mutation_not_a_successful_pipeline(kind):
    class Bad(nn.Module):
        def __init__(self):
            super().__init__(); self.weight=nn.Parameter(torch.ones(1)); self.weight.grad=torch.full_like(self.weight,.5)
        def forward(self,x):
            if kind=='parameter': self.weight.add_(1)
            elif kind=='input': x.add_(1)
            elif kind=='mode': self.training=True
            else: self.weight.grad.add_(1)
            return x*self.weight
    model=Bad(); before=model.weight.detach().clone()
    with pytest.raises(StateMutationError): measure_pipeline(model,source(),protocol=PipelineProtocol(samples=1,warmup_samples=0))
    assert torch.equal(model.weight,before) and model.training


def test_model_output_lifetime_no_gc_required():
    def run():
        outputs=[]
        class Track(nn.Conv2d):
            def forward(self,x):
                value=super().forward(x); outputs.append(weakref.ref(value)); return value
        model=Track(1,2,3,padding=1); refs=[weakref.ref(model),*[weakref.ref(p) for p in model.parameters()]]
        measure_pipeline(model,source(),protocol=PipelineProtocol(samples=4,queue_depth=2))
        assert all(ref() is None for ref in outputs)
        return refs
    enabled=gc.isenabled(); gc.disable()
    try: assert all(ref() is None for ref in run())
    finally:
        gc.collect()
        if enabled: gc.enable()


@pytest.mark.parametrize('kind',['pipeline','stage'])
def test_frozen_defensive_json_csv_publication(pipeline_result,tmp_path,kind):
    result=pipeline_result if kind=='pipeline' else analyze_stages(pipeline_result)
    before=result.to_dict(); copy=result.to_dict(); copy['diagnostics']['warnings'].append('bad')
    assert result.to_dict()==before
    with pytest.raises(FrozenInstanceError): result._payload='bad'
    path=tmp_path/'result.json'; result.to_json(path); result.to_csv(tmp_path/'result.csv')
    assert json.loads(path.read_text(encoding='utf-8'))==before
    with path.with_suffix('.csv').open(encoding='utf-8',newline='') as handle: rows=list(csv.DictReader(handle))
    assert len(rows)==(3 if kind=='pipeline' else 6)
    with pytest.raises(FileExistsError): result.to_json(path)
    result.to_json(path,overwrite=True)


@pytest.mark.parametrize('case',['ID','pending','drop','count','stage','clock','duration','device','nan'])
def test_forged_pipeline_rejected(pipeline_result,case):
    data=pipeline_result.to_dict(); row=data['raw_evidence']['samples'][0]
    if case=='ID': row['stages'][0]['sample_id']='wrong'
    elif case=='pending': data['actual']['pending']=1
    elif case=='drop': data['actual']['dropped']=1
    elif case=='count': data['actual']['completed']+=1
    elif case=='stage': row['stages'].pop()
    elif case=='clock': data['protocol']['clock']='device_time'
    elif case=='duration': row['e2e_ms']+=1
    elif case=='device': row['stages'][0]['device_event_ms']=2
    else: row['e2e_ms']=float('nan')
    with pytest.raises(ValueError): PipelineResult(data)


def test_analyze_does_not_rerun_and_source_unchanged(pipeline_result,monkeypatch):
    before=pipeline_result.to_dict()
    monkeypatch.setattr(engine,'measure_pipeline',lambda *args,**kwargs:pytest.fail('no rerun'))
    assert analyze_stages(pipeline_result).to_dict()['identity']['source_result_id']==before['result_id']
    assert pipeline_result.to_dict()==before


@pytest.mark.parametrize('options',[dict(trials=0),dict(timeout_seconds=0),dict(timeout_seconds=float('inf')),
                                   dict(warm_samples=0),dict(warmup_iterations=-1)])
def test_invalid_cold_protocol(options):
    with pytest.raises(ValueError): ColdStartProtocol(**options)


@pytest.mark.parametrize('factory,kwargs',[('lambda',{}),('torch.nn:Identity',{'model_kwargs':{'tensor':torch.ones(1)}}),
    ('torch.nn:Identity',{'device':'mps'}),('torch.nn:Identity',{'input_shape':(0,1,2,2)}),
    ('torch.nn:Identity',{'artifact_path':'x'}),('torch.nn:Identity',{'artifact_sha256':'a'*64})])
def test_recipe_plain_only(factory,kwargs):
    with pytest.raises((ValueError,TypeError)): ColdStartRecipe(factory,**kwargs)


def test_actual_fresh_cpu_workers_artifact_phases_and_parent_state(tmp_path):
    model=nn.Conv2d(1,2,3,padding=1); path=tmp_path/'state.pt'
    info=measure_footprint(model,serialize_to=path,verify_roundtrip=True).to_dict()['raw_evidence']['artifact']
    before=state(model)
    recipe=ColdStartRecipe('torch.nn:Conv2d',model_kwargs=dict(in_channels=1,out_channels=2,kernel_size=3,padding=1),
                          artifact_path=path,artifact_sha256=info['sha256'])
    result=measure_cold_start(recipe,protocol=ColdStartProtocol(trials=2,warm_samples=2,warmup_iterations=1))
    data=result.to_dict(); rows=data['raw_evidence']['trials']
    assert data['actual']['measured']==2 and len({r['worker_pid'] for r in rows})==2
    assert all(r['worker_pid']!=os.getpid() and r['process_terminated'] for r in rows)
    for row in rows:
        assert row['artifact']['sha256']==info['sha256'] and row['provenance']['forward_count']==4
        assert len(row['warm_samples_ms'])==2 and row['phases']['first_inference']['status']=='measured'
    equal(state(model),before)
    result.to_json(tmp_path/'cold.json'); result.to_csv(tmp_path/'cold.csv')
    assert json.loads((tmp_path/'cold.json').read_text())==data
    bad=result.to_dict(); bad['raw_evidence']['trials'][0]['phases']['first_inference']['elapsed_ms']+=1
    with pytest.raises(ValueError): ColdStartResult(bad)


@pytest.mark.parametrize('case',['missing_artifact','corrupt_artifact','factory_exception','import_exception','timeout','crash'])
def test_cold_failures_are_explicit_terminal_trials(case,tmp_path):
    kwargs={}; factory='torch.nn:Identity'; timeout=60.
    if case.endswith('artifact'):
        path=tmp_path/'artifact.pt'
        if case=='corrupt_artifact': path.write_bytes(b'corrupt')
        kwargs=dict(artifact_path=path,artifact_sha256=hashlib.sha256(b'corrupt').hexdigest())
    elif case=='factory_exception': factory='cold_factories:explode'; kwargs['factory_search_path']=str(Path(__file__).parent)
    elif case=='import_exception': factory='nonexistent_hyso_factory:make'
    elif case=='crash': factory='cold_factories:crash'; kwargs['factory_search_path']=str(Path(__file__).parent)
    else: timeout=.01
    data=measure_cold_start(ColdStartRecipe(factory,**kwargs),protocol=ColdStartProtocol(trials=1,timeout_seconds=timeout)).to_dict()
    trial=data['raw_evidence']['trials'][0]
    assert trial['status']==('timed_out' if case=='timeout' else 'crashed' if case=='crash' else 'failed')
    assert trial['process_terminated'] and data['actual']['measured']==0
    assert data['measurements']['first_inference_mean_ms']['value'] is None
    assert not [t for t in threading.enumerate() if t.name=='hyso-cold-ready']


def test_executable_guide_and_reopen(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    path=Path(__file__).resolve().parents[1]/'docs/guides/performance-pipeline.md'; namespace={}
    for block in re.findall(r'```python\n(.*?)```',path.read_text(encoding='utf-8'),re.S):
        exec(compile(block,str(path),'exec'),namespace)
    from hyso.experiments import ExperimentManager
    manager=ExperimentManager(namespace['root']); run=manager.get_run(namespace['run_id'])
    assert run.to_dict()['status']=='completed' and len(run.read_metrics())==3
    assert all(manager.verify_artifact(identity)['status']=='verified' for identity in namespace['artifact_ids'])
    manager.close()


def test_recipe_defensive_copy_and_cold_launch_failure_cleanup(monkeypatch,tmp_path):
    from hyso.performance import cold as module
    kwargs={'in_channels':1,'out_channels':2,'kernel_size':3}
    recipe=ColdStartRecipe('torch.nn:Conv2d',model_kwargs=kwargs)
    kwargs['in_channels']=99
    assert recipe.to_dict()['model_kwargs']['in_channels']==1
    with pytest.raises(FrozenInstanceError): recipe._payload='changed'
    def fail(*args,**kwargs): raise OSError('controlled launch failure')
    monkeypatch.setattr(module.subprocess,'Popen',fail)
    with pytest.raises(OSError,match='launch failure'):
        measure_cold_start(recipe,protocol=ColdStartProtocol(trials=1))
    assert not [t for t in threading.enumerate() if t.name=='hyso-cold-ready']


def test_measure_stages_convenience_and_no_pipeline_window_io(monkeypatch):
    import builtins
    import sqlite3
    import socket
    from unittest.mock import patch
    original=engine._stage
    def forbidden(*args,**kwargs): raise AssertionError('no tracking I/O in pipeline stage')
    def wrapped(*args,**kwargs):
        with patch.object(builtins,'open',forbidden),patch.object(sqlite3,'connect',forbidden),patch.object(socket,'socket',forbidden):
            return original(*args,**kwargs)
    monkeypatch.setattr(engine,'_stage',wrapped)
    assert isinstance(measure_stages(nn.Identity(),source(),protocol=PipelineProtocol(samples=1)),StageTimingResult)
