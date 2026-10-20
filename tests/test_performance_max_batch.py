"""P17 real CPU workers, failure ownership and finite evidence contracts."""
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from hyso.performance import (ColdStartRecipe, BatchSearchProtocol, CapacityResult,
                              find_max_batch_size)


def recipe(**kw):
    return ColdStartRecipe('batch_factories:Selective', factory_search_path=Path(__file__).parent,
                           input_shape=(1,1,5,7), model_kwargs=kw)


def plan(candidates=(1,2,3), **kw):
    return BatchSearchProtocol(candidates=candidates, maximum_batch=3,
        cpu_memory_limit_bytes=3*1024**3, input_limit_bytes=8*1024**2,
        timeout_seconds=30, warmup_iterations=1, measure_iterations=2, **kw)


def assert_dead(pid):
    if pid is None: return
    import ctypes as c
    from ctypes import wintypes as w
    api=c.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];api.OpenProcess.restype=w.HANDLE
    api.GetExitCodeProcess.argtypes=[w.HANDLE,c.POINTER(w.DWORD)];api.GetExitCodeProcess.restype=w.BOOL
    api.CloseHandle.argtypes=[w.HANDLE]
    handle=api.OpenProcess(0x1000,False,pid)
    if not handle: return
    try:
        code=w.DWORD();assert api.GetExitCodeProcess(handle,c.byref(code))
        assert code.value!=259,'owned worker is still active'
    finally: api.CloseHandle(handle)


@pytest.fixture(scope='module')
def observed():
    return find_max_batch_size(recipe(), protocol=plan())


def test_real_nonmonotonic_candidates_and_cardinality(observed):
    d=observed.to_dict(); rows=d['raw_evidence']['candidates']
    assert [r['batch'] for r in rows]==[1,2,3]
    assert [r['status'] for r in rows]==['feasible','infeasible_oom','feasible']
    assert d['measurements']['maximum_observed_feasible_batch']==3
    assert d['measurements']['maximum_proven_within_range']==3
    assert d['measurements']['range_exhaustive']
    assert len({r['worker_pid'] for r in rows})==3
    assert all(r['process_terminated'] and r['exit_code']==0 for r in rows)
    for r in (rows[0],rows[2]):
        assert r['actual_batch']==r['batch'] and r['forward_completed']
        b=r['sweep']['raw_evidence']['benchmark']
        assert b['configuration']['warmup_iterations']==1
        assert b['configuration']['measure_iterations']==2
        assert r['memory']['rss_after']['rss_bytes']>0
        assert r['memory']['budget_scope']=='windows_job_private_commit'
        assert r['latency_mean_ms']>0


def test_order_cap_gaps_and_first_infeasible():
    d=find_max_batch_size(recipe(),protocol=plan((3,2,1),stop_policy='first_infeasible')).to_dict()
    assert [r['status'] for r in d['raw_evidence']['candidates']]==['feasible','infeasible_oom','not_run']
    assert d['measurements']['maximum_observed_feasible_batch']==3
    assert d['measurements']['maximum_proven_within_range'] is None
    assert d['measurements']['status']=='partial'
    d=find_max_batch_size(recipe(),protocol=plan((1,3))).to_dict()
    assert d['measurements']['candidate_list_complete']
    assert not d['measurements']['range_exhaustive']
    assert d['measurements']['maximum_proven_within_range'] is None


@pytest.mark.parametrize('failure,status,code',[
    ('exception','failed',0),('crash','crashed',73),('timeout','timed_out',None),
    ('cardinality','failed',0)])
def test_failure_aborts_and_no_successful_max(failure,status,code):
    p=plan((2,3))
    if failure=='timeout':
        p=BatchSearchProtocol(candidates=(2,3),maximum_batch=3,cpu_memory_limit_bytes=3*1024**3,
                              input_limit_bytes=8*1024**2,timeout_seconds=15,measure_iterations=1)
    d=find_max_batch_size(recipe(failure=failure),protocol=p).to_dict()
    rows=d['raw_evidence']['candidates']
    assert rows[0]['status']==status and rows[1]['status']=='not_run'
    assert rows[0]['process_terminated']
    assert_dead(rows[0]['worker_pid']);assert_dead(rows[0]['launcher_pid'])
    if code is not None: assert rows[0]['exit_code']==code
    if failure=='timeout': assert rows[0]['runtime_ready']
    assert d['measurements']['status']=='aborted'
    assert d['measurements']['maximum_observed_feasible_batch'] is None
    assert rows[0]['classification']=='unknown'


def test_invalid_model_and_input_budget():
    r=ColdStartRecipe('batch_factories:invalid',factory_search_path=Path(__file__).parent)
    d=find_max_batch_size(r,protocol=plan((1,))).to_dict()
    assert d['raw_evidence']['candidates'][0]['status']=='failed'
    r=ColdStartRecipe('torch.nn:Identity',input_shape=(1,1,10000,10000))
    d=find_max_batch_size(r,protocol=plan((1,))).to_dict()
    assert d['raw_evidence']['candidates'][0]['status']=='resource_limit'
    assert d['measurements']['maximum_proven_within_range'] is None


@pytest.mark.parametrize('kwargs',[
    {'candidates':[]},{'candidates':[1,1]},{'candidates':[True]},
    {'candidates':[0]},{'candidates':[4]}, {'maximum_batch':0},
    {'cpu_memory_limit_bytes':0},{'input_limit_bytes':0},{'timeout_seconds':float('nan')},
    {'stop_policy':'binary'}, {'measure_iterations':0},{'repeats':0},
    {'cuda_memory_fraction':1.1},{'cuda_memory_fraction':0}, {'warmup_iterations':-1}])
def test_invalid_protocol(kwargs):
    base=dict(candidates=[1,2],maximum_batch=3,cpu_memory_limit_bytes=3*1024**3,
              input_limit_bytes=1024**2,timeout_seconds=20)
    with pytest.raises((ValueError,TypeError)): BatchSearchProtocol(**dict(base,**kwargs))


def test_parent_state_rng_and_lifetime():
    model=torch.nn.Linear(2,2).train(); model.weight.grad=torch.ones_like(model.weight)
    before=[v.detach().clone() for v in (*model.parameters(),model.weight.grad)]
    py=random.getstate(); npstate=np.random.get_state(); rng=torch.get_rng_state().clone()
    initialized=torch.cuda.is_initialized()
    d=find_max_batch_size(recipe(reject=0),protocol=plan((1,))).to_dict()
    assert random.getstate()==py and np.array_equal(np.random.get_state()[1],npstate[1])
    assert torch.equal(torch.get_rng_state(),rng) and torch.cuda.is_initialized()==initialized
    assert model.training and all(torch.equal(a,b) for a,b in zip(before,(*model.parameters(),model.weight.grad)))
    assert d['raw_evidence']['candidates'][0]['process_terminated']
    for row in d['raw_evidence']['candidates']:
        assert_dead(row['worker_pid']);assert_dead(row['launcher_pid'])


def test_frozen_plain_export_validation(observed,tmp_path):
    with pytest.raises((FrozenInstanceError,AttributeError)): observed._payload='bad'
    d=observed.to_dict();d['measurements']['maximum_observed_feasible_batch']=999
    assert observed.to_dict()['measurements']['maximum_observed_feasible_batch']==3
    with pytest.raises(ValueError): CapacityResult(d)
    d=observed.to_dict(); d['raw_evidence']['candidates'][0]['latency_mean_ms']=float('nan')
    with pytest.raises(ValueError): CapacityResult(d)
    observed.to_json(tmp_path/'result.json'); observed.to_csv(tmp_path/'result.csv')
    assert CapacityResult(json.loads((tmp_path/'result.json').read_text())).to_dict()==observed.to_dict()
    with pytest.raises(FileExistsError): observed.to_json(tmp_path/'result.json')
    with pytest.raises(FileExistsError): observed.to_csv(tmp_path/'result.csv')
    assert len((tmp_path/'result.csv').read_text().splitlines())==4


def test_experiment_artifact_metrics_reopen(observed,tmp_path):
    from hyso.experiments import ExperimentManager
    root=tmp_path/'store'; m=ExperimentManager(root)
    experiment=m.create_experiment('capacity'); m.close()
    ExperimentManager.migrate_store(root)
    m=ExperimentManager(root);run=m.create_run(experiment['id'],'benchmark',config={});run.start()
    path=tmp_path/'capacity.json';observed.to_json(path)
    a=run.add_artifact(path,kind='performance_result',ownership='copy')
    for i,r in enumerate(observed.to_dict()['raw_evidence']['candidates']):
        if r['status']=='feasible':
            for name,value,unit in [('latency',r['latency_mean_ms'],'ms/batch'),
                ('rss',r['memory']['rss_after']['rss_bytes'],'byte')]:
                run.log_metric(name='capacity.'+name,value=value,unit=unit,step=i,
                    step_kind='observation',source='performance.capacity_v1')
    rid=run.id;run.finish();m.close();m=ExperimentManager(root)
    assert m.verify_artifact(a['id'])['status']=='verified'
    assert len(m.get_run(rid).read_metrics())==4
    m.close()


def test_hard_private_commit_guard():
    import subprocess
    import sys
    worker=Path(__file__).resolve().parents[1]/'hyso/performance/_batch_worker.py'
    code="""
import importlib.util,sys
s=importlib.util.spec_from_file_location('guard',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
guard=m._memory_guard(256*1024**2)
try:
    storage=bytearray(300*1024**2)
except MemoryError:
    print('job_budget_rejected_allocation')
else:
    raise AssertionError('private commit budget did not apply')
"""
    p=subprocess.run([sys.executable,'-I','-B','-c',code,str(worker)],capture_output=True,text=True,timeout=15)
    assert p.returncode==0,p.stderr
    assert p.stdout.strip()=='job_budget_rejected_allocation'


@pytest.mark.parametrize('field,value', [('classification','infeasible'),('actual_batch',2),
    ('exit_code',73),('process_terminated',False),('shape',[2,1,5,7])])
def test_reject_inconsistent_worker_evidence(observed,field,value):
    d=observed.to_dict();d['raw_evidence']['candidates'][0][field]=value
    with pytest.raises(ValueError): CapacityResult(d)


def test_fp64_and_verified_artifact(tmp_path):
    model=torch.nn.Conv2d(1,2,1)
    path=tmp_path/'model.pth';torch.save(model.state_dict(),path)
    import hashlib
    r=ColdStartRecipe('torch.nn:Conv2d',model_kwargs=dict(in_channels=1,out_channels=2,kernel_size=1),
        precision='fp64',artifact_path=path,artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    d=find_max_batch_size(r,protocol=plan((1,))).to_dict()
    row=d['raw_evidence']['candidates'][0]
    assert row['status']=='feasible' and row['sweep']['provenance']['precision']['output_dtype']=='torch.float64'
    bad=ColdStartRecipe('torch.nn:Identity',artifact_path=path,artifact_sha256='0'*64)
    d=find_max_batch_size(bad,protocol=plan((1,))).to_dict()
    assert d['raw_evidence']['candidates'][0]['status']=='failed'
    assert 'SHA256 mismatch' in d['raw_evidence']['candidates'][0]['reason']
