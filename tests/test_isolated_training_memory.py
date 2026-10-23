"""Phase A production admission/limits, real native CPU fit and immutable evidence."""
from fixture_support import load_fixture
import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
from hyso.performance import (TrainingMemoryRecipe,IsolatedTrainingMemoryProtocol,IsolatedTrainingMemoryResult,
    measure_training_memory_isolated,TrainingMemoryProtocol,TrainingMemoryResult)
from hyso.performance import isolated_training_memory as module
from training_memory_factories import tiny,state_digest

TESTS=Path(__file__).parent


def recipe(**kwargs):
    return TrainingMemoryRecipe('training_memory_factories:tiny',factory_search_path=TESTS,epochs=2,seed=71,kwargs=kwargs)


@pytest.mark.parametrize('kwargs',[{'factory':'x'},{'factory':'__main__:x'},{'factory':'x:<lambda>'},
    {'factory':'x:y','device':'cuda'},{'factory':'x:y','kwargs':{'model':torch.nn.Linear(1,1)}},
    {'factory':'x:y','kwargs':{'tensor':torch.ones(1)}},{'factory':'x:y','kwargs':{'device':'cpu'}}])
def test_recipe_rejects_live_or_unimportable_state(kwargs):
    with pytest.raises((ValueError,TypeError)):TrainingMemoryRecipe(**kwargs)


@pytest.mark.parametrize('kwargs',[{'max_processes':0},{'max_log_bytes':True},{'max_result_bytes':100},
    {'timeout_seconds':float('nan')},{'process_memory_bytes':100},{'job_memory_bytes':1024**3},
    {'cuda_memory_fraction':2.}])
def test_protocol_real_limit_admission(kwargs):
    with pytest.raises((ValueError,TypeError)):IsolatedTrainingMemoryProtocol(**kwargs)


@pytest.mark.parametrize('optimizer',['sgd','adam','adamw'])
def test_real_cpu_one_fit_and_exact_state(optimizer,tmp_path):
    random.seed(71);np.random.seed(71);torch.manual_seed(71)
    expected=tiny(device='cpu',optimizer=optimizer)
    expected['trainer'].fit(expected['train_loader'],expected['val_loader'],epochs=2)
    digest=state_digest(expected['trainer'])
    before=(copy.deepcopy(random.getstate()),copy.deepcopy(np.random.get_state()),torch.get_rng_state().clone(),state_digest(expected['trainer']))
    output=tmp_path/'state.json'
    result=measure_training_memory_isolated(recipe(optimizer=optimizer,state_path=str(output)))
    d=result.to_dict();assert d['outcome']=='success',d
    assert json.loads(output.read_text())==dict(state_sha256=digest,fits=1)
    assert before[0]==random.getstate() and before[1][0]==np.random.get_state()[0] and np.array_equal(before[1][1],np.random.get_state()[1])
    assert torch.equal(before[2],torch.get_rng_state()) and before[3]==state_digest(expected['trainer'])
    p19=d['measurement']['source_p19'];assert [r['samples'] for r in p19['observations'] if r['kind']=='window']==[4,1,4,1]
    assert d['receipt']['job']['active_processes']==0 and d['receipt']['exit_code']==0
    boundaries=d['measurement']['boundaries']
    assert boundaries[0]['logical']['summary']['optimizer']['logical_bytes']==0
    if optimizer!='sgd':assert boundaries[-1]['logical']['summary']['optimizer']['logical_bytes']>0


@pytest.mark.parametrize('failure,outcome',[('constructor','workload_exception'),('forward','workload_exception'),
    ('memory','resource_limit'),('logs','resource_limit'),('timeout','timeout')])
def test_real_owned_failure_limits_and_cleanup(failure,outcome,tmp_path,monkeypatch,record_property):
    lifetime=tmp_path/'lifetime.json'
    if failure=='timeout':
        from training_timeout_evidence import capture_launch
        phases=capture_launch(monkeypatch,tmp_path/'timeout')
    p=IsolatedTrainingMemoryProtocol(timeout_seconds=15 if failure=='timeout' else 60,max_log_bytes=1024 if failure=='logs' else 65536)
    d=measure_training_memory_isolated(recipe(failure=failure,lifetime_path=str(lifetime) if failure=='forward' else None),protocol=p).to_dict()
    assert d['outcome']==outcome,d
    assert d['measurement'] is None and d['receipt']['job']['active_processes']==0
    if failure=='timeout':
        _assert_timeout_phases(d,phases,tmp_path,record_property,require_forward=False)
    if failure=='forward':assert all(json.loads(lifetime.read_text()).values())
    if failure=='logs':assert len(d['receipt']['stdout'])<=1024


def _assert_timeout_phases(data,evidence,tmp_path,record_property,*,require_forward):
    from training_timeout_evidence import rows
    parent=rows(evidence['parent']);child=rows(evidence['child'])
    receipt=data['receipt'];pid=receipt['launcher_pid']
    # Retain the receipt and all durable phases, including when an assertion fails.
    (tmp_path/'timeout-result.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    (tmp_path/'timeout-phases.json').write_text(json.dumps(dict(parent=parent,child=child,ready=evidence['ready']),indent=2),encoding='utf-8')
    assert evidence['launches']==[pid]
    assert [r['phase'] for r in parent]==['process_launch_enter','process_launch_returned','job_assignment_completed','launcher_receipt_observed']
    assert all(r['pid']==pid for r in child)
    for entries in (parent,child):
        stamps=[r['monotonic_ns'] for r in entries]
        assert stamps==sorted(stamps)
    phases=[r['phase'] for r in child]
    if evidence['ready'] is not None:
        assert evidence['ready']==dict(token=data['probe_id'],pid=pid,phase='job_admitted_before_import')
    if 'factory_enter' in phases:assert 'hyso_torch_import_completed' in phases
    if 'trainer_fit_enter' in phases:assert 'factory_returned' in phases
    reached='forward_hook_enter' in phases
    record_property('timeout_last_durable_phase',phases[-1] if phases else 'job_assignment_completed')
    record_property('timeout_forward_pre_hook_observed',reached)
    record_property('timeout_budget_seconds',data['protocol']['timeout_seconds'])
    record_property('timeout_job_active_processes',receipt['job']['active_processes'])
    assert data['outcome']=='timeout' and receipt['monitor_outcome']=='timeout'
    assert data['measurement'] is None and receipt['job']['active_processes']==0
    # The 15s clock includes imports. Do not claim forward coverage on that path.
    # Once real pre-hook entry is observed, keep the genuine stdout assertion.
    if reached:
        assert 'trainer_fit_enter' in phases and 'factory_returned' in phases
        assert 'training_forward_started' in receipt['stdout']
    else:
        assert 'training_forward_started' not in receipt['stdout']
    if require_forward:assert reached,dict(phases=phases,receipt=receipt)


def test_default_deadline_timeout_proves_forward_entry_and_cleanup(tmp_path,monkeypatch,record_property):
    from training_timeout_evidence import capture_launch
    phases=capture_launch(monkeypatch,tmp_path/'timeout')
    before=(copy.deepcopy(random.getstate()),copy.deepcopy(np.random.get_state()),torch.get_rng_state().clone())
    # A separate test of the existing default budget, not a retry or an increase
    # to the original 15s total-lifecycle test. Both clocks start before Popen.
    protocol=IsolatedTrainingMemoryProtocol()
    assert protocol.timeout_seconds==60
    data=measure_training_memory_isolated(recipe(failure='timeout'),protocol=protocol).to_dict()
    _assert_timeout_phases(data,phases,tmp_path,record_property,require_forward=True)
    assert random.getstate()==before[0] and np.array_equal(np.random.get_state()[1],before[1][1])
    assert torch.equal(torch.get_rng_state(),before[2])


def test_child_grandchild_job_cleanup_and_success_lifetime(tmp_path):
    path=tmp_path/'lifetime.json'
    d=measure_training_memory_isolated(recipe(failure='tree',lifetime_path=str(path))).to_dict()
    assert d['outcome']=='success',d
    assert d['receipt']['job']['total_processes']>=3 and d['receipt']['job']['active_processes']==0
    assert all(json.loads(path.read_text()).values())


def test_bounded_result_transfer():
    d=measure_training_memory_isolated(recipe(),protocol=IsolatedTrainingMemoryProtocol(max_result_bytes=16384)).to_dict()
    assert d['outcome']=='resource_limit' and d['measurement'] is None


def test_corrupt_receipt_and_abnormal_exit_classification():
    receipt=dict(monitor_outcome=None,exit_code=0,launcher_pid=9)
    assert module._classify(receipt,dict(token='bad',pid=9),'token')[0]=='invalid_receipt'
    receipt['exit_code']=0xC0000409
    assert module._classify(receipt,None,'token')[0]=='native_crash'
    receipt['monitor_outcome']='timeout'
    assert module._classify(receipt,None,'token')[0]=='timeout'


def test_immutable_export_and_tamper(tmp_path):
    result=measure_training_memory_isolated(recipe())
    d=result.to_dict();assert d['outcome']=='success',d
    result.to_json(tmp_path/'isolated.json');result.to_csv(tmp_path/'isolated.csv')
    assert IsolatedTrainingMemoryResult(json.loads((tmp_path/'isolated.json').read_text())).to_dict()==d
    with pytest.raises(FileExistsError):result.to_json(tmp_path/'isolated.json')
    with pytest.raises((AttributeError,TypeError)):result._payload='bad'
    bad=result.to_dict();bad['receipt']['job']['active_processes']=1
    bad['sha256']=module._hash({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(ValueError):IsolatedTrainingMemoryResult(bad)
    bad=result.to_dict();bad['measurement']['boundaries'][0]['logical']['summary']['parameter']['logical_bytes']+=1
    bad['sha256']=module._hash({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(ValueError):IsolatedTrainingMemoryResult(bad)


def test_legacy_p23_schema_fixture():
    d=load_fixture('training-memory-p23-schema1');assert TrainingMemoryResult(d).to_dict()==d


def test_no_live_trainer_admission():
    with pytest.raises(TypeError):measure_training_memory_isolated(tiny(device='cpu')['trainer'])


def test_inconsistent_windows_rss_is_raw_and_explicitly_unsupported(monkeypatch):
    from hyso.performance import memory
    recorder=object.__new__(module._IsolatedRecorder)
    recorder.rss=[];recorder.rss_reason=None
    monkeypatch.setattr(memory,'_windows_rss',lambda:(8192,4096))
    recorder.sample()
    assert recorder.rss[0]['rss_bytes']==8192 and recorder.rss[0]['lifetime_hwm_bytes']==4096
    assert recorder.rss_reason=='inconsistent_native_windows_rss_hwm; raw_tuple_retained'
    assert all(r['value'] is None and r['status']=='unsupported' for r in module._rss_summary(recorder.rss,recorder.rss_reason).values())
