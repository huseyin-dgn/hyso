"""Native payload lifetimes, exact replay, abandoned graphs and bounded evidence."""
import copy
import csv
import gc
import json
from pathlib import Path
import random
import threading
import weakref
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from hyso.performance import TrainingMemoryRecipe, IsolatedTrainingMemoryProtocol, IsolatedTrainingMemoryResult, measure_training_memory_isolated
from hyso.performance import isolated_training_memory as module
from hyso.performance._saved_activation import _SavedActivationRecorder, _validate_activation
from training_memory_factories import tiny, state_digest

TESTS=Path(__file__).parent


def seed():random.seed(71);np.random.seed(71);torch.manual_seed(71)


def protocol(**kwargs):return IsolatedTrainingMemoryProtocol(saved_activation_lifetime=True,process_memory_bytes=4*1024**3,
    job_memory_bytes=8*1024**3,cuda_memory_fraction=.15,**kwargs)


@pytest.mark.parametrize('kwargs',[{},dict(optimizer='adam'),dict(optimizer='adamw'),dict(tied=True),dict(modular=True),dict(modular=True,early=True)])
def test_worker_exact_state_rng_native_saved_lifetime(kwargs,tmp_path):
    seed();w=tiny(device='cpu',**kwargs);w['trainer'].fit(w['train_loader'],w['val_loader'],epochs=2)
    expected=state_digest(w['trainer']);before=copy.deepcopy((random.getstate(),np.random.get_state(),torch.get_rng_state()))
    state=tmp_path/'state.json'
    recipe=TrainingMemoryRecipe('training_memory_factories:tiny',factory_search_path=TESTS,seed=71,epochs=2,
        kwargs=dict(kwargs,state_path=str(state)))
    result=measure_training_memory_isolated(recipe,protocol=protocol());d=result.to_dict()
    assert d['outcome']=='success',d
    assert json.loads(state.read_text())==dict(state_sha256=expected,fits=1)
    assert random.getstate()==before[0] and np.array_equal(np.random.get_state()[1],before[1][1]) and torch.equal(torch.get_rng_state(),before[2])
    a=d['measurement']['activation'];s=a['summary']
    assert s['saved_tensors']>0 and s['pack_events']==s['release_events']==s['saved_tensors']
    assert s['unpack_events']>0 and s['live_payloads_at_end']==0
    assert s['peak_live_saved_storage_bytes']>=s['peak_live_nonparameter_storage_bytes']>0
    assert a['events'][-1]['live_storage_bytes']==0
    assert a['physical_activation_peak']['value'] is None
    if kwargs.get('tied'):
        assert len({r['storage_id'] for r in a['records']})<len(a['records'])
    result.to_json(tmp_path/'result.json');result.to_csv(tmp_path/'result.csv')
    rows=list(csv.DictReader((tmp_path/'result.csv').open(newline='',encoding='utf-8')))
    assert sum(r['kind']=='saved_tensor' for r in rows)==len(a['events'])
    assert IsolatedTrainingMemoryResult(json.loads((tmp_path/'result.json').read_text())).to_dict()==d


def test_worker_checkpoint_resume_exact(tmp_path):
    seed();w=tiny(device='cpu',optimizer='adamw',modular=True);w['trainer'].fit(w['train_loader'],w['val_loader'],epochs=2)
    expected=state_digest(w['trainer']);checkpoint=tmp_path/'last.pth';state=tmp_path/'state.json'
    first=TrainingMemoryRecipe('training_memory_factories:tiny',factory_search_path=TESTS,seed=71,epochs=2,
        save_last_to=str(checkpoint),kwargs=dict(optimizer='adamw',modular=True,interrupt_after_epoch=1))
    assert measure_training_memory_isolated(first,protocol=protocol()).to_dict()['outcome']=='workload_exception'
    second=TrainingMemoryRecipe('training_memory_factories:tiny',factory_search_path=TESTS,seed=71,epochs=2,
        resume_from=str(checkpoint),save_last_to=str(checkpoint),kwargs=dict(optimizer='adamw',modular=True,state_path=str(state)))
    d=measure_training_memory_isolated(second,protocol=protocol()).to_dict();assert d['outcome']=='success',d['reason']
    assert json.loads(state.read_text())['state_sha256']==expected
    assert [e['epoch_id'] for e in d['measurement']['source_p19']['epochs']]==[2]
    done=measure_training_memory_isolated(second,protocol=protocol()).to_dict();assert done['outcome']=='success'
    assert done['measurement']['source_p19']['actual_epochs']==0
    assert done['measurement']['activation']['summary']['saved_tensors']==0


@pytest.mark.parametrize('failure',['after_forward','backward',None])
def test_real_worker_gc_disabled_cleanup(failure,tmp_path):
    lifetime=tmp_path/'lifetime.json'
    recipe=TrainingMemoryRecipe('training_memory_factories:tiny',factory_search_path=TESTS,seed=71,epochs=2,
        kwargs=dict(failure=failure,lifetime_path=str(lifetime)))
    d=measure_training_memory_isolated(recipe,protocol=protocol()).to_dict()
    assert d['outcome']==('workload_exception' if failure else 'success'),d
    assert all(json.loads(lifetime.read_text()).values())
    assert d['receipt']['job']['active_processes']==0
    if failure:assert d['measurement'] is None


@pytest.mark.parametrize('failure',['after_forward','backward','budget',None])
def test_no_tensor_retention_or_hook_after_unwind_without_gc(failure):
    enabled=gc.isenabled();gc.disable()
    try:
        model=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.Sigmoid(),torch.nn.Linear(3,2))
        observer=_SavedActivationRecorder(model,1 if failure=='budget' else 100,400)
        observer.context=dict(epoch_id=1,microbatch_id=1,window_id=1);observer.stage='forward'
        refs=[]
        model.register_forward_hook(lambda m,a,o:refs.append(weakref.ref(o)))
        if failure=='backward':
            def fail(g):raise RuntimeError('backward')
            next(model.parameters()).register_hook(fail)
        model_ref=weakref.ref(model);parameter_ref=weakref.ref(next(model.parameters()))
        output=loss=None
        try:
            with observer.hooks():
                output=model(torch.ones(2,3));loss=output.square().sum()
                if failure=='after_forward':raise RuntimeError('after forward')
                observer.stage='backward';loss.backward()
        except (RuntimeError,OverflowError):
            if failure is None:raise
        finally:
            observer.close()
        loss=output=None
        del model
        assert model_ref() is None and parameter_ref() is None and all(r() is None for r in refs)
        assert not observer.slots and not observer.storages
        seen=[]
        with torch.autograd.graph.saved_tensors_hooks(lambda t:(seen.append(True) or t),lambda t:t):
            t=torch.ones(1,requires_grad=True);(t*t).sum().backward()
        assert seen and not [t for t in threading.enumerate() if t.name=='hyso-training-rss']
    finally:
        if enabled:gc.enable()


def test_retain_graph_is_explicitly_unsupported_and_cleanup_releases():
    model=torch.nn.Linear(2,2);r=_SavedActivationRecorder(model,100,400)
    r.context=dict(epoch_id=1,microbatch_id=1,window_id=1);r.stage='forward'
    with r.hooks():
        output=model(torch.ones(1,2));output.sum().backward(retain_graph=True)
    with pytest.raises(NotImplementedError,match='retained graph'):r.result()
    r.close();assert not r.slots and not r.storages


def test_custom_autograd_is_explicitly_unsupported():
    class Custom(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x):ctx.save_for_backward(x);return x*2
        @staticmethod
        def backward(ctx,g):return ctx.saved_tensors[0]*g
    model=torch.nn.Linear(2,2);r=_SavedActivationRecorder(model,100,400)
    r.context=dict(epoch_id=1,microbatch_id=1,window_id=1);r.stage='forward'
    try:
        with r.hooks():
            out=Custom.apply(model(torch.ones(1,2)))
            with pytest.raises(NotImplementedError,match='custom autograd'):out.sum().backward()
    finally:r.close()


@pytest.mark.parametrize('reentrant',[False,True])
def test_checkpointing_is_explicitly_unsupported(reentrant):
    from torch.utils.checkpoint import checkpoint
    model=torch.nn.Linear(2,2);r=_SavedActivationRecorder(model,100,400)
    r.context=dict(epoch_id=1,microbatch_id=1,window_id=1);r.stage='forward'
    try:
        with pytest.raises(NotImplementedError),r.hooks():
            out=checkpoint(model,torch.ones(1,2,requires_grad=True),use_reentrant=reentrant)
            r.stage='backward';out.sum().backward()
    finally:r.close()


def test_alias_slots_preserve_original_objects_and_deduplicate_storage():
    model=torch.nn.Linear(2,2);r=_SavedActivationRecorder(model,100,400)
    r.context=dict(epoch_id=1,microbatch_id=1,window_id=1);r.stage='forward'
    weight=model.weight;view=weight.t()
    p=r.pack(weight);q=r.pack(view)
    assert r.unpack(p) is weight and r.unpack(q) is view
    assert r.records[0]['storage_id']==r.records[1]['storage_id']
    assert r.live_logical==2*weight.numel()*weight.element_size()
    assert r.live_storage==weight.untyped_storage().nbytes() and r.live_nonparameter==0
    p.clear();q.clear();assert r.result()['summary']['live_payloads_at_end']==0
    r.close()


@pytest.mark.parametrize('mutation',['output','parameter'])
def test_saved_hooks_preserve_native_inplace_version_rejection(mutation):
    for observed in (False,True):
        model=torch.nn.Linear(2,2);value=torch.ones(1,2,requires_grad=True)
        recorder=_SavedActivationRecorder(model,100,400)
        recorder.context=dict(epoch_id=1,microbatch_id=1,window_id=1);recorder.stage='forward'
        try:
            with recorder.hooks() if observed else nullcontext():
                output=value.sigmoid() if mutation=='output' else model(value)
                changed=output if mutation=='output' else model.weight
                with torch.no_grad():changed.add_(.1)
                recorder.stage='backward'
                with pytest.raises(RuntimeError,match='modified by an inplace operation'):
                    output.sum().backward()
        finally:recorder.close()
        assert not recorder.slots and not recorder.storages


def test_abandoned_native_output_cycle_requires_and_releases_owned_cleanup():
    enabled=gc.isenabled();gc.disable()
    try:
        model=torch.nn.Identity();r=_SavedActivationRecorder(model,100,400)
        r.context=dict(epoch_id=1,microbatch_id=1,window_id=1);r.stage='forward'
        with r.hooks():
            value=torch.ones(2,requires_grad=True);output=value.sigmoid()
        output_ref=weakref.ref(output);value_ref=weakref.ref(value)
        output=value=None
        assert output_ref() is not None and r.slots  # Original output closes a native graph cycle.
        with pytest.raises(NotImplementedError,match='retained graph'):r.result()
        r.close();r.close()
        assert output_ref() is None and value_ref() is None
        assert not r.slots and not r.storages
        assert r.live_logical==r.live_storage==r.live_nonparameter==0
        assert all(row['release_kind']=='cleanup' for row in r.records)
    finally:
        if enabled:gc.enable()


def test_payload_release_reentrant_close_is_idempotent(monkeypatch):
    import sys
    unraisable=[]
    monkeypatch.setattr(sys,'unraisablehook',lambda error:unraisable.append(type(error.exc_value).__name__))
    r=_SavedActivationRecorder(torch.nn.Identity(),100,400)
    r.context=dict(epoch_id=1,microbatch_id=1,window_id=1);r.stage='forward'
    first=torch.ones(2);second=torch.ones(3)
    p=r.pack(first);q=r.pack(second)
    callback=weakref.finalize(first,r.close)
    first=second=None
    p.clear()  # Dropping the tensor reenters close before p's release bookkeeping.
    assert not callback.alive and not r.slots and not r.storages
    assert r.live_logical==r.live_storage==r.live_nonparameter==0
    assert [row['release_kind'] for row in r.records]==['native_payload_disposal','cleanup']
    before=copy.deepcopy((r.records,r.events))
    r.close();p.clear();q.clear()
    assert (r.records,r.events)==before and not unraisable


def test_export_tracking_does_not_retrain(activation_result,tmp_path,monkeypatch):
    import hashlib
    from hyso.experiments import ExperimentManager
    result=IsolatedTrainingMemoryResult(activation_result)
    def forbidden(*args,**kwargs):raise AssertionError('retraining')
    monkeypatch.setattr(module,'_launch',forbidden)
    root=tmp_path/'store';manager=ExperimentManager(root);manager.close()
    ExperimentManager.migrate_store(root);manager=ExperimentManager(root)
    run=manager.create_run(manager.create_experiment('isolated P23')['id'],'training',{});run.start()
    for kind in ('json','csv'):
        path=tmp_path/('result.'+kind);getattr(result,'to_'+kind)(path)
        original=path.read_bytes()
        with pytest.raises(FileExistsError):getattr(result,'to_'+kind)(path)
        assert original==path.read_bytes()
        a=run.add_artifact(path,kind='report',ownership='copy')
        assert a['expected_sha256']==hashlib.sha256(original).hexdigest()
        assert manager.verify_artifact(a['id'])['status']=='verified'
    run.log_metric(name='memory.saved_payload.peak_logical',value=activation_result['measurement']['activation']['summary']['peak_live_saved_logical_bytes'],unit='byte',source='P23',epoch=1)
    with monkeypatch.context() as patcher:
        patcher.setattr(run,'log_metric',lambda **kw:(_ for _ in ()).throw(RuntimeError('tracking error')))
        with pytest.raises(RuntimeError):run.log_metric()
    run.finish();identity=run.id;manager.close();manager=ExperimentManager(root)
    assert len(manager.get_run(identity).read_metrics())==1 and len(manager.list_artifacts(identity))==2
    manager.close()


@pytest.fixture(scope='module')
def activation_result():
    recipe=TrainingMemoryRecipe('training_memory_factories:tiny',factory_search_path=TESTS,seed=71,epochs=2)
    d=measure_training_memory_isolated(recipe,protocol=protocol()).to_dict();assert d['outcome']=='success',d
    return d


@pytest.mark.parametrize('field',['logical','storage','event','release','source','physical','summary','unpack','bounds'])
def test_rehashed_saved_evidence_tamper_rejected(activation_result,field):
    d=copy.deepcopy(activation_result);a=d['measurement']['activation']
    if field=='logical':a['records'][0]['logical_bytes']+=1
    elif field=='storage':a['records'][0]['storage_id']+=100
    elif field=='event':a['events'][0]['live_storage_bytes']+=1
    elif field=='release':a['records'][0]['release_ns']=None
    elif field=='source':a['records'][0]['microbatch_id']=1000
    elif field=='physical':a['physical_activation_peak']['value']=1
    elif field=='summary':a['summary']['saved_tensors']+=1
    elif field=='unpack':a['records'][0]['unpack_count']+=1
    else:d['protocol']['max_saved_tensors']=1
    d['sha256']=module._hash({k:v for k,v in d.items() if k!='sha256'})
    with pytest.raises(ValueError):IsolatedTrainingMemoryResult(d)
