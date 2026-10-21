"""P23 native tensor accounting, one-fit replay, bounded cleanup and exports."""
import copy
import gc
import hashlib
import json
from pathlib import Path
import threading
import weakref
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from hyso.training import Trainer
from hyso.performance import (TrainingMemoryProtocol, TrainingMemoryResult, TrainingMemoryError,
                              measure_training_memory, TrainingTimingProtocol)
from hyso.performance import training_memory as module
from trainer_replay_support import seed, equal, snapshot, build, CASES, CRITICAL


def loader(dtype=torch.float32):
    return DataLoader(TensorDataset(torch.arange(15).reshape(5,3).to(dtype)/20,
                                   torch.tensor([0,1,0,1,0])), batch_size=2)


def make(opt=torch.optim.AdamW, dtype=torch.float32, momentum=False):
    model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.2),torch.nn.Linear(4,2)).to(dtype)
    optimizer=opt(model.parameters(),lr=.01,**({'momentum':.9} if momentum else {}))
    return Trainer(model,optimizer=optimizer,loss_fn=torch.nn.CrossEntropyLoss(),accum_steps=2,ema_decay=.8)


def logical(t):
    return module._logical(t.model,t.optimizer,10000)


@pytest.mark.parametrize('opt',[torch.optim.SGD,torch.optim.Adam,torch.optim.AdamW])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_real_logical_gradient_and_lazy_optimizer(opt,dtype):
    t=make(opt,dtype,momentum=opt is torch.optim.SGD)
    before=logical(t)
    expected=sum(p.numel()*p.element_size() for p in t.model.parameters())
    assert before['summary']['parameter']['logical_bytes']==expected
    assert before['summary']['gradient']['logical_bytes']==0
    assert before['summary']['optimizer']['logical_bytes']==0
    x,y=next(iter(loader(dtype)))
    for update in range(2):
        torch.nn.functional.cross_entropy(t.model(x),y).backward()
        assert logical(t)['summary']['gradient']['logical_bytes']==expected
        t.optimizer.step()
        actual=sum(v.numel()*v.element_size() for s in t.optimizer.state.values() for v in s.values() if isinstance(v,torch.Tensor))
        assert logical(t)['summary']['optimizer']['logical_bytes']==actual and actual>0
        t.optimizer.zero_grad(set_to_none=True)
        assert logical(t)['summary']['gradient']['logical_bytes']==0
    if opt is torch.optim.SGD:assert actual==expected


def test_sgd_no_momentum_and_zero_grad_false():
    t=make(torch.optim.SGD);x,y=next(iter(loader()))
    torch.nn.functional.cross_entropy(t.model(x),y).backward();t.optimizer.step()
    assert logical(t)['summary']['optimizer']['logical_bytes']==0
    t.optimizer.zero_grad(set_to_none=False)
    assert logical(t)['summary']['gradient']['logical_bytes']>0


def test_mixed_frozen_tied_view_and_shared_storage():
    m=torch.nn.Module();m.p=torch.nn.Parameter(torch.arange(8,dtype=torch.float64));m.tied=m.p
    m.view=torch.nn.Parameter(m.p[2:6],requires_grad=False)
    m.register_buffer('b',m.p.detach()[1:3]);m.register_buffer('i',torch.ones(3,dtype=torch.int16))
    t=Trainer(m,optimizer=torch.optim.SGD(m.parameters(),lr=.1),loss_fn=torch.nn.CrossEntropyLoss())
    d=logical(t);s=d['summary']
    assert s['parameter']['logical_bytes']==96 and s['parameter']['tensor_count']==2
    assert s['frozen_parameter']['logical_bytes']==32 and s['buffer']['logical_bytes']==22
    assert d['unique_storage_bytes']==70
    p=next(r for r in d['tensors'] if r['names']==['p','tied'])
    view=next(r for r in d['tensors'] if r['names']==['view'])
    b=next(r for r in d['tensors'] if r['names']==['b'])
    assert p['storage_id']==view['storage_id']==b['storage_id'] and view['storage_offset']==2
    assert p['tensor_id']!=view['tensor_id'] and b['stride']==[1]


def test_unsupported_sparse_quantized_custom_optimizer():
    m=torch.nn.Linear(3,2);m.register_buffer('sparse',torch.ones(2,2).to_sparse())
    m.register_buffer('quantized',torch.quantize_per_tensor(torch.ones(2),.1,0,torch.qint8))
    class Custom(torch.optim.SGD):pass
    d=module._logical(m,Custom(m.parameters(),lr=.1),1000)
    assert d['summary']['buffer']['logical_bytes'] is None
    assert d['summary']['optimizer']['status']=='unsupported'


@pytest.mark.parametrize('opt',[torch.optim.SGD,torch.optim.Adam,torch.optim.AdamW])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_linear_exact_one_fit_and_accumulation(opt,dtype):
    outputs=[]
    for observed in (False,True):
        seed(82);t=make(opt,dtype);train=loader(dtype)
        if observed:r=measure_training_memory(t,train,train,epochs=2)
        else:t.fit(train,train,epochs=2)
        outputs.append(snapshot(t))
        assert not hasattr(t,'_timing_recorder') and not hasattr(t,'_stage_recorder')
    equal(*outputs)
    d=r.to_dict();p19=d['source_p19'];rows=d['observations']
    assert p19['counters']['after']['microbatches']==6
    assert [x['samples'] for x in p19['observations'] if x['kind']=='window']==[4,1,4,1]
    back=[x for x in rows if x['boundary']=='post_backward']
    assert all(x['logical']['summary']['gradient']['logical_bytes']>0 for x in back)
    assert all(x['logical']['summary']['gradient']['logical_bytes']==0 for x in rows if x['boundary']=='post_zero_grad')
    assert rows[0]['logical']['summary']['optimizer']['logical_bytes']==0
    if opt is not torch.optim.SGD:
        assert next(x for x in rows if x['boundary']=='post_scaler_step')['logical']['summary']['optimizer']['logical_bytes']>0
    assert d['activation']['value'] is None and d['cuda']['value'] is None
    rss=d['rss'];assert len(rss['observations'])>=2
    assert rss['summary']['sampled_peak_rss_bytes']['value']>=rss['summary']['baseline_rss_bytes']['value']>0


@pytest.mark.parametrize('case',CASES)
def test_modular_cnn_full_replay(case):
    states=[]
    for observed in (False,True):
        seed(82);t,train,val=build(case)
        if observed:measure_training_memory(t,train,val,epochs=5)
        else:t.fit(train,val,epochs=5)
        states.append(snapshot(t))
    equal(*states)


def test_resume_early_stop_completed_checkpoint(tmp_path):
    import hyso.training.resume as resume
    states=[]
    for observed in (False,True):
        seed(65);t,train,val=build(CRITICAL);path=tmp_path/(str(observed)+'.pth')
        native=resume.save_last
        class Stop(Exception):pass
        def cut(*args):
            native(*args)
            if len(args[0].history['train_loss'])==2:raise Stop()
        with patch.object(resume,'save_last',cut),pytest.raises(Stop):
            if observed:measure_training_memory(t,train,val,epochs=5,save_last_to=path)
            else:t.fit(train,val,epochs=5,save_last_to=path)
        t,train,val=build(CRITICAL)
        if observed:r=measure_training_memory(t,train,val,epochs=5,resume_from=path)
        else:t.fit(train,val,epochs=5,resume_from=path)
        states.append(snapshot(t))
    equal(*states);assert [e['epoch_id'] for e in r.to_dict()['source_p19']['epochs']]==[3,4,5]
    t,train,val=build(dict(CRITICAL,early='terminal',best=True))
    path=tmp_path/'done.pth';r=measure_training_memory(t,train,val,epochs=5,save_last_to=path)
    assert r.to_dict()['source_p19']['actual_epochs']==2
    t,train,val=build(dict(CRITICAL,early='terminal',best=True))
    done=measure_training_memory(t,train,val,epochs=5,resume_from=path)
    assert done.to_dict()['source_p19']['actual_epochs']==0
    assert [x['boundary'] for x in done.to_dict()['observations']]==['ready','segment_end']


@pytest.mark.parametrize('kwargs',[{'peak_reset_policy':'reset'}, {'scope':'isolated'}, {'device':'cuda'},
    {'max_samples':1},{'max_boundaries':100001},{'max_tensor_records':True},{'interval_seconds':float('nan')}])
def test_protocol_admission(kwargs):
    with pytest.raises((ValueError,TypeError)):TrainingMemoryProtocol(**kwargs)


@pytest.mark.parametrize('failure',['constructor','model','budget','timing','sampler'])
def test_failure_cleanup_with_gc_disabled(failure,monkeypatch):
    enabled=gc.isenabled();gc.disable()
    try:
        t=make();ref=weakref.ref(t.model);tensor=weakref.ref(next(t.model.parameters()))
        protocol=TrainingMemoryProtocol()
        if failure=='constructor':protocol=TrainingMemoryProtocol(max_tensor_records=1)
        elif failure=='model':
            def fail(*args):raise RuntimeError('injected forward error')
            t.model.register_forward_pre_hook(fail)
        elif failure=='budget':protocol=TrainingMemoryProtocol(max_boundaries=3)
        elif failure=='sampler':
            def fail(self):self.errors.append('injected sampler failure')
            monkeypatch.setattr(module._MemoryRecorder,'sampling',fail)
        with pytest.raises((TrainingMemoryError,RuntimeError)):
            measure_training_memory(t,loader(),epochs=1,protocol=protocol,
                timing_protocol=TrainingTimingProtocol(max_records=2) if failure=='timing' else None)
        assert not hasattr(t,'_stage_recorder') and not hasattr(t,'_timing_recorder')
        assert not [x for x in threading.enumerate() if x.name=='hyso-training-rss']
        assert not module._RUN_LOCK.locked()
        del t
        assert ref() is None and tensor() is None
    finally:
        if enabled:gc.enable()


def test_missing_rss_backend_and_weakrefs(monkeypatch):
    def unavailable():raise module._BackendUnavailable('missing test backend')
    monkeypatch.setattr(module,'_observe',unavailable)
    enabled=gc.isenabled();gc.disable()
    try:
        t=make();ref=weakref.ref(t.model);tensor=weakref.ref(next(t.model.parameters()))
        r=measure_training_memory(t,loader(),epochs=1)
        assert all(v['value'] is None and v['status']=='unsupported' for v in r.to_dict()['rss']['summary'].values())
        del t
        assert ref() is None and tensor() is None
    finally:
        if enabled:gc.enable()


def test_result_does_not_retain_model_tensor_and_positive_control():
    enabled=gc.isenabled();gc.disable()
    try:
        t=make();ref=weakref.ref(t.model);tensor=weakref.ref(next(t.model.parameters()))
        r=measure_training_memory(t,loader(),epochs=1);retained=[t.model]
        del t
        assert ref() is not None and tensor() is not None
        retained.clear()
        assert ref() is None and tensor() is None
        assert r.to_dict()['status']=='partial'
    finally:
        if enabled:gc.enable()


@pytest.mark.parametrize('field',['logical','rss','cuda','source','activation','timestamp','id','attempt'])
def test_tamper_rejected_even_when_rehashed(field):
    d=measure_training_memory(make(),loader(),epochs=1).to_dict()
    if field=='logical':d['observations'][0]['logical']['summary']['parameter']['logical_bytes']+=1
    elif field=='rss':d['rss']['summary']['current_rss_bytes']['value']+=1
    elif field=='cuda':d['cuda']['value']=0
    elif field=='source':d['source_sha256']='0'*64
    elif field=='activation':d['activation']['value']=0
    elif field=='id':next(r for r in d['observations'] if r['boundary']=='post_backward')['microbatch_id']+=1
    elif field=='attempt':next(r for r in d['observations'] if r['boundary']=='post_attempt')['fields']['skipped']=True
    else:d['observations'][1]['query_start_ns']=0
    with pytest.raises(ValueError):TrainingMemoryResult(d)
    d['sha256']=module._digest({k:v for k,v in d.items() if k!='sha256'})
    with pytest.raises(ValueError):TrainingMemoryResult(d)


def test_immutable_atomic_export_and_manager(tmp_path,monkeypatch):
    from hyso.experiments import ExperimentManager
    t=make();r=measure_training_memory(t,loader(),epochs=1);before=snapshot(t)
    d=r.to_dict();d['status']='complete';assert r.to_dict()['status']=='partial'
    with pytest.raises((AttributeError,TypeError)):r._payload='bad'
    root=tmp_path/'store';m=ExperimentManager(root);m.close();ExperimentManager.migrate_store(root);m=ExperimentManager(root)
    run=m.create_run(m.create_experiment('P23')['id'],'training',{});run.start()
    def forbidden(*args,**kwargs):raise AssertionError('retraining')
    monkeypatch.setattr(t,'fit',forbidden)
    for kind in ('json','csv'):
        path=tmp_path/('memory.'+kind);getattr(r,'to_'+kind)(path)
        original=path.read_bytes()
        with pytest.raises(FileExistsError):getattr(r,'to_'+kind)(path)
        assert path.read_bytes()==original
        a=run.add_artifact(path,kind='report',ownership='copy',metadata={'source_segment_id':r.to_dict()['source_segment_id']})
        assert m.verify_artifact(a['id'])['status']=='verified' and a['expected_sha256']==hashlib.sha256(original).hexdigest()
    run.log_metric(name='memory.rss.sampled_peak',value=r.to_dict()['rss']['summary']['sampled_peak_rss_bytes']['value'],unit='byte',source='P23',epoch=1)
    with patch.object(run,'log_metric',side_effect=RuntimeError('tracking error')),pytest.raises(RuntimeError):
        run.log_metric(name='test',value=1,unit='byte',source='P23',epoch=1)
    equal(before,snapshot(t));run.finish();identity=run.id;m.close();m=ExperimentManager(root)
    assert len(m.get_run(identity).read_metrics())==1 and len(m.list_artifacts(identity))==2;m.close()
    bad=r.to_dict();bad['status']='complete';object.__setattr__(r,'_payload',json.dumps(bad))
    with pytest.raises(ValueError):r.to_json(tmp_path/'bad.json')
    assert not (tmp_path/'bad.json').exists()


def test_cpu_does_not_call_cuda_reset_gc_or_cache(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('global mutation')
    for name in ('reset_peak_memory_stats','empty_cache','memory_allocated','memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    monkeypatch.setattr(gc,'collect',forbidden)
    measure_training_memory(make(),loader(),epochs=1)


def test_exclusive_lock_and_fresh_admission():
    with module._RUN_LOCK:
        with pytest.raises(TrainingMemoryError):measure_training_memory(make(),loader(),epochs=1)
    t=make();measure_training_memory(t,loader(),epochs=1)
    with pytest.raises(ValueError):measure_training_memory(t,loader(),epochs=1)
    assert not module._RUN_LOCK.locked()
    with pytest.raises(ValueError):measure_training_memory(make(),loader(),protocol=TrainingMemoryProtocol(device='cuda:0'))


def test_scheduled_gc_diagnostic_two_runs(tmp_path):
    # Test-only diagnostic: RSS/cache fluctuations are observations, not a leak verdict.
    samples=[]
    for index in range(2):
        gc.collect();baseline=module._observe();t=make();ref=weakref.ref(t.model)
        r=measure_training_memory(t,loader(),epochs=1);del t;gc.collect()
        assert ref() is None
        samples.append(dict(run=index,baseline=baseline,end=module._observe(),peak=r.to_dict()['rss']['summary']['sampled_peak_rss_bytes']))
    (tmp_path/'plateau.json').write_text(json.dumps(samples))
    assert all(s['end']['rss_bytes']>0 for s in samples)


def test_accumulation_gradient_lifecycle_and_attempt_state():
    d=measure_training_memory(make(),loader(),epochs=1).to_dict();rows=d['observations']
    pre=[r for r in rows if r['boundary']=='pre_microbatch']
    back=[r for r in rows if r['boundary']=='post_backward']
    size=back[0]['logical']['summary']['gradient']['logical_bytes']
    assert [r['logical']['summary']['gradient']['logical_bytes'] for r in pre]==[0,size,0]
    assert [r['logical']['summary']['gradient']['logical_bytes'] for r in back]==[size]*3
    attempts=[r for r in rows if r['boundary']=='post_attempt']
    assert [r['fields'] for r in attempts]==[dict(applied=True,skipped=False,samples=4),dict(applied=True,skipped=False,samples=1)]


def test_rss_sampler_bound_and_midrun_missing_backend(monkeypatch):
    t=make();p=TrainingMemoryProtocol(max_samples=2,interval_seconds=.0001)
    recorder=module._MemoryRecorder(t,TrainingTimingProtocol(),p)
    assert recorder.stop.wait(.03) is False
    recorder.close();assert recorder.errors and not recorder.thread and recorder.trainer is None
    assert not [x for x in threading.enumerate() if x.name=='hyso-training-rss']
    calls=[];native=module._observe
    def observe():
        calls.append(True)
        if len(calls)>1:raise module._BackendUnavailable('backend disappeared')
        return native()
    monkeypatch.setattr(module,'_observe',observe)
    d=measure_training_memory(make(),loader(),epochs=1).to_dict()
    assert d['rss']['observations'] and d['rss']['summary']['current_rss_bytes']['value'] is None
    assert d['rss']['reason']=='backend disappeared'


def test_cuda_summary_never_invents_window_peak_or_reset_arithmetic():
    rows=[{'cuda':dict(allocated_bytes=a,reserved_bytes=r,lifetime_peak_allocated_bytes=900,lifetime_peak_reserved_bytes=1000)} for a,r in [(10,100),(30,100),(5,100)]]
    d=module._cuda_summary(rows,'cuda:0')
    assert d['sampled_peak_allocated_bytes']==30 and d['end']['allocated_bytes']==5
    assert d['incremental_window_peak']['value'] is None and d['window_peak']['value'] is None
    assert d['reset_performed'] is False


def test_memory_guide():
    import re
    path=Path(__file__).resolve().parents[1]/'docs/guides/performance-training-memory.md'
    namespace={'__name__':'p23_guide'}
    for block in re.findall(r'```python\n(.*?)```',path.read_text(encoding='utf-8'),re.S):
        exec(compile(block,str(path),'exec'),namespace)
    assert namespace['data']['status']=='partial'
