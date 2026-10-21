"""P25 deterministic reduction, real one-fit diagnostics and lifecycle evidence."""
import gc
import hashlib
import json
import math
from pathlib import Path
import weakref
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader,TensorDataset
from hyso.training import Trainer
from hyso.performance import (GradientDiagnosticsProtocol,GradientDiagnosticsResult,
    GradientDiagnosticsError,measure_gradient_diagnostics,TrainingTimingProtocol,TrainingStageProtocol)
from hyso.performance import gradients as module
from trainer_replay_support import seed,snapshot,equal,build,CASES,CRITICAL


def data(dtype=torch.float32):
    return DataLoader(TensorDataset(torch.arange(15).reshape(5,3).to(dtype)/20,
        torch.tensor([0,1,0,1,0])),batch_size=2)


def make(opt=torch.optim.AdamW,dtype=torch.float32,clip=.1):
    model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.2),torch.nn.Linear(4,2)).to(dtype)
    optimizer=opt(model.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,.02,total_steps=4,cycle_momentum=False)
    return Trainer(model,optimizer=optimizer,loss_fn=torch.nn.CrossEntropyLoss(),accum_steps=2,
                   scheduler=scheduler,ema_decay=.8,grad_clip=clip)


def tensor_info(g,chunk=2):
    return module._tensor_gradient(g,chunk)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32,torch.float64])
@pytest.mark.parametrize('chunk',[1,2,65536])
def test_exact_known_norm_zero_and_packed_views(dtype,chunk):
    for g in (torch.tensor([3,4,0],dtype=dtype),torch.tensor([[3,0],[4,0]],dtype=dtype).t()):
        d=tensor_info(g,chunk)
        assert d['global_l2']==5 and d['finite'] and d['status']=='measured'
        assert d['chunks']==math.ceil(g.numel()/chunk)
    assert tensor_info(torch.zeros(3,dtype=dtype),chunk)['global_l2']==0
    assert tensor_info(torch.zeros(0,dtype=dtype),chunk)['global_l2']==0
    assert tensor_info(torch.tensor(3.,dtype=dtype),chunk)['global_l2']==3


@pytest.mark.parametrize('values,nan,inf',[
    ([float('nan'),1],True,False),([float('inf'),1],False,True),
    ([-float('inf'),1],False,True),([float('nan'),float('inf')],True,True)])
def test_nonfinite_is_evidence_not_json_nonfinite(values,nan,inf):
    d=tensor_info(torch.tensor(values));assert d['has_nan']==nan and d['has_inf']==inf
    assert d['global_l2'] is None and d['finite'] is False and d['status']=='nonfinite'
    json.dumps(d,allow_nan=False)


def test_large_small_fp64_stable_reduction_and_range():
    for value in (1e-300,1e200):
        d=tensor_info(torch.tensor([value,value],dtype=torch.float64))
        assert d['global_l2']==pytest.approx(math.sqrt(2)*value,rel=2e-15,abs=0)
    d=tensor_info(torch.tensor([1.7e308,1.7e308],dtype=torch.float64))
    assert d['status']=='unrepresentable' and d['finite'] and d['global_l2'] is None


@pytest.mark.parametrize('kind',['coo','coalesced','csr','gapped','expanded','complex','meta'])
def test_unsupported_never_densifies_or_invents_norm(kind,monkeypatch):
    if kind in ('coo','coalesced'):
        g=torch.sparse_coo_tensor(torch.tensor([[0,0]]),torch.tensor([3.,4.]),(3,))
        if kind=='coalesced':g=g.coalesce()
    elif kind=='csr':g=torch.eye(3).to_sparse_csr()
    elif kind=='gapped':g=torch.arange(8.).reshape(2,4)[:,::2]
    elif kind=='expanded':g=torch.ones(1).expand(4)
    elif kind=='complex':g=torch.ones(2,dtype=torch.complex64)
    else:g=torch.empty(2,device='meta')
    def forbidden(*a,**kw):raise AssertionError('sparse conversion')
    monkeypatch.setattr(torch.Tensor,'to_dense',forbidden)
    monkeypatch.setattr(torch.Tensor,'coalesce',forbidden)
    d=tensor_info(g);assert d['status']=='unsupported' and d['global_l2'] is None and d['finite'] is None


def test_groups_none_mixed_dtype_tied_identity_and_no_mutation():
    p=torch.nn.Parameter(torch.zeros(2));q=torch.nn.Parameter(torch.zeros(1,dtype=torch.float64));missing=torch.nn.Parameter(torch.ones(1))
    p.grad=torch.tensor([3.,4.]);q.grad=torch.tensor([12.],dtype=torch.float64)
    # Snapshot's identity convention is explicit; standard Trainer rejects duplicated groups.
    class Groups:pass
    opt=Groups();opt.param_groups=[{'params':[p,q]},{'params':[p,missing]}]
    before=(p.grad.clone(),q.grad.clone())
    d=module._snapshot(opt,GradientDiagnosticsProtocol(chunk_elements=1),20)
    assert d['summary']['global_l2']['value']==13
    assert d['summary']['with_gradient']==2 and d['summary']['without_gradient']==1
    assert len(d['tensors'][0]['names'])==2
    equal(before,(p.grad,q.grad))


@pytest.mark.parametrize('opt',[torch.optim.SGD,torch.optim.Adam,torch.optim.AdamW])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('clip',[None,.1])
def test_real_linear_exact_one_fit(opt,dtype,clip):
    states=[]
    for observed in (False,True):
        seed(52);t=make(opt,dtype,clip);train=data(dtype)
        calls=[];native=t.fit
        def fit(*a,**kw):calls.append(True);return native(*a,**kw)
        with patch.object(t,'fit',fit):
            if observed:r=measure_gradient_diagnostics(t,train,train,epochs=2)
            else:t.fit(train,train,epochs=2)
        assert len(calls)==1
        states.append(snapshot(t))
        assert not hasattr(t,'_timing_recorder') and not hasattr(t,'_stage_recorder')
        assert not t.optimizer._optimizer_step_pre_hooks and not t.optimizer._optimizer_step_post_hooks
    equal(*states)
    d=r.to_dict();rows=d['observations']
    assert d['actual']==dict(epochs=2,attempts=4,windows=4)
    assert [x['samples'] for x in rows]==[4,1,4,1]
    assert all(x['optimizer_executed'] and x['applied'] and x['scale_before']==x['scale_after']==1 for x in rows)
    assert all((x['postclip'] is not None)==(clip is not None) for x in rows)
    if clip:
        assert all(x['postclip']['summary']['global_l2']['value']<=clip+1e-6 for x in rows)


def test_analytic_uneven_accumulation_preclip_normalization():
    class SumModel(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64))
        def forward(self,x):return (x*self.weight).sum(1,keepdim=True)
    x=torch.tensor([[1,2],[3,4],[5,6],[7,8],[9,10]],dtype=torch.float64)
    loader=DataLoader(TensorDataset(x,torch.zeros(5,dtype=torch.float64)),batch_size=2)
    model=SumModel();t=Trainer(model,optimizer=torch.optim.SGD(model.parameters(),lr=.01),loss_fn=torch.nn.BCEWithLogitsLoss(),task='binary',accum_steps=2,grad_clip=.5)
    rows=measure_gradient_diagnostics(t,loader,epochs=1).to_dict()['observations']
    assert [r['samples'] for r in rows]==[4,1] and [r['microbatch_ids'] for r in rows]==[[1,2],[3]]
    coefficient=.5/(math.hypot(2.,2.5)+1e-6)
    probability=1/(1+math.exp(.01*coefficient*(2*9+2.5*10)))
    for r,reference in zip(rows,([2.,2.5],[9*probability,10*probability])):
        assert r['preclip']['summary']['global_l2']['value']==pytest.approx(math.hypot(*reference),rel=1e-15)
        assert r['postclip']['summary']['global_l2']['value']<.500001


@pytest.mark.parametrize('case',CASES)
def test_modular_native_replay_matrix(case):
    states=[]
    for observed in (False,True):
        seed(73);t,train,val=build(case)
        if observed:measure_gradient_diagnostics(t,train,val,epochs=5)
        else:t.fit(train,val,epochs=5)
        states.append(snapshot(t))
    equal(*states)


@pytest.mark.parametrize('before,after,change',[(1.,2.,'increased'),(2.,1.,'decreased'),(1.,1.,'unchanged'),(0.,0.,'unchanged')])
def test_scale_classification_is_not_execution(before,after,change):
    assert module._scale_change(before,after)==change


def test_native_sparse_training_returns_partial_not_complete_norm():
    class SparseModel(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embedding=torch.nn.Embedding(4,3,sparse=True);self.head=torch.nn.Linear(3,2)
        def forward(self,x):return self.head(self.embedding(x.long().reshape(-1)))
    m=SparseModel();t=Trainer(m,optimizer=torch.optim.SGD(m.parameters(),lr=.01),loss_fn=torch.nn.CrossEntropyLoss())
    train=DataLoader(TensorDataset(torch.tensor([[0.],[0.],[1.]]),torch.tensor([0,1,0])),batch_size=3)
    r=measure_gradient_diagnostics(t,train,epochs=1).to_dict()
    assert r['status']=='partial' and r['observations'][0]['preclip']['summary']['global_l2']['value'] is None


def test_checkpoint_resume_early_stop_completed(tmp_path):
    from hyso.training import resume
    states=[]
    for observed in (False,True):
        seed(61);t,train,val=build(CRITICAL);path=tmp_path/(str(observed)+'.pth');native=resume.save_last
        class Stop(Exception):pass
        def cut(*a):
            native(*a)
            if len(a[0].history['train_loss'])==2:raise Stop()
        with patch.object(resume,'save_last',cut),pytest.raises(Stop):
            if observed:measure_gradient_diagnostics(t,train,val,epochs=5,save_last_to=path)
            else:t.fit(train,val,epochs=5,save_last_to=path)
        t,train,val=build(CRITICAL)
        if observed:r=measure_gradient_diagnostics(t,train,val,epochs=5,resume_from=path)
        else:t.fit(train,val,epochs=5,resume_from=path)
        states.append(snapshot(t))
    equal(*states);assert r.to_dict()['actual']['epochs']==3 and r.to_dict()['observations'][0]['attempt_id']>1
    t,train,val=build(dict(CRITICAL,early='terminal',best=True));path=tmp_path/'done.pth'
    r=measure_gradient_diagnostics(t,train,val,epochs=5,save_last_to=path);assert r.to_dict()['actual']['epochs']==2
    t,train,val=build(dict(CRITICAL,early='terminal',best=True))
    r=measure_gradient_diagnostics(t,train,val,epochs=5,resume_from=path)
    assert r.to_dict()['status']=='no_new_attempts' and r.to_dict()['observations']==[]


@pytest.mark.parametrize('failure',['budget','tensor_budget','forward','reduction','optimizer','event','nonfinite'])
def test_gc_disabled_failure_cleanup(failure,monkeypatch):
    enabled=gc.isenabled();gc.disable()
    try:
        t=make();ref=weakref.ref(t.model);parameter=weakref.ref(next(t.model.parameters()))
        def fail(*a,**kw):raise RuntimeError('injected failure')
        if failure=='forward':t.model.register_forward_pre_hook(fail)
        elif failure=='reduction':monkeypatch.setattr(module,'_tensor_gradient',fail)
        elif failure=='optimizer':t.optimizer.register_step_pre_hook(fail)
        elif failure=='event':monkeypatch.setattr(module._GradientRecorder,'_event',fail)
        elif failure=='nonfinite':next(t.model.parameters()).register_hook(lambda g:torch.full_like(g,float('inf')))
        protocol=GradientDiagnosticsProtocol(max_attempts=1 if failure=='budget' else 10000,max_tensor_records=1 if failure=='tensor_budget' else 100000)
        pre=set(t.optimizer._optimizer_step_pre_hooks);post=set(t.optimizer._optimizer_step_post_hooks)
        with pytest.raises((RuntimeError,GradientDiagnosticsError,FloatingPointError)):
            measure_gradient_diagnostics(t,data(),epochs=2,protocol=protocol)
        assert set(t.optimizer._optimizer_step_pre_hooks)==pre and set(t.optimizer._optimizer_step_post_hooks)==post
        assert not hasattr(t,'_stage_recorder') and not hasattr(t,'_timing_recorder')
        del t
        assert ref() is None and parameter() is None
    finally:
        if enabled:gc.enable()


def test_gc_disabled_success_and_no_default_traversal(monkeypatch):
    enabled=gc.isenabled();gc.disable()
    try:
        t=make();ref=weakref.ref(t.model);r=measure_gradient_diagnostics(t,data(),epochs=2);del t
        assert ref() is None and r.to_dict()['status']=='measured'
    finally:
        if enabled:gc.enable()
    def fail(*a,**kw):raise AssertionError('default traversal')
    monkeypatch.setattr(module,'_snapshot',fail)
    make().fit(data(),epochs=2)
    from hyso.performance import measure_training,measure_training_detailed,measure_training_memory
    measure_training(make(),data(),epochs=2);measure_training_detailed(make(),data(),epochs=2);measure_training_memory(make(),data(),epochs=2)


@pytest.mark.parametrize('mutation',['norm','finite','samples','micro','applied','scale','timing','postclip','source','instrumentation','missing','recovery','disabled_scale'])
def test_reject_tamper_with_recomputed_checksum(mutation):
    d=measure_gradient_diagnostics(make(),data(),epochs=2).to_dict();r=d['observations'][0]
    if mutation=='norm':r['preclip']['summary']['global_l2']['value']+=1
    elif mutation=='finite':r['preclip']['tensors'][0]['gradient']['finite']=False
    elif mutation=='samples':r['samples']+=1
    elif mutation=='micro':r['microbatch_ids']=[99]
    elif mutation=='applied':r['applied']=False
    elif mutation=='scale':r['scale_change']='increased'
    elif mutation=='timing':r['preclip']['query_start_ns']=0
    elif mutation=='postclip':r['postclip']=None
    elif mutation=='source':d['source_sha256']='0'*64
    elif mutation=='instrumentation':d['instrumentation']['scalar_reads']=0
    elif mutation=='recovery':d['summary']['recovered_after_skip_attempt_ids']=[1]
    elif mutation=='disabled_scale':r['scale_before']=r['scale_after']=2.
    else:del r['scale_before']
    with pytest.raises(ValueError):GradientDiagnosticsResult(d)
    d['sha256']=module._hash({k:v for k,v in d.items() if k!='sha256'})
    with pytest.raises(ValueError):GradientDiagnosticsResult(d)


def test_immutable_export_manager_and_tracking_failure(tmp_path,monkeypatch):
    from hyso.experiments import ExperimentManager
    t=make();r=measure_gradient_diagnostics(t,data(),epochs=2);before=snapshot(t)
    changed=r.to_dict();changed['status']='fake';assert r.to_dict()['status']=='measured'
    with pytest.raises((AttributeError,TypeError)):r._payload='bad'
    store=tmp_path/'store';m=ExperimentManager(store);m.close();ExperimentManager.migrate_store(store);m=ExperimentManager(store)
    run=m.create_run(m.create_experiment('P25')['id'],'training',{});run.start()
    def forbidden(*a,**kw):raise AssertionError('retraining')
    monkeypatch.setattr(t,'fit',forbidden)
    for kind in ('json','csv'):
        path=tmp_path/('diagnostics.'+kind);getattr(r,'to_'+kind)(path);raw=path.read_bytes()
        with pytest.raises(FileExistsError):getattr(r,'to_'+kind)(path)
        assert path.read_bytes()==raw
        getattr(r,'to_'+kind)(path,overwrite=True)
        artifact=run.add_artifact(path,kind='report',ownership='copy',metadata={'source_segment_id':r.to_dict()['source_segment_id']})
        assert m.verify_artifact(artifact['id'])['status']=='verified' and artifact['expected_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    for row in r.to_dict()['observations']:
        run.log_metric(name='gradient.preclip_l2',value=row['preclip']['summary']['global_l2']['value'],unit='norm',source='P25',step=row['attempt_id'],step_kind='optimizer_attempt')
    with patch.object(run,'log_metric',side_effect=RuntimeError('tracking failure')),pytest.raises(RuntimeError):
        run.log_metric(name='test',value=1,unit='norm',source='P25',epoch=1)
    equal(before,snapshot(t));run.finish();identity=run.id;m.close();m=ExperimentManager(store)
    assert len(m.get_run(identity).read_metrics())==4 and len(m.list_artifacts(identity))==2;m.close()
    d=r.to_dict();d['status']='fake';object.__setattr__(r,'_payload',json.dumps(d))
    with pytest.raises(ValueError):r.to_csv(tmp_path/'bad.csv')
    assert not (tmp_path/'bad.csv').exists()


@pytest.mark.parametrize('kwargs',[{'max_attempts':0},{'max_tensor_records':True},{'chunk_elements':1048577},{'observe_postclip':1}])
def test_protocol_bounds(kwargs):
    with pytest.raises((ValueError,TypeError)):GradientDiagnosticsProtocol(**kwargs)


def test_postclip_not_requested_and_fresh_admission():
    t=make();d=measure_gradient_diagnostics(t,data(),epochs=2,protocol=GradientDiagnosticsProtocol(observe_postclip=False)).to_dict()
    assert all(r['postclip'] is None and r['postclip_reason']=='not_requested' and r['clipping']['status']=='measured' for r in d['observations'])
    with pytest.raises(ValueError):measure_gradient_diagnostics(t,data(),epochs=2)


def test_global_nan_inf_counts_and_shared_gradient_storage():
    p=torch.nn.Parameter(torch.zeros(2));q=torch.nn.Parameter(torch.zeros(2))
    p.grad=torch.tensor([3.,4.]);q.grad=p.grad.view(2)
    opt=torch.optim.SGD([p,q],lr=.01);protocol=GradientDiagnosticsProtocol()
    s=module._snapshot(opt,protocol,10)['summary']
    assert s['global_l2']['value']==pytest.approx(math.sqrt(50)) and s['with_gradient']==2
    p.grad=torch.tensor([float('nan'),1.]);q.grad=torch.tensor([float('inf'),-float('inf')])
    s=module._snapshot(opt,protocol,10)['summary']
    assert s['nan_tensor_count']==1 and s['inf_tensor_count']==1 and s['finite'] is False
    assert s['global_l2']['value'] is None and s['global_l2']['status']=='nonfinite'


def test_chunk_scratch_bound_and_no_full_cpu_copy(monkeypatch):
    calls=[];native=torch.Tensor.to
    def to(self,*args,**kwargs):
        if kwargs.get('dtype') is torch.float64:calls.append(self.numel())
        return native(self,*args,**kwargs)
    def forbidden(*a,**kw):raise AssertionError('CPU gradient copy')
    monkeypatch.setattr(torch.Tensor,'to',to);monkeypatch.setattr(torch.Tensor,'cpu',forbidden)
    d=tensor_info(torch.ones(101),chunk=7)
    assert d['global_l2']==pytest.approx(math.sqrt(101),rel=1e-15)
    assert max(calls)<=7 and sum(calls)==101


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.])
def test_invalid_scaler_value(value):
    with pytest.raises(GradientDiagnosticsError):module._scale_change(1.,value)


@pytest.mark.parametrize('budget,passes',[(31,False),(32,True)])
def test_exact_tensor_record_budget(budget,passes):
    t=make()
    if passes:
        d=measure_gradient_diagnostics(t,data(),epochs=2,protocol=GradientDiagnosticsProtocol(max_tensor_records=budget)).to_dict()
        assert sum(len(r[k]['tensors']) for r in d['observations'] for k in ('preclip','postclip'))==budget
    else:
        with pytest.raises(GradientDiagnosticsError):measure_gradient_diagnostics(t,data(),epochs=2,protocol=GradientDiagnosticsProtocol(max_tensor_records=budget))


def test_gradient_guide():
    import re
    path=Path(__file__).resolve().parents[1]/'docs/guides/performance-gradient-amp.md'
    namespace={'__name__':'p25_guide'}
    for block in re.findall(r'```python\n(.*?)```',path.read_text(encoding='utf-8'),re.S):
        exec(compile(block,str(path),'exec'),namespace)
    assert namespace['result'].to_dict()['summary']['attempts']==2
