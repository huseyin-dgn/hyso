"""Native single-process DataLoader training, exact state and artifact contracts."""
from fixture_support import example_fixture
import copy
import csv
import json
import hashlib
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from torch.utils.data import DataLoader,TensorDataset,Dataset
from hyso.performance import (measure_training_with_loader,measure_training_detailed,measure_training,
    DataLoaderAnalysisProtocol,DataLoaderAnalysisResult,TrainingTimingError)
from hyso.performance.dataloader import _LoaderMixin
from trainer_replay_support import seed,equal,snapshot,build,CRITICAL,CASES


def loader(shuffle=False):
    return DataLoader(TensorDataset(torch.arange(15).reshape(5,3).float()/20,torch.tensor([0,1,0,1,0])),
        batch_size=2,shuffle=shuffle,generator=torch.Generator().manual_seed(728))


def trainer(epochs=2):
    from hyso.training import Trainer
    m=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.2),torch.nn.Linear(4,2))
    o=torch.optim.AdamW(m.parameters(),lr=.01)
    s=torch.optim.lr_scheduler.OneCycleLR(o,.02,total_steps=epochs*2,cycle_momentum=False)
    return Trainer(m,optimizer=o,loss_fn=torch.nn.CrossEntropyLoss(),scheduler=s,accum_steps=2,ema_decay=.8,grad_clip=.1)


@pytest.mark.parametrize('epochs',[1,2,3])
@pytest.mark.parametrize('shuffle',[False,True])
@pytest.mark.parametrize('validation',[False,True])
@pytest.mark.parametrize('stages',[False,True])
def test_exact_native_loader_state_order_and_counts(epochs,shuffle,validation,stages):
    outputs=[]; observed=[]
    for mode in ('baseline','observed'):
        seed(81);t=trainer(epochs);train=loader(shuffle);val=loader() if validation else None
        content=[];h=t.model.register_forward_pre_hook(lambda m,args:content.append(args[0].detach().clone()))
        try:
            if mode=='baseline':t.fit(train,val,epochs=epochs)
            else: result=measure_training_with_loader(t,train,val,epochs=epochs,stages=stages)
        finally:h.remove()
        outputs.append((snapshot(t),train.generator.get_state(),None if val is None else val.generator.get_state(),content))
        assert all(not hasattr(t,k) for k in ('_loader_recorder','_timing_recorder','_stage_recorder'))
        assert not t.optimizer._optimizer_step_post_hooks and not t.optimizer._optimizer_step_pre_hooks
    equal(*outputs)
    d=result.loader.to_dict()
    assert d['status']=='partial' and d['worker_service']['value'] is None
    assert d['source_p19']==result.timing.to_dict()
    assert (result.stages is not None)==stages
    assert all(e['samples']==5 and e['batches']==3 for e in d['summaries'])
    training=[r for r in d['observations'] if r['phase']=='training' and r['kind']=='batch']
    assert [r['samples'] for r in training]==[2,2,1]*epochs
    assert [r['attempt_id'] for r in training]==[a for i in range(epochs) for a in (i*2+1,i*2+1,i*2+2)]
    assert result.throughput.to_dict()['segment']['completed']['samples']==sum(r['samples'] for r in training)
    for s in d['summaries']:
        assert s['iterator_init_ms']>0 and s['next_wait_ms']>0
        assert s['delivered_wait']['p99']['value'] is None
        if s['phase']=='training':assert 0<=s['training_wall_wait_fraction']['value']<=1


@pytest.mark.parametrize('case',CASES)
def test_loader_native_replay_matrix(case):
    outputs=[]
    for observed in (False,True):
        seed(45);t,train,val=build(case)
        if observed:measure_training_with_loader(t,train,val,epochs=5)
        else:t.fit(train,val,epochs=5)
        outputs.append(snapshot(t))
    equal(*outputs)


def test_default_p19_p20_have_no_loader_observer(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('default loader observation')
    monkeypatch.setattr(_LoaderMixin,'observe_loader',forbidden)
    trainer().fit(loader(),epochs=2)
    measure_training(trainer(),loader(),epochs=2)
    measure_training_detailed(trainer(),loader(),epochs=2)


@pytest.mark.parametrize('failure',['dataset','model','budget','empty','invalid'])
def test_error_cleanup(failure):
    t=trainer();train=loader()
    if failure=='dataset':
        class Bad(Dataset):
            def __len__(self):return 5
            def __getitem__(self,index):
                if index>=2:raise RuntimeError('dataset failure')
                return torch.ones(3),0
        train=DataLoader(Bad(),batch_size=2)
    elif failure=='model':t.model[0].register_forward_pre_hook(lambda *a:(_ for _ in ()).throw(RuntimeError('model failure')))
    elif failure=='empty':train=DataLoader(TensorDataset(torch.zeros(0,3),torch.zeros(0,dtype=torch.long)),batch_size=2)
    elif failure=='invalid':train=[(torch.ones(2,3),torch.tensor([0,1]))]
    with pytest.raises((RuntimeError,ValueError,TypeError)):
        measure_training_with_loader(t,train,epochs=2,loader_protocol=DataLoaderAnalysisProtocol(max_records=2 if failure=='budget' else 100000))
    assert all(not hasattr(t,k) for k in ('_loader_recorder','_timing_recorder','_stage_recorder'))


@pytest.mark.parametrize('field',['wait','sample','id','delivery','summary','worker','hash','stage'])
def test_tamper_rejection(field):
    d=measure_training_with_loader(trainer(),loader(),epochs=2).loader.to_dict()
    row=next(r for r in d['observations'] if r['kind']=='batch')
    if field=='wait':row['host_ms']+=1
    elif field=='sample':row['samples']+=1
    elif field=='id':row['microbatch_id']=100
    elif field=='delivery':row['delivery_ns']+=1
    elif field=='summary':d['summaries'][0]['samples']+=1
    elif field=='worker':d['worker_service']['value']=0
    elif field=='hash':d['source_sha256']='bad'
    else:d['stage_links'][0]['compute']['forward']['host_ms']+=1
    with pytest.raises(ValueError):DataLoaderAnalysisResult(d)


def test_export_roundtrip_and_no_clobber(tmp_path):
    r=measure_training_with_loader(trainer(),loader(),epochs=2).loader
    r.to_json(tmp_path/'loader.json');r.to_csv(tmp_path/'loader.csv')
    assert DataLoaderAnalysisResult(json.loads((tmp_path/'loader.json').read_text())).to_dict()==r.to_dict()
    d=r.to_dict();d['observations'].clear();assert r.to_dict()['observations']
    with pytest.raises((AttributeError,TypeError)):r._payload='bad'
    for method,path in ((r.to_json,tmp_path/'loader.json'),(r.to_csv,tmp_path/'loader.csv')):
        with pytest.raises(FileExistsError):method(path)
    with (tmp_path/'loader.csv').open(newline='') as f:assert len(list(csv.DictReader(f)))==10
    bad=r.to_dict();bad['source_sha256']='=DANGEROUS()'
    object.__setattr__(r,'_payload',json.dumps(bad))
    with pytest.raises(ValueError):r.to_csv(tmp_path/'bad.csv')
    assert not (tmp_path/'bad.csv').exists()
    from hyso.performance.dataloader import _csv_text
    for formula in ('=1+1',' +cmd','-formula','@SUM(A1)'):
        assert "'"+formula in _csv_text(['x'],[dict(x=formula)])


def test_resume_early_stop_and_completed(tmp_path):
    import hyso.training.resume as resume
    outputs=[]
    for observed in (False,True):
        seed(93);t,train,val=build(CRITICAL);path=tmp_path/f'{observed}.pth';original=resume.save_last
        class Stop(Exception):pass
        def cut(*args):
            original(*args)
            if len(args[0].history['train_loss'])==2:raise Stop()
        with patch.object(resume,'save_last',cut),pytest.raises(Stop):
            if observed:measure_training_with_loader(t,train,val,epochs=5,save_last_to=path)
            else:t.fit(train,val,epochs=5,save_last_to=path)
        t,train,val=build(CRITICAL)
        if observed:r=measure_training_with_loader(t,train,val,epochs=5,resume_from=path,save_last_to=tmp_path/'done.pth')
        else:t.fit(train,val,epochs=5,resume_from=path)
        outputs.append(snapshot(t))
    equal(*outputs)
    assert [e['epoch_id'] for e in r.timing.to_dict()['epochs']]==[3,4,5]
    t,train,val=build(CRITICAL);done=measure_training_with_loader(t,train,val,epochs=5,resume_from=tmp_path/'done.pth')
    assert done.loader.to_dict()['observations']==[]
    assert done.throughput.to_dict()['segment']['rates']['samples_per_second']['value'] is None
    states=[]
    for observed in (False,True):
        seed(36);t,train,val=build(dict(CRITICAL,early='terminal',best=True))
        if observed:r=measure_training_with_loader(t,train,val,epochs=5)
        else:t.fit(train,val,epochs=5)
        states.append(snapshot(t))
    equal(*states);assert r.loader.to_dict()['actual']['epochs']==2


def test_manager_after_fit_no_retraining(tmp_path,monkeypatch):
    from hyso.experiments import ExperimentManager
    t=trainer();result=measure_training_with_loader(t,loader(),epochs=2);before=snapshot(t)
    def forbidden(*a,**kw):raise AssertionError('retraining')
    monkeypatch.setattr(t,'fit',forbidden)
    root=tmp_path/'store';m=ExperimentManager(root);m.close();ExperimentManager.migrate_store(root);m=ExperimentManager(root)
    run=m.create_run(m.create_experiment('P21')['id'],'training',{});run.start()
    for kind in ('json','csv'):
        path=tmp_path/('loader.'+kind);getattr(result.loader,'to_'+kind)(path)
        a=run.add_artifact(path,kind='report',ownership='copy',metadata=dict(source_segment_id=result.timing.to_dict()['segment_id']))
        assert m.verify_artifact(a['id'])['status']=='verified' and a['expected_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    for s in result.loader.to_dict()['summaries']:
        if s['phase']=='training':run.log_metric(name='loader.next_wait',value=s['next_wait_ms'],unit='ms',source='P21',epoch=s['epoch_id'])
    with patch.object(run,'log_metric',side_effect=RuntimeError('tracking error')),pytest.raises(RuntimeError):
        run.log_metric(name='test',value=1,unit='ms',source='P21',epoch=1)
    equal(before,snapshot(t));run.finish();identity=run.id;m.close();m=ExperimentManager(root)
    assert len(m.get_run(identity).read_metrics())==2
    assert len(m.list_artifacts(identity))==2
    m.close()


def test_fake_clock_all_loader_and_stage_boundaries(monkeypatch):
    import hyso.performance.dataloader as module
    ticks=iter(range(1000,100000000,100))
    monkeypatch.setattr(module.time,'perf_counter_ns',lambda:next(ticks))
    r=measure_training_with_loader(trainer(1),loader(),epochs=1)
    d=r.loader.to_dict()
    assert all(row['host_ms']==.0001 for row in d['observations'])
    delivered={x['microbatch_id']:x for x in d['observations'] if x['kind']=='batch'}
    for link in d['stage_links']:
        prep=link['input_preparation'];compute=link['compute']
        assert delivered[link['microbatch_id']]['delivery_ns']<=prep['start_ns']<=prep['end_ns']<=compute['forward']['start_ns']<=compute['forward']['end_ns']<=compute['backward']['start_ns']<=compute['backward']['end_ns']
        assert prep['host_ms']==compute['forward']['host_ms']==compute['backward']['host_ms']==.0001


@pytest.mark.parametrize('cap,success',[(4,False),(5,True)])
def test_exact_loader_budget(cap,success):
    t=trainer(1)
    if success:
        r=measure_training_with_loader(t,loader(),epochs=1,loader_protocol=DataLoaderAnalysisProtocol(max_records=cap))
        assert len(r.loader.to_dict()['observations'])==cap
    else:
        with pytest.raises(TrainingTimingError):measure_training_with_loader(t,loader(),epochs=1,loader_protocol=DataLoaderAnalysisProtocol(max_records=cap))


def test_dataloader_guide():
    import re
    guide=example_fixture('docs/guides/performance-dataloader.md')
    namespace={'__name__':'p21_guide_test'}
    for block in re.findall(r'```python\n(.*?)```',guide.read_text(encoding='utf-8'),re.S):
        exec(compile(block,str(guide),'exec'),namespace)
    assert namespace['data']['summaries'][0]['samples']==5
