"""Deterministic tiny CPU configuration/Trainer integration; no performance suite."""
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader

from hyso import TrainingConfig,TrainingConfigError,ModelBuilder,Trainer,ExperimentManager


def dataset():
    g=torch.Generator().manual_seed(41)
    return TensorDataset(torch.randn(5,1,5,5,generator=g),torch.tensor([0,1,0,1,1]))


def modular():
    return ModelBuilder.modular(input_channels=1,num_classes=2,widths=(3,),repeats=(1,),strides=(1,),norm=None)


def test_config_roundtrip_is_immutable_plain_and_separate_from_model(tmp_path):
    p={'momentum':.8}
    c=TrainingConfig(epochs=2,batch_size=2,optimizer='sgd',optimizer_params=p)
    p['momentum']=.1
    path=tmp_path/'training.json';c.to_json(path)
    assert TrainingConfig.from_json(path).to_dict()==c.to_dict()
    assert c.to_dict()['optimizer_params']['momentum']==.8
    with pytest.raises(FileExistsError):c.to_json(path)
    with pytest.raises(FrozenInstanceError):c._payload='bad'
    assert 'model' not in c.to_dict() and 'dataset' not in c.to_dict()


@pytest.mark.parametrize('args',[
    dict(epochs=True),dict(batch_size=0),dict(optimizer='LBFGS'),dict(learning_rate=0),
    dict(learning_rate=float('nan')),dict(learning_rate=10**1000),dict(precision='bf16'),
    dict(precision='amp_fp16'),dict(device='cuda'),dict(device='mps'),dict(num_workers=1),
    dict(accum_steps=0),dict(grad_clip=-1),dict(ema_decay=2),dict(optimizer_params={'fused':True}),
    dict(optimizer='sgd',optimizer_params={'nesterov':True}),dict(optimizer_params={'betas':[.9,1.]}),
    dict(scheduler='lambda'),dict(scheduler_params={'step_size':2}),dict(scheduler='onecycle'),
    dict(scheduler='plateau'),dict(early_stopping={}),dict(validation=True,early_stopping={'patience':0}),
    dict(scheduler='step',scheduler_params={'gamma':-1}),dict(save_last_to=''),
    dict(optimizer_params={'weight_decay':nn.Linear(1,1)}),
])
def test_invalid_requests_fail_without_allocating_model_or_querying_gpu(args,monkeypatch):
    def forbidden(*a,**k):pytest.fail('configuration validation queried CUDA')
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    with pytest.raises(TrainingConfigError):TrainingConfig(**args)


@pytest.mark.parametrize('raw',['{"format":"x","format":"y"}','{"x":NaN}','{"x":1e999}',
                              '{"constructor":"os.system"}', '[]'])
def test_json_rejects_unsafe_or_unknown_content(tmp_path,raw):
    p=tmp_path/'bad.json';p.write_text(raw,encoding='utf-8')
    with pytest.raises(TrainingConfigError):TrainingConfig.from_json(p)
    d=TrainingConfig().to_dict();d['dataset']=[]
    with pytest.raises(TrainingConfigError):TrainingConfig.from_dict(d)


@pytest.mark.parametrize('optimizer',['sgd','adam','adamw'])
@pytest.mark.parametrize('scheduler',[None,'step','cosine','onecycle','plateau'])
def test_supported_native_optimizers_and_schedulers_use_one_trainer(optimizer,scheduler):
    torch.manual_seed(12)
    cfg=TrainingConfig(epochs=2,batch_size=2,optimizer=optimizer,scheduler=scheduler,validation=True,
        scheduler_params={'max_lr':.02} if scheduler=='onecycle' else None,accum_steps=2,grad_clip=1.,ema_decay=.8)
    train,val=cfg.make_loaders(dataset(),dataset())
    m=modular().build();s=cfg.prepare(m,train,val)
    assert type(s.trainer) is Trainer
    assert s.evaluation_model is s.trainer.evaluation_model
    assert s.evaluation_model is not m
    before=s.settings()
    assert before['requested']==cfg.to_dict()
    if scheduler=='onecycle':
        assert before['resolved']['planned_optimizer_updates']==4
        assert before['effective']['optimizer_groups'][0]['lr']!=before['requested']['learning_rate']
    result=s.fit().to_dict()
    assert result['completed_history_epochs']==2 and result['applied_updates']==4
    assert len(result['epoch_events'])==2 and all(e['interval_seconds']>=0 for e in result['epoch_events'])
    assert all(v is not None for v in result['history']['val_loss'])
    with pytest.raises(TrainingConfigError):s.fit()


def test_same_native_fit_state_as_direct_trainer():
    torch.manual_seed(17)
    b=modular();m=b.build();other=b.build();other.load_state_dict(m.state_dict())
    c=TrainingConfig(epochs=2,batch_size=2,optimizer='sgd',learning_rate=.02,accum_steps=2)
    train,val=c.make_loaders(dataset())
    s=c.prepare(m,train);r=s.fit()
    direct=Trainer(other,optimizer=torch.optim.SGD(other.parameters(),lr=.02),loss_fn=nn.CrossEntropyLoss(),device='cpu',accum_steps=2)
    history=direct.fit(train,epochs=2)
    assert r.history==history
    for name,value in m.state_dict().items():assert torch.equal(value,other.state_dict()[name])


def test_built_classifier_and_external_model_do_not_require_resume_profile():
    c=TrainingConfig(epochs=1,batch_size=2)
    train,_=c.make_loaders(dataset())
    for m in (ModelBuilder((1,5,5),2).add('conv2d',in_channels=1,out_channels=2).build(),
              nn.Sequential(nn.Flatten(),nn.Linear(25,2))):
        result=c.prepare(m,train).fit()
        assert result.to_dict()['applied_updates']==3
        assert result.history['val_loss']==[None]


def test_loader_precision_and_validation_mismatches_rejected_before_updates():
    m=modular().build();c=TrainingConfig(batch_size=2)
    with pytest.raises(TrainingConfigError):c.prepare(m,DataLoader(dataset(),batch_size=3))
    train,_=c.make_loaders(dataset())
    with pytest.raises(TrainingConfigError):c.prepare(m.double(),train)
    with pytest.raises(TrainingConfigError):TrainingConfig(validation=True,batch_size=2).prepare(m.float(),train)
    with pytest.raises(TrainingConfigError):c.make_loaders(dataset(),dataset())


def test_fp64_binary_and_explicit_observer():
    cfg=TrainingConfig(epochs=1,batch_size=2,precision='fp64',task='binary')
    ds=TensorDataset(torch.randn(4,2,dtype=torch.float64),torch.tensor([0.,1.,0.,1.],dtype=torch.float64))
    train,_=cfg.make_loaders(ds);events=[]
    m=nn.Linear(2,1).double();result=cfg.prepare(m,train,on_epoch=events.append).fit()
    assert len(events)==1 and result.to_dict()['settings']['effective']['model_floating_dtypes']==['torch.float64']


def test_early_stop_and_observer_failure_do_not_retry():
    c=TrainingConfig(epochs=4,batch_size=2,validation=True,early_stopping={'patience':1,'delta':1e6})
    train,val=c.make_loaders(dataset(),dataset());s=c.prepare(modular().build(),train,val)
    assert s.fit().to_dict()['completed_history_epochs']==2
    c=TrainingConfig(epochs=2,batch_size=2);train,_=c.make_loaders(dataset())
    def fail(row):raise RuntimeError('observer failed')
    s=c.prepare(modular().build(),train,on_epoch=fail)
    with pytest.raises(RuntimeError,match='observer failed'):s.fit()
    assert s.settings()['state']=='failed'
    with pytest.raises(TrainingConfigError):s.fit()


def test_config_checkpoint_uses_unchanged_modular_profile(tmp_path):
    b=modular();path=tmp_path/'last.pth'
    c=TrainingConfig(epochs=1,batch_size=2,save_last_to=str(path))
    train,_=c.make_loaders(dataset())
    result=c.prepare(b.build(),train,model_config=b.to_config()['params']).fit().to_dict()
    assert path.exists() and result['epoch_events']==[]
    assert result['epoch_timing_scope']=='unavailable_callback_free_checkpoint_profile'
    restore=TrainingConfig(epochs=1,batch_size=2,resume_from=str(path))
    restored=restore.prepare(b.build(),train,model_config=b.to_config()['params']).fit()
    assert restored.history==result['history']
    built=ModelBuilder((1,5,5),2).build()
    with pytest.raises(TrainingConfigError,match='BuiltClassifier'):
        c.prepare(built,train,model_config=built.to_config())
    with pytest.raises(TrainingConfigError,match='on_epoch'):
        c.prepare(b.build(),train,model_config=b.to_config()['params'],on_epoch=lambda row:None)


def test_settings_drift_and_onecycle_bad_plan_rejected():
    c=TrainingConfig(batch_size=2);train,_=c.make_loaders(dataset());s=c.prepare(modular().build(),train)
    s.trainer.optimizer.param_groups[0]['lr']=.5
    with pytest.raises(TrainingConfigError,match='changed'):s.fit()
    c=TrainingConfig(epochs=1,batch_size=2,scheduler='onecycle',scheduler_params={'max_lr':.1,'pct_start':.5})
    train,_=c.make_loaders(TensorDataset(torch.randn(4,1,5,5),torch.tensor([0,1,0,1])))
    with pytest.raises(TrainingConfigError,match='nondegenerate'):c.prepare(modular().build(),train)


@pytest.mark.parametrize('scheduler',[None,'step','cosine','plateau'])
def test_config_continues_original_trainer_checkpoint_without_schema_change(tmp_path,monkeypatch,scheduler):
    from hyso.training import resume
    b=modular();path=tmp_path/'legacy-last.pth'
    config=TrainingConfig(epochs=2,batch_size=2,optimizer='sgd',learning_rate=.02,
                          scheduler=scheduler,validation=scheduler=='plateau',resume_from=str(path))
    train,val=config.make_loaders(dataset(),dataset() if scheduler=='plateau' else None)
    def native_trainer():
        model=b.build()
        opt=torch.optim.SGD(model.parameters(),lr=.02)
        native=None
        if scheduler=='step':native=torch.optim.lr_scheduler.StepLR(opt,1)
        elif scheduler=='cosine':native=torch.optim.lr_scheduler.CosineAnnealingLR(opt,2)
        elif scheduler=='plateau':native=torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
        return Trainer(model,optimizer=opt,scheduler=native,loss_fn=nn.CrossEntropyLoss(),
                       device='cpu',model_config=b.to_config()['params'])
    torch.manual_seed(97);baseline=native_trainer();expected=baseline.fit(train,val,epochs=2)
    torch.manual_seed(97);original=native_trainer();save_last=resume.save_last
    def save_then_interrupt(*args):
        save_last(*args)
        raise InterruptedError('controlled completed epoch')
    with monkeypatch.context() as patch:
        patch.setattr(resume,'save_last',save_then_interrupt)
        with pytest.raises(InterruptedError):original.fit(train,val,epochs=2,save_last_to=path)
    session=config.prepare(b.build(),train,val,model_config=b.to_config()['params'])
    result=session.fit()
    assert result.history==expected
    assert result.to_dict()['applied_updates']==6
    for name,value in baseline.model.state_dict().items():
        assert torch.equal(value,session.trainer.model.state_dict()[name])


def test_early_stopping_actual_settings_and_mutation_guard():
    c=TrainingConfig(batch_size=2,validation=True,early_stopping={'patience':2})
    train,val=c.make_loaders(dataset(),dataset())
    s=c.prepare(modular().build(),train,val)
    assert s.settings()['effective']['early_stopping']==dict(patience=2,delta=0.,restore_best_weights=True)
    s.trainer.early_stopping.patience=3
    with pytest.raises(TrainingConfigError,match='changed'):s.fit()
    assert s.trainer.update_count==0


def test_result_logs_real_history_to_existing_manager(tmp_path):
    c=TrainingConfig(epochs=1,batch_size=2);train,_=c.make_loaders(dataset())
    s=c.prepare(modular().build(),train);result=s.fit()
    manager=ExperimentManager(tmp_path/'store')
    try:
        exp=manager.create_experiment('configured training')
        run=manager.create_run(exp['id'],'training',s.settings())
        run.start();records=result.log_to(run);run.finish()
        assert len(records)==3
        assert {r['name'] for r in run.read_metrics()}=={'train_loss','train_acc','epoch_observer_interval'}
        assert run.to_dict()['status']=='completed'
    finally:
        manager.close()
