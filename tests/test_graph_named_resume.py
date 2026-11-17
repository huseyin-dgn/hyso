"""Two named tensors: existing v2 strict epoch replay, not a new serializer."""
import copy
import pytest
import torch
from hyso import TrainingConfig
from hyso.training import NamedTensorDataset, resume
from hyso.checkpoint import TrainingCheckpointError
from test_graph_multi_input_training import build
from test_graph_training_resume import cpu_only
from test_trainer_checkpoint_restore import assert_state


def data(dtype=torch.float32,task='multiclass',swap=False):
    x=torch.arange(36,dtype=dtype).reshape(4,1,3,3)/36
    y=torch.tensor([0,1,1,0],dtype=torch.long if task=='multiclass' else dtype)
    inputs={'right':x*.3+.1,'left':x}
    if swap:inputs=dict(reversed(list(inputs.items())))
    return NamedTensorDataset(inputs,y)


def session(model,path=None,source=None,optimizer='sgd',task='multiclass',dtype=torch.float32,dataset=None,
            epochs=2,scheduler='step',**options):
    c=TrainingConfig(epochs=epochs,batch_size=2,shuffle=True,validation=True,task=task,
        precision='fp64' if dtype==torch.float64 else 'fp32',optimizer=optimizer,
        optimizer_params={'momentum':.8} if optimizer=='sgd' else None,
        scheduler=scheduler,scheduler_params={'step_size':1,'gamma':.7} if scheduler=='step' else None,
        save_last_to=None if path is None else str(path),resume_from=None if source is None else str(source),**options)
    ds=data(dtype,task) if dataset is None else dataset
    train,val=c.make_loaders(ds,ds)
    return c.prepare(model,train,val,model_config=model.graph.to_config())


@pytest.mark.parametrize('optimizer,task,dtype,scheduler', [('sgd','multiclass',torch.float32,None),
    ('adam','binary',torch.float32,'step'),('adamw','multiclass',torch.float64,'step')])
def test_named_v2_resume_matches_uninterrupted(tmp_path,monkeypatch,optimizer,task,dtype,scheduler):
    torch.manual_seed(72)
    model=build(dtype,task);baseline=session(model,optimizer=optimizer,task=task,dtype=dtype,scheduler=scheduler)
    expected=baseline.fit();rng=torch.get_rng_state().clone()
    torch.manual_seed(72)
    model=build(dtype,task);path=tmp_path/'last.pth'
    interrupted=session(model,path=path,optimizer=optimizer,task=task,dtype=dtype,scheduler=scheduler)
    native=resume.save_last
    def stop(*args):
        native(*args);raise InterruptedError('saved epoch one')
    with monkeypatch.context() as patch:
        patch.setattr(resume,'save_last',stop)
        with pytest.raises(InterruptedError):interrupted.fit()
    payload=torch.load(path,weights_only=True)
    assert payload['format_version']==2 and payload['completed_epochs']==1
    assert all(k.startswith('graph.') for k in payload['model']['state_dict'])
    restored=build(dtype,task);ids=[id(p) for p in restored.parameters()]
    continued=session(restored,path=path,source=path,optimizer=optimizer,task=task,dtype=dtype,
                      dataset=data(dtype,task,swap=True),scheduler=scheduler)
    result=continued.fit()
    assert result.history==expected.history
    assert_state(restored.state_dict(),baseline.trainer.model.state_dict())
    assert_state(continued.trainer.optimizer.state_dict(),baseline.trainer.optimizer.state_dict())
    if scheduler is None:assert continued.trainer.scheduler is None
    else:assert_state(continued.trainer.scheduler.state_dict(),baseline.trainer.scheduler.state_dict())
    assert ids==[id(p) for p in restored.parameters()]
    assert torch.equal(torch.get_rng_state(),rng)
    assert torch.load(path,weights_only=True)['completed_epochs']==2


@pytest.mark.parametrize('change', ['names','data','architecture','state','budget'])
def test_named_resume_mismatch_is_transactional(tmp_path,change):
    path=tmp_path/'last.pth';model=build();session(model,path=path).fit()
    model=build();ds=data();options={}
    if change=='names':ds.input_names=('left','wrong')
    if change=='data':ds.tensors[1].add_(.1)
    if change=='architecture':model.graph.nodes['node_a'].stride=(2,2)
    if change=='state':
        payload=torch.load(path,weights_only=True)
        next(iter(payload['model']['state_dict'].values())).add_(1)
        torch.save(payload,path)
    before=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone();s=None
    with pytest.raises((TrainingCheckpointError,ValueError)):
        s=session(model,source=path,dataset=ds,epochs=3 if change=='budget' else 2)
        s.fit()
    assert_state(model.state_dict(),before)
    assert torch.equal(torch.get_rng_state(),rng)
    if s is not None:
        assert not s.trainer.optimizer.state and s.trainer.update_count==0


@pytest.mark.parametrize('problem',['input_shape','target','collate'])
def test_named_checkpoint_data_rejected_before_updates(tmp_path,problem):
    m=build();ds=data();c=TrainingConfig(epochs=1,batch_size=2,validation=False,save_last_to=str(tmp_path/'last.pth'))
    if problem=='input_shape':ds.tensors=(ds.tensors[0][:,:,:2,:],*ds.tensors[1:])
    if problem=='target':ds.tensors=(*ds.tensors[:-1],ds.tensors[-1].float())
    train,_=c.make_loaders(ds)
    if problem=='collate':train.collate_fn=lambda batch:batch
    before=copy.deepcopy(m.state_dict())
    with pytest.raises(TrainingCheckpointError):c.prepare(m,train,model_config=m.graph.to_config())
    assert_state(m.state_dict(),before)
