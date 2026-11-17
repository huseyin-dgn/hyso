"""Existing strict v2 plans compose four explicitly named tensors."""
import copy
import pytest
import torch
from hyso import TrainingConfig, GraphModelBuilder
from hyso.training import NamedTargetDataset, resume
from hyso.checkpoint import TrainingCheckpointError
from test_graph_combined_evaluation import fixture
from test_graph_multi_loss import criteria
from test_graph_training_resume import cpu_only
from test_trainer_checkpoint_restore import assert_state


def session(graph,data,path=None,source=None,optimizer='sgd',binary=False,epochs=2,losses=None,weights=None):
    config=TrainingConfig(epochs=epochs,batch_size=2,optimizer=optimizer,validation=True,
        optimizer_params={'momentum':.8} if optimizer=='sgd' else None,
        save_last_to=None if path is None else str(path),resume_from=None if source is None else str(source))
    train,val=config.make_loaders(data,data)
    return config.prepare(graph,train,val,model_config=graph.to_config(),
        loss_fn=criteria(binary) if losses is None else losses,
        loss_weights={'one':.3,'two':1.2} if weights is None else weights)


@pytest.mark.parametrize('optimizer,binary',[('sgd',False),('adam',True)])
def test_combined_v2_resume_matches_uninterrupted_and_completed_noop(tmp_path,monkeypatch,optimizer,binary):
    torch.manual_seed(13520);graph,data=fixture(binary)
    baseline=session(graph,data,optimizer=optimizer,binary=binary);expected=baseline.fit()
    rng=torch.get_rng_state().clone()
    torch.manual_seed(13520);graph,data=fixture(binary);path=tmp_path/'last.pth'
    cut=session(graph,data,path=path,optimizer=optimizer,binary=binary)
    native=resume.save_last
    def stop(*args):native(*args);raise InterruptedError()
    with monkeypatch.context() as scoped:
        scoped.setattr(resume,'save_last',stop)
        with pytest.raises(InterruptedError):cut.fit()
    payload=torch.load(path,weights_only=True)
    assert payload['format_version']==2 and payload['completed_epochs']==1
    assert payload['metrics']['generic_trainer']['plan']['train_loader']['input_names']==['context','image']
    assert list(payload['model']['state_dict'])==list(graph.state_dict())
    graph,data=fixture(binary);identities=[id(p) for p in graph.parameters()]
    reordered=NamedTargetDataset(dict(reversed(list(zip(data.input_names,data.tensors[:2])))),
        dict(reversed(list(zip(data.target_names,data.tensors[2:])))))
    continued=session(graph,reordered,path=path,source=path,optimizer=optimizer,binary=binary,
        losses=dict(reversed(list(criteria(binary).items()))))
    actual=continued.fit()
    assert actual.history==expected.history
    assert torch.equal(torch.get_rng_state(),rng)
    assert identities==[id(p) for p in graph.parameters()]
    assert_state(graph.state_dict(),baseline.trainer.model.state_dict())
    assert_state(continued.trainer.optimizer.state_dict(),baseline.trainer.optimizer.state_dict())
    assert continued.trainer.scheduler is baseline.trainer.scheduler is None
    assert continued.trainer.update_count==baseline.trainer.update_count==4
    final=torch.load(path,weights_only=True)
    assert final['completed_epochs']==2
    untouched=path.read_bytes()
    graph,data=fixture(binary)
    completed=session(graph,data,source=path,optimizer=optimizer,binary=binary)
    assert completed.fit().history==actual.history and path.read_bytes()==untouched


@pytest.mark.parametrize('problem',['input_names','topology','outputs','target_names','target_mapping','loss_type',
    'weights','optimizer','optimizer_state','corrupt','budget','input_data','input_shape','input_dtype','batch','config','state'])
def test_combined_resume_rejects_mismatch_transactionally(tmp_path,problem):
    graph,data=fixture();path=tmp_path/'last.pth';session(graph,data,path=path).fit()
    graph,data=fixture();options={}
    if problem=='input_names':data.input_names=('context','wrong')
    if problem=='topology':graph.nodes['node_left'].stride=(2,2)
    if problem=='outputs':
        config=graph.to_config();config['outputs']['different']=config['outputs'].pop('two')
        graph=GraphModelBuilder.from_config(config).build()
    if problem=='target_names':data.target_names=('one','different')
    if problem=='target_mapping':data.tensors=(*data.tensors[:2],data.tensors[3],data.tensors[2])
    if problem=='loss_type':options['losses']={'one':torch.nn.CrossEntropyLoss(),'two':torch.nn.BCEWithLogitsLoss()}
    if problem=='weights':options['weights']={'one':1.2,'two':.3}
    if problem in ('optimizer','optimizer_state','state'):
        payload=torch.load(path,weights_only=True)
        if problem=='optimizer':payload['optimizer']=None
        if problem=='optimizer_state':payload['optimizer']['state_dict']['state']={}
        if problem=='state':next(t for t in payload['model']['state_dict'].values() if t.numel()).add_(1)
        torch.save(payload,path)
    if problem=='corrupt':path.write_bytes(b'invalid checkpoint')
    if problem=='budget':options['epochs']=3
    if problem=='input_data':data.tensors[1].add_(.01)
    if problem=='input_shape':data.tensors=(data.tensors[0][:,:,:2,:],*data.tensors[1:])
    if problem=='input_dtype':data.tensors=(data.tensors[0].double(),*data.tensors[1:])
    if problem=='batch':data.tensors=(data.tensors[0][:1],*data.tensors[1:])
    if problem=='config':
        payload=torch.load(path,weights_only=True)
        payload['metrics']['generic_trainer']['plan']['model']['config']['outputs']['one']='head_b'
        torch.save(payload,path)
    before=copy.deepcopy(graph.state_dict());rng=torch.get_rng_state().clone();prepared=None
    with pytest.raises((ValueError,TypeError,TrainingCheckpointError)):
        prepared=session(graph,data,source=path,**options);prepared.fit()
    assert_state(graph.state_dict(),before)
    assert torch.equal(torch.get_rng_state(),rng)
    if prepared is not None:
        assert prepared.trainer.update_count==0 and not prepared.trainer.optimizer.state
