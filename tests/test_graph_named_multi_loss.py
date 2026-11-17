"""Compose existing input-name and loss-name bridges on the same fit loop."""
import copy
import pytest
import torch
from torch.utils.data import DataLoader
from hyso import GraphModelBuilder, TrainingConfig, Trainer
from hyso.training import NamedTargetDataset
from hyso.checkpoint import TrainingCheckpointError
from test_graph_multi_loss import criteria
from test_graph_training_resume import cpu_only
from test_trainer_checkpoint_restore import assert_state


def model():
    return (GraphModelBuilder({'image':[1,3,3],'context':[1,3,3]})
        .add('left','conv2d',['image'],in_channels=1,out_channels=2)
        .add('right','conv2d',['context'],in_channels=1,out_channels=2)
        .add('head_a','classification_head',['left'],in_features=2,num_classes=2)
        .add('head_b','classification_head',['right'],in_features=2,num_classes=3)
        .outputs(one='head_a',two='head_b').build())


def dataset():
    x=torch.arange(27,dtype=torch.float32).reshape(3,1,3,3)/27
    return NamedTargetDataset({'context':x.flip(-1)+.2,'image':x},
        {'two':torch.tensor([2,0,1]),'one':torch.tensor([0,1,0])})


@pytest.mark.parametrize('optimizer',['sgd','adam'])
def test_two_named_inputs_and_losses_match_native_updates(optimizer):
    graph=model();reference=copy.deepcopy(graph);before=copy.deepcopy(graph.state_dict())
    data=dataset();loader=DataLoader(data,batch_size=2);losses=criteria();weights={'one':.3,'two':1.2}
    native={'sgd':torch.optim.SGD,'adam':torch.optim.Adam}[optimizer](reference.parameters(),lr=.01)
    summed=0.
    for inputs,targets in loader:
        native.zero_grad();outputs=reference(dict(reversed(list(inputs.items()))))
        loss=sum(weights[n]*criterion(outputs[n],targets[n]) for n,criterion in losses.items())
        summed+=loss.item()*len(targets['one']);loss.backward();native.step()
    config=TrainingConfig(epochs=1,batch_size=2,shuffle=False,optimizer=optimizer,learning_rate=.01)
    train,_=config.make_loaders(data)
    session=config.prepare(graph,train,loss_fn=losses,loss_weights=weights)
    history=session.fit().history
    assert history['train_loss']==pytest.approx([summed/3])
    assert history['train_acc']==[None]
    assert session.trainer.update_count==2
    assert_state(graph.state_dict(),reference.state_dict())
    for branch in ('node_left','node_right'):
        assert any(not torch.equal(value,before[name]) for name,value in graph.state_dict().items() if branch in name)


@pytest.mark.parametrize('problem',['input','target','batch','dtype','device','output'])
def test_invalid_named_multi_head_batch_fails_before_update(problem):
    graph=model();inputs,targets=next(iter(DataLoader(dataset(),batch_size=2)))
    if problem=='input':inputs['wrong']=inputs.pop('context')
    if problem=='target':targets.pop('two')
    if problem=='batch':inputs['context']=inputs['context'][:1]
    if problem=='dtype':inputs['context']=inputs['context'].double()
    if problem=='device':inputs['context']=torch.empty_like(inputs['context'],device='meta')
    if problem=='output':
        forward=graph.forward
        graph.forward=lambda x:{k:v for k,v in forward(x).items() if k!='two'}
    trainer=Trainer(graph,optimizer=torch.optim.SGD(graph.parameters(),lr=.01),
        loss_fn=criteria(),loss_weights={'one':1.,'two':1.})
    before=copy.deepcopy(graph.state_dict())
    with pytest.raises((ValueError,TypeError,RuntimeError)):
        trainer.fit([(inputs,targets)],epochs=1)
    assert trainer.update_count==0
    assert_state(graph.state_dict(),before)


@pytest.mark.parametrize('option',['save_last_to','resume_from'])
def test_combined_profile_automatic_resume_remains_explicitly_unsupported(tmp_path,option):
    graph=model();trainer=Trainer(graph,optimizer=torch.optim.SGD(graph.parameters(),lr=.01),
        loss_fn=criteria(),loss_weights={'one':1.,'two':1.},model_config=graph.to_config())
    with pytest.raises(TrainingCheckpointError,match='two-input multi-loss automatic'):
        trainer.fit(DataLoader(dataset(),batch_size=2),epochs=1,**{option:tmp_path/'never.pth'})
    assert trainer.update_count==0 and not (tmp_path/'never.pth').exists()
