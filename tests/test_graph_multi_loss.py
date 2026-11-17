"""Bounded two-head losses on the single existing Trainer fit/update loop."""
import copy
import pytest
import torch
from torch.utils.data import Dataset, DataLoader
from hyso import GraphModelBuilder, Trainer
from test_graph_training_resume import cpu_only
from test_trainer_checkpoint_restore import assert_state


def model(binary=False):
    return (GraphModelBuilder({'image':[1,3,3]})
        .add('stem','conv2d',['image'],in_channels=1,out_channels=2)
        .add('left','conv2d',['stem'],in_channels=2,out_channels=2)
        .add('right','conv2d',['stem'],in_channels=2,out_channels=2)
        .add('head_a','classification_head',['left'],in_features=2,num_classes=2)
        .add('head_b','classification_head',['right'],in_features=2,num_classes=1 if binary else 3)
        .outputs(one='head_a',two='head_b',features='stem').build())


class Samples(Dataset):
    def __init__(self,binary=False):
        self.x=torch.arange(27,dtype=torch.float32).reshape(3,1,3,3)/27
        self.a=torch.tensor([0,1,0])
        self.b=torch.tensor([1.,0.,1.]) if binary else torch.tensor([2,0,1])
    def __len__(self):return 3
    def __getitem__(self,i):return self.x[i],{'one':self.a[i],'two':self.b[i]}


def criteria(binary=False):
    return {'two':torch.nn.BCEWithLogitsLoss() if binary else torch.nn.CrossEntropyLoss(),
            'one':torch.nn.CrossEntropyLoss()}


@pytest.mark.parametrize('optimizer,binary',[(torch.optim.SGD,False),(torch.optim.Adam,True)])
def test_two_losses_match_native_updates_and_validation(optimizer,binary):
    graph=model(binary);reference=copy.deepcopy(graph);losses=criteria(binary)
    weights={'one':.3,'two':1.2};loader=DataLoader(Samples(binary),batch_size=2)
    expected_optimizer=optimizer(reference.parameters(),lr=.01)
    expected=0.
    for inputs,targets in loader:
        expected_optimizer.zero_grad();outputs=reference(inputs)
        loss=sum(weights[n]*criterion(outputs[n].flatten() if binary and n=='two' else outputs[n],targets[n])
                 for n,criterion in losses.items())
        expected+=loss.item()*len(inputs);loss.backward();expected_optimizer.step()
    reference.eval();validation=0.
    with torch.no_grad():
        for inputs,targets in loader:
            outputs=reference(inputs)
            validation+=sum(weights[n]*criterion(outputs[n].flatten() if binary and n=='two' else outputs[n],targets[n])
                            for n,criterion in losses.items()).item()*len(inputs)
    before=copy.deepcopy(graph.state_dict());ids=[id(p) for p in graph.parameters()]
    trainer=Trainer(graph,optimizer=optimizer(graph.parameters(),lr=.01),loss_fn=losses,loss_weights=weights)
    history=trainer.fit(loader,loader,epochs=1)
    assert history['train_loss']==pytest.approx([expected/3])
    assert history['val_loss']==pytest.approx([validation/3])
    assert history['train_acc']==history['val_acc']==[None]
    assert trainer.update_count==2 and ids==[id(p) for p in graph.parameters()]
    assert_state(graph.state_dict(),reference.state_dict())
    for branch in ('node_left','node_right'):
        assert any(not torch.equal(value,before[name]) for name,value in graph.state_dict().items() if branch in name)


@pytest.mark.parametrize('problem',['missing_target','extra_target','dtype','batch','class',
    'missing_output','output_type','output_width','nonfinite'])
def test_invalid_multi_output_batch_fails_before_update(problem):
    graph=model();inputs,targets=next(iter(DataLoader(Samples(),batch_size=2)))
    if problem=='missing_target':targets.pop('two')
    if problem=='extra_target':targets['extra']=targets['one']
    if problem=='dtype':targets['two']=targets['two'].float()
    if problem=='batch':targets['two']=targets['two'][:1]
    if problem=='class':targets['two'][0]=100
    if problem=='missing_output':
        native=graph.forward
        graph.forward=lambda x:{k:v for k,v in native(x).items() if k!='two'}
    if problem in ('output_type','output_width','nonfinite'):
        native=graph.forward
        def invalid(x):
            outputs=native(x)
            outputs['two'] = (None if problem=='output_type' else
                outputs['two'][:,:1] if problem=='output_width' else outputs['two']*float('nan'))
            return outputs
        graph.forward=invalid
    trainer=Trainer(graph,optimizer=torch.optim.SGD(graph.parameters(),lr=.1),
                    loss_fn=criteria(),loss_weights={'one':1.,'two':1.})
    before=copy.deepcopy(graph.state_dict())
    with pytest.raises((ValueError,TypeError,FloatingPointError)):trainer.fit([(inputs,targets)],epochs=1)
    assert trainer.update_count==0
    assert_state(graph.state_dict(),before)


@pytest.mark.parametrize('weights',[None,{}, {'one':1.,'two':0.}, {'one':1.,'two':-1.},
    {'one':1.,'two':float('nan')}, {'one':1.,'two':True}])
def test_explicit_valid_weights_required(weights):
    graph=model()
    with pytest.raises(ValueError):
        Trainer(graph,optimizer=torch.optim.SGD(graph.parameters(),lr=.1),loss_fn=criteria(),loss_weights=weights)


@pytest.mark.parametrize('name',['missing','features'])
def test_only_explicit_classification_outputs_can_have_losses(name):
    graph=model();losses=criteria();losses[name]=losses.pop('two')
    with pytest.raises(ValueError,match='classification_head'):
        Trainer(graph,optimizer=torch.optim.SGD(graph.parameters(),lr=.1),loss_fn=losses,
                loss_weights={key:1. for key in losses})


@pytest.mark.parametrize('option',['save_last_to','resume_from'])
def test_multi_loss_automatic_resume_stays_unsupported(tmp_path,option):
    graph=model();trainer=Trainer(graph,optimizer=torch.optim.SGD(graph.parameters(),lr=.1),
        loss_fn=criteria(),loss_weights={'one':1.,'two':1.})
    with pytest.raises(ValueError,match='multi-loss automatic'):
        trainer.fit(DataLoader(Samples(),batch_size=2),epochs=1,**{option:tmp_path/'never.pth'})
    assert trainer.update_count==0 and not (tmp_path/'never.pth').exists()


@pytest.mark.parametrize('problem',['fp64','adamw','scheduler','head_width','weighted_criterion'])
def test_bounded_profile_is_explicit(problem):
    graph=model();losses=criteria()
    if problem=='fp64':graph.double()
    optimizer=(torch.optim.AdamW if problem=='adamw' else torch.optim.SGD)(graph.parameters(),lr=.1)
    options={}
    if problem=='scheduler':options['scheduler']=torch.optim.lr_scheduler.StepLR(optimizer,1)
    if problem=='head_width':losses['two']=torch.nn.BCEWithLogitsLoss()
    if problem=='weighted_criterion':losses['two']=torch.nn.CrossEntropyLoss(weight=torch.ones(3))
    with pytest.raises((ValueError,TypeError)):
        Trainer(graph,optimizer=optimizer,loss_fn=losses,loss_weights={'one':1.,'two':1.},**options)
