"""Independent P26 arithmetic, inference isolation and artifact acceptance."""
import copy
import csv
from fractions import Fraction
import json
import random
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from hyso import evaluate_predictions, evaluate_classification, EvaluationResult, evaluation_capabilities
from hyso.checkpoint import _rng_state
from trainer_replay_support import equal


def evaluate(y,p,**kw):
    options=dict(class_names=['a','b','c'],split='test')
    options.update(kw)
    return evaluate_predictions(y,p,**options).to_dict()


def test_known_confusion_and_all_averages():
    d=evaluate([0,0,0,1,1,2],[0,0,1,1,2,0]);m=d['metrics']
    assert m['confusion_matrix']==[[2,1,0],[0,1,1],[1,0,0]]
    for name in ('precision','recall','f1'):
        assert [r[name] for r in m['per_class']]==pytest.approx([2/3,1/2,0])
        assert m['averages']['macro'][name]==pytest.approx(float(Fraction(7,18)))
        assert m['averages']['weighted'][name]==pytest.approx(.5)
        assert m['averages']['micro'][name]==.5
    assert m['accuracy']==.5
    assert [r['support'] for r in m['per_class']]==[3,2,1]


@pytest.mark.parametrize('zero',[0,1])
def test_imbalanced_and_absent_classes(zero):
    m=evaluate([0]*90+[1]*10,[0]*99+[1],zero_division=zero)['metrics']
    assert m['accuracy']==.91
    assert m['per_class'][1]['precision']==1
    assert m['per_class'][1]['recall']==.1
    assert m['per_class'][1]['f1']==pytest.approx(2/11)
    assert m['per_class'][2]['support']==0
    assert m['per_class'][2]['f1']==zero
    assert m['averages']['macro']['precision']==pytest.approx((90/99+1+zero)/3)
    assert m['averages']['macro']['recall']==pytest.approx((1+.1+zero)/3)
    assert m['averages']['weighted']['f1']==pytest.approx((90*180/189+10*2/11)/100)
    assert m['averages']['weighted']['recall']==.91
    # Independent TP/FP/FN reference for each class and averaging policy.
    y=[0]*90+[1]*10;p=[0]*99+[1]
    for c,row in enumerate(m['per_class']):
        tp=sum(a==c and b==c for a,b in zip(y,p))
        fp=sum(a!=c and b==c for a,b in zip(y,p))
        fn=sum(a==c and b!=c for a,b in zip(y,p))
        ref=lambda n,d:float(Fraction(n,d)) if d else zero
        assert row['precision']==ref(tp,tp+fp)
        assert row['recall']==ref(tp,tp+fn)
        assert row['f1']==ref(2*tp,2*tp+fp+fn)


@pytest.mark.parametrize('shape',[(5,),(5,1)])
def test_binary_threshold_exact_trainer_rule(shape):
    logits=torch.tensor([-100.,-1e-20,0.,1e-20,100.]).reshape(shape)
    d=evaluate_predictions([0.,0.,1.,1.,1.],logits,class_names=['negative','positive'],split='holdout',
                           task='binary',prediction_type='logits').to_dict()
    assert d['metrics']['confusion_matrix']==[[2,0],[0,3]]
    assert d['configuration']['binary_rule']=='logit >= 0'
    assert d['training_source'] is None


def test_class_order_and_ties():
    m=evaluate([0,1],[[4.,4.,0.],[0.,2.,1.]],prediction_type='logits',class_names=['z','x','y'])['metrics']
    assert m['accuracy']==1
    assert [r['class_name'] for r in m['per_class']]==['z','x','y']


@pytest.mark.parametrize('zero',[0,1])
def test_empty_is_not_success(zero):
    d=evaluate([],[],zero_division=zero)
    assert d['status']=='empty' and d['metrics']['accuracy'] is None
    assert all(r['f1'] is None and r['support']==0 for r in d['metrics']['per_class'])
    assert all(r['precision'] is None for r in d['metrics']['averages'].values())


@pytest.mark.parametrize('y,p,kw',[
    ([0],[0,1],{}),([3],[0],{}),([-1],[0],{}),([0.],[0],{}),([True],[0],{}),
    ([[0]],[0],{}),([0],[False],{}),([0],[float('nan')],{}),
    ([0],[[float('inf'),0.,0.]],{'prediction_type':'logits'}),
    ([0],[[0.,0.]],{'prediction_type':'logits'}),([0],[0.],{'prediction_type':'logits'}),
    ([0],[[True,False,False]],{'prediction_type':'logits'}),
    ([0],[0],{'class_names':['a','a']}),([0],[0],{'class_names':{0:'a',1:'b'}}),
    ([0],[0],{'zero_division':True}),([0],[0],{'zero_division':2}),([0],[0],{'split':''}),
    ([0],[0],{'task':'detection'}),([0],[0],{'task':'segmentation'}),
])
def test_invalid_evidence_rejected(y,p,kw):
    with pytest.raises((ValueError,TypeError,NotImplementedError)):evaluate(y,p,**kw)


def loader(data=None,**kw):
    if data is None:data=TensorDataset(torch.arange(15).reshape(5,3).float()/20,torch.tensor([0,1,0,1,0]))
    return DataLoader(data,batch_size=2,**kw)


def model():
    result=nn.Sequential(nn.Linear(3,4),nn.BatchNorm1d(4),nn.Dropout(.3),nn.Linear(4,2))
    result.train();result[1].eval()
    return result


def state(m):
    return (copy.deepcopy(m.state_dict()),[x.training for x in m.modules()],_rng_state(),
            [(id(p),p._version,None if p.grad is None else p.grad.clone()) for p in m.parameters()])


def test_real_cpu_model_state_modes_rng_and_generator_preserved():
    m=model()
    for p in m.parameters():p.grad=torch.ones_like(p)
    generator=torch.Generator().manual_seed(72);data=loader(generator=generator)
    before=state(m);g=generator.get_state().clone()
    r=evaluate_classification(m,data,class_names=['a','b'],split='validation',source_id='fixture-v1')
    equal(state(m),before);assert torch.equal(g,generator.get_state())
    d=r.to_dict();assert d['accounting']==dict(expected_samples=5,batch_sizes=[2,2,1])
    assert d['source']['device']=='cpu' and d['metrics']['sample_count']==5
    assert d['training_source'] is None
    reference=copy.deepcopy(m).eval()
    with torch.inference_mode():
        logits=torch.cat([reference(x) for x,y in data])
    expected=evaluate_predictions([0,1,0,1,0],logits,prediction_type='logits',class_names=['a','b'],split='validation')
    assert expected.to_dict()['metrics']==d['metrics']


class BadModel(nn.Module):
    def __init__(self,fail):
        super().__init__();self.register_buffer('counter',torch.zeros(1));self.fail=fail
    def forward(self,x):
        random.random();np.random.random();torch.rand(1)
        if self.fail=='raise':self.counter.add_(1);raise RuntimeError('model failure')
        if self.fail=='mutate':self.counter.add_(1)
        if self.fail=='nan':return torch.full((len(x),2),float('nan'))
        if self.fail=='shape':return torch.zeros(len(x),3)
        return torch.zeros(len(x),2)


@pytest.mark.parametrize('failure',['raise','mutate','nan','shape'])
def test_model_failures_preserve_original_and_rng(failure):
    m=BadModel(failure);before=state(m)
    with pytest.raises((RuntimeError,ValueError)):
        evaluate_classification(m,loader(),class_names=['a','b'],split='test')
    equal(state(m),before)


class BadData(Dataset):
    def __len__(self):return 5
    def __getitem__(self,i):
        random.random();np.random.random();torch.rand(1)
        if i==3:raise RuntimeError('dataset failure')
        return torch.ones(3),0


@pytest.mark.parametrize('failure',['dataset','collate','missing','batch_shape','input_nan','target','binary_target'])
def test_batch_failures_restore_rng_and_model(failure):
    m=model();before=state(m)
    data=loader(BadData()) if failure=='dataset' else loader()
    if failure in ('collate','missing','batch_shape','input_nan','target','binary_target'):
        def collate(items):
            if failure=='collate':raise RuntimeError('collate failure')
            from torch.utils.data import default_collate
            x,y=default_collate(items)
            if failure=='missing':return x[:1],y[:1]
            if failure=='batch_shape':return x,y,'extra'
            if failure=='input_nan':x[0,0]=float('nan')
            if failure=='target':y[0]=2
            if failure=='binary_target':return x,y
            return x,y
        data=loader(collate_fn=collate)
    with pytest.raises((RuntimeError,ValueError)):
        evaluate_classification(m,data,class_names=['a','b'],split='test',task='binary' if failure=='binary_target' else 'multiclass')
    equal(state(m),before)


def test_empty_model_evaluation_and_binary_model():
    d=evaluate_classification(nn.Identity(),loader(TensorDataset(torch.zeros(0,2),torch.zeros(0,dtype=torch.long))),
                              class_names=['a','b'],split='empty').to_dict()
    assert d['status']=='empty'
    m=nn.Linear(3,1);m.weight.data.zero_();m.bias.data.zero_()
    d=evaluate_classification(m,loader(TensorDataset(torch.ones(3,3),torch.ones(3,1))),
        class_names=['no','yes'],task='binary',split='test').to_dict()
    assert d['metrics']['accuracy']==1 and d['metrics']['confusion_matrix']==[[0,0],[0,3]]


@pytest.mark.parametrize('options',[{'shuffle':True},{'drop_last':True},{'num_workers':1},{'pin_memory':True}])
def test_unsupported_coverage_rejected_before_iteration(options):
    with pytest.raises(NotImplementedError):evaluate_classification(model(),loader(**options),class_names=['a','b'],split='test')


def test_export_roundtrip_tamper_and_csv(tmp_path):
    r=evaluate_predictions([0,1],[0,1],class_names=['=unsafe','b'],split='holdout')
    d=r.to_dict();path=tmp_path/'evaluation.json';csvpath=tmp_path/'evaluation.csv'
    r.to_json(path);r.to_csv(csvpath)
    assert EvaluationResult(json.loads(path.read_text(encoding='utf-8'))).to_dict()==d
    for method,p in ((r.to_json,path),(r.to_csv,csvpath)):
        with pytest.raises(FileExistsError):method(p)
    with csvpath.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==9 and rows[0]['class_name']=="'=unsafe"
    assert sum(int(r['count']) for r in rows if r['kind']=='confusion')==2
    detached=r.to_dict();detached['raw']['targets'].clear();assert r.to_dict()==d
    with pytest.raises((AttributeError,TypeError)):r._payload='bad'
    for damage in ('metric','metric_type','raw','class_map','source','accounting','training'):
        bad=copy.deepcopy(d)
        if damage=='metric':bad['metrics']['accuracy']=.1
        elif damage=='metric_type':bad['metrics']['accuracy']=True
        elif damage=='raw':bad['raw']['predictions'][0]=1
        elif damage=='class_map':bad['configuration']['class_names'].reverse()
        elif damage=='source':bad['source']['source_id']='foreign'
        elif damage=='accounting':bad['accounting']['expected_samples']=1
        else:bad['training_source']='some-training'
        with pytest.raises(ValueError):EvaluationResult(bad)
    object.__setattr__(r,'_payload',json.dumps(bad))
    with pytest.raises(ValueError):r.to_json(tmp_path/'invalid.json')
    assert not (tmp_path/'invalid.json').exists()
    assert evaluation_capabilities()['segmentation']=='unsupported'


def test_length_exception_and_copy_alias_preserve_rng():
    class LengthFailure(BadData):
        def __len__(self):
            random.random();np.random.random();torch.rand(1)
            raise RuntimeError('length failure')
    m=model();before=state(m)
    with pytest.raises(RuntimeError):
        evaluate_classification(m,loader(LengthFailure()),class_names=['a','b'],split='test')
    equal(state(m),before)
    class Aliased(nn.Linear):
        def __deepcopy__(self,memo):return self
    m=Aliased(3,2);before=state(m)
    with pytest.raises(ValueError,match='aliases'):
        evaluate_classification(m,loader(),class_names=['a','b'],split='test')
    equal(state(m),before)
