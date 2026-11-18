"""Named streams preserve two inputs and two independent head evaluations."""
import copy
import pytest
import torch
from torch.utils.data import DataLoader
from hyso import GraphModelBuilder, evaluate_predictions
from hyso.analysis import evaluate_graph_heads, MultiHeadGraphEvaluationResult, compare_graph_heads
from hyso.analysis._codec import digest
from test_graph_named_multi_loss import model, dataset
from test_graph_heads import specs
from test_graph_training_resume import cpu_only
from test_trainer_checkpoint_restore import assert_state


def fixture(binary=False):
    graph=model();data=dataset()
    if binary:
        config=graph.to_config()
        next(n for n in config['nodes'] if n['id']=='head_b')['params']['num_classes']=1
        graph=GraphModelBuilder.from_config(config).build()
        data.tensors=(*data.tensors[:-1],torch.tensor([1.,0.,1.]))
    return graph,data


@pytest.mark.parametrize('binary',[False,True])
def test_named_head_metrics_mapping_state_rng_and_roundtrip(tmp_path,binary):
    graph,data=fixture(binary);graph.train()
    for parameter in graph.parameters():parameter.grad=torch.ones_like(parameter)
    before=copy.deepcopy(graph.state_dict());rng=torch.get_rng_state().clone()
    modes=[m.training for m in graph.modules()];grads=[p.grad.clone() for p in graph.parameters()]
    result=evaluate_graph_heads(graph,DataLoader(data,batch_size=2),heads=specs(binary),capture_structure=True)
    assert_state(graph.state_dict(),before)
    assert modes==[m.training for m in graph.modules()]
    assert all(torch.equal(p.grad,g) for p,g in zip(graph.parameters(),grads))
    assert torch.equal(torch.get_rng_state(),rng)
    with torch.no_grad():outputs=graph({n:t for n,t in zip(data.input_names,data.tensors[:2])})
    for name in ('one','two'):
        value=result.head(name).evaluation.to_dict()
        target=data.tensors[2+data.target_names.index(name)]
        expected=evaluate_predictions(target,outputs[name],class_names=specs(binary)[name]['class_names'],
            task=specs(binary)[name]['task'],split='test',prediction_type='logits')
        assert value['metrics']==expected.to_dict()['metrics']
        stream=value['producer_evidence']['evaluated_stream']
        assert stream['input_names']==['context','image'] and stream['sample_count']==3
        assert stream['method']=='sha256_ordered_named_tensor_samples_le_v1'
        assert value['metrics']['sample_count']==3
    path=tmp_path/'heads.json';result.to_json(path)
    opened=MultiHeadGraphEvaluationResult.from_json(path)
    assert opened.to_dict()['heads']==result.to_dict()['heads']
    assert opened.to_dict()['verification']['graph_v2']=='unverified'


def test_named_hash_is_order_and_partition_invariant_but_binds_each_input():
    graph,data=fixture()
    a=evaluate_graph_heads(graph,DataLoader(data,batch_size=2),heads=specs())
    from hyso.training import NamedTargetDataset
    reordered=NamedTargetDataset(dict(reversed(list(zip(data.input_names,data.tensors[:2])))),
        dict(zip(data.target_names,data.tensors[2:])))
    b=evaluate_graph_heads(graph,DataLoader(reordered,batch_size=3),heads=dict(reversed(list(specs().items()))))
    stream=lambda r:r.head('one').evaluation.to_dict()['producer_evidence']['evaluated_stream']
    assert stream(a)==stream(b)
    data.tensors[0].add_(.1)
    c=evaluate_graph_heads(graph,DataLoader(data,batch_size=2),heads=specs())
    assert stream(a)['inputs_sha256']!=stream(c)['inputs_sha256']
    compared=compare_graph_heads(a,c,head_pairs={'one':'one'})['one'].to_dict()
    assert all(r['status']=='INCOMPATIBLE' and r['difference'] is None for r in compared['metrics'] if r['feature_id']=='P26')


@pytest.mark.parametrize('problem',['input_name','batch','shape','dtype','device','output','target','target_dtype','target_shape','labels'])
def test_combined_evaluation_rejects_invalid_contract_without_source_mutation(problem):
    graph,data=fixture();heads=specs()
    if problem=='input_name':data.input_names=('context','wrong')
    if problem=='batch':data.tensors=(data.tensors[0][:1],*data.tensors[1:])
    if problem=='shape':data.tensors=(data.tensors[0][:,:,:2,:],*data.tensors[1:])
    if problem=='dtype':data.tensors=(data.tensors[0].double(),*data.tensors[1:])
    if problem=='device':data.tensors=(torch.empty_like(data.tensors[0],device='meta'),*data.tensors[1:])
    if problem=='output':heads['unknown']=heads.pop('two')
    if problem=='target':heads['two']['target']='missing'
    if problem=='target_dtype':data.tensors=(*data.tensors[:-1],data.tensors[-1].float())
    if problem=='target_shape':data.tensors=(*data.tensors[:-1],data.tensors[-1][:,None])
    if problem=='labels':heads['two']['class_names']=['wrong','width']
    before=copy.deepcopy(graph.state_dict());rng=torch.get_rng_state().clone()
    with pytest.raises((ValueError,TypeError,RuntimeError)):
        evaluate_graph_heads(graph,DataLoader(data,batch_size=2),heads=heads)
    assert_state(graph.state_dict(),before)
    assert torch.equal(torch.get_rng_state(),rng)


def test_independent_head_accuracy_and_rehashed_name_mismatch():
    graph,data=fixture()
    with torch.no_grad():
        for p in graph.parameters():p.zero_()
    result=evaluate_graph_heads(graph,DataLoader(data,batch_size=2),heads=specs())
    assert result.head('one').evaluation.to_dict()['metrics']['accuracy']==pytest.approx(2/3)
    assert result.head('two').evaluation.to_dict()['metrics']['accuracy']==pytest.approx(1/3)
    assert result.head('one').evaluation.to_dict()['metrics']['averages']['macro']['f1']!=result.head('two').evaluation.to_dict()['metrics']['averages']['macro']['f1']
    raw=result.to_dict();raw.pop('verification')
    raw['graph_contract']['configuration']['inputs']['wrong']=raw['graph_contract']['configuration']['inputs'].pop('context')
    raw['content_sha256']=digest({k:v for k,v in raw.items() if k!='content_sha256'})
    with pytest.raises(ValueError):MultiHeadGraphEvaluationResult(raw)


@pytest.mark.parametrize('names',[None,['image','context'],['image','image'],['image',1]])
def test_named_native_reader_rejects_rehashed_invalid_names(names):
    graph,data=fixture()
    result=evaluate_graph_heads(graph,DataLoader(data,batch_size=2),heads=specs())
    raw=result.head('one').evaluation.to_dict()
    raw['producer_evidence']['evaluated_stream']['input_names']=names
    from hyso.evaluation import EvaluationResult, _hash
    raw['source_sha256']=_hash({k:v for k,v in raw.items() if k not in ('source_sha256','metrics','status')})
    with pytest.raises(ValueError):EvaluationResult(raw)
