"""Explicit head selections reuse original P26 metrics; no global accuracy."""
import copy
import json
import pytest
import torch
from torch.utils.data import DataLoader
from hyso.analysis import MultiHeadGraphEvaluationResult, evaluate_graph_heads
from hyso.analysis._codec import digest
from hyso.checkpoint import save_training_checkpoint, load_training_checkpoint
from test_graph_multi_loss import model
from test_graph_multi_loss_resume import dataset
from test_graph_training_resume import cpu_only
from test_trainer_checkpoint_restore import assert_state


def specs(binary=False):
    return {name:dict(target=name,task='binary' if binary and name=='two' else 'multiclass',
        class_names=['no','yes'] if name=='one' or binary else ['low','medium','high'],
        split='test',dataset_id='synthetic-heads',preprocessing_id='fixed-fp32') for name in ('one','two')}


def capture(binary=False, *, structure=False):
    graph = model(binary)
    result = evaluate_graph_heads(graph,DataLoader(dataset(binary),batch_size=2),
        heads=specs(binary),capture_structure=structure)
    return graph,result


@pytest.mark.parametrize('binary',[False,True])
def test_two_heads_reuse_native_metrics_and_preserve_source(tmp_path,binary):
    graph=model(binary);before=copy.deepcopy(graph.state_dict());rng=torch.get_rng_state().clone()
    result=evaluate_graph_heads(graph,DataLoader(dataset(binary),batch_size=2),heads=specs(binary),capture_structure=True)
    data=result.to_dict()
    assert_state(graph.state_dict(),before)
    assert torch.equal(torch.get_rng_state(),rng)
    assert 'overall_accuracy' not in json.dumps(data)
    assert data['source_receipt']['status']=='missing'
    for name in ('one','two'):
        assert 'verification' not in data['heads'][name]['result']
        evaluation=result.head(name).evaluation.to_dict()
        from hyso.evaluation import evaluate_predictions
        expected=evaluate_predictions(evaluation['raw']['targets'],evaluation['raw']['predictions'],
            class_names=specs(binary)[name]['class_names'],split='test',task=specs(binary)[name]['task'],prediction_type='logits')
        assert evaluation['metrics']==expected.to_dict()['metrics']
        assert evaluation['metrics']['sample_count']==3
        assert len(evaluation['metrics']['confusion_matrix'])==len(specs(binary)[name]['class_names'])
        assert evaluation['source']['source_id']=='synthetic-heads'
        assert evaluation['schema_version']==3
        assert result.analysis_source('model-'+name,head=name).model_id=='model-'+name
    path=tmp_path/'heads.json';result.to_json(path)
    restored=MultiHeadGraphEvaluationResult.from_json(path)
    assert restored.to_dict()['heads']==data['heads']
    assert restored.to_dict()['verification']['graph_v2']=='unverified'


def test_head_metrics_and_declared_protocols_remain_independent():
    graph=model()
    with torch.no_grad():
        for parameter in graph.parameters():parameter.zero_()
    heads=specs();heads['two']['split']='validation';heads['two']['preprocessing_id']='declared-second-protocol'
    result=evaluate_graph_heads(graph,DataLoader(dataset(),batch_size=2),heads=heads)
    a,b=[result.head(name).evaluation.to_dict() for name in ('one','two')]
    assert a['metrics']['accuracy']==pytest.approx(2/3)
    assert b['metrics']['accuracy']==pytest.approx(1/3)
    assert a['metrics']['averages']['macro']['f1']!=b['metrics']['averages']['macro']['f1']
    assert b['configuration']['split']=='validation'
    assert result.to_dict()['heads']['two']['spec']['preprocessing_id']=='declared-second-protocol'


@pytest.mark.parametrize('problem',['output','target','labels','task','target_dtype','target_shape','input_dtype','input_shape','sampler','collate'])
def test_head_contract_rejects_invalid_mapping_or_tensors(problem):
    graph=model();ds=dataset();heads=specs();options={}
    if problem=='output':heads['unknown']=heads.pop('two')
    if problem=='target':heads['two']['target']='missing'
    if problem=='labels':heads['two']['class_names']=['wrong','count']
    if problem=='task':heads['two']['task']='binary'
    if problem=='target_dtype':ds.tensors=(*ds.tensors[:2],ds.tensors[2].float())
    if problem=='target_shape':ds.tensors=(*ds.tensors[:2],ds.tensors[2][:,None])
    if problem=='input_dtype':ds.tensors=(ds.tensors[0].double(),*ds.tensors[1:])
    if problem=='input_shape':ds.tensors=(ds.tensors[0][:,:,:2,:],*ds.tensors[1:])
    if problem=='sampler':options['shuffle']=True
    if problem=='collate':options['collate_fn']=lambda rows:rows
    before=copy.deepcopy(graph.state_dict());rng=torch.get_rng_state().clone()
    with pytest.raises((ValueError,TypeError,NotImplementedError,RuntimeError)):
        evaluate_graph_heads(graph,DataLoader(ds,batch_size=2,**options),heads=heads)
    assert_state(graph.state_dict(),before)
    assert torch.equal(torch.get_rng_state(),rng)


@pytest.mark.parametrize('damage',['mapping','labels','metrics','state','receipt'])
def test_rehashed_invalid_multi_head_envelope_is_rejected(damage):
    graph,result=capture();data=result.to_dict();data.pop('verification')
    if damage=='mapping':data['heads']['two']['spec']['target']='one'
    if damage=='labels':data['heads']['two']['spec']['class_names']=['bad','mapping','other']
    if damage=='metrics':data['heads']['two']['result']['evaluation']['metrics']['accuracy']=.123
    if damage=='state':data['model_state_sha256']='invalid'
    if damage=='receipt':data['source_receipt']['status']='native'
    data['content_sha256']=digest({k:v for k,v in data.items() if k!='content_sha256'})
    with pytest.raises(ValueError):MultiHeadGraphEvaluationResult(data)


def test_raw_graph_checkpoint_receipt_is_not_transferred_to_head_adapters(tmp_path):
    graph=model();path=tmp_path/'model.pth'
    save_training_checkpoint(path,graph,graph_contract=True,include_rng=False)
    load_training_checkpoint(path,graph,capture_evidence=True)
    from hyso.checkpoint import graph_checkpoint_evidence
    sha=graph_checkpoint_evidence(graph)['checkpoint_sha256']
    result=evaluate_graph_heads(graph,DataLoader(dataset(),batch_size=2),heads=specs(),expected_checkpoint_sha256=sha)
    assert result.to_dict()['verification']['graph_v2']=='verified'
    assert all(row['result']['source_receipt']['status']=='missing' for row in result.to_dict()['heads'].values())
    with pytest.raises(ValueError,match='expected_source_identity'):
        evaluate_graph_heads(graph,DataLoader(dataset(),batch_size=2),heads=specs(),expected_checkpoint_sha256='0'*64)
    with torch.no_grad():next(graph.parameters()).add_(1)
    assert result.to_dict()['verification']['graph_v2']=='missing'
