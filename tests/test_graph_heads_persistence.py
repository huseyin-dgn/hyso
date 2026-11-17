"""Existing generic artifacts retain heads, state relations and training metadata."""
import hashlib
import json
import pytest
import torch
from torch.utils.data import DataLoader
from hyso import ExperimentManager, GraphModelBuilder, Trainer, performance
from hyso.models.graph import GraphModel
from hyso.checkpoint import load_training_checkpoint
from hyso.analysis import evaluate_graph_heads, MultiHeadGraphEvaluationResult
from hyso.analysis._codec import digest
from test_graph_multi_loss import model
from test_graph_multi_loss_resume import session, dataset
from test_graph_heads import specs
from test_graph_training_resume import cpu_only
from test_experiment_workflow import manager


@pytest.fixture
def native_heads(tmp_path):
    graph=model();checkpoint=tmp_path/'last.pth'
    training_session=session(graph,path=checkpoint,epochs=1)
    training=training_session.fit()
    load_training_checkpoint(checkpoint,graph,optimizer=training_session.trainer.optimizer,
        scaler=training_session.trainer.scaler,capture_evidence=True)
    result=evaluate_graph_heads(graph,DataLoader(dataset(),batch_size=2),heads=specs(),capture_structure=True)
    return graph,training_session,training,result,checkpoint


def prohibit_workloads(monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('Persistence started model construction, training, inference or profiling')
    for target,name in ((torch.nn.Module,'__init__'),(Trainer,'fit'),(GraphModel,'forward'),
                        (performance,'profile_operators'),(torch,'load')):
        monkeypatch.setattr(target,name,forbidden)


@pytest.mark.parametrize('ownership',['copy','reference'])
def test_multi_head_training_and_evaluation_roundtrip(manager,tmp_path,monkeypatch,native_heads,ownership):
    graph,prepared,training,result,checkpoint=native_heads
    before=result.to_dict();builder=GraphModelBuilder.from_config(graph.to_config())
    # Freeze the native statement before the persistence-only section. A retained
    # result never revalidates or reconstructs the process-local live source.
    result=MultiHeadGraphEvaluationResult(before)
    prohibit_workloads(monkeypatch)
    experiment=manager.create_experiment('Two retained heads')
    run=manager.create_workflow_run(experiment['id'],'training',model_ids=['A'],
        architectures={'A':builder},training_config=prepared.config,
        protocol={'heads':specs(),'association':'caller_declared'})
    run.start();training.log_to(run)
    trained=run.record_result(training)
    cp=run.add_artifact(checkpoint,kind='checkpoint',ownership='reference',retain_locator=True)
    path=tmp_path/'heads.json';result.to_json(path)
    saved=run.add_artifact(path,kind='graph_multi_head_evaluation',ownership=ownership,retain_locator=True,
        metadata={'checkpoint_artifact_id':cp['id'],'association':'retained_source_statement'})
    run.finish()
    assert saved['expected_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert manager.open_result(trained['id'])['result'].to_dict()==training.to_dict()
    root=manager.root_dir;manager.close();reopened=ExperimentManager(root)
    try:
        opened=reopened.open_result(saved['id'])
        assert opened['status']=='verified'
        actual=opened['result'].to_dict()
        assert actual['heads']==before['heads'] and actual['source_receipt']==before['source_receipt']
        assert actual['model_state_sha256']==before['model_state_sha256']
        assert actual['verification']['graph_v2']=='unverified'
        assert actual['source_receipt']['checkpoint_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert all(h['result']['evaluation']['producer_evidence']['checkpoint']['status']=='missing'
                   for h in actual['heads'].values())
        detail=reopened.get_run_details(run.id)
        assert detail['run']['experiment_id']==experiment['id']
        assert detail['run']['config']['architectures']['A']==graph.to_config()
        assert opened['artifact']['metadata']['checkpoint_artifact_id']==cp['id']
    finally:reopened.close()


@pytest.mark.parametrize('damage',['bytes','rehashed_relation'])
def test_multi_head_artifact_integrity_and_semantic_validation(manager,tmp_path,monkeypatch,native_heads,damage):
    graph,prepared,training,result,checkpoint=native_heads
    data=result.to_dict()
    if damage=='rehashed_relation':
        data['source_receipt']['model_state_sha256']='f'*64
        data.pop('verification')
        data['content_sha256']=digest({k:v for k,v in data.items() if k!='content_sha256'})
    path=tmp_path/'heads.json';path.write_text(json.dumps(data),encoding='utf8')
    prohibit_workloads(monkeypatch)
    experiment=manager.create_experiment('Retained head rejection')
    run=manager.create_run(experiment['id'],'evaluation',{})
    run.start();saved=run.add_artifact(path,kind='graph_multi_head_evaluation',ownership='reference',retain_locator=True);run.finish()
    if damage=='bytes':path.write_text('{}',encoding='utf8')
    opened=manager.open_result(saved['id'])
    assert opened['status']==('mismatch' if damage=='bytes' else 'invalid_result')
    assert opened['result'] is None
