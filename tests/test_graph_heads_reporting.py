"""Head-scoped native comparison, existing advice and presentation readers."""
import json
import pytest
import torch
from torch.utils.data import DataLoader
from hyso import GraphModelBuilder, advisor
from hyso.analysis import MultiHeadGraphEvaluationResult, evaluate_graph_heads, compare_graph_heads
from hyso.reporting import create_report, report_experiment, PresentationReport
from hyso.cli import main
from test_graph_multi_loss import model
from test_graph_multi_loss_resume import dataset
from test_graph_heads import specs
from test_graph_heads_persistence import prohibit_workloads
from test_graph_training_resume import cpu_only
from test_experiment_workflow import manager


def pair(change=None):
    left=model();config=left.to_config()
    for node in config['nodes']:
        if node['id']=='left':node['params']['out_channels']=3
        if node['id']=='head_a':node['params']['in_features']=3
    right=GraphModelBuilder.from_config(config).build()
    candidate=specs()
    if change=='labels':candidate['one']['class_names']=['different','meaning']
    if change=='preprocessing':candidate['one']['preprocessing_id']='different'
    if change=='split':candidate['one']['split']='validation'
    if change=='missing':candidate['one']['preprocessing_id']=None
    values=[evaluate_graph_heads(g,DataLoader(dataset(),batch_size=2),heads=s,capture_structure=True)
            for g,s in ((left,specs()),(right,candidate))]
    return [MultiHeadGraphEvaluationResult(v.to_dict()) for v in values]


def test_explicit_head_pairs_retain_native_gates_and_structure(monkeypatch):
    a,b=pair();prohibit_workloads(monkeypatch)
    comparisons=compare_graph_heads(a,b,head_pairs={'one':'one','two':'two'})
    for head,comparison in comparisons.items():
        data=comparison.to_dict()
        assert data['sources']['baseline']['model_id']=='A:'+head
        assert data['sources']['candidate']['model_id']=='B:'+head
        assert data['claims']['global_winner'] is None
        assert data['structure']['status']=='COMPARABLE'
        assert any(row['difference'] for row in data['structure']['comparison']['summary_differences'])
        assert all(r['status']=='COMPARABLE' and r['difference'] is not None for r in data['metrics'] if r['feature_id']=='P26')
        assert all(r['status']=='INSUFFICIENT_EVIDENCE' and r['difference'] is None for r in data['metrics'] if r['feature_id']=='P12')
        advice=advisor.recommend(comparison).to_display_dict(language='en')
        assert all(o['manual_only'] and not o['automatically_applied'] for o in advice['comparison_observations'])
    with pytest.raises(ValueError,match='explicit_equal_head_names'):
        compare_graph_heads(a,b,head_pairs={'one':'two'})
    with pytest.raises(ValueError,match='explicit_equal_head_names'):
        compare_graph_heads(a,b,head_pairs={})
    with pytest.raises(ValueError,match='selected_head_missing'):
        compare_graph_heads(a,b,head_pairs={'absent':'absent'})


@pytest.mark.parametrize('change,status',[('labels','INCOMPATIBLE'),('preprocessing','INCOMPATIBLE'),
    ('split','INCOMPATIBLE'),('missing','INSUFFICIENT_EVIDENCE')])
def test_ineligible_head_never_has_numeric_difference_or_advisor_comparison(change,status,monkeypatch):
    a,b=pair(change);prohibit_workloads(monkeypatch)
    # This case isolates quality advice. Independently eligible structure advice
    # is covered by the positive structure test and must remain available.
    comparison=compare_graph_heads(a,b,head_pairs={'one':'one'},include_structure=False)['one']
    assert all(r['status']==status and r['difference'] is None for r in comparison.to_dict()['metrics'] if r['feature_id']=='P26')
    assert advisor.recommend(comparison).to_display_dict(language='en')['comparison_observations']==[]


@pytest.mark.parametrize('language',['tr','en'])
def test_head_reports_preserve_units_values_authority_and_saved_cli(manager,tmp_path,monkeypatch,capsys,language):
    a,b=pair();prohibit_workloads(monkeypatch)
    comparisons=compare_graph_heads(a,b,head_pairs={'one':'one','two':'two'})
    advice=[advisor.recommend(c) for c in comparisons.values()]
    report=create_report(a,b,*comparisons.values(),*advice,language=language)
    data=report.to_dict()
    assert data['presentation_only'] and not data['authoritative_evidence']
    sections=[s for s in data['sources'] if s['kind']=='graph_multi_head_evaluation']
    assert len(sections)==2
    for section in sections:
        metrics={m['key']:m for m in section['metrics']}
        for head in ('one','two'):
            native=section['raw']['heads'][head]['result']['evaluation']['metrics']
            assert metrics[head+'.macro_precision']['value']==native['averages']['macro']['precision']
            assert metrics[head+'.macro_precision']['scope']['evaluation_id']==section['raw']['heads'][head]['result']['evaluation']['evaluation_id']
            assert metrics[head+'.macro_recall']['unit']=='fraction'
            assert metrics[head+'.confusion_matrix']['value']==native['confusion_matrix']
            assert metrics[head+'.sample_count']['value']==3
        assert not any(m['key']=='overall_accuracy' for m in section['metrics'])
    for section in (s for s in data['sources'] if s['kind']=='comparison'):
        head=section['raw']['sources']['baseline']['model_id'].split(':')[1]
        assert section['raw']==comparisons[head].to_dict()
        assert all(m['value'] is None and m['status']=='INSUFFICIENT_EVIDENCE'
                   for m in section['metrics'] if m['feature_id']=='P12')
    for suffix,export in [('json',report.to_json),('md',report.to_markdown),('html',report.to_html)]:
        export(tmp_path/('report.'+suffix))
    for suffix in ('md','html'):
        text=(tmp_path/('report.'+suffix)).read_text(encoding='utf8')
        assert all(marker in text for marker in ('one','two','fraction','INSUFFICIENT_EVIDENCE','COMPARABLE','unverified'))
        for section in sections:
            for metric in section['metrics']:
                if type(metric['value']) in (int,float):assert str(metric['value']) in text
    saved=tmp_path/'report.json'
    assert PresentationReport.from_json(saved).to_dict()==data
    output=tmp_path/'cli.html'
    assert main(['reports','render',str(saved),'--format','html','--output',str(output),'--json'])==0
    assert json.loads(capsys.readouterr().out)['result']==data
    assert output.read_text(encoding='utf8')==report.to_html()
    experiment=manager.create_experiment('Head comparison report')
    artifacts=[]
    for label,result in (('A',a),('B',b)):
        run=manager.create_run(experiment['id'],'evaluation',{'model_id':label})
        run.start();path=tmp_path/(label+'.json');result.to_json(path)
        artifacts.append(run.add_artifact(path,kind='graph_multi_head_evaluation',ownership='copy')['id'])
        run.finish()
    for comparison in comparisons.values():
        ids=[comparison.to_dict()['sources'][side]['model_id'] for side in ('baseline','candidate')]
        run=manager.create_workflow_run(experiment['id'],'comparison',model_ids=ids,comparison=comparison)
        run.start();run.record_result(comparison,input_artifact_ids=artifacts);run.finish()
    before=manager._core_snapshot(manager._db)
    experiment_report=report_experiment(manager,experiment['id'],language=language).to_dict()
    assert manager._core_snapshot(manager._db)==before
    assert len([s for s in experiment_report['sources'] if s['kind']=='graph_multi_head_evaluation'])==2
    assert len([s for s in experiment_report['sources'] if s['kind']=='comparison'])==2
