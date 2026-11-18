"""Explicit two-input/two-head CPU workflow, then strictly retained reporting."""
import hashlib
import json
import pytest
import torch
from hyso import ExperimentManager
from hyso.analysis import compare_graph_heads
from hyso.reporting import PresentationReport, report_experiment
from hyso.cli import main
from hyso.training import resume
from examples.models.graph_evidence_workflow import train_heads, compare_saved_heads
from test_graph_heads_persistence import prohibit_workloads
from test_graph_training_resume import cpu_only


def test_combined_resume_evaluation_store_comparison_advisor_and_bilingual_exports(tmp_path,monkeypatch,capsys):
    native=resume.save_last
    def stop(*args):native(*args);raise InterruptedError()
    with monkeypatch.context() as scoped:
        scoped.setattr(resume,'save_last',stop)
        with pytest.raises(InterruptedError):train_heads(tmp_path/'cut',two_inputs=True)
    a=train_heads(tmp_path/'A',two_inputs=True,resume_from=tmp_path/'cut/last.pth')
    b=train_heads(tmp_path/'B',two_inputs=True,variant='B')
    assert a['applied_updates']==b['applied_updates']==4
    assert a['history']['train_acc']==b['history']['train_acc']==[None,None]
    assert a['imported_graph_trust']==b['imported_graph_trust']=='unverified'
    managers=[];results=[]
    for label,summary in (('A',a),('B',b)):
        manager=ExperimentManager(tmp_path/label/'experiments');managers.append(manager)
        opened=manager.open_result(summary['evaluation_artifact_id'])
        assert opened['status']=='verified'
        result=opened['result'];results.append(result)
        data=result.to_dict()
        assert data['source_receipt']['checkpoint_sha256']==hashlib.sha256(open(summary['checkpoint'],'rb').read()).hexdigest()
        assert data['source_receipt']['model_state_sha256']==data['model_state_sha256']
        assert set(data['graph_contract']['configuration']['inputs'])=={'hint','image'}
        assert all(h['result']['evaluation']['producer_evidence']['evaluated_stream']['input_names']==['hint','image']
                   for h in data['heads'].values())
        assert all(h['result']['source_receipt']['status']=='missing' for h in data['heads'].values())
        assert manager.verify_artifact(summary['evaluation_artifact_id'])['status']=='verified'
    try:
        prohibit_workloads(monkeypatch)
        summary=compare_saved_heads([tmp_path/'A/heads.json',tmp_path/'B/heads.json'],tmp_path/'reports',
            head_pairs={'disease':'disease','severity':'severity'})
        comparisons=compare_graph_heads(*results,head_pairs={'disease':'disease','severity':'severity'})
        for name,comparison in comparisons.items():
            raw=comparison.to_dict()
            assert raw['claims']['global_winner'] is None and summary[name]['structure']=='COMPARABLE'
            assert all(m['status']=='COMPARABLE' and m['difference'] is not None for m in raw['metrics'] if m['feature_id']=='P26')
            assert all(m['status']=='INSUFFICIENT_EVIDENCE' and m['difference'] is None for m in raw['metrics'] if m['feature_id']=='P12')
            advice=json.loads((tmp_path/'reports'/(name+'-advisor.json')).read_text(encoding='utf8'))
            from hyso.advisor.recommendations import RecommendationReport
            observations=RecommendationReport(advice).to_display_dict(language='en')['comparison_observations']
            assert all(o['manual_only'] and not o['automatically_applied'] for o in observations)
        for language in ('tr','en'):
            path=tmp_path/'reports'/language
            report=PresentationReport.from_json(path.with_suffix('.json'))
            data=report.to_dict()
            assert data['presentation_only'] and not data['authoritative_evidence']
            for section in (s for s in data['sources'] if s['kind']=='comparison'):
                name=section['raw']['sources']['baseline']['model_id'].split(':')[1]
                assert section['raw']==comparisons[name].to_dict()
            for suffix in ('.md','.html'):
                text=path.with_suffix(suffix).read_text(encoding='utf8')
                assert all(x in text for x in ('disease','severity','COMPARABLE','INSUFFICIENT_EVIDENCE','unverified'))
                for section in data['sources']:
                    if section['kind']=='graph_multi_head_evaluation':
                        for m in section['metrics']:
                            if type(m['value']) in (float,int):assert str(m['value']) in text
            exported=tmp_path/('cli-'+language+'.json')
            assert main(['reports','render',str(path.with_suffix('.json')),'--output',str(exported),'--json'])==0
            assert json.loads(capsys.readouterr().out)['result']==data
            assert json.loads(exported.read_text(encoding='utf8'))==data
            before=managers[0]._core_snapshot(managers[0]._db)
            experiment=report_experiment(managers[0],a['experiment_id'],language=language).to_dict()
            assert any(s['kind']=='graph_multi_head_evaluation' for s in experiment['sources'])
            assert managers[0]._core_snapshot(managers[0]._db)==before
    finally:
        for manager in managers:manager.close()
