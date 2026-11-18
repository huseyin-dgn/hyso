import json
import pytest
from hyso.analysis import (AnalysisContext,EvidenceIndex,EvidenceRecord,EvidenceError,
                           UnifiedAnalysisReport)


def context():
    return AnalysisContext(workload_id='acceptance',task='image_classification',
        model={'checkpoint':'caller-declared'},dataset={'split':'test'})


def test_35_slots_missing_is_not_zero_and_no_advisor(tmp_path):
    report=EvidenceIndex().analyze(context=context())
    d=report.to_dict();assert len(d['analysis']['feature_results'])==35
    assert d['analysis']['coverage']['counts']['missing']==35
    assert all(not s['record_ids'] for s in d['analysis']['feature_results'])
    assert d['analysis']['recommendations']=={'status':'not_implemented','items':[]}
    assert d['analysis']['claims']['independently_verified_improvement']=='not_certified'
    report.to_json(tmp_path/'report.json');report.to_markdown(tmp_path/'report.md')
    assert UnifiedAnalysisReport.from_json(tmp_path/'report.json').to_dict()==d
    assert 'not a zero measurement' in (tmp_path/'report.md').read_text()
    assert 'Percentage points differ from relative percent' in (tmp_path/'report.md').read_text()
    with pytest.raises(FileExistsError):report.to_json(tmp_path/'report.json')
    with pytest.raises(FileExistsError):report.to_markdown(tmp_path/'report.md')


def test_unsupported_and_not_run_are_distinct():
    index=EvidenceIndex([EvidenceRecord.unavailable('P08',reason='INT8 unsupported',status='unsupported'),
        EvidenceRecord.unavailable('P24',reason='Epoch durations not measured',status='not_run'),
        EvidenceRecord.unavailable('P18',reason='Qualified compute roof missing',status='unknown'),
        EvidenceRecord.unavailable('P31',reason='7200-second run not executed',status='not_run')])
    d=index.analyze(context=context()).to_dict()['analysis']
    assert d['coverage']['counts']['unsupported']==1 and d['coverage']['counts']['not_run']==2
    assert d['claims']['qualified_stable_roof']=='not_established_by_this_report'
    assert d['claims']['long_run_7200_acceptance']=='not_certified'


def test_report_derived_fields_and_versions_tamper_rejected(tmp_path):
    d=EvidenceIndex().analyze(context=context()).to_dict();d['analysis']['coverage']['counts']['missing']=0
    with pytest.raises(EvidenceError,match='report_derivation_mismatch'):UnifiedAnalysisReport(d)
    d=EvidenceIndex().analyze(context=context()).to_dict();d['schema_version']=2
    with pytest.raises(EvidenceError,match='unsupported_report_version'):UnifiedAnalysisReport(d)


def test_context_is_declared_defensive_and_no_mutation():
    model={'nested':{'width':3}};c=AnalysisContext(workload_id='x',task='classification',model=model,dataset={})
    model['nested']['width']=99
    assert c.to_dict()['model']['nested']['width']==3
    assert 'caller_declared' in c.to_dict()['authority']


def test_existing_manager_can_store_report_without_schema_changes(tmp_path):
    from hyso.experiments import ExperimentManager
    report=EvidenceIndex().analyze(context=context());path=tmp_path/'analysis.json';report.to_json(path)
    manager=ExperimentManager(tmp_path/'store');manager.close()
    ExperimentManager.migrate_store(tmp_path/'store')  # Existing explicit API; no schema code change.
    manager=ExperimentManager(tmp_path/'store');experiment=manager.create_experiment('analysis')
    run=manager.create_run(experiment['id'],'evaluation',{'analysis':'pure-evidence'})
    run.start();artifact=run.add_artifact(path,kind='report',ownership='copy',role='report')
    checked=manager.verify_artifact(artifact['id']);assert checked['status']=='verified'
    run.finish();manager.close()
    reopened=ExperimentManager(tmp_path/'store');assert reopened.verify_artifact(artifact['id'])['status']=='verified';reopened.close()
