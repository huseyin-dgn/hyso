"""Short trusted CPU routing/authorization tests, not full acceptance or GPU tests."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import typing

import pytest
import torch
from torch import nn

from hyso import analysis, advisor, ExperimentManager
from hyso.analysis import execution as engine
from hyso.analysis._codec import EvidenceError, canonical
from hyso.analysis.catalog import EXECUTOR_FEATURES


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('common CPU executor attempted CUDA initialization')
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    monkeypatch.setattr(engine,'memory_availability',lambda:dict(status='observed',
        available_physical_bytes=8*1024**3,available_commit_bytes=8*1024**3,source='test_OS_snapshot'))
    torch.manual_seed(31)


def context(name='A',**kwargs):
    return analysis.ModelContext(name,input_contract=True,**kwargs)


def binding(name='A',model=None):
    return analysis.LiveCPUModel(name,nn.Sequential(nn.Flatten(),nn.Linear(4,2)) if model is None else model,
                                 torch.ones(2,1,2,2))


def outcomes(result):
    return {(r['model_id'],r['feature_id']):r for r in result.to_dict()['features']}


@pytest.mark.parametrize('profile',['basic','inference','training','memory','comparison','full'])
def test_profiles_catalog_and_selection_are_pure(profile,monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('selection performed preflight or execution')
    monkeypatch.setattr(engine,'memory_availability',forbidden)
    monkeypatch.setattr(engine,'_invoke',forbidden)
    plan=analysis.plan_analysis(context(),profile=profile)
    data=plan.to_dict()
    assert data['selected_features']==list(analysis.PROFILES[profile])
    assert data['execution_authorized'] is False and data['acceptance']=='NOT_EVALUATED'
    assert len(analysis.feature_catalog())==35
    for spec in analysis.feature_catalog():
        assert spec.description and spec.cost_class and spec.applicability and spec.executor_scope
        assert spec.supported_devices==(('cpu',) if spec.executor_modes else ())
    assert analysis.feature_catalog('P09').required_capabilities==('gpu',)
    assert analysis.feature_catalog('P31').cost_class=='long_7200s'
    assert analysis.feature_catalog('P32').dependencies==('P31',)


@pytest.mark.parametrize('kwargs',[{},dict(features='P01',profile='basic'),dict(features='P99'),
    dict(features=('P01','P01')),dict(profile='unknown'),dict(features='P01',device='cuda'),
    dict(features='P01',execution_mode='automatic'),dict(features='P01',policy={})])
def test_invalid_selection_rejected(kwargs):
    with pytest.raises(ValueError):analysis.plan_analysis(context(),**kwargs)


def test_old_g1_request_and_dependency_visibility_without_auto_execution(monkeypatch):
    req=analysis.FeaturePlanRequest('single',(context(),),features=('P32',))
    legacy=analysis.plan_features(req).to_dict()
    plan=analysis.plan_analysis(req)
    dry=analysis.preflight_analysis(plan,(binding(),))
    monkeypatch.setattr(engine,'_invoke',lambda *a:pytest.fail('P31/P32 must not execute'))
    result=analysis.execute_analysis(dry,authorized_features=('P32',),explicit_authorization=True)
    rows=outcomes(result)
    assert rows['A','P31']['status']=='SKIPPED'
    assert rows['A','P32']['status']=='MISSING_PREREQUISITE'
    assert rows['A','P32']['executor_support']=='NOT_IMPLEMENTED_IN_EXECUTOR'
    assert analysis.plan_features(req).to_dict()==legacy
    assert legacy['execution_authorized'] is False


def test_authorization_exact_subset_no_implicit_work_and_one_shot(monkeypatch):
    dry=analysis.preflight_analysis(analysis.plan_analysis(context(),features='P11'),(binding(),))
    monkeypatch.setattr(engine,'_invoke',lambda *a:pytest.fail('authorization failed open'))
    for features,consent in [(('P11',),False),(('P01',),True),(('P11','P11'),True),('P11',True)]:
        with pytest.raises(analysis.AnalysisExecutionError):
            analysis.execute_analysis(dry,authorized_features=features,explicit_authorization=consent)
    result=analysis.execute_analysis(dry,authorized_features=())
    assert outcomes(result)['A','P11']['status']=='SKIPPED'
    with pytest.raises(analysis.AnalysisExecutionError):
        analysis.execute_analysis(dry,authorized_features=('P11',),explicit_authorization=True)


def test_only_authorized_row_runs_and_native_state_is_preserved():
    b=binding(); saved={k:v.clone() for k,v in b.model.state_dict().items()}
    mode=b.model.training
    dry=analysis.preflight_analysis(analysis.plan_analysis(context(),profile='basic'),(b,))
    result=analysis.execute_analysis(dry,authorized_features=('P11',),explicit_authorization=True)
    rows=outcomes(result)
    assert rows['A','P11']['status']=='COMPLETED'
    assert {row['status'] for key,row in rows.items() if key[1]!='P11'}=={'SKIPPED'}
    assert b.model.training==mode
    assert all(torch.equal(v,b.model.state_dict()[k]) for k,v in saved.items())


def test_real_four_cpu_adapters_native_evidence_and_unchanged_tail_policy(tmp_path):
    b=binding();mode=b.model.training;before={k:v.clone() for k,v in b.model.state_dict().items()}
    dry=analysis.preflight_analysis(analysis.plan_analysis(context(),profile='basic'),(b,))
    assert {r['status'] for r in dry.to_dict()['features']}=={'SUPPORTED'}
    result=analysis.execute_analysis(dry,authorized_features=('P01','P02','P10','P11'),explicit_authorization=True)
    rows=outcomes(result)
    assert {r['status'] for r in rows.values()}=={'COMPLETED'},result.to_dict()
    latency=rows['A','P01']['evidence_record']['native_payload']
    assert latency['actual']['samples']==1000
    assert latency['protocol']['tail_policy']==dict(min_samples=1000,min_tail_samples=10)
    assert all(m['value'] is not None for m in latency['measurements'])
    assert rows['A','P01']['auxiliary_evidence']['benchmark']['configuration']['memory_probe_count']==0
    memory=rows['A','P10']['evidence_record']
    assert memory['native_payload']['protocol']['workload']=='snapshot'
    assert memory['native_payload']['actual']['observations']==1
    assert memory['bindings']['model_id']=='declared'
    assert b.model.training==mode and all(torch.equal(v,b.model.state_dict()[k]) for k,v in before.items())
    index=result.to_evidence_index()
    assert len(index.records)==4
    index.to_json(tmp_path/'index.json')
    assert analysis.EvidenceIndex.from_json(tmp_path/'index.json').to_dict()==index.to_dict()
    report=result.to_unified_report(context=analysis.AnalysisContext(workload_id='toy',task='classification',
        model={'id':'A'},dataset={'kind':'synthetic'}))
    advice=advisor.advise(report)
    assert advice.to_dict()
    path=tmp_path/'execution.json';result.to_json(path)
    with pytest.raises(FileExistsError):result.to_json(path)
    store=tmp_path/'store'
    ExperimentManager(store).close()
    ExperimentManager.migrate_store(store)  # Explicit, offline and temp-store-only.
    manager=ExperimentManager(store)
    try:
        exp=manager.create_experiment('selected analysis')
        run=manager.create_run(exp['id'],'benchmark',dry.tracking_config())
        run.start()
        artifact=run.add_artifact(path,kind='analysis_execution',ownership='copy',role='output')
        assert artifact
        run.finish()
    finally:manager.close()


def test_insufficient_p01_budget_retains_evidence_without_success_claim():
    policy=analysis.CPUAnalysisPolicy(warmup_iterations=0,latency_measure_iterations=3)
    dry=analysis.preflight_analysis(analysis.plan_analysis(context(),features='P01',policy=policy),(binding(),))
    row=outcomes(analysis.execute_analysis(dry,authorized_features=('P01',),explicit_authorization=True))['A','P01']
    assert row['status']=='MISSING_PREREQUISITE' and row['attempted'] is True
    assert row['evidence_record']['native_payload']['actual']['samples']==3
    assert all(m['value'] is None for m in row['evidence_record']['native_payload']['measurements'])


def test_p10_snapshot_does_not_call_model():
    class NoForward(nn.Module):
        def forward(self,x):pytest.fail('P10 snapshot must not run inference')
    dry=analysis.preflight_analysis(analysis.plan_analysis(context(),features='P10'),(binding(model=NoForward()),))
    result=analysis.execute_analysis(dry,authorized_features=('P10',),explicit_authorization=True)
    assert outcomes(result)['A','P10']['status']=='COMPLETED'


def test_missing_binding_and_resources_do_not_execute(monkeypatch):
    plan=analysis.plan_analysis(context(),features=('P01','P11'))
    assert {r['status'] for r in analysis.preflight_analysis(plan).to_dict()['features']}=={'MISSING_PREREQUISITE'}
    dry=analysis.preflight_analysis(plan,(binding(),))
    monkeypatch.setattr(engine,'memory_availability',lambda:dict(status='unavailable'))
    monkeypatch.setattr(engine,'_invoke',lambda *a:pytest.fail('resources were not rechecked'))
    result=analysis.execute_analysis(dry,authorized_features=('P01','P11'),explicit_authorization=True)
    assert {r['status'] for r in result.to_dict()['features']}=={'MISSING_PREREQUISITE'}


@pytest.mark.parametrize('change',[
    dict(max_input_bytes=1),dict(max_model_bytes=1),dict(min_available_memory_bytes=9*1024**3),
    dict(min_available_commit_bytes=9*1024**3)])
def test_cpu_resource_limits_are_honored(change):
    policy=analysis.CPUAnalysisPolicy(**change)
    dry=analysis.preflight_analysis(analysis.plan_analysis(context(),features='P01',policy=policy),(binding(),))
    assert dry.to_dict()['features'][0]['status']=='MISSING_PREREQUISITE'


@pytest.mark.parametrize('change',[dict(latency_measure_iterations=0),dict(repeats=4),dict(quantiles=(float('nan'),)),
    dict(quantiles=(.5,.5)),dict(min_available_memory_bytes=0)])
def test_invalid_policy(change):
    with pytest.raises(ValueError):analysis.CPUAnalysisPolicy(**change)


def test_binding_identity_and_dtype_errors_before_native_work():
    plan=analysis.plan_analysis(context(),features='P01')
    with pytest.raises(analysis.AnalysisExecutionError):analysis.preflight_analysis(plan,(binding('wrong'),))
    b=binding();b.model.double()
    assert analysis.preflight_analysis(plan,(b,)).to_dict()['features'][0]['status']=='UNSUPPORTED'
    b=analysis.LiveCPUModel('A',nn.Identity(),torch.ones(2,1,2,2,dtype=torch.float64))
    assert analysis.preflight_analysis(plan,(b,)).to_dict()['features'][0]['status']=='MISSING_PREREQUISITE'


def test_failure_does_not_hide_or_stop_independent_features_and_models(monkeypatch):
    plan=analysis.plan_analysis((context('A'),context('B')),features=('P01','P11'),
                                policy=analysis.CPUAnalysisPolicy(latency_measure_iterations=3))
    native=engine._invoke
    def fail_a(plan,context,binding,feature,policy):
        if context.model_id=='A' and feature=='P01':raise RuntimeError('controlled failure')
        return native(plan,context,binding,feature,policy)
    monkeypatch.setattr(engine,'_invoke',fail_a)
    result=analysis.execute_analysis(analysis.preflight_analysis(plan,(binding('B'),binding('A'))),
        authorized_features=('P01','P11'),explicit_authorization=True)
    rows=outcomes(result)
    assert rows['A','P01']['status']=='FAILED'
    assert rows['A','P11']['status']==rows['B','P11']['status']=='COMPLETED'
    assert rows['B','P01']['status']=='MISSING_PREREQUISITE'
    assert len(result.to_evidence_index().records)==4


def test_wrong_native_feature_cannot_be_completed(monkeypatch):
    dry=analysis.preflight_analysis(analysis.plan_analysis(context(),features='P11'),(binding(),))
    monkeypatch.setattr(engine,'_invoke',lambda *a:({'feature_ids':['P02']},{},{'status':'COMPLETED'}))
    row=outcomes(analysis.execute_analysis(dry,authorized_features=('P11',),explicit_authorization=True))['A','P11']
    assert row['status']=='FAILED' and row['evidence_record'] is None


def test_cpu_p09_cuda_and_other_unwired_features_are_never_invoked(monkeypatch):
    monkeypatch.setattr(engine,'_invoke',lambda *a:pytest.fail('unwired/GPU analysis executed'))
    for device,features in [('cpu',('P03','P09','P19','P30','P31','P32','P33','P35')),('cuda:0',('P01','P09'))]:
        plan=analysis.plan_analysis(context(),features=features,device=device)
        result=analysis.execute_analysis(analysis.preflight_analysis(plan,(binding(),)),
            authorized_features=features,explicit_authorization=True)
        assert all(not row['attempted'] for row in result.to_dict()['features'])
        assert outcomes(result)['A','P09']['status']=='UNSUPPORTED'


def test_in_memory_evidence_has_real_byte_identity_and_no_implicit_file(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    result=analysis.execute_analysis(analysis.preflight_analysis(analysis.plan_analysis(context(),features='P11'),(binding(),)),
        authorized_features=('P11',),explicit_authorization=True)
    row=outcomes(result)['A','P11'];record=row['evidence_record']
    raw=canonical(record['native_payload']).encode('utf-8')
    assert record['artifact']['sha256']==hashlib.sha256(raw).hexdigest()
    assert record['artifact']['bytes']==len(raw) and list(tmp_path.iterdir())==[]
    d=record['native_payload'];d['measurements']['parameter_buffer_bytes']['value']+=1
    with pytest.raises(EvidenceError):analysis.EvidenceRecord.from_native(d,feature_id='P11')


def test_public_aliases_policy_immutability_and_annotations():
    from hyso.analysis.native_bridge import execute_scoped_cpu
    from hyso.analysis.isolated_bridge import execute_isolated_cpu
    assert analysis.execute_scoped_cpu is execute_scoped_cpu
    assert analysis.execute_isolated_cpu is execute_isolated_cpu
    policy=analysis.CPUAnalysisPolicy()
    with pytest.raises(FrozenInstanceError):policy.batch_count=5
    for target in (analysis.plan_analysis,analysis.preflight_analysis,analysis.execute_analysis):
        assert typing.get_type_hints(target)['return']


def test_uninitialized_binding_does_not_stop_independent_model():
    plan=analysis.plan_analysis((context('A'),context('B')),features='P11')
    dry=analysis.preflight_analysis(plan,(binding('A',nn.LazyLinear(2)),binding('B')))
    result=analysis.execute_analysis(dry,authorized_features=('P11',),explicit_authorization=True)
    rows=outcomes(result)
    assert rows['A','P11']['status']=='MISSING_PREREQUISITE'
    assert rows['B','P11']['status']=='COMPLETED'


def test_changed_binding_metadata_requires_new_review_before_any_execution(monkeypatch):
    b=binding();plan=analysis.plan_analysis(context(),features='P11')
    dry=analysis.preflight_analysis(plan,(b,))
    b.input_tensor.resize_(1,1,2,2)
    monkeypatch.setattr(engine,'_invoke',lambda *a:pytest.fail('changed metadata was not rejected'))
    with pytest.raises(analysis.AnalysisExecutionError,match='metadata_changed'):
        analysis.execute_analysis(dry,authorized_features=('P11',),explicit_authorization=True)


def test_dependency_only_cannot_be_authorized_implicitly():
    dry=analysis.preflight_analysis(analysis.plan_analysis(context(),features='P32'),(binding(),))
    with pytest.raises(analysis.AnalysisExecutionError,match='subset'):
        analysis.execute_analysis(dry,authorized_features=('P31',),explicit_authorization=True)


def test_interrupt_propagates_without_retry_or_continuation(monkeypatch):
    dry=analysis.preflight_analysis(analysis.plan_analysis(context(),features=('P01','P11')),(binding(),))
    def interrupted(*args):raise KeyboardInterrupt()
    monkeypatch.setattr(engine,'_invoke',interrupted)
    with pytest.raises(KeyboardInterrupt):
        analysis.execute_analysis(dry,authorized_features=('P01','P11'),explicit_authorization=True)
    with pytest.raises(analysis.AnalysisExecutionError,match='one_shot'):
        analysis.execute_analysis(dry,authorized_features=('P11',),explicit_authorization=True)


def test_common_guide_examples_are_cpu_only_and_no_implicit_exports(tmp_path,monkeypatch):
    import ast,re
    path=Path(__file__).resolve().parents[1]/'docs/guides/selective-analysis.md'
    examples=re.findall(r'^```python\n(.*?)^```',path.read_text(encoding='utf-8'),re.M|re.S)
    assert len(examples)==3
    monkeypatch.chdir(tmp_path);namespace={}
    for example in examples:
        ast.parse(example,feature_version=(3,10))
        exec(compile(example,str(path),'exec'),namespace)
    assert namespace['result'].to_dict()['status_counts']['COMPLETED']==1
    assert list(tmp_path.iterdir())==[]


def test_actual_os_resource_query_has_no_execution():
    from hyso.analysis._analysis_resources import memory_availability
    values=memory_availability()
    if os.name=='nt' or __import__('sys').platform.startswith('linux'):
        assert values['status']=='observed'
        assert values['available_physical_bytes']>=0 and values['available_commit_bytes']>=0


@pytest.mark.skipif(os.name!='nt',reason='existing Windows Job backend only')
@pytest.mark.parametrize('feature',['P02','P11'])
def test_common_isolated_real_owned_cpu_worker(feature):
    path=Path(__file__).with_name('g3_trusted_factory.py').resolve()
    b=analysis.TrustedFactoryBinding('A',str(path),hashlib.sha256(path.read_bytes()).hexdigest(),kwargs={'width':2})
    plan=analysis.plan_analysis(context(trusted_factory=True),features=feature,execution_mode='isolated_cpu')
    dry=analysis.preflight_analysis(plan,(b,))
    result=analysis.execute_analysis(dry,authorized_features=(feature,),explicit_authorization=True)
    row=outcomes(result)['A',feature]
    assert row['status']=='COMPLETED',row
    receipt=row['execution_receipt']['worker_receipt']
    assert receipt['assigned_before_resume'] and receipt['active_owned_processes']==0


def test_isolated_dry_run_never_imports_factory_and_no_fallback(tmp_path,monkeypatch):
    p=tmp_path/'factory.py';p.write_text('raise RuntimeError("must not import at preflight")',encoding='utf-8')
    b=analysis.TrustedFactoryBinding('A',str(p.resolve()),hashlib.sha256(p.read_bytes()).hexdigest())
    plan=analysis.plan_analysis(context(trusted_factory=True),features=('P01','P11'),execution_mode='isolated_cpu')
    dry=analysis.preflight_analysis(plan,(b,))
    assert dry.to_dict()['features'][0]['status']=='NOT_IMPLEMENTED_IN_EXECUTOR'
    def failed(*a,**kw):
        from hyso.analysis.isolated_bridge import IsolatedCPUReport
        return IsolatedCPUReport('FAILED','controlled timeout',None,None,('P11',),('P11',))
    monkeypatch.setattr(engine,'execute_isolated_cpu',failed)
    monkeypatch.setattr(engine,'execute_scoped_cpu',lambda *a,**kw:pytest.fail('isolated failure fell back in process'))
    if os.name=='nt':
        row=outcomes(analysis.execute_analysis(dry,authorized_features=('P11',),explicit_authorization=True))['A','P11']
        assert row['status']=='FAILED' and row['execution_receipt']['reason']=='controlled timeout'
    p.write_text('changed',encoding='utf-8')
    assert analysis.preflight_analysis(plan,(b,)).to_dict()['features'][-1]['status']=='MISSING_PREREQUISITE'
