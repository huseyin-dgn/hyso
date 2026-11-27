"""Focused metadata/retained presentation regressions, not acceptance execution."""
import ast
import copy
import json
import subprocess
import typing
from pathlib import Path

import pytest
from tools.product_acceptance import scenarios as s
from tools.product_acceptance.inventory import REPO, build_inventory, export_inventory

@pytest.fixture(autouse=True)
def offline():
    with s.offline_only():yield


def test_root_discovery_resolves_without_workloads_and_preserves_legacy():
    import hyso
    assert len(hyso.__all__)==len(set(hyso.__all__))
    for n in hyso.__all__:
        assert not n.startswith('_') and n in dir(hyso)
        getattr(hyso,n)
    assert hyso.models.__all__==['EBO','ELO','CEMO','EMA','CNNSENormalWithFit']
    for n in ('ModelBuilder','GraphModelBuilder','Trainer','TrainingConfig','analysis','performance','advisor','reporting','evaluation','quality','checkpoint'):
        assert n in hyso.__all__
    with pytest.raises(AttributeError):getattr(hyso,'not_a_public_API')


def test_concrete_result_hints_and_parameter_contracts():
    import hyso
    for name in ('evaluate_classification','evaluate_predictions'):
        assert typing.get_type_hints(getattr(hyso,name))['return'] is hyso.EvaluationResult
    for name in ('measure_time_to_quality','evaluate_robustness','evaluate_precision_quality','compare_performance_quality'):
        assert typing.get_type_hints(getattr(hyso,name))['return'] is hyso.QualityExperimentResult
    for name in ('reopen_stability','measure_streaming','compare_devices','profile_execution_providers'):
        assert typing.get_type_hints(getattr(hyso.performance,name))['return'] is hyso.performance.RuntimeResult
    for namespace in (hyso.advisor,hyso.reporting):
        for name in namespace.__all__:
            value=getattr(namespace,name)
            if isinstance(value,type(lambda:None)) and value.__module__.startswith('hyso.'):
                assert 'return' in typing.get_type_hints(value),name


def test_public_cli_and_python_help_without_workload(tmp_path):
    result=s.U16(tmp_path)
    assert result['guide_topics']==dict.fromkeys(('tr','en','de','es'),len(s.read(s.REPO/'docs/user-guide/content.json')['topics']))


def test_changes_to_existing_computation_are_annotation_or_docstring_only():
    # Unaffected computation remains annotation-only. E01-E06 deliberately
    # factors P35 trace analysis and extends the owned dispatcher; their original
    # safety/provenance semantics have dedicated export/provider regressions.
    paths=['hyso/evaluation.py','hyso/quality.py','hyso/benchmark/runner.py',
        'hyso/performance/devices.py',
        'hyso/performance/streaming.py',*['hyso/advisor/'+n+'.py' for n in
        ('comparison','goals','performance','presentation','profiles','validation')]]
    class Normalize(ast.NodeTransformer):
        def visit_FunctionDef(self,n):
            self.generic_visit(n);n.returns=None
            for a in (*n.args.posonlyargs,*n.args.args,*n.args.kwonlyargs):a.annotation=None
            n.body=[x for x in n.body if not(isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str))]
            return n
        def visit_ClassDef(self,n):
            self.generic_visit(n)
            n.body=[x for x in n.body if not(isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str))]
            return n
        def visit_ImportFrom(self,n):
            if n.module=='runtime_result':n.names=[a for a in n.names if a.name!='RuntimeResult']
            return n
    for path in paths:
        prior=subprocess.check_output(['git','show','HEAD:'+path],cwd=REPO).decode('utf-8')
        before=ast.dump(Normalize().visit(ast.parse(prior)),include_attributes=False)
        after=ast.dump(Normalize().visit(ast.parse((REPO/path).read_text(encoding='utf-8'))),include_attributes=False)
        assert before==after,path


@pytest.mark.parametrize('language',['tr','en','de','es'])
def test_all_feature_explanations_and_retained_evidence_localization(language,tmp_path):
    from hyso import analysis,advisor,reporting
    raw=s.retained('U09','advisor-raw.json');source=analysis.UnifiedAnalysisReport(raw['source'])
    before=source.to_dict();advice=advisor.recommend(source)
    en=reporting.create_report(source,advice,language='en').to_dict()
    report=reporting.create_report(source,advice,language=language)
    assert s.presentation_projection(en)==s.presentation_projection(report.to_dict())
    assert source.to_dict()==before
    rows=reporting.feature_explanations(language=language)
    assert len(rows)==35
    assert [(r['feature_id'],r['units']) for r in rows]==[(r['feature_id'],r['units']) for r in reporting.feature_explanations(language='en')]
    assert all(r['definition'] and r['limits'] and r['purpose'] and r['explanations'] for r in rows)
    p=tmp_path/'report.json';report.to_json(p)
    assert reporting.PresentationReport.from_json(p).to_dict()==report.to_dict()
    assert report.to_markdown() and report.to_html()
    assert advice.to_dict()['claims']==dict(global_winner=None,numerical_improvement=None,automatically_applied=False)


def test_corrupt_provenance_and_unknown_languages_rejected():
    from hyso import analysis,advisor,reporting,QualityExperimentResult
    raw=s.retained('U09','advisor-raw.json')
    corrupt=copy.deepcopy(raw['source']);corrupt['report_id']='forged'
    with pytest.raises(ValueError):analysis.UnifiedAnalysisReport(corrupt)
    source=analysis.UnifiedAnalysisReport(raw['source'])
    with pytest.raises(ValueError):reporting.create_report(source,language='fr')
    with pytest.raises(ValueError):advisor.recommend(source).to_display_dict(language='fr')
    raw=s.retained('U08','two-models-raw.json')['comparison'];QualityExperimentResult(raw)
    corrupt=copy.deepcopy(raw);corrupt['sources']['cohort']['inputs'][0][0][0][0]=123.0
    with pytest.raises(ValueError):QualityExperimentResult(corrupt)


@pytest.mark.parametrize('field,value',[('manual_only',False),('estimated_improvement',0.3),
    ('verified_effect','proven'),('retained_record_ids',['forged'])])
def test_advice_must_keep_bound_manual_unmeasured_claims(field,value):
    from hyso import analysis,advisor
    source=analysis.UnifiedAnalysisReport(s.retained('U09','advisor-raw.json')['source'])
    body=advisor.recommend(source).to_dict()
    s.validate_manual_advice(body,source.to_dict())
    body['recommendations'][0][field]=value
    with pytest.raises(AssertionError):s.validate_manual_advice(body,source.to_dict())


def test_inventory_distinguishes_exports_sources_and_captured_scope():
    import hyso
    from tools.cuda_acceptance.runner import file_hash
    inventory=build_inventory()
    assert len(inventory['components'])==21 and len(inventory['performance_features'])==35
    assert all(r['implemented'] and r['guide']['present'] for r in inventory['components'])
    assert not inventory['full_cuda_acceptance'] and not inventory['typing']['real_IDE_tested']
    apis={r['feature']:r['public_api'] for r in inventory['performance_features']}
    assert apis['P09']=='hyso.BenchmarkRunner'
    assert apis['P13']=='hyso.performance.profile_layers'
    assert apis['P15']=='hyso.performance.measure_transfers'
    assert apis['P24']=='hyso.measure_time_to_quality'
    for name in apis.values():
        value=hyso
        for part in name.split('.')[1:]:value=getattr(value,part)
        assert callable(value)
    for r in inventory['performance_features']:
        record=r['release_support'].get('current_cuda_acceptance')
        if record:
            assert record['status']=='PASS' and record['sha256']==file_hash(record['evidence'])
    assert len(export_inventory())==10
    assert all(r['hyso_conversion_api']=='MISSING' for r in export_inventory())
    assert 'not equivalence' in export_inventory()[0]['output_equivalence']
