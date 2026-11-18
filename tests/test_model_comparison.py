"""Retained fixtures only: no live model, training, benchmark, CUDA or checkpoint I/O."""
import copy
import ast
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import re
import typing

import pytest
import torch

from hyso import analysis
from hyso.analysis._codec import canonical, digest
from hyso.analysis.model_comparison import _benchmark_diagnostic
from hyso.benchmark import BenchmarkResult
from hyso.performance.result import _latency_metrics, LatencyAnalysis
from test_advisor_comparison import payload as old_latency_fixture

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_workload(monkeypatch):
    from hyso import benchmark, training, performance, quality, evaluation, structure
    def forbidden(*a,**kw):pytest.fail('comparison attempted live workload/device/capture/checkpoint access')
    for target,name in ((benchmark.BenchmarkRunner,'run'),(training.Trainer,'fit'),
            (torch.nn.Module,'_call_impl'),(torch,'load'),(quality,'compare_performance_quality'),
            (evaluation,'evaluate_classification'),(structure,'capture_model_structure')):
        monkeypatch.setattr(target,name,forbidden)
    for name in ('init','_lazy_init','is_available','device_count','current_device'):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    for name in ('measure_throughput','measure_memory','measure_footprint'):
        monkeypatch.setattr(performance,name,forbidden)


def source(model_id, feature, native, *, structure=None, benchmark=None):
    record=analysis.EvidenceRecord.from_native(native,feature_id=feature,
        scope=analysis.EvidenceScope(model_id=model_id))
    report=analysis.EvidenceIndex([record]).analyze(context=analysis.AnalysisContext(
        workload_id='retained-test',task='classification',model={'id':model_id},dataset={'kind':'fixture'}))
    return analysis.ModelComparisonSource(model_id,report,structure=structure,benchmark=benchmark)


def latency(run,value=10.,count=1000):
    n=old_latency_fixture(run,value)
    rows=[[value]*count]
    n['protocol'].update(quantiles=[.5,.95,.99],tail_policy=dict(min_samples=1000,min_tail_samples=10))
    n['requested'].update(samples=count);n['actual'].update(samples=count,samples_per_repeat=[count])
    n['provenance']['configuration']['measure_iterations']=count
    n['raw_evidence'].update(latency_samples_ms=rows,source_sha256=digest(dict(run=run,rows=rows)))
    n['provenance']['comparison_capture']['sample_sha256']=digest(rows)
    n['measurements']=_latency_metrics(rows,[.5,.95,.99],n['protocol']['tail_policy'])
    LatencyAnalysis(n)
    return n


def pair_latency(a=None,b=None):
    return source('A','P01',latency('1') if a is None else a),source('B','P01',latency('2',7.) if b is None else b)


def bound(name):
    return json.loads((ROOT/'tests/fixtures/model-comparison-bound-v1.json').read_text(encoding='utf-8'))['payloads'][name]


def test_latency_uses_original_gate_math_directions_and_no_winner():
    a,b=pair_latency()
    result=analysis.compare_models(a,b,features='P01')
    data=result.to_dict()
    assert len(data['metrics'])==3
    for row in data['metrics']:
        assert row['status']=='COMPARABLE'
        assert row['difference']['difference']==-3.
        assert row['difference']['percent_change']==-30.
        assert row['preferred_direction']=='lower' and row['difference']['metric_effect']=='favorable'
    assert data['claims']==dict(global_winner=None,statistical_superiority='not_established',acceptance='NOT_EVALUATED')
    assert analysis.compare_models(a,b,features='P01').to_json()==result.to_json()
    with pytest.raises(FrozenInstanceError):result._payload='forged'


def throughput(run,seconds):
    """Synthetic completed-window ledger, never a measured hardware claim."""
    n=json.loads((ROOT/'tests/fixtures/advisor-native-v1.json').read_text())['payloads']['P02']
    n['result_id']=run+'0000000-0000-0000-0000-000000000000'
    n['protocol']['configuration']['batch_count']=500;n['requested']['batch_count']=500
    batch=n['provenance']['input']['shape'][0]
    for i,w in enumerate(n['raw_evidence']['windows']):
        start=(int(run)*10+i*3)*10**9;end=start+int(seconds*10**9)
        w.update(start_ns=start,end_ns=end,admission_end_ns=end-100,admitted_batches=500,
            completed_batches=500,completed_images=500*batch,completed_id_range=[0,499],
            completion_groups=500,elapsed_wall_seconds=seconds,tail_seconds=100/1e9,
            images_per_second=500*batch/seconds)
    n['actual'].update(completed_batches=1000,completed_images=1000*batch)
    n['measurements'].update(value=1000*batch/(2*seconds),elapsed_wall_seconds=2*seconds)
    common=old_latency_fixture(run,10.)['provenance']
    n['provenance']['configuration']=common['configuration']
    n['provenance']['configuration'].update(n['protocol']['configuration'])
    n['provenance']['input']['layout']='torch.strided'
    n['provenance']['environment'].update(common['environment'])
    n['provenance']['comparison_capture']=common['comparison_capture']
    n['provenance']['comparison_capture'].update(run_id=n['result_id'],sample_sha256=digest(n['raw_evidence']))
    return n


def test_p02_completed_window_protocol_and_higher_direction():
    a,b=source('A','P02',throughput('1',2.)),source('B','P02',throughput('2',1.))
    row=analysis.compare_models(a,b,features='P02').to_dict()['metrics'][0]
    assert row['status']=='COMPARABLE'
    assert row['preferred_direction']=='higher' and row['difference']['metric_effect']=='favorable'
    assert row['difference']['percent_change']==100.
    assert row['comparison']['contract']['min_samples']==1000
    assert row['difference']['baseline']['protocol_json']==row['difference']['candidate']['protocol_json']


def test_rss_footprint_and_gpu_domains_are_not_interchangeable():
    payloads=json.loads((ROOT/'tests/fixtures/advisor-native-v1.json').read_text())['payloads']
    for feature in ('P10','P11'):
        a=source('A',feature,payloads[feature]);other=copy.deepcopy(payloads[feature])
        other['result_id']='80000000-0000-0000-0000-000000000000'
        result=analysis.compare_models(a,source('B',feature,other),features=(feature,'P09'))
        rows=result.to_dict()['metrics']
        assert all(r['difference'] is None for r in rows)
        assert all(r.get('domain')==('process_rss' if feature=='P10' else
            'serialized_artifact_bytes' if r['metric']=='serialized_artifact_bytes' else 'model_logical_bytes')
            for r in rows if r['feature_id']==feature)
        assert rows[-1]['status']=='NOT_SUPPORTED_IN_COMPARISON'
        if feature=='P10':assert all(r['scope']=='process_wide_not_model_exclusive' for r in rows[:-1])


@pytest.mark.parametrize('field,value',[
    ('hardware_id','8'*64),('dataset_id','other'),('split','train'),('cohort_sha256','7'*64),
    ('preprocessing_sha256','9'*64),('input_family_sha256','6'*64),('clock_id','other')])
def test_native_condition_mismatch_withholds_differences(field,value):
    b=latency('2',7.)
    b['provenance']['comparison_capture'][field]=value
    result=analysis.compare_models(*pair_latency(b=b),features='P01')
    assert all(row['status']=='INCOMPATIBLE' and row['difference'] is None for row in result.to_dict()['metrics'])


@pytest.mark.parametrize('section,field,value',[
    ('input','shape',[2,1,2,2]),('environment','device','cuda:0'),
    ('configuration','precision','fp64'),('configuration','warmup_iterations',2)])
def test_shape_device_precision_and_protocol_are_fixed(section,field,value):
    b=latency('2',7.);b['provenance'][section][field]=value
    result=analysis.compare_models(*pair_latency(b=b),features='P01')
    assert all(row['status']=='INCOMPATIBLE' and row['difference'] is None for row in result.to_dict()['metrics'])


def test_missing_capture_and_short_tails_are_not_repaired():
    for n in (latency('2',7.,count=20),latency('2',7.)):
        if n['actual']['samples']==1000:n['provenance'].pop('comparison_capture')
        a,b=pair_latency(b=n)
        original=b.to_dict()
        data=analysis.compare_models(a,b,features='P01').to_dict()
        assert all(row['difference'] is None and row['status']=='INSUFFICIENT_EVIDENCE' for row in data['metrics'])
        assert original==b.to_dict()


def test_same_capture_cannot_be_relabeled_as_two_models():
    n=latency('1')
    result=analysis.compare_models(source('A','P01',n),source('B','P01',n),features='P01')
    assert all(r['status']=='INCOMPATIBLE' and r['difference'] is None for r in result.to_dict()['metrics'])


def test_bound_quality_and_registered_parameter_counts():
    a,b=source('A','P26',bound('A')),source('B','P26',bound('B'))
    data=analysis.compare_models(a,b,features='P26',include_structure=True).to_dict()
    accuracy=next(r for r in data['metrics'] if r['metric']=='test_accuracy')
    assert accuracy['status']=='COMPARABLE'
    assert accuracy['difference']['difference']==.25
    assert accuracy['difference']['percent_change']==pytest.approx(100/3)
    assert accuracy['difference']['percentage_point_change']==25.
    assert accuracy['preferred_direction']=='higher'
    structure=data['structure']
    assert structure['scope']=='native_P26_bound_registered_metadata'
    count=next(r for r in structure['comparison']['summary_differences'] if r['metric']=='parameter_elements')
    assert (count['baseline']['value'],count['candidate']['value'],count['difference'])==(18,10,-8)
    assert count['percent_change']==pytest.approx(-800/18)
    assert 'source_snapshots' not in structure['comparison']


def test_quality_zero_baseline_has_absolute_and_pp_but_no_percent():
    result=analysis.compare_models(source('Zero','P26',bound('Zero')),source('B','P26',bound('B')),features='P26')
    accuracy=next(r for r in result.to_dict()['metrics'] if r['metric']=='test_accuracy')
    assert accuracy['difference']['baseline']['value']==0.
    assert accuracy['difference']['difference']==1.
    assert accuracy['difference']['percent_change'] is None
    assert accuracy['difference']['percentage_point_change']==100.
    assert 'zero_baseline_percent_undefined' in accuracy['comparison']['warnings']


def test_unselected_p26_and_structure_are_not_compared(monkeypatch):
    import hyso.advisor as advisor
    from hyso.evaluation import EvaluationResult
    a,b=bound('A'),bound('B')
    def forbidden(*a,**kw):pytest.fail('unselected quality comparison called')
    monkeypatch.setattr(advisor,'compare_performance',forbidden)
    monkeypatch.setattr(advisor,'compare_p26_model_profiles',forbidden)
    left=source('A','P26',a,structure=EvaluationResult(a).structure_snapshot)
    right=source('B','P26',b,structure=EvaluationResult(b).structure_snapshot)
    data=analysis.compare_models(left,right,features='P11').to_dict()
    assert data['structure']['status']=='NOT_SELECTED'
    assert {r['feature_id'] for r in data['metrics']}=={'P11'}
    assert all(r['status']=='INSUFFICIENT_EVIDENCE' for r in data['metrics'])
    result=analysis.compare_models(left,right,features='P11',include_structure=True)
    assert result.structure_comparison is not None
    assert result.to_dict()['structure']['scope'].startswith('explicit_retained')


def test_missing_and_unconnected_features_have_no_invented_numbers():
    data=analysis.compare_models(*pair_latency(),features=('P02','P09','P11','P19','P31','P32','P35')).to_dict()
    assert all(r['difference'] is None for r in data['metrics'])
    assert {r['feature_id'] for r in data['metrics']}=={'P02','P09','P11','P19','P31','P32','P35'}
    assert data['structure']['status']=='NOT_SELECTED'


def test_model_ids_record_selection_and_two_model_limit():
    a,b=pair_latency()
    with pytest.raises(ValueError):analysis.compare_models(a,a,features='P01')
    with pytest.raises(TypeError):analysis.compare_models(a,b,a,features='P01')
    with pytest.raises(ValueError):analysis.compare_models(a,b,features=('P01','P01'))
    with pytest.raises(ValueError):analysis.compare_models(a,b,features='P99')
    report=analysis.UnifiedAnalysisReport(a.to_dict()['report'])
    identity=a.to_dict()['records']['P01'][0]
    with pytest.raises(analysis.ModelComparisonError):analysis.ModelComparisonSource('B',report,records={'P01':identity})


def test_ambiguous_captures_require_explicit_selection():
    sources=[source('A','P01',latency('1')),source('A','P01',latency('3',9.))]
    records=[analysis.EvidenceRecord(r) for s in sources for r in s.to_dict()['report']['index']['records']]
    report=analysis.EvidenceIndex(records).analyze(context=analysis.AnalysisContext(
        workload_id='multiple-captures',task='classification',model={},dataset={}))
    a=analysis.ModelComparisonSource('A',report);b=pair_latency()[1]
    data=analysis.compare_models(a,b,features='P01').to_dict()
    assert all(r['difference'] is None for r in data['metrics'])
    selected=analysis.ModelComparisonSource('A',report,records={'P01':records[0].record_id})
    assert all(r['status']=='COMPARABLE' for r in analysis.compare_models(selected,b,features='P01').to_dict()['metrics'])


def test_json_reader_rederives_and_rejects_forged_winner_arithmetic_and_versions(tmp_path):
    result=analysis.compare_models(*pair_latency(),features='P01')
    path=tmp_path/'comparison.json';result.to_json(path)
    assert analysis.ModelComparisonResult.from_json(path).to_json()==result.to_json()
    with pytest.raises(FileExistsError):result.to_json(path)
    for field in ('winner','delta','version','extra'):
        d=result.to_dict()
        if field=='winner':d['claims']['global_winner']='B'
        elif field=='delta':d['metrics'][0]['difference']['difference']=-999
        elif field=='version':d['schema_version']=2
        else:d['unrecognized']=True
        d.pop('content_sha256');d['content_sha256']=digest(d)
        with pytest.raises(analysis.ModelComparisonError):analysis.ModelComparisonResult(d)


def test_bilingual_markdown_and_display_reuse_advisor_without_claiming_winner(tmp_path):
    result=analysis.compare_models(source('A','P26',bound('A')),source('B','P26',bound('B')),features='P26',include_structure=True)
    for language in ('tr','en'):
        view=result.to_display_dict(language=language)
        assert view['presentation_only'] and not view['authoritative_evidence']
        assert view['advisor_observations']['baseline']['format']=='hyso.advisor.display'
        assert any(row['metric_effect']=='favorable' for row in view['metrics'])
        path=tmp_path/(language+'.md');raw=result.to_markdown(path,language=language)
        assert path.read_text(encoding='utf-8')==raw
        assert 'statistical' in raw if language=='en' else 'istatistiksel' in raw
        assert 'Percentage-point' in raw if language=='en' else 'Yüzde puan' in raw
    with pytest.raises(ValueError):result.to_display_dict(language='fr')


def test_type_hints_and_no_mutable_source_payload_alias():
    a,b=pair_latency();d=a.to_dict();d['records'].clear()
    assert a.to_dict()['records']
    assert typing.get_type_hints(analysis.compare_models)['return'] is analysis.ModelComparisonResult


def test_retained_execution_adapter_filters_models_without_running_missing_features():
    rows=[]
    for model_id,native in (('A',latency('1')),('B',latency('2',7.))):
        record=analysis.EvidenceRecord.from_native(native,feature_id='P01',
            scope=analysis.EvidenceScope(model_id=model_id))
        rows.extend([
            dict(model_id=model_id,feature_id='P01',status='COMPLETED',reasons=[],
                 evidence_record=record.to_dict(),auxiliary_evidence={}),
            dict(model_id=model_id,feature_id='P11',status='SKIPPED',reasons=['not_authorized'],
                 evidence_record=None,auxiliary_evidence={})])
    retained=analysis.AnalysisExecutionResult(canonical(dict(
        format='hyso.analysis_execution',schema_version=1,features=rows)))
    before=retained.to_dict()
    a,b=(analysis.ModelComparisonSource.from_execution(retained,name) for name in ('A','B'))
    for name,item in (('A',a),('B',b)):
        assert {r['scope']['model_id'] for r in item.to_dict()['report']['index']['records']}=={name}
    result=analysis.compare_models(a,b,features=('P01','P11')).to_dict()
    assert all(r['status']=='COMPARABLE' for r in result['metrics'] if r['feature_id']=='P01')
    assert all(r['status']=='INSUFFICIENT_EVIDENCE' and r['difference'] is None
               for r in result['metrics'] if r['feature_id']=='P11')
    assert retained.to_dict()==before
    with pytest.raises(analysis.ModelComparisonError,match='model_not_present'):
        analysis.ModelComparisonSource.from_execution(retained,'missing')


@pytest.mark.parametrize('document',['README.md','docs/guides/python-api.md'])
def test_documented_saved_report_comparison_executes_without_workloads(document,tmp_path):
    text=(ROOT/document).read_text(encoding='utf-8')
    examples=re.findall(r'^```python\n(.*?)^```',text,re.M|re.S)
    example,=[code for code in examples if 'def compare_saved_models(' in code]
    ast.parse(example,feature_version=(3,10))
    namespace={}
    exec(compile(example,document,'exec'),namespace)
    paths=[]
    for name in ('A','B'):
        report=analysis.UnifiedAnalysisReport(source(name,'P26',bound(name)).to_dict()['report'])
        path=tmp_path/(name+'.json');report.to_json(path);paths.append(path)
    result=namespace['compare_saved_models'](*paths)
    data=result.to_dict()
    accuracy=next(row for row in data['metrics'] if row['metric']=='test_accuracy')
    assert accuracy['status']=='COMPARABLE' and accuracy['difference']['percentage_point_change']==25.
    assert all(row['difference'] is None for row in data['metrics'] if row['feature_id']!='P26')
    assert data['structure']['status']=='COMPARABLE' and data['claims']['global_winner'] is None
    exported=tmp_path/'comparison.json';result.to_json(exported)
    assert analysis.ModelComparisonResult.from_json(exported).to_dict()==data
    for language in ('tr','en'):
        result.to_markdown(tmp_path/(language+'.md'),language=language)


def test_comparison_guide_projection_matches_retained_result():
    text=(ROOT/'docs/guides/two-model-comparison.md').read_text(encoding='utf-8')
    example,=re.findall(r'^```json\n(.*?)^```',text,re.M|re.S)
    projection=json.loads(example)
    result=analysis.compare_models(source('A','P26',bound('A')),
        source('B','P26',bound('B')),features='P26',include_structure=True).to_dict()
    row=next(r for r in result['metrics'] if r['metric']==projection['metric'])
    for key,value in projection.items():
        if key=='claims':
            assert all(result['claims'][k]==v for k,v in value.items())
        elif key=='difference':
            assert all(row['difference'][k]==v for k,v in value.items())
        else:
            assert row[key]==value
