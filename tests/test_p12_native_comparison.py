"""Tiny owned CPU P12 captures, then read-only comparison and presentation."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from hyso import analysis, advisor, performance
from hyso.reporting import create_report, PresentationReport
from hyso.analysis._codec import digest
from hyso.advisor.p12_observation import observe_p12


def capture_pair(precision):
    protocol=performance.ProfilingProtocol(warmup_iterations=0,measure_iterations=1,timeout_seconds=30,
        cpu_memory_limit_bytes=3*1024**3,input_limit_bytes=4096,max_events=1000,max_trace_bytes=1024**2)
    result=[]
    for outputs in (4,2):
        recipe=performance.ColdStartRecipe('profiling_factories:spatial',model_kwargs={'outputs':outputs},
            factory_search_path=Path(__file__).parent,input_shape=(1,2,3,3),device='cpu',precision=precision)
        profile=performance.profile_operators(recipe,protocol=protocol)
        result.append((profile,performance.analyze_flops(profile)))
    return result


@pytest.fixture(scope='module')
def captures():
    return capture_pair('fp32')


@pytest.fixture(scope='module')
def captures_fp64():
    return capture_pair('fp64')


@pytest.fixture(scope='module')
def zero_capture(tmp_path_factory):
    artifact=tmp_path_factory.mktemp('native-p12-checkpoint')/'identity.pth'
    torch.save({},artifact)
    sha=hashlib.sha256(artifact.read_bytes()).hexdigest()
    recipe=performance.ColdStartRecipe('torch.nn:Identity',input_shape=(1,2,3,3),
        artifact_path=artifact,artifact_sha256=sha,device='cpu',precision='fp32')
    protocol=performance.ProfilingProtocol(warmup_iterations=0,measure_iterations=1,timeout_seconds=30,
        cpu_memory_limit_bytes=3*1024**3,input_limit_bytes=4096,max_events=1000,max_trace_bytes=1024**2)
    profile=performance.profile_operators(recipe,protocol=protocol)
    return performance.analyze_flops(profile).to_dict(),sha


@pytest.fixture(autouse=True)
def read_only_after_capture(captures,monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('retained P12 comparison attempted live work')
    from hyso import training, benchmark, evaluation
    for target,name in ((torch,'load'),(torch.nn.Module,'_call_impl'),(training.Trainer,'fit'),
                        (benchmark.BenchmarkRunner,'run'),(evaluation,'evaluate_classification'),
                        (performance,'profile_operators'),(performance.profiling,'_execute_profile')):
        monkeypatch.setattr(target,name,forbidden)
    for name in ('init','_lazy_init','is_available','device_count','current_device','synchronize'):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    assert not torch.cuda.is_initialized()


def source(label,native,**scope):
    record=analysis.EvidenceRecord.from_native(native,feature_id='P12',scope=analysis.EvidenceScope(model_id=label,**scope))
    report=analysis.EvidenceIndex([record]).analyze(context=analysis.AnalysisContext(
        workload_id='native-CPU-P12',task='analytic observed operations',model={'id':label},dataset={}))
    return analysis.ModelComparisonSource(label,report)


def pair(captures):
    return [source(label,item[1].to_dict()) for label,item in zip(('A','B'),captures)]


def test_native_receipt_and_two_model_counts(captures):
    for profile,result in captures:
        p,n=profile.to_dict(),result.to_dict()
        binding=n['raw_evidence']['p12_binding']
        assert binding==p['raw_evidence']['p12_binding']
        assert binding['source_profile_id']==p['result_id']==n['identity']['source_result_id']
        assert binding['model_state_sha256']==binding['post_model_state_sha256']==n['provenance']['model']['weights_sha256']
        assert binding['input_sha256']==binding['post_input_sha256']==n['provenance']['input']['content_sha256']
        assert binding['device']=='cpu' and binding['threads']==1
        assert binding['checkpoint_artifact_sha256'] is None
    sources=pair(captures);before=[s.to_dict() for s in sources]
    result=analysis.compare_models(*sources,features='P12')
    assert result.to_dict()['schema_version']==3
    rows={r['metric']:r for r in result.to_dict()['metrics']}
    for key,values in (('macs_per_batch',(648,324,-324)),('logical_flops_per_batch',(1332,666,-666))):
        row=rows[key];diff=row['difference']
        assert row['status']=='COMPARABLE'
        assert (diff['baseline']['value'],diff['candidate']['value'],diff['difference'])==values
        assert diff['percent_change']==-50 and diff['metric_effect']=='favorable'
        assert diff['baseline']['reference']['native_binding']['source_profile_id']
    assert [s.to_dict() for s in sources]==before


def test_native_fp64_pair_and_cross_precision_rejection(captures,captures_fp64):
    result=analysis.compare_models(*pair(captures_fp64),features='P12')
    assert all(r['status']=='COMPARABLE' and r['difference']['percent_change']==-50 for r in result.to_dict()['metrics'])
    mixed=analysis.compare_models(source('A',captures[0][1].to_dict()),
        source('B',captures_fp64[1][1].to_dict()),features='P12')
    assert all(r['status']=='INCOMPATIBLE' and r['difference'] is None for r in mixed.to_dict()['metrics'])


@pytest.mark.parametrize('field', ['model_state_sha256','post_model_state_sha256','structure_sha256',
                                  'input_sha256','aten_calls_sha256','trace_sha256','schema_version','threads'])
def test_native_receipt_tampering_is_rejected(captures,field):
    native=copy.deepcopy(captures[0][1].to_dict())
    native['raw_evidence']['p12_binding'][field]=True if field in ('schema_version','threads') else 'b'*64
    with pytest.raises(ValueError,match='invalid_native_evidence'):
        source('forged',native)


@pytest.mark.parametrize('declaration', [False,True])
def test_missing_native_receipt_cannot_be_repaired_by_declarations(captures,declaration):
    sources=[]
    for label,item in zip(('A','B'),captures):
        native=item[1].to_dict();native['raw_evidence'].pop('p12_binding')
        options={} if not declaration else dict(checkpoint_sha256='b'*64,dataset_id='same',split='test')
        sources.append(source(label,native,**options))
    result=analysis.compare_models(*sources,features='P12')
    assert all(r['status']=='INSUFFICIENT_EVIDENCE' and r['difference'] is None
               and any('native_p12_binding_missing' in reason for reason in r['reasons']) for r in result.to_dict()['metrics'])


def test_declared_state_cannot_override_native_identity(captures):
    with pytest.raises(ValueError,match='native_scope_mismatch:model_state_sha256'):
        source('caller-label',captures[0][1].to_dict(),model_state_sha256='b'*64)


def test_checkpoint_declaration_remains_missing_without_artifact_receipt(captures):
    result=analysis.compare_models(source('A',captures[0][1].to_dict(),checkpoint_sha256='b'*64),
                                  source('B',captures[1][1].to_dict(),checkpoint_sha256='b'*64),features='P12')
    assert all(r['status']=='INSUFFICIENT_EVIDENCE' and r['difference'] is None
               and any('native_checkpoint_association_missing' in reason for reason in r['reasons'])
               for r in result.to_dict()['metrics'])


@pytest.mark.parametrize('field', ['input','configuration','torch'])
def test_known_condition_mismatch_wins_over_missing_receipt(captures,field):
    native=captures[1][1].to_dict();native['raw_evidence'].pop('p12_binding')
    if field=='input': native['provenance']['input']['content_sha256']='b'*64
    if field=='configuration': native['protocol']['configuration']['timeout_seconds']=29
    if field=='torch': native['provenance']['environment']['torch']='different-runtime'
    result=analysis.compare_models(source('A',captures[0][1].to_dict()),source('B',native),features='P12')
    assert all(r['status']=='INCOMPATIBLE' and r['difference'] is None
               and 'condition_mismatch:'+field in r['reasons'] for r in result.to_dict()['metrics'])


def test_partial_operation_coverage_keeps_totals_and_differences_null(captures):
    from hyso.performance._operation_models import summarize
    from hyso.performance.profiling import _hash
    # Controlled reader mutation, not a new capture or accepted native evidence.
    native=captures[1][1].to_dict();raw=native['raw_evidence']
    raw['aten_calls'][0]['operator']='test.unmodeled'
    raw['operations'],native['measurements']=summarize(raw['aten_calls'],1,1,model_version=3)
    raw['p12_binding']['aten_calls_sha256']=_hash(raw['aten_calls'])
    result=analysis.compare_models(source('A',captures[0][1].to_dict()),source('B',native),features='P12')
    assert all(r['status']=='INSUFFICIENT_EVIDENCE' and r['difference'] is None for r in result.to_dict()['metrics'])
    observation=observe_p12(analysis.EvidenceRecord.from_native(native,feature_id='P12'),version=1)
    assert all(m['value'] is None for m in observation.to_dict()['metrics'].values())


def test_same_capture_is_not_two_models(captures):
    native=captures[0][1].to_dict()
    result=analysis.compare_models(source('A',native),source('B',native),features='P12')
    assert all(r['status']=='INCOMPATIBLE' and r['difference'] is None
               and any('same_underlying_capture' in reason for reason in r['reasons']) for r in result.to_dict()['metrics'])


@pytest.mark.parametrize('damage', ['difference','receipt_reference','eligibility','policy'])
def test_rehashed_comparison_cannot_forge_results(captures,damage):
    data=analysis.compare_models(*pair(captures),features='P12').to_dict()
    if damage=='difference':data['metrics'][0]['difference']['difference']=0
    if damage=='receipt_reference':data['metrics'][0]['difference']['baseline']['reference']['native_binding']['input_sha256']='b'*64
    if damage=='eligibility':data['metrics'][0]['status']='INSUFFICIENT_EVIDENCE'
    if damage=='policy':data['p12_comparison_policy']['allowed_changes'].append('precision')
    data['content_sha256']=digest({k:v for k,v in data.items() if k!='content_sha256'})
    with pytest.raises(analysis.ModelComparisonError):analysis.ModelComparisonResult(data)


@pytest.mark.parametrize('language', ['tr','en'])
def test_native_P12_reporting_advisor_and_saved_roundtrip(captures,tmp_path,language):
    result=analysis.compare_models(*pair(captures),features='P12',include_structure=True)
    before=result.to_json();path=tmp_path/'comparison.json';result.to_json(path)
    assert analysis.ModelComparisonResult.from_json(path).to_json()==before
    recommendations=advisor.recommend(result)
    observations=recommendations.to_display_dict(language=language)['comparison_observations']
    assert {o['metric'] for o in observations}=={'macs_per_batch','logical_flops_per_batch'}
    assert all(o['manual_only'] and not o['automatically_applied'] for o in observations)
    assert all(('do not establish lower latency' if language=='en' else 'daha düşük gecikme') in o['statement'] for o in observations)
    report=create_report(result,recommendations,language=language)
    data=report.to_dict();assert data['presentation_only'] and not data['authoritative_evidence']
    assert data['sources'][0]['raw']==result.to_dict()
    assert data['sources'][0]['raw']['structure']['status']=='INSUFFICIENT_EVIDENCE'
    saved=tmp_path/'presentation.json';report.to_json(saved)
    reopened=PresentationReport.from_json(saved)
    for text in (reopened.to_json(),reopened.to_markdown(),reopened.to_html()):
        assert 'MAC/batch' in text and 'logical_FLOP/batch' in text and 'COMPARABLE' in text
        assert '648' in text and '324' in text and '-50' in text
        assert 'source_profile_id' in text
    assert result.to_json()==before


def test_historical_P12_comparison_reader_is_not_silently_upgraded(captures):
    from hyso.analysis.model_comparison import _derive
    old,_=_derive(*pair(captures),('P12',),False)
    assert old['schema_version']==1 and old['metrics'][0]['status']=='NOT_SUPPORTED_IN_COMPARISON'
    assert analysis.ModelComparisonResult(old).to_dict()==old


def test_missing_selected_P12_evidence_has_no_difference(captures):
    empty=analysis.EvidenceIndex([]).analyze(context=analysis.AnalysisContext(
        workload_id='not-captured',task='analytic operations',model={},dataset={}))
    result=analysis.compare_models(source('A',captures[0][1].to_dict()),
        analysis.ModelComparisonSource('B',empty),features='P12')
    assert all(r['status']=='INSUFFICIENT_EVIDENCE' and r['difference'] is None for r in result.to_dict()['metrics'])


def test_observed_zero_work_and_real_checkpoint_binding(captures,zero_capture):
    native,artifact_sha=zero_capture
    binding=native['raw_evidence']['p12_binding']
    assert native['raw_evidence']['aten_calls']==[]
    assert binding['checkpoint_artifact_sha256']==artifact_sha
    assert binding['model_state_sha256']!=artifact_sha
    result=analysis.compare_models(source('A',native,checkpoint_sha256=artifact_sha),
        source('B',captures[1][1].to_dict()),features='P12')
    assert all(r['status']=='COMPARABLE' and r['difference']['baseline']['value']==0
               and r['difference']['percent_change'] is None for r in result.to_dict()['metrics'])
    invalid=analysis.compare_models(source('A',native,checkpoint_sha256='b'*64),
        source('B',captures[1][1].to_dict()),features='P12')
    assert all(r['status']=='INCOMPATIBLE' and r['difference'] is None for r in invalid.to_dict()['metrics'])


@pytest.mark.parametrize('format', ['json','markdown','html'])
def test_native_P12_saved_comparison_uses_existing_CLI(captures,tmp_path,capsys,format):
    from hyso.cli import main
    result=analysis.compare_models(*pair(captures),features='P12')
    path=tmp_path/'comparison.json';result.to_json(path)
    expected=create_report(result,language='en')
    output=tmp_path/('report.'+format)
    assert main(['reports','render',str(path),'--language','en','--format',format,
                 '--output',str(output),'--json'])==0
    data=json.loads(capsys.readouterr().out)['result']
    assert data==expected.to_dict()
    assert output.read_text(encoding='utf8')==getattr(expected,'to_'+format)()


def test_strict_P12_schema_preserves_separate_missing_quality_gate(captures):
    result=analysis.compare_models(*pair(captures),features=('P12','P26'),context_policy='strict')
    data=result.to_dict()
    assert data['schema_version']==3 and data['context_policy']['id']=='strict_declared_context_v1'
    assert all(r['status']=='COMPARABLE' for r in data['metrics'] if r['feature_id']=='P12')
    assert all(r['status']=='INSUFFICIENT_EVIDENCE' and r['difference'] is None
               for r in data['metrics'] if r['feature_id']=='P26')
    assert analysis.ModelComparisonResult(data).to_dict()==data


@pytest.mark.parametrize('damage', ['source_label','invalid_source_hash'])
def test_caller_label_or_invalid_hash_is_not_native_source_identity(captures,damage):
    native=captures[0][1].to_dict()
    if damage=='source_label':
        native['identity']['source_result_id']='caller-label'
        native['raw_evidence']['source_result_id']='caller-label'
        native['raw_evidence']['p12_binding']['source_profile_id']='caller-label'
    else:
        native['raw_evidence']['source_sha256']='g'*64
    with pytest.raises(ValueError,match='invalid_native_evidence'):
        source('forged',native)
