"""Integration over retained evidence only; no model, capture or workload access."""
import pytest
import torch

from hyso import analysis, advisor, performance
from hyso.analysis._codec import digest
from hyso.evaluation import EvaluationResult
from hyso.reporting import create_report, PresentationReport
from test_model_comparison import bound, source, latency, no_workload


@pytest.fixture(autouse=True)
def no_live_sources(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('analyzer integration attempted model construction or a new measurement')
    monkeypatch.setattr(torch.nn.Module, '__init__', forbidden)
    for name in ('profile_operators', 'analyze_flops'):
        monkeypatch.setattr(performance, name, forbidden)


def comparison(*, features='P26', include_structure=True):
    return analysis.compare_models(source('A', 'P26', bound('A')),
        source('B', 'P26', bound('B')), features=features, include_structure=include_structure)


@pytest.mark.parametrize('missing', ['baseline', 'candidate', 'both'])
def test_missing_structure_snapshot_is_not_captured_or_inferred(missing):
    sources = []
    for side, label in (('baseline', 'A'), ('candidate', 'B')):
        snapshot = None if missing in (side, 'both') else EvaluationResult(bound(label)).structure_snapshot
        sources.append(source(label, 'P26', bound(label), structure=snapshot))
    before = [s.to_dict() for s in sources]
    # P26 is deliberately unselected: its bound snapshot must not be inferred.
    result = analysis.compare_models(*sources, features='P11', include_structure=True)
    assert result.to_dict()['structure']['status'] == 'INSUFFICIENT_EVIDENCE'
    assert result.structure_comparison is None
    assert [s.to_dict() for s in sources] == before


@pytest.mark.parametrize('language', ['tr', 'en'])
def test_advisor_comparison_explanations_reuse_eligible_evidence(language):
    result = comparison()
    before = result.to_json()
    recommendations = advisor.recommend(result)
    raw = recommendations.to_json()
    observations = recommendations.to_display_dict(language=language)['comparison_observations']
    assert {o['metric'] for o in observations} == {'test_accuracy', 'test_macro_f1', 'parameter_elements'}
    quality = next(o for o in observations if o['metric'] == 'test_accuracy')
    assert ('higher' if language == 'en' else 'daha yüksek') in quality['statement']
    for observation in observations:
        assert observation['evidence']['comparison_sha256'] == result.to_dict()['content_sha256']
        assert observation['manual_only'] and not observation['automatically_applied']
        assert observation['guidance']
    assert recommendations.to_json() == raw and result.to_json() == before


@pytest.mark.parametrize('problem', ['missing', 'incompatible', 'unsupported'])
def test_advisor_does_not_explain_unsupported_comparison_as_an_advantage(problem):
    native = latency('2', 7.)
    if problem == 'missing': native['provenance'].pop('comparison_capture')
    if problem == 'incompatible': native['provenance']['comparison_capture']['dataset_id'] = 'other'
    result = analysis.compare_models(source('A', 'P01', latency('1')), source('B', 'P01', native),
                                    features='P09' if problem == 'unsupported' else 'P01')
    view = advisor.recommend(result).to_display_dict(language='en')
    assert view['comparison_observations'] == []
    assert all(row['difference'] is None and row['reasons'] for row in result.to_dict()['metrics'])


def test_advisor_latency_observation_is_separate_from_quality_or_winner():
    result = analysis.compare_models(source('A', 'P01', latency('1')),
        source('B', 'P01', latency('2', 7.)), features='P01')
    observations = advisor.recommend(result).to_display_dict(language='en')['comparison_observations']
    assert len(observations) == 3
    assert all('lower observed' in o['statement'] for o in observations)
    assert result.to_dict()['claims']['global_winner'] is None


@pytest.mark.parametrize('language', ['tr', 'en'])
@pytest.mark.parametrize('level', ['beginner', 'technical', 'expert'])
def test_unified_comparison_formats_preserve_structure_values_and_advice(tmp_path, language, level):
    result = comparison(features=('P26', 'P09'))
    before = result.to_json()
    report = create_report(result, advisor.recommend(result), language=language, level=level)
    data = report.to_dict()
    section = data['sources'][0]
    assert data['presentation_only'] and not data['authoritative_evidence']
    assert section['raw'] == result.to_dict()
    parameters = next(m for m in section['metrics'] if m['key'] == 'parameter_elements')
    assert (parameters['baseline'], parameters['candidate'], parameters['value']) == (18, 10, -8)
    assert any(m['key'] == 'trainable_elements' for m in section['metrics'])
    unsupported = next(m for m in section['metrics'] if m['feature_id'] == 'P09')
    assert unsupported['value'] is None and unsupported['baseline'] is None and unsupported['candidate'] is None
    assert unsupported['status'] == 'NOT_SUPPORTED_IN_COMPARISON'
    tables = section['tables']
    assert any(t['title'] == ('State tensor metadata farkları' if language == 'tr' else 'State tensor metadata differences')
               and any(row[1] == 'shape' for row in t['rows']) for t in tables)
    snapshot_hash = section['raw']['structure']['comparison']['baseline']['content_sha256']
    path = tmp_path / 'report.json'
    report.to_json(path)
    reopened = PresentationReport.from_json(path)
    assert reopened.to_dict() == data
    markdown, html = reopened.to_markdown(), reopened.to_html()
    for text in (markdown, html):
        assert 'B−A' in text and snapshot_hash in text
        assert 'NOT_SUPPORTED_IN_COMPARISON' in text
        assert 'torch.float32' in text
        assert ('Model B has fewer' if language == 'en' else 'daha az parametre') in text
        assert ('manual' if language == 'en' else 'kendiniz') in text
    assert result.to_json() == before


@pytest.mark.parametrize('language', ['tr', 'en'])
def test_retained_P12_values_are_reported_without_comparison_or_hardware_claim(language):
    from test_advisor_p12_observation import saved, report as analysis_report
    sources = []
    for label, kind in (('A', 'sequential'), ('B', 'graph')):
        record = saved(kind, scope=analysis.EvidenceScope(model_id=label))
        sources.append(analysis.ModelComparisonSource(label, analysis_report([record])))
    result = analysis.compare_models(*sources, features='P12')
    before = result.to_json()
    report = create_report(result, language=language)
    section = report.to_dict()['sources'][0]
    assert section['metrics'][0]['value'] is None
    assert section['metrics'][0]['status'] == 'INSUFFICIENT_EVIDENCE'
    observations = section['p12_observations']
    assert [p['observation']['metrics']['macs_per_batch']['value'] for p in observations] == [25856, 17664]
    assert all(not p['observation']['claims']['hardware_performance'] for p in observations)
    assert section['advisor']['comparison_observations'] == []
    for text in (report.to_markdown(), report.to_html()):
        assert '25856' in text and '17664' in text and 'INSUFFICIENT_EVIDENCE' in text
    assert result.to_json() == before


def test_old_saved_comparison_presentation_still_renders_A_and_B():
    data = create_report(comparison(), language='en').to_dict()
    section = data['sources'][0]
    for key in ('tables', 'p12_observations'):
        section.pop(key)
    section['advisor'].pop('comparison_observations')
    for metric in section['metrics']:
        metric.pop('baseline')
        metric.pop('candidate')
    data['content_sha256'] = digest({k: v for k, v in data.items() if k != 'content_sha256'})
    report = PresentationReport(data)
    before = report.to_json()
    assert '| 18 | 10 | -8 |' in report.to_markdown()
    assert '<td>18</td><td>10</td><td>-8</td>' in report.to_html()
    assert report.to_json() == before


def test_new_comparison_tables_and_advice_escape_model_labels():
    result = analysis.compare_models(source('A<script>|', 'P26', bound('A')),
        source('B<script>|', 'P26', bound('B')), features='P26', include_structure=True)
    report = create_report(result, language='en')
    for text in (report.to_markdown(), report.to_html()):
        assert '<script>' not in text and '&lt;script&gt;' in text


@pytest.mark.parametrize('missing', ['trace','device','scope'])
def test_partial_legacy_P12_metadata_stays_insufficient(missing):
    from test_advisor_p12_observation import saved
    sources=[]
    for label,kind in (('A','sequential'),('B','graph')):
        native=saved(kind).to_dict()['native_payload']
        if missing=='trace':native['raw_evidence']['trace']=None
        if missing=='device':native['provenance']['environment'].pop('device')
        if missing=='scope':native['protocol'].pop('scope')
        sources.append(source(label,'P12',native))
    result=analysis.compare_models(*sources,features='P12')
    assert all(r['status']=='INSUFFICIENT_EVIDENCE' and r['difference'] is None
               for r in result.to_dict()['metrics'])
