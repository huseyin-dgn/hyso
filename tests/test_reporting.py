"""Deterministic presentation from retained fixtures, with live workloads forbidden."""
import copy
import json
from html.parser import HTMLParser
from pathlib import Path
import re
import runpy
import typing

import pytest
import torch

from hyso import analysis, reporting, TrainingConfig, ExperimentManager
from hyso.reporting import PresentationReport, ReportError, create_report, feature_explanations, report_experiment
from hyso.training import TrainingResult
from hyso.analysis._codec import canonical, digest
from test_model_comparison import source, bound, latency
from test_experiment_workflow import training_run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_workloads(monkeypatch):
    from hyso import training, benchmark, performance, quality, evaluation
    def forbidden(*args, **kwargs):
        pytest.fail('Reporting attempted a live workload, device query or checkpoint load')
    for target, name in ((training.Trainer, 'fit'), (training.TrainingSession, 'fit'),
                         (benchmark.BenchmarkRunner, 'run'), (analysis, 'execute_analysis'),
                         (torch.nn.Module, '_call_impl'), (torch, 'load'),
                         (quality, 'compare_performance_quality'), (evaluation, 'evaluate_classification')):
        monkeypatch.setattr(target, name, forbidden)
    for name in ('init', '_lazy_init', 'is_available', 'device_count', 'current_device'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    for name in ('measure_throughput', 'measure_memory', 'measure_footprint'):
        monkeypatch.setattr(performance, name, forbidden)


def context():
    return analysis.AnalysisContext(workload_id='retained-fixture', task='classification', model={}, dataset={})


def training():
    raw = (ROOT / 'examples/reporting/fixtures/training-result-v1.json').read_text(encoding='utf-8')
    return TrainingResult(raw)


def native_report(features=('P01', 'P02', 'P10', 'P11')):
    payloads = json.loads((ROOT / 'tests/fixtures/advisor-native-v1.json').read_text(encoding='utf-8'))['payloads']
    records = [analysis.EvidenceRecord.from_native(payloads[feature], feature_id=feature,
               scope=analysis.EvidenceScope(model_id='retained-A')) for feature in features]
    return analysis.EvidenceIndex(records).analyze(context=context())


def test_all_35_bilingual_explanations_and_existing_catalog_scopes():
    for language in ('tr', 'en'):
        catalog = feature_explanations(language=language)
        assert [r['feature_id'] for r in catalog] == [f'P{i:02d}' for i in range(1, 36)]
        for row in catalog:
            assert set(row['explanations']) == {'beginner', 'technical', 'expert'}
            assert all(row[key] for key in ('definition', 'purpose', 'limits', 'units'))
            guide = row['guide_url'].split('/docs/guides/')[1]
            assert (ROOT / 'docs/guides' / guide).is_file()
        assert catalog[8]['executor_modes'] == []
        assert 'trusted_cpu' in catalog[9]['executor_modes']
    assert feature_explanations(language='tr')[9]['name'] != feature_explanations(language='en')[9]['name']
    with pytest.raises(ValueError):
        feature_explanations(language='fr')


@pytest.mark.parametrize('language', ['tr', 'en'])
@pytest.mark.parametrize('level', ['beginner', 'technical', 'expert'])
def test_training_levels_exports_and_exact_source_preservation(tmp_path, language, level):
    result = training()
    before = result.to_dict()
    report = create_report(result, language=language, level=level)
    data = report.to_dict()
    source_data = data['sources'][0]
    assert source_data['raw'] == before == result.to_dict()
    assert {c['domain'] for c in source_data['charts']} == {'training_loss', 'training_accuracy', 'ux_observer_interval'}
    assert source_data['features'] == []
    assert all(c['unit'] == 's' for c in source_data['charts'] if c['domain'] == 'ux_observer_interval')
    assert source_data['metrics'][2]['unit'] == 'percent'
    assert source_data['metrics'][2]['value'] == 80.
    path = tmp_path / 'report.json'
    report.to_json(path)
    assert PresentationReport.from_json(path).to_json() == report.to_json()
    for suffix, writer in [('md', report.to_markdown), ('html', report.to_html)]:
        target = tmp_path / ('report.' + suffix)
        raw = writer(target)
        assert target.read_text(encoding='utf-8') == raw
        assert 'Synthetic reporting fixture' in raw
        with pytest.raises(FileExistsError):
            writer(target)
    with pytest.raises(FileExistsError):
        report.to_json(path)
    assert create_report(result, language=language, level=level).to_json() == report.to_json()


def test_missing_validation_is_a_curve_gap_and_missing_data_has_no_charts():
    data = training().to_dict()
    data['history']['val_loss'] = [.95, None, .65]
    data['history']['val_acc'] = [None, None, None]
    data['epoch_events'] = []
    report = create_report(TrainingResult(canonical(data)))
    charts = report.to_dict()['sources'][0]['charts']
    assert charts[0]['series'][1]['y'] == [.95, None, .65]
    assert len(charts[1]['series']) == 1
    assert 'ux_observer_interval' not in {c['domain'] for c in charts}
    assert report.to_html().count('<svg') == 2
    empty = analysis.EvidenceIndex([
        analysis.EvidenceRecord.unavailable('P09', status='unsupported', reason='no retained CUDA capture'),
        analysis.EvidenceRecord.unavailable('P19', status='not_run', reason='training not run'),
        analysis.EvidenceRecord.unavailable('P31', status='invalid', reason='incomplete duration')]).analyze(context=context())
    display = create_report(empty).to_dict()['sources'][0]
    assert display['metrics'] == [] and display['charts'] == []
    statuses = {r['feature_id']: r['status'] for r in display['features']}
    assert statuses['P09'] == 'unsupported' and statuses['P19'] == 'not_run'
    assert statuses['P31'] == 'invalid' and statuses['P01'] == 'missing'
    assert '<svg' not in create_report(empty).to_html()


def test_native_metric_values_latency_throughput_memory_and_advisor_reuse():
    original = native_report()
    report = create_report(original, language='en')
    section = report.to_dict()['sources'][0]
    assert section['advisor']['format'] == 'hyso.advisor.display'
    assert section['raw'] == original.to_dict()
    hist, = [c for c in section['charts'] if c['kind'] == 'histogram']
    record = next(r for r in original.to_dict()['index']['records'] if r['feature_id'] == 'P01')
    assert hist['series'][0]['y'] == record['native_payload']['raw_evidence']['latency_samples_ms'][0]
    domains = {c['domain'] for c in section['charts']}
    assert {'throughput', 'process_rss', 'model_logical_bytes', 'serialized_artifact_bytes'} <= domains
    assert 'cuda_allocator' not in domains
    for original_record in original.to_dict()['index']['records']:
        retained = [m for m in section['metrics'] if m['record_id'] == original_record['record_id']]
        assert [(m['key'], m['value'], m['unit'], m['status']) for m in retained] == [
            (m['path'], m['value'], m['unit'], m['status']) for m in original_record['metrics']]
    assert all(m['recommendation_links'] for m in section['metrics'])
    assert '<svg' in report.to_html()


def test_tail_ineligibility_keeps_raw_samples_without_inventing_p95():
    retained = source('A', 'P01', latency('1', count=20))
    report = analysis.UnifiedAnalysisReport(retained.to_dict()['report'])
    section = create_report(report).to_dict()['sources'][0]
    assert all(m['value'] is None for m in section['metrics'])
    assert sum(len(s['y']) for s in section['charts'][0]['series']) == 20


def test_comparison_charts_follow_existing_eligibility_and_arithmetic():
    result = analysis.compare_models(source('A', 'P26', bound('A')), source('B', 'P26', bound('B')), features=('P26', 'P09'), include_structure=True)
    section = create_report(result, language='en').to_dict()['sources'][0]
    accuracy = next(m for m in section['metrics'] if m['key'] == 'test_accuracy')
    assert accuracy['value'] == .25 and accuracy['percentage_point_change'] == 25.
    assert accuracy['percent_change'] == pytest.approx(100 / 3)
    assert section['raw']['claims']['global_winner'] is None
    count = next(m for m in section['metrics'] if m['key'] == 'parameter_elements')
    assert count['value'] == -8 and count['domain'] == 'registered_structure'
    assert all(c['domain'] != 'cuda_allocator' for c in section['charts'])
    left = source('A', 'P01', latency('1'))
    changed = latency('2')
    changed['provenance']['comparison_capture']['hardware_id'] = '9' * 64
    incompatible = analysis.compare_models(left, source('B', 'P01', changed), features='P01')
    blocked = create_report(incompatible).to_dict()['sources'][0]
    assert blocked['charts'] == []
    assert all(m['value'] is None and m['status'] == 'INCOMPATIBLE' for m in blocked['metrics'])


def test_index_context_and_direct_advisor_input_preserve_existing_api(tmp_path):
    report = native_report(('P11',))
    index = analysis.EvidenceIndex(analysis.EvidenceRecord(r) for r in report.to_dict()['index']['records'])
    with pytest.raises(ReportError, match='analysis_context_required'):
        create_report(index)
    wrapper = create_report(index, context=context())
    assert wrapper.to_dict()['sources'][0]['raw'] == index.analyze(context=context()).to_dict()
    from hyso import advisor
    advice = advisor.advise(report)
    assert create_report(advice).to_dict()['sources'][0]['advisor']['format'] == 'hyso.advisor.display'
    report.to_json(tmp_path / 'native.json')
    assert analysis.UnifiedAnalysisReport.from_json(tmp_path / 'native.json').to_dict() == report.to_dict()
    with pytest.raises(analysis.EvidenceError):
        analysis.UnifiedAnalysisReport(wrapper.to_dict())


def test_experiment_reads_missing_artifacts_and_original_states_without_writes(tmp_path):
    root = tmp_path / 'store'
    manager = ExperimentManager(root)
    manager.close()
    ExperimentManager.migrate_store(root)
    manager = ExperimentManager(root)
    try:
        run, result = training_run(manager)
        artifact = run.record_result(result)
        run.finish()
        experiment_id = run.to_dict()['experiment_id']
        before = manager._core_snapshot(manager._db)
        presented = report_experiment(manager, experiment_id)
        assert presented.to_dict()['experiment']['status'] == 'completed'
        assert len(presented.to_dict()['sources']) == 1
        assert manager._core_snapshot(manager._db) == before
        assert manager._db.execute('SELECT count(*) FROM artifact_checks').fetchone()[0] == 0
        (root / artifact['managed_relative_path']).rename(tmp_path / 'moved-result.json')
        missing = report_experiment(manager, experiment_id)
        assert missing.to_dict()['sources'] == []
        assert missing.to_dict()['experiment']['status'] == 'completed'
        item = missing.to_dict()['experiment']['runs'][0]['artifacts'][0]
        assert item['result_status'] == 'missing'
        assert 'missing' in missing.to_html()
        assert manager._core_snapshot(manager._db) == before
    finally:
        manager.close()


def test_html_escaping_no_remote_dependencies_and_determinism():
    report = create_report(training(), title="<script>alert('title')</script> $USD$", language='en')
    first = report.to_html()
    assert first == report.to_html()
    assert '<script>' not in first and '&lt;script&gt;' in first
    class Document(HTMLParser):
        def handle_starttag(self, tag, attrs):
            assert tag not in ('script', 'iframe', 'object', 'embed', 'link')
            for key, value in attrs:
                assert not key.startswith('on')
                if key in ('href', 'src'):
                    assert value.startswith('#')
    Document().feed(first)


def test_presentation_digest_versions_options_and_no_mutable_alias(tmp_path):
    report = create_report(training())
    data = report.to_dict()
    data['sources'].clear()
    assert report.to_dict()['sources']
    with pytest.raises(ReportError, match='integrity'):
        PresentationReport(data)
    data = report.to_dict()
    data['schema_version'] = 2
    with pytest.raises(ReportError):
        PresentationReport(data)
    for kwargs in (dict(language='fr'), dict(level='unknown')):
        with pytest.raises(ReportError):
            create_report(training(), **kwargs)
    with pytest.raises(ReportError):
        create_report({'untyped': True})
    assert typing.get_type_hints(create_report)['return'] is PresentationReport
    assert typing.get_type_hints(report_experiment)['manager'] is ExperimentManager


@pytest.mark.parametrize('language,level', [('tr', 'beginner'), ('en', 'expert')])
def test_retained_example_script_outputs_three_formats_without_workloads(tmp_path, language, level):
    example = runpy.run_path(str(ROOT / 'examples/reporting/retained_results.py'))
    target = tmp_path / 'example'
    example['main'](['--output', str(target), '--language', language, '--level', level])
    assert {p.name for p in target.iterdir()} == {'report.json', 'report.md', 'report.html'}
    report = PresentationReport.from_json(target / 'report.json')
    assert report.to_dict()['language'] == language
    assert (target / 'report.html').read_text(encoding='utf-8').count('<svg') == 3


def test_history_example_existing_schema1_store(tmp_path):
    root = tmp_path / 'store'
    manager = ExperimentManager(root)
    study = manager.create_experiment('existing')
    run = manager.create_run(study['id'], 'evaluation', {})
    run.fail('retained failure')
    before = manager._core_snapshot(manager._db)
    manager.close()
    example = runpy.run_path(str(ROOT / 'examples/reporting/experiment_history.py'))
    target = tmp_path / 'history'
    example['main'](['--store', str(root), '--experiment', study['id'], '--output', str(target)])
    report = PresentationReport.from_json(target / 'report.json')
    assert report.to_dict()['experiment']['status'] == 'failed'
    manager = ExperimentManager(root)
    assert manager._schema_version == 1 and manager._core_snapshot(manager._db) == before
    manager.close()


def test_native_p19_timings_and_ux_intervals_have_separate_domains():
    data = json.loads((ROOT / 'tests/fixtures/training-p19-schema1.json').read_text(encoding='utf-8'))
    record = analysis.EvidenceRecord.from_native(data, feature_id='P19')
    report = create_report(training(), analysis.EvidenceIndex([record]), context=context())
    domains = {c['domain'] for s in report.to_dict()['sources'] for c in s['charts']}
    assert {'training_timing', 'ux_observer_interval'} <= domains
    timings = report.to_dict()['sources'][1]['metrics']
    assert all(m['unit'] == 's' for m in timings)


def test_executor_failure_and_unconnected_statuses_remain_explicit():
    from hyso.analysis.execution import STATUSES
    plan = analysis.plan_analysis(analysis.ModelContext('A'), features=('P11', 'P19', 'P09'))
    dry = analysis.preflight_analysis(plan)
    rows = [dict(model_id='A', feature_id=feature, status=status, reasons=['retained fixture reason'],
                 evidence_record=None, auxiliary_evidence={})
            for feature, status in [('P11', 'FAILED'), ('P19', 'NOT_IMPLEMENTED_IN_EXECUTOR'), ('P09', 'UNSUPPORTED')]]
    result = analysis.AnalysisExecutionResult(canonical(dict(format='hyso.analysis_execution', schema_version=1,
        dry_run=dry.to_dict(), features=rows, authorized_features=['P11'],
        status_counts={s: sum(r['status'] == s for r in rows) for s in STATUSES})))
    with pytest.raises(ReportError):
        create_report(result)
    view = create_report(result, context=context())
    section = view.to_dict()['sources'][0]
    assert section['execution'] == result.to_dict()
    assert section['charts'] == [] and section['metrics'] == []
    assert all(row['status'] in view.to_html() for row in rows)


def test_reporting_guide_and_saved_input_example(tmp_path):
    report = native_report(('P11',))
    comparison = analysis.compare_models(source('A', 'P26', bound('A')), source('B', 'P26', bound('B')), features='P26')
    guide = ROOT / 'docs/guides/result-reporting.md'
    namespace = {}
    for block in re.findall(r'^```python\n(.*?)^```', guide.read_text(encoding='utf-8'), re.M | re.S):
        exec(compile(block, str(guide), 'exec'), namespace)
    result = namespace['export_results'](tmp_path / 'guide', training(), report, comparison)
    assert len(result.to_dict()['sources']) == 3
    report.to_json(tmp_path / 'analysis.json')
    comparison.to_json(tmp_path / 'comparison.json')
    example = runpy.run_path(str(ROOT / 'examples/reporting/retained_results.py'))
    example['main'](['--output', str(tmp_path / 'saved'), '--input', str(tmp_path / 'analysis.json'),
                     '--input', str(tmp_path / 'comparison.json')])
    reopened = PresentationReport.from_json(tmp_path / 'saved/report.json')
    assert {s['kind'] for s in reopened.to_dict()['sources']} == {'analysis', 'comparison'}
    readme = ROOT / 'README.md'
    block, = [code for code in re.findall(r'^```python\n(.*?)^```', readme.read_text(encoding='utf-8'), re.M | re.S)
               if 'def export_saved_results(' in code]
    exec(compile(block, str(readme), 'exec'), namespace)
    assert namespace['export_saved_results'](tmp_path / 'readme', training()).to_dict()['sources']


def test_presentation_links_only_target_local_suggestions():
    data = create_report(training()).to_dict()
    data['sources'][0]['metrics'][0]['recommendation_links'] = ['javascript:alert(1)']
    data.pop('content_sha256')
    data['content_sha256'] = digest(data)
    with pytest.raises(ReportError, match='local_recommendation_link'):
        PresentationReport(data).to_html()


def test_full_experiment_pagination_and_scalar_table(tmp_path):
    manager = ExperimentManager(tmp_path / 'paged')
    try:
        study = manager.create_experiment('paged')
        run = manager.create_run(study['id'], 'training', {})
        run.start()
        run.log_metrics([dict(name='retained_scalar', value=float(i), unit='mean_loss', source='caller',
                             step=i, step_kind='observation') for i in range(101)])
        run.finish()
        for _ in range(100):
            manager.create_run(study['id'], 'evaluation', {})
        before = manager._core_snapshot(manager._db)
        report = report_experiment(manager, study['id'])
        experiment = report.to_dict()['experiment']
        assert len(experiment['runs']) == 101
        assert len(experiment['runs'][0]['metrics']) == 101
        assert 'retained_scalar' in report.to_html() and 'retained_scalar' in report.to_markdown()
        assert 'Bu bölüm için sayısal veri sağlanmadı.' not in report.to_html()
        assert report.to_dict()['sources'] == []
        assert manager._core_snapshot(manager._db) == before
    finally:
        manager.close()
