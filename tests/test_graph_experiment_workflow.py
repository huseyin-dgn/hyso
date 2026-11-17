"""Saved Graph workflow integration; no model construction, capture or workload."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from hyso import GraphModelBuilder, ModelBuilder, ExperimentManager, analysis, advisor, performance
from hyso.architecture_comparison import compare_architectures
from hyso.analysis.graph_evidence import GraphEvaluationResult, GraphComparisonResult, compare_graph_evaluations
from hyso.analysis._codec import digest
from hyso.reporting import create_report, report_experiment, PresentationReport
from test_experiment_workflow import manager, no_workloads


FIXTURE = Path(__file__).parent / 'fixtures/graph-cli-comparison-v1.json'


@pytest.fixture(autouse=True)
def retained_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('saved Graph workflow attempted construction, profiling or a new measurement')
    for target, name in ((torch.nn.Module, '__init__'), (GraphModelBuilder, 'build'),
                         (ModelBuilder, 'build'), (performance, 'profile_operators'),
                         (performance.profiling, '_execute_profile')):
        monkeypatch.setattr(target, name, forbidden)


@pytest.fixture
def graphs():
    data = json.loads(FIXTURE.read_text(encoding='utf8'))
    return [GraphEvaluationResult(data['sources'][side]) for side in ('baseline', 'candidate')]


def attach(run, result, kind, path, ownership='copy'):
    result.to_json(path)
    return run.add_artifact(path, kind=kind, ownership=ownership, retain_locator=True,
                            metadata={'association': 'caller_declared'})


def test_graph_workflow_architecture_is_retained_metadata(manager, tmp_path, graphs):
    contract = graphs[0].to_dict()['source_receipt']['contract']
    config = contract['configuration']
    builder = GraphModelBuilder.from_config(config)
    experiment = manager.create_experiment('Graph metadata')
    run = manager.create_workflow_run(experiment['id'], 'model', model_ids=['A'],
        architectures={'A': builder}, protocol={'adapter_output': contract['adapter_output'],
                                                'state_dict_prefix': 'graph.'})
    snapshot = run.to_dict()['config']
    assert snapshot['architectures']['A'] == config
    assert snapshot['context']['model_ids'] == ['A']
    assert snapshot['context']['association'] == 'caller_declared'
    assert snapshot['context']['protocol'] == {'adapter_output': contract['adapter_output'], 'state_dict_prefix': 'graph.'}
    run.start()
    path = tmp_path / 'architecture.json';path.write_text(json.dumps(config), encoding='utf8')
    artifact = run.add_artifact(path, kind='graph_architecture', ownership='copy')
    assert manager.open_result(artifact['id'])['status'] == 'unsupported'
    assert manager.verify_artifact(artifact['id'])['status'] == 'verified'
    run.finish()


@pytest.mark.parametrize('ownership', ['copy', 'reference'])
@pytest.mark.parametrize('kind', ['graph_evaluation_association', 'graph_evidence_comparison',
                                 'analysis_report', 'presentation_report'])
def test_graph_generic_artifacts_reopen_with_existing_readers(manager, tmp_path, graphs, ownership, kind):
    values = {
        'graph_evaluation_association': graphs[0],
        'graph_evidence_comparison': GraphComparisonResult.from_json(FIXTURE),
        'analysis_report': analysis.UnifiedAnalysisReport(graphs[0].analysis_source('A').to_dict()['report']),
        'presentation_report': create_report(graphs[0], language='en'),
    }
    result = values[kind];before = result.to_dict()
    experiment = manager.create_experiment('Saved Graph result')
    run = manager.create_run(experiment['id'], 'evaluation', {'model_id': 'A', 'association': 'caller_declared'})
    run.start()
    path = tmp_path / 'result.json'
    artifact = attach(run, result, kind, path, ownership)
    run.finish()
    opened = manager.open_result(artifact['id'])
    assert opened['status'] == 'verified' and type(opened['result']) is type(result)
    assert opened['result'].to_dict() == before
    assert artifact['expected_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    if kind == 'graph_evaluation_association':
        data = opened['result'].to_dict()
        assert data['verification']['graph_v2'] == 'unverified'
        assert data['source_receipt'] == before['source_receipt']
        assert data['source_receipt']['model_state_sha256'] == data['evaluation']['source']['model_state_sha256']
        assert data['evaluation']['producer_evidence']['checkpoint']['status'] == 'missing'
        assert data['source_receipt']['contract']['adapter_output'] == 'logits'
    if kind == 'presentation_report':
        assert opened['result'].to_dict()['presentation_only'] is True
        assert opened['result'].to_dict()['authoritative_evidence'] is False
    detail = manager.get_run_details(run.id)
    assert detail['run']['experiment_id'] == experiment['id']
    assert detail['artifacts'][0]['artifact']['id'] == artifact['id']


@pytest.mark.parametrize('damage', ['schema', 'state', 'kind'])
def test_verified_artifact_bytes_do_not_override_Graph_contracts(manager, tmp_path, graphs, damage):
    data = graphs[0].to_dict();data.pop('verification');data.pop('relation')
    if damage == 'schema':data['schema_version'] = 999
    if damage == 'state':data['source_receipt']['model_state_sha256'] = 'b' * 64
    data['content_sha256'] = digest({k: v for k, v in data.items() if k != 'content_sha256'})
    path = tmp_path / 'invalid.json';path.write_text(json.dumps(data), encoding='utf8')
    experiment = manager.create_experiment('Graph contract rejection')
    run = manager.create_run(experiment['id'], 'evaluation', {})
    run.start()
    kind = 'graph_evidence_comparison' if damage == 'kind' else 'graph_evaluation_association'
    artifact = run.add_artifact(path, kind=kind, ownership='copy')
    assert manager.verify_artifact(artifact['id'])['status'] == 'verified'
    opened = manager.open_result(artifact['id'])
    assert opened['status'] == 'invalid_result' and opened['result'] is None
    run.fail('invalid retained statement')


def stored_pair(manager, tmp_path, graphs):
    experiment = manager.create_experiment('Retained Graph experiment')
    artifacts, runs = [], []
    for label, graph in zip(('A', 'B'), graphs):
        config = graph.to_dict()['source_receipt']['contract']['configuration']
        run = manager.create_workflow_run(experiment['id'], 'model', model_ids=[label],
            architectures={label: GraphModelBuilder.from_config(config)},
            protocol={'adapter_output': 'logits', 'state_dict_prefix': 'graph.'})
        run.start()
        artifacts.append(attach(run, graph, 'graph_evaluation_association', tmp_path / (label + '.json')))
        run.log_metric(name='test_accuracy', value=graph.evaluation.to_dict()['metrics']['accuracy'],
                       unit='fraction', source='retained_P26:' + graph.to_dict()['record']['record_id'],
                       step=0, step_kind='observation')
        run.finish()
        runs.append(run.id)
    return experiment, artifacts, runs


def test_saved_graph_results_reuse_existing_comparison_and_architecture_APIs(manager, tmp_path, graphs):
    experiment, artifacts, run_ids = stored_pair(manager, tmp_path, graphs)
    before = manager._core_snapshot(manager._db)
    loaded = [manager.open_result(a['id'])['result'] for a in artifacts]
    configs = [g.to_dict()['source_receipt']['contract']['configuration'] for g in loaded]
    architecture = compare_architectures(configs, names=['A', 'B'])
    assert all(c['valid'] and c['node_count'] > 0 for c in architecture['columns'])
    assert configs[0] != configs[1]
    compared = analysis.compare_models(*[g.analysis_source(label) for label, g in zip(('A', 'B'), loaded)],
        features=('P26', 'P12', 'P11'), include_structure=True, context_policy='strict')
    data = compared.to_dict()
    assert all(row['status'] == 'COMPARABLE' for row in data['metrics'] if row['feature_id'] == 'P26')
    assert all(row['status'] == 'INSUFFICIENT_EVIDENCE' and row['difference'] is None
               for row in data['metrics'] if row['feature_id'] in ('P12', 'P11'))
    assert data['structure']['status'] == 'INSUFFICIENT_EVIDENCE'
    assert manager._core_snapshot(manager._db) == before
    run = manager.create_workflow_run(experiment['id'], 'comparison', model_ids=['A', 'B'], comparison=compared)
    run.start()
    saved = run.record_result(compared, input_artifact_ids=[a['id'] for a in artifacts])
    assert run.record_result(compared, input_artifact_ids=[a['id'] for a in artifacts])['id'] == saved['id']
    run.finish()
    assert manager.open_result(saved['id'])['result'].to_json() == compared.to_json()
    assert {r['run']['id'] for r in [manager.get_run_details(i) for i in run_ids]} == set(run_ids)
    root = manager.root_dir
    manager.close()
    reopened = ExperimentManager(root)
    try:
        assert reopened.open_result(saved['id'])['result'].to_json() == compared.to_json()
        assert all(reopened.open_result(a['id'])['result'].to_dict()['verification']['graph_v2'] == 'unverified'
                   for a in artifacts)
    finally:
        reopened.close()


@pytest.mark.parametrize('context_value,status', [('different', 'INCOMPATIBLE'), (None, 'INSUFFICIENT_EVIDENCE')])
def test_saved_graph_comparison_preserves_declared_context_gates(manager, tmp_path, graphs, context_value, status):
    data = graphs[1].to_dict();data.pop('verification');data.pop('relation')
    data['context']['preprocessing_id'] = context_value
    data['content_sha256'] = digest({k: v for k, v in data.items() if k != 'content_sha256'})
    values = [graphs[0], GraphEvaluationResult(data)]
    experiment, artifacts, _ = stored_pair(manager, tmp_path, values)
    reopened = [manager.open_result(a['id'])['result'] for a in artifacts]
    result = compare_graph_evaluations(*reopened, context_policy='strict')
    assert result.to_dict()['status'] == status
    assert all(m['difference'] is None for m in result.to_dict()['metrics'])
    run = manager.create_run(experiment['id'], 'benchmark', {'association': 'caller_declared'})
    run.start()
    saved = attach(run, result, 'graph_evidence_comparison', tmp_path / 'comparison.json')
    run.finish()
    assert manager.open_result(saved['id'])['result'].to_dict() == result.to_dict()


@pytest.mark.parametrize('language', ['tr', 'en'])
def test_experiment_Graph_reporting_reuses_results_and_Advisor(manager, tmp_path, graphs, language):
    experiment, artifacts, _ = stored_pair(manager, tmp_path, graphs)
    opened = [manager.open_result(a['id'])['result'] for a in artifacts]
    comparison = analysis.compare_models(*[g.analysis_source(label) for label, g in zip(('A', 'B'), opened)],
        features=('P26', 'P12'), include_structure=True, context_policy='strict')
    run = manager.create_workflow_run(experiment['id'], 'comparison', model_ids=['A', 'B'], comparison=comparison)
    run.start()
    recorded = run.record_result(comparison, input_artifact_ids=[a['id'] for a in artifacts])
    graph_comparison = compare_graph_evaluations(*opened, context_policy='strict')
    graph_artifact = attach(run, graph_comparison, 'graph_evidence_comparison', tmp_path / 'graph-comparison.json')
    analysis_report = analysis.UnifiedAnalysisReport(opened[0].analysis_source('A').to_dict()['report'])
    analysis_artifact = attach(run, analysis_report, 'analysis_report', tmp_path / 'analysis.json')
    run.finish()
    restored_comparison = manager.open_result(recorded['id'])['result']
    recommendations = advisor.recommend(restored_comparison)
    observations = recommendations.to_display_dict(language=language)['comparison_observations']
    assert {o['metric'] for o in observations} == {'test_accuracy', 'test_macro_f1'}
    assert all(o['manual_only'] and not o['automatically_applied'] for o in observations)
    direct = create_report(restored_comparison, recommendations, language=language)
    assert direct.to_dict()['sources'][0]['raw'] == comparison.to_dict()
    before = manager._core_snapshot(manager._db)
    report = report_experiment(manager, experiment['id'], language=language, level='expert')
    assert manager._core_snapshot(manager._db) == before
    data = report.to_dict()
    assert data['presentation_only'] is True and data['authoritative_evidence'] is False
    assert data['experiment']['experiment']['id'] == experiment['id']
    assert len(data['sources']) == 5
    linked = {identity for section in data['sources'] for identity in section['artifact_ids']}
    assert linked == {a['id'] for a in artifacts} | {recorded['id'], graph_artifact['id'], analysis_artifact['id']}
    graph_sections = [s for s in data['sources'] if s['kind'] == 'graph_evidence']
    assert len(graph_sections) == 3
    assert all(s['graph_evidence']['graph_v2'] == 'unverified' for s in graph_sections[:2])
    typed_section = next(s for s in data['sources'] if s['kind'] == 'comparison')
    assert all(m['value'] is None and m['status'] == 'INSUFFICIENT_EVIDENCE'
               for m in typed_section['metrics'] if m['feature_id'] == 'P12')
    assert typed_section['advisor']['comparison_observations'] == observations
    for text in (report.to_json(), report.to_markdown(), report.to_html()):
        assert experiment['id'] in text
        assert 'graph.' in text and 'adapter_output' in text
        assert 'INSUFFICIENT_EVIDENCE' in text and 'unverified' in text
        assert ('manual' if language == 'en' else 'kendiniz') in text
    cache = manager.create_run(experiment['id'], 'benchmark', {'scope': 'saved_presentation_only'})
    cache.start()
    saved = attach(cache, report, 'presentation_report', tmp_path / 'presentation.json')
    cache.finish()
    reopened = manager.open_result(saved['id'])['result']
    assert type(reopened) is PresentationReport and reopened.to_json() == report.to_json()
    # Cached presentations are snapshots, never recursive report inputs.
    assert len(report_experiment(manager, experiment['id'], language=language).to_dict()['sources']) == 5


def test_experiment_report_keeps_invalid_Graph_artifacts_explicit(manager, tmp_path, graphs):
    experiment, artifacts, _ = stored_pair(manager, tmp_path, graphs)
    run = manager.create_run(experiment['id'], 'evaluation', {})
    run.start()
    path = tmp_path / 'invalid-version.json'
    data = graphs[0].to_dict();data['schema_version'] = 999
    path.write_text(json.dumps(data), encoding='utf8')
    invalid = run.add_artifact(path, kind='graph_evaluation_association', ownership='copy')
    run.finish()
    report = report_experiment(manager, experiment['id'], language='en').to_dict()
    rows = [a for detail in report['experiment']['runs'] for a in detail['artifacts']]
    row = next(a for a in rows if a['artifact']['id'] == invalid['id'])
    assert row['availability'] == 'verified' and row['result_status'] == 'invalid_result'
    assert len(report['sources']) == 2


def test_artifact_integrity_cannot_restore_serialized_Graph_trust(manager, tmp_path, graphs):
    data = graphs[0].to_dict()
    data['verification']['graph_v2'] = 'verified'
    data['relation']['status'] = 'verified'
    path = tmp_path / 'claimed-trust.json';path.write_text(json.dumps(data), encoding='utf8')
    experiment = manager.create_experiment('Imported trust stays unverified')
    run = manager.create_run(experiment['id'], 'evaluation', {})
    run.start()
    artifact = run.add_artifact(path, kind='graph_evaluation_association', ownership='copy')
    run.finish()
    opened = manager.open_result(artifact['id'])
    assert opened['status'] == 'verified'
    assert opened['result'].to_dict()['verification']['graph_v2'] == 'unverified'
    assert opened['result'].to_dict()['relation']['status'] != 'verified'


@pytest.mark.parametrize('field,value', [('presentation_only', False), ('authoritative_evidence', True)])
def test_saved_presentation_artifact_cannot_elevate_authority(manager, tmp_path, graphs, field, value):
    data = create_report(graphs[0]).to_dict();data[field] = value
    data['content_sha256'] = digest({k: v for k, v in data.items() if k != 'content_sha256'})
    path = tmp_path / 'authority.json';path.write_text(json.dumps(data), encoding='utf8')
    experiment = manager.create_experiment('Presentation authority')
    run = manager.create_run(experiment['id'], 'benchmark', {})
    run.start()
    artifact = run.add_artifact(path, kind='presentation_report', ownership='copy')
    run.finish()
    assert manager.verify_artifact(artifact['id'])['status'] == 'verified'
    assert manager.open_result(artifact['id'])['status'] == 'invalid_result'


def test_existing_experiment_CLI_reads_and_reports_Graph_artifacts(manager, tmp_path, graphs, capsys):
    from hyso.cli import main
    experiment, artifacts, _ = stored_pair(manager, tmp_path, graphs)
    root = manager.root_dir
    manager.close()
    database = root / 'experiments.sqlite3'
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    assert main(['experiments', 'open', artifacts[0]['id'], '--root', str(root), '--json']) == 0
    opened = json.loads(capsys.readouterr().out)['result']
    assert opened['status'] == 'verified'
    assert opened['result']['verification']['graph_v2'] == 'unverified'
    assert main(['reports', 'experiment', experiment['id'], '--root', str(root),
                 '--language', 'en', '--format', 'json', '--json']) == 0
    report = json.loads(capsys.readouterr().out)['result']
    assert report['experiment']['experiment']['id'] == experiment['id']
    assert len(report['sources']) == 2
    assert report['presentation_only'] and not report['authoritative_evidence']
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
