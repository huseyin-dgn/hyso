"""Stage 7 local history and retained fixtures. No live workloads or checkpoints."""
import json
from pathlib import Path
import re
import typing
import uuid

import pytest
import torch

from hyso import ExperimentManager, ModelBuilder, TrainingConfig, analysis
from hyso.experiments import ExperimentError, ExperimentRun
from hyso.training import TrainingResult
from hyso.analysis._codec import canonical
from hyso.analysis.execution import STATUSES
from test_model_comparison import source, bound


@pytest.fixture(autouse=True)
def no_workloads(monkeypatch):
    from hyso import training, benchmark, evaluation, performance
    def forbidden(*args, **kwargs):
        pytest.fail("Experiment history attempted a live workload or checkpoint access")
    for target, name in ((training.Trainer, 'fit'), (training.TrainingSession, 'fit'),
                         (benchmark.BenchmarkRunner, 'run'), (torch.nn.Module, '_call_impl'),
                         (torch, 'load'), (evaluation, 'evaluate_classification'),
                         (analysis, 'execute_analysis')):
        monkeypatch.setattr(target, name, forbidden)
    for name in ('init', '_lazy_init', 'is_available', 'device_count', 'current_device'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    for name in ('measure_throughput', 'measure_memory', 'measure_footprint'):
        monkeypatch.setattr(performance, name, forbidden)


@pytest.fixture
def manager(tmp_path):
    root = tmp_path / 'store'
    first = ExperimentManager(root)
    first.close()
    ExperimentManager.migrate_store(root)
    current = ExperimentManager(root)
    yield current
    current.close()


def builder():
    return ModelBuilder((1, 4, 4), 2).add('conv2d', in_channels=1, out_channels=2)


def training_fixture(config):
    """Synthetic UX observations for storage checks, with no training claim."""
    return TrainingResult(canonical(dict(format='hyso.training_session_result', schema_version=1,
        settings=dict(state='completed', requested=config.to_dict(),
                      effective=dict(device='cpu', precision='fp32')),
        history=dict(train_loss=[.5], train_acc=[75.], val_loss=[None], val_acc=[None]),
        completed_history_epochs=1, applied_updates=2,
        epoch_events=[dict(epoch=1, interval_seconds=.1)],
        fixture_scope='synthetic observations, no training performed')))


def training_run(manager, experiment_id=None, **kwargs):
    config = TrainingConfig(epochs=1, batch_size=2)
    identity = experiment_id or manager.create_experiment('study')['id']
    run = manager.create_workflow_run(identity, 'training', model_ids=('A',),
        architectures={'A': builder()}, training_config=config,
        dataset={'id': 'synthetic-v1', 'split': 'train'}, device='cpu', precision='fp32', **kwargs)
    run.start()
    return run, training_fixture(config)


def test_schema1_snapshots_no_automatic_migration_and_explicit_migration(tmp_path):
    root = tmp_path / 'schema1'
    m = ExperimentManager(root)
    run, result = training_run(m)
    before = m._core_snapshot(m._db)
    with pytest.raises(ExperimentError, match='unsupported_b2'):
        run.record_result(result)
    assert m._core_snapshot(m._db) == before
    assert m.get_run_details(run.id)['artifact_support'] == 'schema2_required'
    result.log_to(run)  # The original scalar path still works on schema1.
    run.finish()
    snapshot = m._core_snapshot(m._db)
    m.close()
    m = ExperimentManager(root)
    assert m._schema_version == 1
    assert m.get_run(run.id).to_dict()['status'] == 'completed'
    assert m._core_snapshot(m._db) == snapshot
    m.close()
    migration = ExperimentManager.migrate_store(root)
    m = ExperimentManager(root)
    assert m._core_snapshot(m._db) == snapshot
    assert m._schema_version == 2 and Path(migration['backup']).is_file()
    assert m._db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    assert m._db.execute('PRAGMA foreign_key_check').fetchall() == []
    m.close()


def test_training_recording_replay_and_read_only_reopen(manager):
    run, result = training_run(manager)
    artifact = run.record_result(result)
    assert run.record_result(result)['id'] == artifact['id']
    assert len(run.read_metrics()) == 3
    run.finish()
    before = manager._core_snapshot(manager._db)
    detail = manager.get_run_details(run.id, limit=2)
    assert detail['run']['status'] == 'completed'
    assert detail['run']['config']['context']['dataset']['id'] == 'synthetic-v1'
    assert detail['artifacts'][0]['roles'] == ['output']
    assert detail['artifacts'][0]['availability'] == 'verified'
    assert len(manager.get_run_details(run.id, after_metric_seq=detail['next_metric_seq'])['metrics']) == 1
    assert manager.open_result(artifact['id'])['result'].to_dict() == result.to_dict()
    assert manager._core_snapshot(manager._db) == before
    assert manager._db.execute('SELECT count(*) FROM artifact_checks').fetchone()[0] == 0
    root = manager.root_dir
    manager.close()
    reopened = ExperimentManager(root)
    try:
        assert reopened.get_run_details(run.id)['run']['status'] == 'completed'
        assert reopened.open_result(artifact['id'])['result'].history == result.history
        assert reopened._core_snapshot(reopened._db) == before
    finally:
        reopened.close()


def test_analysis_comparison_and_same_experiment_artifact_links(manager):
    experiment = manager.create_experiment('shared')
    plan = analysis.plan_analysis(analysis.ModelContext('A'), features='P11')
    dry = analysis.preflight_analysis(plan)
    retained = json.loads((Path(__file__).parent / 'fixtures/advisor-native-v1.json').read_text(encoding='utf-8'))['payloads']['P11']
    record = analysis.EvidenceRecord.from_native(retained, feature_id='P11', scope=analysis.EvidenceScope(model_id='A'))
    row = dict(model_id='A', feature_id='P11', status='COMPLETED', reasons=[],
               evidence_record=record.to_dict(), auxiliary_evidence={})
    # Retained fixture envelope, never a call to the current executor.
    result = analysis.AnalysisExecutionResult(canonical(dict(format='hyso.analysis_execution', schema_version=1,
        dry_run=dry.to_dict(), features=[row], authorized_features=['P11'],
        status_counts={s: int(s == 'COMPLETED') for s in STATUSES})))
    run = manager.create_workflow_run(experiment['id'], 'analysis', model_ids=['A'], analysis=dry,
        protocol={'scope': 'retained fixture'})
    run.start()
    evidence = run.record_result(result)
    run.finish()
    loaded = manager.open_result(evidence['id'])
    assert loaded['status'] == 'verified'
    assert loaded['result'].to_evidence_index().to_dict() == result.to_evidence_index().to_dict()
    comparison = analysis.compare_models(source('A', 'P26', bound('A')), source('B', 'P26', bound('B')), features='P26')
    other = manager.create_workflow_run(experiment['id'], 'comparison', model_ids=['A', 'B'], comparison=comparison)
    other.start()
    saved = other.record_result(comparison, input_artifact_ids=[evidence['id']])
    assert other.record_result(comparison, input_artifact_ids=[evidence['id']])['id'] == saved['id']
    other.finish()
    details = manager.get_run_details(other.id)
    assert {tuple(a['roles']) for a in details['artifacts']} == {('input',), ('output',)}
    assert manager.open_result(saved['id'])['result'].to_json() == comparison.to_json()
    assert details['run']['config']['context']['dataset'] is None
    assert manager.get_experiment_history(experiment['id'])['status'] == 'completed'


@pytest.mark.parametrize('damage,expected', [('missing', 'missing'), ('changed', 'mismatch'), ('moved', 'missing')])
def test_missing_changed_moved_artifacts_have_no_result(manager, tmp_path, damage, expected):
    run, result = training_run(manager)
    artifact = run.record_result(result)
    run.finish()
    path = manager.root_dir / artifact['managed_relative_path']
    if damage == 'changed':
        path.write_bytes(b'corrupt')
    else:
        path.rename(tmp_path / 'relocated.json')
    loaded = manager.open_result(artifact['id'])
    assert loaded['status'] == expected and loaded['result'] is None and loaded['reason']
    assert manager.get_run_details(run.id)['artifacts'][0]['availability'] == expected


def test_reference_relocation_is_explicit_and_preserves_identity(manager, tmp_path):
    run, result = training_run(manager)
    path = tmp_path / 'result.json'
    path.write_text(canonical(result.to_dict()), encoding='utf-8')
    reference = run.add_artifact(path, kind='training_result', ownership='reference', retain_locator=True)
    moved = path.rename(tmp_path / 'moved.json')
    before = manager.get_artifact(reference['id'])
    assert manager.open_result(reference['id'])['status'] == 'missing'
    assert manager.open_result(reference['id'], path=moved)['result'].history == result.history
    assert manager.get_artifact(reference['id']) == before
    hidden = run.add_artifact(moved, kind='training_result', ownership='reference')
    assert manager.open_result(hidden['id'])['status'] == 'unavailable'
    assert manager.open_result(hidden['id'], path=moved)['status'] == 'verified'
    run.interrupt()


@pytest.mark.parametrize('payload,status', [(b'not json', 'invalid_result'), (b'{}', 'invalid_result')])
def test_hash_valid_but_invalid_results_are_rejected(manager, tmp_path, payload, status):
    run, _ = training_run(manager)
    path = tmp_path / 'invalid.json'
    path.write_bytes(payload)
    artifact = run.add_artifact(path, kind='analysis_execution', ownership='copy')
    assert manager.open_result(artifact['id'])['status'] == status
    assert manager.open_result(artifact['id'])['result'] is None
    run.fail('invalid fixture')


def test_result_conflicts_and_foreign_inputs_do_not_change_recorded_result(manager):
    first, result = training_run(manager)
    artifact = first.record_result(result)
    changed = result.to_dict()
    changed['history']['train_loss'] = [.9]
    with pytest.raises(ExperimentError, match='workflow_result_conflict'):
        first.record_result(TrainingResult(canonical(changed)))
    other, other_result = training_run(manager)
    with pytest.raises(ExperimentError, match='experiment_mismatch'):
        other.record_result(other_result, input_artifact_ids=[artifact['id']])
    assert manager.list_artifacts(other.id) == []
    first.finish()
    other.interrupt()


def test_failed_ingest_and_metrics_failure_preserve_explicit_recovery(manager, monkeypatch):
    run, result = training_run(manager)
    with monkeypatch.context() as patch:
        patch.setattr(TrainingResult, 'log_to', lambda *a: (_ for _ in ()).throw(OSError('tracking failure')))
        with pytest.raises(OSError):
            run.record_result(result)
    assert run.to_dict()['status'] == 'running' and run.read_metrics() == []
    assert run.record_result(result)['ingest_state'] == 'ready'
    assert len(manager.list_artifacts(run.id)) == 1
    assert len(run.read_metrics()) == 3
    run.finish()
    broken, broken_result = training_run(manager)
    with monkeypatch.context() as patch:
        patch.setattr(manager, '_publish', lambda *a: (_ for _ in ()).throw(OSError('disk full')))
        with pytest.raises(ExperimentError, match='artifact_io_error'):
            broken.record_result(broken_result)
    with pytest.raises(ExperimentError, match='result_storage_incomplete'):
        broken.record_result(broken_result)
    assert broken.read_metrics() == []
    with pytest.raises(ExperimentError, match='outputs_not_ready'):
        broken.finish()
    broken.interrupt()
    manager.reconcile_storage()
    assert broken.to_dict()['status'] == 'interrupted'


def test_experiment_filters_cursor_summary_and_revision_recovery(manager):
    empty = manager.create_experiment('empty')
    experiment = manager.create_experiment('lifecycle')
    run = manager.create_run(experiment['id'], 'training', {})
    assert manager.get_experiment_history(empty['id'])['status'] == 'created'
    assert manager.get_experiment_history(experiment['id'])['status'] == 'created'
    run.start()
    assert manager.get_experiment_history(experiment['id'])['status'] == 'running'
    old = run.to_dict()
    foreign = ExperimentManager(manager.root_dir)
    try:
        assert foreign.get_experiment_history(experiment['id'])['runs'][0]['writer_liveness'] == 'unknown'
        assert run.to_dict() == old
        foreign.recover_interrupted(run.id, old['revision'], 'coordinated writer shutdown')
    finally:
        foreign.close()
    assert manager.get_experiment_history(experiment['id'])['status'] == 'interrupted'
    retry = manager.create_run(experiment['id'], 'training', {}, retry_of_run_id=run.id)
    retry.start()
    retry.fail('failed fixture')
    view = manager.get_experiment_history(experiment['id'], limit=1)
    assert view['status'] == 'failed' and view['status_counts']['failed'] == 1
    assert manager.get_experiment_history(experiment['id'], after_seq=view['next_after_seq'])['runs'][0]['id'] == retry.id
    assert manager.list_experiments(name='lifecycle', kind='training', status='failed') == [experiment]
    assert manager.list_experiments(kind='benchmark', status='failed') == []
    page = manager.list_experiments(limit=1)
    assert manager.list_experiments(after_seq=page[-1]['seq']) == [experiment]
    with pytest.raises(ExperimentError):
        manager.list_experiments(status='unknown')
    with pytest.raises(ExperimentError):
        run.finish()


def test_metadata_validation_identity_replay_and_type_hints(manager):
    experiment = manager.create_experiment('typed')
    request = uuid.uuid4().hex
    arch = builder()
    run = manager.create_workflow_run(experiment['id'], 'model', model_ids=['A'], architectures={'A': arch}, request_id=request)
    assert manager.create_workflow_run(experiment['id'], 'model', model_ids=['A'], architectures={'A': arch}, request_id=request).id == run.id
    before = run.to_dict()
    arch.add('relu')
    assert run.to_dict() == before
    for kwargs in (dict(model_ids=['A', 'A']), dict(model_ids=['A'], architectures={'B': arch}),
                   dict(model_ids=['A'], architectures={'A': object()})):
        with pytest.raises(ExperimentError):
            manager.create_workflow_run(experiment['id'], 'model', **kwargs)
    for target in (ExperimentManager.create_workflow_run, ExperimentManager.get_run_details,
                   ExperimentManager.open_result, ExperimentRun.record_result):
        assert typing.get_type_hints(target)['return']
    run.start()
    run.finish()


def test_large_history_batches_replay_after_partial_metric_commit(manager, monkeypatch):
    config = TrainingConfig(epochs=150, batch_size=2)
    experiment = manager.create_experiment('retained long history')
    run = manager.create_workflow_run(experiment['id'], 'training', model_ids=['A'], training_config=config)
    data = training_fixture(config).to_dict()
    data['history'] = dict(train_loss=[.5] * 150, train_acc=[75.] * 150,
                           val_loss=[None] * 150, val_acc=[None] * 150)
    data['completed_history_epochs'] = 150
    data['epoch_events'] = [dict(epoch=i, interval_seconds=.1) for i in range(1, 151)]
    result = TrainingResult(canonical(data))
    run.start()
    original = run.log_metrics
    calls = []
    def fail_second_batch(records):
        calls.append(len(records))
        if len(calls) == 2:
            raise OSError('metric storage failed')
        return original(records)
    with monkeypatch.context() as patch:
        patch.setattr(run, 'log_metrics', fail_second_batch)
        with pytest.raises(OSError):
            run.record_result(result)
    assert len(run.read_metrics()) == 256 and calls == [256, 194]
    run.record_result(result)
    assert len(run.read_metrics()) == 450
    assert len(manager.list_artifacts(run.id)) == 1
    run.finish()


@pytest.mark.parametrize('change,error', [('settings', 'training_settings_mismatch'),
                                         ('device', 'training_context_mismatch'),
                                         ('kind', 'workflow_result_stage_mismatch')])
def test_result_context_mismatches_are_rejected_before_artifact_writes(manager, change, error):
    experiment = manager.create_experiment('mismatch')
    config = TrainingConfig(epochs=1, batch_size=2)
    if change == 'kind':
        run = manager.create_workflow_run(experiment['id'], 'model', model_ids=['A'], architectures={'A': builder()})
    else:
        run = manager.create_workflow_run(experiment['id'], 'training', model_ids=['A'], training_config=config,
            device='cuda:0' if change == 'device' else 'cpu')
    run.start()
    result = training_fixture(TrainingConfig(epochs=2) if change == 'settings' else config)
    with pytest.raises(ExperimentError, match=error):
        run.record_result(result)
    assert manager.list_artifacts(run.id) == [] and run.read_metrics() == []
    run.fail('context mismatch')


def test_history_documentation_functions_execute_with_retained_result(tmp_path):
    root = Path(__file__).resolve().parents[1]
    guide = root / 'docs/guides/experiment-workflow.md'
    text = guide.read_text(encoding='utf-8')
    namespace = {}
    for code in re.findall(r'^```python\n(.*?)^```', text, re.M | re.S):
        exec(compile(code, str(guide), 'exec'), namespace)
    store = tmp_path / 'documented-store'
    experiment = namespace['create_study'](store)
    assert namespace['enable_result_artifacts'](store)['status'] == 'migrated'
    config = TrainingConfig(epochs=1, batch_size=2)
    run_id, artifact_id = namespace['record_saved_training'](store, experiment, builder(), config, training_fixture(config))
    experiments, history, details, opened = namespace['read_study'](store, experiment, artifact_id)
    assert experiments[0]['id'] == experiment and history['status'] == 'completed'
    assert details[-1]['run']['id'] == run_id and opened['status'] == 'verified'
    readme = (root / 'README.md').read_text(encoding='utf-8')
    code, = [code for code in re.findall(r'^```python\n(.*?)^```', readme, re.M | re.S)
              if 'def review_experiment(' in code]
    exec(compile(code, 'README.md', 'exec'), namespace)
    assert namespace['review_experiment'](store, experiment)[0] == history
