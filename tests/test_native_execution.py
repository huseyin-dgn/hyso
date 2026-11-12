"""Stage 10 closed routing, source identity and offline integration regressions."""
from dataclasses import replace
from dataclasses import asdict
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from hyso import analysis as api, performance as perf
from hyso.analysis import native_adapters as adapters
from hyso.analysis._codec import digest
from fixture_support import load_fixture


@pytest.fixture(autouse=True)
def no_workloads(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('offline adapter regression attempted a workload or device query')
    from hyso.training import Trainer
    from hyso.benchmark import BenchmarkRunner
    monkeypatch.setattr(Trainer, 'fit', forbidden)
    monkeypatch.setattr(BenchmarkRunner, 'run', forbidden)
    monkeypatch.setattr(torch.nn.Module, '_call_impl', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    for name in ('_lazy_init', 'is_available', 'device_count', 'current_device', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def make_binding(source, features, **kwargs):
    return api.NativeBinding('operation-A', ('A',), tuple(features), source, **kwargs)


def dry(binding, features=None, **kwargs):
    plan = api.plan_analysis(api.ModelContext('A'), features=features or binding.features,
                             execution_mode='native', device=binding.device, **kwargs)
    return api.preflight_analysis(plan, (binding,))


def execute(binding, **kwargs):
    return api.execute_analysis(dry(binding), authorized_features=binding.features,
                                 explicit_authorization=True, **kwargs)


def selected(result):
    return {row['feature_id']: row for row in result.to_dict()['features'] if row['selection'] != 'prerequisite'}


def test_retained_identity_partial_and_reopen_without_fit():
    source = load_fixture('training-p19-schema1')
    binding = make_binding(api.RetainedSource(source), ('P19', 'P22'))
    result = execute(binding)
    rows = selected(result)
    assert {row['status'] for row in rows.values()} == {'COMPLETED'}
    assert rows['P19']['evidence_record']['native_payload'] == source
    assert rows['P19']['evidence_record']['origin'] == 'reused'
    assert rows['P19']['execution_receipt'] == rows['P22']['execution_receipt']
    assert rows['P19']['execution_receipt']['source_sha256'] == digest(source)
    assert result.to_evidence_index().to_dict()['records']


def test_stale_binding_rejected_before_execution():
    source = load_fixture('training-p19-schema1')
    binding = make_binding(api.RetainedSource(source), ('P19',))
    review = dry(binding)
    source['limitations'].append('changed after review')
    with pytest.raises(api.AnalysisExecutionError, match='metadata_changed'):
        api.execute_analysis(review, authorized_features=('P19',), explicit_authorization=True)


def test_invalid_native_family_fails_closed():
    result = execute(make_binding(api.RetainedSource(load_fixture('training-p19-schema1')), ('P11',)))
    assert selected(result)['P11']['status'] == 'FAILED'
    assert selected(result)['P11']['evidence_record'] is None


def test_authorization_one_shot_and_missing_source(monkeypatch):
    binding = make_binding(api.RetainedSource(load_fixture('training-p19-schema1')), ('P19',))
    review = dry(binding)
    with pytest.raises(api.AnalysisExecutionError):
        api.execute_analysis(review, authorized_features=('P22',), explicit_authorization=True)
    monkeypatch.setattr(adapters, 'invoke_operation', lambda *a: pytest.fail('unauthorized producer'))
    assert selected(api.execute_analysis(review, authorized_features=()))['P19']['status'] == 'SKIPPED'
    with pytest.raises(api.AnalysisExecutionError, match='one_shot'):
        api.execute_analysis(review, authorized_features=())
    plan = api.plan_analysis(api.ModelContext('A'), features='P19', execution_mode='native')
    assert api.preflight_analysis(plan).to_dict()['features'][0]['status'] == 'MISSING_PREREQUISITE'


def test_duplicate_operation_and_model_identity():
    binding = make_binding(api.RetainedSource(load_fixture('training-p19-schema1')), ('P19',))
    plan = dry(binding)._plan
    with pytest.raises(api.AnalysisExecutionError, match='duplicate_operation'):
        api.preflight_analysis(plan, (binding, binding))
    with pytest.raises(api.AnalysisExecutionError, match='model_identity'):
        api.preflight_analysis(plan, (replace(binding, model_ids=('B',)),))
    with pytest.raises(ValueError, match='duplicate_model'):
        api.plan_analysis((api.ModelContext('A'), api.ModelContext('A')), features='P19')


def test_artifact_missing_and_changed_metadata(tmp_path):
    path = tmp_path / 'source.json'
    path.write_text('{}', encoding='utf-8')
    reference = api.ArtifactSource.from_path(path)
    binding = make_binding(api.RetainedSource(load_fixture('training-p19-schema1')), ('P19',), artifacts=(reference,))
    review = dry(binding)
    path.write_text('{"changed":true}', encoding='utf-8')
    with pytest.raises(api.AnalysisExecutionError, match='metadata_changed'):
        api.execute_analysis(review, authorized_features=('P19',), explicit_authorization=True)
    path.unlink()
    assert dry(binding).to_dict()['features'][0]['status'] == 'MISSING_PREREQUISITE'


def saved(feature):
    return json.loads((Path(__file__).parent / 'fixtures/advisor-native-v1.json').read_text(encoding='utf-8'))['payloads'][feature]


@pytest.mark.parametrize('feature', ['P01', 'P02', 'P03', 'P08', 'P10', 'P11'])
def test_saved_native_cpu_formats(feature):
    payload = saved(feature)
    result = execute(make_binding(api.RetainedSource(payload), (feature,)))
    row = selected(result)[feature]
    assert row['status'] == 'COMPLETED', row['reasons']
    assert row['evidence_record']['native_payload'] == payload


def test_pipeline_shared_capture_and_authorized_projection(monkeypatch):
    capture = perf.PipelineResult(saved('P03'))
    producer = Mock(return_value=capture)
    monkeypatch.setattr(perf, 'measure_pipeline', producer)
    source = api.PipelineSource(torch.nn.Identity(), lambda: pytest.fail('sample source ran'))
    binding = make_binding(source, ('P03', 'P04'), policy=api.OperationPolicy(allow_capture=True))
    result = execute(binding)
    rows = selected(result)
    assert producer.call_count == 1
    assert {row['status'] for row in rows.values()} == {'COMPLETED'}, rows
    assert rows['P04']['evidence_record']['native_payload']['identity']['source_result_id'] == capture.to_dict()['result_id']
    assert rows['P03']['execution_receipt'] == rows['P04']['execution_receipt']
    producer.reset_mock()
    result = api.execute_analysis(dry(binding), authorized_features=('P04',), explicit_authorization=True)
    assert selected(result)['P03']['status'] == 'SKIPPED'
    assert producer.call_count == 1


def test_retained_pipeline_never_calls_capture(monkeypatch):
    monkeypatch.setattr(perf, 'measure_pipeline', lambda *a, **k: pytest.fail('retained capture'))
    result = execute(make_binding(api.RetainedSource(saved('P03')), ('P04',)))
    assert selected(result)['P04']['status'] == 'COMPLETED'


@pytest.mark.parametrize('feature,axis', [('P06', 'batch'), ('P07', 'resolution'), ('P08', 'precision')])
def test_sweep_routing_keeps_explicit_points_and_protocol(monkeypatch, feature, axis):
    producer = Mock(return_value=Mock(to_dict=lambda: dict(route_fixture=True)))
    monkeypatch.setattr(perf, 'run_sweep', producer)
    factory = lambda *args: pytest.fail('input factory ran')
    protocol = perf.SweepProtocol(warmup_iterations=0, measure_iterations=1)
    source = api.InferenceSource(torch.nn.Identity(), input_factory=factory,
                                 options=dict(points=(1, 2), protocol=protocol))
    binding = make_binding(source, (feature,), policy=api.OperationPolicy(allow_capture=True))
    assert dry(binding).to_dict()['features'][0]['status'] == 'SUPPORTED'
    adapters.invoke_operation(binding, (feature,))
    assert producer.call_args.kwargs == dict(points=(1, 2), protocol=protocol, axis=axis)


@pytest.mark.parametrize('feature,producer', [('P05', 'measure_cold_start'), ('P17', 'find_max_batch_size')])
def test_owned_recipe_routing(monkeypatch, feature, producer):
    target = Mock(return_value=Mock(to_dict=lambda: {'route_fixture': True}))
    monkeypatch.setattr(perf, producer, target)
    protocol = perf.ColdStartProtocol(trials=1) if feature == 'P05' else perf.BatchSearchProtocol(
        candidates=(1, 2), maximum_batch=2, cpu_memory_limit_bytes=512*1024**2,
        input_limit_bytes=1024, timeout_seconds=5.)
    source = api.RecipeSource(perf.ColdStartRecipe('absent:factory'), options=dict(protocol=protocol))
    binding = make_binding(source, (feature,), policy=api.OperationPolicy(allow_capture=True))
    assert dry(binding).to_dict()['features'][0]['status'] == 'SUPPORTED'
    adapters.invoke_operation(binding, (feature,))
    target.assert_called_once_with(source.recipe, protocol=protocol)


def test_classification_predictions_safe_normal_entry():
    source = api.EvaluationSource(targets=[0, 0, 1, 1], predictions=[0, 1, 1, 1],
                                  options=dict(class_names=['a', 'b'], split='held-out'))
    result = execute(make_binding(source, ('P26',), policy=api.OperationPolicy(allow_capture=True)))
    row = selected(result)['P26']
    assert row['status'] == 'COMPLETED', row['reasons']
    assert row['evidence_record']['native_payload']['metrics']['accuracy'] == .75


@pytest.mark.parametrize('feature,name', [('P27', 'evaluate_robustness'), ('P28', 'evaluate_precision_quality'),
                                         ('P29', 'compare_performance_quality')])
def test_quality_native_routing_and_pair_shared_operation(monkeypatch, feature, name):
    from hyso import quality
    target = Mock(return_value=Mock(to_dict=lambda: {'route_fixture': True}))
    monkeypatch.setattr(quality, name, target)
    source = api.QualitySource(torch.nn.Identity(), object(), candidate=torch.nn.Identity(), perturbation=lambda x: x)
    binding = make_binding(source, (feature,), policy=api.OperationPolicy(allow_capture=True))
    if feature == 'P29':
        binding = replace(binding, model_ids=('A', 'B'))
    adapters.invoke_operation(binding, (feature,))
    assert target.call_count == 1
    assert target.call_args.args[0] is source.model


def test_CUDA_preflight_has_no_device_queries_or_fallback():
    source = api.RecipeSource(perf.ColdStartRecipe('absent:factory', device='cuda:0'))
    binding = make_binding(source, ('P14',), device='cuda:0', policy=api.OperationPolicy(allow_capture=True))
    assert dry(binding).to_dict()['features'][0]['status'] == 'MISSING_PREREQUISITE'
    binding = replace(binding, policy=api.OperationPolicy(allow_capture=True, allow_cuda=True))
    assert dry(binding).to_dict()['features'][0]['status'] == 'SUPPORTED'
    binding = replace(binding, source=api.RecipeSource(perf.ColdStartRecipe('absent:factory')))
    assert dry(binding).to_dict()['features'][0]['status'] == 'MISSING_PREREQUISITE'


def test_allocator_peak_reset_consent_is_process_specific():
    source = api.InferenceSource(torch.nn.Identity(), torch.ones(1, 1, 2, 2))
    binding = make_binding(source, ('P09',), device='cuda:0',
                           policy=api.OperationPolicy(allow_capture=True, allow_cuda=True))
    row = dry(binding).to_dict()['features'][0]
    assert 'current_process_allocator_reset_consent_required' in row['reasons']


@pytest.mark.parametrize('feature', ['P15', 'P16'])
def test_protocol_only_native_routing_has_no_dummy_model(monkeypatch, feature):
    protocol = perf.TransferProtocol() if feature == 'P15' else perf.UtilizationProtocol()
    name = 'measure_transfers' if feature == 'P15' else 'measure_utilization'
    producer = Mock(return_value=Mock(to_dict=lambda: {'route_fixture': True}))
    monkeypatch.setattr(perf, name, producer)
    binding = make_binding(api.ProtocolSource(protocol), (feature,),
        device='cuda:0' if feature == 'P15' else 'cpu',
        policy=api.OperationPolicy(allow_capture=True, allow_cuda=True))
    assert dry(binding).to_dict()['features'][0]['status'] == 'SUPPORTED'
    adapters.invoke_operation(binding, (feature,))
    producer.assert_called_once_with(protocol=protocol)


def test_P34_rejects_logical_aliases_and_duplicate_physical_targets(tmp_path):
    source = api.RuntimeSource(perf.WorkloadRecipe('absent:factory'), str(tmp_path / 'new-run'),
                               devices=('cuda:0', 'cuda:1'))
    policy = api.OperationPolicy(allow_capture=True, allow_cuda=True, allow_file_writes=True)
    binding = make_binding(source, ('P34',), policy=policy)
    assert dry(binding).to_dict()['features'][0]['status'] == 'UNSUPPORTED'
    source = replace(source, devices=('cpu', 'cpu'))
    assert dry(replace(binding, source=source)).to_dict()['features'][0]['status'] == 'MISSING_PREREQUISITE'


def test_saved_P34_native_reader():
    payload = json.loads((Path(__file__).parent / 'fixtures/device-comparison-v1.json').read_text(encoding='utf-8'))
    row = selected(execute(make_binding(api.RetainedSource(payload), ('P34',))))['P34']
    assert row['status'] == 'COMPLETED', row['reasons']
    assert row['evidence_record']['native_payload'] == payload


def test_isolated_P23_unsupported_envelope_retained(tmp_path):
    from test_isolated_training_memory_reader import saved_result
    payload = saved_result(tmp_path / 'absent')
    row = selected(execute(make_binding(api.RetainedSource(payload), ('P23',))))['P23']
    assert row['status'] == 'UNSUPPORTED', row['reasons']
    assert row['evidence_record']['native_payload'] == payload
    assert row['auxiliary_evidence']['native_diagnostics']['diagnostics']['outcome'] == 'unsupported'


def trainer_source():
    from hyso.training import Trainer
    model = torch.nn.Linear(2, 2)
    trainer = Trainer(model, optimizer=torch.optim.SGD(model.parameters(), lr=.01),
                      loss_fn=torch.nn.CrossEntropyLoss(), device='cpu')
    return api.TrainingSource(trainer, [], options=dict(epochs=1))


def test_training_source_requires_explicit_episode_and_blocks_incompatible_recorders():
    source = trainer_source()
    policy = api.OperationPolicy(allow_capture=True, allow_training=True)
    for features in (('P21', 'P23'), ('P23', 'P20'), ('P22',)):
        binding = make_binding(source, features, policy=policy)
        rows = dry(binding).to_dict()['features']
        assert all(row['status'] == 'MISSING_PREREQUISITE' for row in rows)
    binding = make_binding(source, ('P19',), policy=api.OperationPolicy(allow_capture=True))
    assert dry(binding).to_dict()['features'][0]['status'] == 'MISSING_PREREQUISITE'


def test_one_training_capture_derived_P22_and_no_retry(monkeypatch):
    producer = Mock(return_value=perf.TrainingPerformanceResult(load_fixture('training-p19-schema1')))
    monkeypatch.setattr(perf, 'measure_training', producer)
    source = trainer_source()
    binding = make_binding(source, ('P19', 'P22'), policy=api.OperationPolicy(allow_capture=True, allow_training=True))
    rows = selected(execute(binding))
    assert {row['status'] for row in rows.values()} == {'COMPLETED'}, rows
    assert producer.call_count == 1
    assert rows['P19']['execution_receipt'] == rows['P22']['execution_receipt']
    assert dry(binding).to_dict()['features'][0]['status'] == 'MISSING_PREREQUISITE'


def test_failed_training_operation_is_not_retried_for_projection(monkeypatch):
    producer = Mock(side_effect=RuntimeError('after optimizer update'))
    monkeypatch.setattr(perf, 'measure_training', producer)
    binding = make_binding(trainer_source(), ('P19', 'P22'),
                           policy=api.OperationPolicy(allow_capture=True, allow_training=True))
    rows = selected(execute(binding))
    assert {row['status'] for row in rows.values()} == {'FAILED'}
    assert producer.call_count == 1
    assert all(row['execution_receipt']['status'] == 'FAILED' for row in rows.values())


def test_CUDA_saved_activation_production_guard_is_pre_admission_blocked():
    recipe = perf.TrainingMemoryRecipe('absent:factory', device='cuda:0')
    protocol = perf.IsolatedTrainingMemoryProtocol(saved_activation_lifetime=True)
    binding = make_binding(api.RecipeSource(recipe, options=dict(protocol=protocol)), ('P23',), device='cuda:0',
        policy=api.OperationPolicy(allow_capture=True, allow_training=True, allow_cuda=True))
    row = next(row for row in dry(binding).to_dict()['features'] if row['feature_id'] == 'P23')
    assert row['status'] == 'UNSUPPORTED'
    assert 'Basic_only_saved_activation_lifetime_disabled' in row['reasons']


@pytest.mark.parametrize('fail', [False, True])
def test_session_lifecycle_uses_native_operation_once(monkeypatch, fail):
    from hyso.training import TrainingConfig
    from torch.utils.data import DataLoader, TensorDataset
    config = TrainingConfig(epochs=1, batch_size=2, shuffle=False)
    loader = DataLoader(TensorDataset(torch.ones(2, 2), torch.zeros(2, dtype=torch.long)), batch_size=2)
    session = config.prepare(torch.nn.Linear(2, 2), loader)
    source = api.TrainingSource.from_session(session)
    def producer(*args, **kwargs):
        assert session.settings()['state'] == 'running'
        if fail:
            raise RuntimeError('fixture failure after state advance')
        session.trainer.history = dict(train_loss=[1.], train_acc=[50.], val_loss=[None], val_acc=[None])
        return perf.TrainingPerformanceResult(load_fixture('training-p19-schema1'))
    monkeypatch.setattr(perf, 'measure_training', producer)
    binding = make_binding(source, ('P19',), policy=api.OperationPolicy(allow_capture=True, allow_training=True))
    row = selected(execute(binding))['P19']
    assert row['status'] == ('FAILED' if fail else 'COMPLETED'), row['reasons']
    assert session.settings()['state'] == ('failed' if fail else 'completed')
    if not fail:
        assert row['auxiliary_evidence']['training_session']['settings']['state'] == 'completed'
    with pytest.raises(ValueError, match='one-shot'):
        session.fit()


def aborted_stability():
    from hyso.performance.runtime_result import _result
    token = 'offline-interrupted-fixture'
    evidence = dict(request=dict(token=token, protocol=asdict(perf.StabilityProtocol()),
                                 recipe=asdict(perf.WorkloadRecipe('absent:factory'))),
        controller=dict(token=token, launcher_pid=123, cleanup=dict(active_processes=0),
                        exit_code=1, kill_on_close=True, reason='fixture interruption'),
        rows=[], header=None, terminal=None, chunks=[])
    return _result('P31', evidence)


def test_interrupted_retained_runtime_never_becomes_success_or_new_capture(monkeypatch):
    monkeypatch.setattr(perf, 'measure_stability', lambda *a, **k: pytest.fail('retained runtime capture'))
    payload = aborted_stability().to_dict()
    row = selected(execute(make_binding(api.RetainedSource(payload), ('P31',))))['P31']
    assert row['status'] == 'FAILED'
    assert row['evidence_record']['native_payload'] == payload


def test_camera_and_long_run_permissions_are_separate(tmp_path):
    policy = api.OperationPolicy(allow_capture=True, allow_file_writes=True)
    source = api.RuntimeSource(perf.WorkloadRecipe('absent:trained_frames'), str(tmp_path / 'camera'),
        stream=perf.StreamSource('camera', 0, 'user-selected source'))
    row = dry(make_binding(source, ('P30',), policy=policy)).to_dict()['features'][0]
    assert row['reasons'] == ['explicit_camera_source_consent_required']
    source = api.RuntimeSource(perf.WorkloadRecipe('absent:factory'), str(tmp_path / 'long'),
        options=dict(protocol=perf.StabilityProtocol(duration_seconds=7200, interval_seconds=1.)))
    row = dry(make_binding(source, ('P31',), policy=policy)).to_dict()['features'][0]
    assert row['reasons'] == ['separate_long_run_consent_required']


def test_GPU_stability_power_sampling_requires_explicit_consent(tmp_path):
    source = api.RuntimeSource(perf.WorkloadRecipe('absent:factory', gpu_uuid='GPU-'+'a'*32),
        str(tmp_path / 'GPU'), options=dict(protocol=perf.StabilityProtocol()))
    binding = make_binding(source, ('P31',), device='cuda:0',
        policy=api.OperationPolicy(allow_capture=True, allow_file_writes=True, allow_cuda=True))
    assert 'GPU_stability_includes_power_sampling_consent_required' in dry(binding).to_dict()['features'][0]['reasons']


def test_runtime_capture_uses_existing_owned_launcher_only(monkeypatch, tmp_path):
    capture = aborted_stability()
    producer = Mock(return_value=capture)
    monkeypatch.setattr(perf, 'measure_stability', producer)
    source = api.RuntimeSource(perf.WorkloadRecipe('absent:factory'), str(tmp_path / 'new'),
                               options=dict(protocol=perf.StabilityProtocol()))
    binding = make_binding(source, ('P31',), policy=api.OperationPolicy(allow_capture=True, allow_file_writes=True))
    row = selected(execute(binding))['P31']
    assert producer.call_count == 1
    assert row['status'] == 'FAILED'
    assert row['evidence_record']['native_payload'] == capture.to_dict()


@pytest.mark.parametrize('family', ['training', 'runtime'])
def test_derived_only_authorization_cannot_inherit_capture(monkeypatch, family, tmp_path):
    if family == 'training':
        binding = make_binding(trainer_source(), ('P19', 'P22'),
            policy=api.OperationPolicy(allow_capture=True, allow_training=True))
        authorized = ('P22',)
    else:
        source = api.RuntimeSource(perf.WorkloadRecipe('absent:factory'), str(tmp_path / 'new'),
                                   options=dict(protocol=perf.StabilityProtocol()))
        binding = make_binding(source, ('P31', 'P32'),
            policy=api.OperationPolicy(allow_capture=True, allow_file_writes=True))
        authorized = ('P32',)
    monkeypatch.setattr(adapters, 'invoke_operation', lambda *a: pytest.fail('derived-only authorization started capture'))
    row = selected(api.execute_analysis(dry(binding), authorized_features=authorized, explicit_authorization=True))[authorized[0]]
    assert row['status'] == 'MISSING_PREREQUISITE'
    assert row['attempted'] is False


def test_ORT_preflight_and_launcher_routing_without_export_or_session(monkeypatch, tmp_path):
    producer = Mock(return_value=Mock(to_dict=lambda: {'route_fixture': True}))
    monkeypatch.setattr(perf, 'profile_execution_providers', producer)
    recipe = perf.WorkloadRecipe('absent:onnx_bytes_and_feed')
    options = dict(providers=['CPUExecutionProvider'], expected={'*': 'CPUExecutionProvider'},
                   optimization='disabled', protocol=perf.ProviderProtocol(warmup=0, runs=1))
    binding = make_binding(api.RuntimeSource(recipe, str(tmp_path / 'ORT'), options=options), ('P35',),
        policy=api.OperationPolicy(allow_capture=True, allow_file_writes=True))
    assert dry(binding).to_dict()['features'][0]['status'] == 'SUPPORTED'
    adapters.invoke_operation(binding, ('P35',))
    producer.assert_called_once_with(recipe, str(tmp_path / 'ORT'), **options)
    options['providers'] = ['CUDAExecutionProvider']
    assert dry(binding).to_dict()['features'][0]['status'] == 'MISSING_PREREQUISITE'


def test_native_snapshot_roundtrip_tamper_and_artifact_availability(tmp_path):
    path = tmp_path / 'source.json'
    payload = load_fixture('training-p19-schema1')
    path.write_text(json.dumps(payload), encoding='utf-8')
    binding = make_binding(api.RetainedSource(payload), ('P19', 'P22'),
                           artifacts=(api.ArtifactSource.from_path(path),))
    result = execute(binding)
    result_path = tmp_path / 'execution.json'
    result.to_json(result_path)
    assert api.AnalysisExecutionResult.from_json(result_path).to_dict() == result.to_dict()
    assert result.verify_artifacts()[0]['status'] == 'verified'
    path.unlink()
    assert result.verify_artifacts()[0]['status'] == 'missing'
    assert api.AnalysisExecutionResult.from_json(result_path).to_dict() == result.to_dict()
    changed = result.to_dict()
    changed['features'][0]['execution_receipt']['source_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='source_sha256'):
        api.AnalysisExecutionResult.from_dict(changed)


def test_two_callers_cannot_execute_same_ticket(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    binding = make_binding(api.RetainedSource(load_fixture('training-p19-schema1')), ('P19',))
    review = dry(binding)
    entered, release = Event(), Event()
    original = adapters.invoke_operation
    def delayed(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(adapters, 'invoke_operation', delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(api.execute_analysis, review, authorized_features=('P19',), explicit_authorization=True)
        try:
            assert entered.wait(5)
            with pytest.raises(api.AnalysisExecutionError, match='in_progress'):
                api.execute_analysis(review, authorized_features=('P19',), explicit_authorization=True)
        finally:
            release.set()
        assert selected(future.result())['P19']['status'] == 'COMPLETED'


def test_projection_failure_preserves_shared_capture_and_other_feature(monkeypatch):
    producer = Mock(return_value=perf.PipelineResult(saved('P03')))
    monkeypatch.setattr(perf, 'measure_pipeline', producer)
    monkeypatch.setattr(perf, 'analyze_stages', Mock(side_effect=ValueError('stage projection unavailable')))
    source = api.PipelineSource(torch.nn.Identity(), lambda: pytest.fail('sample source ran'))
    binding = make_binding(source, ('P03', 'P04'), policy=api.OperationPolicy(allow_capture=True))
    result = execute(binding)
    rows = selected(result)
    assert rows['P03']['status'] == 'COMPLETED'
    assert rows['P04']['status'] == 'FAILED'
    assert producer.call_count == 1
    assert rows['P04']['auxiliary_evidence']['native_sources']['pipeline'] == saved('P03')
    assert api.AnalysisExecutionResult.from_dict(result.to_dict()).to_dict() == result.to_dict()


def test_retained_partial_P23_is_kept_with_Basic_diagnostics():
    payload = load_fixture('training-memory-p23-schema1')
    row = selected(execute(make_binding(api.RetainedSource(payload), ('P23',))))['P23']
    assert row['status'] == 'COMPLETED', row['reasons']
    assert row['native_coverage'] == 'partial'
    assert row['evidence_record']['native_payload'] == payload
    assert row['acceptance'] == 'NOT_EVALUATED'


def test_manager_artifact_links_and_reporting_advisor_comparison(tmp_path):
    from hyso import ExperimentManager, ModelBuilder, advisor
    from hyso.reporting import create_report, report_experiment
    path = tmp_path / 'logical.json'
    path.write_text(json.dumps(saved('P11')), encoding='utf-8')
    binding = make_binding(api.RetainedSource(saved('P11')), ('P11',), artifacts=(api.ArtifactSource.from_path(path),))
    review = dry(binding)
    result = api.execute_analysis(review, authorized_features=binding.features, explicit_authorization=True)
    root = tmp_path / 'store'
    ExperimentManager(root).close()
    ExperimentManager.migrate_store(root)
    manager = ExperimentManager(root)
    try:
        experiment = manager.create_experiment('native integration')
        builder = ModelBuilder((1, 2, 2), 2).add('conv2d', in_channels=1, out_channels=2)
        run = manager.create_workflow_run(experiment['id'], 'analysis', model_ids=['A'],
            architectures={'A': builder}, analysis=review)
        run.start()
        reference = result.artifact_sources()[0]
        item = run.add_artifact(reference.path, kind='analysis_source', ownership='reference', role='input', retain_locator=True)
        assert item['expected_sha256'] == reference.sha256
        output = run.record_result(result, input_artifact_ids=[item['id']])
        assert run.record_result(result, input_artifact_ids=[item['id']])['id'] == output['id']
        run.finish()
        reopened = manager.open_result(output['id'])
        assert reopened['status'] == 'verified'
        assert reopened['result'].to_dict() == result.to_dict()
        report_experiment(manager, experiment['id']).to_dict()
        assert {tuple(row['roles']) for row in manager.get_run_details(run.id)['artifacts']} == {('input',), ('output',)}
    finally:
        manager.close()
    context = api.AnalysisContext(workload_id='fixture', task='classification', model={}, dataset={})
    advice = advisor.recommend(result.to_unified_report(context=context))
    report = create_report(result, advice, context=context)
    report.to_json(tmp_path / 'report.json')
    report.to_markdown(tmp_path / 'report.md')
    report.to_html(tmp_path / 'report.html')
    source = api.ModelComparisonSource.from_execution(result, 'A')
    candidate_binding = api.NativeBinding('retained-B', ('B',), ('P11',), api.RetainedSource(saved('P11')))
    candidate_plan = api.plan_analysis(api.ModelContext('B'), features='P11', execution_mode='native')
    candidate_result = api.execute_analysis(api.preflight_analysis(candidate_plan, (candidate_binding,)),
        authorized_features=('P11',), explicit_authorization=True)
    candidate = api.ModelComparisonSource.from_execution(candidate_result, 'B')
    comparison = api.compare_models(source, candidate, features=('P11', 'P35'))
    assert comparison.to_dict()
    assert 'NOT_SUPPORTED_IN_COMPARISON' in json.dumps(comparison.to_dict())


@pytest.mark.parametrize('command', [
    ['cpu'], ['cpu', '--features', 'P03', 'P04', 'P26'], ['training'], ['quality'],
    ['cuda-preflight', '--factory', 'absent:factory'],
    ['providers', '--factory', 'absent:onnx', '--directory', 'new-ORT-directory'],
])
def test_safe_default_examples_do_not_run_workloads(monkeypatch, command):
    import sys
    from examples.analysis import native_workflow
    monkeypatch.setattr(sys, 'argv', ['native_workflow.py', *command])
    native_workflow.main()


def test_example_retained_native_exports(tmp_path, monkeypatch):
    import sys
    from examples.analysis import native_workflow
    source = tmp_path / 'P03.json'
    source.write_text(json.dumps(saved('P03')), encoding='utf-8')
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['native_workflow.py', 'artifact', '--input', str(source),
        '--features', 'P03', 'P04', '--execute', '--output', str(output)])
    native_workflow.main()
    assert {path.name for path in output.iterdir()} == {
        'execution.json', 'evidence.json', 'recommendations.json', 'report.json', 'report.md', 'report.html'}
    assert api.AnalysisExecutionResult.from_json(output / 'execution.json').to_dict()['status_counts']['COMPLETED'] == 2


@pytest.mark.parametrize('feature', ['P26', 'P28'])
def test_CPU_binding_cannot_hide_CUDA_device_option(feature):
    if feature == 'P26':
        source = api.EvaluationSource(torch.nn.Identity(), [], options=dict(device='cuda:0'))
    else:
        source = api.QualitySource(torch.nn.Identity(), object(), options=dict(device='cuda:0'))
    binding = make_binding(source, (feature,), policy=api.OperationPolicy(allow_capture=True))
    row = next(row for row in dry(binding).to_dict()['features'] if row['feature_id'] == feature)
    assert row['status'] == 'MISSING_PREREQUISITE'


def test_predictions_are_a_source_analysis_without_capture_consent():
    source = api.EvaluationSource(targets=[0, 1], predictions=[0, 1],
        options=dict(class_names=['a', 'b'], split='synthetic'))
    result = execute(make_binding(source, ('P26',)))
    row = selected(result)['P26']
    assert row['status'] == 'COMPLETED'
    assert row['execution_receipt']['operation_kind'] == 'retained'
    assert api.AnalysisExecutionResult.from_dict(result.to_dict()).to_dict() == result.to_dict()


def test_source_publication_limit_fails_explicitly_without_truncation(monkeypatch):
    payload = saved('P03')
    original = json.dumps(payload, sort_keys=True)
    monkeypatch.setattr(adapters, 'MAX_NATIVE_BYTES', 64)
    row = selected(execute(make_binding(api.RetainedSource(payload), ('P03',))))['P03']
    assert row['status'] == 'FAILED'
    assert 'native_source_exceeds_64_MiB_evidence_limit' in str(row['reasons'])
    assert row['evidence_record'] is None
    assert json.dumps(payload, sort_keys=True) == original


def test_isolated_unsupported_timing_members_remain_unsupported(monkeypatch, tmp_path):
    from test_isolated_training_memory_reader import saved_result
    producer = Mock(return_value=perf.IsolatedTrainingMemoryResult(saved_result(tmp_path)))
    monkeypatch.setattr(perf, 'measure_training_memory_isolated', producer)
    source = api.RecipeSource(perf.TrainingMemoryRecipe('absent:factory'),
                              options=dict(protocol=perf.IsolatedTrainingMemoryProtocol()))
    binding = make_binding(source, ('P19', 'P22', 'P23'),
                           policy=api.OperationPolicy(allow_capture=True, allow_training=True))
    result = execute(binding)
    assert {row['status'] for row in selected(result).values()} == {'UNSUPPORTED'}
    assert producer.call_count == 1
    assert api.AnalysisExecutionResult.from_dict(result.to_dict()).to_dict() == result.to_dict()
