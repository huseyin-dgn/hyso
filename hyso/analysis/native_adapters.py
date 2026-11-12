"""Closed native producer/reader dispatch. No source-declared executor imports."""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

from ._codec import digest
from ._native import validate_native, restore_native
from .sources import (ArtifactSource, RetainedSource, InferenceSource, PipelineSource,
                      RecipeSource, EvaluationSource, QualitySource, TrainingSource,
                      ProtocolSource, RooflineSource, RuntimeSource)

MAX_NATIVE_BYTES = 64 * 1024**2


def _check_payload_budget(payload):
    """Reject an oversized native source before duplicating it into envelopes."""
    size = 0
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    for chunk in encoder.iterencode(payload):
        size += len(chunk.encode('utf-8'))
        if size > MAX_NATIVE_BYTES:
            raise ValueError('native_source_exceeds_64_MiB_evidence_limit')


FORMATS = {
    'P01': ('hyso.performance.latency_analysis',),
    'P02': ('hyso.performance.throughput',), 'P03': ('hyso.performance.pipeline',),
    'P04': ('hyso.performance.stage_timing',), 'P05': ('hyso.performance.cold_start',),
    'P06': ('hyso.performance.sweep',), 'P07': ('hyso.performance.sweep',),
    'P08': ('hyso.performance.sweep',), 'P09': (None,),
    'P10': ('hyso.performance.memory',), 'P11': ('hyso.performance.footprint',),
    'P12': ('hyso.performance.operation_count',),
    'P13': ('hyso.performance.layer_profile', 'hyso.performance.layer_profile.nsys'),
    'P14': ('hyso.performance.operator_profile', 'hyso.performance.operator_profile.nsys'),
    'P15': ('hyso.performance.transfer',),
    'P16': ('hyso.performance.utilization',), 'P17': ('hyso.performance.capacity',),
    'P18': ('hyso.performance.roofline',), 'P19': ('hyso.performance.training_timing',),
    'P20': ('hyso.performance.training_stages',), 'P21': ('hyso.performance.dataloader',),
    'P22': ('hyso.performance.training_throughput',),
    'P23': ('hyso.performance.training_memory', 'hyso.performance.isolated_training_memory'),
    'P24': ('hyso.quality_experiment',), 'P25': ('hyso.performance.gradient_diagnostics',),
    'P26': ('hyso.evaluation.classification',),
    'P27': ('hyso.quality_experiment',), 'P28': ('hyso.quality_experiment',),
    'P29': ('hyso.quality_experiment',),
    **{f'P{i}': ('hyso.performance.runtime',) for i in range(30, 36)},
}


def validate_projection(payload, feature):
    if payload.get('format') not in FORMATS[feature] or feature not in validate_native(payload)[1]:
        raise ValueError('native_feature_or_result_family_mismatch')


def admission(binding, feature, device, context):
    """Validate typed resources and consent without touching a native producer."""
    source = binding.source
    checks = [item.verify() for item in binding.artifacts]
    bad_artifacts = [item['status'] for item in checks if item['status'] != 'verified']
    if bad_artifacts:
        return 'MISSING_PREREQUISITE', ['artifact_' + status for status in bad_artifacts]
    if binding.device != device:
        return 'MISSING_PREREQUISITE', ['operation_device_must_match_plan']
    if type(source) is RetainedSource:
        if device != 'cpu':
            return 'UNSUPPORTED', ['retained_analysis_execution_device_is_CPU']
        return 'SUPPORTED', []
    if type(source) is RooflineSource:
        if device != 'cpu':
            return 'UNSUPPORTED', ['offline_profile_analysis_execution_device_is_CPU']
        if source.kind not in ('counter', 'nsys_operators', 'nsys_modules'):
            return 'UNSUPPORTED', ['unknown_offline_profile_kind']
        allowed = {'counter': {'P18'}, 'nsys_operators': {'P14'}, 'nsys_modules': {'P13'}}
        if not set(binding.features).issubset(allowed[source.kind]):
            return 'UNSUPPORTED', ['offline_source_feature_mismatch']
        if source.kind != 'counter' and not binding.artifacts:
            return 'MISSING_PREREQUISITE', ['hash_pinned_Nsight_artifacts_required']
        return 'SUPPORTED', []
    if type(source) is ProtocolSource:
        return _protocol_admission(binding, feature)
    if type(source) is RuntimeSource:
        return _runtime_admission(binding, feature)
    if type(source) is TrainingSource:
        return _training_admission(binding)
    if type(source) is RecipeSource and 'P23' in binding.features:
        return _isolated_training_admission(binding)
    if type(source) not in (InferenceSource, PipelineSource, RecipeSource, EvaluationSource, QualitySource):
        return 'MISSING_PREREQUISITE', ['capture_adapter_not_bound']
    prediction_analysis = type(source) is EvaluationSource and source.model is None
    if not binding.policy.allow_capture and not prediction_analysis:
        return 'MISSING_PREREQUISITE', ['operation_capture_consent_required']
    if device != 'cpu' and not binding.policy.allow_cuda:
        return 'MISSING_PREREQUISITE', ['explicit_CUDA_operation_consent_required']
    supported = {
        InferenceSource: {'P01', 'P02', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11'},
        PipelineSource: {'P03', 'P04'}, RecipeSource: {'P05', 'P12', 'P13', 'P14', 'P17', 'P18'},
        EvaluationSource: {'P26'}, QualitySource: {'P24', 'P27', 'P28', 'P29'},
    }
    if not set(binding.features).issubset(supported[type(source)]):
        return 'UNSUPPORTED', ['source_family_cannot_produce_selected_features']
    if len(binding.model_ids) > 1 and not (type(source) is QualitySource and binding.features == ('P29',)):
        return 'MISSING_PREREQUISITE', ['shared_model_pair_requires_P29']
    if feature in ('P24', 'P26', 'P27', 'P28', 'P29') and context.task != 'classification':
        return 'UNSUPPORTED', ['classification_task_required']
    if type(source) in (InferenceSource, EvaluationSource, QualitySource):
        if context.backend != 'pytorch':
            return 'UNSUPPORTED', ['pytorch_native_source_required']
    options = source.options
    if type(options) is not dict:
        return 'MISSING_PREREQUISITE', ['native_options_mapping_required']
    if type(source) is QualitySource and feature in ('P24', 'P27', 'P29') and device != 'cpu':
        return 'UNSUPPORTED', ['quality_operation_CPU_only']
    if feature == 'P11' and device != 'cpu':
        return 'UNSUPPORTED', ['logical_footprint_CPU_model_required']
    if feature == 'P09' and device == 'cpu':
        return 'UNSUPPORTED', ['CUDA_allocator_capture_requires_CUDA']
    if feature in ('P26', 'P28') and str(options.get('device', 'cpu')) != device:
        return 'MISSING_PREREQUISITE', ['native_device_option_must_match_operation_device']
    if device != 'cpu':
        if feature in ('P02', 'P03', 'P04', 'P06', 'P07', 'P08') and options.get('protocol') is None:
            return 'MISSING_PREREQUISITE', ['explicit_CUDA_native_protocol_required']
        if feature in ('P26', 'P28') and str(options.get('device')) != device:
            return 'MISSING_PREREQUISITE', ['explicit_native_device_option_required']
    protocol = options.get('protocol')
    declared_device = getattr(protocol, 'device', device)
    if declared_device != device:
        return 'MISSING_PREREQUISITE', ['native_protocol_device_mismatch']
    if feature == 'P09' or getattr(protocol, 'measure_cuda_memory', False):
        if binding.policy.peak_reset_pid != os.getpid():
            return 'MISSING_PREREQUISITE', ['current_process_allocator_reset_consent_required']
    if type(source) is InferenceSource:
        if feature != 'P10' and (source.model is None or (feature in ('P01', 'P02', 'P09') and source.input_tensor is None)):
            return 'MISSING_PREREQUISITE', ['resident_model_and_input_required']
        if feature in ('P06', 'P07', 'P08'):
            if len(binding.features) != 1 or not callable(source.input_factory) or not options.get('points'):
                return 'MISSING_PREREQUISITE', ['one_explicit_sweep_axis_and_input_factory_required']
        elif len(binding.features) != 1:
            return 'UNSUPPORTED', ['inference_options_require_one_operation_per_protocol']
        import torch
        expected = 'cpu' if feature in ('P06', 'P07', 'P08', 'P11') else device
        if isinstance(source.model, torch.nn.Module):
            if any(str(tensor.device) != expected for tensor in (*source.model.parameters(), *source.model.buffers())):
                return 'MISSING_PREREQUISITE', ['resident_model_device_mismatch']
        if source.input_tensor is not None and str(source.input_tensor.device) != device:
            return 'MISSING_PREREQUISITE', ['resident_input_device_mismatch']
    if type(source) is PipelineSource and (source.model is None or not callable(source.source)):
        return 'MISSING_PREREQUISITE', ['pipeline_model_and_sample_source_required']
    if type(source) is RecipeSource:
        from hyso.performance import ColdStartRecipe, BatchSearchProtocol
        if type(source.recipe) is not ColdStartRecipe:
            return 'MISSING_PREREQUISITE', ['typed_cold_start_recipe_required']
        if source.recipe.to_dict()['device'] != device:
            return 'MISSING_PREREQUISITE', ['native_recipe_device_mismatch']
        if feature == 'P17' and type(options.get('protocol')) is not BatchSearchProtocol:
            return 'MISSING_PREREQUISITE', ['bounded_batch_search_protocol_required']
        if any(f in binding.features for f in ('P05', 'P17')) and len(binding.features) != 1:
            return 'UNSUPPORTED', ['fresh_process_operation_cannot_share_unrelated_capture']
    if type(source) is EvaluationSource:
        if prediction_analysis and device != 'cpu':
            return 'UNSUPPORTED', ['prediction_analysis_execution_device_is_CPU']
        if source.model is None and (source.targets is None or source.predictions is None):
            return 'MISSING_PREREQUISITE', ['classification_model_loader_or_predictions_required']
        if source.model is not None and source.loader is None:
            return 'MISSING_PREREQUISITE', ['labeled_loader_required']
        if prediction_analysis and any(getattr(value, 'device', None) is not None and str(value.device) != 'cpu'
                                       for value in (source.targets, source.predictions)):
            return 'UNSUPPORTED', ['retained_predictions_require_CPU_data']
    if type(source) is QualitySource:
        if len(binding.features) != 1 or source.model is None or source.cohort is None:
            return 'MISSING_PREREQUISITE', ['one_quality_operation_with_model_and_cohort_required']
        if feature == 'P27' and not callable(source.perturbation):
            return 'MISSING_PREREQUISITE', ['trusted_perturbation_required']
        if feature == 'P29' and (source.candidate is None or len(binding.model_ids) != 2):
            return 'MISSING_PREREQUISITE', ['explicit_baseline_candidate_model_pair_required']
        if feature == 'P24' and (not binding.policy.allow_training or not callable(source.model)):
            return 'MISSING_PREREQUISITE', ['bounded_multi_seed_training_factory_and_consent_required']
    if (options.get('serialize_to') is not None or options.get('trace_path') is not None) and not binding.policy.allow_file_writes:
        return 'MISSING_PREREQUISITE', ['operation_file_write_consent_required']
    return 'SUPPORTED', []


def _protocol_admission(binding, feature):
    from hyso import performance as perf
    protocol = binding.source.protocol
    expected = {'P15': perf.TransferProtocol, 'P16': perf.UtilizationProtocol}
    if len(binding.features) != 1 or feature not in expected or type(protocol) is not expected[feature]:
        return 'UNSUPPORTED', ['typed_transfer_or_utilization_protocol_required']
    if not binding.policy.allow_capture:
        return 'MISSING_PREREQUISITE', ['operation_capture_consent_required']
    if feature == 'P15':
        if binding.device != protocol.device or protocol.device != 'cuda:0':
            return 'UNSUPPORTED', ['native_transfer_cuda_0_profile_required']
    elif binding.device != 'cpu':
        return 'UNSUPPORTED', ['telemetry_runs_in_current_CPU_process']
    if (feature == 'P15' or protocol.gpu) and not binding.policy.allow_cuda:
        return 'MISSING_PREREQUISITE', ['explicit_GPU_resource_consent_required']
    return 'SUPPORTED', []


def _runtime_admission(binding, feature):
    from hyso import performance as perf
    source = binding.source
    features = set(binding.features)
    if source.recipe is None:
        if binding.device != 'cpu':
            return 'UNSUPPORTED', ['retained_runtime_analysis_execution_device_is_CPU']
        if not features.issubset({'P31', 'P32', 'P33'}) or not source.directory or not binding.artifacts:
            return 'MISSING_PREREQUISITE', ['retained_P31_directory_and_pinned_journal_sources_required']
        return 'SUPPORTED', []
    if not (features.issubset({'P31', 'P32', 'P33'}) and 'P31' in features
            or binding.features in (('P30',), ('P34',), ('P35',))):
        return 'MISSING_PREREQUISITE', ['runtime_adapter_not_bound']
    if type(source.recipe) is not perf.WorkloadRecipe or not source.directory:
        return 'MISSING_PREREQUISITE', ['typed_runtime_recipe_and_output_directory_required']
    if not binding.policy.allow_capture or not binding.policy.allow_file_writes:
        return 'MISSING_PREREQUISITE', ['runtime_capture_and_file_write_consent_required']
    if Path(source.directory).exists():
        return 'MISSING_PREREQUISITE', ['native_runtime_requires_new_output_directory']
    if feature == 'P35':
        providers = source.options.get('providers', ('CPUExecutionProvider',))
        gpu_providers = {'CUDAExecutionProvider', 'TensorrtExecutionProvider'}
        if set(providers or ()) - {'CPUExecutionProvider', 'AzureExecutionProvider', *gpu_providers}:
            return 'UNSUPPORTED', ['provider_resource_admission_requires_supported_CPU_or_CUDA_profile']
        uses_gpu = bool(set(providers or ()) & gpu_providers)
        if (uses_gpu or source.recipe.gpu_uuid is not None) and not binding.policy.allow_cuda:
            return 'MISSING_PREREQUISITE', ['explicit_GPU_provider_consent_required']
        if binding.device != 'cpu' and not uses_gpu:
            return 'MISSING_PREREQUISITE', ['selected_CUDA_provider_required']
        if uses_gpu and binding.device == 'cpu':
            return 'MISSING_PREREQUISITE', ['CUDA_provider_requires_explicit_CUDA_operation_device']
        if source.recipe.gpu_uuid is not None and binding.device not in ('cpu', 'cuda:0'):
            return 'UNSUPPORTED', ['UUID_visibility_uses_worker_cuda_0']
        return 'SUPPORTED', []
    if feature == 'P30':
        if type(source.stream) is not perf.StreamSource:
            return 'MISSING_PREREQUISITE', ['explicit_video_or_camera_source_required']
        if source.recipe.gpu_uuid is not None or binding.device != 'cpu':
            return 'UNSUPPORTED', ['native_stream_processor_CPU_only']
        if source.stream.kind == 'camera' and not binding.policy.allow_camera:
            return 'MISSING_PREREQUISITE', ['explicit_camera_source_consent_required']
        return 'SUPPORTED', []
    if 'P31' in features:
        protocol = source.options.get('protocol')
        if type(protocol) is not perf.StabilityProtocol:
            return 'MISSING_PREREQUISITE', ['explicit_stability_duration_and_journal_protocol_required']
        if protocol.duration_seconds >= 7200 and not binding.policy.allow_long_run:
            return 'MISSING_PREREQUISITE', ['separate_long_run_consent_required']
        if source.recipe.gpu_uuid is not None:
            if binding.device != 'cuda:0':
                return 'UNSUPPORTED', ['UUID_visibility_uses_worker_cuda_0']
            if not binding.policy.allow_cuda or not binding.policy.allow_power_sampling:
                return 'MISSING_PREREQUISITE', ['GPU_stability_includes_power_sampling_consent_required']
        elif binding.device != 'cpu':
            return 'MISSING_PREREQUISITE', ['explicit_physical_GPU_recipe_required']
        return 'SUPPORTED', []
    from hyso.performance.devices import _device_id
    try:
        devices = tuple(_device_id(item) for item in source.devices)
    except ValueError:
        return 'UNSUPPORTED', ['physical_device_UUID_required']
    if not 2 <= len(devices) <= 8 or len(set(devices)) != len(devices):
        return 'MISSING_PREREQUISITE', ['at_least_two_distinct_physical_devices_required']
    if any(item != 'cpu' for item in devices) and not binding.policy.allow_cuda:
        return 'MISSING_PREREQUISITE', ['explicit_GPU_resource_consent_required']
    return 'SUPPORTED', []


def _training_admission(binding):
    from hyso.training import Trainer, TrainingSession
    source, policy, features = binding.source, binding.policy, set(binding.features)
    if len(binding.model_ids) != 1 or not features.issubset({'P19', 'P20', 'P21', 'P22', 'P23', 'P25'}):
        return 'UNSUPPORTED', ['one_model_training_episode_required']
    if (len(features & {'P21', 'P23', 'P25'}) > 1 or 'P23' in features and 'P20' in features):
        return 'MISSING_PREREQUISITE', ['incompatible_recorders_require_separate_fresh_episodes']
    if features == {'P22'}:
        return 'MISSING_PREREQUISITE', ['P22_requires_retained_P19_or_explicit_training_capture']
    if not policy.allow_capture or not policy.allow_training:
        return 'MISSING_PREREQUISITE', ['explicit_training_episode_consent_required']
    if type(source.trainer) is not Trainer or source.train_loader is None:
        return 'MISSING_PREREQUISITE', ['standard_fresh_Trainer_and_train_loader_required']
    if source.trainer._epochs is not None or getattr(source.trainer, '_analysis_episode_used', False):
        return 'MISSING_PREREQUISITE', ['fresh_one_shot_Trainer_required']
    if str(source.trainer.device) != binding.device:
        return 'MISSING_PREREQUISITE', ['Trainer_device_mismatch']
    if binding.device != 'cpu' and not policy.allow_cuda:
        return 'MISSING_PREREQUISITE', ['explicit_CUDA_training_consent_required']
    if type(source.options.get('epochs')) is not int or source.options['epochs'] < 1:
        return 'MISSING_PREREQUISITE', ['explicit_positive_epoch_budget_required']
    if source.options.get('save_last_to') is not None and not policy.allow_file_writes:
        return 'MISSING_PREREQUISITE', ['checkpoint_write_consent_required']
    if source.session is not None:
        if (type(source.session) is not TrainingSession or source.session.trainer is not source.trainer
                or source.session.train_loader is not source.train_loader or source.session.val_loader is not source.val_loader
                or source.session.settings()['state'] != 'prepared'):
            return 'MISSING_PREREQUISITE', ['prepared_session_source_mismatch']
        settings = source.session.settings()['requested']
        if any(source.options.get(key) != settings[key] for key in ('epochs', 'save_last_to', 'resume_from')):
            return 'MISSING_PREREQUISITE', ['session_fit_options_must_match_prepared_configuration']
    return 'SUPPORTED', []


def _isolated_training_admission(binding):
    from hyso import performance as perf
    source, policy = binding.source, binding.policy
    if not set(binding.features).issubset({'P19', 'P22', 'P23'}) or type(source.recipe) is not perf.TrainingMemoryRecipe:
        return 'UNSUPPORTED', ['isolated_P23_recipe_and_compatible_projections_required']
    if not policy.allow_capture or not policy.allow_training:
        return 'MISSING_PREREQUISITE', ['explicit_training_episode_consent_required']
    if source.recipe.to_dict()['device'] != binding.device:
        return 'MISSING_PREREQUISITE', ['native_recipe_device_mismatch']
    if binding.device != 'cpu' and not policy.allow_cuda:
        return 'MISSING_PREREQUISITE', ['explicit_CUDA_training_consent_required']
    protocol = source.options.get('protocol')
    if type(protocol) is not perf.IsolatedTrainingMemoryProtocol:
        return 'MISSING_PREREQUISITE', ['explicit_isolated_memory_protocol_required']
    if protocol.saved_activation_lifetime:
        return 'UNSUPPORTED', ['Basic_only_saved_activation_lifetime_disabled']
    if source.recipe.to_dict()['save_last_to'] is not None and not policy.allow_file_writes:
        return 'MISSING_PREREQUISITE', ['checkpoint_write_consent_required']
    return 'SUPPORTED', []


def _retained(source, features):
    from hyso import performance as perf
    payload = source.payload
    _check_payload_budget(payload)
    native = restore_native(payload)
    projections, errors = {}, {}
    for feature in features:
        try:
            projections[feature] = _retained_projection(source, native, feature)
        except Exception as error:
            errors[feature] = type(error).__name__ + ':' + str(error)[:240]
    companions = {}
    for payload in source.companions:
        result_type, ids = validate_native(payload)
        if ids != ['P21'] or 'worker_' not in payload.get('format', ''):
            raise ValueError('P21_worker_companion_required')
        if payload.get('source_p21') != source.payload:
            raise ValueError('worker_companion_source_mismatch')
        key = result_type + ':' + digest(payload)
        if key in companions:
            raise ValueError('duplicate_worker_companion')
        companions[key] = payload
    return projections, dict(native_sources=dict(source=source.payload, **companions), projection_errors=errors)


def _retained_projection(source, native, feature):
    from hyso import performance as perf
    payload, options = source.payload, _derive_options(source.options, feature)
    if payload.get('format') in FORMATS[feature] and feature in validate_native(payload)[1]:
        return payload
    if feature == 'P01' and payload.get('benchmark_type') == 'inference':
        return perf.analyze_latency(native, **options).to_dict()
    if feature == 'P04' and payload.get('format') == 'hyso.performance.pipeline':
        return perf.analyze_stages(native).to_dict()
    if feature in ('P12', 'P13', 'P18') and payload.get('format') == 'hyso.performance.operator_profile':
        producer = {'P12': perf.analyze_flops, 'P13': perf.profile_layers, 'P18': perf.analyze_roofline}[feature]
        return producer(native, **options).to_dict()
    if feature == 'P22' and payload.get('format') == 'hyso.performance.training_timing':
        return perf.analyze_training_throughput(native).to_dict()
    if feature in ('P32', 'P33') and payload.get('feature_id') == 'P31':
        producer = perf.analyze_memory_trend if feature == 'P32' else perf.analyze_energy
        return producer(native, **options).to_dict()
    raise ValueError('retained_source_cannot_produce_selected_feature')


def invoke_operation(binding, features):
    source = binding.source
    if type(source) is RuntimeSource and source.recipe is None:
        from hyso.performance import reopen_stability
        snapshot = reopen_stability(source.directory).to_dict()
        projections, auxiliary = _retained(RetainedSource(snapshot, source.options), features)
        return projections, auxiliary, dict(operation_kind='retained', source_sha256=digest(snapshot),
            artifact_receipts=[item.verify() for item in binding.artifacts])
    if type(source) is RetainedSource:
        projections, auxiliary = _retained(source, features)
        if 'P01' in features and source.payload.get('benchmark_type') == 'inference':
            auxiliary['benchmark'] = source.payload
        if 'P23' in projections:
            native = restore_native(projections['P23'])
            auxiliary['native_diagnostics'] = dict(basic_metrics=native.basic_metrics(), diagnostics=native.diagnostics())
        return projections, auxiliary, dict(operation_kind='retained',
            source_sha256=digest(source.payload), artifact_receipts=[item.verify() for item in binding.artifacts])
    if type(source) is RooflineSource:
        projections, auxiliary = _offline_profile(source, features)
        for payload in projections.values():
            _check_payload_budget(payload)
        return projections, auxiliary, dict(operation_kind='retained',
            source_sha256=digest(auxiliary['native_sources']),
            artifact_receipts=[item.verify() for item in binding.artifacts])
    projections, auxiliary = _capture(source, features, binding)
    artifacts = [item.verify() for item in binding.artifacts]
    if type(source) is RuntimeSource:
        artifacts.extend(ArtifactSource.from_path(path).verify() for path in sorted(Path(source.directory).rglob('*')) if path.is_file())
    for name in ('trace_path', 'serialize_to', 'save_last_to'):
        path = getattr(source, 'options', {}).get(name)
        if path is not None and Path(path).is_file():
            artifacts.append(ArtifactSource.from_path(path).verify())
    kind = 'retained' if type(source) is EvaluationSource and source.model is None else 'capture'
    return projections, auxiliary, dict(operation_kind=kind,
        source_sha256=digest(auxiliary.get('native_sources', projections)),
        artifact_receipts=artifacts)


def _derive_options(options, feature):
    if any(key in FORMATS for key in options):
        return options.get(feature, {})
    return options


def _call(producer, *args, options):
    """Reject unknown native options using the producer's public signature."""
    inspect.signature(producer).bind(*args, **options)
    return producer(*args, **options)


def _capture(source, features, binding):
    from hyso import performance as perf, evaluation, quality
    options = dict(getattr(source, 'options', {}))
    projections, sources, errors, projection_statuses = {}, {}, {}, {}
    if type(source) is PipelineSource:
        capture = _call(perf.measure_pipeline, source.model, source.source, options=options)
        sources['pipeline'] = capture.to_dict()
        if 'P03' in features:
            projections['P03'] = capture.to_dict()
        if 'P04' in features:
            _project(projections, errors, 'P04', perf.analyze_stages, capture)
    elif type(source) is RecipeSource:
        if 'P23' in features:
            capture = _call(perf.measure_training_memory_isolated, source.recipe, options=options)
            projections['P23'] = capture.to_dict()
            sources['isolated_memory'] = capture.to_dict()
            measurement = capture.to_dict().get('measurement')
            if measurement is not None:
                _training_projections(projections, measurement['source_p19'], None, features)
            else:
                outcome = capture.to_dict()['outcome']
                for feature in features:
                    if feature != 'P23':
                        errors[feature] = 'isolated_training_timing_unavailable:' + outcome
                        projection_statuses[feature] = 'UNSUPPORTED' if outcome == 'unsupported' else 'FAILED'
        elif features == ('P05',):
            capture = _call(perf.measure_cold_start, source.recipe, options=options)
            projections['P05'] = capture.to_dict()
        elif features == ('P17',):
            capture = _call(perf.find_max_batch_size, source.recipe, options=options)
            projections['P17'] = capture.to_dict()
        elif features == ('P18',):
            capture = _call(perf.measure_roofline, source.recipe, options=options)
            projections['P18'] = capture.to_dict()
        else:
            capture = _call(perf.profile_operators, source.recipe, options=options)
            sources['operator_profile'] = capture.to_dict()
            for feature in features:
                producer = {'P12': perf.analyze_flops, 'P13': perf.profile_layers, 'P18': perf.analyze_roofline}
                if feature == 'P14':
                    projections[feature] = capture.to_dict()
                else:
                    _project(projections, errors, feature, producer[feature], capture,
                             options=source.analysis_options.get(feature, {}))
    elif type(source) is InferenceSource:
        feature = features[0]
        if feature in ('P01', 'P09'):
            from hyso.benchmark import BenchmarkRunner
            quantiles = options.pop('quantiles', (.5, .95, .99))
            capture = BenchmarkRunner(source.model, device=binding.device, example_input=source.input_tensor,
                                      measure_cuda_memory=feature == 'P09', **options).run()
            sources['benchmark'] = capture.to_dict()
            projections[feature] = (perf.analyze_latency(capture, quantiles=quantiles).to_dict()
                                    if feature == 'P01' else capture.to_dict())
        elif feature == 'P02':
            projections[feature] = _call(perf.measure_throughput, source.model, source.input_tensor, options=options).to_dict()
        elif feature in ('P06', 'P07', 'P08'):
            axis = {'P06': 'batch', 'P07': 'resolution', 'P08': 'precision'}[feature]
            projections[feature] = _call(perf.run_sweep, source.model, source.input_factory,
                                        options=dict(options, axis=axis)).to_dict()
        elif feature == 'P10':
            projections[feature] = _call(perf.measure_memory, options=options).to_dict()
        elif feature == 'P11':
            projections[feature] = _call(perf.measure_footprint, source.model, options=options).to_dict()
    elif type(source) is EvaluationSource:
        producer, args = (evaluation.evaluate_classification, (source.model, source.loader)) if source.model is not None else (
            evaluation.evaluate_predictions, (source.targets, source.predictions))
        projections['P26'] = _call(producer, *args, options=options).to_dict()
    elif type(source) is QualitySource:
        feature = features[0]
        producer, args = {
            'P24': (quality.measure_time_to_quality, (source.model, source.cohort)),
            'P27': (quality.evaluate_robustness, (source.model, source.cohort, source.perturbation)),
            'P28': (quality.evaluate_precision_quality, (source.model, source.cohort)),
            'P29': (quality.compare_performance_quality, (source.model, source.candidate, source.cohort)),
        }[feature]
        projections[feature] = _call(producer, *args, options=options).to_dict()
    elif type(source) is TrainingSource:
        source.trainer._analysis_episode_used = True
        if source.session is None:
            return _training_capture(source, features)
        native, session_result = source.session._fit_with_native(lambda: _training_capture(source, features))
        native[1]['training_session'] = session_result.to_dict()
        return native
    elif type(source) is ProtocolSource:
        producer = perf.measure_transfers if features == ('P15',) else perf.measure_utilization
        projections[features[0]] = producer(protocol=source.protocol).to_dict()
    elif type(source) is RuntimeSource:
        analysis_options = options.pop('analysis_options', {})
        if features == ('P30',):
            capture = _call(perf.measure_streaming, source.stream, source.recipe, source.directory, options=options)
            projections['P30'] = capture.to_dict()
        elif 'P31' in features:
            capture = _call(perf.measure_stability, source.recipe, source.directory, options=options)
            projections, derived = _retained(RetainedSource(capture.to_dict(), analysis_options), features)
            errors.update(derived['projection_errors'])
        elif features == ('P35',):
            capture = _call(perf.profile_execution_providers, source.recipe, source.directory, options=options)
            projections['P35'] = capture.to_dict()
        else:
            capture = _call(perf.compare_devices, source.recipe, source.devices, source.directory, options=options)
            projections['P34'] = capture.to_dict()
        sources['runtime'] = capture.to_dict()
    else:
        raise ValueError('capture_adapter_not_bound')
    if not sources:
        sources['capture'] = next(iter(projections.values()))
    for payload in sources.values():
        _check_payload_budget(payload)
    auxiliary = dict(native_sources=sources, projection_errors=errors, projection_statuses=projection_statuses)
    if 'P01' in features and 'benchmark' in sources:
        auxiliary['benchmark'] = sources['benchmark']
    return projections, auxiliary


def _project(projections, errors, feature, producer, *args, options=None):
    try:
        projections[feature] = _call(producer, *args, options=options or {}).to_dict()
    except Exception as error:
        errors[feature] = type(error).__name__ + ':' + str(error)[:240]


def _training_projections(projections, timing, stages, features, throughput=None):
    from hyso import performance as perf
    timing = perf.TrainingPerformanceResult(timing)
    if 'P19' in features:
        projections['P19'] = timing.to_dict()
    if 'P20' in features:
        if stages is None:
            raise ValueError('selected_training_stages_not_produced')
        projections['P20'] = stages
    if 'P22' in features:
        projections['P22'] = (throughput or perf.analyze_training_throughput(timing)).to_dict()


def _training_capture(source, features):
    from hyso import performance as perf
    options = dict(source.options)
    if 'P21' in features:
        producer = perf.measure_training_with_loader
        options['stages'] = 'P20' in features
    elif 'P25' in features:
        producer = perf.measure_gradient_diagnostics
    elif 'P23' in features:
        producer = perf.measure_training_memory
    elif 'P20' in features:
        producer = perf.measure_training_detailed
    else:
        producer = perf.measure_training
    result = _call(producer, source.trainer, source.train_loader, source.val_loader, options=options)
    projections = {}
    if 'P21' in features:
        projections['P21'] = result.loader.to_dict()
        timing, stages = result.timing.to_dict(), None if result.stages is None else result.stages.to_dict()
        primary = result.loader.to_dict()
    elif 'P25' in features or 'P23' in features:
        primary = result.to_dict()
        projections['P25' if 'P25' in features else 'P23'] = primary
        timing, stages = primary['source_p19'], primary.get('source_p20')
    elif 'P20' in features:
        timing, stages = result.timing.to_dict(), result.stages.to_dict()
        primary = stages
    else:
        timing, stages = result.to_dict(), None
        primary = timing
    _training_projections(projections, timing, stages, features, getattr(result, 'throughput', None))
    auxiliary = dict(native_sources=dict(training_capture=primary))
    _check_payload_budget(primary)
    if 'P23' in features:
        auxiliary['native_diagnostics'] = dict(basic_metrics=result.basic_metrics(), diagnostics=result.diagnostics())
    return projections, auxiliary


def _offline_profile(source, features):
    from hyso import performance as perf
    if source.kind == 'counter':
        result = _call(perf.analyze_counter_roofline, source.source, options=source.options)
        return {'P18': result.to_dict()}, dict(native_sources=dict(roofline=result.to_dict()))
    if source.kind == 'nsys_modules':
        result = _call(perf.analyze_nsys_modules, source.source, options=source.options)
        return {'P13': result}, dict(native_sources=dict(layer=result))
    result = _call(perf.analyze_nsys_operators, source.source, options=source.options)
    return {'P14': result}, dict(native_sources=dict(operator_profile=result))
