"""Metadata admission and one-shot scheduling of closed native adapters."""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

from ._codec import canonical, digest
from ._native import validate_native, native_outcome
from .catalog import feature_catalog
from .evidence import EvidenceRecord, EvidenceScope
from .feature_plan import plan_features
from .sources import NativeBinding, RetainedSource, OperationPolicy


def preflight_native(plan, bindings, isolated_policy):
    """Read binding metadata and file identities without invoking producers."""
    from .execution import AnalysisDryRun, AnalysisExecutionError
    from .native_adapters import admission

    if any(type(binding) is not NativeBinding for binding in bindings):
        raise AnalysisExecutionError('native_mode_requires_native_bindings')
    models = {context.model_id for context in plan.request.models}
    operation_ids = [binding.operation_id for binding in bindings]
    if len(set(operation_ids)) != len(operation_ids):
        raise AnalysisExecutionError('duplicate_operation_id')
    lookup = {}
    trainers = set()
    from .sources import TrainingSource
    for binding in bindings:
        if not set(binding.model_ids).issubset(models):
            raise AnalysisExecutionError('binding_model_identity_mismatch')
        if not set(binding.features).issubset(plan.selected_features):
            raise AnalysisExecutionError('binding_features_must_be_explicitly_selected')
        if type(binding.source) is TrainingSource:
            identity = id(binding.source.trainer)
            if identity in trainers:
                raise AnalysisExecutionError('distinct_episodes_require_distinct_trainers')
            trainers.add(identity)
        for model in binding.model_ids:
            for feature in binding.features:
                if (model, feature) in lookup:
                    raise AnalysisExecutionError('duplicate_model_feature_binding')
                lookup[model, feature] = binding
    rows = []
    for feature in plan_features(plan.request).features:
        for context, original in zip(plan.request.models, feature.models):
            binding = lookup.get((context.model_id, feature.feature_id))
            status, reasons = 'MISSING_PREREQUISITE', ['missing_native_source_binding']
            resolutions = []
            if feature.selection == 'prerequisite':
                status, reasons = 'SKIPPED', ['dependency_only_not_selected;never_auto_execute']
            elif binding is not None:
                status, reasons = admission(binding, feature.feature_id, plan.device, context)
                resolutions = [dict(reason=reason,
                    resolution='retained_native_validation' if type(binding.source) is RetainedSource
                    else 'typed_source_and_execution_time_native_admission',
                    status='pending_native_validation' if status == 'SUPPORTED' else 'blocked')
                    for reason in original.reasons] if original.status == 'BLOCKED' else []
            spec = feature_catalog(feature.feature_id)
            rows.append(dict(model_id=context.model_id, feature_id=feature.feature_id,
                selection=feature.selection, executor_support='SUPPORTED', status=status, reasons=reasons,
                dependencies=list(spec.dependencies), cost_class=spec.cost_class,
                executor_scope=spec.executor_scope, planner_status=original.status,
                planner_reasons=list(original.reasons), blocker_resolutions=resolutions,
                operation_id=None if binding is None else binding.operation_id))
    data = dict(format='hyso.analysis_dry_run', schema_version=1, plan=plan.to_dict(),
                features=rows, bindings=[binding.metadata() for binding in bindings],
                resources=dict(status='native_protocol_bounds;no_resource_reservation'), isolated_policy=None,
                execution_authorized=False, acceptance='NOT_EVALUATED',
                trust_scope='trusted caller resources; native ownership and timeout semantics')
    return AnalysisDryRun(plan, tuple(bindings), isolated_policy, data)


def _outcome(payload):
    """Map explicit terminal failures while retaining valid partial measurements."""
    outcome = payload.get('outcome', native_outcome(payload))
    if outcome in ('unsupported', 'unavailable', 'not_supported'):
        return 'UNSUPPORTED', [str(payload.get('reason') or outcome)]
    if outcome in ('aborted', 'failed', 'timeout', 'timed_out', 'oom', 'native_crash',
                   'workload_exception', 'measurement_error', 'resource_limit', 'invalid_receipt',
                   'interrupted', 'crashed'):
        return 'FAILED', [str(payload.get('reason') or outcome)]
    if outcome in ('not_executed', 'not_run', 'no_new_epochs'):
        return 'MISSING_PREREQUISITE', [str(payload.get('reason') or outcome)]
    return 'COMPLETED', []


def _exception_status(error):
    if isinstance(error, (ImportError, NotImplementedError)):
        return 'UNSUPPORTED'
    if isinstance(error, FileNotFoundError):
        return 'MISSING_PREREQUISITE'
    if getattr(error, 'code', None) == 'native_reader_unavailable':
        return 'UNSUPPORTED'
    return 'FAILED'


def execute_native(dry_run, authorized):
    from .execution import AnalysisExecutionError, AnalysisExecutionResult, STATUSES
    from .native_adapters import invoke_operation, validate_projection

    fresh = preflight_native(dry_run._plan, dry_run._bindings, dry_run._isolated_policy)
    data, reviewed = fresh.to_dict(), dry_run.to_dict()
    if data['bindings'] != reviewed['bindings'] or data['plan']['plan_id'] != reviewed['plan']['plan_id']:
        raise AnalysisExecutionError('plan_or_binding_metadata_changed;review_a_new_dry_run')
    dry_run._used = True
    lookup = {binding.operation_id: binding for binding in dry_run._bindings}
    cache = {}
    rows = []
    for before in data['features']:
        row = dict(before, evidence_record=None, auxiliary_evidence={}, execution_receipt=None, attempted=False)
        feature = row['feature_id']
        if feature not in authorized and row['selection'] != 'prerequisite':
            row.update(status='SKIPPED', reasons=['not_authorized'])
        elif row['status'] == 'SUPPORTED':
            binding = lookup[row['operation_id']]
            selected = tuple(f for f in binding.features if f in authorized and any(
                item['operation_id'] == binding.operation_id and item['feature_id'] == f
                and item['status'] == 'SUPPORTED' for item in data['features']))
            # Admission is repeated for the actual authorized subset. P22-only
            # and retained P32/P33 requests cannot inherit a sibling's capture.
            from .native_adapters import admission
            context = next(item for item in dry_run._plan.request.models if item.model_id == row['model_id'])
            if binding.operation_id not in cache:
                status, reasons = admission(replace(binding, features=selected), feature, dry_run._plan.device, context)
                if status != 'SUPPORTED':
                    row.update(status=status, reasons=reasons)
                    rows.append(row)
                    continue
            row['attempted'] = True
            if binding.operation_id not in cache:
                receipt = dict(operation_id=binding.operation_id, operation_kind='pending',
                    binding_sha256=digest(binding.metadata()), selected_features=list(selected),
                    model_ids=list(binding.model_ids), execution_device=binding.device,
                    controller_pid=os.getpid(), acceptance='NOT_EVALUATED')
                try:
                    projections, auxiliary, details = invoke_operation(binding, selected)
                    receipt.update(details, status='COMPLETED')
                    cache[binding.operation_id] = (projections, auxiliary, receipt, None)
                except Exception as error:
                    receipt.update(status=_exception_status(error), reason=type(error).__name__ + ':' + str(error)[:240])
                    cache[binding.operation_id] = ({}, {}, receipt, error)
            projections, auxiliary, receipt, error = cache[binding.operation_id]
            row['execution_receipt'] = receipt
            row['auxiliary_evidence'] = auxiliary
            try:
                if error is not None:
                    raise AnalysisExecutionError(receipt['reason'])
                if feature in auxiliary.get('projection_errors', {}):
                    row.update(status=auxiliary.get('projection_statuses', {}).get(feature, 'FAILED'),
                        reasons=[auxiliary['projection_errors'][feature]])
                    rows.append(row)
                    continue
                native = projections[feature]
                validate_projection(native, feature)
                record = EvidenceRecord.from_native(native, feature_id=feature,
                    scope=EvidenceScope(model_id=row['model_id']),
                    origin='reused' if receipt['operation_kind'] == 'retained' else 'new')
                status, reasons = _outcome(native)
                if feature == 'P09' and native.get('cuda_memory') is None:
                    status, reasons = 'UNSUPPORTED', ['native_cuda_allocator_evidence_unavailable']
                if feature == 'P01' and any(metric['value'] is None for metric in native['measurements']):
                    status, reasons = 'MISSING_PREREQUISITE', ['requested_quantile_native_tail_policy_not_met;raw_samples_retained']
                if feature == 'P10' and native['actual']['observations'] == 0:
                    status, reasons = 'UNSUPPORTED', ['native_RSS_backend_unavailable']
                coverage = native_outcome(native)
                if native.get('format') == 'hyso.performance.isolated_training_memory':
                    coverage = (native.get('measurement') or {}).get('coverage', 'none')
                row.update(evidence_record=record.to_dict(), status=status, reasons=reasons,
                    native_coverage=coverage, acceptance='NOT_EVALUATED')
            except Exception as error:
                row.update(status=receipt['status'] if error is not None and receipt['status'] != 'COMPLETED'
                    else _exception_status(error), reasons=[type(error).__name__ + ':' + str(error)[:240]])
        rows.append(row)
    result = dict(format='hyso.analysis_execution', schema_version=1, dry_run=data,
        authorized_features=list(authorized), features=rows,
        status_counts={status: sum(row['status'] == status for row in rows) for status in STATUSES},
        acceptance='NOT_EVALUATED', comparison_authority='per_model_observations_only;no_certified_winner',
        reporting_scope='native evidence and coverage with operation receipts')
    return AnalysisExecutionResult(canonical(result))


def validate_operation_row(row, data):
    """Validate saved auxiliary native sources and operation/source associations."""
    from .native_adapters import validate_projection
    receipt = row.get('execution_receipt')
    auxiliary = row.get('auxiliary_evidence', {})
    sources = auxiliary.get('native_sources', {})
    for payload in sources.values():
        validate_native(payload)
        if payload.get('format') in ('hyso.performance.worker_trace', 'hyso.performance.worker_tasks'):
            base = row.get('evidence_record')
            if base is None or row['feature_id'] != 'P21' or payload['source_p21'] != base['native_payload']:
                raise ValueError('worker_companion_source_mismatch')
    if 'training_session' in auxiliary:
        from hyso.experiments.workflow import _restore_result
        _restore_result('training_result', auxiliary['training_session'])
    record = row.get('evidence_record')
    if record is not None:
        validate_projection(record['native_payload'], row['feature_id'])
    if receipt is None:
        if row['attempted']:
            raise ValueError('attempted_operation_receipt_required')
        return
    bindings = {binding['operation_id']: binding for binding in data['dry_run']['bindings']}
    binding = bindings.get(receipt['operation_id'])
    if (binding is None or receipt['binding_sha256'] != digest(binding)
            or row['model_id'] not in receipt['model_ids']
            or receipt['model_ids'] != binding['model_ids']
            or row['feature_id'] not in receipt['selected_features']
            or len(set(receipt['selected_features'])) != len(receipt['selected_features'])
            or not set(receipt['selected_features']).issubset(binding['features'])
            or not set(receipt['selected_features']).issubset(data['authorized_features'])):
        raise ValueError('operation_binding_or_selection_mismatch')
    if receipt['status'] == 'COMPLETED':
        expected = digest(sources['source']) if receipt['operation_kind'] == 'retained' and 'source' in sources else digest(sources)
        if receipt['source_sha256'] != expected:
            raise ValueError('operation_source_sha256_mismatch')
        if receipt['operation_kind'] == 'retained' and binding['source'].get('type') == 'RetainedSource':
            if receipt['source_sha256'] != binding['source']['payload_sha256']:
                raise ValueError('retained_binding_source_identity_mismatch')
            companions = [digest(payload) for payload in sources.values()
                if payload.get('format') in ('hyso.performance.worker_trace', 'hyso.performance.worker_tasks')]
            if sorted(companions) != sorted(binding['source']['companions']):
                raise ValueError('retained_worker_companion_identity_mismatch')
