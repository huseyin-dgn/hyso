"""Read-only eligibility under separate schema-2 and bound schema-3 policies.

No scope declarations repair evidence. Equality here is of recorded ordered
tensor content and configured metric semantics, not lineage or preprocessing.
"""
from ._json import AdvisorError
from .comparison import (ComparisonReason, ComparisonResult, ComparisonStatus,
                         P26BoundQualityComparisonContract)
from .resolver import NativeMetricResolver


def _capture(resolver, record_id, contract, side):
    reasons = []
    def missing(code, field):
        reasons.append(ComparisonReason(code, field, side))
    record = resolver.record(record_id)
    if record is None:
        missing('record_not_found', 'record_id')
        return None, None, reasons
    n = record['native_payload']
    schema = 3 if type(contract) is P26BoundQualityComparisonContract else 2
    if (record['feature_id'] != 'P26' or record['validation']['status'] != 'native_validated'
            or not n or n.get('format') != 'hyso.evaluation.classification' or n.get('schema_version') != schema):
        missing('p26_schema%d_producer_evidence_required' % schema, 'native_payload')
        return None, None, reasons
    if schema == 3:
        # Recheck the complete native constructor, including the bound snapshot
        # and all stream/state/run peer links, before admitting schema 3.
        from hyso.evaluation import EvaluationResult
        EvaluationResult(n)
    # The immutable resolver already reran the native constructor, including
    # source SHA, derived metric, stream method/version/scope and status checks.
    e = n['producer_evidence']; s = e['evaluated_stream']
    if n['status'] != 'complete' or s['sample_count'] < contract.min_samples:
        missing('sample_requirement_not_met', 'sample_count')
    if record['bindings']['model_state_sha256'] != 'native' or e['model_state']['status'] != 'native':
        missing('native_model_state_required', 'model_state_sha256')
    model_type = n['source']['model_type']
    if type(model_type) is not str or not model_type.strip() or model_type.lower() in ('unknown', 'not_collected'):
        missing('native_model_type_missing', 'source.model_type')
    resolution = resolver.resolve(record_id, contract.metric)
    metric = resolution.resolved
    if resolution.status != 'supported':
        missing('metric_not_observable', contract.metric.value)
    elif resolver.check_metric(metric).status != 'supported':
        missing('invalid_native_evidence', 'metric')
        metric = None
    # Preserve the native payload's declared/missing authority. In particular,
    # source_id and split never stand in for evaluation-run or stream identity.
    conditions = dict(
        sample_count=s['sample_count'], inputs_sha256=s['inputs_sha256'],
        targets_sha256=s['targets_sha256'], paired_sha256=s['paired_sha256'],
        stream_semantics={key: s[key] for key in ('method', 'scope', 'source', 'batch_partition_invariant')},
        evidence_format=(e['format'], e['schema_version']),
        configuration=n['configuration'], batch_sizes=n['accounting']['batch_sizes'],
        runtime=n['provenance'], execution={key: n['source'][key] for key in
            ('device', 'execution', 'sampler', 'num_workers')},
        model=(model_type, e['model_state']['method'], e['model_state']['sha256']))
    if metric is not None:
        conditions['metric_semantics'] = (metric.metric.value, metric.reference.metric_path,
            metric.reference.unit, metric.reference.domain, metric.reference.status,
            metric.reference.protocol_ref, metric.protocol, metric.native_context.get('averaging'))
    return metric, conditions, reasons


def compare_p26_records(left_resolver, left_id, right_resolver, right_id, contract):
    if type(left_resolver) is not NativeMetricResolver or type(right_resolver) is not NativeMetricResolver:
        raise AdvisorError('validated_native_resolvers_required')
    lm, left, lr = _capture(left_resolver, left_id, contract, 'left')
    rm, right, rr = _capture(right_resolver, right_id, contract, 'right')
    reasons = lr + rr
    incompatible = False
    a, b = left_resolver.record(left_id), right_resolver.record(right_id)
    if a is not None and b is not None:
        same = (left_id == right_id or a['artifact'] and b['artifact'] and
                a['artifact']['sha256'] == b['artifact']['sha256'] or
                a['source'] and b['source'] and a['source']['payload_sha256'] == b['source']['payload_sha256'])
        an, bn = a['native_payload'] or {}, b['native_payload'] or {}
        run = an.get('evaluation_id')
        if same or run is not None and run == bn.get('evaluation_id'):
            reasons.append(ComparisonReason('same_underlying_capture', 'record/run/artifact'))
            incompatible = True
    if left is not None and right is not None:
        for key in sorted((set(left) & set(right)) - {'model'}):
            if left[key] != right[key]:
                reasons.append(ComparisonReason('condition_mismatch', key))
                incompatible = True
        if left['model'] != right['model']:
            allowed = 'model' in contract.allowed_changes
            reasons.append(ComparisonReason('allowed_change' if allowed else 'unpermitted_change', 'model'))
            incompatible |= not allowed
        elif 'model' not in contract.allowed_changes:
            # Equal named tensors and type names cannot prove equal model
            # behavior/architecture. This capture lacks a full code identity.
            reasons.append(ComparisonReason('model_behavior_identity_not_captured', 'model'))
    insufficient = bool(lr or rr) or left is None or right is None or lm is None or rm is None
    if 'model' not in contract.allowed_changes:
        insufficient = True
    status = (ComparisonStatus.INCOMPATIBLE if incompatible else
              ComparisonStatus.INSUFFICIENT_EVIDENCE if insufficient else ComparisonStatus.COMPARABLE)
    if status == ComparisonStatus.COMPARABLE:
        reasons.append(ComparisonReason('exact_observed_stream_compatible', contract.policy))
    return ComparisonResult(status, contract.content_sha256, tuple(reasons),
                            lm.reference if lm else None, rm.reference if rm else None)
