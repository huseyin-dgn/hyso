"""Record-local goal evaluation v1. No deployment or future feasibility proof."""
from fractions import Fraction
import hashlib

from ._json import AdvisorError, canonical
from .contracts import (AdvisorRequest, EvaluationProtocol, EvidenceSupport,
                        Feasibility, GoalEvaluation, InputValidity, Metric,
                        MetricScope, ObservedGoal, UserGoal, _METRICS)
from .resolver import NativeMetric, NativeMetricResolver

GOAL_RULE_ID = 'record-local-goal'
GOAL_RULE_VERSION = '1'
_SUPPORTED = {
    Metric.RESIDENT_FORWARD_P95: ('P01', NativeMetric.FORWARD_P95, 'inference'),
    Metric.E2E_P95: ('P03', NativeMetric.E2E_P95, 'inference'),
    Metric.IMAGE_THROUGHPUT: ('P02', NativeMetric.COMPLETED_IMAGES_PER_SECOND, 'inference'),
    Metric.ACCURACY: ('P26', NativeMetric.TEST_ACCURACY, 'evaluation'),
    Metric.PROCESS_RSS_PEAK: ('P10', NativeMetric.RSS_SAMPLED_PEAK, 'process'),
    Metric.LOGICAL_TENSOR_BYTES: ('P11', NativeMetric.MODEL_LOGICAL_BYTES, 'static'),
}


def _context(resolver, resolved, metric):
    """Project explicit selectors without changing any EvidenceRecord binding."""
    record = resolver.record(resolved.reference.record_id)
    scope = dict(record['scope'])
    native = resolved.native_context
    scope['phase'] = _SUPPORTED[metric][2]
    # These are local native measurement facts, not upgraded index associations.
    for key in ('device', 'precision', 'batch', 'input_shape'):
        if native.get(key) is not None:
            scope[key] = native[key]
    payload = record['native_payload']
    env = payload.get('provenance', {}).get('environment', {})
    backend = env.get('backend')
    if type(backend) is dict:
        # A settings fingerprint is not an inferred backend name/capability.
        scope['backend'] = 'native-settings-sha256:' + hashlib.sha256(canonical(backend).encode()).hexdigest()
    elif type(backend) is str:
        scope['backend'] = backend
    phase = scope['phase']
    if metric == Metric.ACCURACY:
        scope['split'] = native['split']
    selected = {key: scope.get(key) for key in MetricScope.__dataclass_fields__}
    selected['input_shape'] = tuple(selected['input_shape'] or ())
    selected['phase'] = phase
    request_scope = MetricScope(**selected)
    protocol = resolved.protocol
    sample_policy = canonical(dict(
        native_metric=resolved.metric.value,
        native_protocol_sha256=hashlib.sha256(canonical(protocol).encode()).hexdigest(),
        sample_count=native.get('sample_count'),
    ))
    request_protocol = EvaluationProtocol(
        protocol_id=resolved.reference.protocol_ref,
        version=protocol.get('version', 1),
        description='Exact retained native capture profile; record-local assessment only.',
        sample_policy=sample_policy,
        window=native.get('measurement_scope'),
        evaluator='hyso.evaluation.classification:schema1' if metric == Metric.ACCURACY else None,
        averaging='sample_accuracy' if metric == Metric.ACCURACY else None,
    )
    return request_scope, request_protocol


def goal_context(resolver, record_id, metric):
    """Explicitly request a retained record's scope/protocol selectors.

    This convenience call adds no target and leaves missing fields missing.
    In particular it cannot repair a missing backend, native clock or model link.
    Model/workload/dataset labels retain their original authority in evidence.
    """
    if type(resolver) is not NativeMetricResolver:
        raise AdvisorError('typed_resolver_required')
    try:
        metric = Metric(metric)
    except (ValueError, TypeError) as exc:
        raise AdvisorError('unsupported_goal_metric') from exc
    if metric not in _SUPPORTED:
        raise AdvisorError('unsupported_goal_metric')
    feature, identity, _ = _SUPPORTED[metric]
    record = resolver.record(record_id)
    if record is None or record['feature_id'] != feature:
        raise AdvisorError('goal_record_metric_mismatch')
    result = resolver.resolve(record_id, identity)
    if result.status != 'supported':
        raise AdvisorError('goal_context_unavailable')
    return _context(resolver, result.resolved, metric)


def _match(goal, scope, protocol):
    reasons = []
    for name in MetricScope.__dataclass_fields__:
        requested = getattr(goal.scope, name)
        if requested is None or name == 'input_shape' and not requested:
            continue
        actual = getattr(scope, name)
        if actual is None or name == 'input_shape' and not actual:
            reasons.append('scope_missing:' + name)
        elif actual != requested:
            reasons.append('scope_mismatch:' + name)
    # Description is human explanation, never protocol identity or instructions.
    for name in ('protocol_id', 'version', 'sample_policy', 'window', 'evaluator', 'averaging'):
        if getattr(goal.protocol, name) != getattr(protocol, name):
            reasons.append('protocol_mismatch:' + name)
    return tuple(reasons)


def _threshold(goal, resolved):
    # Exact declared unit scales only. No FPS/images/s, fraction/percent or
    # cross-memory-domain conversion. The closed goal/metric map fixes identity.
    scales = _METRICS[goal.metric][0]
    if resolved.reference.unit not in scales:
        raise AdvisorError('goal_native_unit_mismatch')
    actual = Fraction(str(resolved.value)) * Fraction(str(scales[resolved.reference.unit]))
    target = Fraction(str(goal.target)) * Fraction(str(scales[goal.unit]))
    return {'<=': actual <= target, '>=': actual >= target,
            '<': actual < target, '>': actual > target, '==': actual == target}[goal.operator]


def _unknown(goal, support, reasons, evidence=()):
    return GoalEvaluation(goal.goal_id, goal.input_validity, evidence_support=support,
                          observed=ObservedGoal.UNKNOWN, feasibility=Feasibility.UNKNOWN,
                          evidence=evidence, reasons=tuple(reasons))


def evaluate_goal(resolver, goal):
    """Evaluate one goal in exactly one eligible native capture, or abstain."""
    if type(resolver) is not NativeMetricResolver or type(goal) is not UserGoal:
        raise AdvisorError('typed_goal_and_resolver_required')
    if goal.input_validity != InputValidity.VALID:
        return _unknown(goal, EvidenceSupport.INSUFFICIENT_EVIDENCE,
                        tuple('missing:' + name for name in goal.missing_scope_fields))
    if goal.metric not in _SUPPORTED:
        return _unknown(goal, EvidenceSupport.UNSUPPORTED_CAPABILITY, ('unsupported_goal_metric',))
    if goal.value_basis != 'measured':
        return _unknown(goal, EvidenceSupport.UNSUPPORTED_CAPABILITY, ('estimated_goal_assessment_not_supported',))
    if goal.baseline_id is not None:
        return _unknown(goal, EvidenceSupport.UNSUPPORTED_CAPABILITY, ('baseline_comparison_requires_ComparisonContract',))
    feature, identity, _ = _SUPPORTED[goal.metric]
    candidates = []; failures = []; seen = False
    for record_id, _ in resolver._records:
        record = resolver.record(record_id)
        if record['feature_id'] != feature:
            continue
        seen = True
        result = resolver.resolve(record_id, identity)
        if result.status != 'supported':
            failures.extend(result.reasons)
            continue
        resolved = result.resolved
        eligibility = resolver.check_metric(resolved)
        if eligibility.status != EvidenceSupport.SUPPORTED:
            failures.extend(eligibility.reasons)
            continue
        scope, protocol = _context(resolver, resolved, goal.metric)
        mismatch = _match(goal, scope, protocol)
        if mismatch:
            failures.extend(mismatch)
            continue
        candidates.append(resolved)
    if not candidates:
        reasons = tuple(dict.fromkeys(failures)) if seen else ('no_native_record_for_goal_metric',)
        return _unknown(goal, EvidenceSupport.INSUFFICIENT_EVIDENCE, reasons)
    if len(candidates) != 1:
        # No cherry-picking, averaging or conjunction across unpaired captures.
        return _unknown(goal, EvidenceSupport.INSUFFICIENT_EVIDENCE,
                        ('ambiguous_matching_records; refine explicit scope/protocol selectors',),
                        evidence=tuple(m.reference for m in candidates))
    resolved = candidates[0]
    observed = ObservedGoal.MET if _threshold(goal, resolved) else ObservedGoal.MISSED
    bindings = [b.field + '=' + b.authority for b in resolved.reference.bindings if b.authority != 'native']
    reasons = (
        'assessment_scope:single_native_record',
        'observed_value:' + str(resolved.value) + ' ' + resolved.reference.unit,
        'target:' + goal.operator + ' ' + str(goal.target) + ' ' + goal.unit,
        'scope_and_protocol:matched_capture_selectors_only',
        'unproven_associations:' + ','.join(bindings),
        'deployment_acceptance:not_evaluated; declared labels are not native associations',
        'future_feasibility:unknown; one observed outcome does not prove general possibility or impossibility',
    )
    return GoalEvaluation(goal.goal_id, goal.input_validity, EvidenceSupport.SUPPORTED,
                          observed, Feasibility.UNKNOWN, (resolved.reference,), reasons)


def evaluate_goals(resolver, request):
    if type(request) is not AdvisorRequest or type(resolver) is not NativeMetricResolver:
        raise AdvisorError('typed_request_and_resolver_required')
    return tuple(evaluate_goal(resolver, goal) for goal in request.goals)
