"""Deterministic observations and record-local goals; no execution engine."""
import hashlib

from ._json import AdvisorError, canonical
from .contracts import (AdvisorReport, AdvisorRequest, EvidenceSupport, Finding,
                        FindingKind, Priority, RuleVersion)
from .resolver import NativeMetric, NativeMetricResolver, NativeObservation
from .validation import validate_request
from .goals import evaluate_goals as _evaluate_goals, GOAL_RULE_ID, GOAL_RULE_VERSION

RULE_ID = 'native-observation'
RULE_VERSION = '1'
_METRICS = {
    'P01': (NativeMetric.FORWARD_P50, NativeMetric.FORWARD_P95, NativeMetric.FORWARD_P99),
    'P02': (NativeMetric.COMPLETED_IMAGES_PER_SECOND,),
    'P03': (NativeMetric.E2E_P50, NativeMetric.E2E_P95, NativeMetric.E2E_P99),
    'P08': (NativeMetric.PRECISION_MEAN, NativeMetric.PRECISION_THROUGHPUT),
    'P10': (NativeMetric.RSS_CURRENT, NativeMetric.RSS_SAMPLED_PEAK),
    'P11': (NativeMetric.MODEL_LOGICAL_BYTES, NativeMetric.ARTIFACT_BYTES),
    'P26': (NativeMetric.TEST_ACCURACY, NativeMetric.TEST_MACRO_F1),
}


def _identity(*values):
    return hashlib.sha256(canonical(values).encode('utf-8')).hexdigest()


def advise(report, request=None):
    """Explain eligible existing native measurements without running a workload.

    Advanced goals are assessed within single matching native captures only.
    Priority changes order only. Each observation is local to its source capture.
    """
    request = AdvisorRequest() if request is None else request
    if type(request) is not AdvisorRequest:
        raise AdvisorError('typed_request_required')
    resolver = NativeMetricResolver(report)
    return _advise_validated(resolver, request, plan_experiments=True)


def _advise_validated(resolver, request, *, evaluate_goals=True, plan_experiments=False):
    """Internal snapshot reuse; legacy rule reproduction never changes v1 data."""
    source = resolver.report
    observations = []; actions = []; limitations = [
        'Supported means eligible for a local descriptive observation, not complete provenance or joint deployment acceptance.',
        'No independent improvement, causal diagnosis, hardware enforcement or numerical model score is produced.',
        'Original artifact bytes are not automatically reread; integrity is relative to this validated source report.',
        'Resolver v1 supports P01/P02/P03/P08/P10/P11/P26 only; other features remain outside metric-resolution coverage.',
    ]
    active_goals = evaluate_goals and bool(request.goals)
    if active_goals:
        limitations.append('Goal results describe single retained records only; deployment acceptance and future feasibility are not established. Resource-budget assessment/enforcement remains NOT IMPLEMENTED.')
    elif request.goals or request.budgets:
        limitations.append('Advanced goal and resource-budget assessment is NOT IMPLEMENTED (3B.3); only static request validation is included.')
    for record in source['index']['records']:
        rid = record['record_id']; feature = record['feature_id']; n = record['native_payload']
        if feature == 'P05':
            limitations.append('P05 record ' + rid + ': input_kind=' + str(record['scope']['input_kind']) +
                               '; synthetic recipe input must not be labeled a real MNIST image.')
        if record['validation']['status'] != 'native_validated':
            limitations.append(feature + ': ' + record['validation']['status'] + ': ' + record['validation']['reason'])
        if feature not in _METRICS:
            continue
        points = [None]
        if feature == 'P08' and n is not None and n.get('format') == 'hyso.performance.sweep':
            points = [p['point_id'] for p in n.get('raw_evidence', {}).get('points', [])]
        for point_id in points:
            for metric in _METRICS[feature]:
                result = resolver.resolve(rid, metric, point_id=point_id)
                if result.status != 'supported':
                    reason = '; '.join(result.reasons)
                    limitations.append(feature + '/' + metric.value + (('/' + point_id) if point_id else '') + ': ' + reason)
                    # This is a conditional evidence gap, never an execution plan.
                    if reason.startswith(('native_status_not_observable:insufficient_samples', 'unavailable_record:not_run')):
                        actions.append(Finding(
                            _identity(rid, metric.value, point_id, 'gap'), FindingKind.MEASURE_NEXT,
                            feature + ': if this metric is needed, obtain eligible ' + metric.value +
                            ' evidence with its required protocol/sample policy. No run is scheduled or authorized.',
                            limitations=(reason, 'Scope and resource budget must be specified before any future measurement.'),
                            rule_id=RULE_ID, rule_version=RULE_VERSION,
                            evidence_support=EvidenceSupport.INSUFFICIENT_EVIDENCE))
                    continue
                m = result.resolved
                eligibility = resolver.check_reference(m.reference, m.metric)
                if eligibility.status != EvidenceSupport.SUPPORTED:
                    limitations.extend(eligibility.reasons)
                    continue
                context = m.native_context
                display = format(m.value, '.12g') + ' ' + m.reference.unit
                if m.metric == NativeMetric.TEST_ACCURACY:
                    display += ' (' + format(m.value * 100, '.8g') + '%)'
                statement = feature + ' ' + m.metric.value + ' = ' + display + '; origin=' + m.origin + '.'
                statement += ' Measurement context: ' + canonical(context)
                detail = m.limitations + ('Original record scope: ' + m.record_scope_json,
                                          'Native metric protocol: ' + m.protocol_json)
                observations.append((feature, NativeObservation(
                    _identity(rid, metric.value, point_id, RULE_VERSION), FindingKind.OBSERVATION, statement,
                    evidence=(m.reference,), limitations=detail, rule_id=RULE_ID, rule_version=RULE_VERSION,
                    evidence_support=EvidenceSupport.SUPPORTED, verification='not_verified', resolved_metric=m)))
    if any(any(v != 'native' for v in r['bindings'].values()) for r in source['index']['records']):
        limitations.append('Native/declared/missing bindings remain unchanged; identical declarations do not establish compatibility.')
        actions.append(Finding(
            _identity(resolver.source_report_id, 'association-gap'), FindingKind.MEASURE_NEXT,
            'Before a cross-record comparison, establish the required native associations from producer evidence. '
            'Inspect retained evidence first; if absent, a separately authorized capture may be needed. '
            'Repeating declarations cannot repair missing provenance.',
            limitations=('No capture is started; compare_evidence remains fail-closed.',),
            rule_id=RULE_ID, rule_version=RULE_VERSION, evidence_support=EvidenceSupport.INSUFFICIENT_EVIDENCE))
    for slot in source['analysis']['feature_results']:
        if slot['status'] in ('missing', 'unsupported', 'not_run', 'invalid', 'unknown'):
            limitations.append(slot['feature_id'] + ' coverage=' + slot['status'] + ': ' + slot['reason'])
    groups = {
        Priority.SPEED: ('P01', 'P02', 'P03', 'P08', 'P10', 'P11', 'P26'),
        Priority.MEMORY: ('P10', 'P11', 'P01', 'P02', 'P03', 'P08', 'P26'),
        Priority.PRESERVE_ACCURACY: ('P26', 'P01', 'P02', 'P03', 'P08', 'P10', 'P11'),
    }
    order = groups.get(request.priority)
    if order:
        observations.sort(key=lambda item: order.index(item[0]))
    risks = list(source['analysis']['risks'])
    risks.extend(name + '=' + value for name, value in source['analysis']['claims'].items())
    validation = validate_request(request)
    evaluations = validation.goals
    rule_versions = (RuleVersion(RULE_ID, RULE_VERSION),)
    if active_goals:
        evaluations = _evaluate_goals(resolver, request)
        rule_versions += (RuleVersion(GOAL_RULE_ID, GOAL_RULE_VERSION),)
        goals_by_id = {g.goal_id: g for g in request.goals}
        for evaluation in evaluations:
            if evaluation.evidence_support != EvidenceSupport.SUPPORTED:
                actions.append(Finding(
                    _identity(goals_by_id[evaluation.goal_id].content_sha256, evaluation.goal_id, GOAL_RULE_VERSION), FindingKind.MEASURE_NEXT,
                    'Goal ' + evaluation.goal_id + ': clarify the requested scope/protocol and inspect retained native evidence. '
                    'If the required capability or evidence is absent, obtain an explicitly scoped, separately authorized capture. '
                    'No measurement is started.', evidence=evaluation.evidence, limitations=evaluation.reasons,
                    rule_id=GOAL_RULE_ID, rule_version=GOAL_RULE_VERSION,
                    evidence_support=EvidenceSupport.INSUFFICIENT_EVIDENCE))
        if validation.conflicting_goal_ids:
            limitations.append('Hard goal intervals are logically inconsistent for the same scope/protocol. Individual observed outcomes are not joint acceptance.')
    result = AdvisorReport(
        resolver.source_report_id, resolver.source_report_sha256, request,
        findings=tuple(f for _, f in observations) + tuple(actions),
        goal_evaluations=evaluations, request_validation=validation,
        evidence_limitations=tuple(dict.fromkeys(limitations)), risks=tuple(dict.fromkeys(risks)),
        rule_versions=rule_versions,
        evidence_status=EvidenceSupport.SUPPORTED if observations else EvidenceSupport.INSUFFICIENT_EVIDENCE,
        evaluation_status='goals_evaluated' if active_goals else 'observations_only')
    if plan_experiments:
        from .planning import _add_plans
        return _add_plans(resolver, result)
    return result


def validate_advisor_report(result, source):
    """Reproduce engine output against the actual source, not merely JSON hashes."""
    if type(result) is not AdvisorReport:
        raise AdvisorError('typed_advisor_report_required')
    if result.evaluation_status not in ('observations_only', 'goals_evaluated', 'planning_only'):
        raise AdvisorError('not_an_observation_engine_report')
    expected = _advise_validated(NativeMetricResolver(source), result.request,
                                evaluate_goals=result.evaluation_status != 'observations_only',
                                plan_experiments=result.evaluation_status == 'planning_only')
    if expected.to_dict() != result.to_dict():
        raise AdvisorError('advisor_derivation_mismatch')
    return True
