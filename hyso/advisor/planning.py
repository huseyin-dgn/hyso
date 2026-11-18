"""Versioned, read-only decision plans. No execution or comparative certification."""
from dataclasses import dataclass, replace
import hashlib

from ._json import AdvisorError, Contract, canonical, register
from .contracts import (EvidenceSupport, Finding, FindingKind, Metric, Priority,
                        RuleVersion, _choice, _sha, _strings, _text)
from .resolver import NativeMetric as M, NativeObservation

PLANNER_RULES = (
    'plan.latency-stage', 'plan.batch-throughput', 'plan.precision-quality',
    'plan.memory-domain', 'plan.accuracy-preservation', 'plan.hard-goal-consistency',
)
PLANNER_VERSION = '1'
_STAGES = {'retained_review': 0, 'association_validation': 1, 'new_measurement': 2, 'conditional_experiment': 3}


@register
@dataclass(frozen=True)
class DecisionPlan(Contract):
    domain: str
    stage: str
    decision: str
    rationale: str
    change_parameter: str | None = None
    changed_variables: tuple = ()
    fixed_conditions: tuple = ()
    remeasure_metrics: tuple = ()
    missing_evidence: tuple = ()
    risks: tuple = ()
    acceptance_conditions: tuple = ()
    goal_ids: tuple = ()
    budget_ids: tuple = ()
    capture_records: tuple = ()
    authorization: str = 'not_authorized'
    readiness: str = 'blocked'
    cost: str = 'unknown'
    expected_effect: str = 'unknown'
    verified_effect: str = 'not_measured'

    def __post_init__(self):
        _choice(self, 'domain', ('latency', 'throughput', 'precision', 'memory', 'quality', 'provenance'))
        _choice(self, 'stage', tuple(_STAGES))
        for name in ('decision', 'rationale'):
            _text(getattr(self, name), name)
        _text(self.change_parameter, 'change_parameter', optional=True)
        for name in ('changed_variables', 'fixed_conditions', 'remeasure_metrics', 'missing_evidence',
                     'risks', 'acceptance_conditions', 'goal_ids', 'budget_ids', 'capture_records'):
            _strings(self, name)
        for name, allowed in (('authorization', ('not_authorized',)), ('readiness', ('blocked',)),
                              ('cost', ('unknown',)), ('expected_effect', ('unknown',)), ('verified_effect', ('not_measured',))):
            _choice(self, name, allowed)
        for record_id in self.capture_records:
            _sha(record_id, 'capture_record_id')
        if self.stage == 'conditional_experiment':
            if (self.change_parameter != 'batch_size' or self.changed_variables != ('input.batch_size',)
                    or not self.fixed_conditions or not self.remeasure_metrics or not self.missing_evidence
                    or not self.risks or not self.acceptance_conditions or not self.capture_records):
                raise AdvisorError('incomplete_batch_experiment_plan')
        elif self.change_parameter is not None or self.changed_variables:
            raise AdvisorError('evidence_review_cannot_mutate_configuration')


@register
@dataclass(frozen=True)
class PlannedFinding(Finding):
    plan: DecisionPlan | None = None

    def __post_init__(self):
        super().__post_init__()
        if type(self.plan) is not DecisionPlan or self.verification != 'not_verified':
            raise AdvisorError('typed_unverified_plan_required')
        expected = FindingKind.EXPERIMENT_CANDIDATE if self.plan.stage == 'conditional_experiment' else FindingKind.MEASURE_NEXT
        if self.kind != expected:
            raise AdvisorError('plan_kind_mismatch')
        if self.kind == FindingKind.EXPERIMENT_CANDIDATE and (
                not self.evidence or self.evidence_support != EvidenceSupport.SUPPORTED):
            raise AdvisorError('experiment_requires_local_native_basis')


def _id(*items):
    return hashlib.sha256(canonical(items).encode()).hexdigest()


def _add_plans(resolver, base):
    """Internal: base was generated from this validated immutable resolver."""
    if base.source_report_id != resolver.source_report_id or base.source_report_sha256 != resolver.source_report_sha256:
        raise AdvisorError('planner_source_mismatch')
    observations = [f for f in base.findings if type(f) is NativeObservation]
    metrics = []
    for f in observations:
        if resolver.check_metric(f.resolved_metric).status != EvidenceSupport.SUPPORTED:
            raise AdvisorError('planner_metric_not_eligible')
        metrics.append(f.resolved_metric)
    request = base.request
    evaluations = {g.goal_id: g for g in base.goal_evaluations}
    areas = {Metric.RESIDENT_FORWARD_P95: 'latency', Metric.E2E_P95: 'latency',
             Metric.IMAGE_THROUGHPUT: 'throughput', Metric.PROCESS_RSS_PEAK: 'memory',
             Metric.LOGICAL_TENSOR_BYTES: 'memory', Metric.ACCURACY: 'quality'}
    missed = [g for g in request.goals if evaluations[g.goal_id].observed == 'observed_missed'
              and evaluations[g.goal_id].evidence_support == EvidenceSupport.SUPPORTED]
    interests = {areas[g.metric] for g in missed if g.metric in areas}
    if not request.goals:
        interests = {'latency', 'throughput', 'precision', 'memory', 'quality'}
    elif interests & {'latency', 'throughput', 'quality'}:
        interests.add('precision')
    plans = []; limits = []
    def select(*identities):
        return [m for m in metrics if m.metric in identities]
    def goal_ids(domain, refs=()):
        ids = {r.record_id for r in refs}
        return tuple(g.goal_id for g in missed if areas.get(g.metric) == domain and
                     (not ids or any(r.record_id in ids for r in evaluations[g.goal_id].evidence)))
    def local_for(domain, values):
        if not request.goals:
            return values
        ids = {r.record_id for g in missed if areas.get(g.metric) == domain for r in evaluations[g.goal_id].evidence}
        return [m for m in values if m.reference.record_id in ids]
    def emit(rule, domain, decision, rationale, selected=(), *, stage='retained_review', missing=(),
             fixed=(), remeasure=(), risks=(), acceptance=(), records=(), candidate=False, related=None):
        refs = tuple(m.reference for m in selected)
        related = goal_ids(domain, refs) if related is None else related
        record_ids = tuple(dict.fromkeys(tuple(r.record_id for r in refs) + tuple(records)))
        plan = DecisionPlan(domain, stage, decision, rationale,
            change_parameter='batch_size' if candidate else None,
            changed_variables=('input.batch_size',) if candidate else (), fixed_conditions=tuple(fixed),
            remeasure_metrics=tuple(remeasure), missing_evidence=tuple(missing), risks=tuple(risks),
            acceptance_conditions=tuple(acceptance), goal_ids=related,
            budget_ids=tuple(b.budget_id for b in request.budgets), capture_records=record_ids)
        kind = FindingKind.EXPERIMENT_CANDIDATE if candidate else FindingKind.MEASURE_NEXT
        plans.append(PlannedFinding(
            _id(resolver.source_report_id, rule, PLANNER_VERSION, plan.to_dict(), [r.to_dict() for r in refs]),
            kind, decision, evidence=refs, limitations=(rationale, 'No capture, mutation or execution is authorized.'),
            rule_id=rule, rule_version=PLANNER_VERSION,
            evidence_support=EvidenceSupport.SUPPORTED if candidate else EvidenceSupport.INSUFFICIENT_EVIDENCE,
            verification='not_verified', plan=plan))
    conflict = base.request_validation and base.request_validation.conflicting_goal_ids
    if conflict:
        emit('plan.hard-goal-consistency', 'provenance', 'Resolve contradictory hard goals before selecting experiments.',
             'No experiment can satisfy the empty interval as written; individual historical goal outcomes remain unchanged.',
             missing=('A logically consistent explicit hard-goal set',), related=tuple(conflict))
        interests = set()
    if request.goals:
        for g in request.goals:
            e = evaluations[g.goal_id]
            if e.observed == 'observed_met':
                limits.append('Goal ' + g.goal_id + ' is met only in its retained record; no new target is invented.')
            elif e.observed == 'unknown':
                limits.append('Goal ' + g.goal_id + ': no optimization prescription while its evidence/capability is unknown; use its existing evidence-clarification action.')
    if 'latency' in interests:
        local = local_for('latency', select(M.E2E_P95)) or local_for('latency', select(M.FORWARD_P95))
        if local:
            emit('plan.latency-stage', 'latency', 'Inspect retained pipeline stages before choosing a latency experiment.',
                 'Resident forward and read-to-post E2E are different captures. Their difference is not a diagnosed cause.', local,
                 missing=('Stage-boundary/clock attribution from retained P03/P04; if absent, a separately authorized stage profile',),
                 fixed=('Keep every source capture separate; do not subtract P01 from P03.',),
                 remeasure=('Only if retained stage evidence is insufficient: scoped source/preprocess/inference/postprocess timings',),
                 risks=('Instrumentation overhead and unmatched clocks can change observed timing.',))
    if 'throughput' in interests:
        for m in select(M.COMPLETED_IMAGES_PER_SECOND):
            record = resolver.record(m.reference.record_id)
            env = record['native_payload']['provenance']['environment']
            required = ('device', 'precision', 'batch', 'input_shape', 'model_state_sha256', 'protocol_id', 'protocol_sha256')
            binding = {b.field:b.authority for b in m.reference.bindings}
            missing = tuple('native association:' + k for k in required if binding.get(k) != 'native')
            if type(env.get('backend')) is not str or not env['backend'].strip():
                missing += ('Explicit retained backend identity',)
            if m.native_context.get('device') != 'cpu' or record['scope']['precision'] not in ('fp32', 'fp64'):
                missing += ('CPU FP32/FP64 planning capability; risky GPU paths are outside planner v1',)
            linked = goal_ids('throughput', (m.reference,))
            if request.goals and not linked:
                continue
            if missing:
                emit('plan.batch-throughput', 'throughput', 'Review throughput recipe and native associations before a batch experiment.',
                     'A completed images/s observation alone does not establish backend, batch-change capability or safe resource limits.', (m,),
                     missing=missing + ('Inspect retained P06 sweep before selecting any new batch points; resolver v1 does not resolve P06.',), related=linked)
                continue
            hard = tuple('Hard goal ' + g.goal_id + ': ' + g.metric.value + ' ' + g.operator + ' ' + str(g.target) + ' ' + g.unit for g in request.goals if g.constraint == 'hard')
            emit('plan.batch-throughput', 'throughput', 'Inspect retained P06 evidence before choosing any new batch points.',
                 'P06 is not resolved by resolver v1. Retained sweep evidence may answer the batch question without rerunning it.', (m,),
                 missing=('Retained P06 point capability, scopes and failure outcomes; do not invent sweep results.',), related=linked)
            emit('plan.batch-throughput', 'throughput', 'Consider a bounded, single-variable batch-size experiment after prerequisites are resolved.',
                 'This native CPU throughput capture supplies a baseline. Changing batch is a question to test, not a speedup recommendation.', (m,),
                 stage='conditional_experiment', candidate=True, related=linked,
                 missing=('Review retained P06 sweep first; P06 is outside resolver v1.',
                          'Caller-selected finite batch points and explicit experiment resource/time/trial budget',
                          'Native model/input/workload bindings and candidate execution capability',
                          'Linked baseline/candidate memory and quality acceptance evidence',
                          'Future ComparisonContract and independent validation; no verified comparison is currently available'),
                 fixed=('Baseline model-state SHA: ' + record['scope']['model_state_sha256'],
                        'Baseline device/precision: ' + record['scope']['device'] + '/' + record['scope']['precision'],
                        'Recorded backend: ' + env['backend'],
                        'Hold model, dtype, preprocessing, input semantics, backend and timing method fixed; only batch may vary.',
                        'Do not silently resize input, change precision or relax any hard quality constraint.'),
                 remeasure=('Completed images/s and completion/pending ledger', 'Resident P95 at each explicit batch',
                            'Process RSS in a matching capture', 'Quality on a linked baseline/candidate cohort',
                            'Bounded stability/failure status; not 7200-second acceptance'),
                 risks=('Larger batches may worsen latency, RSS, OOM risk, runtime and short-run stability.',
                        'No speedup or memory reduction is guaranteed; numerical behavior/quality must be checked.',
                        'Deployment budgets are not experiment limits; actual enforcement is unknown.'),
                 acceptance=hard + ('Caller must specify acceptance thresholds; none are invented.',
                    'Meet every hard constraint on linked evidence; a local observed_met is not joint deployment success.',
                    'Use separate confirmation captures after exploration; do not certify effect without the future comparison gate.'))
    if 'precision' in interests:
        points = select(M.PRECISION_MEAN)
        by_record = {}
        for m in points:
            by_record.setdefault(m.reference.record_id, []).append(m)
        for rid, local in by_record.items():
            raw = resolver.record(rid)['native_payload']
            unavailable = tuple(p['precision'] + ':' + p['status'] for p in raw['raw_evidence']['points'] if p['status'] != 'measured')
            emit('plan.precision-quality', 'precision', 'Inspect retained precision capability and linked quality evidence before considering a precision change.',
                 'Each P08 point retains its own batch/precision and unverified capture. BF16 is not automatically faster; no speedup is certified.', local,
                 missing=('Native point-to-quality/cohort and goal-capture bindings; inspect retained P28 before any new measurement.',) + unavailable,
                 fixed=('Do not use the sweep default precision as the point precision.',
                        'Unsupported INT8 remains unsupported; no precision candidate is declared runnable.'),
                 remeasure=('Only after capability/authorization: matched precision-quality and timing captures',),
                 risks=('Precision changes can alter accuracy and supported operators; no default accuracy-drop tolerance exists.',),
                 related=tuple(g.goal_id for g in missed if areas.get(g.metric) in ('latency', 'throughput', 'quality')))
    if 'memory' in interests:
        local = local_for('memory', select(M.RSS_SAMPLED_PEAK, M.MODEL_LOGICAL_BYTES))
        if local:
            emit('plan.memory-domain', 'memory', 'Review process memory and tensor accounting separately before prescribing memory changes.',
                 'Process RSS includes runtime costs; logical parameter/buffer bytes are not RSS or VRAM. A mechanism has not been diagnosed.', local,
                 missing=('Window/ownership/association evidence for a specific memory mechanism; inspect retained P10/P11 first.',),
                 fixed=('Keep source scopes separate; never subtract logical bytes from RSS to claim an attributable saving.',),
                 remeasure=('Only if retained evidence is inadequate: scoped allocation/lifetime/RSS profile with an explicit experiment budget',),
                 risks=('Sampled peaks may miss transients; allocation growth does not prove a causal leak.',))
    if 'quality' in interests:
        local = local_for('quality', select(M.TEST_ACCURACY))
        if local:
            emit('plan.accuracy-preservation', 'quality', 'Validate baseline/candidate cohort bindings before claiming accuracy preservation.',
                 'P26 is a recorded quality observation, not a guarantee of zero accuracy loss after a change.', local,
                 stage='association_validation',
                 missing=('Linked model/checkpoint/cohort/preprocessing and explicit caller quality acceptance criteria',),
                 fixed=('Preserve split/evaluator/class mapping; priority does not create a numerical accuracy tolerance.',),
                 acceptance=('Every explicit hard quality bound must pass on linked evidence before overall candidate acceptance.',),
                 risks=('Repeated selection on the evaluation cohort can bias acceptance; independent confirmation is still required.',))
    # Dependency order first, priority only breaks ties within that order.
    domain_order = {
        Priority.MEMORY: ('memory', 'quality', 'latency', 'throughput', 'precision', 'provenance'),
        Priority.PRESERVE_ACCURACY: ('quality', 'precision', 'memory', 'latency', 'throughput', 'provenance'),
    }.get(request.priority, ('latency', 'throughput', 'precision', 'memory', 'quality', 'provenance'))
    association_id = _id(resolver.source_report_id, 'association-gap')
    # Legacy _identity hashes the tuple identically; existing findings stay intact.
    def order(f):
        if type(f) is PlannedFinding:
            return (_STAGES[f.plan.stage], domain_order.index(f.plan.domain), f.rule_id, f.finding_id)
        if f.rule_id == 'record-local-goal':
            return (0, 5, f.rule_id, f.finding_id)
        return (1 if f.finding_id == association_id else 2, 5, f.rule_id or '', f.finding_id)
    legacy_actions = [f for f in base.findings if f.kind != FindingKind.OBSERVATION]
    ordered = sorted(legacy_actions + plans, key=order)
    return replace(base, findings=tuple(observations) + tuple(ordered), evaluation_status='planning_only',
        evidence_limitations=base.evidence_limitations + tuple(limits) + (
            'Planner v1 proposes decisions only. All plans are blocked/not_authorized; expected effect and cost are unknown.',
            'No diagnosis, verified improvement, automatic goal, configuration change or execution is produced.'),
        rule_versions=base.rule_versions + tuple(RuleVersion(r, PLANNER_VERSION) for r in PLANNER_RULES))
