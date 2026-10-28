"""Stage 3B.1: immutable data contracts, not an Advisor implementation."""
from dataclasses import dataclass, field
from enum import Enum
import math
import re

from ._json import AdvisorError, Contract, register


class Priority(str, Enum):
    SPEED = 'speed'
    MEMORY = 'memory'
    PRESERVE_ACCURACY = 'preserve_accuracy'
    BALANCED = 'balanced'


class Metric(str, Enum):
    RESIDENT_FORWARD_P95 = 'resident_forward_p95'
    E2E_P95 = 'e2e_p95'
    IMAGE_THROUGHPUT = 'image_throughput'
    CAMERA_FPS = 'camera_fps'
    TRAINING_THROUGHPUT = 'training_throughput'
    TRAINING_STEP_TIME = 'training_step_time'
    ACCURACY = 'accuracy'
    ACCURACY_DROP = 'accuracy_drop'
    PROCESS_RSS_PEAK = 'process_rss_peak'
    CUDA_ALLOCATED_PEAK = 'cuda_allocated_peak'
    CUDA_RESERVED_PEAK = 'cuda_reserved_peak'
    LOGICAL_TENSOR_BYTES = 'logical_tensor_bytes'
    ENERGY_PER_INFERENCE = 'energy_per_inference'
    POWER = 'power'


class FindingKind(str, Enum):
    OBSERVATION = 'OBSERVATION'
    HYPOTHESIS = 'HYPOTHESIS'
    DIAGNOSIS = 'DIAGNOSIS'
    MEASURE_NEXT = 'MEASURE_NEXT'
    EXPERIMENT_CANDIDATE = 'EXPERIMENT_CANDIDATE'
    VERIFIED_IMPROVEMENT = 'VERIFIED_IMPROVEMENT'


class InputValidity(str, Enum):
    VALID = 'valid'
    INVALID = 'invalid'
    NEEDS_SCOPE = 'needs_scope'


class EvidenceSupport(str, Enum):
    SUPPORTED = 'supported'
    INSUFFICIENT_EVIDENCE = 'insufficient_evidence'
    UNSUPPORTED_CAPABILITY = 'unsupported_capability'


class ObservedGoal(str, Enum):
    MET = 'observed_met'
    MISSED = 'observed_missed'
    UNKNOWN = 'unknown'


class Feasibility(str, Enum):
    DEMONSTRATED = 'demonstrated_in_scope'
    NOT_DEMONSTRATED = 'not_demonstrated'
    UNKNOWN = 'unknown'


class Consistency(str, Enum):
    NO_CONFLICT = 'no_conflict_identified'
    POTENTIAL = 'potentially_conflicting'
    INCONSISTENT = 'logically_inconsistent'


def _text(value, name, optional=False):
    if value is None and optional:
        return
    if type(value) is not str or not value.strip() or len(value) > 4096:
        raise AdvisorError('invalid_' + name)
    try:
        value.encode('utf-8')
    except UnicodeError as exc:
        raise AdvisorError('invalid_' + name) from exc


def _choice(obj, name, choices):
    value = getattr(obj, name)
    if type(value) is not str or value not in choices:
        raise AdvisorError('invalid_' + name)


def _enum(obj, name, enum, optional=False):
    value = getattr(obj, name)
    if value is None and optional:
        return
    try:
        object.__setattr__(obj, name, enum(value))
    except (ValueError, TypeError) as exc:
        raise AdvisorError('invalid_' + name) from exc


def _number(value, name, positive=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or (value <= 0 if positive else value < 0):
        raise AdvisorError('invalid_' + name)


def _sha(value, name):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise AdvisorError('invalid_' + name)


def _sequence(obj, name, cls, unique=None):
    values = getattr(obj, name)
    allowed = cls if type(cls) is tuple else (cls,)
    if type(values) not in (tuple, list) or len(values) > 1024 or any(type(v) not in allowed for v in values):
        raise AdvisorError('invalid_' + name)
    values = tuple(values)
    if unique and len({getattr(v, unique) for v in values}) != len(values):
        raise AdvisorError('duplicate_' + name)
    object.__setattr__(obj, name, values)


def _strings(obj, name):
    _sequence(obj, name, str)
    for value in getattr(obj, name):
        _text(value, name)


@register
@dataclass(frozen=True)
class MetricScope(Contract):
    phase: str
    device: str | None = None
    backend: str | None = None
    precision: str | None = None
    batch: int | None = None
    input_shape: tuple = ()
    workload_id: str | None = None
    dataset_id: str | None = None
    split: str | None = None
    preprocessing_id: str | None = None
    clock_id: str | None = None
    model_id: str | None = None

    def __post_init__(self):
        _choice(self, 'phase', ('inference', 'training', 'evaluation', 'process', 'static'))
        for name in ('device', 'backend', 'precision', 'workload_id', 'dataset_id',
                     'split', 'preprocessing_id', 'clock_id', 'model_id'):
            _text(getattr(self, name), name, optional=True)
        if self.batch is not None and (type(self.batch) is not int or self.batch < 1):
            raise AdvisorError('invalid_batch')
        _sequence(self, 'input_shape', int)
        if any(n < 1 for n in self.input_shape):
            raise AdvisorError('invalid_input_shape')
        if self.input_shape and self.batch is not None and self.input_shape[0] != self.batch:
            raise AdvisorError('batch_shape_mismatch')


@register
@dataclass(frozen=True)
class EvaluationProtocol(Contract):
    protocol_id: str
    version: int
    description: str
    sample_policy: str | None = None
    window: str | None = None
    evaluator: str | None = None
    averaging: str | None = None

    def __post_init__(self):
        for name in ('protocol_id', 'description'):
            _text(getattr(self, name), name)
        if type(self.version) is not int or self.version < 1:
            raise AdvisorError('invalid_protocol_version')
        for name in ('sample_policy', 'window', 'evaluator', 'averaging'):
            _text(getattr(self, name), name, optional=True)


# No unit guessing or FPS/images/s conversion. Memory scales are exact.
_MEMORY_UNITS = {'byte': 1, 'GB': 10**9, 'GiB': 2**30}
_TIME_UNITS = {'s': 1, 'ms': 0.001}
_METRICS = {
    Metric.RESIDENT_FORWARD_P95: (_TIME_UNITS, 'inference'),
    Metric.E2E_P95: (_TIME_UNITS, 'inference'),
    Metric.IMAGE_THROUGHPUT: ({'images/s': 1}, 'inference'),
    Metric.CAMERA_FPS: ({'frames/s': 1}, 'inference'),
    Metric.TRAINING_THROUGHPUT: ({'images/s': 1}, 'training'),
    Metric.TRAINING_STEP_TIME: (_TIME_UNITS, 'training'),
    Metric.ACCURACY: ({'fraction': 1}, 'evaluation'),
    Metric.ACCURACY_DROP: ({'percentage_point': 1}, 'evaluation'),
    Metric.PROCESS_RSS_PEAK: (_MEMORY_UNITS, None),
    Metric.CUDA_ALLOCATED_PEAK: (_MEMORY_UNITS, None),
    Metric.CUDA_RESERVED_PEAK: (_MEMORY_UNITS, None),
    Metric.LOGICAL_TENSOR_BYTES: (_MEMORY_UNITS, None),
    Metric.ENERGY_PER_INFERENCE: ({'J/inference': 1}, 'inference'),
    Metric.POWER: ({'W': 1}, None),
}


@register
@dataclass(frozen=True)
class UserGoal(Contract):
    goal_id: str
    metric: Metric
    operator: str
    target: float
    unit: str
    constraint: str = 'hard'
    scope: MetricScope | None = None
    protocol: EvaluationProtocol | None = None
    value_basis: str = 'measured'
    baseline_id: str | None = None

    def __post_init__(self):
        _text(self.goal_id, 'goal_id')
        _enum(self, 'metric', Metric)
        _choice(self, 'operator', ('<=', '>=', '<', '>', '=='))
        _choice(self, 'constraint', ('hard', 'soft'))
        _choice(self, 'value_basis', ('measured', 'estimated'))
        _number(self.target, 'target')
        units, phase = _METRICS[self.metric]
        if type(self.unit) is not str or self.unit not in units:
            raise AdvisorError('metric_unit_mismatch')
        if self.metric == Metric.ACCURACY and self.target > 1:
            raise AdvisorError('accuracy_fraction_out_of_range')
        if self.metric == Metric.ACCURACY_DROP and self.target > 100:
            raise AdvisorError('accuracy_drop_out_of_range')
        if self.scope is not None and type(self.scope) is not MetricScope:
            raise AdvisorError('typed_scope_required')
        if self.protocol is not None and type(self.protocol) is not EvaluationProtocol:
            raise AdvisorError('typed_protocol_required')
        if self.scope is not None and phase and self.scope.phase != phase:
            raise AdvisorError('metric_phase_mismatch')
        _text(self.baseline_id, 'baseline_id', optional=True)
        if self.metric == Metric.ACCURACY_DROP and self.baseline_id is None:
            raise AdvisorError('baseline_required')

    @property
    def input_validity(self):
        return InputValidity.NEEDS_SCOPE if self.missing_scope_fields else InputValidity.VALID

    @property
    def missing_scope_fields(self):
        """Request completeness only; never native evidence eligibility."""
        if self.scope is None or self.protocol is None:
            return tuple(name for name in ('scope', 'protocol') if getattr(self, name) is None)
        required = ['workload_id']
        if self.metric not in (Metric.CAMERA_FPS, Metric.POWER):
            required.append('model_id')
        if self.metric in (Metric.ACCURACY, Metric.ACCURACY_DROP):
            required += ['dataset_id', 'split', 'preprocessing_id']
            protocol_fields = ('sample_policy', 'evaluator', 'averaging')
        else:
            required.append('device')
            protocol_fields = ('sample_policy',)
        if self.metric in (Metric.RESIDENT_FORWARD_P95, Metric.IMAGE_THROUGHPUT,
                           Metric.TRAINING_THROUGHPUT, Metric.TRAINING_STEP_TIME):
            required += ['backend', 'precision', 'batch', 'input_shape']
        missing = ['scope.' + name for name in required if not getattr(self.scope, name)]
        missing += ['protocol.' + name for name in protocol_fields if not getattr(self.protocol, name)]
        return tuple(missing)


@register
@dataclass(frozen=True)
class ResourceBudget(Contract):
    budget_id: str
    purpose: str
    domain: str
    limit: float
    unit: str
    allowed_devices: tuple = ()
    allowed_backends: tuple = ()
    allowed_precisions: tuple = ()
    window_seconds: float | None = None
    enforcement: str = 'not_evaluated'
    applicability: str = 'not_evaluated'

    def __post_init__(self):
        _text(self.budget_id, 'budget_id')
        _choice(self, 'purpose', ('deployment', 'experiment'))
        domains = {'process_rss': _MEMORY_UNITS, 'cuda_allocated': _MEMORY_UNITS,
                   'cuda_reserved': _MEMORY_UNITS, 'device_memory': _MEMORY_UNITS,
                   'logical_tensor_bytes': _MEMORY_UNITS, 'wall_time': {'s': 1},
                   'energy': {'J': 1}, 'power': {'W': 1}, 'trial_count': {'count': 1}}
        if type(self.domain) is not str or self.domain not in domains:
            raise AdvisorError('invalid_resource_domain')
        if type(self.unit) is not str or self.unit not in domains[self.domain]:
            raise AdvisorError('resource_unit_mismatch')
        _number(self.limit, 'limit', positive=True)
        if self.domain == 'trial_count' and type(self.limit) is not int:
            raise AdvisorError('integer_trial_count_required')
        for name in ('allowed_devices', 'allowed_backends', 'allowed_precisions'):
            _strings(self, name)
        if self.window_seconds is not None:
            _number(self.window_seconds, 'window_seconds', positive=True)
        # 3B.1 cannot attest runtime enforcement, even on deserialization.
        _choice(self, 'enforcement', ('not_evaluated', 'unavailable', 'advisory_only'))
        _choice(self, 'applicability', ('not_evaluated', 'unavailable', 'unsupported_capability'))


@register
@dataclass(frozen=True)
class AdvisorRequest(Contract):
    goals: tuple = ()
    budgets: tuple = ()
    priority: Priority | None = None

    def __post_init__(self):
        _sequence(self, 'goals', UserGoal, 'goal_id')
        _sequence(self, 'budgets', ResourceBudget, 'budget_id')
        _enum(self, 'priority', Priority, optional=True)

    @property
    def request_id(self):
        return 'sha256:' + self.content_sha256

    @property
    def mode(self):
        return 'advanced' if self.goals or self.budgets else 'priority' if self.priority else 'automatic'


@register
@dataclass(frozen=True)
class EvidenceBinding(Contract):
    field: str
    authority: str

    def __post_init__(self):
        _text(self.field, 'binding_field')
        _choice(self, 'authority', ('native', 'declared', 'missing'))


@register
@dataclass(frozen=True)
class EvidenceRef(Contract):
    report_id: str
    record_id: str
    metric_path: str
    artifact_sha256: str
    payload_sha256: str
    domain: str
    unit: str
    status: str
    bindings: tuple
    sample_ref: str | None = None
    point_ref: str | None = None
    protocol_ref: str | None = None

    def __post_init__(self):
        for name in ('report_id', 'record_id', 'artifact_sha256', 'payload_sha256'):
            _sha(getattr(self, name), name)
        for name in ('metric_path', 'domain', 'unit'):
            _text(getattr(self, name), name)
        # This is a reference path, never an expression or executable attribute lookup.
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+))*', self.metric_path) is None:
            raise AdvisorError('invalid_metric_path')
        _choice(self, 'status', ('measured', 'derived', 'estimated', 'unknown', 'unsupported',
                                'not_run', 'invalid', 'insufficient_samples'))
        _sequence(self, 'bindings', EvidenceBinding, 'field')
        for name in ('sample_ref', 'point_ref', 'protocol_ref'):
            _text(getattr(self, name), name, optional=True)


@register
@dataclass(frozen=True)
class Finding(Contract):
    finding_id: str
    kind: FindingKind
    statement: str
    evidence: tuple = ()
    limitations: tuple = ()
    rule_id: str | None = None
    rule_version: str | None = None
    evidence_support: EvidenceSupport | None = None
    verification: str = 'not_evaluated'

    def __post_init__(self):
        _text(self.finding_id, 'finding_id')
        _text(self.statement, 'statement')
        _enum(self, 'kind', FindingKind)
        _sequence(self, 'evidence', EvidenceRef)
        _strings(self, 'limitations')
        _enum(self, 'evidence_support', EvidenceSupport, optional=True)
        for name in ('rule_id', 'rule_version'):
            _text(getattr(self, name), name, optional=True)
        if (self.rule_id is None) != (self.rule_version is None):
            raise AdvisorError('rule_identity_incomplete')
        _choice(self, 'verification', ('not_evaluated', 'unavailable', 'not_verified'))
        if self.kind == FindingKind.VERIFIED_IMPROVEMENT:
            raise AdvisorError('comparison_contract_unavailable')
        if self.evidence_support == EvidenceSupport.SUPPORTED and not self.evidence:
            raise AdvisorError('support_requires_references')


@register
@dataclass(frozen=True)
class GoalEvaluation(Contract):
    goal_id: str
    input_validity: InputValidity
    evidence_support: EvidenceSupport | None = None
    observed: ObservedGoal = ObservedGoal.UNKNOWN
    feasibility: Feasibility = Feasibility.UNKNOWN
    evidence: tuple = ()
    reasons: tuple = ()

    def __post_init__(self):
        _text(self.goal_id, 'goal_id')
        _enum(self, 'input_validity', InputValidity)
        _enum(self, 'evidence_support', EvidenceSupport, optional=True)
        _enum(self, 'observed', ObservedGoal)
        _enum(self, 'feasibility', Feasibility)
        _sequence(self, 'evidence', EvidenceRef)
        _strings(self, 'reasons')
        if (self.evidence_support == EvidenceSupport.SUPPORTED or self.observed != ObservedGoal.UNKNOWN
                or self.feasibility == Feasibility.DEMONSTRATED) and not self.evidence:
            raise AdvisorError('assessment_requires_references')
        if (self.observed != ObservedGoal.UNKNOWN or self.feasibility == Feasibility.DEMONSTRATED) and (
                self.input_validity != InputValidity.VALID or self.evidence_support != EvidenceSupport.SUPPORTED):
            raise AdvisorError('assessment_requires_valid_supported_goal')


@register
@dataclass(frozen=True)
class RequestValidation(Contract):
    goals: tuple
    consistency: Consistency
    conflicting_goal_ids: tuple = ()
    reasons: tuple = ()

    def __post_init__(self):
        _sequence(self, 'goals', GoalEvaluation, 'goal_id')
        _enum(self, 'consistency', Consistency)
        _strings(self, 'conflicting_goal_ids')
        _strings(self, 'reasons')
        ids = {g.goal_id for g in self.goals}
        if (not set(self.conflicting_goal_ids) <= ids or
                len(set(self.conflicting_goal_ids)) != len(self.conflicting_goal_ids)):
            raise AdvisorError('invalid_conflicting_goal_ids')
        if bool(self.conflicting_goal_ids) != (self.consistency == Consistency.INCONSISTENT):
            raise AdvisorError('consistency_status_mismatch')


@register
@dataclass(frozen=True)
class RuleVersion(Contract):
    rule_id: str
    version: str

    def __post_init__(self):
        _text(self.rule_id, 'rule_id')
        _text(self.version, 'rule_version')


@register
@dataclass(frozen=True)
class AdvisorReport(Contract):
    source_report_id: str
    source_report_sha256: str
    request: AdvisorRequest
    findings: tuple = ()
    goal_evaluations: tuple = ()
    evidence_limitations: tuple = ()
    risks: tuple = ()
    rule_versions: tuple = ()
    evidence_status: EvidenceSupport | None = None
    evaluation_status: str = 'not_evaluated'
    request_validation: RequestValidation | None = None
    request_id: str = field(init=False)
    request_sha256: str = field(init=False)
    assessment_authority: str = field(init=False, default='caller_supplied_not_verified')

    def __post_init__(self):
        _sha(self.source_report_id, 'source_report_id')
        _sha(self.source_report_sha256, 'source_report_sha256')
        if type(self.request) is not AdvisorRequest:
            raise AdvisorError('typed_request_required')
        object.__setattr__(self, 'request_id', self.request.request_id)
        object.__setattr__(self, 'request_sha256', self.request.content_sha256)
        from .resolver import NativeObservation
        from .planning import PlannedFinding
        _sequence(self, 'findings', (Finding, NativeObservation, PlannedFinding), 'finding_id')
        _sequence(self, 'goal_evaluations', GoalEvaluation, 'goal_id')
        _sequence(self, 'rule_versions', RuleVersion, 'rule_id')
        for name in ('evidence_limitations', 'risks'):
            _strings(self, name)
        _enum(self, 'evidence_status', EvidenceSupport, optional=True)
        _choice(self, 'evaluation_status', ('not_evaluated', 'unavailable', 'caller_supplied', 'observations_only', 'goals_evaluated', 'planning_only'))
        if self.evaluation_status == 'planning_only':
            object.__setattr__(self, 'assessment_authority', 'native_planning_rules_v1; not_authorized')
            if len(self.goal_evaluations) != len(self.request.goals):
                raise AdvisorError('complete_goal_evaluations_required')
            if any(f.kind not in (FindingKind.OBSERVATION, FindingKind.MEASURE_NEXT, FindingKind.EXPERIMENT_CANDIDATE, FindingKind.HYPOTHESIS) for f in self.findings):
                raise AdvisorError('unsupported_planner_finding_kind')
            if any(g.feasibility != Feasibility.UNKNOWN for g in self.goal_evaluations):
                raise AdvisorError('future_feasibility_not_evaluated')
            for finding in self.findings:
                if finding.kind == FindingKind.EXPERIMENT_CANDIDATE and type(finding) is not PlannedFinding:
                    raise AdvisorError('typed_experiment_plan_required')
                if type(finding) is PlannedFinding and not set(finding.plan.goal_ids) <= {g.goal_id for g in self.request.goals}:
                    raise AdvisorError('plan_goal_reference_mismatch')
                if type(finding) is PlannedFinding and not set(finding.plan.budget_ids) <= {b.budget_id for b in self.request.budgets}:
                    raise AdvisorError('plan_budget_reference_mismatch')
        if self.evaluation_status == 'goals_evaluated':
            object.__setattr__(self, 'assessment_authority', 'native_goal_rules_v1; record_local_only')
            if not self.request.goals or len(self.goal_evaluations) != len(self.request.goals):
                raise AdvisorError('complete_goal_evaluations_required')
            if any(f.kind not in (FindingKind.OBSERVATION, FindingKind.MEASURE_NEXT) for f in self.findings):
                raise AdvisorError('unsupported_goal_engine_finding_kind')
            if any(g.feasibility != Feasibility.UNKNOWN for g in self.goal_evaluations):
                raise AdvisorError('future_feasibility_not_evaluated')
            if any(g.observed != ObservedGoal.UNKNOWN and len(g.evidence) != 1 for g in self.goal_evaluations):
                raise AdvisorError('single_record_goal_evidence_required')
        if self.evaluation_status == 'observations_only':
            object.__setattr__(self, 'assessment_authority', 'native_observation_rules_v1; not_independent_verification')
            if any(f.kind not in (FindingKind.OBSERVATION, FindingKind.MEASURE_NEXT) for f in self.findings):
                raise AdvisorError('unsupported_observation_finding_kind')
            if any(g.evidence_support is not None or g.observed != ObservedGoal.UNKNOWN
                   or g.feasibility != Feasibility.UNKNOWN for g in self.goal_evaluations):
                raise AdvisorError('advanced_goal_evaluation_not_implemented')
        goals = {g.goal_id: g for g in self.request.goals}
        if self.request_validation is not None:
            from .validation import validate_request
            if type(self.request_validation) is not RequestValidation or self.request_validation != validate_request(self.request):
                raise AdvisorError('request_validation_mismatch')
        for evaluation in self.goal_evaluations:
            if evaluation.goal_id not in goals or evaluation.input_validity != goals[evaluation.goal_id].input_validity:
                raise AdvisorError('goal_evaluation_mismatch')
        rules = {r.rule_id: r.version for r in self.rule_versions}
        for finding in self.findings:
            if finding.rule_id is not None and rules.get(finding.rule_id) != finding.rule_version:
                raise AdvisorError('finding_rule_version_mismatch')
        refs = [r for f in self.findings for r in f.evidence] + [r for g in self.goal_evaluations for r in g.evidence]
        if any(r.report_id != self.source_report_id for r in refs):
            raise AdvisorError('source_report_reference_mismatch')
        assessed = any(g.evidence_support is not None or g.observed != ObservedGoal.UNKNOWN
                       or g.feasibility != Feasibility.UNKNOWN for g in self.goal_evaluations)
        if self.evaluation_status not in ('caller_supplied', 'observations_only', 'goals_evaluated', 'planning_only') and (assessed or self.evidence_status is not None
                or any(f.evidence_support is not None for f in self.findings)):
            raise AdvisorError('assessment_status_mismatch')

    @property
    def report_id(self):
        return self.content_sha256

    def to_markdown(self, path=None):
        """Explicit readable export of this same report; never run analysis."""
        from .render import render_markdown
        return render_markdown(self, path)
