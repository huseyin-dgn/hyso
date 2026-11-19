"""Stage 3B.1: immutable data contracts, not an Advisor implementation."""
import typing as _typing
from dataclasses import dataclass, field
from enum import Enum
import math
import re

from ._json import AdvisorError, Contract, register


class Priority(str, Enum):
    """Select the user priority for speed, memory, accuracy preservation or a balanced goal.

    Returns:
        An instance of Priority with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        Priority.SPEED

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en

    Members:
        SPEED, MEMORY, PRESERVE_ACCURACY, BALANCED.
    """
    SPEED = 'speed'
    MEMORY = 'memory'
    PRESERVE_ACCURACY = 'preserve_accuracy'
    BALANCED = 'balanced'


class Metric(str, Enum):
    """Select a supported metric identity; latency, throughput, quality, memory and energy retain distinct units and scopes.

    Returns:
        An instance of Metric with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        Metric.RESIDENT_FORWARD_P95

    HYSo Guide: metrics.interpretation
    Offline: hyso docs show metrics.interpretation --lang en
    Open: hyso docs open metrics.interpretation --lang en

    Members:
        RESIDENT_FORWARD_P95, E2E_P95, IMAGE_THROUGHPUT, CAMERA_FPS, TRAINING_THROUGHPUT, TRAINING_STEP_TIME, ACCURACY, ACCURACY_DROP, PROCESS_RSS_PEAK, CUDA_ALLOCATED_PEAK, CUDA_RESERVED_PEAK, LOGICAL_TENSOR_BYTES, ENERGY_PER_INFERENCE, POWER.

    Metric boundaries:
        P95 is a latency quantile, not a deadline guarantee. images/s is not camera FPS.
        Accuracy percentages and percentage-point changes are distinct. Process RSS,
        CUDA allocated/reserved peaks and logical tensor bytes are different measures.
        Power (W), energy (J) and energy per inference (J/inference) require valid sensor coverage.
    """
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
    """Distinguish observations, hypotheses, proposed measurements and verified improvements.

    Returns:
        An instance of FindingKind with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        FindingKind.OBSERVATION

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en

    Members:
        OBSERVATION, HYPOTHESIS, DIAGNOSIS, MEASURE_NEXT, EXPERIMENT_CANDIDATE, VERIFIED_IMPROVEMENT.
    """
    OBSERVATION = 'OBSERVATION'
    HYPOTHESIS = 'HYPOTHESIS'
    DIAGNOSIS = 'DIAGNOSIS'
    MEASURE_NEXT = 'MEASURE_NEXT'
    EXPERIMENT_CANDIDATE = 'EXPERIMENT_CANDIDATE'
    VERIFIED_IMPROVEMENT = 'VERIFIED_IMPROVEMENT'


class InputValidity(str, Enum):
    """Describe whether an input is valid, invalid or missing required scope.

    Returns:
        An instance of InputValidity with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        InputValidity.VALID

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en

    Members:
        VALID, INVALID, NEEDS_SCOPE.
    """
    VALID = 'valid'
    INVALID = 'invalid'
    NEEDS_SCOPE = 'needs_scope'


class EvidenceSupport(str, Enum):
    """Describe evidence support separately from a user goal or workload success.

    Returns:
        An instance of EvidenceSupport with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        EvidenceSupport.SUPPORTED

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en

    Members:
        SUPPORTED, INSUFFICIENT_EVIDENCE, UNSUPPORTED_CAPABILITY.
    """
    SUPPORTED = 'supported'
    INSUFFICIENT_EVIDENCE = 'insufficient_evidence'
    UNSUPPORTED_CAPABILITY = 'unsupported_capability'


class ObservedGoal(str, Enum):
    """Record whether the measured goal was met, missed or remains unknown.

    Returns:
        An instance of ObservedGoal with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        ObservedGoal.MET

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en

    Members:
        MET, MISSED, UNKNOWN.
    """
    MET = 'observed_met'
    MISSED = 'observed_missed'
    UNKNOWN = 'unknown'


class Feasibility(str, Enum):
    """Distinguish feasibility demonstrated in scope from unproven or unknown feasibility.

    Returns:
        An instance of Feasibility with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        Feasibility.DEMONSTRATED

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en

    Members:
        DEMONSTRATED, NOT_DEMONSTRATED, UNKNOWN.
    """
    DEMONSTRATED = 'demonstrated_in_scope'
    NOT_DEMONSTRATED = 'not_demonstrated'
    UNKNOWN = 'unknown'


class Consistency(str, Enum):
    """Describe whether declared goals conflict under the available evidence.

    Returns:
        An instance of Consistency with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        Consistency.NO_CONFLICT

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en

    Members:
        NO_CONFLICT, POTENTIAL, INCONSISTENT.
    """
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
    """Bind a metric to its phase, device, precision, batch and workload/dataset context.

    Args:
        phase (str): Explicit trace phase label. Required.
        device (str | None): Optional device selector following this API's native placement/default rules; legacy EBO may choose CUDA. Pass cpu explicitly for CPU use. Default: None.
        backend (str | None): Explicit runtime/backend identity. Default: None.
        precision (str | None): Supported numerical format; runtime/task/device restrictions remain authoritative. Default: None.
        batch (int | None): Batch dimension of the measured input. Default: None.
        input_shape (tuple): Per-sample (C, H, W) dimensions; batch is excluded. Default: ().
        workload_id (str | None): Exact workload identity for scope alignment. Default: None.
        dataset_id (str | None): Exact dataset identity used for evidence alignment. Default: None.
        split (str | None): Explicit dataset split label used in evaluation identity. Default: None.
        preprocessing_id (str | None): Identity of the preprocessing used to produce this fixed cohort. Default: None.
        clock_id (str | None): Clock-domain identity used for timing alignment. Default: None.
        model_id (str | None): Stable model label bound to this result/context. Default: None.

    Returns:
        An instance of MetricScope with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        MetricScope(phase=phase)

    HYSo Guide: metrics.interpretation
    Offline: hyso docs show metrics.interpretation --lang en
    Open: hyso docs open metrics.interpretation --lang en
    """
    phase: _typing.Literal['inference', 'training', 'evaluation', 'process', 'static']
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
    """Identify the evaluation method, version, sample policy and averaging convention.

    Args:
        protocol_id (str): Exact measurement/evaluation protocol identity. Required.
        version (int): Positive native schema/protocol/rule version. Required.
        description (str): Human-readable specification description. Required.
        sample_policy (str | None): Sampling convention used by the measurement/evaluation. Default: None.
        window (str | None): Exact measurement window definition. Default: None.
        evaluator (str | None): Explicit evaluation implementation/protocol identity. Default: None.
        averaging (str | None): Metric averaging convention, such as macro or per-class. Default: None.

    Returns:
        An instance of EvaluationProtocol with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        EvaluationProtocol(protocol_id=protocol_id, version=version, description=description)

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en
    """
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
    """Declare a measurable target with its metric, unit, operator and applicable scope.

    Args:
        goal_id (str): Exact user-goal identity. Required.
        metric (Metric): Exact supported metric selector; units and missing-value status are preserved. Required.
        operator (str): Supported comparison operator for the goal. Required.
        target (float): Explicit target value/model/context required by this operation. Required.
        unit (str): Recorded metric unit; fractions, percentages and percentage points are distinct. Required.
        constraint (str): Explicit restriction applied to the advisory request. Default: 'hard'.
        scope (MetricScope | None): Exact measurement boundary/meaning; different scopes are not interchangeable. Default: None.
        protocol (EvaluationProtocol | None): Typed native measurement protocol; scope, budgets and device guards remain unchanged. Default: None.
        value_basis (str): Evidence basis and interpretation of the reported value. Default: 'measured'.
        baseline_id (str | None): Exact retained baseline identity. Default: None.

    Returns:
        An instance of UserGoal with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        UserGoal(goal_id=goal_id, metric=metric, operator=operator, target=target, unit=unit)

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en
    """
    goal_id: str
    metric: Metric
    operator: _typing.Literal['<=', '>=', '<', '>', '==']
    target: float
    unit: _typing.Literal['byte', 'GB', 'GiB', 's', 'ms', 'images/s', 'frames/s', 'fraction', 'percentage_point', 'J/inference', 'W']
    constraint: _typing.Literal['hard', 'soft'] = 'hard'
    scope: MetricScope | None = None
    protocol: EvaluationProtocol | None = None
    value_basis: _typing.Literal['measured', 'estimated'] = 'measured'
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
    def input_validity(self) -> '_typing.Any':
        """Read the retained input validity from this UserGoal record.

        Returns:
            The retained input validity value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.input_validity

        HYSo Guide: advisor.recommendations
        Offline: hyso docs show advisor.recommendations --lang en
        Open: hyso docs open advisor.recommendations --lang en
        """
        return InputValidity.NEEDS_SCOPE if self.missing_scope_fields else InputValidity.VALID

    @property
    def missing_scope_fields(self) -> '_typing.Any':
        """Request completeness only; never native evidence eligibility.

        Returns:
            The retained missing scope fields value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.missing_scope_fields

        HYSo Guide: advisor.recommendations
        Offline: hyso docs show advisor.recommendations --lang en
        Open: hyso docs open advisor.recommendations --lang en
        """
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
    """Declare a resource budget and its enforcement/verification status; a declaration alone enforces nothing.

    Args:
        budget_id (str): Stable resource-budget identity. Required.
        purpose (str): Explicit reason for the operation or evidence selection. Required.
        domain (str): Feature/metric domain, separate from runtime support. Required.
        limit (float): Bounded page/record count; native Manager limits still apply. Required.
        unit (str): Recorded metric unit; fractions, percentages and percentage points are distinct. Required.
        allowed_devices (tuple): Explicit device choices permitted by this policy. Default: ().
        allowed_backends (tuple): Closed backend choices permitted by this request. Default: ().
        allowed_precisions (tuple): Explicit precision choices permitted by this request. Default: ().
        window_seconds (float | None): Window in seconds under the native protocol. Default: None.
        enforcement (str): Declared enforcement mechanism; unsupported enforcement is not inferred. Default: 'not_evaluated'.
        applicability (str): Conditions under which the finding/evidence applies. Default: 'not_evaluated'.

    Returns:
        An instance of ResourceBudget with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        ResourceBudget(budget_id=budget_id, purpose=purpose, domain=domain, limit=limit, unit=unit)

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en
    """
    budget_id: str
    purpose: _typing.Literal['deployment', 'experiment']
    domain: _typing.Literal['process_rss', 'cuda_allocated', 'cuda_reserved', 'device_memory', 'logical_tensor_bytes', 'wall_time', 'energy', 'power', 'trial_count']
    limit: float
    unit: _typing.Literal['byte', 'GB', 'GiB', 's', 'J', 'W', 'count']
    allowed_devices: tuple = ()
    allowed_backends: tuple = ()
    allowed_precisions: tuple = ()
    window_seconds: float | None = None
    enforcement: _typing.Literal['not_evaluated', 'unavailable', 'advisory_only'] = 'not_evaluated'
    applicability: _typing.Literal['not_evaluated', 'unavailable', 'unsupported_capability'] = 'not_evaluated'

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
    """Describe the requested advisory mode, goals, allowed changes and constraints without authorizing execution.

    Args:
        goals (tuple): Explicit user goals; unmet prerequisites are retained. Default: ().
        budgets (tuple): Explicit resource budgets with their enforcement status. Default: ().
        priority (Priority | None): Explicit user priority for advisory ranking. Default: None.

    Returns:
        An instance of AdvisorRequest with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        AdvisorRequest()

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en
    """
    goals: tuple = ()
    budgets: tuple = ()
    priority: Priority | None = None

    def __post_init__(self):
        _sequence(self, 'goals', UserGoal, 'goal_id')
        _sequence(self, 'budgets', ResourceBudget, 'budget_id')
        _enum(self, 'priority', Priority, optional=True)

    @property
    def request_id(self) -> '_typing.Any':
        """Read the retained request id from this AdvisorRequest record.

        Returns:
            The retained request id value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.request_id

        HYSo Guide: advisor.recommendations
        Offline: hyso docs show advisor.recommendations --lang en
        Open: hyso docs open advisor.recommendations --lang en
        """
        return 'sha256:' + self.content_sha256

    @property
    def mode(self) -> '_typing.Any':
        """Read the retained mode from this AdvisorRequest record.

        Returns:
            The retained mode value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.mode

        HYSo Guide: advisor.recommendations
        Offline: hyso docs show advisor.recommendations --lang en
        Open: hyso docs open advisor.recommendations --lang en
        """
        return 'advanced' if self.goals or self.budgets else 'priority' if self.priority else 'automatic'


@register
@dataclass(frozen=True)
class EvidenceBinding(Contract):
    """Bind an advisory statement to exact retained evidence and its metric scope.

    Args:
        field (str): Exact retained field/path selected by the resolver. Required.
        authority (str): Declared authority/provenance level of the evidence. Required.

    Returns:
        An instance of EvidenceBinding with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        EvidenceBinding(field=field, authority=authority)

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en
    """
    field: str
    authority: _typing.Literal['native', 'declared', 'missing']

    def __post_init__(self):
        _text(self.field, 'binding_field')
        _choice(self, 'authority', ('native', 'declared', 'missing'))


@register
@dataclass(frozen=True)
class EvidenceRef(Contract):
    """Reference retained evidence by identity and its supported applicability.

    Args:
        report_id (str): Exact retained report identity. Required.
        record_id (str): Exact retained evidence identity. Required.
        metric_path (str): Exact native metric path in the retained result. Required.
        artifact_sha256 (str): Exact serialized artifact digest. Required.
        payload_sha256 (str): Exact SHA-256 digest of the payload. Required.
        domain (str): Feature/metric domain, separate from runtime support. Required.
        unit (str): Recorded metric unit; fractions, percentages and percentage points are distinct. Required.
        status (str): Exact lifecycle/status filter accepted by this API. Required.
        bindings (tuple): Typed live/isolated/native bindings; identities and operation permissions must match the plan. Required.
        sample_ref (str | None): Exact retained sample reference. Default: None.
        point_ref (str | None): Reference to the exact retained sweep/calibration point. Default: None.
        protocol_ref (str | None): Reference to the exact retained protocol. Default: None.

    Returns:
        An instance of EvidenceRef with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        EvidenceRef(report_id=report_id, record_id=record_id, metric_path=metric_path, artifact_sha256=artifact_sha256, payload_sha256=payload_sha256, domain=domain, unit=unit, status=status, bindings=bindings)

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en
    """
    report_id: str
    record_id: str
    metric_path: str
    artifact_sha256: str
    payload_sha256: str
    domain: str
    unit: str
    status: _typing.Literal['measured', 'derived', 'estimated', 'unknown', 'unsupported', 'not_run', 'invalid', 'insufficient_samples']
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
    """Record one advisory finding with its rationale, evidence, limitations and proposed next measurement.

    Args:
        finding_id (str): Stable advisory-finding identity. Required.
        kind (FindingKind): Existing run or artifact kind accepted by this operation. Required.
        statement (str): Human-readable finding/observation statement. Required.
        evidence (tuple): Typed retained evidence; missing measurements remain unavailable. Default: ().
        limitations (tuple): Explicit limitations of the evidence, comparison or proposed operation. Default: ().
        rule_id (str | None): Exact advisory rule identity. Default: None.
        rule_version (str | None): Exact advisory rule version. Default: None.
        evidence_support (EvidenceSupport | None): Assessment of whether the supplied evidence supports the claim. Default: None.
        verification (str): Retained verification state; declarations are not successful execution evidence. Default: 'not_evaluated'.

    Returns:
        An instance of Finding with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        Finding(finding_id=finding_id, kind=kind, statement=statement)

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en
    """
    finding_id: str
    kind: FindingKind
    statement: str
    evidence: tuple = ()
    limitations: tuple = ()
    rule_id: str | None = None
    rule_version: str | None = None
    evidence_support: EvidenceSupport | None = None
    verification: _typing.Literal['not_evaluated', 'unavailable', 'not_verified'] = 'not_evaluated'

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
    """Record validity, support, observed goal outcome and feasibility as separate assessments.

    Args:
        goal_id (str): Exact user-goal identity. Required.
        input_validity (InputValidity): Retained input validity assessment. Required.
        evidence_support (EvidenceSupport | None): Assessment of whether the supplied evidence supports the claim. Default: None.
        observed (ObservedGoal): Observed outcome under the supplied evidence scope. Default: ObservedGoal.UNKNOWN.
        feasibility (Feasibility): Feasibility assessment within the exact evidence scope. Default: Feasibility.UNKNOWN.
        evidence (tuple): Typed retained evidence; missing measurements remain unavailable. Default: ().
        reasons (tuple): Structured eligibility/compatibility reasons. Default: ().

    Returns:
        An instance of GoalEvaluation with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        GoalEvaluation(goal_id=goal_id, input_validity=input_validity)

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en
    """
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
    """Retain request validation and goal consistency findings without starting a workload.

    Args:
        goals (tuple): Explicit user goals; unmet prerequisites are retained. Required.
        consistency (Consistency): Goal-consistency assessment separate from feasibility. Required.
        conflicting_goal_ids (tuple): Exact identities of conflicting goals. Default: ().
        reasons (tuple): Structured eligibility/compatibility reasons. Default: ().

    Returns:
        An instance of RequestValidation with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        RequestValidation(goals=goals, consistency=consistency)

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en
    """
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
    """Identify the advisory rule and version used to produce a finding.

    Args:
        rule_id (str): Exact advisory rule identity. Required.
        version (str): Positive native schema/protocol/rule version. Required.

    Returns:
        An instance of RuleVersion with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        RuleVersion(rule_id=rule_id, version=version)

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en
    """
    rule_id: str
    version: str

    def __post_init__(self):
        _text(self.rule_id, 'rule_id')
        _text(self.version, 'rule_version')


@register
@dataclass(frozen=True)
class AdvisorReport(Contract):
    """Retain an advisory decision, findings, goal evaluations and the exact request/evidence identities.

    Args:
        source_report_id (str): Exact source-report identity. Required.
        source_report_sha256 (str): Exact SHA-256 digest of the source report. Required.
        request (AdvisorRequest): Validated advisory/selection request; does not authorize execution. Required.
        findings (tuple): Typed retained advisory findings. Default: ().
        goal_evaluations (tuple): Separate validity, support, outcome and feasibility assessments. Default: ().
        evidence_limitations (tuple): Limitations attached to the retained evidence. Default: ().
        risks (tuple): Known limitations/risks attached to the proposal. Default: ().
        rule_versions (tuple): Exact rule identities/versions used by the report. Default: ().
        evidence_status (EvidenceSupport | None): Retained availability/coverage status. Default: None.
        evaluation_status (str): Retained evaluation status, including unavailable or partial results. Default: 'not_evaluated'.
        request_validation (RequestValidation | None): Retained validation and consistency results for the request. Default: None.
        request_id (str): Optional idempotency key for the stored operation; not a model or artifact identity. Default: field(init=False).
        request_sha256 (str): Exact SHA-256 digest of the request. Default: field(init=False).
        assessment_authority (str): Authority level of the assessment; caller claims are not producer attestation. Default: field(init=False, default='caller_supplied_not_verified').

    Returns:
        An instance of AdvisorReport with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        AdvisorReport(source_report_id=source_report_id, source_report_sha256=source_report_sha256, request=request)

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en
    """
    source_report_id: str
    source_report_sha256: str
    request: AdvisorRequest
    findings: tuple = ()
    goal_evaluations: tuple = ()
    evidence_limitations: tuple = ()
    risks: tuple = ()
    rule_versions: tuple = ()
    evidence_status: EvidenceSupport | None = None
    evaluation_status: _typing.Literal['not_evaluated', 'unavailable', 'caller_supplied', 'observations_only', 'goals_evaluated', 'planning_only'] = 'not_evaluated'
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
    def report_id(self) -> '_typing.Any':
        """Read the retained report id from this AdvisorReport record.

        Returns:
            The retained report id value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.report_id

        HYSo Guide: advisor.recommendations
        Offline: hyso docs show advisor.recommendations --lang en
        Open: hyso docs open advisor.recommendations --lang en
        """
        return self.content_sha256

    def to_markdown(self, path: '_typing.Any'=None, *, language: _typing.Literal['tr', 'en'] | None=None, audience: _typing.Literal['summary', 'technical']=None) -> '_typing.Any':
        """Explicit readable export of this same report; never run analysis.

        Args:
            path (_typing.Any): Filesystem path for the specific reader/exporter; no configuration code is executed. Default: None.
            language (str | None): Presentation language supported by this API, normally tr/en; guide languages do not expand report APIs. Default: None.
            audience (_typing.Any): Supported presentation audience/detail level. Default: None.

        Returns:
            The localized Markdown report string.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_markdown()

        HYSo Guide: advisor.recommendations
        Offline: hyso docs show advisor.recommendations --lang en
        Open: hyso docs open advisor.recommendations --lang en
        """
        if language is not None or audience is not None:
            from .presentation import to_markdown
            return to_markdown(self, path, language='en' if language is None else language,
                               audience='summary' if audience is None else audience)
        from .render import render_markdown
        return render_markdown(self, path)

    def to_display_dict(self, *, language: _typing.Literal['tr', 'en']='en', audience: _typing.Literal['summary', 'technical']='summary') -> 'dict[str, _typing.Any]':
        """Presentation-only localized JSON data; never canonical evidence.

        Args:
            language (str): Presentation language supported by this API, normally tr/en; guide languages do not expand report APIs. Default: 'en'.
            audience (str): Supported presentation audience/detail level. Default: 'summary'.

        Returns:
            Localized presentation data; canonical comparison values and units are retained.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_display_dict()

        HYSo Guide: advisor.recommendations
        Offline: hyso docs show advisor.recommendations --lang en
        Open: hyso docs open advisor.recommendations --lang en
        """
        from .presentation import to_display_dict
        return to_display_dict(self, language=language, audience=audience)

    def to_display_json(self, path: '_typing.Any'=None, *, language: _typing.Literal['tr', 'en']='en', audience: _typing.Literal['summary', 'technical']='summary') -> 'str':
        """Serialize localized display text while preserving the retained canonical data.

        Args:
            path (_typing.Any): Filesystem path for the specific reader/exporter; no configuration code is executed. Default: None.
            language (str): Presentation language supported by this API, normally tr/en; guide languages do not expand report APIs. Default: 'en'.
            audience (str): Supported presentation audience/detail level. Default: 'summary'.

        Returns:
            The localized JSON string, also written to path when supplied.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_display_json()

        HYSo Guide: advisor.recommendations
        Offline: hyso docs show advisor.recommendations --lang en
        Open: hyso docs open advisor.recommendations --lang en
        """
        from .presentation import to_display_json
        return to_display_json(self, path, language=language, audience=audience)
