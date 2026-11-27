"""G2 scoped native bridge: two *existing* CPU producers, never a new benchmark.

P02/P11 only. Explicit trusted-in-process consent is required. This is not a
sandbox, arbitrary-model executor, long-run scheduler, or 35-feature acceptance.
No model factory/module path is imported from user-supplied strings.
"""
import typing as _typing
from dataclasses import asdict, dataclass
import hashlib
import json
import math

from .feature_plan import FeaturePlan, FeaturePlanError


class NativeBridgeError(ValueError):
    """Report a rejected native bridge request.
    """
    pass


SUPPORTED = frozenset(('P02', 'P11'))


@dataclass(frozen=True)
class TrustedCPUExecutionPolicy:
    """Bounds operation counts and input size, NOT time/RAM/hangs/crashes.

    Args:
        explicit_trusted_inprocess_consent (bool): Explicit trust/consent for this bounded in-process
            CPU operation. Default: False.
        warmup_iterations (int): Unreported warmup iteration count before measured samples. Default: 1.
        batch_count (int): Number of batches within the explicit measurement boundary. Default: 2.
        repeats (int): Number of independently recorded measurement windows/repeats. Default: 1.
        queue_depth (int): Bound on in-flight pipeline work. Default: 2.
        max_input_bytes (int): Max input in bytes; this does not imply a GPU hardware quota. Default: 8
            * 1024 * 1024.

    Returns:
        An instance of TrustedCPUExecutionPolicy with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        TrustedCPUExecutionPolicy()

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """

    explicit_trusted_inprocess_consent: bool = False
    warmup_iterations: int = 1
    batch_count: int = 2
    repeats: int = 1
    queue_depth: int = 2
    max_input_bytes: int = 8 * 1024 * 1024

    def __post_init__(self):
        if type(self.explicit_trusted_inprocess_consent) is not bool:
            raise NativeBridgeError('consent_must_be_boolean')
        for name, lo, hi in (('warmup_iterations', 0, 5), ('batch_count', 1, 16),
                             ('repeats', 1, 3), ('queue_depth', 2, 8),
                             ('max_input_bytes', 1, 32 * 1024 * 1024)):
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise NativeBridgeError('invalid_bounded_policy:' + name)


@dataclass(frozen=True)
class LiveCPUModel:
    """Bind a caller-owned live CPU model and its input to the scoped CPU execution contract.

    Args:
        model_id (str): Stable model label bound to this result/context. Required.
        model (object): Caller-owned model; device, dtype and task must match the native operation.
            Required.
        input_tensor (object): Caller-supplied input Tensor with the declared shape/device/dtype.
            Default: None.

    Returns:
        An instance of LiveCPUModel with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        LiveCPUModel(model_id=model_id, model=model)

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """
    model_id: str
    model: object
    input_tensor: object = None

    def __post_init__(self):
        if type(self.model_id) is not str or not self.model_id.strip():
            raise NativeBridgeError('model_id_required')


@dataclass(frozen=True)
class CapturedNativeResult:
    """Store captured native result evidence as an immutable typed record.

    Args:
        model_id (str): Caller label identifying a model within this evidence context; not a weight or
            hardware identity. Required.
        feature_id (str): P01–P35 catalog identifier for this evidence or plan item. Required.
        status (str): Recorded evidence/coverage status; unavailable values must not be interpreted as
            zero. Required.
        source_sha256 (str): Digest of the exact source evidence content. Required.
        result (str): Retained HYSo result/evidence record; rendering/advice does not run measurements.
            Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    model_id: str
    feature_id: str
    status: str
    source_sha256: str
    result: str  # canonical JSON owned copy, not a shared mutable dictionary

    def to_dict(self) -> 'dict[str, _typing.Any]':
        """Return detached JSON-compatible evidence without rerunning the source workload.
        """
        return dict(model_id=self.model_id, feature_id=self.feature_id, status=self.status,
                    source_sha256=self.source_sha256, native_result=json.loads(self.result))


@dataclass(frozen=True)
class ObservedModelDifference:
    """Store observed model difference evidence as an immutable typed record.

    Args:
        feature_id (str): P01–P35 catalog identifier for this evidence or plan item. Required.
        metric (str): Exact recorded metric whose baseline/candidate values are compared. Required.
        baseline_model_id (str): Model label for the reference evidence. Required.
        candidate_model_id (str): Model label for the candidate evidence. Required.
        baseline (float | int | None): Reference evidence used as the denominator/origin of candidate
            differences; missing/incompatible evidence is not inferred. Required.
        candidate (float | int | None): Evidence or runtime being compared to the reference;
            measurements remain bound to their recorded conditions. Required.
        observed_difference (float | int | None): Recorded candidate-minus-baseline difference, limited
            to the comparable evidence. Required.
        status (str): Recorded evidence/coverage status; unavailable values must not be interpreted as
            zero. Required.
        reason (str): Recorded explanation for rejection, absence or incomplete evidence; None only
            where the schema allows it. Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    feature_id: str
    metric: str
    baseline_model_id: str
    candidate_model_id: str
    baseline: float | int | None
    candidate: float | int | None
    observed_difference: float | int | None
    status: str
    reason: str


@dataclass(frozen=True)
class NativeBridgeReport:
    """Store native bridge report evidence as an immutable typed record.

    Args:
        requested_feature_ids (tuple): Feature identifiers explicitly requested by the caller. Required.
        executed_feature_ids (tuple): Feature identifiers actually executed by the native bridge.
            Required.
        omitted_features (tuple): Requested features not executed, with explicit reasons. Required.
        captures (tuple): Captured source-bound native result records. Required.
        comparisons (tuple): Explicit pairwise comparisons from compatible captured evidence. Required.
        scope (str): Declared evidence coverage; does not authorize execution. Default:
            'trusted_inprocess_cpu;not_isolated_no_hard_runtime_memory_limit'.
        acceptance (str): Recorded acceptance scope; not a new execution or certification. Default:
            'NOT_EVALUATED'.
        authorization (str): Recorded caller authorization requirements; planning grants no execution
            permission. Default: 'EXPLICIT_CPU_P02_P11_ONLY'.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    requested_feature_ids: tuple
    executed_feature_ids: tuple
    omitted_features: tuple  # explicit NOT_EXECUTED + reason, not successes
    captures: tuple
    comparisons: tuple
    scope: str = 'trusted_inprocess_cpu;not_isolated_no_hard_runtime_memory_limit'
    acceptance: str = 'NOT_EVALUATED'
    authorization: str = 'EXPLICIT_CPU_P02_P11_ONLY'

    def to_dict(self) -> 'dict[str, _typing.Any]':
        """Return detached JSON-compatible evidence without rerunning the source workload.
        """
        return dict(requested_feature_ids=list(self.requested_feature_ids),
                    executed_feature_ids=list(self.executed_feature_ids),
                    omitted_features=[dict(feature_id=f, status='NOT_EXECUTED', reason=r) for f,r in self.omitted_features],
                    captures=[x.to_dict() for x in self.captures],
                    comparisons=[asdict(x) for x in self.comparisons],
                    scope=self.scope, acceptance=self.acceptance,
                    authorization=self.authorization)


def _json_copy(data):
    if type(data) is not dict:
        raise NativeBridgeError('native_result_must_be_dictionary')
    return json.dumps(data, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(',', ':'))


def _checked_capture(model_id, feature_id, native):
    if not hasattr(native, 'to_dict') or not callable(native.to_dict):
        raise NativeBridgeError('native_producer_result_required')
    payload = native.to_dict()
    content = _json_copy(payload)
    if payload.get('feature_ids') != [feature_id] or payload.get('schema_version') != 1:
        raise NativeBridgeError('wrong_native_feature_or_schema')
    if feature_id == 'P02':
        metrics = payload.get('measurements', {})
        if (metrics.get('status') != 'measured' or metrics.get('unit') != 'images/s'
                or type(metrics.get('value')) not in (float, int)
                or not math.isfinite(metrics['value']) or metrics['value'] <= 0
                or payload.get('actual', {}).get('pending_batches') != 0
                or payload.get('actual', {}).get('completed_batches', 0) < 1):
            raise NativeBridgeError('invalid_native_throughput_evidence')
    elif feature_id == 'P11':
        metrics = payload.get('measurements', {}).get('parameter_buffer_bytes', {})
        if (metrics.get('status') != 'derived' or metrics.get('unit') != 'byte'
                or type(metrics.get('value')) is not int or metrics['value'] < 0):
            raise NativeBridgeError('invalid_native_footprint_evidence')
    else:
        raise NativeBridgeError('unimplemented_native_feature')
    return CapturedNativeResult(model_id, feature_id, 'MEASURED_NATIVE_SCOPE',
                                hashlib.sha256(content.encode('utf-8')).hexdigest(), content)


def _preflight_binding(plan, binding, policy, needs_inference):
    import torch
    if type(binding) is not LiveCPUModel:
        raise NativeBridgeError('typed_live_model_required')
    if not isinstance(binding.model, torch.nn.Module):
        raise NativeBridgeError('torch_module_required')
    # Registered tensors are inspected without running user-defined forward.
    # Custom parameter accessors and forward hooks are rejected in the underlying producers.
    for tensor in tuple(binding.model.parameters()) + tuple(binding.model.buffers()):
        if tensor.device.type != 'cpu':
            raise NativeBridgeError('cpu_only_no_implicit_device_transfer')
    if needs_inference:
        x = binding.input_tensor
        if (type(x) is not torch.Tensor or x.device.type != 'cpu'
                or x.dtype != torch.float32 or x.layout != torch.strided
                or not x.is_contiguous() or x.ndim != 4 or x.shape[0] < 1
                or any(d < 1 for d in x.shape)
                or x.numel() * x.element_size() > policy.max_input_bytes):
            raise NativeBridgeError('bounded_cpu_fp32_nchw_input_required')
    if plan.backend != 'pytorch' or plan.task != 'classification':
        raise NativeBridgeError('pytorch_classification_only_in_G2')


def _invoke_native(feature_id, binding, policy):
    """No arbitrary import path or new measurement implementation is introduced."""
    if feature_id == 'P11':
        from hyso.performance import measure_footprint
        return measure_footprint(binding.model)
    if feature_id == 'P02':
        from hyso.performance import ThroughputProtocol, measure_throughput
        return measure_throughput(binding.model, binding.input_tensor,
            protocol=ThroughputProtocol(device='cpu', precision='fp32',
                    warmup_iterations=policy.warmup_iterations,
                    batch_count=policy.batch_count, repeats=policy.repeats,
                    queue_depth=policy.queue_depth, concurrency=1))
    raise NativeBridgeError('native_producer_not_implemented')


def _observed_pair(feature, a, b):
    x = json.loads(a.result); y = json.loads(b.result)
    if feature == 'P02':
        identity = lambda z: (z['provenance']['input']['shape'],
                              z['provenance']['input']['dtype'],
                              z['provenance']['input']['content_sha256'],
                              z['provenance']['environment']['device'],
                              z['protocol']['configuration'])
        same = identity(x) == identity(y)
        metric = 'completed_images_per_second'
        v1 = x['measurements']['value']; v2 = y['measurements']['value']
        reason = ('same_input_and_native_protocol;no_independent_repeats_or_speedup_certification'
                  if same else 'input_or_protocol_not_comparable')
    elif feature == 'P11':
        metric = 'parameter_buffer_bytes'
        v1 = x['measurements'][metric]['value']; v2 = y['measurements'][metric]['value']
        same = (x['protocol']['byte_accounting'] == y['protocol']['byte_accounting']
                and x['provenance']['environment']['device'] == y['provenance']['environment']['device'])
        reason = ('same_logical_byte_convention_not_physical_RAM' if same
                  else 'incompatible_footprint_convention')
    else:
        raise NativeBridgeError('feature_not_comparable_in_G2')
    return ObservedModelDifference(feature, metric, a.model_id, b.model_id,
                                   v1 if same else None, v2 if same else None,
                                   (v2 - v1) if same else None,
                                   'OBSERVED_DESCRIPTIVE_ONLY' if same else 'BLOCKED', reason)


def execute_scoped_cpu(plan: '_typing.Any', bindings: '_typing.Any', *, authorized_features: '_typing.Any', policy: '_typing.Any'=TrustedCPUExecutionPolicy()) -> '_typing.Any':
    """Runs only individually authorized P02/P11, for trusted, local CPU models.

    This is intentionally NOT safe for untrusted code: execution is in-process,
    no hard watchdog, no OS memory isolation, and no promise against side effects.
    This function never writes checkpoints or modifies model weights intentionally.

    Args:
        plan (_typing.Any): Typed selection plan; a plan alone grants no execution permission. Required.
        bindings (_typing.Any): Typed live/isolated/native bindings; identities and operation
            permissions must match the plan. Required.
        authorized_features (_typing.Any): Explicit unique selected feature subset authorized for this
            call only. Required.
        policy (_typing.Any): Existing protocol/resource policy; defaults do not grant additional
            consent. Default: TrustedCPUExecutionPolicy().

    Returns:
        The retained execute scoped cpu value, without recomputing measurements.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        execute_scoped_cpu(plan, bindings, authorized_features=authorized_features)

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """
    if type(plan) is not FeaturePlan or type(policy) is not TrustedCPUExecutionPolicy:
        raise NativeBridgeError('typed_G1_plan_and_G2_policy_required')
    if not policy.explicit_trusted_inprocess_consent:
        raise NativeBridgeError('trusted_inprocess_execution_requires_explicit_consent')
    if type(bindings) not in (tuple, list) or type(authorized_features) not in (tuple, list):
        raise NativeBridgeError('typed_bindings_and_authorizations_required')
    selected = tuple(x.feature_id for x in plan.features)
    authorized = tuple(authorized_features)
    if (len(authorized) != len(set(authorized)) or
            not set(authorized).issubset(SUPPORTED & set(selected))):
        raise NativeBridgeError('unauthorized_or_unimplemented_feature_requested')
    if len(bindings) != len(plan.models) or any(type(x) is not LiveCPUModel for x in bindings):
        raise NativeBridgeError('one_exact_binding_per_planned_model_required')
    if tuple(x.model_id for x in bindings) != tuple(x.model_id for x in plan.models):
        raise NativeBridgeError('model_identity_mismatch')
    if len({id(x.model) for x in bindings}) != len(bindings):
        raise NativeBridgeError('shared_model_object_cannot_be_two_distinct_models')
    # Entire request is validated before the first native producer is invoked.
    for b, m in zip(bindings, plan.models):
        _preflight_binding(m, b, policy, 'P02' in authorized)
    blocked = []
    for feature in authorized:
        entry = next(x for x in plan.features if x.feature_id == feature)
        if any(x.status != 'PROPOSED' for x in entry.models):
            blocked.append(feature)
    if blocked:
        raise NativeBridgeError('G1_prerequisites_not_proposed:' + ','.join(blocked))
    captures = []
    for feature in selected:
        if feature not in authorized:
            continue
        for binding in bindings:
            captures.append(_checked_capture(binding.model_id, feature,
                                            _invoke_native(feature, binding, policy)))
    comparisons = []
    if len(bindings) == 2:
        for feature in authorized:
            a, b = (next(x for x in captures if x.model_id == binding.model_id
                         and x.feature_id == feature) for binding in bindings)
            comparisons.append(_observed_pair(feature, a, b))
    omitted = tuple((f, 'not_authorized' if f in SUPPORTED else 'native_adapter_not_implemented')
                    for f in selected if f not in authorized)
    return NativeBridgeReport(selected, authorized, omitted, tuple(captures), tuple(comparisons))
