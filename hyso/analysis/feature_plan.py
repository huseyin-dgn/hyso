"""Pure P01-P35 feature selection and prerequisite planning.

This is NOT an execution manager, model loader, capability attestation or
acceptance certificate. Nothing in this module imports Torch, executes models,
opens files, queries devices, grants authorization or modifies a model.

G1 integration boundary: describe what to assess before an explicitly approved
native run; carry unknown and missing prerequisites instead of faking results.
"""

from dataclasses import asdict, dataclass


class FeaturePlanError(ValueError):
    """Invalid, ambiguous or unsupported planning request."""


def _tokens(value, allowed, label, *, allow_empty=True):
    if type(value) not in (tuple, list):
        raise FeaturePlanError(label + '_must_be_a_sequence')
    items = tuple(value)
    if (not allow_empty and not items) or any(type(x) is not str or x not in allowed for x in items):
        raise FeaturePlanError('invalid_' + label)
    if len(items) != len(set(items)):
        raise FeaturePlanError('duplicate_' + label)
    return items


_TASKS = frozenset(('classification', 'detection', 'segmentation', 'other'))
_BACKENDS = frozenset(('pytorch', 'onnx'))
_CAPABILITIES = frozenset(('gpu', 'ncu', 'power_sensor', 'multiple_devices', 'ort_cpu', 'ort_cuda'))
_GOALS = frozenset(('latency', 'throughput', 'memory', 'bottleneck', 'quality',
                    'training', 'stability', 'energy', 'providers', 'comparison', 'streaming'))
_MODES = frozenset(('single', 'selected', 'automatic', 'full'))


@dataclass(frozen=True)
class ModelContext:
    """Caller DECLARATIONS only. These are NOT verified model facts."""

    model_id: str
    task: str = 'classification'
    backend: str = 'pytorch'
    input_contract: bool = False
    adapter_contract: bool = False
    labeled_evaluation: bool = False
    training_workload: bool = False
    stage_adapters: bool = False
    trusted_factory: bool = False
    checkpoint: bool = False
    trained_frame_adapter: bool = False
    recorded_video: bool = False
    live_camera: bool = False
    onnx_export_contract: bool = False
    retained_long_run: bool = False

    def __post_init__(self):
        if (type(self.model_id) is not str or not self.model_id.strip()
                or len(self.model_id) > 160):
            raise FeaturePlanError('invalid_model_id')
        if type(self.task) is not str or self.task not in _TASKS:
            raise FeaturePlanError('invalid_model_task')
        if type(self.backend) is not str or self.backend not in _BACKENDS:
            raise FeaturePlanError('invalid_model_backend')
        for key in ('input_contract', 'adapter_contract', 'labeled_evaluation',
                    'training_workload', 'stage_adapters', 'trusted_factory',
                    'checkpoint', 'trained_frame_adapter', 'recorded_video',
                    'live_camera', 'onnx_export_contract', 'retained_long_run'):
            if type(getattr(self, key)) is not bool:
                raise FeaturePlanError('expected_boolean_' + key)


@dataclass(frozen=True)
class FeatureSpec:
    feature_id: str
    name: str
    domain: str
    needs: tuple = ()
    capabilities: tuple = ()
    limitations: tuple = ()


# Minimal prerequisites for selecting a plausible next validation step, NOT a
# promise of full native support. More-specific runtime requirements remain
# authoritative in each measurement protocol and must be validated at execution.
_ROWS = (
    ('P01', 'Inference latency P50/P95/P99', 'inference', ('input',), (), ()),
    ('P02', 'Completed inference throughput', 'inference', ('input',), (), ('images/s is not camera FPS',)),
    ('P03', 'End-to-end pipeline latency', 'inference', ('input', 'stages'), (), ('serial pipeline scope',)),
    ('P04', 'Preprocessing / postprocessing', 'inference', ('input', 'stages'), (), ()),
    ('P05', 'Cold start and model loading', 'inference', ('input', 'factory', 'checkpoint'), (), ('OS page-cache coldness unproven',)),
    ('P06', 'Batch-size sweep', 'inference', ('input',), (), ('finite explicit points, not global optimal batch',)),
    ('P07', 'Resolution sweep', 'inference', ('input',), (), ('resolution must preserve task semantics',)),
    ('P08', 'Precision benchmark', 'inference', ('input',), (), ('numeric quality is assessed separately',)),
    ('P09', 'CUDA allocator peak memory', 'memory', ('input',), ('gpu',), ('allocated/reserved is not total physical VRAM',)),
    ('P10', 'Process RSS and sampled peak', 'memory', ('input',), (), ('sampled peak is not exact lifetime maximum',)),
    ('P11', 'Model tensor and artifact footprint', 'memory', (), (), ('logical bytes are not resident RAM',)),
    ('P12', 'FLOPs / MACs', 'profiling', ('input',), (), ('unknown operator cost cannot be counted as zero',)),
    ('P13', 'Module/layer profiling', 'profiling', ('input',), (), ('CUDA module-to-kernel attribution not established',)),
    ('P14', 'Operator/kernel profiling', 'profiling', ('input',), (), ('native CUDA kernel attribution may be unsupported',)),
    ('P15', 'CPU-GPU transfer analysis', 'device', ('input',), ('gpu',), ('copy/compute overlap not certified',)),
    ('P16', 'CPU/GPU utilization', 'device', ('input',), (), ('GPU board utilization is not model-exclusive',)),
    ('P17', 'Maximum feasible batch search', 'inference', ('input', 'factory', 'resource_policy'), (), ('finite bounded search; resource policy required',)),
    ('P18', 'Arithmetic intensity / roofline', 'profiling', ('input', 'target_counters'), ('gpu', 'ncu'), ('reference-kernel counters do not prove target-model roofline',)),
    ('P19', 'Epoch / training step timing', 'training', ('training',), (), ()),
    ('P20', 'Forward / backward / optimizer stages', 'training', ('training',), (), ()),
    ('P21', 'DataLoader bottleneck', 'training', ('training',), (), ('worker service/prefetch scope may be partial',)),
    ('P22', 'Training throughput', 'training', ('training',), (), ()),
    ('P23', 'Training memory analysis', 'training', ('training',), (), ('CUDA saved-activation coverage incomplete',)),
    ('P24', 'Time to target accuracy', 'training', ('training', 'labels'), (), ('censored unattained targets must be retained',)),
    ('P25', 'Gradient / AMP diagnostics', 'training', ('training',), (), ('sparse/custom gradient coverage limited',)),
    ('P26', 'Accuracy / precision / recall / F1', 'quality', ('input', 'labels'), (), ('classification metrics are not detection mAP or segmentation IoU',)),
    ('P27', 'Robustness measurements', 'quality', ('input', 'labels'), (), ('perturbations must be task-appropriate',)),
    ('P28', 'Precision accuracy change', 'quality', ('input', 'labels'), (), ('requires matched baseline and candidate predictions',)),
    ('P29', 'Quality/speed/memory tradeoff', 'comparison', ('input', 'labels', 'matched_evidence'), (), ('observed difference is not verified improvement',)),
    ('P30', 'Live camera/recorded video FPS and drops', 'streaming', ('frame_adapter', 'stream'), (), ('only application queue drops measured',)),
    ('P31', 'Uninterrupted long-run stability', 'stability', ('input', 'factory', '7200_authorization'), (), ('must be one continuous 7200-second run',)),
    ('P32', 'Memory growth analysis', 'stability', ('retained_7200',), (), ('growth alone is not a causal leak',)),
    ('P33', 'Energy per inference', 'energy', ('input', 'matched_power_window'), ('gpu', 'power_sensor'), ('board energy is not model-exclusive',)),
    ('P34', 'Multiple-device comparison', 'comparison', ('input',), ('multiple_devices',), ('same method and distinct actual devices required',)),
    ('P35', 'ONNX Runtime node/provider placement', 'provider', ('onnx_model',), ('ort_cpu',), ('provider list alone does not prove execution',)),
)
CATALOG = tuple(FeatureSpec(*row) for row in _ROWS)
SPECS = {row.feature_id: row for row in CATALOG}
assert len(SPECS) == len(CATALOG) == 35
assert tuple(SPECS) == tuple('P%02d' % i for i in range(1, 36))

# Minimum evidence prerequisites, not a runtime scheduler. Dependencies are
# visible in the plan; selected modes never silently run additional features.
DEPENDENCIES = {
    'P24': ('P19', 'P26'), 'P28': ('P08', 'P26'),
    'P29': ('P01', 'P02', 'P09', 'P10', 'P26'),
    'P32': ('P31',), 'P33': ('P16',),
}
GOAL_FEATURES = {
    'latency': ('P01', 'P03', 'P04', 'P05'),
    'throughput': ('P02', 'P06', 'P07', 'P17'),
    'memory': ('P09', 'P10', 'P11'),
    'bottleneck': ('P12', 'P13', 'P14', 'P15', 'P16', 'P18', 'P21'),
    'quality': ('P26', 'P27', 'P28'),
    'training': ('P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25'),
    'stability': ('P16', 'P31', 'P32'),
    'energy': ('P16', 'P33'),
    'providers': ('P35',),
    'comparison': ('P01', 'P02', 'P09', 'P10', 'P11', 'P26', 'P29'),
    'streaming': ('P30',),
}


@dataclass(frozen=True)
class FeaturePlanRequest:
    mode: str
    models: tuple
    features: tuple = ()
    goals: tuple = ()
    capabilities: tuple = ()

    def __post_init__(self):
        if type(self.mode) is not str or self.mode not in _MODES:
            raise FeaturePlanError('invalid_plan_mode')
        if type(self.models) not in (tuple, list) or not (1 <= len(self.models) <= 2):
            raise FeaturePlanError('one_or_two_models_required')
        if any(type(m) is not ModelContext for m in self.models):
            raise FeaturePlanError('typed_model_context_required')
        if len({m.model_id for m in self.models}) != len(self.models):
            raise FeaturePlanError('duplicate_model_id')
        object.__setattr__(self, 'models', tuple(self.models))
        for field, allowed in (('features', SPECS), ('goals', _GOALS),
                               ('capabilities', _CAPABILITIES)):
            object.__setattr__(self, field, _tokens(getattr(self, field), allowed, field))
        if self.mode == 'single' and (len(self.features) != 1 or self.goals):
            raise FeaturePlanError('single_mode_requires_exactly_one_feature')
        if self.mode == 'selected' and (not self.features or self.goals):
            raise FeaturePlanError('selected_mode_requires_features')
        if self.mode == 'automatic' and (not self.goals or self.features):
            raise FeaturePlanError('automatic_mode_requires_goals')
        if self.mode == 'full' and (self.features or self.goals):
            raise FeaturePlanError('full_mode_has_no_selection_arguments')


@dataclass(frozen=True)
class ModelAssessment:
    model_id: str
    status: str  # PROPOSED / BLOCKED / NOT_APPLICABLE
    reasons: tuple


@dataclass(frozen=True)
class PlannedFeature:
    feature_id: str
    name: str
    domain: str
    selection: str  # requested / recommended / prerequisite
    models: tuple
    comparison: str  # NOT_REQUESTED / PENDING_COMPATIBILITY_EVIDENCE / BLOCKED
    limitations: tuple
    authorization: str = 'NOT_AUTHORIZED'
    execution: str = 'NOT_RUN'
    acceptance: str = 'NOT_EVALUATED'


@dataclass(frozen=True)
class FeaturePlan:
    mode: str
    models: tuple
    features: tuple
    goals: tuple
    capability_basis: str = 'CALLER_DECLARED_NOT_VERIFIED'
    execution_authorized: bool = False
    comparison_authority: str = 'NO_NATIVE_COMPARISON_OR_CERTIFIED_WINNER'

    def to_dict(self):
        return asdict(self)


def _missing(model, spec, capabilities):
    reasons = []
    if model.backend == 'onnx' and spec.feature_id != 'P35':
        return 'NOT_APPLICABLE', ('ONNX backend has no validated native producer for this P feature in planner v1',)
    if model.backend == 'pytorch' and model.task != 'classification' and spec.feature_id != 'P35':
        if not model.adapter_contract:
            reasons.append('non_classification_model_requires_task_specific_adapter')
    if spec.domain == 'training' and model.backend != 'pytorch':
        return 'NOT_APPLICABLE', ('ONNX inference is not a training workload',)
    for required in spec.needs:
        if required == 'input' and not model.input_contract:
            reasons.append('missing_declared_input_contract')
        elif required == 'stages' and not model.stage_adapters:
            reasons.append('missing_declared_pipeline_stage_adapters')
        elif required == 'factory' and not model.trusted_factory:
            reasons.append('missing_trusted_factory_contract')
        elif required == 'checkpoint' and not model.checkpoint:
            reasons.append('missing_checkpoint_contract')
        elif required == 'training' and not model.training_workload:
            reasons.append('missing_training_workload')
        elif required == 'labels' and not model.labeled_evaluation:
            reasons.append('missing_labeled_evaluation_cohort')
        elif required == 'resource_policy':
            reasons.append('bounded_worker_resource_policy_not_attested')
        elif required == 'target_counters':
            reasons.append('target_model_kernel_counter_binding_not_attested')
        elif required == 'matched_evidence':
            reasons.append('matched_independent_performance_quality_memory_evidence_not_attested')
        elif required == 'frame_adapter' and not model.trained_frame_adapter:
            reasons.append('missing_task_appropriate_trained_frame_adapter')
        elif required == 'stream' and not (model.recorded_video or model.live_camera):
            reasons.append('missing_recorded_video_or_live_camera')
        elif required == '7200_authorization':
            reasons.append('uninterrupted_7200_second_run_requires_separate_authorization')
        elif required == 'retained_7200':
            if not model.retained_long_run:
                reasons.append('missing_retained_7200_second_source')
            else:
                reasons.append('retained_7200_source_requires_identity_and_continuity_validation')
        elif required == 'matched_power_window':
            reasons.append('matching_idle_active_model_power_windows_not_attested')
        elif required == 'onnx_model':
            if model.backend != 'onnx' and not model.onnx_export_contract:
                reasons.append('missing_declared_onnx_export_contract')
    for cap in spec.capabilities:
        if cap not in capabilities:
            reasons.append('missing_declared_capability:' + cap)
        else:
            reasons.append('runtime_capability_requires_native_verification:' + cap)
    if spec.feature_id == 'P30' and model.backend == 'onnx':
        return 'NOT_APPLICABLE', ('current streaming implementation has no ONNX frame processor',)
    if spec.feature_id == 'P26' and model.task != 'classification':
        reasons.append('classification_metrics_not_a_substitute_for_task_specific_quality')
    if spec.feature_id == 'P35' and model.backend == 'pytorch' and not model.onnx_export_contract:
        reasons.append('trained_pytorch_export_and_ort_execution_not_attested')
    # Capability verification and target evidence are blockers by design; the
    # planner must NEVER infer native readiness from caller declarations.
    if reasons:
        return 'BLOCKED', tuple(dict.fromkeys(reasons))
    return 'PROPOSED', ('prerequisites_caller_declared_only;native_validation_and_explicit_approval_required',)


def plan_features(request: FeaturePlanRequest) -> FeaturePlan:
    """Return a *planning-only* decision; never execute or authorize a workload."""
    if type(request) is not FeaturePlanRequest:
        raise FeaturePlanError('typed_feature_plan_request_required')
    reasons = {}
    if request.mode in ('single', 'selected'):
        for f in request.features:
            reasons[f] = 'requested'
    elif request.mode == 'full':
        for f in SPECS:
            reasons[f] = 'requested'
    else:
        for goal in request.goals:
            for f in GOAL_FEATURES[goal]:
                reasons.setdefault(f, 'recommended')
    # Expose all required feature measurements in the plan, without silently
    # approving them or falsely treating a newly proposed step as prior evidence.
    changed = True
    while changed:
        changed = False
        for f in tuple(reasons):
            for parent in DEPENDENCIES.get(f, ()):
                if parent not in reasons:
                    reasons[parent] = 'prerequisite'
                    changed = True
    rows = []
    for f in SPECS:
        if f not in reasons:
            continue
        spec = SPECS[f]
        per_model = []
        for model in request.models:
            status, why = _missing(model, spec, frozenset(request.capabilities))
            per_model.append(ModelAssessment(model.model_id, status, why))
        if len(per_model) == 1:
            comparison = 'NOT_REQUESTED'
        elif all(x.status == 'PROPOSED' for x in per_model):
            comparison = 'PENDING_COMPATIBILITY_EVIDENCE'
        else:
            comparison = 'BLOCKED'
        rows.append(PlannedFeature(f, spec.name, spec.domain, reasons[f], tuple(per_model),
                                   comparison, spec.limitations))
    return FeaturePlan(request.mode, request.models, tuple(rows), request.goals)
