"""Public descriptive catalog, profiles and executor coverage; no hardware queries."""
import typing as _typing
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

from .feature_plan import CATALOG, DEPENDENCIES, GOAL_FEATURES, FeaturePlanError


PROFILES = MappingProxyType({
    'basic': ('P01', 'P02', 'P10', 'P11'),
    'inference': tuple('P%02d' % i for i in range(1, 9)),
    'training': GOAL_FEATURES['training'],
    'memory': ('P09', 'P10', 'P11', 'P23', 'P32'),
    'comparison': GOAL_FEATURES['comparison'],
    'full': tuple(spec.feature_id for spec in CATALOG),
})
EXECUTOR_FEATURES = MappingProxyType({
    'trusted_cpu': ('P01', 'P02', 'P10', 'P11'),
    'isolated_cpu': ('P02', 'P11'),
    'native': tuple(spec.feature_id for spec in CATALOG),
})
NATIVE_CUDA_FEATURES = frozenset(('P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08',
    'P09', 'P12', 'P13', 'P14', 'P15', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22',
    'P23', 'P25', 'P26', 'P28', 'P31', 'P32', 'P33', 'P34', 'P35'))
_SCOPES = {
    'P01': 'resident CPU benchmark samples, then native latency analysis; tail policy unchanged',
    'P02': 'existing G2/G3 completed CPU inference throughput',
    'P10': 'one current-process RSS/HWM snapshot; no inference or model-only peak attribution',
    'P11': 'existing G2/G3 logical CPU tensor footprint; no artifact serialization',
}


@dataclass(frozen=True)
class AnalysisFeature:
    """Catalog metadata, not native readiness or an acceptance certificate.

    supported_devices/modes refer only to the common executor. Empty coverage
    does not mean the original performance API is absent. Requirements and
    applicability still need plan and runtime validation.

    Args:
        feature_id (str): Exact P01-P35 feature identifier. Required.
        name (str): Registered component name, exact metric label, or stored entity name in this API. Required.
        description (str): Human-readable specification description. Required.
        domain (str): Feature/metric domain, separate from runtime support. Required.
        requirements (tuple[str, ...]): Explicit prerequisite contracts for the operation. Required.
        required_capabilities (tuple[str, ...]): Explicit prerequisite capabilities; declarations are not runtime proof. Required.
        supported_devices (tuple[str, ...]): Declared native device support; current acceptance/admission is separate. Required.
        executor_modes (tuple[str, ...]): Explicit supported execution profiles for this feature. Required.
        cost_class (str): Native cost category of the feature/operation. Required.
        dependencies (tuple[str, ...]): Explicit prerequisite identities for this proposed step. Required.
        applicability (str): Conditions under which the finding/evidence applies. Required.
        executor_scope (str): Native execution boundary of this feature. Required.
        limitations (tuple[str, ...]): Explicit limitations of the evidence, comparison or proposed operation. Required.

    Returns:
        An instance of AnalysisFeature with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        AnalysisFeature(feature_id=feature_id, name=name, description=description, domain=domain, requirements=requirements, required_capabilities=required_capabilities, supported_devices=supported_devices, executor_modes=executor_modes, cost_class=cost_class, dependencies=dependencies, applicability=applicability, executor_scope=executor_scope, limitations=limitations)

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """
    feature_id: str
    name: str
    description: str
    domain: str
    requirements: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    supported_devices: tuple[str, ...]
    executor_modes: tuple[str, ...]
    cost_class: str
    dependencies: tuple[str, ...]
    applicability: str
    executor_scope: str
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached plain-data view of the stored state.

        Returns:
            A detached plain-data view of this validated object; native units and unavailable values are retained.

        Raises:
            No additional validation for a valid object; corrupt retained state or native copy/serialization failures propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_dict()

        HYSo Guide: analyzer.workflow
        Offline: hyso docs show analyzer.workflow --lang en
        Open: hyso docs open analyzer.workflow --lang en
        """
        return asdict(self)


def feature_catalog(feature_id: _typing.Literal['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35'] | None = None) -> tuple[AnalysisFeature, ...] | AnalysisFeature:
    """Describe all 35 features, or one ID. Does not load models or probe devices.

    Args:
        feature_id (str | None): Exact P01-P35 feature identifier. Default: None.

    Returns:
        The retained feature catalog value, without recomputing measurements.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        feature_catalog()

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """
    result = []
    for spec in CATALOG:
        modes = tuple(mode for mode, ids in EXECUTOR_FEATURES.items() if spec.feature_id in ids)
        cost = ('training' if spec.domain == 'training' else
                'long_7200s' if spec.feature_id == 'P31' else
                'offline_analysis' if spec.feature_id == 'P32' else
                'snapshot' if spec.feature_id == 'P10' else
                'metadata' if spec.feature_id == 'P11' else
                'multi_point' if spec.feature_id in ('P06','P07','P08','P17','P27','P28','P29','P34') else
                'profiling' if spec.domain in ('profiling','provider') else
                'streaming' if spec.feature_id == 'P30' else
                'sensor_window' if spec.feature_id == 'P33' else 'inference')
        applicability = ('requires CUDA and runtime-verified ' + ', '.join(spec.capabilities)
                         if 'gpu' in spec.capabilities else 'producer-specific task/backend and protocol prerequisites')
        description = spec.name + ('. ' + '; '.join(spec.limitations) if spec.limitations else '.')
        result.append(AnalysisFeature(spec.feature_id, spec.name, description, spec.domain,
            tuple(spec.needs), tuple(spec.capabilities),
            ('cpu', 'cuda') if spec.feature_id in NATIVE_CUDA_FEATURES else ('cpu',) if modes else (), modes, cost,
            DEPENDENCIES.get(spec.feature_id, ()), applicability,
            _SCOPES.get(spec.feature_id, 'typed native operation; source, protocol and explicit execution admission required'),
            tuple(spec.limitations)))
    if feature_id is None:
        return tuple(result)
    for row in result:
        if row.feature_id == feature_id:
            return row
    raise FeaturePlanError('invalid_feature_id')
