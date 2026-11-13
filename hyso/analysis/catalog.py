"""Public descriptive catalog, profiles and executor coverage; no hardware queries."""
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
        return asdict(self)


def feature_catalog(feature_id: str | None = None) -> tuple[AnalysisFeature, ...] | AnalysisFeature:
    """Describe all 35 features, or one ID. Does not load models or probe devices."""
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
