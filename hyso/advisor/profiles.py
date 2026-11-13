"""Read-only, source-bound structure and P26 quality comparison, version 1.

Only schema-3 native evaluations are admitted. No model code, artifact paths,
checkpoint loaders or device APIs are followed. Source reports are retained once
per identity so original metric references remain independently resolvable.
"""
from dataclasses import asdict, dataclass
import json

from hyso.analysis import UnifiedAnalysisReport
from hyso.analysis._codec import canonical, digest, plain, read_json, export
from hyso.evaluation import EvaluationResult
from hyso.structure_comparison import ModelStructureComparison, compare_model_structures
from ._json import AdvisorError, Contract, register
from .comparison import P26BoundQualityComparisonContract
from .performance import PerformanceComparison, compare_performance
from .resolver import NativeMetric, NativeMetricResolver


_MAX_BYTES = 128 * 1024**2
_LIMITATIONS = (
    'sections_have_separate_eligibility_no_model_wide_verdict',
    'structure_is_registered_metadata_not_an_operator_graph_or_behavior_identity',
    'structure_describes_isolated_forward_receiver_at_recorded_boundaries',
    'original_model_structure_or_behavior_equivalence_not_certified',
    'structure_state_and_checkpoint_hashes_are_distinct_identities',
    'quality_is_descriptive_for_exact_observed_ordered_stream_only',
    'split_and_source_id_are_caller_declared_dataset_origin_and_preprocessing_not_verified',
    'imported_consistency_is_not_cryptographic_live_producer_authenticity',
    'distinct_evaluation_identities_do_not_certify_statistical_independence',
    'causality_significance_and_overall_superiority_not_established',
    'VERIFIED_IMPROVEMENT_and_P29_independent_acceptance_not_certified',
    'latency_throughput_RSS_and_P11_are_not_joined',
)


@register
@dataclass(frozen=True)
class P26ModelProfileComparisonContract(Contract):
    allowed_changes: tuple = ()
    min_samples: int = 1
    policy: str = 'p26_bound_model_profiles_v1'

    def __post_init__(self):
        quality = P26BoundQualityComparisonContract(
            allowed_changes=self.allowed_changes, min_samples=self.min_samples)
        object.__setattr__(self, 'allowed_changes', quality.allowed_changes)
        if self.policy != 'p26_bound_model_profiles_v1':
            raise AdvisorError('unsupported_model_profile_policy')


@dataclass(frozen=True)
class BoundEvaluationReference:
    """Derived source addresses, validated against retained reports by the reader.

    A reference alone is not an attestation. Checkpoint SHA may be absent and
    never falls back to a structure or model-state SHA.
    """
    report_id: str
    report_sha256: str
    record_id: str
    artifact_sha256: str
    payload_sha256: str
    evaluation_id: str
    evaluation_source_sha256: str
    structure_sha256: str
    model_state_method: str
    model_state_sha256: str
    checkpoint_status: str
    checkpoint_sha256: str | None
    binding_method: str
    binding_scope: str
    snapshot_path: str = 'native_payload.structure_binding.snapshot'


def _source(resolver, record_id):
    if type(resolver) is not NativeMetricResolver:
        raise AdvisorError('validated_native_resolvers_required')
    r = resolver.record(record_id)
    n = None if r is None else r['native_payload']
    if (r is None or r['feature_id'] != 'P26' or r['validation']['status'] != 'native_validated'
            or not n or n.get('format') != 'hyso.evaluation.classification'
            or type(n.get('schema_version')) is not int or n['schema_version'] != 3):
        raise AdvisorError('model_profiles_require_native_p26_schema3')
    result = EvaluationResult(n)  # includes raw/metric/accounting/snapshot/peer checks
    b = n['structure_binding']
    ref = BoundEvaluationReference(resolver.source_report_id, resolver.source_report_sha256,
        record_id, r['artifact']['sha256'], r['source']['payload_sha256'], n['evaluation_id'],
        n['source_sha256'], b['structure_sha256'], b['model_state_method'], b['model_state_sha256'],
        b['checkpoint']['status'], b['checkpoint']['sha256'], b['method'], b['scope'])
    return ref, result.structure_snapshot


def _derive(left, left_id, right, right_id, contract):
    if type(contract) is not P26ModelProfileComparisonContract:
        raise AdvisorError('typed_model_profile_contract_required')
    a, sa = _source(left, left_id)
    b, sb = _source(right, right_id)
    if (a.record_id == b.record_id or a.evaluation_id == b.evaluation_id
            or a.artifact_sha256 == b.artifact_sha256 or a.payload_sha256 == b.payload_sha256):
        raise AdvisorError('model_profiles_require_distinct_evaluations')
    structure = compare_model_structures(sa, sb)
    # The original D.2 object/hash remains unchanged. Only its wire projection
    # omits snapshots already present in the native schema-3 source reports.
    structure_data = structure.to_dict()
    del structure_data['source_snapshots']
    quality = {}
    typed_quality = []
    for name, metric in (('accuracy', NativeMetric.TEST_ACCURACY), ('macro_f1', NativeMetric.TEST_MACRO_F1)):
        qc = P26BoundQualityComparisonContract(metric, contract.allowed_changes, contract.min_samples)
        comparison = compare_performance(left, left_id, right, right_id, contract=qc)
        typed_quality.append(comparison)
        quality[name] = dict(evidence_level='native_derived_recorded_quality',
            scope=qc.policy, comparison=comparison.to_dict())
    reports = {r.source_report_id: r.report for r in (left, right)}
    data = plain(dict(format='hyso.unified_model_comparison', schema_version=1,
        contract=contract.to_dict(), source_reports={k: reports[k] for k in sorted(reports)},
        baseline=asdict(a), candidate=asdict(b),
        structure=dict(eligibility='COMPARABLE', evidence_level='native_bound_registered_metadata',
            scope='literal_registered_path_structure_v1', baseline=asdict(a), candidate=asdict(b),
            comparison=structure_data), quality=quality, limitations=_LIMITATIONS))
    data['content_sha256'] = digest(data)
    return data, structure, tuple(typed_quality), a, b


@dataclass(frozen=True, init=False)
class UnifiedModelComparison:
    """Validated immutable result; every section is rederived on import.

    Retained reports keep original EvidenceRefs intact; shared reports and bound
    snapshots are stored once. The structure wire projection is not a standalone
    D.2 JSON object; .structure_comparison returns the complete validated object.
    """
    _payload: str
    _structure: ModelStructureComparison
    _accuracy: PerformanceComparison
    _macro_f1: PerformanceComparison
    _baseline: BoundEvaluationReference
    _candidate: BoundEvaluationReference

    def __init__(self, data):
        try:
            d = plain(data)
            raw = canonical(d)
            if len(raw.encode('utf-8')) > _MAX_BYTES:
                raise AdvisorError('model_profile_size_limit')
            if (d.get('format') != 'hyso.unified_model_comparison'
                    or type(d.get('schema_version')) is not int or d['schema_version'] != 1):
                raise AdvisorError('unsupported_model_profile_version')
            sources = d['source_reports']
            if type(sources) is not dict or not 1 <= len(sources) <= 2:
                raise AdvisorError('invalid_model_profile_sources')
            resolvers = {}
            for key, report in sources.items():
                resolver = NativeMetricResolver(UnifiedAnalysisReport(report))
                if key != resolver.source_report_id:
                    raise AdvisorError('model_profile_source_identity_mismatch')
                resolvers[key] = resolver
            contract = P26ModelProfileComparisonContract.from_dict(d['contract'])
            a, b = d['baseline'], d['candidate']
            expected, structure, quality, ar, br = _derive(
                resolvers[a['report_id']], a['record_id'],
                resolvers[b['report_id']], b['record_id'], contract)
            # Rederivation verifies identities, refs, eligibility, all differences
            # and limitations, even when an attacker recomputes every outer hash.
            if raw != canonical(expected):
                raise AdvisorError('model_profile_derivation_mismatch')
        except AdvisorError:
            raise
        except (ValueError, KeyError, TypeError, AttributeError, IndexError, RecursionError,
                RuntimeError, OverflowError) as exc:
            raise AdvisorError('invalid_model_profile_evidence') from exc
        for field, value in (('_payload', raw), ('_structure', structure),
                ('_accuracy', quality[0]), ('_macro_f1', quality[1]), ('_baseline', ar), ('_candidate', br)):
            object.__setattr__(self, field, value)

    @property
    def baseline(self): return self._baseline

    @property
    def candidate(self): return self._candidate

    @property
    def structure_comparison(self): return self._structure

    @property
    def structure_eligibility(self): return 'COMPARABLE'

    @property
    def accuracy_comparison(self): return self._accuracy

    @property
    def macro_f1_comparison(self): return self._macro_f1

    @property
    def contract(self): return P26ModelProfileComparisonContract.from_dict(self.to_dict()['contract'])

    @property
    def limitations(self): return tuple(self.to_dict()['limitations'])

    @property
    def content_sha256(self): return self.to_dict()['content_sha256']

    def to_dict(self): return json.loads(self._payload)

    def to_json(self, path=None):
        raw = self._payload + '\n'
        if path is not None: export(path, raw, overwrite=False)
        return raw

    @classmethod
    def from_dict(cls, data): return cls(data)

    @classmethod
    def from_json(cls, path):
        data, _, _ = read_json(path, max_bytes=_MAX_BYTES)
        return cls(data)


def compare_p26_model_profiles(baseline_resolver, baseline_record_id,
                               candidate_resolver, candidate_record_id, *, contract=None):
    """Compare two distinct, bound native evaluations without rerunning models.

    Structure eligibility is independent of quality. Accuracy and macro-F1 use
    the existing safety gate under a separate schema-3 policy. Without explicit
    model-change permission they never produce an eligible numerical difference.
    """
    contract = P26ModelProfileComparisonContract() if contract is None else contract
    data, *_ = _derive(baseline_resolver, baseline_record_id,
                       candidate_resolver, candidate_record_id, contract)
    return UnifiedModelComparison(data)
