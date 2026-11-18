"""Read-only, source-bound structure and P26 quality comparison, version 1.

Only schema-3 native evaluations are admitted. No model code, artifact paths,
checkpoint loaders or device APIs are followed. Source reports are retained once
per identity so original metric references remain independently resolvable.
"""
import typing as _typing
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
    """Declare the P26 evaluation alignment required to compare two model profiles.

    Args:
        allowed_changes (tuple): Explicit model/configuration changes that an advisory proposal may suggest; none are applied automatically. Default: ().
        min_samples (int): Minimum observations required for the statistic. Default: 1.
        policy (str): Existing protocol/resource policy; defaults do not grant additional consent. Default: 'p26_bound_model_profiles_v1'.

    Returns:
        An instance of P26ModelProfileComparisonContract with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        P26ModelProfileComparisonContract()

    HYSo Guide: comparison.models
    Offline: hyso docs show comparison.models --lang en
    Open: hyso docs open comparison.models --lang en
    """
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

    Args:
        report_id (str): Exact retained report identity. Required.
        report_sha256 (str): Exact SHA-256 digest of the report. Required.
        record_id (str): Exact retained evidence identity. Required.
        artifact_sha256 (str): Exact serialized artifact digest. Required.
        payload_sha256 (str): Exact SHA-256 digest of the payload. Required.
        evaluation_id (str): Exact evaluation identity. Required.
        evaluation_source_sha256 (str): Exact SHA-256 digest of the evaluation source. Required.
        structure_sha256 (str): Exact SHA-256 digest of the structure. Required.
        model_state_method (str): Method used to identify the model state. Required.
        model_state_sha256 (str): Exact SHA-256 digest of the model state. Required.
        checkpoint_status (str): Retained checkpoint availability/compatibility status. Required.
        checkpoint_sha256 (str | None): Exact SHA-256 digest of the checkpoint. Required.
        binding_method (str): Method used to bind this evidence to its source. Required.
        binding_scope (str): Exact scope within which the binding is valid. Required.
        snapshot_path (str): Local retained structure-snapshot path. Default: 'native_payload.structure_binding.snapshot'.

    Returns:
        An instance of BoundEvaluationReference with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        BoundEvaluationReference(report_id=report_id, report_sha256=report_sha256, record_id=record_id, artifact_sha256=artifact_sha256, payload_sha256=payload_sha256, evaluation_id=evaluation_id, evaluation_source_sha256=evaluation_source_sha256, structure_sha256=structure_sha256, model_state_method=model_state_method, model_state_sha256=model_state_sha256, checkpoint_status=checkpoint_status, checkpoint_sha256=checkpoint_sha256, binding_method=binding_method, binding_scope=binding_scope)

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en
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

    Args:
        data (dict[str, _typing.Any]): Detached plain-data payload accepted by the existing version/schema validator. Required.

    Returns:
        An instance of UnifiedModelComparison with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        UnifiedModelComparison(data)

    HYSo Guide: comparison.models
    Offline: hyso docs show comparison.models --lang en
    Open: hyso docs open comparison.models --lang en
    """
    _payload: str
    _structure: ModelStructureComparison
    _accuracy: PerformanceComparison
    _macro_f1: PerformanceComparison
    _baseline: BoundEvaluationReference
    _candidate: BoundEvaluationReference

    def __init__(self, data: 'dict[str, _typing.Any]') -> 'None':
        """Initialize UnifiedModelComparison with the supplied settings.

        Args:
            data (dict[str, _typing.Any]): Detached plain-data payload accepted by the existing version/schema validator. Required.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            UnifiedModelComparison(data)

        HYSo Guide: comparison.models
        Offline: hyso docs show comparison.models --lang en
        Open: hyso docs open comparison.models --lang en
        """
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
    def baseline(self) -> '_typing.Any':
        """Read the retained baseline from this UnifiedModelComparison record.

        Returns:
            The retained baseline value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.baseline

        HYSo Guide: comparison.models
        Offline: hyso docs show comparison.models --lang en
        Open: hyso docs open comparison.models --lang en
        """
        return self._baseline

    @property
    def candidate(self) -> '_typing.Any':
        """Read the retained candidate from this UnifiedModelComparison record.

        Returns:
            The retained candidate value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.candidate

        HYSo Guide: comparison.models
        Offline: hyso docs show comparison.models --lang en
        Open: hyso docs open comparison.models --lang en
        """
        return self._candidate

    @property
    def structure_comparison(self) -> '_typing.Any':
        """Read the retained structure comparison from this UnifiedModelComparison record.

        Returns:
            The retained structure comparison value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.structure_comparison

        HYSo Guide: comparison.models
        Offline: hyso docs show comparison.models --lang en
        Open: hyso docs open comparison.models --lang en
        """
        return self._structure

    @property
    def structure_eligibility(self) -> '_typing.Any':
        """Read the retained structure eligibility from this UnifiedModelComparison record.

        Returns:
            The retained structure eligibility value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.structure_eligibility

        HYSo Guide: comparison.models
        Offline: hyso docs show comparison.models --lang en
        Open: hyso docs open comparison.models --lang en
        """
        return 'COMPARABLE'

    @property
    def accuracy_comparison(self) -> '_typing.Any':
        """Read the retained accuracy comparison from this UnifiedModelComparison record.

        Returns:
            The retained accuracy comparison value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.accuracy_comparison

        HYSo Guide: comparison.models
        Offline: hyso docs show comparison.models --lang en
        Open: hyso docs open comparison.models --lang en
        """
        return self._accuracy

    @property
    def macro_f1_comparison(self) -> '_typing.Any':
        """Read the retained macro f1 comparison from this UnifiedModelComparison record.

        Returns:
            The retained macro f1 comparison value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.macro_f1_comparison

        HYSo Guide: comparison.models
        Offline: hyso docs show comparison.models --lang en
        Open: hyso docs open comparison.models --lang en
        """
        return self._macro_f1

    @property
    def contract(self) -> '_typing.Any':
        """Read the retained contract from this UnifiedModelComparison record.

        Returns:
            The retained contract value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.contract

        HYSo Guide: comparison.models
        Offline: hyso docs show comparison.models --lang en
        Open: hyso docs open comparison.models --lang en
        """
        return P26ModelProfileComparisonContract.from_dict(self.to_dict()['contract'])

    @property
    def limitations(self) -> '_typing.Any':
        """Read the retained limitations from this UnifiedModelComparison record.

        Returns:
            The retained limitations value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.limitations

        HYSo Guide: comparison.models
        Offline: hyso docs show comparison.models --lang en
        Open: hyso docs open comparison.models --lang en
        """
        return tuple(self.to_dict()['limitations'])

    @property
    def content_sha256(self) -> '_typing.Any':
        """Return the canonical content digest of this retained record.

        Returns:
            The canonical SHA-256 digest string of this retained content.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.content_sha256

        HYSo Guide: comparison.models
        Offline: hyso docs show comparison.models --lang en
        Open: hyso docs open comparison.models --lang en
        """
        return self.to_dict()['content_sha256']

    def to_dict(self) -> 'dict[str, _typing.Any]':
        """Return a detached plain-data view of the stored state.

        Returns:
            A detached plain-data view of this validated object; native units and unavailable values are retained.

        Raises:
            No additional validation for a valid object; corrupt retained state or native copy/serialization failures propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_dict()

        HYSo Guide: comparison.models
        Offline: hyso docs show comparison.models --lang en
        Open: hyso docs open comparison.models --lang en
        """
        return json.loads(self._payload)

    def to_json(self, path: '_typing.Any'=None) -> '_typing.Any':
        """Serialize this object using its native JSON format and overwrite policy.

        Args:
            path (_typing.Any): Filesystem path for the specific reader/exporter; no configuration code is executed. Default: None.

        Returns:
            A JSON string when path is None, otherwise None after writing the file.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_json()

        HYSo Guide: comparison.models
        Offline: hyso docs show comparison.models --lang en
        Open: hyso docs open comparison.models --lang en
        """
        raw = self._payload + '\n'
        if path is not None: export(path, raw, overwrite=False)
        return raw

    @classmethod
    def from_dict(cls, data: 'dict[str, _typing.Any]') -> 'UnifiedModelComparison':
        """Validate plain retained data and reconstruct the typed result.

        Args:
            data (dict[str, _typing.Any]): Detached plain-data payload accepted by the existing version/schema validator. Required.

        Returns:
            A validated UnifiedModelComparison instance reconstructed from the supplied data.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            UnifiedModelComparison.from_dict(data)

        HYSo Guide: comparison.models
        Offline: hyso docs show comparison.models --lang en
        Open: hyso docs open comparison.models --lang en
        """
        return cls(data)

    @classmethod
    def from_json(cls, path: '_typing.Any') -> 'UnifiedModelComparison':
        """Read and validate this type from a local JSON file.

        Args:
            path (_typing.Any): Filesystem path for the specific reader/exporter; no configuration code is executed. Required.

        Returns:
            A validated UnifiedModelComparison instance reconstructed from the supplied data.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            UnifiedModelComparison.from_json(path)

        HYSo Guide: comparison.models
        Offline: hyso docs show comparison.models --lang en
        Open: hyso docs open comparison.models --lang en
        """
        data, _, _ = read_json(path, max_bytes=_MAX_BYTES)
        return cls(data)


def compare_p26_model_profiles(baseline_resolver: '_typing.Any', baseline_record_id: '_typing.Any',
                               candidate_resolver: '_typing.Any', candidate_record_id: '_typing.Any', *, contract: '_typing.Any'=None) -> '_typing.Any':
    """Compare two distinct, bound native evaluations without rerunning models.

    Structure eligibility is independent of quality. Accuracy and macro-F1 use
    the existing safety gate under a separate schema-3 policy. Without explicit
    model-change permission they never produce an eligible numerical difference.

    Args:
        baseline_resolver (_typing.Any): Native metric resolver for baseline evidence. Required.
        baseline_record_id (_typing.Any): Exact baseline evidence-record identity. Required.
        candidate_resolver (_typing.Any): Native metric resolver for candidate evidence. Required.
        candidate_record_id (_typing.Any): Exact candidate evidence-record identity. Required.
        contract (_typing.Any): Typed comparison/evaluation contract required by the operation. Default: None.

    Returns:
        The retained compare p26 model profiles value, without recomputing measurements.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        compare_p26_model_profiles(baseline_resolver, baseline_record_id, candidate_resolver, candidate_record_id)

    HYSo Guide: comparison.models
    Offline: hyso docs show comparison.models --lang en
    Open: hyso docs open comparison.models --lang en
    """
    contract = P26ModelProfileComparisonContract() if contract is None else contract
    data, *_ = _derive(baseline_resolver, baseline_record_id,
                       candidate_resolver, candidate_record_id, contract)
    return UnifiedModelComparison(data)
