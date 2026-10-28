"""Descriptive differences gated by the existing comparison contract.

No workloads, model objects, significance tests or optimization certificates.
"""
from dataclasses import dataclass
from decimal import Decimal, localcontext
import math

from ._json import AdvisorError, Contract, register
from .contracts import _enum, _strings
from .comparison import (ComparisonContract, P26QualityComparisonContract,
                         ComparisonResult, ComparisonStatus, compare_records)
from .resolver import NativeMetric, ResolvedMetric


# Closed semantics: a caller cannot select a more favorable direction or unit.
_SEMANTICS = {
    NativeMetric.FORWARD_P50: ('ms', 'latency', 'lower'),
    NativeMetric.FORWARD_P95: ('ms', 'latency', 'lower'),
    NativeMetric.FORWARD_P99: ('ms', 'latency', 'lower'),
    NativeMetric.COMPLETED_IMAGES_PER_SECOND: ('images/s', 'throughput', 'higher'),
    NativeMetric.RSS_BASELINE: ('byte', 'process_rss', 'lower'),
    NativeMetric.RSS_CURRENT: ('byte', 'process_rss', 'lower'),
    NativeMetric.RSS_SAMPLED_PEAK: ('byte', 'process_rss', 'lower'),
    NativeMetric.PARAMETER_BYTES: ('byte', 'model_logical_bytes', 'lower'),
    NativeMetric.BUFFER_BYTES: ('byte', 'model_logical_bytes', 'lower'),
    NativeMetric.MODEL_LOGICAL_BYTES: ('byte', 'model_logical_bytes', 'lower'),
    NativeMetric.ARTIFACT_BYTES: ('byte', 'serialized_artifact_bytes', 'lower'),
    NativeMetric.TEST_ACCURACY: ('fraction', 'quality', 'higher'),
    NativeMetric.TEST_MACRO_F1: ('fraction', 'quality', 'higher'),
}


def _numbers(baseline, candidate, quality):
    for value in (baseline, candidate):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or quality and value > 1:
            raise AdvisorError('invalid_performance_value')
    with localcontext() as ctx:
        ctx.prec = 40
        a, b = Decimal(str(baseline)), Decimal(str(candidate))
        change = b - a
        values = (change, None if not a else change / a * 100,
                  change * 100 if quality else None)
        result = tuple(None if v is None else float(v) for v in values)
    if any(v is not None and not math.isfinite(v) for v in result):
        raise AdvisorError('nonfinite_performance_difference')
    return result


@register
@dataclass(frozen=True)
class MetricDifference(Contract):
    """Values and source contexts; direct construction is not certification."""
    baseline: ResolvedMetric
    candidate: ResolvedMetric
    difference: float
    percent_change: float | None
    percentage_point_change: float | None
    preferred_direction: str
    observed_direction: str
    metric_effect: str

    def __post_init__(self):
        if type(self.baseline) is not ResolvedMetric or type(self.candidate) is not ResolvedMetric:
            raise AdvisorError('typed_native_metrics_required')
        a, b = self.baseline, self.candidate
        if a.metric != b.metric or a.metric not in _SEMANTICS:
            raise AdvisorError('unsupported_or_different_metrics')
        unit, domain, preferred = _SEMANTICS[a.metric]
        if any(m.reference.unit != unit or m.reference.domain != domain for m in (a,b)):
            raise AdvisorError('metric_unit_or_domain_mismatch')
        if (a.native_context.get('measurement_scope') != b.native_context.get('measurement_scope') or
                a.protocol != b.protocol):
            raise AdvisorError('different_measurement_scopes')
        expected = _numbers(a.value, b.value, domain == 'quality')
        actual = (self.difference, self.percent_change, self.percentage_point_change)
        if any(v is not None and type(v) not in (int,float) for v in actual) or actual != expected:
            raise AdvisorError('incorrect_performance_arithmetic')
        observed = 'increase' if b.value > a.value else 'decrease' if b.value < a.value else 'unchanged'
        effect = ('unchanged' if observed == 'unchanged' else 'favorable' if
                  (observed == 'decrease') == (preferred == 'lower') else 'unfavorable')
        if (self.preferred_direction, self.observed_direction, self.metric_effect) != (preferred, observed, effect):
            raise AdvisorError('incorrect_metric_direction')


def _difference(a, b):
    unit, domain, preferred = _SEMANTICS[a.metric]
    values = _numbers(a.value, b.value, domain == 'quality')
    observed = 'increase' if b.value > a.value else 'decrease' if b.value < a.value else 'unchanged'
    effect = ('unchanged' if observed == 'unchanged' else 'favorable' if
              (observed == 'decrease') == (preferred == 'lower') else 'unfavorable')
    return MetricDifference(a, b, *values, preferred, observed, effect)


_LIMITS = ('single_metric_only', 'allowed_changes_limit_interpretation',
           'statistical_independence_not_certified', 'statistical_significance_not_assessed',
           'causality_not_established', 'optimization_success_not_certified',
           'P29_independent_validation_incomplete', 'producer_authenticity_external')
_MISSING = ('paired_quality_acceptance', 'independent_acceptance_evidence', 'uncertainty_estimate')
_P26_LIMITS = _LIMITS + ('exact_observed_ordered_stream_only', 'dataset_origin_not_verified',
    'split_and_source_id_are_caller_declared', 'preprocessing_implementation_not_verified',
    'checkpoint_artifact_not_required_or_inferred_from_model_state', 'model_behavior_may_differ')
_P26_MISSING = _MISSING + ('verified_dataset_origin', 'verified_split_membership',
                         'preprocessing_implementation_identity')


@register
@dataclass(frozen=True)
class PerformanceComparison(Contract):
    contract: ComparisonContract | P26QualityComparisonContract
    eligibility: ComparisonResult
    difference: MetricDifference | None
    warnings: tuple
    limitations: tuple = _LIMITS
    missing_evidence: tuple = _MISSING

    def __post_init__(self):
        if type(self.contract) not in (ComparisonContract, P26QualityComparisonContract) or type(self.eligibility) is not ComparisonResult:
            raise AdvisorError('typed_comparison_required')
        if self.eligibility.contract_sha256 != self.contract.content_sha256:
            raise AdvisorError('comparison_contract_mismatch')
        if self.difference is not None:
            if type(self.difference) is not MetricDifference or self.eligibility.status != ComparisonStatus.COMPARABLE:
                raise AdvisorError('difference_requires_comparable')
            d = self.difference
            if (d.baseline.metric != self.contract.metric or d.baseline.reference != self.eligibility.left or
                    d.candidate.reference != self.eligibility.right):
                raise AdvisorError('difference_evidence_mismatch')
        _strings(self, 'warnings'); _strings(self, 'limitations'); _strings(self, 'missing_evidence')
        p26 = type(self.contract) is P26QualityComparisonContract
        if self.limitations != (_P26_LIMITS if p26 else _LIMITS) or self.missing_evidence != (_P26_MISSING if p26 else _MISSING):
            raise AdvisorError('performance_boundaries_required')


def compare_performance(left_resolver, left_record_id, right_resolver, right_record_id, *, contract=None):
    """Recheck eligibility and resolve both scalars; never trust caller results.

    Returns no difference for ineligible pairs or unsafe arithmetic. A favorable
    direction is an observation for this metric/scope, not verified improvement.
    Multiple calls retain separate effects; no model-wide aggregate is inferred.
    P26QualityComparisonContract restricts eligibility to the exact observed
    ordered P26 stream and retains its additional origin/preprocessing limits.
    """
    contract = ComparisonContract() if contract is None else contract
    eligible = compare_records(left_resolver, left_record_id, right_resolver, right_record_id, contract=contract)
    warnings = tuple('allowed_change:' + r.field for r in eligible.reasons if r.code == 'allowed_change')
    difference = None
    if eligible.status == ComparisonStatus.COMPARABLE:
        a = left_resolver.resolve(left_record_id, contract.metric).resolved
        b = right_resolver.resolve(right_record_id, contract.metric).resolved
        try:
            difference = _difference(a, b)
        except (AdvisorError, OverflowError):
            warnings += ('numeric_comparison_unavailable',)
        if difference is not None and a.value == 0:
            warnings += ('zero_baseline_percent_undefined',)
    else:
        warnings += ('comparison_not_eligible',)
    if type(contract) is P26QualityComparisonContract:
        return PerformanceComparison(contract, eligible, difference, warnings, _P26_LIMITS, _P26_MISSING)
    return PerformanceComparison(contract, eligible, difference, warnings)
