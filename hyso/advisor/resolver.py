"""Closed native metric resolution and per-claim evidence checks (resolver v1)."""
import typing as _typing
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math

from ._json import AdvisorError, Contract, canonical, register
from .contracts import (EvidenceBinding, EvidenceRef, EvidenceSupport, Finding, FindingKind,
                        _choice, _enum, _number, _sequence, _strings, _text)

RESOLVER_VERSION = '1'


class NativeMetric(str, Enum):
    """Select a native metric without inferring unavailable measurements or converting incompatible scopes.

    Returns:
        An instance of NativeMetric with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        NativeMetric.FORWARD_P50

    HYSo Guide: metrics.interpretation
    Offline: hyso docs show metrics.interpretation --lang en
    Open: hyso docs open metrics.interpretation --lang en

    Members:
        FORWARD_P50, FORWARD_P95, FORWARD_P99, COMPLETED_IMAGES_PER_SECOND, E2E_P50, E2E_P95, E2E_P99,
        PRECISION_MEAN, PRECISION_THROUGHPUT, RSS_BASELINE, RSS_CURRENT, RSS_SAMPLED_PEAK,
        RSS_LIFETIME_HWM, PARAMETER_BYTES, BUFFER_BYTES, MODEL_LOGICAL_BYTES, ARTIFACT_BYTES,
        TEST_ACCURACY, TEST_MACRO_F1.
    """
    FORWARD_P50 = 'resident_forward_p50'
    FORWARD_P95 = 'resident_forward_p95'
    FORWARD_P99 = 'resident_forward_p99'
    COMPLETED_IMAGES_PER_SECOND = 'completed_images_per_second'
    E2E_P50 = 'e2e_p50'
    E2E_P95 = 'e2e_p95'
    E2E_P99 = 'e2e_p99'
    PRECISION_MEAN = 'precision_forward_mean'
    PRECISION_THROUGHPUT = 'precision_completed_images_per_second'
    RSS_BASELINE = 'process_rss_baseline'
    RSS_CURRENT = 'process_rss_current'
    RSS_SAMPLED_PEAK = 'process_rss_sampled_peak'
    RSS_LIFETIME_HWM = 'process_rss_lifetime_hwm'
    PARAMETER_BYTES = 'model_parameter_logical_bytes'
    BUFFER_BYTES = 'model_buffer_logical_bytes'
    MODEL_LOGICAL_BYTES = 'model_parameter_buffer_logical_bytes'
    ARTIFACT_BYTES = 'serialized_artifact_bytes'
    TEST_ACCURACY = 'test_accuracy'
    TEST_MACRO_F1 = 'test_macro_f1'


@register
@dataclass(frozen=True)
class ResolvedMetric(Contract):
    """A detached metric description; construction alone is not certification.

    Args:
        metric (NativeMetric): Exact supported metric selector; units and missing-value status are
            preserved. Required.
        value (float): Observed metric value; unavailable values retain their native null/status policy.
            Required.
        reference (EvidenceRef): Exact retained reference evidence/identity. Required.
        record_scope_json (str): Canonical metric scope serialized as JSON. Required.
        protocol_json (str): Canonical protocol metadata serialized as JSON. Required.
        native_context_json (str): Canonical native context serialized as JSON. Required.
        origin (str): Explicit evidence origin/provenance. Required.
        limitations (tuple): Explicit limitations of the evidence, comparison or proposed operation.
            Default: ().
        resolver_version (str): Exact resolver/rule version. Default: RESOLVER_VERSION.

    Returns:
        An instance of ResolvedMetric with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        ResolvedMetric(metric=metric, value=value, reference=reference,
        record_scope_json=record_scope_json, protocol_json=protocol_json,
        native_context_json=native_context_json, origin=origin)

    HYSo Guide: metrics.interpretation
    Offline: hyso docs show metrics.interpretation --lang en
    Open: hyso docs open metrics.interpretation --lang en
    """
    metric: NativeMetric
    value: float
    reference: EvidenceRef
    record_scope_json: str
    protocol_json: str
    native_context_json: str
    origin: _typing.Literal['new', 'reused']
    limitations: tuple = ()
    resolver_version: str = RESOLVER_VERSION

    def __post_init__(self):
        _enum(self, 'metric', NativeMetric)
        _number(self.value, 'metric_value')
        if type(self.reference) is not EvidenceRef:
            raise AdvisorError('typed_reference_required')
        for name in ('record_scope_json', 'protocol_json', 'native_context_json'):
            _text(getattr(self, name), name)
            try:
                data = json.loads(getattr(self, name))
            except (ValueError, TypeError) as exc:
                raise AdvisorError('invalid_metric_context') from exc
            if type(data) is not dict or canonical(data) != getattr(self, name):
                raise AdvisorError('noncanonical_metric_context')
        _choice(self, 'origin', ('new', 'reused'))
        _strings(self, 'limitations')
        if self.resolver_version != RESOLVER_VERSION:
            raise AdvisorError('unsupported_resolver_version')

    @property
    def record_scope(self) -> '_typing.Any':
        """Read the retained record scope from this ResolvedMetric record.

        Returns:
            The retained record scope value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.record_scope

        HYSo Guide: metrics.interpretation
        Offline: hyso docs show metrics.interpretation --lang en
        Open: hyso docs open metrics.interpretation --lang en
        """
        return json.loads(self.record_scope_json)

    @property
    def protocol(self) -> '_typing.Any':
        """Read the retained protocol from this ResolvedMetric record.

        Returns:
            The retained protocol value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.protocol

        HYSo Guide: metrics.interpretation
        Offline: hyso docs show metrics.interpretation --lang en
        Open: hyso docs open metrics.interpretation --lang en
        """
        return json.loads(self.protocol_json)

    @property
    def native_context(self) -> '_typing.Any':
        """Read the retained native context from this ResolvedMetric record.

        Returns:
            The retained native context value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.native_context

        HYSo Guide: metrics.interpretation
        Offline: hyso docs show metrics.interpretation --lang en
        Open: hyso docs open metrics.interpretation --lang en
        """
        return json.loads(self.native_context_json)


@register
@dataclass(frozen=True)
class MetricResolution(Contract):
    """Retain a metric resolution with its supported value, unit, source and unavailability reasons.

    Args:
        status (str): Exact lifecycle/status filter accepted by this API. Required.
        resolved (ResolvedMetric | None): Retained resolved value/status under the requested scope.
            Default: None.
        reasons (tuple): Structured eligibility/compatibility reasons. Default: ().

    Returns:
        An instance of MetricResolution with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        MetricResolution(status=status)

    HYSo Guide: metrics.interpretation
    Offline: hyso docs show metrics.interpretation --lang en
    Open: hyso docs open metrics.interpretation --lang en
    """
    status: _typing.Literal['supported', 'unsupported_metric', 'insufficient_evidence']
    resolved: ResolvedMetric | None = None
    reasons: tuple = ()

    def __post_init__(self):
        _choice(self, 'status', ('supported', 'unsupported_metric', 'insufficient_evidence'))
        _strings(self, 'reasons')
        if (self.status == 'supported') != (type(self.resolved) is ResolvedMetric):
            raise AdvisorError('resolution_status_mismatch')
        if self.resolved is not None and type(self.resolved) is not ResolvedMetric:
            raise AdvisorError('typed_resolved_metric_required')


@register
@dataclass(frozen=True)
class NativeObservation(Finding):
    """Add structured scalar/context to Finding without changing its v1 fields.

    Args:
        resolved_metric (ResolvedMetric | None): Native resolved metric retaining unit and scope.
            Default: None.

    Returns:
        An instance of NativeObservation with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        NativeObservation(finding_id, kind, statement)

    HYSo Guide: metrics.interpretation
    Offline: hyso docs show metrics.interpretation --lang en
    Open: hyso docs open metrics.interpretation --lang en
    """
    resolved_metric: ResolvedMetric | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.kind != FindingKind.OBSERVATION or type(self.resolved_metric) is not ResolvedMetric:
            raise AdvisorError('typed_native_observation_required')
        if (self.evidence != (self.resolved_metric.reference,) or
                self.evidence_support != EvidenceSupport.SUPPORTED or self.verification != 'not_verified'):
            raise AdvisorError('native_observation_reference_mismatch')


@register
@dataclass(frozen=True)
class EvidenceEligibility(Contract):
    """Describe why native evidence is eligible, incompatible or insufficient for a requested metric.

    Args:
        status (EvidenceSupport): Exact lifecycle/status filter accepted by this API. Required.
        claim (str): Explicit evidence-backed statement to assess. Required.
        reasons (tuple): Structured eligibility/compatibility reasons. Default: ().

    Returns:
        An instance of EvidenceEligibility with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        EvidenceEligibility(status=status, claim=claim)

    HYSo Guide: advisor.recommendations
    Offline: hyso docs show advisor.recommendations --lang en
    Open: hyso docs open advisor.recommendations --lang en
    """
    status: EvidenceSupport
    claim: _typing.Literal['observation', 'comparison']
    reasons: tuple = ()

    def __post_init__(self):
        _enum(self, 'status', EvidenceSupport)
        _choice(self, 'claim', ('observation', 'comparison'))
        _strings(self, 'reasons')


_FORWARD = {NativeMetric.FORWARD_P50: .5, NativeMetric.FORWARD_P95: .95, NativeMetric.FORWARD_P99: .99}
_E2E = {NativeMetric.E2E_P50: 'p50_ms', NativeMetric.E2E_P95: 'p95_ms', NativeMetric.E2E_P99: 'p99_ms'}
_RSS = {NativeMetric.RSS_BASELINE: 'baseline_rss_bytes', NativeMetric.RSS_CURRENT: 'current_rss_bytes',
        NativeMetric.RSS_SAMPLED_PEAK: 'sampled_peak_rss_bytes', NativeMetric.RSS_LIFETIME_HWM: 'current_lifetime_hwm_bytes'}
_FOOTPRINT = {NativeMetric.PARAMETER_BYTES: 'parameter_bytes', NativeMetric.BUFFER_BYTES: 'buffer_bytes',
              NativeMetric.MODEL_LOGICAL_BYTES: 'parameter_buffer_bytes', NativeMetric.ARTIFACT_BYTES: 'serialized_artifact_bytes'}


def _at(data, path):
    for key in path.split('.'):
        data = data[int(key)] if type(data) is list else data[key]
    return data


def _unsupported(reason='native_format_or_metric_not_supported'):
    return MetricResolution('unsupported_metric', reasons=(reason,))


def _insufficient(reason):
    return MetricResolution('insufficient_evidence', reasons=(reason,))


@dataclass(frozen=True, init=False)
class NativeMetricResolver:
    """Revalidate a complete report once; keep only immutable JSON snapshots.

    No source paths, factory strings, model classes or device APIs are followed.
    Native constructors may do bounded CPU arithmetic on recorded evidence.

    Args:
        report (_typing.Any): Validated retained analysis/advisory report. Required.

    Returns:
        An instance of NativeMetricResolver with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        NativeMetricResolver(report)

    HYSo Guide: metrics.interpretation
    Offline: hyso docs show metrics.interpretation --lang en
    Open: hyso docs open metrics.interpretation --lang en
    """
    _payload: str
    _records: tuple
    source_report_id: str
    source_report_sha256: str

    def __init__(self, report: '_typing.Any') -> 'None':
        """Initialize NativeMetricResolver with the supplied settings.

        Args:
            report (_typing.Any): Validated retained analysis/advisory report. Required.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            NativeMetricResolver(report)

        HYSo Guide: metrics.interpretation
        Offline: hyso docs show metrics.interpretation --lang en
        Open: hyso docs open metrics.interpretation --lang en
        """
        from hyso.analysis import UnifiedAnalysisReport
        if type(report) is not UnifiedAnalysisReport:
            raise AdvisorError('typed_analysis_report_required')
        try:
            data = UnifiedAnalysisReport(report.to_dict()).to_dict()
        except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
            raise AdvisorError('invalid_analysis_report') from exc
        raw = canonical(data)
        object.__setattr__(self, '_payload', raw)
        object.__setattr__(self, '_records', tuple((r['record_id'], canonical(r)) for r in data['index']['records']))
        object.__setattr__(self, 'source_report_id', data['report_id'])
        object.__setattr__(self, 'source_report_sha256', hashlib.sha256(raw.encode('utf-8')).hexdigest())

    @property
    def report(self) -> '_typing.Any':
        """Read the retained report from this NativeMetricResolver record.

        Returns:
            The retained report value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.report

        HYSo Guide: metrics.interpretation
        Offline: hyso docs show metrics.interpretation --lang en
        Open: hyso docs open metrics.interpretation --lang en
        """
        return json.loads(self._payload)

    def record(self, record_id: '_typing.Any') -> '_typing.Any':
        """Read the retained record from this NativeMetricResolver record.

        Args:
            record_id (_typing.Any): Exact retained evidence identity. Required.

        Returns:
            The retained record value, without recomputing measurements.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.record(record_id)

        HYSo Guide: metrics.interpretation
        Offline: hyso docs show metrics.interpretation --lang en
        Open: hyso docs open metrics.interpretation --lang en
        """
        for identity, raw in self._records:
            if identity == record_id:
                return json.loads(raw)
        return None

    def resolve(self, record_id: '_typing.Any', metric: NativeMetric, *, point_id: '_typing.Any'=None) -> '_typing.Any':
        """Resolve the requested metric from retained native evidence without guessing a missing value.

        Args:
            record_id (_typing.Any): Exact retained evidence identity. Required.
            metric (_typing.Any): Exact supported metric selector; units and missing-value status are
                preserved. Required.
            point_id (_typing.Any): Exact sweep/calibration point identity. Default: None.

        Returns:
            A MetricResolution with the native value/unit/scope or explicit unavailability reasons.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.resolve(record_id, metric)

        HYSo Guide: metrics.interpretation
        Offline: hyso docs show metrics.interpretation --lang en
        Open: hyso docs open metrics.interpretation --lang en
        """
        try:
            metric = NativeMetric(metric)
        except (ValueError, TypeError):
            return _unsupported('unknown_metric_identity')
        record = self.record(record_id)
        if record is None:
            return _insufficient('unknown_record_id')
        if record['validation']['status'] != 'native_validated' or record['native_payload'] is None:
            return _insufficient('unavailable_record:' + record['validation']['status'])
        try:
            return self._resolve(record, metric, point_id)
        except (KeyError, IndexError, TypeError) as exc:
            return _insufficient('missing_native_metric_semantics:' + type(exc).__name__)

    def _resolve(self, r, metric, point_id):
        n = r['native_payload']; p = n.get('protocol', {})
        feature, fmt = r['feature_id'], n.get('format')
        context = dict(measurement_scope=p.get('scope'), device=r['scope']['device'],
                       input_kind=r['scope']['input_kind'], sample_count=n.get('actual', {}).get('samples'))
        path = None; domain = unit = None; value = status = None
        point_ref = None; protocol_ref = p.get('id')
        notes = list(r['limitations']) + [
            'Artifact SHA is bound to this validated report; original artifact bytes are not reread unless explicitly supplied.',
            'This observation does not establish cross-record compatibility or independent improvement.',
        ]
        if point_id is not None and metric not in (NativeMetric.PRECISION_MEAN, NativeMetric.PRECISION_THROUGHPUT):
            return _insufficient('point_reference_not_applicable')
        if metric in _FORWARD and feature == 'P01' and fmt == 'hyso.performance.latency_analysis':
            if (p.get('id') != 'resident_tail' or p.get('scope') not in ('synchronous_host_forward', 'serial_current_stream_forward')
                    or p.get('method') != 'linear_(n-1)*q'):
                return _insufficient('resident_forward_protocol_not_supported')
            q = _FORWARD[metric]
            matches = [(i, row) for i, row in enumerate(n['measurements']) if row.get('quantile') == q]
            if q not in p['quantiles'] or len(matches) != 1:
                return _insufficient('quantile_identity_missing_or_ambiguous')
            i, row = matches[0]; path = 'measurements.%d.value' % i
            value, unit, status = row['value'], row['unit'], row['status']; domain = 'latency'
            context.update(quantile=q, effective_tail_count=row.get('effective_tail_count'),
                           repeats=n['actual']['repeats'], precision=n['provenance']['configuration']['precision'],
                           batch=r['scope']['batch'])
            notes.append('Sample quantile is descriptive; tail eligibility is not an independence or confidence-interval certificate.')
        elif metric == NativeMetric.COMPLETED_IMAGES_PER_SECOND and feature == 'P02' and fmt == 'hyso.performance.throughput':
            if (p.get('id') != 'continuous_completion_window' or p.get('scope') != 'resident_forward_completion_wall'
                    or n['measurements'].get('aggregation') != 'total_completed_images_over_total_wall_seconds'):
                return _insufficient('completion_protocol_mismatch')
            path, domain, unit = 'measurements.value', 'throughput', 'images/s'
            row = n['measurements']; value, status = row['value'], row['status']
            if row['unit'] != unit:
                return _insufficient('native_unit_mismatch')
            context.update(batch=n['provenance']['input']['shape'][0], precision=p['configuration']['precision'],
                           aggregation=row['aggregation'], elapsed_wall_seconds=row['elapsed_wall_seconds'])
            notes.append('Completed resident images/s is not camera FPS, request latency or E2E throughput.')
        elif metric in _E2E and feature == 'P03' and fmt == 'hyso.performance.pipeline':
            if p.get('id') != 'serial_bounded_pipeline' or p.get('scope') != 'read_start_to_post_completion':
                return _insufficient('e2e_protocol_mismatch')
            key = _E2E[metric]; path = 'measurements.' + key + '.value'
            row = n['measurements'][key]; value, unit, status = row['value'], row['unit'], row['status']; domain = 'e2e'
            context.update(sample_count=n['actual']['completed'], unit_of_work=n['provenance']['input']['unit_of_work'])
            notes.append('One pipeline ID may contain a batch; this is read-start to post-completion, not resident forward latency.')
        elif metric in (NativeMetric.PRECISION_MEAN, NativeMetric.PRECISION_THROUGHPUT) and feature == 'P08' and fmt == 'hyso.performance.sweep':
            if (p.get('id') != 'finite_isolated_points' or p.get('axis') != 'precision'
                    or p.get('scope') != 'ordered_resident_point_sweep'):
                return _insufficient('precision_sweep_protocol_mismatch')
            if point_id is None:
                return _insufficient('exact_point_id_required')
            matches = [(i, v) for i, v in enumerate(n['raw_evidence']['points']) if v['point_id'] == point_id]
            if len(matches) != 1:
                return _insufficient('unknown_point_id')
            i, point = matches[0]
            if point['status'] != 'measured' or point['result'] is None:
                return _insufficient('point_' + point['status'])
            result = point['result']; p = result['protocol']
            if (p.get('id') != 'isolated_precision_point' or p.get('scope') != 'resident_forward'
                    or p.get('precision') != point['precision'] or result['provenance']['input']['shape'] != point['shape']):
                return _insufficient('point_scope_mismatch')
            point_ref = point_id; protocol_ref = p['id']
            name = 'latency_mean_ms' if metric == NativeMetric.PRECISION_MEAN else 'sustained_images_per_second'
            row = result['measurements'][name]
            path = 'raw_evidence.points.%d.result.measurements.%s.value' % (i, name)
            domain, unit = ('latency', 'ms/batch') if metric == NativeMetric.PRECISION_MEAN else ('throughput', 'images/s')
            value, status = row['value'], row['status']
            if row['unit'] != unit:
                return _insufficient('native_unit_mismatch')
            context.update(measurement_scope=p['scope'], precision=point['precision'], batch=point['shape'][0],
                           input_shape=point['shape'], point_id=point_id, sample_count=result['actual']['samples'],
                           cross_point_compatibility=n['protocol'].get('cross_point_compatibility'))
            notes.append('Point context is native to this point; original record scope/bindings are unchanged. No precision recommendation or cross-point speedup is inferred.')
            projected = next((v for v in r['metrics'] if v['path'] == path), None)
            if projected and projected['domain'] != domain:
                notes.append('Projection domain disagrees with native metric semantics; resolver v1 uses the validated point unit/protocol, not that projection label.')
        elif metric in _RSS and feature == 'P10' and fmt == 'hyso.performance.memory':
            if p.get('id') != 'process_rss' or p.get('scope') != 'current_process_rss':
                return _insufficient('rss_protocol_mismatch')
            path = 'measurements.' + _RSS[metric] + '.value'
            row = n['measurements'][_RSS[metric]]; value, unit, status = row['value'], row['unit'], row['status']; domain = 'process_rss'
            notes.append('Process RSS includes runtime/allocator costs. Sampled peaks may miss transients; lifetime HWM is not a workload-reset peak or CUDA allocation.')
        elif metric in _FOOTPRINT and feature == 'P11' and fmt == 'hyso.performance.footprint':
            if (p.get('id') != 'logical_and_artifact_bytes' or p.get('scope') != 'dense_cpu_model_and_optional_artifact'
                    or p.get('byte_accounting') != 'unique_tensor_object_logical'):
                return _insufficient('logical_bytes_protocol_mismatch')
            path = 'measurements.' + _FOOTPRINT[metric] + '.value'
            row = n['measurements'][_FOOTPRINT[metric]]; value, unit, status = row['value'], row['unit'], row['status']
            domain = 'serialized_artifact_bytes' if metric == NativeMetric.ARTIFACT_BYTES else 'model_logical_bytes'
            notes.append('Logical tensor bytes, serialized file bytes, process RSS and physical/device memory are distinct quantities.')
        elif metric in (NativeMetric.TEST_ACCURACY, NativeMetric.TEST_MACRO_F1) and feature == 'P26' and fmt == 'hyso.evaluation.classification':
            config = n['configuration']
            if config.get('split') != 'test' or config.get('task') not in ('multiclass', 'binary'):
                return _insufficient('test_split_or_classification_identity_missing')
            path = 'metrics.accuracy' if metric == NativeMetric.TEST_ACCURACY else 'metrics.averages.macro.f1'
            value = _at(n, path); unit, domain, status = 'fraction', 'quality', 'derived'
            if not 0 <= value <= 1:
                return _insufficient('quality_fraction_out_of_range')
            p = config; protocol_ref = 'hyso.evaluation.classification:schema' + str(n['schema_version'])
            context.update(split='test', task=config['task'], sample_count=n['metrics']['sample_count'],
                           averaging='macro' if metric == NativeMetric.TEST_MACRO_F1 else 'sample_accuracy')
            notes.append('Quality is descriptive of this recorded test cohort; it is not automatically paired with another performance capture.')
            if n['schema_version'] in (2, 3):
                notes.append('The test split is caller-declared. Ordered tensor content does not verify dataset origin or preprocessing code.')
            if n['schema_version'] == 3:
                from hyso._evaluation_structure import resolved_structure_context
                context['structure_binding'] = resolved_structure_context(n['structure_binding'])
                notes.append('Structure is bound to the isolated forward receiver at pre/post evaluation boundaries; original-model architecture/behavior equivalence is not certified.')
                notes.append('Imported binding validation proves consistency, not cryptographic live-model origin or the executed operator graph.')
                notes.append('Individual snapshot fields retain their unsupported/missing statuses; a native association does not upgrade their coverage.')
        else:
            return _unsupported()
        expected_unit = ('ms' if metric in _FORWARD or metric in _E2E else 'byte' if metric in _RSS or metric in _FOOTPRINT else unit)
        if unit != expected_unit:
            return _insufficient('native_unit_mismatch')
        if status not in ('measured', 'derived') or value is None:
            return _insufficient('native_status_not_observable:' + str(status))
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            return _insufficient('invalid_metric_value')
        ref = EvidenceRef(self.source_report_id, r['record_id'], path, r['artifact']['sha256'],
                          r['source']['payload_sha256'], domain, unit, status,
                          tuple(EvidenceBinding(k, v) for k, v in sorted(r['bindings'].items())),
                          point_ref=point_ref, protocol_ref=protocol_ref)
        return MetricResolution('supported', ResolvedMetric(metric, value, ref, canonical(r['scope']),
                                canonical(p), canonical(context), r['origin'], tuple(notes)))

    def check_reference(self, reference: '_typing.Any', metric: NativeMetric, *, required_native: '_typing.Sequence[str]'=(), artifact_bytes: '_typing.Any'=None) -> '_typing.Any':
        """Check the entire reference against a regenerated native resolution.

        Original artifact bytes are optional explicit input, never read by path.
        Without them, only the artifact hash's binding to the report is checked.

        Args:
            reference (_typing.Any): Exact retained reference evidence/identity. Required.
            metric (_typing.Any): Exact supported metric selector; units and missing-value status are
                preserved. Required.
            required_native (_typing.Sequence[str]): Exact native evidence prerequisites. Default: ().
            artifact_bytes (_typing.Any): Artifact in bytes; this does not imply a GPU hardware quota.
                Default: None.

        Returns:
            The retained check reference value, without recomputing measurements.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.check_reference(reference, metric)

        HYSo Guide: metrics.interpretation
        Offline: hyso docs show metrics.interpretation --lang en
        Open: hyso docs open metrics.interpretation --lang en
        """
        if type(reference) is not EvidenceRef:
            raise AdvisorError('typed_reference_required')
        resolution = self.resolve(reference.record_id, metric, point_id=reference.point_ref)
        reasons = []
        if resolution.status != 'supported':
            reasons.extend(resolution.reasons)
        elif reference != resolution.resolved.reference:
            reasons.append('reference_does_not_match_native_metric')
        if type(required_native) not in (tuple, list) or any(type(k) is not str for k in required_native):
            raise AdvisorError('invalid_required_native_fields')
        bindings = {b.field: b.authority for b in reference.bindings}
        reasons.extend('association_not_native:' + k for k in required_native if bindings.get(k) != 'native')
        if artifact_bytes is not None:
            if type(artifact_bytes) is not bytes or len(artifact_bytes) > 64 * 1024**2:
                raise AdvisorError('invalid_artifact_bytes')
            if hashlib.sha256(artifact_bytes).hexdigest() != reference.artifact_sha256:
                reasons.append('artifact_sha256_mismatch')
            else:
                from hyso.analysis._codec import decode, digest
                try:
                    if digest(decode(artifact_bytes)) != reference.payload_sha256:
                        reasons.append('artifact_payload_mismatch')
                except ValueError:
                    reasons.append('invalid_artifact_json')
        return EvidenceEligibility(EvidenceSupport.INSUFFICIENT_EVIDENCE if reasons else EvidenceSupport.SUPPORTED,
                                   'observation', tuple(reasons))

    def check_metric(self, resolved: '_typing.Any') -> '_typing.Any':
        """Also reject tampered scalar, point context, origin or protocol claims.

        Args:
            resolved (_typing.Any): Retained resolved value/status under the requested scope. Required.

        Returns:
            The retained check metric value, without recomputing measurements.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.check_metric(resolved)

        HYSo Guide: metrics.interpretation
        Offline: hyso docs show metrics.interpretation --lang en
        Open: hyso docs open metrics.interpretation --lang en
        """
        if type(resolved) is not ResolvedMetric:
            raise AdvisorError('typed_resolved_metric_required')
        fresh = self.resolve(resolved.reference.record_id, resolved.metric, point_id=resolved.reference.point_ref)
        if fresh.status != 'supported' or fresh.resolved != resolved:
            return EvidenceEligibility('insufficient_evidence', 'observation', ('resolved_metric_mismatch',))
        return self.check_reference(resolved.reference, resolved.metric)

    def compare(self, left: '_typing.Any', left_metric: '_typing.Any', right: '_typing.Any', right_metric: '_typing.Any') -> '_typing.Any':
        """Compatibility only, never independence, ranking or verified effect.

        Args:
            left (_typing.Any): Left retained input/source in this comparison. Required.
            left_metric (_typing.Any): Resolved left-side metric with its unit and scope. Required.
            right (_typing.Any): Right retained input/source in this comparison. Required.
            right_metric (_typing.Any): Resolved right-side metric with its unit and scope. Required.

        Returns:
            The retained compare value, without recomputing measurements.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.compare(left, left_metric, right, right_metric)

        HYSo Guide: metrics.interpretation
        Offline: hyso docs show metrics.interpretation --lang en
        Open: hyso docs open metrics.interpretation --lang en
        """
        a = self.check_reference(left, left_metric); b = self.check_reference(right, right_metric)
        reasons = list(a.reasons + b.reasons)
        if not reasons:
            lm = self.resolve(left.record_id, left_metric, point_id=left.point_ref).resolved
            rm = self.resolve(right.record_id, right_metric, point_id=right.point_ref).resolved
            if lm.metric != rm.metric or left.domain != right.domain or left.unit != right.unit:
                reasons.append('metric_semantics_incompatible')
            if lm.native_context != rm.native_context:
                reasons.append('native_metric_context_differs')
            from hyso.analysis import EvidenceRecord, compare_evidence
            comparison = compare_evidence(EvidenceRecord(self.record(left.record_id)),
                                          EvidenceRecord(self.record(right.record_id)), domain=left.domain)
            if comparison['status'] != 'compatible':
                reasons.append('native_association_' + comparison['status'])
        return EvidenceEligibility(EvidenceSupport.INSUFFICIENT_EVIDENCE if reasons else EvidenceSupport.SUPPORTED,
                                   'comparison', tuple(reasons))
