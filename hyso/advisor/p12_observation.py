"""Explicit, versioned P12 observations; no legacy resolver or comparison policy.

Use ``observe_p12(record, version=1)`` on a retained EvidenceRecord. This module
does not import a recipe factory, execute a model or repair producer metadata.
"""
from dataclasses import dataclass
import json
from typing import Any

from ..analysis.evidence import EvidenceRecord
from ..analysis._codec import canonical, digest, export, plain, read_json
from ._json import AdvisorError

_FORMAT = 'hyso.advisor.p12_observation'
_MAX_BYTES = 64 * 1024**2
_LIMITS = (
    'analytic_observed_operations;not_hardware_counters_or_runtime_performance',
    'coverage_is_observed_ATen_call_fraction;not_FLOPs_or_hardware_coverage',
    'executed_recipe_input_paths_only;not_general_model_acceptance',
    'native_means_recorded_producer_field;not_authenticity_attestation',
    'source_UUID_SHA_and_trace_references_are_retained;external_files_not_followed',
    'caller_labels_QA_config_state_hashes_and_checkpoint_identity_are_not_inferred',
    'no_comparison_recommendation_global_winner_speedup_or_optimization_success',
)


def _version(version):
    if type(version) is not int or version != 1:
        raise AdvisorError('unsupported_p12_observation_version')


def _field(value):
    return dict(value=value, authority='missing' if value is None else 'native')


def _derive(record, version):
    _version(version)
    if type(record) is not EvidenceRecord:
        raise AdvisorError('p12_evidence_record_required')
    # Revalidate the original record/reader/bindings; never trust a projection.
    retained = EvidenceRecord(record.to_dict()).to_dict()
    native = retained['native_payload']
    if (retained['feature_id'] != 'P12' or native is None
            or native.get('format') != 'hyso.performance.operation_count'):
        raise AdvisorError('native_p12_operation_count_required')
    measurements = native['measurements']
    coverage = measurements['coverage']
    complete = measurements['status'] == 'complete' and coverage['unknown_calls'] == 0
    reason = measurements['reason']
    metrics = {}
    for name, unit in (('logical_flops_per_batch', 'logical_FLOP/batch'),
                       ('macs_per_batch', 'MAC/batch')):
        key = 'total_flops_per_batch' if name == 'logical_flops_per_batch' else 'total_macs_per_batch'
        value = measurements[key] if complete else None
        metrics[name] = dict(value=value, unit=unit,
                             status='DERIVED' if complete else 'UNKNOWN',
                             reason=None if complete else reason or 'incomplete_operator_coverage')
    provenance = native['provenance']
    model, inputs = provenance['model'], provenance['input']
    raw, protocol = native['raw_evidence'], native['protocol']
    shape = inputs.get('shape')
    identities = dict(
        producer_result_id=_field(native.get('result_id')),
        source_result_id=_field(native['identity'].get('source_result_id')),
        source_payload_sha256=_field(raw.get('source_sha256')),
        trace_sha256=_field((raw.get('trace') or {}).get('sha256')),
        protocol_id=_field(protocol.get('id')),
        native_protocol_sha256=dict(value=retained['scope']['protocol_sha256'],
                                    authority=retained['bindings']['protocol_sha256']),
    )
    body = dict(
        format=_FORMAT, schema_version=1, adapter_version=version,
        feature_id='P12', measurement_kind='analytic_logical_operation_count',
        availability='AVAILABLE' if complete else 'PARTIAL',
        reason=None if complete else reason or 'incomplete_operator_coverage',
        metrics=metrics, operator_coverage=coverage,
        counting_convention=protocol['convention'],
        operation_model_version=protocol.get('operation_model_version', 1),
        input=dict(shape=_field(shape), batch=_field(shape[0] if shape else None),
                   dtype=_field(inputs.get('dtype')),
                   device=dict(value=retained['scope']['device'], authority=retained['bindings']['device']),
                   precision=dict(value=retained['scope']['precision'], authority=retained['bindings']['precision']),
                   content_sha256=_field(inputs.get('content_sha256'))),
        model_state=_field(model.get('weights_sha256', model.get('state_sha256'))),
        model_descriptor=_field(model.get('structural_descriptor')),
        producer_identity=identities,
        producer_identity_status='AVAILABLE' if all(
            v['authority'] == 'native' for v in identities.values()) else 'PARTIAL',
        native_protocol=protocol,
        native_protocol_sha_definition='Analysis.native_scope: digest(protocol,configuration,training,optimizer);not_external_QA_protocol_SHA',
        scope=retained['scope'], scope_bindings=retained['bindings'],
        declared_scope=retained['declared_scope'],
        evidence_reference=dict(record_id=retained['record_id'], artifact=retained['artifact'],
                                payload_sha256=retained['source']['payload_sha256']),
        evidence_record=retained,
        comparison_eligibility='NOT_SUPPORTED_IN_COMPARISON',
        claims=dict(hardware_performance=False, model_comparison=False,
                    model_performance_recommendation=False, global_winner=None,
                    speedup=None, statistical_superiority='not_established',
                    optimization_success='not_assessed'),
        limitations=list(_LIMITS),
    )
    if raw.get('p12_binding') is not None:
        # Observation v1 still makes no pairwise eligibility claim. The native
        # receipt is exposed only when supplied and validated by the producer.
        body['native_cpu_binding']=_field(raw['p12_binding'])
    body['content_sha256'] = digest(body)
    return body


@dataclass(frozen=True, init=False)
class P12Observation:
    """Detached schema1/adapter1 observation, rederived on JSON restoration.

    Identity availability and operation coverage are separate. Source locators
    are retained as data and never opened. Raw native payload values are exact;
    observation JSON is not a recreation of original artifact whitespace.
    """
    _payload: str

    def __init__(self, data: dict[str, Any]) -> None:
        try:
            data = plain(data)
            if (type(data) is not dict or data.get('format') != _FORMAT
                    or type(data.get('schema_version')) is not int or data['schema_version'] != 1):
                raise AdvisorError('unsupported_p12_observation_schema')
            _version(data.get('adapter_version'))
            if len(canonical(data).encode('utf8')) > _MAX_BYTES:
                raise AdvisorError('p12_observation_size_limit')
            expected = _derive(EvidenceRecord(data['evidence_record']), data['adapter_version'])
            if canonical(data) != canonical(expected):
                raise AdvisorError('p12_observation_derivation_or_integrity_mismatch')
        except AdvisorError:
            raise
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, RecursionError, OverflowError) as error:
            raise AdvisorError('invalid_p12_observation_evidence') from error
        object.__setattr__(self, '_payload', canonical(expected))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._payload)

    def to_json(self, path=None, *, overwrite: bool = False) -> str:
        raw = self._payload + '\n'
        if path is not None:
            export(path, raw, overwrite)
        return raw

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'P12Observation':
        return cls(data)

    @classmethod
    def from_json(cls, path) -> 'P12Observation':
        data, _, _ = read_json(path, max_bytes=_MAX_BYTES)
        return cls(data)


def observe_p12(record: EvidenceRecord, *, version: int) -> P12Observation:
    """Opt in to a P12 observation; ``version=1`` must be supplied explicitly.

    Legacy NativeMetricResolver/advise behavior is unchanged. Pairwise P12
    eligibility belongs to compare_models, not this standalone observation.
    Partial operator coverage yields null model totals, never zero-filled costs.
    This is not a ResolvedMetric or a general comparison-contract input.
    """
    return P12Observation(_derive(record, version))
