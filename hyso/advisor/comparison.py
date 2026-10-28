"""General v1 and explicit P26 observed-quality eligibility; no effects."""
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import re

from ._json import AdvisorError, Contract, canonical, register
from .contracts import EvidenceRef, _enum, _sequence, _sha, _strings, _text
from .resolver import NativeMetric, NativeMetricResolver
from hyso.analysis._codec import digest


class ComparisonStatus(str, Enum):
    COMPARABLE = 'COMPARABLE'
    INSUFFICIENT_EVIDENCE = 'INSUFFICIENT_EVIDENCE'
    INCOMPATIBLE = 'INCOMPATIBLE'


_QUANTILES = {NativeMetric.FORWARD_P50: .5, NativeMetric.FORWARD_P95: .95, NativeMetric.FORWARD_P99: .99}
_SCALARS = {NativeMetric.COMPLETED_IMAGES_PER_SECOND, NativeMetric.RSS_BASELINE,
    NativeMetric.RSS_CURRENT, NativeMetric.RSS_SAMPLED_PEAK, NativeMetric.PARAMETER_BYTES,
    NativeMetric.BUFFER_BYTES, NativeMetric.MODEL_LOGICAL_BYTES, NativeMetric.ARTIFACT_BYTES,
    NativeMetric.TEST_ACCURACY, NativeMetric.TEST_MACRO_F1}


@register
@dataclass(frozen=True)
class ComparisonContract(Contract):
    metric: NativeMetric = NativeMetric.FORWARD_P95
    allowed_changes: tuple = ()
    min_samples: int = 1000
    min_tail_samples: int = 10

    def __post_init__(self):
        _enum(self, 'metric', NativeMetric)
        if self.metric not in _QUANTILES and self.metric not in _SCALARS:
            raise AdvisorError('unsupported_comparison_metric')
        _strings(self, 'allowed_changes')
        if len(set(self.allowed_changes)) != len(self.allowed_changes) or any(
                k not in ('model', 'batch', 'precision') for k in self.allowed_changes):
            raise AdvisorError('unsupported_or_duplicate_allowed_change')
        for value in (self.min_samples, self.min_tail_samples):
            if type(value) is not int or value < 1:
                raise AdvisorError('invalid_sample_requirement')


@register
@dataclass(frozen=True)
class P26QualityComparisonContract(Contract):
    """Explicit observed-stream quality policy; never general performance v1."""
    metric: NativeMetric = NativeMetric.TEST_ACCURACY
    allowed_changes: tuple = ()
    min_samples: int = 1
    policy: str = 'p26_exact_observed_stream_v1'

    def __post_init__(self):
        _enum(self, 'metric', NativeMetric)
        if self.metric not in (NativeMetric.TEST_ACCURACY, NativeMetric.TEST_MACRO_F1):
            raise AdvisorError('p26_policy_requires_quality_metric')
        _strings(self, 'allowed_changes')
        if self.allowed_changes not in ((), ('model',)):
            raise AdvisorError('p26_policy_only_allows_model_change')
        if type(self.min_samples) is not int or self.min_samples < 1:
            raise AdvisorError('invalid_sample_requirement')
        if self.policy != 'p26_exact_observed_stream_v1':
            raise AdvisorError('unsupported_p26_comparison_policy')


@register
@dataclass(frozen=True)
class ComparisonReason(Contract):
    code: str
    field: str
    side: str = 'pair'

    def __post_init__(self):
        _text(self.code, 'reason_code'); _text(self.field, 'reason_field')
        if self.side not in ('left', 'right', 'pair'):
            raise AdvisorError('invalid_reason_side')


@register
@dataclass(frozen=True)
class ComparisonResult(Contract):
    status: ComparisonStatus
    contract_sha256: str
    reasons: tuple
    left: EvidenceRef | None = None
    right: EvidenceRef | None = None
    performance_effect: str = 'not_computed'
    statistical_independence: str = 'not_certified'

    def __post_init__(self):
        _enum(self, 'status', ComparisonStatus); _sha(self.contract_sha256, 'contract_sha256')
        _sequence(self, 'reasons', ComparisonReason)
        for ref in (self.left, self.right):
            if ref is not None and type(ref) is not EvidenceRef:
                raise AdvisorError('typed_reference_required')
        if self.status == ComparisonStatus.COMPARABLE and (self.left is None or self.right is None):
            raise AdvisorError('comparable_requires_two_references')
        if self.performance_effect != 'not_computed' or self.statistical_independence != 'not_certified':
            raise AdvisorError('effect_and_statistical_certification_out_of_scope')


_HASH_FIELDS = ('hardware_id', 'checkpoint_sha256', 'cohort_sha256', 'preprocessing_sha256', 'input_family_sha256')
_TEXT_FIELDS = ('dataset_id', 'split', 'workload_id', 'clock_id')
_CAPTURE_FIELDS = {'format', 'schema_version', 'authority', 'run_id', 'sample_sha256', *_HASH_FIELDS, *_TEXT_FIELDS}


def _capture(resolver, record_id, contract, side):
    reasons = []; bad = False
    record = resolver.record(record_id)
    if record is None:
        return None, None, [ComparisonReason('record_not_found', 'record_id', side)], False
    resolution = resolver.resolve(record_id, contract.metric)
    if resolution.status != 'supported':
        return None, None, [ComparisonReason('metric_not_observable', contract.metric.value, side)], False
    m = resolution.resolved
    if resolver.check_metric(m).status != 'supported':
        return None, None, [ComparisonReason('invalid_native_evidence', 'metric', side)], True
    n = record['native_payload']; p = n.get('provenance', {})
    metadata = p.get('comparison_capture')
    if not isinstance(metadata, dict):
        return m, None, [ComparisonReason('native_capture_metadata_missing', 'provenance.comparison_capture', side)], False
    if (metadata.get('format') != 'hyso.native_comparison_capture' or
            type(metadata.get('schema_version')) is not int or metadata['schema_version'] != 1):
        return m, None, [ComparisonReason('unsupported_capture_version', 'comparison_capture', side)], False
    if metadata.get('authority') != 'native_capture':
        return m, None, [ComparisonReason('capture_not_native', 'comparison_capture.authority', side)], False
    if set(metadata) != _CAPTURE_FIELDS:
        return m, None, [ComparisonReason('incomplete_capture_fields', 'comparison_capture', side)], False
    try:
        for key in _HASH_FIELDS + ('sample_sha256',):
            _sha(metadata[key], key)
        for key in _TEXT_FIELDS + ('run_id',):
            _text(metadata[key], key)
            if metadata[key].strip().lower() in ('unknown', 'not_collected', 'unavailable', 'n/a'):
                raise AdvisorError('unknown_capture_value')
    except AdvisorError:
        return m, None, [ComparisonReason('invalid_capture_value', 'comparison_capture', side)], False
    # Metadata is admitted only as producer capture data, never filled from
    # EvidenceScope declarations or report context. Bind it to the actual run
    # and retained samples; matching labels alone do not establish independence.
    raw = n.get('raw_evidence', n.get('raw', {}))
    latency = contract.metric in _QUANTILES
    rows = raw['latency_samples_ms'] if latency else raw
    try:
        if latency:
            _sha(raw['source_sha256'], 'native_source_sha256')
    except AdvisorError:
        return m, None, [ComparisonReason('native_identity_missing', 'source_sha256', side)], False
    run_id = raw['source_result_id'] if latency else n.get('result_id', n.get('evaluation_id'))
    if metadata['run_id'] != run_id or metadata['sample_sha256'] != digest(rows):
        reasons.append(ComparisonReason('capture_binding_mismatch', 'run_id/sample_sha256', side)); bad = True
    for key in ('checkpoint_sha256', 'dataset_id', 'split', 'workload_id'):
        declared = record['declared_scope'][key]
        if declared is not None and declared != metadata[key]:
            reasons.append(ComparisonReason('declaration_conflicts_with_capture', key, side)); bad = True
    count = (sum(len(row) for row in rows) if latency else
             m.native_context.get('sample_count') or n.get('actual', {}).get('observations') or
             n.get('actual', {}).get('completed_batches'))
    if type(count) is not int or count < contract.min_samples or (latency and
            (1 - Fraction(str(_QUANTILES[contract.metric]))) * count < contract.min_tail_samples):
        reasons.append(ComparisonReason('sample_requirement_not_met', 'samples/tail_samples', side))
    env, model, inp, config = (p.get(k, {}) for k in ('environment','model','input','configuration'))
    if any(type(v) is not dict for v in (env,model,inp,config)):
        return m, None, reasons + [ComparisonReason('native_identity_missing', 'native_conditions', side)], bad
    protocol_config = n.get('protocol', {}).get('configuration', {})
    for key in sorted(set(config) & set(protocol_config)):
        if config[key] != protocol_config[key]:
            reasons.append(ComparisonReason('native_configuration_conflict', key, side)); bad = True
    state = model.get('weights_sha256', model.get('state_sha256'))
    try:
        _sha(state, 'model_state_sha256'); _sha(inp['content_sha256'], 'input_content_sha256')
        shape = inp.get('shape')
        if type(shape) is not list or len(shape) < 2 or any(type(v) is not int or v < 1 for v in shape):
            raise AdvisorError('missing_input_shape')
        if not isinstance(env.get('backend'), (dict, str)) or not env['backend']:
            raise AdvisorError('missing_backend')
        if not model.get('structural_descriptor'):
            raise AdvisorError('missing_model_structure')
        device = env.get('device')
        if type(device) is not str or device != 'cpu' and re.fullmatch(r'cuda:[0-9]+', device) is None:
            raise AdvisorError('missing_device')
        if config.get('precision') not in (('fp32','fp64') if device=='cpu' else ('fp32','amp_fp16')):
            raise AdvisorError('missing_precision')
        for value in (inp['dtype'], inp['layout'], env['torch_version'], env['python_version'], env['operating_system']):
            _text(value,'native_condition')
            if value.strip().lower() in ('unknown','not_collected','unavailable','n/a'):
                raise AdvisorError('unknown_native_condition')
        if type(env['backend']) is str and env['backend'].strip().lower() in ('unknown','not_collected','unavailable','n/a'):
            raise AdvisorError('unknown_backend')
        for key in ('measure_iterations','repeats','warmup_iterations','inference_mode','forced_model_mode','input_policy','autocast_enabled'):
            if key not in config:
                raise AdvisorError('incomplete_native_configuration')
    except (AdvisorError, KeyError, TypeError):
        return m, None, reasons + [ComparisonReason('native_identity_missing', 'model/input/backend', side)], bad
    # Full protocol/configuration comparison except the explicitly variable
    # precision; batch is represented independently from nonbatch dimensions.
    fixed_config = {k:v for k,v in config.items() if k not in ('precision', 'autocast_enabled')}
    conditions = dict(metadata)
    conditions.update(model=(state, digest(model['structural_descriptor']), metadata['checkpoint_sha256']),
        device=env.get('device'), backend=env['backend'], runtime={k:v for k,v in env.items() if k not in ('device','backend')},
        batch=inp['shape'][0], nonbatch_shape=inp['shape'][1:], input_sha256=inp['content_sha256'],
        input_dtype=inp['dtype'], input_layout=inp['layout'], precision=config['precision'],
        autocast_enabled=config['autocast_enabled'],
        protocol=n.get('protocol', n.get('configuration')), configuration=fixed_config, unit=m.reference.unit, domain=m.reference.domain,
        measurement_scope=m.native_context['measurement_scope'], metric_context={k:v for k,v in m.native_context.items()
            if k not in ('sample_count','effective_tail_count','elapsed_wall_seconds','batch','precision')})
    return m, conditions, reasons, bad


def compare_records(left_resolver, left_record_id, right_resolver, right_record_id, *, contract=None):
    """Assess fair recorded conditions, never delta, significance or speedup.

    Resolvers validate native records using unchanged 3A constructors. Optional
    comparison capture metadata is a producer schema, not a caller-scope repair.
    Its authenticity is external, just as for existing native result artifacts.
    P26QualityComparisonContract explicitly selects its separate schema-2
    observed-stream policy; it never changes the default general v1 contract.
    """
    contract = ComparisonContract() if contract is None else contract
    if type(contract) is P26QualityComparisonContract:
        from ._p26_comparison import compare_p26_records
        return compare_p26_records(left_resolver, left_record_id, right_resolver, right_record_id, contract)
    if type(contract) is not ComparisonContract:
        raise AdvisorError('typed_comparison_contract_required')
    if type(left_resolver) is not NativeMetricResolver or type(right_resolver) is not NativeMetricResolver:
        raise AdvisorError('validated_native_resolvers_required')
    lm, left, lr, lb = _capture(left_resolver, left_record_id, contract, 'left')
    rm, right, rr, rb = _capture(right_resolver, right_record_id, contract, 'right')
    reasons = lr + rr
    bad = lb or rb
    lrecord, rrecord = left_resolver.record(left_record_id), right_resolver.record(right_record_id)
    if lrecord is not None and rrecord is not None:
        # Distinct analysis IDs or import timestamps cannot turn a reused
        # underlying capture into two independent measurement records.
        same = (left_record_id == right_record_id or
                lrecord['artifact'] and rrecord['artifact'] and lrecord['artifact']['sha256'] == rrecord['artifact']['sha256'] or
                lrecord['source'] and rrecord['source'] and lrecord['source']['payload_sha256'] == rrecord['source']['payload_sha256'])
        ln = lrecord['native_payload'] or {}; rn = rrecord['native_payload'] or {}
        lid = ln.get('raw_evidence', {}).get('source_result_id') or ln.get('result_id', ln.get('evaluation_id'))
        rid = rn.get('raw_evidence', {}).get('source_result_id') or rn.get('result_id', rn.get('evaluation_id'))
        lhash = ln.get('raw_evidence', {}).get('source_sha256')
        rhash = rn.get('raw_evidence', {}).get('source_sha256')
        duplicated_timing = (contract.metric in (NativeMetric.COMPLETED_IMAGES_PER_SECOND,
            NativeMetric.RSS_BASELINE, NativeMetric.RSS_CURRENT, NativeMetric.RSS_SAMPLED_PEAK)
            and ln.get('raw_evidence') and ln.get('raw_evidence') == rn.get('raw_evidence'))
        if same or lid and lid == rid or lhash and lhash == rhash or duplicated_timing:
            reasons.append(ComparisonReason('same_underlying_capture', 'record/run/artifact')); bad = True
        # Known disallowed native differences still reject a pair when other
        # critical capture metadata is missing. No equality of missing data is
        # interpreted as verified compatibility.
        if lm and rm and (left is None or right is None):
            lp, rp = ln.get('provenance', {}), rn.get('provenance', {})
            if not all(k in p for p in (lp,rp) for k in ('environment','configuration','input')):
                return ComparisonResult(ComparisonStatus.INCOMPATIBLE if bad else ComparisonStatus.INSUFFICIENT_EVIDENCE,
                    contract.content_sha256, tuple(reasons), lm.reference, rm.reference)
            fixed = [('protocol', ln.get('protocol'), rn.get('protocol'))]
            for key in ('device','backend'):
                fixed.append((key,lp['environment'].get(key),rp['environment'].get(key)))
            for key in ('warmup_iterations','measure_iterations','repeats','inference_mode','input_policy'):
                fixed.append((key,lp['configuration'].get(key),rp['configuration'].get(key)))
            for key in ('batch','precision','model_state_sha256'):
                group = 'model' if key == 'model_state_sha256' else key
                if group not in contract.allowed_changes:
                    fixed.append((key,lm.native_context.get(key,lm.record_scope.get(key)),
                                  rm.native_context.get(key,rm.record_scope.get(key))))
            li, ri = lp['input'], rp['input']
            ls, rs = li.get('shape'), ri.get('shape')
            if ls and rs:
                fixed.append(('nonbatch_shape', ls[1:], rs[1:]))
            changing_batch = ls and rs and ls[0] != rs[0] and 'batch' in contract.allowed_changes
            changing_precision = lp['configuration'].get('precision') != rp['configuration'].get('precision') and 'precision' in contract.allowed_changes
            fixed.append(('input_layout',li.get('layout'),ri.get('layout')))
            if not (changing_batch or changing_precision):
                fixed.append(('input_sha256',li.get('content_sha256'),ri.get('content_sha256')))
            if not changing_precision:
                fixed.append(('input_dtype',li.get('dtype'),ri.get('dtype')))
            for field, a, b in fixed:
                if a is not None and b is not None and a != b:
                    reasons.append(ComparisonReason('condition_mismatch',field)); bad = True
    if left is not None and right is not None:
        model_changed = left['model'] != right['model']
        batch_changed = left['batch'] != right['batch']
        precision_changed = left['precision'] != right['precision']
        for field, changed in (('model', model_changed), ('batch', batch_changed), ('precision', precision_changed)):
            if changed:
                allowed = field in contract.allowed_changes
                reasons.append(ComparisonReason('allowed_change' if allowed else 'unpermitted_change', field))
                bad |= not allowed
        variable = {'format','schema_version','authority','run_id','sample_sha256','checkpoint_sha256',
                    'model','batch','precision','input_sha256','input_dtype','autocast_enabled'}
        for key in sorted(set(left) - variable):
            if left[key] != right[key]:
                reasons.append(ComparisonReason('condition_mismatch', key)); bad = True
        if not (batch_changed and 'batch' in contract.allowed_changes or precision_changed and 'precision' in contract.allowed_changes):
            if left['input_sha256'] != right['input_sha256']:
                reasons.append(ComparisonReason('condition_mismatch', 'input_sha256')); bad = True
        if not (precision_changed and 'precision' in contract.allowed_changes) and left['input_dtype'] != right['input_dtype']:
            reasons.append(ComparisonReason('condition_mismatch','input_dtype')); bad = True
        if not (precision_changed and 'precision' in contract.allowed_changes) and left['autocast_enabled'] != right['autocast_enabled']:
            reasons.append(ComparisonReason('condition_mismatch','autocast_enabled')); bad = True
    missing = bool(lr or rr) or left is None or right is None
    status = ComparisonStatus.INCOMPATIBLE if bad else ComparisonStatus.INSUFFICIENT_EVIDENCE if missing else ComparisonStatus.COMPARABLE
    if status == ComparisonStatus.COMPARABLE:
        reasons.append(ComparisonReason('recorded_conditions_compatible', 'contract'))
    return ComparisonResult(status, contract.content_sha256, tuple(reasons),
                            lm.reference if lm else None, rm.reference if rm else None)
