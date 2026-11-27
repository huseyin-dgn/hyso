"""Versioned immutable evidence references, with declared/native trust separation."""
import typing as _typing
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from ._codec import EvidenceError, canonical, decode, digest, fingerprint, plain, read_json, export
from ._native import validate_native, native_scope, metrics, limitations, native_outcome

FEATURES = tuple('P%02d' % i for i in range(1, 36))
_PRECISIONS = {'float32':'fp32','torch.float32':'fp32','float64':'fp64','torch.float64':'fp64'}


@dataclass(frozen=True)
class EvidenceScope:
    """Bind retained evidence to its model, phase, device, precision, batch and source context.

    Args:
        model_id (str | None): Stable model label bound to this result/context. Default: None.
        model_state_sha256 (str | None): Exact SHA-256 digest of the model state. Default: None.
        checkpoint_sha256 (str | None): Exact SHA-256 digest of the checkpoint. Default: None.
        dataset_id (str | None): Exact dataset identity used for evidence alignment. Default: None.
        split (str | None): Explicit dataset split label used in evaluation identity. Default: None.
        preprocessing_id (str | None): Identity of the preprocessing used to produce this fixed cohort.
            Default: None.
        device (str | None): Optional device selector following this API's native placement/default
            rules; legacy EBO may choose CUDA. Pass cpu explicitly for CPU use. Default: None.
        backend (str | None): Explicit runtime/backend identity. Default: None.
        precision (str | None): Supported numerical format; runtime/task/device restrictions remain
            authoritative. Default: None.
        batch (int | None): Batch dimension of the measured input. Default: None.
        input_shape (tuple | None): Per-sample (C, H, W) dimensions; batch is excluded. Default: None.
        workload_id (str | None): Exact workload identity for scope alignment. Default: None.
        protocol_id (str | None): Exact measurement/evaluation protocol identity. Default: None.
        protocol_sha256 (str | None): Exact SHA-256 digest of the protocol. Default: None.
        phase (str | None): Explicit trace phase label. Default: None.
        input_kind (str | None): Explicit supported input-source kind. Default: None.
        clock_id (str | None): Clock-domain identity used for timing alignment. Default: None.
        source_revision (str | None): Exact source code/configuration revision. Default: None.

    Returns:
        An instance of EvidenceScope with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        EvidenceScope()

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """
    model_id: str | None = None
    model_state_sha256: str | None = None
    checkpoint_sha256: str | None = None
    dataset_id: str | None = None
    split: str | None = None
    preprocessing_id: str | None = None
    device: str | None = None
    backend: str | None = None
    precision: str | None = None
    batch: int | None = None
    input_shape: tuple | None = None
    workload_id: str | None = None
    protocol_id: str | None = None
    protocol_sha256: str | None = None
    phase: str | None = None
    input_kind: str | None = None
    clock_id: str | None = None
    source_revision: str | None = None

    def __post_init__(self):
        for key, value in asdict(self).items():
            if value is None: continue
            if key.endswith('_sha256'): fingerprint(value)
            elif key == 'batch':
                if type(value) is not int or value <= 0: raise EvidenceError('invalid_batch')
            elif key == 'input_shape':
                if type(value) not in (list,tuple) or not value or any(type(n) is not int or n <= 0 for n in value):
                    raise EvidenceError('invalid_input_shape')
                object.__setattr__(self, key, tuple(value))
            elif type(value) is not str or not value.strip(): raise EvidenceError('invalid_scope_field')
        if self.input_shape and self.batch is not None and self.input_shape[0] != self.batch:
            raise EvidenceError('batch_shape_mismatch')
        if self.precision is not None:
            object.__setattr__(self, 'precision', _PRECISIONS.get(self.precision, self.precision))

    def to_dict(self) -> 'dict[str, _typing.Any]':
        """Return a detached plain-data view of the stored state.

        Returns:
            A detached plain-data view of this validated object; native units and unavailable values are
            retained.

        Raises:
            No additional validation for a valid object; corrupt retained state or native copy/serialization
            failures propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_dict()

        HYSo Guide: analyzer.workflow
        Offline: hyso docs show analyzer.workflow --lang en
        Open: hyso docs open analyzer.workflow --lang en
        """
        return plain(asdict(self))


def _scope(data, feature, declared):
    native = native_scope(data, feature)
    # Multi-point bundles cannot be silently assigned one batch/shape/precision.
    points = data.get('raw_evidence', {}).get('points', [])
    varying = set()
    for key in ('precision', 'shape'):
        if len({canonical(p.get(key)) for p in points}) > 1:
            varying.add('precision' if key == 'precision' else 'input_shape')
            if key == 'shape': varying.add('batch')
    values, bindings = {}, {}
    declarations = declared.to_dict()
    if feature == 'P26' and data.get('schema_version') in (2, 3):
        split = data['configuration']['split']
        if declarations['split'] is not None and declarations['split'] != split:
            raise EvidenceError('declared_split_mismatch')
        declarations['split'] = split
    for key in declarations:
        actual, supplied = native.get(key), declarations[key]
        if key in varying:
            if supplied is not None: raise EvidenceError('varying_scope_requires_point_selection')
            actual = None
        if key == 'precision': actual = _PRECISIONS.get(actual, actual)
        if supplied is not None and actual is not None and supplied != actual:
            raise EvidenceError('native_scope_mismatch:' + key)
        values[key] = actual if actual is not None else supplied
        bindings[key] = 'native' if actual is not None else 'declared' if supplied is not None else 'missing'
    result = EvidenceScope(**values).to_dict()
    return result, bindings


def _record(feature, payload, artifact, declarations, origin, imported_at):
    if feature not in FEATURES: raise EvidenceError('invalid_feature_id')
    if payload.get('format') in ('hyso.performance.worker_trace', 'hyso.performance.worker_tasks'):
        raise EvidenceError('worker_companion_requires_base_record')
    if origin not in ('new', 'reused'): raise EvidenceError('invalid_origin')
    try:
        if datetime.fromisoformat(imported_at).tzinfo is None:raise ValueError()
    except (ValueError,TypeError):raise EvidenceError('invalid_import_time')
    if (set(artifact) != {'artifact_id','name','sha256','bytes','media_type'} or type(artifact['name']) is not str
            or not artifact['name'] or type(artifact['bytes']) is not int or artifact['bytes']<1
            or artifact['media_type']!='application/json'):
        raise EvidenceError('invalid_artifact_reference')
    fingerprint(artifact['sha256'])
    if artifact['artifact_id'] != 'sha256:'+artifact['sha256']:raise EvidenceError('invalid_artifact_identity')
    result_type, features = validate_native(payload)
    if feature not in features: raise EvidenceError('source_feature_mismatch')
    scope, bindings = _scope(payload, feature, declarations)
    result_id = next((payload.get(k) for k in ('result_id','segment_id','evaluation_id','experiment_id') if payload.get(k)), None)
    data = dict(format='hyso.evidence_record', record_version=1, adapter_version=1,
                feature_id=feature, origin=origin, imported_at_utc=imported_at,
                observed_at_utc=None, clock_name=payload.get('protocol',{}).get('clock'), source=dict(result_type=result_type,
                format=payload.get('format','hyso.benchmark.inference'), schema_version=payload['schema_version'],
                result_id=result_id, native_status=native_outcome(payload), source_id=payload.get('source_segment_id', payload.get('identity',{}).get('source_result_id')),
                declared_source_sha256=payload.get('source_sha256'), payload_sha256=digest(payload)),
                artifact=artifact, declared_scope=declarations.to_dict(), scope=scope, bindings=bindings,
                validation=dict(status='native_validated', reason=None,
                    provenance='Some associations may be caller-declared; no authenticity attestation.'),
                native_payload=payload, metrics=metrics(payload,feature), limitations=limitations(feature,payload)+[
                    'Capture UTC time and clock-domain identity are unknown unless natively recorded; import time is not measurement time.'])
    if payload.get('format') == 'hyso.performance.isolated_training_memory':
        measurement = payload.get('measurement') or {}
        data['source'].update(result_id=payload['probe_id'], source_id=measurement.get('source_segment_id'),
                              declared_source_sha256=measurement.get('source_sha256'))
    data['record_id'] = digest(data)
    return data


@dataclass(frozen=True, init=False)
class EvidenceRecord:
    """Validate and retain one measurement record with identity, scope, protocol and limitations.

    Args:
        data (dict[str, _typing.Any]): Detached plain-data payload accepted by the existing
            version/schema validator. Required.

    Returns:
        An instance of EvidenceRecord with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        EvidenceRecord(data)

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """
    _payload: str

    def __init__(self, data: 'dict[str, _typing.Any]') -> 'None':
        """Initialize EvidenceRecord with the supplied settings.

        Args:
            data (dict[str, _typing.Any]): Detached plain-data payload accepted by the existing
                version/schema validator. Required.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            EvidenceRecord(data)

        HYSo Guide: analyzer.workflow
        Offline: hyso docs show analyzer.workflow --lang en
        Open: hyso docs open analyzer.workflow --lang en
        """
        d = plain(data)
        if type(d) is not dict: raise EvidenceError('invalid_record')
        if d.get('record_version') != 1 or type(d.get('record_version')) is not int:
            raise EvidenceError('unsupported_record_version')
        if d.get('format') != 'hyso.evidence_record' or type(d.get('adapter_version')) is not int or d.get('adapter_version') != 1:
            raise EvidenceError('invalid_record')
        original = dict(d); identity = original.pop('record_id', None)
        if identity != digest(original): raise EvidenceError('record_integrity_mismatch')
        if d['native_payload'] is None:
            rebuilt = self.unavailable(d['feature_id'], reason=d['validation']['reason'],
                       status=d['validation']['status'], scope=EvidenceScope(**d['declared_scope'])).to_dict()
            if rebuilt != d: raise EvidenceError('invalid_unavailable_record')
        else:
            fingerprint(d['artifact']['sha256'])
            rebuilt = _record(d['feature_id'], d['native_payload'], d['artifact'],
                              EvidenceScope(**d['declared_scope']), d['origin'], d['imported_at_utc'])
            if rebuilt != d: raise EvidenceError('record_projection_mismatch')
        object.__setattr__(self, '_payload', canonical(d))

    @classmethod
    def from_native(cls, payload: dict, *, feature_id: _typing.Literal['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35'], scope: EvidenceScope | None = None,
                    origin: _typing.Literal['new', 'reused'] = 'new') -> 'EvidenceRecord':
        """Validate an in-memory native result using the existing reader contracts.

        Artifact identity describes canonical JSON bytes retained in this record;
        it does not claim a file was written. Export explicitly when persistence
        is needed. Import time is not capture time; associations remain declared.

        Args:
            payload (dict): Plain retained evidence payload. Required.
            feature_id (str): Exact P01-P35 feature identifier. Required.
            scope (EvidenceScope | None): Exact measurement boundary/meaning; different scopes are not
                interchangeable. Default: None.
            origin (str): Explicit evidence origin/provenance. Default: 'new'.

        Returns:
            A validated EvidenceRecord instance reconstructed from the supplied data.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            EvidenceRecord.from_native(payload, feature_id=feature_id)

        HYSo Guide: analyzer.workflow
        Offline: hyso docs show analyzer.workflow --lang en
        Open: hyso docs open analyzer.workflow --lang en
        """
        if scope is not None and type(scope) is not EvidenceScope:
            raise EvidenceError('typed_scope_required')
        raw = canonical(payload).encode('utf-8')
        if len(raw) > 64*1024**2:
            raise EvidenceError('artifact_size_limit')
        # Project metrics from canonical key order, just as on subsequent JSON
        # reloads. Native dictionaries can have a different insertion order.
        data = json.loads(raw)
        sha = digest(data)
        artifact = dict(artifact_id='sha256:'+sha,name='in-memory-'+sha+'.json',
                        sha256=sha,bytes=len(raw),media_type='application/json')
        return cls(_record(feature_id,data,artifact,scope or EvidenceScope(),origin,
                           datetime.now(timezone.utc).isoformat()))

    @classmethod
    def from_json(cls, path: '_typing.Any', *, feature_id: _typing.Literal['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35'], expected_sha256: '_typing.Any', scope: '_typing.Any'=None, origin: _typing.Literal['new', 'reused']='new', max_bytes: '_typing.Any'=64*1024**2) -> 'EvidenceRecord':
        """Read and validate this type from a local JSON file.

        Args:
            path (_typing.Any): Filesystem path for the specific reader/exporter; no configuration code is
                executed. Required.
            feature_id (_typing.Any): Exact P01-P35 feature identifier. Required.
            expected_sha256 (_typing.Any): Exact expected artifact/content digest. Required.
            scope (_typing.Any): Exact measurement boundary/meaning; different scopes are not
                interchangeable. Default: None.
            origin (str): Explicit evidence origin/provenance. Default: 'new'.
            max_bytes (_typing.Any): Maximum accepted or retained byte count. Default: 64 * 1024 ** 2.

        Returns:
            A validated EvidenceRecord instance reconstructed from the supplied data.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            EvidenceRecord.from_json(path, feature_id=feature_id, expected_sha256=expected_sha256)

        HYSo Guide: analyzer.workflow
        Offline: hyso docs show analyzer.workflow --lang en
        Open: hyso docs open analyzer.workflow --lang en
        """
        if scope is not None and type(scope) is not EvidenceScope:raise EvidenceError('typed_scope_required')
        fingerprint(expected_sha256)
        payload, actual, size = read_json(path, expected_sha256, max_bytes)
        if type(payload) is not dict: raise EvidenceError('invalid_native_evidence')
        artifact = dict(artifact_id='sha256:'+actual,name=Path(path).name, sha256=actual, bytes=size, media_type='application/json')
        return cls(_record(feature_id, payload, artifact, scope or EvidenceScope(), origin,
                           datetime.now(timezone.utc).isoformat()))

    @classmethod
    def unavailable(cls, feature_id: _typing.Literal['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35'], *, reason: '_typing.Any', status: _typing.Literal['not_run', 'unsupported', 'invalid', 'unknown']='not_run', scope: '_typing.Any'=None) -> 'EvidenceRecord':
        """Create an explicit unavailable evidence record with its reason and scope.

        Args:
            feature_id (_typing.Any): Exact P01-P35 feature identifier. Required.
            reason (_typing.Any): Explicit human-readable lifecycle reason; not an automatic workload
                diagnosis. Required.
            status (str): Exact lifecycle/status filter accepted by this API. Default: 'not_run'.
            scope (_typing.Any): Exact measurement boundary/meaning; different scopes are not
                interchangeable. Default: None.

        Returns:
            An EvidenceRecord with the explicit unavailable status, reason and scope.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            EvidenceRecord.unavailable(feature_id, reason=reason)

        HYSo Guide: analyzer.workflow
        Offline: hyso docs show analyzer.workflow --lang en
        Open: hyso docs open analyzer.workflow --lang en
        """
        if feature_id not in FEATURES: raise EvidenceError('invalid_feature_id')
        if status not in ('not_run','unsupported','invalid','unknown') or type(reason) is not str or not reason.strip():
            raise EvidenceError('unavailable_requires_reason')
        scope = scope or EvidenceScope()
        d = dict(format='hyso.evidence_record', record_version=1, adapter_version=1,
                 feature_id=feature_id, origin='none', imported_at_utc=None, observed_at_utc=None,clock_name=None,
                 source=None, artifact=None, declared_scope=scope.to_dict(), scope=scope.to_dict(),
                 bindings={k:'declared' if v is not None else 'missing' for k,v in scope.to_dict().items()},
                 validation=dict(status=status,reason=reason),native_payload=None,metrics=[],limitations=[reason])
        d['record_id'] = digest(d)
        # Build directly; __init__ verifies unavailable round-trips with this method.
        result = object.__new__(cls);object.__setattr__(result,'_payload',canonical(d));return result

    def to_dict(self) -> 'dict[str, _typing.Any]':
        """Return a detached plain-data view of the stored state.

        Returns:
            A detached plain-data view of this validated object; native units and unavailable values are
            retained.

        Raises:
            No additional validation for a valid object; corrupt retained state or native copy/serialization
            failures propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_dict()

        HYSo Guide: analyzer.workflow
        Offline: hyso docs show analyzer.workflow --lang en
        Open: hyso docs open analyzer.workflow --lang en
        """
        return json.loads(self._payload)

    @property
    def record_id(self) -> '_typing.Any':
        """Read the retained record id from this EvidenceRecord record.

        Returns:
            The retained record id value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.record_id

        HYSo Guide: analyzer.workflow
        Offline: hyso docs show analyzer.workflow --lang en
        Open: hyso docs open analyzer.workflow --lang en
        """
        return self.to_dict()['record_id']


def compare_evidence(left: '_typing.Any', right: '_typing.Any', *, domain: '_typing.Any') -> '_typing.Any':
    """Fail closed. Agreement between caller declarations is not native proof.

    Args:
        left (_typing.Any): Left retained input/source in this comparison. Required.
        right (_typing.Any): Right retained input/source in this comparison. Required.
        domain (_typing.Any): Feature/metric domain, separate from runtime support. Required.

    Returns:
        A plain comparison of the retained records, including compatibility and missing-evidence
        reasons.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        compare_evidence(left, right, domain=domain)

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """
    if type(left) is not EvidenceRecord or type(right) is not EvidenceRecord:
        raise EvidenceError('typed_records_required')
    a,b=left.to_dict(),right.to_dict();mismatch=[];missing=[];declared=[]
    if type(domain) is not str or not domain: raise EvidenceError('measurement_domain_required')
    for record in (a,b):
        if not any(m['domain']==domain for m in record['metrics']):mismatch.append('measurement_domain')
    required=('model_id','checkpoint_sha256','model_state_sha256','dataset_id','split','preprocessing_id',
              'device','backend','precision','batch','input_shape','workload_id','protocol_id',
              'protocol_sha256','phase','input_kind','clock_id','source_revision')
    for key in required:
        x,y=a['scope'][key],b['scope'][key]
        if x is None or y is None: missing.append(key)
        elif x!=y: mismatch.append(key)
        elif a['bindings'][key]!='native' or b['bindings'][key]!='native': declared.append(key)
    status='incompatible' if mismatch else 'insufficient_evidence' if missing or declared else 'compatible'
    return dict(status=status,domain=domain,mismatched_fields=sorted(set(mismatch)),
                missing_fields=missing,declared_only_fields=declared,
                reason='No cross-domain conversion, imputation or recommendation is performed.')


@dataclass(frozen=True, init=False)
class EvidenceIndex:
    """Collect retained records and expose evidence availability without executing missing analyses.

    Args:
        records (_typing.Sequence[int]): Bounded sequence of plain observation records. Default: ().

    Returns:
        An instance of EvidenceIndex with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        EvidenceIndex()

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """
    _payload: str

    def __init__(self, records: '_typing.Sequence[EvidenceRecord]'=()) -> 'None':
        """Initialize EvidenceIndex with the supplied settings.

        Args:
            records (_typing.Sequence[EvidenceRecord]): Bounded sequence of plain observation records.
                Default: ().

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            EvidenceIndex()

        HYSo Guide: analyzer.workflow
        Offline: hyso docs show analyzer.workflow --lang en
        Open: hyso docs open analyzer.workflow --lang en
        """
        values=[];seen=set()
        for r in records:
            if type(r) is not EvidenceRecord: raise EvidenceError('typed_records_required')
            d=EvidenceRecord(r.to_dict()).to_dict()
            if d['record_id'] in seen: raise EvidenceError('duplicate_record')
            seen.add(d['record_id']);values.append(d)
        payload=canonical(dict(format='hyso.evidence_index',schema_version=1,records=values))
        if len(payload.encode('utf-8'))>128*1024**2:raise EvidenceError('index_size_limit')
        object.__setattr__(self,'_payload',payload)

    def to_dict(self) -> 'dict[str, _typing.Any]':
        """Return a detached plain-data view of the stored state.

        Returns:
            A detached plain-data view of this validated object; native units and unavailable values are
            retained.

        Raises:
            No additional validation for a valid object; corrupt retained state or native copy/serialization
            failures propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_dict()

        HYSo Guide: analyzer.workflow
        Offline: hyso docs show analyzer.workflow --lang en
        Open: hyso docs open analyzer.workflow --lang en
        """
        return json.loads(self._payload)

    @property
    def records(self) -> '_typing.Any':
        """Read the retained records from this EvidenceIndex record.

        Returns:
            The retained records value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.records

        HYSo Guide: analyzer.workflow
        Offline: hyso docs show analyzer.workflow --lang en
        Open: hyso docs open analyzer.workflow --lang en
        """
        return tuple(EvidenceRecord(d) for d in self.to_dict()['records'])

    def add(self, record: '_typing.Any') -> 'EvidenceIndex':
        """Add one validated retained evidence record to this index without measuring a workload.

        Args:
            record (_typing.Any): Typed retained measurement/evidence record. Required.

        Returns:
            A new EvidenceIndex containing the previous records and this record; this index is unchanged.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.add(record)

        HYSo Guide: analyzer.workflow
        Offline: hyso docs show analyzer.workflow --lang en
        Open: hyso docs open analyzer.workflow --lang en
        """
        return EvidenceIndex((*self.records,record))

    def to_json(self, path: '_typing.Any', *, overwrite: 'bool'=False) -> 'None':
        """Serialize this object using its native JSON format and overwrite policy.

        Args:
            path (_typing.Any): Filesystem path for the specific reader/exporter; no configuration code is
                executed. Required.
            overwrite (bool): Whether an existing output file may be replaced explicitly; inputs remain
                protected. Default: False.

        Returns:
            None after the native file export succeeds.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_json(path)

        HYSo Guide: analyzer.workflow
        Offline: hyso docs show analyzer.workflow --lang en
        Open: hyso docs open analyzer.workflow --lang en
        """
        export(path,self._payload+'\n',overwrite)

    @classmethod
    def from_json(cls,path: '_typing.Any') -> 'EvidenceIndex':
        """Read and validate this type from a local JSON file.

        Args:
            path (_typing.Any): Filesystem path for the specific reader/exporter; no configuration code is
                executed. Required.

        Returns:
            A validated EvidenceIndex instance reconstructed from the supplied data.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            EvidenceIndex.from_json(path)

        HYSo Guide: analyzer.workflow
        Offline: hyso docs show analyzer.workflow --lang en
        Open: hyso docs open analyzer.workflow --lang en
        """
        d,_,_=read_json(path,max_bytes=128*1024**2)
        if set(d)!={'format','schema_version','records'} or d['format']!='hyso.evidence_index' or type(d['schema_version']) is not int or d['schema_version']!=1:
            raise EvidenceError('unsupported_index_version')
        return cls(EvidenceRecord(r) for r in d['records'])

    def analyze(self, *, context: '_typing.Any', title: 'str'='HYSo unified analysis', risks: '_typing.Sequence[str]'=()) -> '_typing.Any':
        """Summarize retained evidence availability and limitations without executing missing analyses.

        Args:
            context (_typing.Any): Explicit analysis/evaluation context; caller declarations are not
                producer attestation. Required.
            title (str): Human-readable report title. Default: 'HYSo unified analysis'.
            risks (_typing.Sequence[str]): Known limitations/risks attached to the proposal. Default: ().

        Returns:
            A UnifiedAnalysisReport describing supplied evidence, coverage, missing features and
            limitations.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.analyze(context=context)

        HYSo Guide: analyzer.workflow
        Offline: hyso docs show analyzer.workflow --lang en
        Open: hyso docs open analyzer.workflow --lang en
        """
        from .report import UnifiedAnalysisReport
        return UnifiedAnalysisReport.build(self,context=context,title=title,risks=risks)
