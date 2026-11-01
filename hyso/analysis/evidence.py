"""Versioned immutable evidence references, with declared/native trust separation."""
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

    def to_dict(self): return plain(asdict(self))


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
    data['record_id'] = digest(data)
    return data


@dataclass(frozen=True, init=False)
class EvidenceRecord:
    _payload: str

    def __init__(self, data):
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
    def from_json(cls, path, *, feature_id, expected_sha256, scope=None, origin='new', max_bytes=64*1024**2):
        if scope is not None and type(scope) is not EvidenceScope:raise EvidenceError('typed_scope_required')
        fingerprint(expected_sha256)
        payload, actual, size = read_json(path, expected_sha256, max_bytes)
        if type(payload) is not dict: raise EvidenceError('invalid_native_evidence')
        artifact = dict(artifact_id='sha256:'+actual,name=Path(path).name, sha256=actual, bytes=size, media_type='application/json')
        return cls(_record(feature_id, payload, artifact, scope or EvidenceScope(), origin,
                           datetime.now(timezone.utc).isoformat()))

    @classmethod
    def unavailable(cls, feature_id, *, reason, status='not_run', scope=None):
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

    def to_dict(self): return json.loads(self._payload)

    @property
    def record_id(self): return self.to_dict()['record_id']


def compare_evidence(left, right, *, domain):
    """Fail closed. Agreement between caller declarations is not native proof."""
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
    _payload: str

    def __init__(self, records=()):
        values=[];seen=set()
        for r in records:
            if type(r) is not EvidenceRecord: raise EvidenceError('typed_records_required')
            d=EvidenceRecord(r.to_dict()).to_dict()
            if d['record_id'] in seen: raise EvidenceError('duplicate_record')
            seen.add(d['record_id']);values.append(d)
        payload=canonical(dict(format='hyso.evidence_index',schema_version=1,records=values))
        if len(payload.encode('utf-8'))>128*1024**2:raise EvidenceError('index_size_limit')
        object.__setattr__(self,'_payload',payload)

    def to_dict(self): return json.loads(self._payload)

    @property
    def records(self): return tuple(EvidenceRecord(d) for d in self.to_dict()['records'])

    def add(self, record): return EvidenceIndex((*self.records,record))

    def to_json(self, path, *, overwrite=False): export(path,self._payload+'\n',overwrite)

    @classmethod
    def from_json(cls,path):
        d,_,_=read_json(path,max_bytes=128*1024**2)
        if set(d)!={'format','schema_version','records'} or d['format']!='hyso.evidence_index' or type(d['schema_version']) is not int or d['schema_version']!=1:
            raise EvidenceError('unsupported_index_version')
        return cls(EvidenceRecord(r) for r in d['records'])

    def analyze(self, *, context, title='HYSo unified analysis', risks=()):
        from .report import UnifiedAnalysisReport
        return UnifiedAnalysisReport.build(self,context=context,title=title,risks=risks)
