"""Additive P30--P33 immutable evidence; independent of legacy schema v1."""
from dataclasses import dataclass
import hashlib
import json
import math
import uuid

from hyso.benchmark.result import _write_export, _csv_text


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _int(value, minimum=0, maximum=2**63-1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('integer outside supported bounds')


def _number(value, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError('finite measurement required')


def _name(value):
    if type(value) is not str or not value.strip() or len(value)>4096:
        raise ValueError('explicit bounded identity required')


def _digest(value):
    if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('SHA256 required')


def _derive(feature, evidence):
    if feature=='P30':
        from .streaming import _analyze_stream
        return _analyze_stream(evidence)
    if feature=='P31':
        from .stability import _analyze_stability
        return _analyze_stability(evidence)
    if feature=='P32':
        from .memory_trend import _analyze_trend
        return _analyze_trend(evidence)
    if feature=='P33':
        from .energy import _analyze_energy
        return _analyze_energy(evidence)
    if feature=='P34':
        from .devices import _analyze_devices
        return _analyze_devices(evidence)
    if feature=='P35':
        from .providers import _analyze_providers
        return _analyze_providers(evidence)
    raise ValueError('unsupported runtime feature')


@dataclass(frozen=True, init=False)
class RuntimeResult:
    """Self-contained snapshot. Hashes detect inconsistency, not signed authenticity."""
    __slots__=('_payload',)
    _payload: str

    def __init__(self, data):
        try:
            value=json.loads(json.dumps(data, allow_nan=False))
            if (set(value)!={'format','schema_version','feature_id','result_id','evidence','source_sha256','analysis'}
                    or value['format']!='hyso.performance.runtime' or type(value['schema_version']) is not int
                    or value['schema_version']!=1 or str(uuid.UUID(value['result_id']))!=value['result_id']):
                raise ValueError('invalid runtime result schema')
            if value['source_sha256']!=_hash(dict(feature_id=value['feature_id'],evidence=value['evidence'])):
                raise ValueError('source hash mismatch')
            if _hash(value['analysis'])!=_hash(_derive(value['feature_id'],value['evidence'])):
                raise ValueError('derived runtime analysis mismatch')
        except (KeyError, TypeError, IndexError, OverflowError) as exc:
            raise ValueError('malformed runtime evidence') from exc
        object.__setattr__(self, '_payload', json.dumps(value, allow_nan=False))

    def to_dict(self):
        return json.loads(self._payload)

    def stability_diagnostics(self):
        """P31 descriptive diagnostics; legacy JSON/CSV schemas stay intact."""
        data=RuntimeResult(self.to_dict()).to_dict()
        if data['feature_id']!='P31':raise TypeError('P31 result required')
        from ._stability_diagnostics import _diagnostics
        return _diagnostics(data['evidence'],data['analysis'])

    def provider_diagnostics(self):
        """P35 observed providers, host Node durations and explicit fallback limits."""
        data=RuntimeResult(self.to_dict()).to_dict()
        if data['feature_id']!='P35':raise TypeError('P35 result required')
        from .providers import _diagnostics
        return _diagnostics(data['evidence'],data['analysis'])

    def device_diagnostics(self, *, baseline_device=None):
        """P34 offline pairwise comparisons; never launches inference or sensors."""
        data=RuntimeResult(self.to_dict()).to_dict()
        if data['feature_id']!='P34':raise TypeError('P34 result required')
        from ._device_diagnostics import _diagnostics
        return _diagnostics(data['evidence'],data['analysis'],baseline_device=baseline_device)

    def to_json(self, path, *, overwrite=False):
        _write_export(path, json.dumps(self.to_dict(), indent=2, allow_nan=False)+'\n', overwrite)

    def to_csv(self, path, *, overwrite=False):
        data=self.to_dict();rows=[]
        if data['feature_id'] in ('P34','P35') or (data['feature_id']=='P33' and
                data['evidence'].get('energy_version')==2) or (data['feature_id']=='P32' and
                data['evidence'].get('trend_version')==2) or (data['feature_id']=='P31' and
                ((data['evidence']['header'] is not None and data['evidence']['header'].get('stability_version')==2)
                 or 'resource_monitor' in data['evidence']['controller'])) or (data['feature_id']=='P30'
                and 'max_journal_bytes' in data['evidence']['request']['protocol']):
            # A canonical envelope preserves types, empty collections and raw events.
            _write_export(path,_csv_text(['feature_id','result_id','source_sha256','payload'],
                [dict(feature_id=data['feature_id'],result_id=data['result_id'],
                      source_sha256=data['source_sha256'],payload=self._payload)]),overwrite)
            return
        def walk(value, field):
            if type(value) is dict:
                for key, item in value.items():walk(item,field+'.'+key)
            elif type(value) is list:
                for i, item in enumerate(value):walk(item,field+'.'+str(i))
            else:rows.append(dict(feature_id=data['feature_id'],result_id=data['result_id'],
                source_sha256=data['source_sha256'],field=field,value=value))
        walk(data['analysis'],'analysis');walk(data['evidence'],'evidence')
        _write_export(path,_csv_text(['feature_id','result_id','source_sha256','field','value'],rows),overwrite)

    @classmethod
    def from_json(cls, path):
        from .stability import _load
        return cls(_load(path))

    @classmethod
    def from_csv(cls, path):
        import csv
        from pathlib import Path
        if Path(path).stat().st_size>128*1024**2:raise ValueError('oversized evidence')
        previous=csv.field_size_limit()
        try:
            csv.field_size_limit(128*1024**2)
            with open(path,encoding='utf-8',newline='') as source:
                reader=csv.DictReader(source)
                if reader.fieldnames!=['feature_id','result_id','source_sha256','payload']:
                    raise ValueError('lossless runtime evidence CSV required')
                rows=list(reader)
        finally:csv.field_size_limit(previous)
        if len(rows)!=1:raise ValueError('one evidence envelope required')
        row=rows[0];result=cls(json.loads(row['payload']));data=result.to_dict()
        if any(row[k]!=data[k] for k in ('feature_id','result_id','source_sha256')):
            raise ValueError('CSV identity mismatch')
        return result


def _result(feature, evidence):
    return RuntimeResult(dict(format='hyso.performance.runtime',schema_version=1,feature_id=feature,
        result_id=str(uuid.uuid4()),evidence=evidence,
        source_sha256=_hash(dict(feature_id=feature,evidence=evidence)),analysis=_derive(feature,evidence)))
