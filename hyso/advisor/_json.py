"""Bounded, deterministic JSON for Advisor contracts. No evidence is resolved."""
from dataclasses import fields
from enum import Enum
import hashlib
import json
from pathlib import Path

MAX_BYTES = 2 * 1024 * 1024
MAX_DEPTH = 24
_TYPES = {}


class AdvisorError(ValueError):
    """A stable contract validation error."""


def register(cls):
    _TYPES[cls.__name__] = cls
    return cls


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def _encode(value):
    if isinstance(value, Contract):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise AdvisorError('unsupported_json_value')


def _decode(value, depth=0):
    if depth > MAX_DEPTH:
        raise AdvisorError('nesting_limit')
    if type(value) is list:
        return tuple(_decode(v, depth + 1) for v in value)
    if type(value) is dict:
        name = value.get('format', '').removeprefix('hyso.advisor.')
        cls = _TYPES.get(name)
        if cls is None or value.get('format') != 'hyso.advisor.' + name:
            raise AdvisorError('unknown_contract_format')
        if type(value.get('schema_version')) is not int or value['schema_version'] != 1:
            raise AdvisorError('unsupported_schema_version')
        expected = {f.name for f in fields(cls)} | {'format', 'schema_version', 'content_sha256'}
        if set(value) != expected:
            raise AdvisorError('invalid_contract_fields')
        body = {k: v for k, v in value.items() if k != 'content_sha256'}
        if hashlib.sha256(canonical(body).encode('utf-8')).hexdigest() != value['content_sha256']:
            raise AdvisorError('contract_integrity_mismatch')
        kwargs = {f.name: _decode(value[f.name], depth + 1) for f in fields(cls) if f.init}
        try:
            result = cls(**kwargs)
        except (TypeError, AttributeError) as exc:
            raise AdvisorError('invalid_contract_value') from exc
        if result.to_dict() != value:
            raise AdvisorError('derived_contract_mismatch')
        return result
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        import math
        if math.isfinite(value):
            return value
    raise AdvisorError('not_finite_plain_json')


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise AdvisorError('duplicate_json_key')
        result[key] = value
    return result


class Contract:
    """Frozen subclasses store only typed contracts, scalars and tuples."""

    def _body(self):
        return dict(format='hyso.advisor.' + type(self).__name__, schema_version=1,
                    **{f.name: _encode(getattr(self, f.name)) for f in fields(self)})

    @property
    def content_sha256(self):
        return hashlib.sha256(canonical(self._body()).encode('utf-8')).hexdigest()

    def to_dict(self):
        return dict(self._body(), content_sha256=self.content_sha256)

    @classmethod
    def from_dict(cls, data):
        try:
            raw = canonical(data).encode('utf-8')
        except (TypeError, ValueError, RecursionError) as exc:
            raise AdvisorError('not_finite_plain_json') from exc
        if len(raw) > MAX_BYTES:
            raise AdvisorError('contract_size_limit')
        try:
            result = _decode(data)
        except (RecursionError, TypeError, AttributeError) as exc:
            raise AdvisorError('invalid_contract_value') from exc
        if type(result) is not cls:
            raise AdvisorError('wrong_contract_type')
        return result

    def to_json(self, path=None):
        raw = canonical(self.to_dict()) + '\n'
        if len(raw.encode('utf-8')) > MAX_BYTES:
            raise AdvisorError('contract_size_limit')
        if path is not None:
            # Explicit exclusive export; never overwrite caller files.
            with Path(path).open('x', encoding='utf-8', newline='\n') as target:
                target.write(raw)
        return raw

    @classmethod
    def from_json(cls, path):
        with Path(path).open('rb') as source:
            raw = source.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise AdvisorError('contract_size_limit')
        try:
            data = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs,
                              parse_constant=lambda _: (_ for _ in ()).throw(AdvisorError('nonfinite_json')))
        except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise AdvisorError('malformed_json') from exc
        return cls.from_dict(data)
