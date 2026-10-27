"""Bounded plain JSON and integrity helpers; no workload execution."""
import hashlib
import json
import math
from pathlib import Path


class EvidenceError(ValueError):
    """Stable error code; raw source payloads are not included in messages."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def plain(value, depth=0):
    if depth > 128:
        raise EvidenceError('nesting_limit')
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list, tuple):
        return [plain(v, depth + 1) for v in value]
    if type(value) is dict and all(type(k) is str for k in value):
        return {k: plain(v, depth + 1) for k, v in value.items()}
    raise EvidenceError('not_plain_finite_json')


def canonical(value):
    return json.dumps(plain(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def fingerprint(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise EvidenceError('invalid_sha256')
    return value


def _pairs(items):
    result = {}
    for k, v in items:
        if k in result:
            raise EvidenceError('duplicate_json_key')
        result[k] = v
    return result


def decode(raw):
    try:
        return plain(json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs,
                                parse_constant=lambda _: (_ for _ in ()).throw(EvidenceError('nonfinite_json'))))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise EvidenceError('malformed_json') from exc


def read_json(path, expected_sha256=None, max_bytes=64 * 1024 ** 2):
    if type(max_bytes) is not int or not 1 <= max_bytes <= 128 * 1024 ** 2:
        raise EvidenceError('invalid_size_limit')
    try:
        with Path(path).open('rb') as source:
            raw = source.read(max_bytes + 1)
    except OSError as exc:
        raise EvidenceError('unreadable_artifact') from exc
    if len(raw) > max_bytes:
        raise EvidenceError('artifact_size_limit')
    actual = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual != fingerprint(expected_sha256):
        raise EvidenceError('artifact_sha256_mismatch')
    return decode(raw), actual, len(raw)


def export(path, text, overwrite=False):
    # Preserve the established atomic UTF-8/no-clobber behavior. No legacy
    # serializers or columns are changed by this separate format.
    from hyso.benchmark.result import _write_export
    _write_export(path, text, overwrite)
