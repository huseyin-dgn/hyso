"""Stdlib-only campaign primitives. Importing this package never runs a test."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEGACY = REPO / 'outputs/release-acceptance-20261125'
ROOT = LEGACY / 'final-campaign'
STATUSES = ('PASS', 'FAIL', 'ERROR', 'SKIP', 'BLOCKED', 'TIMEOUT', 'NOT_RUN')

# The product suite intentionally patches os/json/hashlib/time. Evidence is a
# separate persistence boundary: bind its primitives before any test fixtures run.
_ENCODE = json.JSONEncoder(sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode
_PRETTY_ENCODE = json.JSONEncoder(indent=2,ensure_ascii=False,allow_nan=False).encode
_DECODE = json.JSONDecoder().decode
_SHA256 = hashlib.sha256
_FSYNC, _REPLACE, _SLEEP = os.fsync, os.replace, time.sleep
_OPEN_FD, _WRITE_FD, _CLOSE_FD = os.open, os.write, os.close
_UTC_NOW, _UTC = datetime.now, timezone.utc


def utc():
    return _UTC_NOW(_UTC).isoformat()


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return _DECODE(path.read_text(encoding='utf-8-sig'))


def canonical(value):
    return _ENCODE(value).encode('utf-8')


def digest(value):
    return _SHA256(canonical(value)).hexdigest()


def file_hash(path):
    h = _SHA256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _PRETTY_ENCODE(value) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                dir=path.parent, prefix='.' + path.name + '-', suffix='.tmp', delete=False) as out:
            temporary = Path(out.name)
            out.write(payload)
            out.flush()
            _FSYNC(out.fileno())
        for attempt in range(6):
            try:
                _REPLACE(temporary, path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                _SLEEP(.05 * (2 ** attempt))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as out:
        out.write(_PRETTY_ENCODE(value)+'\n')
        out.flush()
        _FSYNC(out.fileno())


def git(*args):
    return subprocess.check_output(['git', '--no-optional-locks', *args], cwd=REPO,
                                   timeout=45).decode('utf-8').strip()


def relative(path):
    return Path(path).resolve().relative_to(REPO).as_posix()


def inside(path, root=ROOT):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError('path escapes campaign workspace: ' + str(path))
    return path


def tree_bytes(root):
    """Account only ordinary files without following reparse points."""
    total = 0
    if not Path(root).exists():
        return total
    for directory, children, files in os.walk(root, followlinks=False):
        children[:] = [n for n in children if not (Path(directory) / n).is_symlink()
                       and not (getattr((Path(directory) / n).stat(), 'st_file_attributes', 0) & 0x400)]
        for name in files:
            path = Path(directory) / name
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                pass  # A worker may atomically rename its own temporary file.
    return total


def source_snapshot():
    tracked = git('ls-files', '-z').split('\0')
    # Campaign infrastructure is deliberately included even before it is committed.
    prepared = list((REPO / 'tools/final_acceptance').rglob('*.py'))
    prepared += list((REPO / 'tools/final_acceptance').glob('*.md'))
    prepared += list((REPO / 'tools/final_acceptance').glob('*.json'))
    prepared += list((REPO / 'tools/final_acceptance/tests/fixtures').glob('*.json'))
    prepared += [REPO / 'tools/final_acceptance/campaign_config.json', REPO / 'run_final_acceptance.ps1']
    prepared += [LEGACY / n for n in ('bootstrap.py', 'source-runtime.json',
        'readonly-test-environment.json', 'blocked-nodeids.json', 'package.py', 'package-config.json',
        'collected-tests.json', 'complete-static-inventory.json')]
    prepared += [REPO / 'outputs/package-clean-cpu-54e64e9-20261125' / n
                 for n in ('verify_installed.py','clean_fixture.py','requirements.txt')]
    names = {n for n in tracked if n and (REPO / n).is_file()}
    names.update(relative(p) for p in prepared if p.is_file())
    return {n: file_hash(REPO / n) for n in sorted(names)}


def user_snapshot():
    names = git('ls-files', '--others', '--exclude-standard', '-z').split('\0')
    return {n: file_hash(REPO / n) for n in names if n and
            not n.startswith('tools/final_acceptance/') and n != 'run_final_acceptance.ps1'
            and (REPO / n).is_file()}


class Journal:
    """A separate single-writer, fsynced hash chain for every unique attempt."""
    def __init__(self, path, identity):
        self.path = Path(path)
        self.identity = identity
        self.previous = '0' * 64
        self.sequence = 0
        self.failed = False
        self.closed = False
        self.fd = None
        if self.path.exists():
            raise FileExistsError('never append a new attempt to an old journal')
        self.path.parent.mkdir(parents=True,exist_ok=True)

    def emit(self, kind, **fields):
        if self.failed or self.closed:
            raise RuntimeError('JOURNAL_POISONED: no append after a failed write or fsync')
        record = dict(schema_version=2, sequence=self.sequence+1, previous_sha256=self.previous,
                      time_utc=utc(), kind=kind, **self.identity, **fields)
        record['sha256'] = digest(record)
        payload=canonical(record)+b'\n'
        try:
            flags=os.O_WRONLY|getattr(os,'O_BINARY',0)
            flags|=os.O_CREAT|os.O_EXCL if self.sequence==0 else os.O_APPEND
            self.fd=_OPEN_FD(self.path,flags,0o600)
            offset=0
            while offset<len(payload):
                written=_WRITE_FD(self.fd,payload[offset:])
                if not written:raise OSError('short journal write')
                offset+=written
            _FSYNC(self.fd)
        except BaseException:
            self.failed=True
            raise
        finally:
            if self.fd is not None:
                try:_CLOSE_FD(self.fd)
                except BaseException:
                    self.failed=True
                    raise
                finally:self.fd=None
        self.sequence = record['sequence']
        self.previous = record['sha256']
        return record

    def close(self):
        if self.fd is not None:
            _CLOSE_FD(self.fd)
            self.fd=None
        self.closed=True


def read_journal(path):
    records, previous, issues = [], '0' * 64, []
    path = Path(path)
    if not path.exists():
        return records, issues
    with path.open('rb') as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith(b'\n'):
                issues.append(dict(reason='INTERRUPTED_LAST_WRITE', line=number))
                break
            try:
                row = _DECODE(line.decode('utf-8'))
                if not isinstance(row,dict) or type(row.get('sequence')) is not int or not isinstance(row.get('kind'),str):
                    raise ValueError('invalid journal envelope')
                checksum = row.pop('sha256')
                valid = checksum == digest(row) and row['previous_sha256'] == previous
                valid = valid and row['sequence'] == len(records) + 1
                row['sha256'] = checksum
                if not valid or row.get('schema_version') not in (1,2):
                    raise ValueError('broken journal hash chain')
            except (ValueError, KeyError, TypeError,AttributeError) as exc:
                issues.append(dict(reason='EVIDENCE_INVALID', line=number, error=str(exc),
                    verified_prefix_events=len(records),verified_prefix_head=previous))
                break
            records.append(row)
            previous = checksum
    return records, issues
