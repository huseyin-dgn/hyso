"""Synchronous schema1 tracking on a trusted, dedicated local directory.

Use a non-synced local disk. Network shares, hostile filesystem races and
physical power-loss guarantees are outside this API's contract. All successful
mutations are committed; closing a session never completes its runs.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import tempfile
import threading
import uuid


class ExperimentError(RuntimeError):
    """Safe domain error. Payloads and underlying storage messages are omitted."""

    def __init__(self, code, operation, *, entity_id=None, request_id=None,
                 event_id=None, retryable=False, commit_state="not_committed"):
        super().__init__(f"{operation}: {code}")
        self.code = code
        self.operation = operation
        self.entity_id = entity_id
        self.request_id = request_id
        self.event_id = event_id
        self.retryable = retryable
        self.commit_state = commit_state


_KINDS = ("training", "evaluation", "benchmark")
_STATES = ("created", "running", "completed", "failed", "interrupted")
_STEPS = ("applied_update", "optimizer_attempt", "microbatch", "observation")
_SECRET = re.compile(r"password|passwd|token|secret|api[_-]?key|auth", re.I)
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*(?:/[A-Za-z][A-Za-z0-9_.-]*)*\Z", re.ASCII)
_MAX_METRICS = 1_000_000
_MAX_INT = 2**63 - 1
_SCHEMA = (
    """CREATE TABLE store_meta (
        singleton INTEGER PRIMARY KEY CHECK(singleton=1), store_id TEXT NOT NULL UNIQUE,
        canonical_json_version INTEGER NOT NULL CHECK(canonical_json_version=1),
        created_at_utc TEXT NOT NULL, metric_limit INTEGER NOT NULL CHECK(metric_limit>0))""",
    """CREATE TABLE experiments (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL, created_at_utc TEXT NOT NULL,
        request_id TEXT NOT NULL UNIQUE, request_hash TEXT NOT NULL)""",
    """CREATE TABLE runs (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
        experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE RESTRICT,
        kind TEXT NOT NULL CHECK(kind IN ('training','evaluation','benchmark')),
        status TEXT NOT NULL CHECK(status IN ('created','running','completed','failed','interrupted')),
        config_json TEXT NOT NULL, config_hash TEXT NOT NULL,
        provenance_json TEXT NOT NULL, provenance_hash TEXT NOT NULL,
        created_at_utc TEXT NOT NULL, started_at_utc TEXT, ended_at_utc TEXT,
        revision INTEGER NOT NULL CHECK(revision>=0), writer_session_id TEXT,
        reason_json TEXT NOT NULL, request_id TEXT NOT NULL UNIQUE, request_hash TEXT NOT NULL,
        CHECK(status!='running' OR (writer_session_id IS NOT NULL AND started_at_utc IS NOT NULL)),
        CHECK(status!='created' OR (revision=0 AND writer_session_id IS NULL AND started_at_utc IS NULL)),
        CHECK(status!='completed' OR (writer_session_id IS NOT NULL AND started_at_utc IS NOT NULL)),
        CHECK((status IN ('completed','failed','interrupted')) = (ended_at_utc IS NOT NULL)))""",
    """CREATE TABLE run_events (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, operation_id TEXT NOT NULL UNIQUE,
        request_hash TEXT NOT NULL, run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE RESTRICT,
        revision INTEGER NOT NULL CHECK(revision>0),
        from_status TEXT NOT NULL CHECK(from_status IN ('created','running')),
        to_status TEXT NOT NULL CHECK(to_status IN ('running','completed','failed','interrupted')),
        recorded_at_utc TEXT NOT NULL, reason_json TEXT NOT NULL, UNIQUE(run_id,revision),
        CHECK((from_status='created' AND to_status IN ('running','failed','interrupted')) OR
              (from_status='running' AND to_status IN ('completed','failed','interrupted'))))""",
    """CREATE TABLE metric_series (
        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE RESTRICT,
        name TEXT NOT NULL CHECK(length(name) BETWEEN 1 AND 128), unit TEXT NOT NULL,
        axis_kind TEXT NOT NULL CHECK(axis_kind IN
            ('epoch','applied_update','optimizer_attempt','microbatch','observation')),
        source TEXT NOT NULL, PRIMARY KEY(run_id,name))""",
    """CREATE TABLE metric_events (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
        run_id TEXT NOT NULL, name TEXT NOT NULL,
        axis_index INTEGER NOT NULL CHECK(axis_index>=0),
        epoch_context INTEGER CHECK(epoch_context>=1), value REAL NOT NULL,
        observed_at_utc TEXT, recorded_at_utc TEXT NOT NULL, semantic_json TEXT NOT NULL,
        FOREIGN KEY(run_id,name) REFERENCES metric_series(run_id,name) ON DELETE RESTRICT,
        UNIQUE(run_id,name,axis_index))""",
    "CREATE INDEX runs_experiment_seq ON runs(experiment_id,seq)",
    "CREATE INDEX metrics_run_seq ON metric_events(run_id,seq)",
)


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _error(code="invalid_input", operation="validate", **kwargs):
    raise ExperimentError(code, operation, **kwargs)


def _label(value, maximum=128):
    if (type(value) is not str or not value or len(value.encode("utf-8", errors="replace")) > maximum
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 or 0xD800 <= ord(char) <= 0xDFFF
                   for char in value)):
        _error()
    return value


def _id(value):
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{32}", value):
        _error("invalid_id")
    return value


def _request(value):
    return uuid.uuid4().hex if value is None else _id(value)


def _plain(value, depth=0):
    if depth > 16:
        _error("snapshot_depth")
    kind = type(value)
    if kind is str:
        try:
            size = len(value.encode("utf-8"))
        except UnicodeError:
            _error("invalid_string")
        if size > 65536:
            _error("snapshot_string_limit")
        return value
    if value is None or kind in (bool, int):
        return value
    if kind is float and math.isfinite(value):
        return value
    if kind is list:
        return [_plain(item, depth + 1) for item in value]
    if kind is dict and all(type(key) is str for key in value):
        return {_plain(key, depth + 1): _plain(item, depth + 1) for key, item in value.items()}
    _error("unsafe_json")


def _json(value):
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False)
    except (ValueError, OverflowError):
        # Respect the interpreter's integer encoding bound without changing its
        # process-global security policy or coercing an integer to a float.
        _error("snapshot_encoding_limit")
    if len(encoded.encode("utf-8")) > 1048576:
        _error("snapshot_byte_limit")
    return encoded


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _snapshot(value, paths):
    if type(value) is not dict:
        _error("snapshot_requires_dict")
    copied = _plain(value)
    _json(copied)  # Apply original size limits before redaction as well.
    if type(paths) not in (tuple, list) or len(paths) > 256:
        _error("invalid_redact_paths")
    for pointer in paths:
        if type(pointer) is not str or not pointer.startswith("/") or len(pointer) > 65536:
            _error("invalid_redact_path")
        if re.search(r"~(?![01])", pointer):
            _error("invalid_redact_path")
        parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]
        node = copied
        for index, part in enumerate(parts):
            if type(node) is dict and part in node:
                key = part
            elif type(node) is list and re.fullmatch(r"0|[1-9][0-9]*", part) and len(part) < 10 and int(part) < len(node):
                key = int(part)
            else:
                _error("invalid_redact_path")
            if index == len(parts) - 1:
                node[key] = {"value": None, "status": "redacted", "reason": "caller_redaction"}
            else:
                node = node[key]

    def secrets(node):
        if type(node) is dict:
            for key, child in node.items():
                if _SECRET.search(key) and child != {"value": None, "status": "redacted", "reason": "caller_redaction"}:
                    _error("sensitive_key")
                secrets(child)
        elif type(node) is list:
            for child in node:
                secrets(child)

    secrets(copied)
    encoded = _json(_plain(copied))
    return encoded, _hash(encoded)


def _page(after_seq, limit, default):
    if limit is None:
        limit = default
    if type(limit) is not int or not 1 <= limit <= 1000:
        _error("invalid_limit")
    if after_seq is None:
        after_seq = 0
    if type(after_seq) is not int or not 0 <= after_seq <= _MAX_INT:
        _error("invalid_cursor")
    return after_seq, limit


def _safe_path(path):
    # lstat every existing component, before resolve could hide a reparse point.
    absolute = Path(os.path.abspath(path))
    if str(absolute).startswith(("\\\\", "//")):
        _error("unsupported_filesystem", "open")
    for component in (absolute, *absolute.parents):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            _error("unsafe_path", "open")
    for key in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        synced = os.environ.get(key)
        if synced and absolute.is_relative_to(Path(synced).resolve()):
            _error("synced_root", "open")
    if os.name == "nt":
        import ctypes
        if ctypes.windll.kernel32.GetDriveTypeW(str(absolute.anchor)) == 4:
            _error("unsupported_filesystem", "open")
    return absolute


def _settings(connection):
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    connection.execute("PRAGMA synchronous=FULL")


class ExperimentManager:
    """Local schema1 store; each manager and its handles belong to one thread/process.

    request_id/event_id are lowercase UUID hex strings. redact_paths accepts a
    sequence of config JSON Pointers, or a dict with config/provenance Pointer
    lists. max_metrics_per_run is an
    explicit persisted policy: reopening must specify the same nondefault value.
    Root ACLs and absence of synchronization software are the caller's duty;
    known OneDrive roots, network drives and reparse points are rejected.
    """

    def __init__(self, root_dir, *, max_metrics_per_run=_MAX_METRICS):
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._lock = threading.Lock()
        self._closed = True
        self._db = None
        self.session_id = uuid.uuid4().hex
        if type(max_metrics_per_run) is not int or not 1 <= max_metrics_per_run <= _MAX_INT:
            _error("invalid_policy", "open")
        self._metric_limit = max_metrics_per_run
        try:
            self.root_dir = _safe_path(root_dir)
            self.root_dir.mkdir(parents=True, exist_ok=True)
            allowed = {"experiments.sqlite3", "experiments.sqlite3-journal"}
            for item in self.root_dir.iterdir():
                if item.name not in allowed:
                    _error("root_not_dedicated", "open")
                _safe_path(item)
                # A concurrent writer may remove its rollback journal between
                # directory enumeration and this check.
                if not item.exists() and item.name == "experiments.sqlite3-journal":
                    continue
                if not item.is_file():
                    _error("unsafe_path", "open")
            path = self.root_dir / "experiments.sqlite3"
            _safe_path(path)
            if not path.exists():
                if (self.root_dir / "experiments.sqlite3-journal").exists() and not path.exists():
                    _error("unrecognized_store", "open")
                self._initialize(path)
            self._db = sqlite3.connect(path, isolation_level=None, timeout=5)
            _settings(self._db)
            self._verify()
            self._closed = False
        except ExperimentError:
            if self._db is not None:
                self._db.close()
            raise
        except (OSError, sqlite3.Error, ValueError, TypeError):
            if self._db is not None:
                self._db.close()
            raise ExperimentError("invalid_store", "open") from None

    def _initialize(self, destination):
        # Publish a complete closed database without replacing any existing file.
        # Concurrent first-open sessions can only observe a fully initialized DB.
        fd, name = tempfile.mkstemp(prefix=".hyso-init-", dir=self.root_dir.parent)
        os.close(fd)
        temporary = Path(name)
        connection = None
        try:
            connection = sqlite3.connect(temporary, isolation_level=None)
            _settings(connection)
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("BEGIN IMMEDIATE")
            for statement in _SCHEMA:
                connection.execute(statement)
            connection.execute("INSERT INTO store_meta VALUES(1,?,1,?,?)",
                               (uuid.uuid4().hex, _now(), self._metric_limit))
            connection.execute("PRAGMA user_version=1")
            connection.commit()
            connection.close()
            connection = None
            with temporary.open("rb+") as stream:
                os.fsync(stream.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
        finally:
            if connection is not None:
                connection.close()
            temporary.unlink()

    def _verify(self):
        if self._db.execute("PRAGMA user_version").fetchone()[0] != 1:
            _error("unsupported_schema", "open")
        expected = {}
        for statement in _SCHEMA:
            name = statement.split()[2]
            expected[name] = statement
        actual = {row["name"]: row["sql"] for row in self._db.execute(
            "SELECT name,sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'")}
        if actual != expected:
            _error("invalid_schema", "open")
        rows = self._db.execute("SELECT * FROM store_meta").fetchall()
        if len(rows) != 1 or rows[0]["canonical_json_version"] != 1:
            _error("invalid_metadata", "open")
        _id(rows[0]["store_id"])
        if rows[0]["metric_limit"] != self._metric_limit:
            _error("policy_conflict", "open")
        if self._db.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
            _error("unsupported_journal", "open")
        if self._db.execute("PRAGMA quick_check").fetchall()[0][0] != "ok":
            _error("corrupt_store", "open")
        if self._db.execute("PRAGMA foreign_key_check").fetchone() is not None:
            _error("invalid_foreign_keys", "open")

    @contextmanager
    def _access(self, operation, *, write=False, entity_id=None, request_id=None, event_id=None):
        details = dict(entity_id=entity_id, request_id=request_id, event_id=event_id)
        if os.getpid() != self._pid or threading.get_ident() != self._thread:
            _error("session_context", operation, **details)
        if self._closed:
            _error("closed_session", operation, **details)
        if not self._lock.acquire(blocking=False):
            _error("reentrant_session", operation, **details)
        committing = False
        try:
            self._db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield self._db
            if write:
                committing = True
                self._commit()
            else:
                self._db.rollback()
        except BaseException as exc:
            try:
                self._db.rollback()
            except sqlite3.Error:
                pass
            if committing:
                raise ExperimentError("commit_unknown", operation, retryable=True,
                                      commit_state="unknown", **details) from None
            if isinstance(exc, ExperimentError):
                raise
            if isinstance(exc, sqlite3.Error):
                busy = isinstance(exc, sqlite3.OperationalError) and any(
                    word in str(exc).lower() for word in ("locked", "busy"))
                code = "storage_busy" if busy else (
                    "integrity_conflict" if isinstance(exc, sqlite3.IntegrityError) else "storage_error")
                raise ExperimentError(code, operation, retryable=busy, **details) from None
            raise
        finally:
            self._lock.release()

    def _commit(self):
        """A separate boundary permits real commit/ACK failure injection."""
        self._db.commit()

    def close(self):
        if os.getpid() != self._pid or threading.get_ident() != self._thread:
            _error("session_context", "close")
        if self._closed:
            return
        with self._access("close"):
            pass
        self._db.close()
        self._closed = True

    @staticmethod
    def _row(db, table, identity):
        # table is internal, never caller input.
        row = db.execute(f"SELECT * FROM {table} WHERE id=?", (identity,)).fetchone()
        if row is None:
            _error("not_found", "read", entity_id=identity)
        return dict(row)

    @staticmethod
    def _experiment(row):
        return {key: row[key] for key in ("id", "seq", "name", "created_at_utc", "request_id")}

    def create_experiment(self, name, *, request_id=None):
        name = _label(name, 256)
        request_id, identity = _request(request_id), uuid.uuid4().hex
        digest = _hash(_json({"name": name}))
        with self._access("create_experiment", write=True, entity_id=identity, request_id=request_id) as db:
            old = db.execute("SELECT * FROM experiments WHERE request_id=?", (request_id,)).fetchone()
            if old is not None:
                if old["request_hash"] != digest:
                    _error("request_conflict", "create_experiment", request_id=request_id)
                return self._experiment(old)
            db.execute("INSERT INTO experiments(id,name,created_at_utc,request_id,request_hash) VALUES(?,?,?,?,?)",
                       (identity, name, _now(), request_id, digest))
            result = self._experiment(self._row(db, "experiments", identity))
        return result

    def get_experiment(self, id):
        identity = _id(id)
        with self._access("get_experiment") as db:
            return self._experiment(self._row(db, "experiments", identity))

    def create_run(self, experiment_id, kind, config, provenance=None, *, redact_paths=(),
                   retry_of_run_id=None, resume_artifact_id=None, request_id=None):
        if retry_of_run_id is not None or resume_artifact_id is not None:
            _error("unsupported_b2", "create_run")
        experiment_id = _id(experiment_id)
        if type(kind) is not str or kind not in _KINDS:
            _error("invalid_kind", "create_run")
        if type(redact_paths) in (tuple, list):
            redact_paths = {"config": redact_paths}
        if type(redact_paths) is not dict or set(redact_paths) - {"config", "provenance"}:
            _error("invalid_redact_paths", "create_run")
        config_json, config_hash = _snapshot(config, redact_paths.get("config", ()))
        provenance_json, provenance_hash = _snapshot({} if provenance is None else provenance,
                                                     redact_paths.get("provenance", ()))
        request_id, identity = _request(request_id), uuid.uuid4().hex
        digest = _hash(_json([experiment_id, kind, config_json, provenance_json]))
        with self._access("create_run", write=True, entity_id=identity, request_id=request_id) as db:
            old = db.execute("SELECT * FROM runs WHERE request_id=?", (request_id,)).fetchone()
            if old is not None:
                if old["request_hash"] != digest:
                    _error("request_conflict", "create_run", request_id=request_id)
                return ExperimentRun(self, old["id"])
            self._row(db, "experiments", experiment_id)
            db.execute("""INSERT INTO runs(id,experiment_id,kind,status,config_json,config_hash,
                       provenance_json,provenance_hash,created_at_utc,revision,reason_json,request_id,request_hash)
                       VALUES(?,?,?,'created',?,?,?,?,?,0,'{}',?,?)""",
                       (identity, experiment_id, kind, config_json, config_hash, provenance_json,
                        provenance_hash, _now(), request_id, digest))
        return ExperimentRun(self, identity)

    def get_run(self, id):
        identity = _id(id)
        with self._access("get_run") as db:
            self._row(db, "runs", identity)
        return ExperimentRun(self, identity)

    @staticmethod
    def _run_record(db, row):
        record = {key: value for key, value in row.items()
                  if key not in ("config_json", "provenance_json", "reason_json", "request_hash")}
        for key in ("config", "provenance", "reason"):
            record[key] = json.loads(row[key + "_json"])
        record.update(format="hyso.experiment", schema_version=1)
        record["metric_count"] = db.execute("SELECT count(*) FROM metric_events WHERE run_id=?", (row["id"],)).fetchone()[0]
        record["writer_liveness"] = "unknown" if row["status"] == "running" else "not_applicable"
        return record

    def list_runs(self, experiment_id=None, *, kind=None, status=None, after_seq=None, limit=100):
        after_seq, limit = _page(after_seq, limit, 100)
        clauses, values = ["seq> ?"], [after_seq]
        for key, value, options in (("experiment_id", experiment_id, None), ("kind", kind, _KINDS), ("status", status, _STATES)):
            if value is not None:
                if options is None:
                    _id(value)
                elif type(value) is not str or value not in options:
                    _error("invalid_filter")
                clauses.append(key + "=?")
                values.append(value)
        with self._access("list_runs") as db:
            rows = db.execute("SELECT * FROM runs WHERE " + " AND ".join(clauses) + " ORDER BY seq LIMIT ?", (*values, limit)).fetchall()
            return [self._run_record(db, dict(row)) for row in rows]


class ExperimentRun:
    """Manager-created runtime handle; all reads reflect persisted records."""

    def __init__(self, manager, identity):
        self._manager = manager
        self._id = identity

    @property
    def id(self):
        return self._id

    def to_dict(self):
        with self._manager._access("get_run", entity_id=self.id) as db:
            return self._manager._run_record(db, self._manager._row(db, "runs", self.id))

    def _owner(self, row, operation):
        if row["status"] == "running" and row["writer_session_id"] != self._manager.session_id:
            _error("foreign_writer", operation, entity_id=self.id)

    def _transition(self, target, reason, request_id):
        reason_json = _json(_plain(reason))
        request_id = _request(request_id)
        digest = _hash(_json([self.id, target, reason_json]))
        with self._manager._access(target, write=True, entity_id=self.id, request_id=request_id) as db:
            old = db.execute("SELECT * FROM run_events WHERE operation_id=?", (request_id,)).fetchone()
            if old is not None:
                if old["request_hash"] != digest:
                    _error("request_conflict", target, request_id=request_id)
                return json.loads(old["reason_json"])
            row = self._manager._row(db, "runs", self.id)
            self._owner(row, target)
            allowed = {"created": ("running", "failed", "interrupted"),
                       "running": ("completed", "failed", "interrupted")}
            if target not in allowed.get(row["status"], ()):
                _error("invalid_transition", target, entity_id=self.id)
            now, revision = _now(), row["revision"] + 1
            started = now if target == "running" else row["started_at_utc"]
            ended = None if target == "running" else now
            owner = self._manager.session_id if target == "running" else row["writer_session_id"]
            changed = db.execute("""UPDATE runs SET status=?,revision=?,started_at_utc=?,ended_at_utc=?,
                writer_session_id=?,reason_json=? WHERE id=? AND revision=? AND status=?""",
                (target, revision, started, ended, owner, reason_json, self.id, row["revision"], row["status"]))
            if changed.rowcount != 1:
                _error("stale_revision", target, entity_id=self.id)
            db.execute("""INSERT INTO run_events(operation_id,request_hash,run_id,revision,from_status,
                       to_status,recorded_at_utc,reason_json) VALUES(?,?,?,?,?,?,?,?)""",
                       (request_id, digest, self.id, revision, row["status"], target, now, reason_json))
        return json.loads(reason_json)

    def start(self, *, request_id=None):
        return self._transition("running", {}, request_id)

    def finish(self, reason="completed", *, request_id=None):
        return self._transition("completed", {"reason": _label(reason, 256)}, request_id)

    def fail(self, code, *, failure_domain="workload", workload_outcome="failed", request_id=None):
        if failure_domain not in ("workload", "tracking", "artifact") or workload_outcome not in ("failed", "succeeded", "unknown"):
            _error("invalid_failure")
        return self._transition("failed", dict(code=_label(code), failure_domain=failure_domain,
                                               workload_outcome=workload_outcome), request_id)

    def interrupt(self, reason="interrupted", *, request_id=None):
        return self._transition("interrupted", {"reason": _label(reason, 256)}, request_id)

    @staticmethod
    def _metric(record):
        if type(record) is not dict or set(record) - {
                "name", "value", "unit", "source", "epoch", "step", "step_kind", "event_id", "observed_at_utc"}:
            _error("invalid_metric")
        if not {"name", "value", "unit", "source"} <= set(record):
            _error("invalid_metric")
        name = _label(record["name"])
        if not _NAME.fullmatch(name):
            _error("invalid_metric_name")
        unit, source = _label(record["unit"]), _label(record["source"])
        value = record["value"]
        if (type(value) not in (int, float) or
                (type(value) is int and abs(value) > 2**53) or not math.isfinite(value)):
            _error("invalid_metric_value")
        epoch, step, step_kind = (record.get(key) for key in ("epoch", "step", "step_kind"))
        if epoch is not None and (type(epoch) is not int or not 1 <= epoch <= _MAX_INT):
            _error("invalid_axis")
        if step is None:
            if epoch is None or step_kind is not None:
                _error("invalid_axis")
            axis_kind, axis_index, context = "epoch", epoch, None
        else:
            if type(step) is not int or not 0 <= step <= _MAX_INT or type(step_kind) is not str or step_kind not in _STEPS:
                _error("invalid_axis")
            axis_kind, axis_index, context = step_kind, step, epoch
        observed = record.get("observed_at_utc")
        if observed is not None:
            _label(observed, 64)
            try:
                parsed = datetime.fromisoformat(observed.replace("Z", "+00:00"))
                if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
                    _error("invalid_observed_time")
                observed = parsed.isoformat().replace("+00:00", "Z")
            except ValueError:
                _error("invalid_observed_time")
        event = record.get("event_id")
        result = dict(name=name, value=float(value), unit=unit, source=source, axis_kind=axis_kind,
                      axis_index=axis_index, epoch_context=context, observed_at_utc=observed)
        return result, _json(result), None if event is None else _id(event)

    def log_metric(self, *, name, value, unit, source, epoch=None, step=None,
                   step_kind=None, event_id=None, observed_at_utc=None):
        return self.log_metrics([dict(name=name, value=value, unit=unit, source=source,
                                     epoch=epoch, step=step, step_kind=step_kind,
                                     event_id=event_id, observed_at_utc=observed_at_utc)])[0]

    @staticmethod
    def _metric_record(row):
        result = json.loads(row["semantic_json"])
        result.update({key: row[key] for key in ("seq", "event_id", "run_id", "recorded_at_utc")})
        return result

    def log_metrics(self, records):
        if type(records) not in (list, tuple) or not 1 <= len(records) <= 256:
            _error("invalid_batch")
        prepared = [self._metric(record) for record in records]
        event_ids = [event or uuid.uuid4().hex for _, _, event in prepared]
        output = []
        with self._manager._access("log_metrics", write=True, entity_id=self.id,
                                   event_id=event_ids[0]) as db:
            row = self._manager._row(db, "runs", self.id)
            count = db.execute("SELECT count(*) FROM metric_events WHERE run_id=?", (self.id,)).fetchone()[0]
            inserted = 0
            for (metric, semantic, explicit), event_id in zip(prepared, event_ids):
                by_id = db.execute("SELECT * FROM metric_events WHERE event_id=?", (event_id,)).fetchone()
                old = db.execute("SELECT * FROM metric_events WHERE run_id=? AND name=? AND axis_index=?",
                                 (self.id, metric["name"], metric["axis_index"])).fetchone()
                if by_id is not None and (by_id["run_id"] != self.id or by_id["semantic_json"] != semantic):
                    _error("event_conflict", "log_metrics", event_id=event_id)
                if old is not None:
                    if old["semantic_json"] != semantic or (explicit is not None and old["event_id"] != explicit):
                        _error("metric_conflict", "log_metrics", event_id=event_id)
                    output.append(self._metric_record(old))
                    continue
                self._owner(row, "log_metrics")
                if row["status"] != "running":
                    _error("run_not_running", "log_metrics", entity_id=self.id)
                if count + inserted >= self._manager._metric_limit:
                    _error("metric_limit", "log_metrics", entity_id=self.id)
                series = db.execute("SELECT * FROM metric_series WHERE run_id=? AND name=?", (self.id, metric["name"])).fetchone()
                if series is not None:
                    if any(series[key] != metric[key] for key in ("unit", "source", "axis_kind")):
                        _error("series_conflict", "log_metrics")
                    maximum = db.execute("SELECT max(axis_index) FROM metric_events WHERE run_id=? AND name=?", (self.id, metric["name"])).fetchone()[0]
                    if maximum is not None and metric["axis_index"] <= maximum:
                        _error("out_of_order", "log_metrics")
                else:
                    db.execute("INSERT INTO metric_series VALUES(?,?,?,?,?)",
                               (self.id, metric["name"], metric["unit"], metric["axis_kind"], metric["source"]))
                db.execute("""INSERT INTO metric_events(event_id,run_id,name,axis_index,epoch_context,
                           value,observed_at_utc,recorded_at_utc,semantic_json) VALUES(?,?,?,?,?,?,?,?,?)""",
                           (event_id, self.id, metric["name"], metric["axis_index"], metric["epoch_context"],
                            metric["value"], metric["observed_at_utc"], _now(), semantic))
                saved = db.execute("SELECT * FROM metric_events WHERE event_id=?", (event_id,)).fetchone()
                output.append(self._metric_record(saved))
                inserted += 1
            if inserted:
                changed = db.execute("UPDATE runs SET revision=revision+1 WHERE id=? AND revision=? AND writer_session_id=? AND status='running'",
                                     (self.id, row["revision"], self._manager.session_id))
                if changed.rowcount != 1:
                    _error("stale_revision", "log_metrics")
        return output

    def read_metrics(self, name=None, *, after_seq=None, limit=1000):
        after_seq, limit = _page(after_seq, limit, 1000)
        if name is not None:
            _label(name)
            if not _NAME.fullmatch(name):
                _error("invalid_metric_name")
        with self._manager._access("read_metrics", entity_id=self.id) as db:
            rows = db.execute("SELECT * FROM metric_events WHERE run_id=? AND seq>?" +
                              (" AND name=?" if name is not None else "") + " ORDER BY seq LIMIT ?",
                              (self.id, after_seq, *((name,) if name is not None else ()), limit)).fetchall()
            return [self._metric_record(row) for row in rows]
