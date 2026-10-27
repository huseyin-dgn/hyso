"""Synchronous local experiment tracking on a trusted, dedicated local directory.

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


_SCHEMA2 = (
    """CREATE TABLE artifacts (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
        ownership TEXT NOT NULL CHECK(ownership IN ('reference','copy')),
        kind TEXT NOT NULL, producer_run_id TEXT REFERENCES runs(id) ON DELETE RESTRICT,
        original_locator TEXT, managed_relative_path TEXT UNIQUE,
        ingest_state TEXT NOT NULL CHECK(ingest_state IN ('pending','staged','ready','failed')),
        expected_size_bytes INTEGER CHECK(expected_size_bytes>=0),
        expected_sha256 TEXT CHECK(length(expected_sha256)=64 AND expected_sha256 NOT GLOB '*[^0-9a-f]*'),
        created_at_utc TEXT NOT NULL, ready_at_utc TEXT,
        request_id TEXT NOT NULL UNIQUE, request_hash TEXT NOT NULL, metadata_json TEXT NOT NULL,
        writer_session_id TEXT NOT NULL,
        CHECK((ownership='copy') = (managed_relative_path IS NOT NULL)),
        CHECK(ingest_state NOT IN ('staged','ready') OR
              (expected_size_bytes IS NOT NULL AND expected_sha256 IS NOT NULL)),
        CHECK((ingest_state='ready') = (ready_at_utc IS NOT NULL)))""",
    """CREATE TABLE run_artifacts (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE RESTRICT,
        artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE RESTRICT,
        role TEXT NOT NULL CHECK(role IN ('input','output','report')),
        created_at_utc TEXT NOT NULL, UNIQUE(run_id,artifact_id,role))""",
    """CREATE TABLE artifact_checks (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE RESTRICT,
        status TEXT NOT NULL CHECK(status IN ('verified','mismatch','missing','unreadable','unavailable')),
        observed_size_bytes INTEGER CHECK(observed_size_bytes>=0), observed_sha256 TEXT,
        recorded_at_utc TEXT NOT NULL, source TEXT NOT NULL CHECK(source IN ('verify','reconcile','ingest')))""",
    """CREATE TABLE run_attempts (
        run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE RESTRICT,
        retry_of_run_id TEXT UNIQUE REFERENCES runs(id) ON DELETE RESTRICT,
        lineage_id TEXT NOT NULL REFERENCES runs(id) ON DELETE RESTRICT,
        attempt_no INTEGER NOT NULL CHECK(attempt_no>=1),
        resume_artifact_id TEXT REFERENCES artifacts(id) ON DELETE RESTRICT,
        UNIQUE(lineage_id,attempt_no), CHECK(run_id IS NOT retry_of_run_id),
        CHECK((retry_of_run_id IS NULL) = (attempt_no=1)))""",
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
    """Local schema1/schema2 store; handles belong to one thread/process.

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
            allowed = {"experiments.sqlite3", "experiments.sqlite3-journal", ".staging", "artifacts", "backups"}
            for item in self.root_dir.iterdir():
                if item.name not in allowed:
                    _error("root_not_dedicated", "open")
                _safe_path(item)
                # A concurrent writer may remove its rollback journal between
                # directory enumeration and this check.
                if not item.exists() and item.name == "experiments.sqlite3-journal":
                    continue
                if not (item.is_dir() if item.name in (".staging", "artifacts", "backups") else item.is_file()):
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
        self._schema_version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if self._schema_version not in (1, 2):
            _error("unsupported_schema", "open")
        expected = {}
        for statement in _SCHEMA + (_SCHEMA2 if self._schema_version == 2 else ()):
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
            if self._db.execute("PRAGMA user_version").fetchone()[0] != self._schema_version:
                _error("stale_schema", operation)
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
            self._b2()
            if retry_of_run_id is not None:
                _id(retry_of_run_id)
            if resume_artifact_id is not None:
                _id(resume_artifact_id)
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
        payload = [experiment_id, kind, config_json, provenance_json]
        if retry_of_run_id is not None or resume_artifact_id is not None:
            payload += [retry_of_run_id, resume_artifact_id]
        digest = _hash(_json(payload))
        resume_identity = None
        if resume_artifact_id is not None:
            # Exact committed replay remains a read even if bytes later disappear.
            with self._access("resume_replay") as db:
                old = db.execute("SELECT * FROM runs WHERE request_id=?", (request_id,)).fetchone()
                if old is not None:
                    if old["request_hash"] != digest:
                        _error("request_conflict", "create_run", request_id=request_id)
                    return ExperimentRun(self, old["id"])
            with self._access("resume_artifact") as db:
                artifact = self._row(db, "artifacts", resume_artifact_id)
            if artifact["kind"] != "checkpoint" or artifact["ingest_state"] != "ready" or self._observe(artifact)["status"] != "verified":
                _error("invalid_resume_artifact", "create_run")
            resume_identity = (artifact["expected_size_bytes"], artifact["expected_sha256"])
        with self._access("create_run", write=True, entity_id=identity, request_id=request_id) as db:
            old = db.execute("SELECT * FROM runs WHERE request_id=?", (request_id,)).fetchone()
            if old is not None:
                if old["request_hash"] != digest:
                    _error("request_conflict", "create_run", request_id=request_id)
                return ExperimentRun(self, old["id"])
            self._row(db, "experiments", experiment_id)
            lineage, attempt = identity, 1
            if retry_of_run_id is not None:
                parent = self._row(db, "runs", retry_of_run_id)
                if (parent["status"] not in ("failed", "interrupted") or
                        parent["experiment_id"] != experiment_id or parent["kind"] != kind):
                    _error("invalid_retry", "create_run")
                previous = db.execute("SELECT * FROM run_attempts WHERE run_id=?", (retry_of_run_id,)).fetchone()
                lineage, attempt = previous["lineage_id"], previous["attempt_no"] + 1
            if resume_artifact_id is not None:
                artifact = self._row(db, "artifacts", resume_artifact_id)
                if artifact["kind"] != "checkpoint" or artifact["ingest_state"] != "ready":
                    _error("invalid_resume_artifact", "create_run")
                if (artifact["expected_size_bytes"], artifact["expected_sha256"]) != resume_identity:
                    _error("invalid_resume_artifact", "create_run")
            db.execute("""INSERT INTO runs(id,experiment_id,kind,status,config_json,config_hash,
                       provenance_json,provenance_hash,created_at_utc,revision,reason_json,request_id,request_hash)
                       VALUES(?,?,?,'created',?,?,?,?,?,0,'{}',?,?)""",
                       (identity, experiment_id, kind, config_json, config_hash, provenance_json,
                        provenance_hash, _now(), request_id, digest))
            if self._schema_version == 2:
                db.execute("INSERT INTO run_attempts VALUES(?,?,?,?,?)",
                           (identity, retry_of_run_id, lineage, attempt, resume_artifact_id))
                if resume_artifact_id is not None:
                    db.execute("INSERT INTO run_artifacts(run_id,artifact_id,role,created_at_utc) VALUES(?,?,'input',?)",
                               (identity, resume_artifact_id, _now()))
        return ExperimentRun(self, identity)


    def _b2(self):
        if self._schema_version != 2:
            _error("unsupported_b2", "schema2_required")

    @classmethod
    def migrate_store(cls, root_dir):
        """Explicit offline schema1 -> schema2. New stores intentionally use schema1.

        Stop/coordinate writers first. Persisted running records must be explicitly
        recovered, even if their process has exited. Idle old schema1 sessions are
        fenced at their next access. A SQLite reserved lock excludes new writers
        throughout backup/DDL; it is not a process liveness detector.
        Backups remain after any failure; no automatic restore/delete is performed.
        """
        manager = None
        backup = None
        try:
            root = _safe_path(root_dir)
            if not _safe_path(root / "experiments.sqlite3").is_file():
                _error("not_found", "migrate")
            # Determine the persisted policy without changing it.
            probe = sqlite3.connect(root / "experiments.sqlite3")
            try:
                policy = probe.execute("SELECT metric_limit FROM store_meta").fetchone()[0]
            finally:
                probe.close()
            manager = cls(root, max_metrics_per_run=policy)
            with manager._access("migrate", write=True) as db:
                if manager._schema_version == 2:
                    return {"status": "already_migrated", "schema_version": 2, "backup": None}
                manager._offline(db)
                before = manager._core_snapshot(db)
                directory = _safe_path(root / "backups")
                directory.mkdir(exist_ok=True)
                backup = directory / (uuid.uuid4().hex + ".sqlite3")
                with backup.open("xb"):
                    pass
                source = sqlite3.connect(root / "experiments.sqlite3")
                target = sqlite3.connect(backup)
                try:
                    source.backup(target)
                    if target.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                        _error("invalid_backup", "migrate")
                    if (target.execute("PRAGMA user_version").fetchone()[0] != 1 or
                            manager._core_snapshot(target) != before):
                        _error("invalid_backup", "migrate")
                finally:
                    target.close()
                    source.close()
                with backup.open("rb+") as stream:
                    os.fsync(stream.fileno())
                for statement in _SCHEMA2:
                    db.execute(statement)
                db.execute("INSERT INTO run_attempts(run_id,lineage_id,attempt_no) SELECT id,id,1 FROM runs")
                if manager._core_snapshot(db) != before:
                    _error("migration_data_changed", "migrate")
                for row in db.execute("SELECT config_json,config_hash,provenance_json,provenance_hash FROM runs"):
                    if _hash(row[0]) != row[1] or _hash(row[2]) != row[3]:
                        _error("snapshot_hash_mismatch", "migrate")
                if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
                    _error("invalid_foreign_keys", "migrate")
                db.execute("PRAGMA user_version=2")
            manager._schema_version = 2
            manager.close()
            manager = None
            reopened = cls(root, max_metrics_per_run=policy)
            try:
                if reopened._core_snapshot(reopened._db) != before:
                    _error("migration_data_changed", "migrate")
            finally:
                reopened.close()
            return {"status": "migrated", "schema_version": 2, "backup": str(backup)}
        except ExperimentError:
            raise
        except (OSError, sqlite3.Error, ValueError, TypeError):
            raise ExperimentError("migration_failed", "migrate") from None
        finally:
            if manager is not None and manager._db is not None:
                # Do not mask an ambiguous commit or the original failure.
                manager._db.close()
                manager._closed = True

    @staticmethod
    def _core_snapshot(db):
        return {table: [tuple(row) for row in db.execute("SELECT * FROM " + table + " ORDER BY 1")]
                for table in ("store_meta", "experiments", "runs", "run_events", "metric_series", "metric_events")}

    @staticmethod
    def _offline(db):
        if db.execute("SELECT 1 FROM runs WHERE status='running' LIMIT 1").fetchone():
            _error("offline_required", "maintenance")

    def recover_interrupted(self, run_id, expected_revision, reason):
        """Fence tracking writes only; caller must coordinate the old workload.

        Also available on schema1 so an abandoned running record can be explicitly
        recovered before offline migration. Does not kill a process or cancel GPU work.
        """
        run_id = _id(run_id)
        if type(expected_revision) is not int or not 0 <= expected_revision < _MAX_INT:
            _error("invalid_revision", "recover")
        reason_json = _json({"reason": _label(reason, 256)})
        with self._access("recover", write=True, entity_id=run_id) as db:
            row = self._row(db, "runs", run_id)
            if row["revision"] != expected_revision:
                _error("stale_revision", "recover")
            if row["status"] != "running":
                _error("invalid_transition", "recover")
            now, operation_id = _now(), uuid.uuid4().hex
            changed = db.execute("UPDATE runs SET status='interrupted',revision=revision+1,"
                "writer_session_id=NULL,ended_at_utc=?,reason_json=? WHERE id=? AND revision=? AND status='running'",
                (now, reason_json, run_id, expected_revision))
            if changed.rowcount != 1:
                _error("stale_revision", "recover")
            db.execute("INSERT INTO run_events(operation_id,request_hash,run_id,revision,from_status,"
                "to_status,recorded_at_utc,reason_json) VALUES(?,?,?,?,'running','interrupted',?,?)",
                (operation_id, _hash(reason_json), run_id, expected_revision + 1, now, reason_json))
            result = self._run_record(db, self._row(db, "runs", run_id))
        return result

    @staticmethod
    def _artifact_record(row):
        result = dict(row)
        result["artifact_id"] = result["id"]
        result["metadata"] = json.loads(result.pop("metadata_json"))
        result.pop("request_hash")
        result.pop("writer_session_id")
        return result

    def get_artifact(self, artifact_id):
        self._b2()
        with self._access("get_artifact") as db:
            return self._artifact_record(self._row(db, "artifacts", _id(artifact_id)))

    def list_artifacts(self, run_id=None, *, after_seq=None, limit=100):
        self._b2()
        cursor, limit = _page(after_seq, limit, 100)
        if run_id is not None:
            _id(run_id)
        with self._access("list_artifacts") as db:
            rows = db.execute("SELECT a.* FROM artifacts a WHERE a.seq>?" +
                (" AND EXISTS(SELECT 1 FROM run_artifacts l WHERE l.artifact_id=a.id AND l.run_id=?)" if run_id else "") +
                " ORDER BY a.seq LIMIT ?", (cursor, *((run_id,) if run_id else ()), limit)).fetchall()
            return [self._artifact_record(row) for row in rows]

    @staticmethod
    def _role(role):
        if type(role) is not str or role not in ("input", "output", "report"):
            _error("invalid_role", "artifact")

    def _link(self, db, run_id, artifact_id, role):
        old = db.execute("SELECT * FROM run_artifacts WHERE run_id=? AND artifact_id=? AND role=?",
                         (run_id, artifact_id, role)).fetchone()
        if old is not None:
            return dict(old)
        run = self._row(db, "runs", run_id)
        ExperimentRun(self, run_id)._owner(run, "link_artifact")
        if run["status"] not in ("created", "running") or (role == "output" and run["status"] != "running"):
            _error("run_not_running", "link_artifact")
        artifact = self._row(db, "artifacts", artifact_id)
        if artifact["ingest_state"] != "ready":
            _error("artifact_not_ready", "link_artifact")
        db.execute("INSERT INTO run_artifacts(run_id,artifact_id,role,created_at_utc) VALUES(?,?,?,?)",
                   (run_id, artifact_id, role, _now()))
        if run["status"] == "running":
            db.execute("UPDATE runs SET revision=revision+1 WHERE id=?", (run_id,))
        return dict(db.execute("SELECT * FROM run_artifacts WHERE run_id=? AND artifact_id=? AND role=?",
                              (run_id, artifact_id, role)).fetchone())

    def link_artifact(self, run_id, artifact_id, *, role="input"):
        self._b2()
        run_id, artifact_id = _id(run_id), _id(artifact_id)
        self._role(role)
        with self._access("link_artifact", write=True) as db:
            result = self._link(db, run_id, artifact_id, role)
        return result

    def _managed(self, artifact_id, *, staging=False):
        artifact_id = _id(artifact_id)
        relative = Path(".staging") / (artifact_id + ".part") if staging else Path("artifacts") / artifact_id / "payload"
        path = _safe_path(self.root_dir / relative)
        if not path.is_relative_to(self.root_dir):
            _error("unsafe_path", "artifact")
        return path

    @staticmethod
    def _file_path(path):
        # Unlike storage roots, external files may reside in synced folders.
        absolute = Path(os.path.abspath(path))
        for component in (absolute, *absolute.parents):
            info = component.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                _error("unsafe_path", "artifact")
        info = absolute.stat()
        if not stat.S_ISREG(info.st_mode):
            _error("non_regular_file", "artifact")
        return absolute

    @staticmethod
    def _stream(path, destination=None, limit=None):
        """Streaming observation, not an atomic snapshot against hostile mutation."""
        path = ExperimentManager._file_path(path)
        digest, size = hashlib.sha256(), 0
        original = path.lstat()
        with path.open("rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):
                _error("non_regular_file", "artifact")
            if limit is not None and before.st_size > limit:
                _error("artifact_size_limit", "artifact")
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if limit is not None and size > limit:
                    _error("artifact_size_limit", "artifact")
                digest.update(block)
                if destination is not None:
                    if destination.write(block) != len(block):
                        _error("partial_write", "artifact")
            after = os.fstat(source.fileno())
            current = ExperimentManager._file_path(path).stat()
            signature = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            if signature(original) != signature(before) or signature(before) != signature(after) or signature(after) != signature(current) or size != after.st_size:
                _error("source_changed", "artifact")
        return size, digest.hexdigest()

    def _ingest_fence(self, db, artifact_id, state):
        row = self._row(db, "artifacts", artifact_id)
        if row["writer_session_id"] != self.session_id or row["ingest_state"] != state:
            _error("stale_ingest", "artifact")
        run = self._row(db, "runs", row["producer_run_id"] or
                        db.execute("SELECT run_id FROM run_artifacts WHERE artifact_id=? ORDER BY seq LIMIT 1", (artifact_id,)).fetchone()[0])
        ExperimentRun(self, run["id"])._owner(run, "artifact")
        if run["status"] != "running":
            _error("run_not_running", "artifact")
        db.execute("UPDATE runs SET revision=revision+1 WHERE id=?", (run["id"],))
        return row

    def _add_artifact(self, run_id, path, *, kind, ownership, role, request_id,
                      metadata, retain_locator, max_size_bytes):
        self._b2()
        self._role(role)
        kind = _label(kind)
        if type(ownership) is not str or ownership not in ("reference", "copy"):
            _error("invalid_ownership", "artifact")
        if type(retain_locator) is not bool or type(max_size_bytes) is not int or not 1 <= max_size_bytes <= 1024**3:
            _error("invalid_policy", "artifact")
        metadata_json, _ = _snapshot({} if metadata is None else metadata, ())
        try:
            locator = str(Path(os.path.abspath(path)))
        except (TypeError, ValueError, OSError):
            _error("invalid_path", "artifact")
        request_id, identity = _request(request_id), uuid.uuid4().hex
        digest = _hash(_json([run_id, _hash(locator), kind, ownership, role, metadata_json, retain_locator, max_size_bytes]))
        relative = "artifacts/" + identity + "/payload" if ownership == "copy" else None
        with self._access("reserve_artifact", write=True, entity_id=identity, request_id=request_id) as db:
            old = db.execute("SELECT * FROM artifacts WHERE request_id=?", (request_id,)).fetchone()
            if old is not None:
                if old["request_hash"] != digest:
                    _error("request_conflict", "artifact", request_id=request_id)
                return self._artifact_record(old)
            run = self._row(db, "runs", run_id)
            ExperimentRun(self, run_id)._owner(run, "artifact")
            if run["status"] != "running":
                _error("run_not_running", "artifact")
            db.execute("""INSERT INTO artifacts(id,ownership,kind,producer_run_id,original_locator,
                managed_relative_path,ingest_state,created_at_utc,request_id,request_hash,metadata_json,writer_session_id)
                VALUES(?,?,?,?,?,?,'pending',?,?,?,?,?)""",
                (identity, ownership, kind, run_id if role == "output" else None,
                 locator if retain_locator and ownership == "reference" else None, relative, _now(),
                 request_id, digest, metadata_json, self.session_id))
            db.execute("INSERT INTO run_artifacts(run_id,artifact_id,role,created_at_utc) VALUES(?,?,?,?)",
                       (run_id, identity, role, _now()))
            db.execute("UPDATE runs SET revision=revision+1 WHERE id=?", (run_id,))
        try:
            if ownership == "copy":
                staging = self._managed(identity, staging=True)
                staging.parent.mkdir(exist_ok=True)
                with staging.open("xb") as output:
                    size, checksum = self._stream(locator, output, max_size_bytes)
                    output.flush()
                    os.fsync(output.fileno())
                with self._access("stage_artifact", write=True, entity_id=identity) as db:
                    self._ingest_fence(db, identity, "pending")
                    db.execute("UPDATE artifacts SET ingest_state='staged',expected_size_bytes=?,expected_sha256=? WHERE id=?",
                               (size, checksum, identity))
                self._publish(identity)
                if self._stream(self._managed(identity)) != (size, checksum):
                    _error("publication_mismatch", "artifact")
                state = "staged"
            else:
                size, checksum = self._stream(locator)
                state = "pending"
            with self._access("ready_artifact", write=True, entity_id=identity) as db:
                self._ingest_fence(db, identity, state)
                db.execute("UPDATE artifacts SET ingest_state='ready',expected_size_bytes=?,expected_sha256=?,ready_at_utc=? WHERE id=?",
                           (size, checksum, _now(), identity))
                result = self._artifact_record(self._row(db, "artifacts", identity))
            return result
        except (OSError, ExperimentError) as exc:
            if isinstance(exc, ExperimentError) and exc.commit_state == "unknown":
                raise
            # Best-effort observation must never hide a failure or uncertain commit.
            with self._access("fail_artifact", write=True, entity_id=identity) as db:
                row = self._row(db, "artifacts", identity)
                if row["writer_session_id"] == self.session_id and row["ingest_state"] in ("pending", "staged"):
                    db.execute("UPDATE artifacts SET ingest_state='failed' WHERE id=?", (identity,))
                    self._check(db, identity, {"status": "unavailable"}, "ingest")
            code = exc.code if isinstance(exc, ExperimentError) else "artifact_io_error"
            raise ExperimentError(code, "artifact", entity_id=identity, request_id=request_id) from None

    def _publish(self, artifact_id):
        staging, final = self._managed(artifact_id, staging=True), self._managed(artifact_id)
        final.parent.parent.mkdir(exist_ok=True)
        final.parent.mkdir(exist_ok=True)
        # link is atomic no-clobber on supported local FS. No overwrite fallback.
        # Keep the staging link for explicit maintenance; no automatic deletion.
        os.link(staging, final)

    def _observe(self, row, path=None):
        if row["ownership"] == "copy":
            expected = "artifacts/" + _id(row["id"]) + "/payload"
            if row["managed_relative_path"] != expected:
                _error("unsafe_path", "verify_artifact")
            path = self._managed(row["id"])
        elif path is None:
            path = row["original_locator"]
        if path is None or row["expected_sha256"] is None:
            return {"status": "unavailable"}
        try:
            size, checksum = self._stream(path)
            return {"status": "verified" if (size, checksum) == (row["expected_size_bytes"], row["expected_sha256"]) else "mismatch",
                    "observed_size_bytes": size, "observed_sha256": checksum}
        except FileNotFoundError:
            return {"status": "missing"}
        except (OSError, ExperimentError):
            return {"status": "unreadable"}

    @staticmethod
    def _check(db, artifact_id, observation, source):
        cursor = db.execute("INSERT INTO artifact_checks(artifact_id,status,observed_size_bytes,observed_sha256,recorded_at_utc,source) VALUES(?,?,?,?,?,?)",
            (artifact_id, observation["status"], observation.get("observed_size_bytes"), observation.get("observed_sha256"), _now(), source))
        return dict(db.execute("SELECT * FROM artifact_checks WHERE seq=?", (cursor.lastrowid,)).fetchone())

    def verify_artifact(self, artifact_id, *, path=None):
        """Observe bytes only. Redacted references require caller-supplied path.

        This never deserializes a checkpoint or establishes model compatibility.
        Every observation is appended; expected identity remains immutable.
        """
        self._b2()
        artifact_id = _id(artifact_id)
        with self._access("verify_artifact") as db:
            row = self._row(db, "artifacts", artifact_id)
        observation = self._observe(row, path)
        with self._access("verify_artifact", write=True, entity_id=artifact_id) as db:
            result = self._check(db, artifact_id, observation, "verify")
        return result

    def reconcile_storage(self, *, limit=1000):
        """Explicit offline bounded reconciliation; reports unknown files, never deletes.

        One reserved transaction prevents cooperating writers from starting while
        observations/publication are made. This may hold the lock during bounded
        hashing; callers budget disk/time and coordinate maintenance themselves.
        No failed ingest is adopted. Pending bytes without expected identity are
        quarantined by report only. Ready identity is never rewritten.
        """
        self._b2()
        _, limit = _page(None, limit, 1000)
        reports = []
        with self._access("reconcile", write=True) as db:
            self._offline(db)
            rows = db.execute("SELECT * FROM artifacts ORDER BY seq LIMIT ?", (limit + 1,)).fetchall()
            if len(rows) > limit:
                _error("reconcile_limit", "reconcile")
            known = {row["id"] for row in rows if row["ownership"] == "copy"}
            entries = 0
            for directory_name in (".staging", "artifacts"):
                directory = _safe_path(self.root_dir / directory_name)
                if directory.exists():
                    for item in directory.iterdir():
                        entries += 1
                        if entries > limit * 3:
                            _error("reconcile_limit", "reconcile")
                        _safe_path(item)
                        identity = item.name[:-5] if directory_name == ".staging" and item.name.endswith(".part") else item.name
                        if identity not in known:
                            reports.append({"status": "orphan_staging" if directory_name == ".staging" else "orphan_managed", "name": item.name})
                        elif directory_name == "artifacts" and item.is_dir():
                            for child in item.iterdir():
                                entries += 1
                                if entries > limit * 3:
                                    _error("reconcile_limit", "reconcile")
                                _safe_path(child)
                                if child.name != "payload":
                                    reports.append({"status": "unexpected_file", "artifact_id": identity})
            for raw in rows:
                row = dict(raw)
                identity, state = row["id"], row["ingest_state"]
                if state == "staged" and row["ownership"] == "copy":
                    final, staged = self._managed(identity), self._managed(identity, staging=True)
                    observation = None
                    if not final.exists() and staged.exists():
                        try:
                            size, checksum = self._stream(staged)
                            if (size, checksum) == (row["expected_size_bytes"], row["expected_sha256"]):
                                self._publish(identity)
                            else:
                                observation = {"status": "mismatch", "observed_size_bytes": size,
                                               "observed_sha256": checksum}
                        except (OSError, ExperimentError):
                            reports.append({"artifact_id": identity, "status": "publication_failed"})
                    if observation is None:
                        observation = self._observe(row)
                    if observation["status"] == "verified":
                        changed = db.execute("UPDATE artifacts SET ingest_state='ready',ready_at_utc=?,writer_session_id=? WHERE id=? AND ingest_state='staged' AND writer_session_id=?",
                            (_now(), self.session_id, identity, row["writer_session_id"]))
                        if changed.rowcount != 1:
                            _error("stale_ingest", "reconcile")
                    self._check(db, identity, observation, "reconcile")
                    reports.append({"artifact_id": identity, **observation})
                elif state == "ready":
                    observation = self._observe(row)
                    self._check(db, identity, observation, "reconcile")
                    reports.append({"artifact_id": identity, **observation})
                else:
                    report = {"artifact_id": identity, "status": "quarantined_" + state}
                    if row["ownership"] == "copy":
                        report.update(staging_present=self._managed(identity, staging=True).exists(),
                                      final_present=self._managed(identity).exists())
                    reports.append(report)
        return {"status": "reconciled", "observations": reports}

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
        if db.execute("PRAGMA user_version").fetchone()[0] == 2:
            attempt = db.execute("SELECT * FROM run_attempts WHERE run_id=?", (row["id"],)).fetchone()
            record.update(dict(attempt))
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
            if target == "completed" and self._manager._schema_version == 2:
                if db.execute("SELECT 1 FROM run_artifacts l JOIN artifacts a ON a.id=l.artifact_id "
                              "WHERE l.run_id=? AND l.role='output' AND a.ingest_state!='ready'", (self.id,)).fetchone():
                    _error("outputs_not_ready", target)
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

    def add_artifact(self, path, *, kind="checkpoint", ownership="reference", role="output",
                     request_id=None, metadata=None, retain_locator=False,
                     max_size_bytes=1024**3):
        return self._manager._add_artifact(self.id, path, kind=kind, ownership=ownership,
                    role=role, request_id=request_id, metadata=metadata,
                    retain_locator=retain_locator, max_size_bytes=max_size_bytes)

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
