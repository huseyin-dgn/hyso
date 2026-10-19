"""Real SQLite faults, independent processes, locks and commit-ACK ambiguity."""
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import uuid

import pytest

from hyso.experiments import ExperimentError, ExperimentManager
import hyso.experiments.manager as core


CORE_PATH = Path(core.__file__).resolve()
BOOT = """
import importlib.util,json,os,pathlib,sqlite3,sys
spec=importlib.util.spec_from_file_location('standalone_core',sys.argv[1])
core=importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
ExperimentManager=core.ExperimentManager
ExperimentError=core.ExperimentError
root=pathlib.Path(sys.argv[2])
"""


def launch(script, root, *args):
    return subprocess.Popen([sys.executable, "-I", "-B", "-u", "-c", BOOT + script,
                             str(CORE_PATH), str(root), *args], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def finish(process, stdin=""):
    out, err = process.communicate(stdin, timeout=30)
    assert process.returncode == 0, (out, err)
    return out


def new_run(manager):
    return manager.create_run(manager.create_experiment("storage")["id"], "training", {})


def test_schema_settings_constraints_and_no_extra_tables(tmp_path):
    manager = ExperimentManager(tmp_path / "store")
    db = manager._db
    assert db.execute("PRAGMA user_version").fetchone()[0] == 1
    for pragma, expected in (("foreign_keys", 1), ("journal_mode", "delete"), ("synchronous", 2), ("busy_timeout", 5000)):
        assert db.execute("PRAGMA " + pragma).fetchone()[0] == expected
    names = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert names == {"store_meta", "experiments", "runs", "run_events", "metric_series", "metric_events", "sqlite_sequence"}
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO metric_series VALUES(?,?,?,?,?)", (uuid.uuid4().hex, "loss", "loss", "epoch", "caller"))
    run = new_run(manager)
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE runs SET status='invented' WHERE id=?", (run.id,))
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("DELETE FROM experiments")
    manager.close()


@pytest.mark.parametrize("damage", ["corrupt", "empty", "version", "metadata", "constraint", "foreign_key"])
def test_fail_closed_never_replaces_db(tmp_path, damage):
    root = tmp_path / "store"
    manager = ExperimentManager(root)
    run = new_run(manager)
    manager.close()
    path = root / "experiments.sqlite3"
    if damage in ("corrupt", "empty"):
        path.write_bytes(b"not a SQLite database" if damage == "corrupt" else b"")
    else:
        with sqlite3.connect(path) as db:
            if damage == "version":
                db.execute("PRAGMA user_version=2")
            elif damage == "metadata":
                db.execute("DELETE FROM store_meta")
            elif damage == "constraint":
                db.execute("DROP INDEX runs_experiment_seq")
            else:
                db.execute("UPDATE runs SET experiment_id=? WHERE id=?", (uuid.uuid4().hex, run.id))
    original = path.read_bytes()
    with pytest.raises(ExperimentError):
        ExperimentManager(root)
    assert path.read_bytes() == original


def test_non_dedicated_root_and_paths(tmp_path, monkeypatch):
    root = tmp_path / "occupied"
    root.mkdir()
    sentinel = root / "user.txt"
    sentinel.write_bytes(b"preserve")
    with pytest.raises(ExperimentError, match="root_not_dedicated"):
        ExperimentManager(root)
    assert sentinel.read_bytes() == b"preserve"
    assert not (root / "experiments.sqlite3").exists()
    monkeypatch.setenv("OneDrive", str(tmp_path / "synced"))
    with pytest.raises(ExperimentError, match="synced_root"):
        ExperimentManager(tmp_path / "synced" / "store")
    with pytest.raises(ExperimentError, match="unsupported_filesystem"):
        ExperimentManager("\\\\server\\share\\store")
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    (orphan / "experiments.sqlite3-journal").write_bytes(b"preserve")
    with pytest.raises(ExperimentError, match="unrecognized_store"):
        ExperimentManager(orphan)
    assert not (orphan / "experiments.sqlite3").exists()


def test_windows_junction_rejection(tmp_path):
    # Junctions do not need Windows Developer Mode or symlink privileges.
    if os.name == "nt":
        target, link = tmp_path / "target", tmp_path / "junction"
        target.mkdir()
        made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
        assert made.returncode == 0, made.stdout + made.stderr
    else:
        target, link = tmp_path / "target", tmp_path / "junction"
        target.mkdir()
        link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ExperimentError, match="unsafe_path"):
        ExperimentManager(link / "store")
    assert list(target.iterdir()) == []


def test_uuid_collision_is_domain_error(tmp_path, monkeypatch):
    manager = ExperimentManager(tmp_path / "store")
    first = manager.create_experiment("first")
    collision = uuid.UUID(first["id"])
    request = uuid.uuid4().hex
    monkeypatch.setattr(core.uuid, "uuid4", lambda: collision)
    with pytest.raises(ExperimentError) as caught:
        manager.create_experiment("second", request_id=request)
    assert caught.value.code == "integrity_conflict"
    assert caught.value.entity_id == first["id"] and caught.value.commit_state == "not_committed"
    manager.close()
    reopened = ExperimentManager(tmp_path / "store")
    assert reopened.get_experiment(first["id"])["name"] == "first"
    assert reopened._db.execute("SELECT count(*) FROM experiments").fetchone()[0] == 1
    reopened.close()


@pytest.mark.parametrize("operation", ["create", "transition", "batch"])
def test_sqlite_trigger_fault_rolls_back_and_reopens(tmp_path, operation):
    manager = ExperimentManager(tmp_path / "store")
    run = new_run(manager)
    run.start()
    snapshot = run.to_dict()
    db = manager._db
    if operation == "create":
        trigger = "CREATE TEMP TRIGGER fail_insert AFTER INSERT ON experiments BEGIN SELECT RAISE(ABORT,'injected'); END"
        action = lambda: manager.create_experiment("not committed")
    elif operation == "transition":
        trigger = "CREATE TEMP TRIGGER fail_insert BEFORE INSERT ON run_events BEGIN SELECT RAISE(ABORT,'injected'); END"
        action = run.finish
    else:
        trigger = "CREATE TEMP TRIGGER fail_insert BEFORE INSERT ON metric_events WHEN NEW.axis_index=2 BEGIN SELECT RAISE(ABORT,'injected'); END"
        action = lambda: run.log_metrics([dict(name="loss", value=.4, unit="loss", source="caller", epoch=i) for i in (1, 2)])
    db.execute(trigger)
    with pytest.raises(ExperimentError, match="integrity_conflict"):
        action()
    assert run.to_dict() == snapshot and run.read_metrics() == []
    assert db.execute("SELECT count(*) FROM metric_series").fetchone()[0] == 0
    manager.close()
    reopened = ExperimentManager(tmp_path / "store")
    assert reopened.get_run(run.id).to_dict() == snapshot
    assert reopened._db.execute("SELECT count(*) FROM experiments").fetchone()[0] == 1
    reopened.close()


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("operation", ["experiment", "run", "transition", "metric"])
def test_commit_ack_unknown_and_exact_retry(tmp_path, monkeypatch, committed, operation):
    manager = ExperimentManager(tmp_path / "store")
    run = new_run(manager)
    run.start()
    request = uuid.uuid4().hex
    if operation == "experiment":
        action = lambda: manager.create_experiment("ack", request_id=request)
    elif operation == "run":
        action = lambda: manager.create_run(run.to_dict()["experiment_id"], "training", {"ack": True}, request_id=request)
    elif operation == "transition":
        action = lambda: run.finish(request_id=request)
    else:
        action = lambda: run.log_metric(name="loss", value=.4, unit="loss", source="caller", epoch=1, event_id=request)
    original = manager._commit
    def fault():
        if committed:
            original()
        raise OSError("disk/ACK injected")
    monkeypatch.setattr(manager, "_commit", fault)
    with pytest.raises(ExperimentError) as caught:
        action()
    error = caught.value
    assert error.code == "commit_unknown" and error.commit_state == "unknown" and error.retryable
    assert error.entity_id is not None
    assert (error.event_id if operation == "metric" else error.request_id) == request
    # Reopen independently before retry; no completion/recovery mutation occurs.
    other = ExperimentManager(manager.root_dir)
    assert other.get_run(run.id).to_dict()["status"] == ("completed" if committed and operation == "transition" else "running")
    assert other.get_run(run.id).to_dict()["metric_count"] == int(committed and operation == "metric")
    other.close()
    monkeypatch.setattr(manager, "_commit", original)
    first = action()
    second = action()
    if operation == "run":
        assert first.id == second.id
    else:
        assert first == second
    assert manager._db.execute("SELECT count(*) FROM experiments").fetchone()[0] == (2 if operation == "experiment" else 1)
    assert manager._db.execute("SELECT count(*) FROM runs").fetchone()[0] == (2 if operation == "run" else 1)
    manager.close()


def test_concurrent_first_open_and_create(tmp_path):
    root = tmp_path / "concurrent"
    script = """
print('ready',flush=True)
sys.stdin.readline()
m=ExperimentManager(root)
e=m.create_experiment('shared name')
r=m.create_run(e['id'],'training',{})
print(json.dumps({'experiment':e['id'],'run':r.id}),flush=True)
m.close()
"""
    workers = [launch(script, root) for _ in range(4)]
    try:
        for process in workers:
            assert process.stdout.readline().strip() == "ready"
        for process in workers:
            process.stdin.write("go\n")
            process.stdin.flush()
        records = [json.loads(finish(process).strip()) for process in workers]
    finally:
        for process in workers:
            if process.poll() is None:
                process.kill()
                process.communicate()
    manager = ExperimentManager(root)
    assert len(manager.list_runs()) == 4
    assert len({row['run'] for row in records}) == len({row['experiment'] for row in records}) == 4
    assert manager._db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    manager.close()


def test_concurrent_same_request_one_row(tmp_path):
    root = tmp_path / "requests"
    manager = ExperimentManager(root)
    experiment = manager.create_experiment("shared")
    request = uuid.uuid4().hex
    script = """
m=ExperimentManager(root)
print('ready',flush=True)
sys.stdin.readline()
r=m.create_run(sys.argv[3],'training',{'same':True},request_id=sys.argv[4])
print(r.id,flush=True)
m.close()
"""
    workers = [launch(script, root, experiment["id"], request) for _ in range(3)]
    try:
        for process in workers:
            assert process.stdout.readline().strip() == "ready"
        for process in workers:
            process.stdin.write("go\n")
            process.stdin.flush()
        ids = [finish(process).strip() for process in workers]
    finally:
        for process in workers:
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert len(set(ids)) == 1 and len(manager.list_runs()) == 1
    manager.close()


def test_lock_contention_real_timeout(tmp_path):
    root = tmp_path / "locks"
    manager = ExperimentManager(root)
    script = """
c=sqlite3.connect(root/'experiments.sqlite3',isolation_level=None)
c.execute('BEGIN IMMEDIATE')
print('locked',flush=True)
sys.stdin.readline()
c.rollback()
c.close()
"""
    process = launch(script, root)
    try:
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(ExperimentError) as caught:
            manager.create_experiment("timeout")
        assert caught.value.code == "storage_busy" and caught.value.retryable
        assert caught.value.commit_state == "not_committed"
        finish(process, "release\n")
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert manager._db.execute("SELECT count(*) FROM experiments").fetchone()[0] == 0
    assert manager.create_experiment("after release")['name'] == 'after release'
    manager.close()


@pytest.mark.parametrize("commit", [False, True])
def test_process_death_transaction_boundary(tmp_path, commit):
    root = tmp_path / "killed"
    manager = ExperimentManager(root)
    run = new_run(manager)
    run.start()
    manager.close()
    request = uuid.uuid4().hex
    script = """
m=ExperimentManager(root)
original=m._commit
def boundary():
    if sys.argv[4]=='yes': original()
    print('boundary',flush=True)
    sys.stdin.readline()
m._commit=boundary
m.create_experiment('killed',request_id=sys.argv[3])
"""
    process = launch(script, root, request, "yes" if commit else "no")
    try:
        assert process.stdout.readline().strip() == "boundary"
        process.kill()
        process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    reopened = ExperimentManager(root)
    assert reopened._db.execute("SELECT count(*) FROM experiments WHERE request_id=?", (request,)).fetchone()[0] == int(commit)
    assert reopened.get_run(run.id).to_dict()["status"] == "running"
    result = reopened.create_experiment("killed", request_id=request)
    assert reopened.create_experiment("killed", request_id=request) == result
    assert reopened._db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    reopened.close()


def test_core_standalone_stdlib_only_no_rng_side_effects(tmp_path):
    script = """
import random
random.seed(123)
before=random.getstate()
m=ExperimentManager(root)
e=m.create_experiment('standalone')
r=m.create_run(e['id'],'training',{})
r.start()
r.log_metric(name='loss',value=.4,unit='loss',source='caller',epoch=1)
r.finish()
m.close()
assert random.getstate()==before
assert not any(k in sys.modules for k in ('torch','numpy','hyso.training','hyso.benchmark'))
print(json.dumps({'sqlite':sqlite3.sqlite_version,'stdlib_only':True}))
"""
    observed = json.loads(finish(launch(script, tmp_path / "stdlib")))
    assert observed["stdlib_only"] is True


def test_fork_context_check(tmp_path, monkeypatch):
    # Cross-platform simulation of inherited PID; real multi-process DB writes
    # are independently tested above, without inherited SQLite connections.
    manager = ExperimentManager(tmp_path / "pid")
    run = new_run(manager)
    original = manager._pid
    monkeypatch.setattr(manager, "_pid", original + 1)
    with pytest.raises(ExperimentError, match="session_context"):
        run.start()
    monkeypatch.setattr(manager, "_pid", original)
    assert run.to_dict()["status"] == "created"
    manager.close()
