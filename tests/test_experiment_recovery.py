"""Real process termination at database/filesystem boundaries; no fixed sleeps."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from hyso.experiments import ExperimentManager, ExperimentError
import hyso.experiments.manager as core

BOOT = """
import importlib.util,json,pathlib,sqlite3,sys,os
spec=importlib.util.spec_from_file_location('standalone_core',sys.argv[1])
core=importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
ExperimentManager=core.ExperimentManager
root=pathlib.Path(sys.argv[2])
def pause():
    print('boundary',flush=True)
    sys.stdin.readline()
    raise RuntimeError('termination barrier must not resume')
"""


def launch(script, root, *args):
    return subprocess.Popen([sys.executable, "-I", "-B", "-u", "-c", BOOT + script,
        str(Path(core.__file__).resolve()), str(root), *args], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def schema2(root):
    m = ExperimentManager(root)
    m.close()
    ExperimentManager.migrate_store(root)


@pytest.mark.parametrize("phase", ["before_reservation", "pending", "pending_with_staging", "staged", "published", "before_ready_commit"])
def test_kill_ingest_and_explicit_reconcile(tmp_path, phase):
    root = tmp_path / "store"
    schema2(root)
    source = tmp_path / "source"
    source.write_bytes(b"untrusted checkpoint" * 100)
    script = """
m=ExperimentManager(root)
r=m.create_run(m.create_experiment('kill')['id'],'training',{})
r.start()
phase=sys.argv[3]
source=pathlib.Path(sys.argv[4])
print(r.id,flush=True)
commit=m._commit
stream=m._stream
publish=m._publish
fence=m._ingest_fence
def patched_commit():
    rows=m._db.execute('SELECT ingest_state FROM artifacts').fetchall()
    if rows and ((phase=='before_reservation' and rows[0][0]=='pending') or
                 (phase=='before_ready_commit' and rows[0][0]=='ready') or
                 (phase=='pending_with_staging' and rows[0][0]=='staged')):
        pause()
    commit()
def patched_stream(path,*args):
    if phase=='pending': pause()
    return stream(path,*args)
def patched_publish(identity):
    if phase=='staged': pause()
    publish(identity)
    if phase=='published': pause()
m._commit=patched_commit
m._stream=patched_stream
m._publish=patched_publish
r.add_artifact(source,ownership='copy')
"""
    process = launch(script, root, phase, str(source))
    try:
        run_id = process.stdout.readline().strip()
        assert len(run_id) == 32
        assert process.stdout.readline().strip() == "boundary"
        process.kill()
        assert process.wait(timeout=20) != 0
        process.communicate(timeout=20)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    m = ExperimentManager(root)
    r = m.get_run(run_id)
    assert r.to_dict()["status"] == "running"
    with pytest.raises(ExperimentError, match="offline_required"):
        m.reconcile_storage()
    m.recover_interrupted(run_id, r.to_dict()["revision"], "old writer stopped")
    result = m.reconcile_storage()
    artifacts = m.list_artifacts()
    if phase == "before_reservation":
        assert artifacts == []
    else:
        expected = "pending" if phase in ("pending", "pending_with_staging") else "ready"
        assert artifacts[0]["ingest_state"] == expected
        if expected == "ready":
            assert m.verify_artifact(artifacts[0]["id"])["status"] == "verified"
    assert r.to_dict()["status"] == "interrupted"
    assert source.read_bytes() == b"untrusted checkpoint" * 100
    assert m._db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    m.close()


def test_migration_process_kill_rolls_back_partial_ddl(tmp_path):
    root = tmp_path / "store"
    m = ExperimentManager(root)
    m.create_experiment("preserved")
    before = m._core_snapshot(m._db)
    m.close()
    script = """
ddl=core._SCHEMA2
def interrupted():
    yield ddl[0]
    pause()
    yield from ddl[1:]
core._SCHEMA2=interrupted()
ExperimentManager.migrate_store(root)
"""
    process = launch(script, root)
    try:
        assert process.stdout.readline().strip() == "boundary"
        process.kill()
        assert process.wait(timeout=20) != 0
        process.communicate(timeout=20)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    m = ExperimentManager(root)
    assert m._schema_version == 1 and m._core_snapshot(m._db) == before
    assert len(list((root / "backups").iterdir())) == 1
    m.close()
    ExperimentManager.migrate_store(root)


def test_expected_revision_recovery_and_old_writer_fencing(tmp_path):
    root = tmp_path / "store"
    schema2(root)
    old = ExperimentManager(root)
    r = old.create_run(old.create_experiment("x")["id"], "training", {})
    r.start()
    r.log_metric(name="loss", value=1, unit="loss", source="caller", epoch=1)
    other = ExperimentManager(root)
    revision = r.to_dict()["revision"]
    with pytest.raises(ExperimentError, match="stale_revision"):
        other.recover_interrupted(r.id, revision - 1, "stale")
    result = other.recover_interrupted(r.id, revision, "coordinated")
    assert result["status"] == "interrupted" and result["writer_session_id"] is None
    with pytest.raises(ExperimentError): r.finish()
    with pytest.raises(ExperimentError):
        r.log_metric(name="loss", value=.5, unit="loss", source="caller", epoch=2)
    assert len(r.read_metrics()) == 1
    with pytest.raises(ExperimentError, match="stale_revision"):
        other.recover_interrupted(r.id, revision, "again")
    other.close()
    old.close()


def test_orphans_mismatch_missing_failed_and_no_delete(tmp_path):
    root = tmp_path / "store"
    schema2(root)
    m = ExperimentManager(root)
    r = m.create_run(m.create_experiment("x")["id"], "training", {})
    r.start()
    source = tmp_path / "source"
    source.write_bytes(b"original")
    a = r.add_artifact(source, ownership="copy")
    b = r.add_artifact(source, ownership="copy")
    r.interrupt()
    (root / a["managed_relative_path"]).write_bytes(b"bad")
    (root / b["managed_relative_path"]).unlink()
    (root / ".staging" / "orphan.part").write_bytes(b"orphan")
    (root / "artifacts" / "orphan").mkdir()
    (root / a["managed_relative_path"]).parent.joinpath("extra").write_bytes(b"extra")
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file() and p.suffix != '.sqlite3'}
    reports = m.reconcile_storage()["observations"]
    statuses = {o["status"] for o in reports}
    assert {"mismatch", "missing", "orphan_staging", "orphan_managed", "unexpected_file"} <= statuses
    after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file() and p.suffix != '.sqlite3'}
    assert before == after
    assert source.read_bytes() == b"original"
    assert m.get_artifact(a["id"])["expected_sha256"] == a["expected_sha256"]
    m.close()


def test_bounded_reconciliation(tmp_path):
    root = tmp_path / "store"
    schema2(root)
    m = ExperimentManager(root)
    (root / ".staging").mkdir()
    for n in range(4): (root / ".staging" / str(n)).write_bytes(b"x")
    with pytest.raises(ExperimentError, match="reconcile_limit"):
        m.reconcile_storage(limit=1)
    assert len(list((root / ".staging").iterdir())) == 4
    m.close()


@pytest.mark.parametrize("committed", [False, True])
def test_migration_commit_ack_loss_keeps_recoverable_store(tmp_path, monkeypatch, committed):
    root = tmp_path / "store"
    m = ExperimentManager(root)
    m.create_experiment("old")
    before = m._core_snapshot(m._db)
    m.close()
    original = ExperimentManager._commit
    def fault(self):
        if committed: original(self)
        raise OSError("ACK lost")
    monkeypatch.setattr(ExperimentManager, "_commit", fault)
    with pytest.raises(ExperimentError) as caught:
        ExperimentManager.migrate_store(root)
    assert caught.value.commit_state == "unknown"
    monkeypatch.setattr(ExperimentManager, "_commit", original)
    m = ExperimentManager(root)
    assert m._schema_version == (2 if committed else 1)
    assert m._core_snapshot(m._db) == before
    assert len(list((root / "backups").iterdir())) == 1
    m.close()


def test_migration_rejects_real_sqlite_writer_lock(tmp_path):
    root = tmp_path / "store"
    m = ExperimentManager(root)
    m.close()
    script = """
db=sqlite3.connect(root/'experiments.sqlite3',isolation_level=None)
db.execute('BEGIN IMMEDIATE')
pause()
"""
    worker = launch(script, root)
    try:
        assert worker.stdout.readline().strip() == "boundary"
        with pytest.raises(ExperimentError, match="storage_busy"):
            ExperimentManager.migrate_store(root)
    finally:
        worker.kill()
        worker.wait(timeout=20)
        worker.communicate()
    m = ExperimentManager(root)
    assert m._schema_version == 1
    m.close()


@pytest.mark.parametrize("state", ["pending", "staged", "failed"])
def test_unknown_or_mismatched_bytes_are_never_adopted(tmp_path, monkeypatch, state):
    root = tmp_path / "store"
    schema2(root)
    m = ExperimentManager(root)
    r = m.create_run(m.create_experiment("x")["id"], "training", {})
    r.start()
    source = tmp_path / "source"
    source.write_bytes(b"original")
    original = m._commit
    if state == "failed":
        def publish_fail(identity): raise PermissionError("denied")
        monkeypatch.setattr(m, "_publish", publish_fail)
    else:
        def commit_fail():
            row = m._db.execute("SELECT ingest_state FROM artifacts").fetchone()
            if row is not None and row[0] == ("staged" if state == "pending" else "ready"):
                raise OSError("commit failure")
            original()
        monkeypatch.setattr(m, "_commit", commit_fail)
    with pytest.raises(ExperimentError):
        r.add_artifact(source, ownership="copy")
    monkeypatch.setattr(m, "_commit", original)
    a = m.list_artifacts()[0]
    staged = m._managed(a["id"], staging=True)
    assert staged.exists()
    staged.write_bytes(b"mismatch")
    r.interrupt()
    reports = m.reconcile_storage()["observations"]
    assert m.get_artifact(a["id"])["ingest_state"] == state
    assert staged.read_bytes() == b"mismatch"
    assert any(o["status"] == ("mismatch" if state == "staged" else "quarantined_" + state) for o in reports)
    assert source.read_bytes() == b"original"
    m.close()


def test_staged_only_mismatch_does_not_publish(tmp_path, monkeypatch):
    root = tmp_path / "store"
    schema2(root)
    m = ExperimentManager(root)
    r = m.create_run(m.create_experiment("x")["id"], "training", {})
    r.start()
    source = tmp_path / "source"
    source.write_bytes(b"original")
    def stop_before_publish(identity):
        raise KeyboardInterrupt()
    monkeypatch.setattr(m, "_publish", stop_before_publish)
    with pytest.raises(KeyboardInterrupt):
        r.add_artifact(source, ownership="copy")
    a = m.list_artifacts()[0]
    assert a["ingest_state"] == "staged"
    staged = m._managed(a["id"], staging=True)
    staged.write_bytes(b"bad bytes")
    r.interrupt()
    reports = m.reconcile_storage()["observations"]
    assert any(o["status"] == "mismatch" for o in reports)
    assert not m._managed(a["id"]).exists()
    assert m.get_artifact(a["id"])["ingest_state"] == "staged"
    m.close()
