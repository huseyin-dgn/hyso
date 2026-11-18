"""Real schema1 snapshots, backups, rollback and offline fencing."""
import sqlite3
import uuid
import pytest
from hyso.experiments import ExperimentManager, ExperimentError
import hyso.experiments.manager as core


def populated(root):
    m = ExperimentManager(root)
    e = m.create_experiment("old")
    r = m.create_run(e["id"], "training", {"unicode": "\u015f", "x": [1, 2.]})
    r.start()
    r.log_metric(name="loss", value=.5, unit="loss", source="caller", epoch=1)
    r.finish()
    return m, r


def test_migration_preserves_all_core_bytes_and_backup(tmp_path):
    root = tmp_path / "store"
    m, run = populated(root)
    before = m._core_snapshot(m._db)
    m.close()
    result = ExperimentManager.migrate_store(root)
    assert result["status"] == "migrated"
    with sqlite3.connect(result["backup"]) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
        assert m._core_snapshot(db) == before
    m = ExperimentManager(root)
    assert m._core_snapshot(m._db) == before
    assert m._db.execute("PRAGMA user_version").fetchone()[0] == 2
    record = m.get_run(run.id).to_dict()
    assert record["attempt_no"] == 1 and record["lineage_id"] == run.id
    assert ExperimentManager.migrate_store(root)["status"] == "already_migrated"
    assert len(list((root / "backups").iterdir())) == 1
    m.close()


@pytest.mark.parametrize("point", range(4))
def test_partial_ddl_rollback(tmp_path, monkeypatch, point):
    root = tmp_path / "store"
    m, _ = populated(root)
    before = m._core_snapshot(m._db)
    m.close()
    ddl = core._SCHEMA2
    monkeypatch.setattr(core, "_SCHEMA2", ddl[:point] + ("INVALID SQL",) + ddl[point:])
    with pytest.raises(ExperimentError, match="storage_error"):
        ExperimentManager.migrate_store(root)
    monkeypatch.setattr(core, "_SCHEMA2", ddl)
    m = ExperimentManager(root)
    assert m._schema_version == 1 and m._core_snapshot(m._db) == before
    assert len(list((root / "backups").iterdir())) == 1
    m.close()
    assert ExperimentManager.migrate_store(root)["status"] == "migrated"


def test_offline_boundary_and_old_session_fencing(tmp_path):
    root = tmp_path / "store"
    old = ExperimentManager(root)
    r = old.create_run(old.create_experiment("x")["id"], "training", {})
    r.start()
    with pytest.raises(ExperimentError, match="offline_required"):
        ExperimentManager.migrate_store(root)
    other = ExperimentManager(root)
    other.recover_interrupted(r.id, r.to_dict()["revision"], "operator coordinated")
    other.close()
    ExperimentManager.migrate_store(root)
    with pytest.raises(ExperimentError, match="stale_schema"):
        old.create_experiment("late")
    old._db.close()


@pytest.mark.parametrize("version", [0, 3, 999])
def test_unknown_schema_fail_closed(tmp_path, version):
    root = tmp_path / "store"
    m, _ = populated(root)
    m.close()
    with sqlite3.connect(root / "experiments.sqlite3") as db:
        db.execute("PRAGMA user_version=" + str(version))
    before = (root / "experiments.sqlite3").read_bytes()
    with pytest.raises(ExperimentError):
        ExperimentManager.migrate_store(root)
    assert (root / "experiments.sqlite3").read_bytes() == before


def test_schema1_stays_explicit_and_new_store_policy(tmp_path):
    m = ExperimentManager(tmp_path / "store")
    assert m._schema_version == 1
    m.close()
    m = ExperimentManager(tmp_path / "store")
    assert m._schema_version == 1
    m.close()


def test_schema2_foreign_keys_and_attempt_constraints(tmp_path):
    root = tmp_path / "store"
    m, r = populated(root)
    m.close()
    ExperimentManager.migrate_store(root)
    m = ExperimentManager(root)
    with pytest.raises(sqlite3.IntegrityError):
        m._db.execute("INSERT INTO run_attempts VALUES(?,?,?,1,NULL)", (uuid.uuid4().hex, None, r.id))
    with pytest.raises(sqlite3.IntegrityError):
        m._db.execute("UPDATE run_attempts SET retry_of_run_id=run_id WHERE run_id=?", (r.id,))
    assert m._db.execute("PRAGMA foreign_key_check").fetchone() is None
    m.close()
