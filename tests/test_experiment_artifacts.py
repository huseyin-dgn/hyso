"""Streaming identity, ownership, safe filesystem and lifecycle boundaries."""
import hashlib
import os
from pathlib import Path
import sqlite3
import uuid
import pytest
from hyso.experiments import ExperimentManager, ExperimentError


@pytest.fixture
def m(tmp_path):
    root = tmp_path / "store"
    manager = ExperimentManager(root)
    manager.close()
    ExperimentManager.migrate_store(root)
    manager = ExperimentManager(root)
    yield manager
    manager.close()


def run(m):
    r = m.create_run(m.create_experiment("artifacts")["id"], "training", {})
    r.start()
    return r


@pytest.mark.parametrize("ownership", ["reference", "copy"])
def test_identity_source_privacy_and_defensive_records(m, tmp_path, ownership):
    r = run(m)
    source = tmp_path / "untrusted.ckpt"
    payload = b"not pickle or a valid model" * 100000
    source.write_bytes(payload)
    a = r.add_artifact(source, ownership=ownership, metadata={"origin": "caller"})
    assert a["expected_sha256"] == hashlib.sha256(payload).hexdigest()
    assert a["expected_size_bytes"] == len(payload) and a["ingest_state"] == "ready"
    assert a["producer_run_id"] == r.id and a["original_locator"] is None
    assert source.read_bytes() == payload
    if ownership == "copy":
        assert (m.root_dir / a["managed_relative_path"]).read_bytes() == payload
    else:
        assert a["managed_relative_path"] is None
        assert m.verify_artifact(a["id"])["status"] == "unavailable"
    assert m.verify_artifact(a["id"], path=source)["status"] == "verified"
    a["metadata"]["origin"] = "mutated"
    assert m.get_artifact(a["id"])["metadata"] == {"origin": "caller"}
    r.finish()


@pytest.mark.parametrize("change,status", [("change", "mismatch"), ("remove", "missing")])
@pytest.mark.parametrize("ownership", ["reference", "copy"])
def test_verification_observations_append_only(m, tmp_path, change, status, ownership):
    r = run(m)
    source = tmp_path / "source"
    source.write_bytes(b"original")
    a = r.add_artifact(source, ownership=ownership, retain_locator=True)
    assert m.verify_artifact(a["id"])["status"] == "verified"
    path = m.root_dir / a["managed_relative_path"] if ownership == "copy" else source
    if change == "change":
        path.write_bytes(b"different")
    else:
        path.unlink()
    assert m.verify_artifact(a["id"])["status"] == status
    assert m.get_artifact(a["id"])["expected_sha256"] == a["expected_sha256"]
    assert m._db.execute("SELECT count(*) FROM artifact_checks").fetchone()[0] == 2


def test_producer_consumers_links_and_terminal_replay(m, tmp_path):
    a_run, b_run = run(m), run(m)
    source = tmp_path / "source"
    source.write_bytes(b"checkpoint")
    request = uuid.uuid4().hex
    artifact = a_run.add_artifact(source, ownership="copy", request_id=request)
    first = m.link_artifact(b_run.id, artifact["id"])
    assert m.link_artifact(b_run.id, artifact["id"]) == first
    assert m.get_artifact(artifact["id"])["producer_run_id"] == a_run.id
    assert m.list_artifacts(b_run.id)[0]["id"] == artifact["id"]
    a_run.finish()
    assert a_run.add_artifact(source, ownership="copy", request_id=request) == artifact
    with pytest.raises(ExperimentError, match="request_conflict"):
        a_run.add_artifact(source, request_id=request)
    with pytest.raises(ExperimentError, match="run_not_running"):
        a_run.add_artifact(source)


@pytest.mark.parametrize("fault", ["limit", "fsync", "publish", "partial", "permission"])
def test_ingest_failures_block_completion_preserve_source(m, tmp_path, monkeypatch, fault):
    r = run(m)
    source = tmp_path / "source"
    source.write_bytes(b"source intact")
    kwargs = {"ownership": "copy"}
    def fail(*args, **kwargs):
        raise PermissionError("private path SECRET")
    if fault == "limit":
        kwargs["max_size_bytes"] = 1
    elif fault == "fsync":
        monkeypatch.setattr(os, "fsync", fail)
    elif fault == "publish":
        monkeypatch.setattr(os, "link", fail)
    elif fault == "permission":
        monkeypatch.setattr(m, "_stream", fail)
    elif fault == "partial":
        original = m._stream
        def partial(path, destination=None, limit=None):
            destination.write(b"part")
            raise OSError("disk full")
        monkeypatch.setattr(m, "_stream", partial)
    with pytest.raises(ExperimentError) as caught:
        r.add_artifact(source, **kwargs)
    assert "SECRET" not in str(caught.value)
    assert source.read_bytes() == b"source intact"
    a = m.list_artifacts(r.id)[0]
    assert a["ingest_state"] == "failed"
    assert m._db.execute("SELECT status FROM artifact_checks").fetchone()[0] == "unavailable"
    with pytest.raises(ExperimentError, match="outputs_not_ready"):
        r.finish()


def test_no_clobber_publication(m, tmp_path, monkeypatch):
    r = run(m)
    source = tmp_path / "source"
    source.write_bytes(b"original")
    original = m._publish
    def conflict(identity):
        destination = m._managed(identity)
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"foreign")
        original(identity)
    monkeypatch.setattr(m, "_publish", conflict)
    with pytest.raises(ExperimentError):
        r.add_artifact(source, ownership="copy")
    a = m.list_artifacts()[0]
    assert (m.root_dir / a["managed_relative_path"]).read_bytes() == b"foreign"
    assert source.read_bytes() == b"original"


@pytest.mark.parametrize("bad", ["directory", "missing", "metadata", "role", "ownership"])
def test_invalid_inputs(m, tmp_path, bad):
    r = run(m)
    source = tmp_path / "source"
    source.write_bytes(b"x")
    options = {}
    if bad == "directory": source = tmp_path
    if bad == "missing": source = tmp_path / "absent"
    if bad == "metadata": options["metadata"] = {"secret": "never persist"}
    if bad == "role": options["role"] = "../output"
    if bad == "ownership": options["ownership"] = "../copy"
    with pytest.raises(ExperimentError):
        r.add_artifact(source, **options)
    assert (tmp_path / "source").read_bytes() == b"x"


def test_foreign_writer_and_foreign_store(m, tmp_path):
    r = run(m)
    source = tmp_path / "source"
    source.write_bytes(b"x")
    other = ExperimentManager(m.root_dir)
    with pytest.raises(ExperimentError, match="foreign_writer"):
        other.get_run(r.id).add_artifact(source)
    a = r.add_artifact(source)
    with pytest.raises(ExperimentError, match="not_found"):
        m.link_artifact(r.id, uuid.uuid4().hex)
    other.close()


def test_managed_escape_and_checksum_constraints(m, tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"x")
    a = run(m).add_artifact(source, ownership="copy")
    with pytest.raises(sqlite3.IntegrityError):
        m._db.execute("UPDATE artifacts SET expected_sha256='INVALID' WHERE id=?", (a["id"],))
    m._db.execute("UPDATE artifacts SET managed_relative_path='../escape' WHERE id=?", (a["id"],))
    with pytest.raises(ExperimentError, match="unsafe_path"):
        m.verify_artifact(a["id"])


def test_windows_junction_source_and_managed_root(m, tmp_path):
    # Native junctions need no symbolic-link privilege on Windows.
    import subprocess
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "source").write_bytes(b"external")
    junction = tmp_path / "junction"
    if os.name == "nt":
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(outside)], capture_output=True)
        assert result.returncode == 0
    else:
        junction.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ExperimentError, match="unsafe_path"):
        run(m).add_artifact(junction / "source")
    assert (outside / "source").read_bytes() == b"external"


def test_source_replacement_detected(m, tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"x")
    original = os.fstat
    count = 0
    def mutate(fd):
        nonlocal count
        info = original(fd)
        count += 1
        if count == 2:
            source.write_bytes(b"replacement with different size")
        return info
    monkeypatch.setattr(os, "fstat", mutate)
    with pytest.raises(ExperimentError, match="source_changed"):
        run(m).add_artifact(source)


@pytest.mark.parametrize("identity", ["../escape", "/absolute", "..", "a" * 31, "A" * 32])
def test_caller_cannot_choose_managed_destination(m, identity):
    with pytest.raises(ExperimentError, match="invalid_id"):
        m._managed(identity)


def test_symlink_mode_rejection_without_platform_privilege(m, tmp_path, monkeypatch):
    import stat
    source = tmp_path / "source"
    source.write_bytes(b"external")
    original = Path.lstat
    def symlink_info(path, *args, **kwargs):
        info = original(path, *args, **kwargs)
        if path == source:
            fields = list(info)
            fields[0] = stat.S_IFLNK | 0o777
            return os.stat_result(fields)
        return info
    monkeypatch.setattr(Path, "lstat", symlink_info)
    with pytest.raises(ExperimentError, match="unsafe_path"):
        run(m).add_artifact(source)
    assert source.read_bytes() == b"external"


def test_streaming_ingest_does_not_hold_sqlite_transaction(m, tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"x" * 2000000)
    original = m._stream
    observations = []
    def inspect(*args, **kwargs):
        observations.append(m._db.in_transaction)
        return original(*args, **kwargs)
    monkeypatch.setattr(m, "_stream", inspect)
    run(m).add_artifact(source, ownership="copy")
    assert observations and not any(observations)


def test_verify_unreadable_observation(m, tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"x")
    a = run(m).add_artifact(source, retain_locator=True)
    def deny(*args, **kwargs): raise PermissionError("secret path")
    monkeypatch.setattr(m, "_stream", deny)
    observation = m.verify_artifact(a["id"])
    assert observation["status"] == "unreadable"
    assert observation["observed_sha256"] is None


@pytest.mark.parametrize("phase", ["pending", "staged", "ready"])
@pytest.mark.parametrize("committed", [False, True])
def test_ingest_commit_uncertainty_never_claims_success(m, tmp_path, monkeypatch, phase, committed):
    source = tmp_path / "source"
    source.write_bytes(b"immutable")
    r = run(m)
    request = uuid.uuid4().hex
    original = m._commit
    triggered = False
    def fault():
        nonlocal triggered
        row = m._db.execute("SELECT ingest_state FROM artifacts WHERE request_id=?", (request,)).fetchone()
        if not triggered and row is not None and row[0] == phase:
            triggered = True
            if committed: original()
            raise OSError("ACK lost")
        original()
    monkeypatch.setattr(m, "_commit", fault)
    with pytest.raises(ExperimentError) as caught:
        r.add_artifact(source, ownership="copy", request_id=request)
    assert caught.value.commit_state == "unknown" and triggered
    monkeypatch.setattr(m, "_commit", original)
    rows = m.list_artifacts()
    if phase == "pending" and not committed:
        assert rows == []
        assert r.add_artifact(source, ownership="copy", request_id=request)["ingest_state"] == "ready"
    else:
        assert len(rows) == 1
        replay = r.add_artifact(source, ownership="copy", request_id=request)
        assert replay["id"] == rows[0]["id"]
        assert replay["ingest_state"] == (phase if committed else {"staged": "pending", "ready": "staged"}[phase])
    assert source.read_bytes() == b"immutable"
