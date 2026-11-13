"""Immutable parents, deterministic attempts, resume identity and real races."""
import json
import uuid
import pytest
from hyso.experiments import ExperimentManager, ExperimentError
from test_experiment_artifacts import m, run
from test_experiment_recovery import launch


@pytest.mark.parametrize("terminal", ["failed", "interrupted"])
def test_retry_new_snapshot_and_attempt_sequence(m, terminal):
    parent = run(m)
    getattr(parent, "fail" if terminal == "failed" else "interrupt")("reason")
    before = parent.to_dict()
    e = before["experiment_id"]
    child = m.create_run(e, "training", {"new": [2]}, retry_of_run_id=parent.id)
    assert child.id != parent.id
    assert child.to_dict()["lineage_id"] == parent.id and child.to_dict()["attempt_no"] == 2
    assert child.to_dict()["config"] == {"new": [2]}
    assert parent.to_dict() == before
    child.start()
    child.interrupt()
    third = m.create_run(e, "training", {}, retry_of_run_id=child.id)
    assert third.to_dict()["attempt_no"] == 3 and third.to_dict()["lineage_id"] == parent.id
    with pytest.raises(ExperimentError, match="integrity_conflict"):
        m.create_run(e, "training", {}, retry_of_run_id=parent.id)


@pytest.mark.parametrize("invalid", ["completed", "running", "created", "kind", "experiment", "self", "missing"])
def test_invalid_retry_parent(m, invalid):
    p = run(m)
    e = p.to_dict()["experiment_id"]
    parent_id, kind = p.id, "training"
    if invalid == "completed": p.finish()
    elif invalid == "created":
        parent_id = m.create_run(e, kind, {}).id
    elif invalid in ("kind", "experiment"):
        p.fail("bad")
        if invalid == "kind": kind = "evaluation"
        else: e = m.create_experiment("other")["id"]
    elif invalid in ("self", "missing"): parent_id = uuid.uuid4().hex
    with pytest.raises(ExperimentError):
        m.create_run(e, kind, {}, retry_of_run_id=parent_id)


def test_retry_request_replay_and_changed_payload(m):
    p = run(m)
    p.fail("bad")
    e = p.to_dict()["experiment_id"]
    request = uuid.uuid4().hex
    child = m.create_run(e, "training", {}, retry_of_run_id=p.id, request_id=request)
    child.start()
    child.finish()
    assert m.create_run(e, "training", {}, retry_of_run_id=p.id, request_id=request).id == child.id
    with pytest.raises(ExperimentError, match="request_conflict"):
        m.create_run(e, "training", {"changed": True}, retry_of_run_id=p.id, request_id=request)


@pytest.mark.parametrize("ownership", ["copy", "reference"])
def test_resume_artifact_identity_and_producer(m, tmp_path, ownership):
    p = run(m)
    source = tmp_path / "malicious-checkpoint"
    source.write_bytes(b"untrusted bytes never deserialize")
    a = p.add_artifact(source, ownership=ownership, retain_locator=True)
    p.fail("workload")
    child = m.create_run(p.to_dict()["experiment_id"], "training", {}, retry_of_run_id=p.id, resume_artifact_id=a["id"])
    assert child.to_dict()["status"] == "created" and child.to_dict()["metric_count"] == 0
    assert child.to_dict()["resume_artifact_id"] == a["id"]
    assert m.list_artifacts(child.id)[0]["producer_run_id"] == p.id
    assert m._db.execute("SELECT role FROM run_artifacts WHERE run_id=?", (child.id,)).fetchone()[0] == "input"
    assert m.verify_artifact(a["id"])["status"] == "verified"


@pytest.mark.parametrize("invalid", ["missing", "kind", "changed", "redacted", "failed"])
def test_invalid_resume_artifact(m, tmp_path, invalid):
    p = run(m)
    source = tmp_path / "source"
    source.write_bytes(b"original")
    a = p.add_artifact(source, kind="report" if invalid == "kind" else "checkpoint", retain_locator=invalid != "redacted")
    if invalid == "changed": source.write_bytes(b"changed")
    if invalid == "failed":
        m._db.execute("UPDATE artifacts SET ingest_state='failed',ready_at_utc=NULL WHERE id=?", (a["id"],))
    p.fail("bad")
    with pytest.raises(ExperimentError):
        m.create_run(p.to_dict()["experiment_id"], "training", {}, retry_of_run_id=p.id,
                     resume_artifact_id=uuid.uuid4().hex if invalid == "missing" else a["id"])
    assert len(m.list_runs()) == 1


def test_concurrent_retry_parent_collision(m):
    p = run(m)
    p.fail("bad")
    e = p.to_dict()["experiment_id"]
    script = """
m=ExperimentManager(root)
print('ready',flush=True)
sys.stdin.readline()
try:
    r=m.create_run(sys.argv[3],'training',{},retry_of_run_id=sys.argv[4])
    print(json.dumps({'id':r.id}),flush=True)
except core.ExperimentError as exc:
    print(json.dumps({'error':exc.code}),flush=True)
m.close()
"""
    workers = [launch(script, m.root_dir, e, p.id) for _ in range(2)]
    try:
        for worker in workers: assert worker.stdout.readline().strip() == "ready"
        for worker in workers:
            worker.stdin.write("go\n")
            worker.stdin.flush()
        rows = []
        for worker in workers:
            out, err = worker.communicate(timeout=30)
            assert worker.returncode == 0, err
            rows.append(json.loads(out))
        assert sum("id" in row for row in rows) == 1
        assert [r["error"] for r in rows if "error" in r] == ["integrity_conflict"]
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.communicate()
    assert len(m.list_runs()) == 2


def test_resume_exact_replay_after_checkpoint_disappears(m, tmp_path):
    p = run(m)
    source = tmp_path / "source"
    source.write_bytes(b"identity")
    a = p.add_artifact(source, ownership="copy")
    p.fail("failed")
    request = uuid.uuid4().hex
    e = p.to_dict()["experiment_id"]
    first = m.create_run(e, "training", {}, retry_of_run_id=p.id, resume_artifact_id=a["id"], request_id=request)
    first.start()
    first.finish()
    (m.root_dir / a["managed_relative_path"]).unlink()
    second = m.create_run(e, "training", {}, retry_of_run_id=p.id, resume_artifact_id=a["id"], request_id=request)
    assert first.id == second.id
    assert len(m.list_runs()) == 2
