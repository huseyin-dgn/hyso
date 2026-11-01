"""Identity, immutable snapshots, lifecycle and ownership of schema1 runs."""
import hashlib
import json
import sqlite3
import threading
import uuid

import pytest

from hyso.experiments import ExperimentError, ExperimentManager


@pytest.fixture
def manager(tmp_path):
    manager = ExperimentManager(tmp_path / "runs")
    yield manager
    manager.close()


def make_run(manager, **kwargs):
    experiment = manager.create_experiment("study")
    return manager.create_run(experiment["id"], "training", kwargs.pop("config", {}), **kwargs)


def test_public_example_and_defensive_snapshots(manager):
    experiment = manager.create_experiment("CNN Attention Comparison")
    config = {"model": "ModularCNN", "attention": "eca", "optimizer": "AdamW", "seed": 42, "nested": [1]}
    provenance = {"dataset": {"status": "not_collected", "value": None}}
    run = manager.create_run(experiment["id"], "training", config, provenance)
    config["nested"].append(2)
    provenance["dataset"]["status"] = "changed"
    snapshot = run.to_dict()
    snapshot["config"]["nested"].append(3)
    experiment["name"] = "changed"
    assert manager.get_experiment(experiment["id"])["name"] == "CNN Attention Comparison"
    assert run.to_dict()["config"]["nested"] == [1]
    assert run.to_dict()["provenance"]["dataset"]["status"] == "not_collected"
    run.start()
    run.log_metric(name="train/loss", value=.42, epoch=1, unit="mean_loss", source="caller")
    run.log_metric(name="validation/accuracy", value=91., epoch=1, unit="percent", source="caller")
    run.finish(reason="completed")
    final = run.to_dict()
    assert final["status"] == "completed" and final["metric_count"] == 2
    assert final["schema_version"] == 1 and final["format"] == "hyso.experiment"
    manager.close()
    reopened = ExperimentManager(manager.root_dir)
    assert reopened.get_run(run.id).to_dict() == final
    assert len(reopened.get_run(run.id).read_metrics()) == 2
    reopened.close()


@pytest.mark.parametrize("kind", ["training", "evaluation", "benchmark"])
def test_identity_names_fk_and_kinds(manager, kind):
    first = manager.create_experiment("../../display only")
    second = manager.create_experiment("../../display only")
    assert first["id"] != second["id"]
    run = manager.create_run(first["id"], kind, {})
    record = manager.get_run(run.id).to_dict()
    assert record["experiment_id"] == first["id"] and record["kind"] == kind
    assert uuid.UUID(run.id).hex == run.id
    assert record["seq"] > 0 and record["revision"] == 0
    with pytest.raises(ExperimentError, match="not_found"):
        manager.create_run(uuid.uuid4().hex, kind, {})
    assert len(manager.list_runs()) == 1


def test_canonical_hash_and_request_idempotency(manager):
    request = uuid.uuid4().hex
    experiment = manager.create_experiment("study", request_id=request)
    assert manager.create_experiment("study", request_id=request) == experiment
    with pytest.raises(ExperimentError, match="request_conflict"):
        manager.create_experiment("different", request_id=request)
    request = uuid.uuid4().hex
    run = manager.create_run(experiment["id"], "training", {"z": [1], "a": "ş"}, request_id=request)
    same = manager.create_run(experiment["id"], "training", {"a": "ş", "z": [1]}, request_id=request)
    assert same.id == run.id
    canonical = json.dumps({"a": "ş", "z": [1]}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert run.to_dict()["config_hash"] == hashlib.sha256(canonical.encode()).hexdigest()
    with pytest.raises(ExperimentError, match="request_conflict"):
        manager.create_run(experiment["id"], "training", {"z": [2]}, request_id=request)
    assert len(manager.list_runs()) == 1


def transition(run, target, **kwargs):
    return {"running": run.start, "completed": run.finish,
            "failed": lambda **kw: run.fail("failure", **kw), "interrupted": run.interrupt}[target](**kwargs)


@pytest.mark.parametrize("initial", ["created", "running", "completed", "failed", "interrupted"])
@pytest.mark.parametrize("target", ["running", "completed", "failed", "interrupted"])
def test_all_lifecycle_edges(manager, initial, target):
    run = make_run(manager)
    if initial != "created":
        run.start()
        if initial != "running":
            transition(run, initial)
    before = run.to_dict()
    allowed = {"created": {"running", "failed", "interrupted"}, "running": {"completed", "failed", "interrupted"}}
    if target not in allowed.get(initial, set()):
        with pytest.raises(ExperimentError, match="invalid_transition"):
            transition(run, target)
        assert run.to_dict() == before
    else:
        request = uuid.uuid4().hex
        result = transition(run, target, request_id=request)
        after = run.to_dict()
        assert after["status"] == target and after["revision"] == before["revision"] + 1
        assert transition(run, target, request_id=request) == result
        assert run.to_dict() == after
        assert (after["ended_at_utc"] is None) == (target == "running")
        with sqlite3.connect(manager.root_dir / "experiments.sqlite3") as db:
            assert db.execute("SELECT count(*) FROM run_events WHERE operation_id=?", (request,)).fetchone()[0] == 1


def test_lifecycle_request_conflicts_and_old_retry(manager):
    run = make_run(manager)
    req = uuid.uuid4().hex
    run.start(request_id=req)
    run.finish()
    before = run.to_dict()
    assert run.start(request_id=req) == {}
    assert run.to_dict() == before
    with pytest.raises(ExperimentError, match="request_conflict"):
        run.interrupt(request_id=req)
    other = make_run(manager)
    with pytest.raises(ExperimentError, match="request_conflict"):
        other.start(request_id=req)


def test_no_auto_completion_and_foreign_writer(manager):
    run = make_run(manager)
    run.start()
    before = run.to_dict()
    other = ExperimentManager(manager.root_dir)
    foreign = other.get_run(run.id)
    assert foreign.to_dict() == before
    for action in (foreign.finish, foreign.interrupt, lambda: foreign.fail("fault"),
                   lambda: foreign.log_metric(name="loss", value=1., epoch=1, unit="loss", source="caller")):
        with pytest.raises(ExperimentError, match="foreign_writer"):
            action()
    other.close()
    manager.close()
    reopened = ExperimentManager(manager.root_dir)
    assert reopened.get_run(run.id).to_dict() == before
    with pytest.raises(ExperimentError, match="foreign_writer"):
        reopened.get_run(run.id).finish()
    reopened.close()


def test_closed_and_thread_bound(manager):
    run = make_run(manager)
    errors = []
    def worker():
        for action in (run.to_dict, run.start, manager.close):
            try:
                action()
            except ExperimentError as exc:
                errors.append(exc.code)
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive() and errors == ["session_context"] * 3
    with manager._access("outer"):
        with pytest.raises(ExperimentError, match="reentrant_session"):
            run.to_dict()
    manager.close()
    manager.close()
    for action in (run.to_dict, run.start, lambda: manager.create_experiment("closed"), run.read_metrics):
        with pytest.raises(ExperimentError, match="closed_session"):
            action()


@pytest.mark.parametrize("key", ["password", "TOKEN", "secret", "api_key", "auth"])
def test_secret_rejection_and_redaction(manager, key):
    secret = "DO_NOT_PERSIST_this_secret"
    with pytest.raises(ExperimentError) as caught:
        make_run(manager, config={key: secret})
    assert caught.value.code == "sensitive_key" and secret not in str(caught.value)
    run = make_run(manager, config={key: secret}, redact_paths={"config": ["/" + key]})
    assert run.to_dict()["config"][key] == {"value": None, "status": "redacted", "reason": "caller_redaction"}
    assert secret.encode() not in (manager.root_dir / "experiments.sqlite3").read_bytes()


def test_json_pointer_lists_escaping_and_provenance(manager):
    run = make_run(manager, config={"a/b": [{"~key": "hidden"}]},
                   provenance={"auth": "hidden"},
                   redact_paths={"config": ["/a~1b/0/~0key"], "provenance": ["/auth"]})
    assert run.to_dict()["config"]["a/b"][0]["~key"]["value"] is None
    assert run.to_dict()["provenance"]["auth"]["status"] == "redacted"


@pytest.mark.parametrize("value", [b"bytes", {1, 2}, (1, 2), object(), lambda: 1, float("nan"), float("inf"), {1: "bad"}])
def test_unsafe_config_rejected(manager, value):
    with pytest.raises(ExperimentError):
        make_run(manager, config={"value": value})
    assert manager.list_runs() == []


@pytest.mark.parametrize("case", ["string", "bytes", "depth", "pointer", "escape", "surrogate"])
def test_snapshot_limits(manager, case):
    config, paths = {}, {}
    if case == "string":
        config = {"value": "ş" * 32769}
    elif case == "bytes":
        config = {str(i): "x" * 65536 for i in range(16)}
    elif case == "depth":
        config = {"v": 0}
        for _ in range(17):
            config = {"v": config}
    elif case == "pointer":
        paths = {"config": ["/missing"]}
    elif case == "escape":
        paths = {"config": ["/bad~2escape"]}
    else:
        config = {"v": "\ud800"}
    with pytest.raises(ExperimentError):
        make_run(manager, config=config, redact_paths=paths)
    assert manager.list_runs() == []


def test_b2_is_not_implemented(manager):
    exp = manager.create_experiment("study")
    for param in ("retry_of_run_id", "resume_artifact_id"):
        with pytest.raises(ExperimentError, match="unsupported_b2"):
            manager.create_run(exp["id"], "training", {}, **{param: uuid.uuid4().hex})
    # Preserve this B1 testcase identity and its schema1 feature boundary.
    # B2 management methods now exist, but opening schema1 does not migrate it.
    assert manager._db.execute("PRAGMA user_version").fetchone()[0] == 1
    for action in (manager.reconcile_storage,
                   lambda: manager.link_artifact(uuid.uuid4().hex, uuid.uuid4().hex)):
        with pytest.raises(ExperimentError, match="unsupported_b2"):
            action()
    assert all(callable(getattr(manager, name)) for name in
               ("recover_interrupted", "migrate_store", "reconcile_storage", "link_artifact"))


def test_bounded_run_queries(manager):
    exp = manager.create_experiment("study")
    ids = [manager.create_run(exp["id"], kind, {}).id for kind in ("training", "evaluation", "benchmark", "training")]
    manager.get_run(ids[-1]).fail("preflight")
    first = manager.list_runs(exp["id"], limit=2)
    second = manager.list_runs(exp["id"], after_seq=first[-1]["seq"], limit=2)
    assert [r["id"] for r in first + second] == ids
    assert [r["id"] for r in manager.list_runs(exp["id"], status="failed", kind="training")] == ids[-1:]
    first[0]["config"]["new"] = 5
    assert manager.get_run(ids[0]).to_dict()["config"] == {}
    for kwargs in ({"limit": 0}, {"limit": 1001}, {"limit": True}, {"after_seq": -1}, {"kind": "bad"}, {"status": "bad"}):
        with pytest.raises(ExperimentError):
            manager.list_runs(**kwargs)


@pytest.mark.parametrize("kind", ["train", "", None, 1, []])
def test_invalid_run_kinds(manager, kind):
    experiment = manager.create_experiment("invalid")
    with pytest.raises(ExperimentError, match="invalid_kind"):
        manager.create_run(experiment["id"], kind, {})
    assert manager.list_runs() == []


def test_no_arbitrary_repr_or_implicit_tensor_serialization(manager):
    import torch
    class Hostile:
        def __repr__(self):
            raise AssertionError("repr must not run")
    for value in (Hostile(), torch.tensor(1), torch.nn.Linear(1, 1)):
        with pytest.raises(ExperimentError, match="unsafe_json"):
            make_run(manager, config={"v": value})
    assert manager.list_runs() == []


def test_config_hash_is_sanitized_not_original(manager):
    hashes = [make_run(manager, config={"token": secret}, redact_paths={"config": ["/token"]}).to_dict()["config_hash"]
              for secret in ("secret-one", "secret-two")]
    assert hashes[0] == hashes[1]


def test_large_json_integer_keeps_exact_type_and_sequence_redaction(manager):
    run = make_run(manager, config={"integer": 10**1500, "token": "hide"}, redact_paths=("/token",))
    value = run.to_dict()["config"]["integer"]
    assert type(value) is int and value == 10**1500
    assert run.to_dict()["config"]["token"]["value"] is None


def test_failure_domain_records_workload_success(manager):
    run = make_run(manager)
    run.start()
    run.fail("metric_store_unavailable", failure_domain="tracking", workload_outcome="succeeded")
    assert run.to_dict()["reason"] == {"code": "metric_store_unavailable", "failure_domain": "tracking", "workload_outcome": "succeeded"}


def test_default_run_query_limit(manager):
    experiment = manager.create_experiment("bounded")
    for _ in range(101):
        manager.create_run(experiment["id"], "evaluation", {})
    page = manager.list_runs()
    assert len(page) == 100
    assert len(manager.list_runs(after_seq=page[-1]["seq"])) == 1
