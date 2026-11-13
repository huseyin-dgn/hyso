"""Synthetic receipt-chain tests; never native P13 acceptance evidence."""
import sys

import numpy as np
import pytest

from tools.full_acceptance import core, registry, p13_capture_recover as recovery


@pytest.fixture
def capture(tmp_path, monkeypatch):
    run = tmp_path / "run"
    stage = run / "stages" / recovery.STAGE
    stage.mkdir(parents=True)
    snapshot = run / "source"
    source = snapshot / "tools/full_acceptance/p13_capture_worker.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic source only\n")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"synthetic checkpoint; never loaded")
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(dataset, x_test=np.zeros((1, 28, 28), dtype=np.uint8))
    files = {source.relative_to(snapshot).as_posix(): core.sha(source)}
    manifest = run / "manifest.json"
    m = dict(format="hyso.full_acceptance.manifest", schema_version=1, run_id="test-run", output=str(run),
        repository=str(tmp_path / "evolved-live-source"),
        source_snapshot=dict(root=str(snapshot), files=files, tree_sha256=core.digest(files)),
        runtime=dict(python=dict(path=sys.executable)),
        hardware=dict(gpu_uuid="GPU-fixture"), assets=dict(models=dict(B=dict(**core.record(checkpoint),
            recorded_state_sha256="state")), dataset=dict(**core.record(dataset), preprocessing="fixture preprocessing")),
        experiments=registry.experiments())
    for exp in m["experiments"]:
        exp["output"] = str(run / "stages" / exp["id"])
        exp["command"] = [sys.executable, "-I", "-S", "-B", str(snapshot / "tools/run_full_acceptance.py"),
            "--worker", exp["id"], "--manifest", str(manifest), "--stage-output", exp["output"]]
    exp = next(x for x in m["experiments"] if x["id"] == recovery.STAGE)
    exp["authorization"] = dict(approved=True, basis="synthetic unit fixture")
    core.publish(manifest, m)
    mh = core.sha(manifest)
    core.publish(run / "run-status.json", dict(state="RUNNING", controller_active=True, controller_pid=9,
        manifest_sha256=mh, run_id=m["run_id"]))
    core.publish(run / "STARTED.json", dict(pid=9, manifest_sha256=mh))
    gate = dict(assigned=True, exact_job_membership=True, kill_on_close=True)
    core.publish(stage / "ready.json", dict(pid=10, gate=gate, manifest_sha256=mh))
    core.publish(stage / "admit.json", dict(pid=10, parent_pid=9, token="fixture"))
    core.publish(stage / "request.json", dict(token="fixture", snapshot=str(snapshot), manifest=str(manifest),
        manifest_sha256=mh, experiment_id=recovery.STAGE, operation="p13_capture", output=str(stage)))
    np.save(stage / "p13-input.npy", np.zeros((1, 1, 28, 28), dtype=np.float32), allow_pickle=False)
    worker = dict(status="completed", pid=11, forward_labels=recovery.LABELS, output_equal=[True]*3,
        cudnn_tf32_disabled=True, checkpoint_sha256=core.sha(checkpoint), original_state_sha256="state",
        worker_source_sha256=core.sha(source), input_sha256=core.sha(stage / "p13-input.npy"),
        gpu_uuid="GPU-fixture", captured_gpu_identity=dict(physical_id="GPU-fixture"),
        preprocessing="fixture preprocessing", input_shape=[1, 1, 28, 28], input_dtype="torch.float32")
    core.publish(stage / "p13-worker.json", worker)
    for name in ("p13-capture.nsys-rep", "p13-export.sqlite"):
        (stage / name).write_bytes(b"synthetic; reader replaced in these chain tests")
    core.publish(stage / "p13-capture-configuration-preliminary.json", dict(global_pid=(1 << 48) | (11 << 24),
        forward_labels=recovery.LABELS, sqlite=core.record(stage / "p13-export.sqlite"),
        report=core.record(stage / "p13-capture.nsys-rep")))
    core.publish(stage / "p13-capture-analysis-preliminary.json", dict(attribution_complete=True))
    terminal = dict(id=recovery.STAGE, status="PASS", manifest_sha256=mh, gate=gate,
        nsys_exit_code=0, nsys_export_exit_code=0, criteria=dict(P13_native_capture_only=True),
        evidence=[core.record(stage / name) for name in recovery.ARTIFACTS])
    core.publish(stage / "result.json", terminal)
    sup = dict(experiment=recovery.STAGE, operation="p13_capture", manifest_sha256=mh,
        command=exp["command"], pid=10, owned_job_only=True, native_crash=False, exit_code=0,
        reason=None, error=None, automatic_retry=False, cleanup=dict(active_processes=0))
    core.publish(stage / "supervisor-receipt.json", sup)
    core.publish(run / "stage-results.json", dict(run_id=m["run_id"], manifest_sha256=mh,
        experiments=[dict(terminal, attempted=True, supervisor=sup)]))
    from hyso.performance import _module_activity
    monkeypatch.setattr(_module_activity, "analyze", lambda cfg: dict(attribution_complete=True,
        status="partial", provenance_verified=False))
    return manifest, mh, tmp_path / "recovery"


def test_recovery_keeps_interrupted_state_and_all_inputs_read_only(capture):
    manifest, mh, output = capture
    before = {p: core.sha(p) for p in manifest.parent.rglob("*") if p.is_file()}
    result = recovery.audit(manifest, mh, output)
    assert result["status"] == "PARTIAL" and not result["errors"]
    assert not result["controller_terminal_verified"] and not result["capture_receipt_created"]
    assert result["inputs_unchanged"] and not result["provenance_verified"]
    assert before == {p: core.sha(p) for p in before}
    assert not list(output.glob("*receipt*"))


@pytest.mark.parametrize("damage", ["disk_receipt", "gate", "token", "controller_pid", "artifact", "missing_receipt"])
def test_broken_receipt_chain_never_recovers_as_pass(capture, damage):
    manifest, mh, output = capture
    stage = manifest.parent / "stages" / recovery.STAGE
    if damage == "artifact":
        (stage / "p13-capture.nsys-rep").write_bytes(b"changed")
    elif damage == "missing_receipt":
        (stage / "supervisor-receipt.json").unlink()
    else:
        name = {"disk_receipt": "supervisor-receipt", "gate": "ready", "token": "admit",
                "controller_pid": "admit"}[damage]
        path = stage / (name + ".json")
        value = core.read(path)
        if damage == "disk_receipt": value["cleanup"]["active_processes"] = 1
        elif damage == "gate": value["gate"]["kill_on_close"] = False
        elif damage == "token": value["token"] = "other"
        else: value["parent_pid"] = 77
        core.publish(path, value)
    result = recovery.audit(manifest, mh, output)
    assert result["status"] == "FAIL" and result["errors"]
    assert not result["capture_receipt_created"]


def test_recovery_rejects_manifest_and_snapshot_drift_before_output(capture):
    manifest, mh, output = capture
    with pytest.raises(ValueError, match="SHA256"):
        recovery.audit(manifest, "0"*64, output)
    source = manifest.parent / "source/tools/full_acceptance/p13_capture_worker.py"
    source.write_text("changed")
    with pytest.raises(ValueError, match="snapshot source changed"):
        recovery.audit(manifest, mh, output)
    assert not output.exists()


def test_recovery_rejects_overlapping_or_existing_output(capture):
    manifest, mh, output = capture
    for target in (manifest.parent, manifest.parent / "recovery", manifest.parent.parent):
        with pytest.raises(ValueError, match="separate"):
            recovery.audit(manifest, mh, target)
    output.mkdir()
    with pytest.raises(FileExistsError):
        recovery.audit(manifest, mh, output)


def test_finalizer_checks_real_disk_receipt(capture):
    manifest, mh, _ = capture
    m = core.read(manifest)
    stage = manifest.parent / "stages" / recovery.STAGE
    value = core.read(stage / "supervisor-receipt.json")
    value["pid"] = 999
    core.publish(stage / "supervisor-receipt.json", value)
    with pytest.raises(ValueError, match="disk/controller"):
        recovery.validate_capture(m, manifest)
