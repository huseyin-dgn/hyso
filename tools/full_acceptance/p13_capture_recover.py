"""Read-only interrupted P13 audit. Never resumes a run or creates a receipt."""
from pathlib import Path
import argparse
import json
import sys

_toolroot = str(Path(__file__).resolve().parents[1])
if _toolroot not in sys.path:
    sys.path.insert(0, _toolroot)

from .core import read, record, publish, sha, verify, now
from .p13_capture_native import LABELS

STAGE = "06.p13_capture"
ARTIFACTS = ("p13-worker.json", "p13-input.npy", "p13-capture.nsys-rep",
             "p13-export.sqlite", "p13-capture-configuration-preliminary.json",
             "p13-capture-analysis-preliminary.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_capture(m, manifest):
    """Validate retained attestations against the actual on-disk Job receipt.

    These are unsigned historical records, not independently observed process
    membership. OS PIDs and admission handles are never reopened or reused.
    """
    prior = Path(m["output"]).resolve()
    stage = prior / "stages" / STAGE
    mh = sha(manifest)
    report = read(prior / "stage-results.json")
    require(report.get("manifest_sha256") == mh and report.get("run_id") == m["run_id"],
            "controller report identity mismatch")
    rows = [x for x in report["experiments"] if x["id"] == STAGE]
    require(len(rows) == 1, "missing/duplicate capture row")
    row = rows[0]
    exp, = [x for x in m["experiments"] if x["id"] == STAGE]
    require(exp["authorization"]["approved"] is True and bool(exp["authorization"].get("basis")),
            "capture authorization missing")
    require(row.get("status") == "PASS" and row.get("attempted") is True
            and row.get("criteria", {}).get("P13_native_capture_only") is True,
            "native capture did not pass")
    sup = read(stage / "supervisor-receipt.json")
    require(sup == row.get("supervisor"), "disk/controller supervisor receipt mismatch")
    require(sup.get("experiment") == STAGE and sup.get("operation") == "p13_capture"
            and sup.get("manifest_sha256") == mh and sup.get("command") == exp["command"]
            and sup.get("owned_job_only") is True and sup.get("native_crash") is False
            and type(sup.get("exit_code")) is int and sup["exit_code"] == 0
            and sup.get("reason") is None and sup.get("error") is None
            and sup.get("automatic_retry") is False
            and type(sup.get("cleanup", {}).get("active_processes")) is int
            and sup["cleanup"]["active_processes"] == 0, "invalid owned Job supervisor receipt")
    ready, admission, request, terminal = [read(stage / name) for name in
        ("ready.json", "admit.json", "request.json", "result.json")]
    require(type(sup.get("pid")) is int and sup["pid"] > 0
            and sup["pid"] == ready.get("pid") == admission.get("pid"), "stage PID mismatch")
    require(request.get("manifest_sha256") == mh and request.get("manifest") == str(manifest)
            and request.get("snapshot") == m["source_snapshot"]["root"]
            and request.get("experiment_id") == STAGE and request.get("operation") == "p13_capture"
            and request.get("output") == str(stage)
            and bool(request.get("token")) and request["token"] == admission.get("token"),
            "Job admission/request identity mismatch")
    require(ready.get("manifest_sha256") == mh
            and all(ready.get("gate", {}).get(k) is True for k in
                    ("assigned", "exact_job_membership", "kill_on_close"))
            and terminal.get("gate") == ready["gate"], "Job admission gate mismatch")
    require(terminal.get("id") == STAGE and terminal.get("manifest_sha256") == mh
            and terminal.get("status") == "PASS"
            and type(terminal.get("nsys_exit_code")) is int and terminal["nsys_exit_code"] == 0
            and type(terminal.get("nsys_export_exit_code")) is int and terminal["nsys_export_exit_code"] == 0,
            "Nsight terminal identity/exit mismatch")
    for key, value in terminal.items():
        require(row.get(key) == value, "controller/worker terminal mismatch: " + key)
    refs = terminal.get("evidence", [])
    require(len(refs) == len(ARTIFACTS) and len({r["path"] for r in refs}) == len(refs),
            "missing/duplicate capture artifact")
    indexed = {str(Path(r["path"]).resolve()): r for r in refs}
    for name in ARTIFACTS:
        path = stage / name
        require(not path.is_symlink() and path.is_file(), "missing/linked artifact: " + name)
        require(indexed.get(str(path)) == record(path), "capture artifact changed: " + name)
    worker = read(stage / "p13-worker.json")
    cp = m["assets"]["models"]["B"]
    source = Path(m["source_snapshot"]["root"]) / "tools/full_acceptance/p13_capture_worker.py"
    require(sha(cp["path"]) == cp["sha256"], "Model B checkpoint changed")
    require(worker.get("status") == "completed" and type(worker.get("pid")) is int and worker["pid"] > 0
            and worker.get("forward_labels") == LABELS
            and type(worker.get("output_equal")) is list and len(worker["output_equal"]) == 3
            and all(x is True for x in worker["output_equal"])
            and worker.get("cudnn_tf32_disabled") is True
            and worker.get("checkpoint_sha256") == cp["sha256"]
            and worker.get("original_state_sha256") == cp["recorded_state_sha256"]
            and worker.get("worker_source_sha256") == sha(source)
            and worker.get("input_sha256") == sha(stage / "p13-input.npy")
            and worker.get("gpu_uuid") == m["hardware"]["gpu_uuid"]
            and worker.get("captured_gpu_identity", {}).get("physical_id") == m["hardware"]["gpu_uuid"]
            and worker.get("preprocessing") == m["assets"]["dataset"]["preprocessing"],
            "native worker model/input/source/GPU/output binding mismatch")
    # NPY header + bounded array validation only; no model forward or torch.load.
    import numpy as np
    require((stage / "p13-input.npy").stat().st_size <= 1024**2, "input byte budget exceeded")
    sample = np.load(stage / "p13-input.npy", allow_pickle=False)
    require(list(sample.shape) == worker.get("input_shape") == [1, 1, 28, 28]
            and sample.dtype == np.dtype("float32") and worker.get("input_dtype") == "torch.float32"
            and np.isfinite(sample).all(), "captured input contract mismatch")
    dataset = m["assets"]["dataset"]
    require(sha(dataset["path"]) == dataset["sha256"], "captured dataset changed")
    import zipfile
    with zipfile.ZipFile(dataset["path"]) as archive:
        require(archive.getinfo("x_test.npy").file_size <= 64*1024**2, "test input expansion budget exceeded")
    with np.load(dataset["path"], allow_pickle=False) as data:
        raw = data["x_test"]
        require(raw.dtype == np.uint8 and raw.shape[1:] == (28, 28) and len(raw) > 0,
                "test dataset shape/dtype mismatch")
        expected_input = raw[:1, None].astype(np.float32) / np.float32(255)
    require(np.array_equal(sample, expected_input), "captured input differs from sealed test preprocessing")
    cfg = read(stage / "p13-capture-configuration-preliminary.json")
    require(not cfg.get("capture") and cfg.get("forward_labels") == LABELS,
            "recovery requires preliminary configuration, not a manufactured receipt")
    for key, name in (("sqlite", "p13-export.sqlite"), ("report", "p13-capture.nsys-rep")):
        require(cfg.get(key) == record(stage / name), "configuration artifact routing mismatch")
    gp = cfg.get("global_pid")
    require(type(gp) is int and ((gp >> 24) & 0xffffff) == worker["pid"], "encoded worker PID mismatch")
    return cfg, sup, admission, worker


def audit(manifest, expected_sha, output):
    manifest = Path(manifest).resolve(strict=True)
    prior = manifest.parent
    output = Path(output).resolve()
    require(output != prior and prior not in output.parents and output not in prior.parents,
            "recovery output must be separate from consumed evidence")
    if output.exists():
        raise FileExistsError("recovery requires a new directory")
    # Validate only the sealed snapshot here. Live source evolution does not
    # invalidate a historical offline audit. No runtime DLLs are executed/read.
    m = verify(manifest, expected=expected_sha, live=False, assets=False)
    inputs = [manifest, prior / "run-status.json", prior / "stage-results.json", prior / "STARTED.json"]
    stage = prior / "stages" / STAGE
    inputs += [p for p in stage.iterdir() if p.is_file()]
    inputs += [Path(m["source_snapshot"]["root"]) / rel for rel in m["source_snapshot"]["files"]]
    inputs += [Path(m["assets"]["models"]["B"]["path"]), Path(m["assets"]["dataset"]["path"])]
    before = [record(p) for p in inputs]
    state = read(prior / "run-status.json")
    started = read(prior / "STARTED.json")
    checks, errors, analysis = [], [], None
    try:
        require(state.get("manifest_sha256") == expected_sha and state.get("run_id") == m["run_id"]
                and started.get("manifest_sha256") == expected_sha, "controller manifest/run identity mismatch")
        cfg, sup, admission, worker = validate_capture(m, manifest)
        require(type(started.get("pid")) is int and started["pid"] == state.get("controller_pid")
                == admission.get("parent_pid"), "recorded controller/admission PID mismatch")
        checks += ["sealed_manifest_and_snapshot_hashes", "disk_Job_receipt_matches_controller",
                   "stage_request_admission_ready_terminal_identity", "capture_artifact_hashes",
                   "Model_B_checkpoint_and_original_state", "input_shape_dtype_hash_preprocessing",
                   "worker_source_and_physical_GPU_UUID", "three_true_output_equivalence_attestations"]
        from hyso.performance._module_activity import analyze
        analysis = analyze(cfg)
        require(analysis.get("attribution_complete") is True, "incomplete measured CUDA attribution")
        checks.append("offline_CUDA_kernel_module_attribution")
    except (ValueError, OSError, KeyError, TypeError) as error:
        errors.append(type(error).__name__ + ": " + str(error))
    unchanged = before == [record(p) for p in inputs]
    if not unchanged:
        errors.append("input evidence changed during audit")
    terminal_verified = (state.get("controller_active") is False and state.get("preservation_status") == "PASS"
                         and not state.get("finalization_errors") and state.get("state") not in ("RUNNING", "FINALIZING"))
    result = dict(format="hyso.p13.interrupted_capture_audit", version=1, audited_at=now(),
        status="FAIL" if errors else "PARTIAL", p13_acceptance="FAIL" if errors else "PARTIAL",
        suite_complete=False, manifest=record(manifest), verified=checks, errors=errors,
        controller_terminal_verified=terminal_verified, recorded_controller_state=state,
        process_liveness="not_inferred_from_reusable_PID; no current OS ownership claim",
        missing=["authorized_06.layer_CPU_ownership_calibration_and_verified_CUDA_acceptance"]
            + ([] if terminal_verified else ["clean_controller_terminal_and_preservation_receipt"]),
        report_sqlite_relationship="worker export success + pinned pair; binary report export relationship not independently decoded",
        output_equivalence="retained worker attestation; outputs not retained for independent recomputation",
        provenance_verified=False, capture_receipt_created=False, inputs_unchanged=unchanged,
        input_inventory=before, snapshot_files_checked=len(m["source_snapshot"]["files"]))
    output.mkdir(parents=True, exist_ok=False)
    if analysis is not None:
        publish(output / "p13-recovered-attribution.json", analysis, new=True)
    publish(output / "p13-recovery-audit.json", result, new=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--approved-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.manifest, args.approved_sha256, args.output)
    print(json.dumps({k: result[k] for k in ("status", "p13_acceptance", "controller_terminal_verified", "errors")}))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
