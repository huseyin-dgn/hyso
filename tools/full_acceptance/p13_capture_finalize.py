"""P13 post-supervisor provenance binding. No model/CUDA/NSYS invocation.

Run only after the P13 native capture's outer acceptance controller has stopped.
Original run, source, checkpoint and previous evidence remain read-only.
"""
from pathlib import Path
import argparse
import json
import sys

# core.verify uses the existing full_acceptance.registry import contract.
# Keep this read-only offline tool usable as `python -m tools.full_acceptance...`.
_toolroot = str(Path(__file__).resolve().parents[1])
if _toolroot not in sys.path:
    sys.path.insert(0, _toolroot)

from .core import read, record, publish, sha, verify
from .p13_capture_native import LABELS


def build(manifest, expected_sha, output):
    manifest = Path(manifest).resolve()
    m = verify(manifest, expected=expected_sha, live=True)
    prior = Path(m["output"]).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("P13 finalized evidence requires a new directory")
    if output == prior or prior in output.parents:
        raise ValueError("do not write into consumed acceptance run")
    state = read(prior / "run-status.json")
    if (state.get("controller_active") is not False or state.get("preservation_status") != "PASS"
            or state.get("manifest_sha256") != sha(manifest)
            or state.get("finalization_errors")):
        raise ValueError("P13 outer acceptance did not finalize cleanly")
    rows = read(prior / "stage-results.json")
    row, = [x for x in rows["experiments"] if x["id"] == "06.p13_capture"]
    authorization, = [e["authorization"] for e in m["experiments"] if e["id"] == "06.p13_capture"]
    sup = row.get("supervisor", {})
    cleanup = sup.get("cleanup", {})
    if (row.get("status") != "PASS" or row.get("attempted") is not True
            or row.get("criteria", {}).get("P13_native_capture_only") is not True
            or authorization.get("approved") is not True
            or not authorization.get("basis")
            or sup.get("owned_job_only") is not True or sup.get("native_crash") is not False
            or sup.get("exit_code") != 0 or sup.get("reason") is not None
            or type(cleanup.get("active_processes")) is not int or cleanup["active_processes"] != 0):
        raise ValueError("P13 native capture or exact owned Job supervisor did not pass")
    stage = prior / "stages" / "06.p13_capture"
    stage_result = read(stage / "result.json")
    ready = read(stage / "ready.json")
    if (ready.get("gate", {}).get("exact_job_membership") is not True
            or ready.get("gate", {}).get("kill_on_close") is not True
            or stage_result.get("nsys_exit_code") != 0
            or stage_result.get("nsys_export_exit_code") != 0
            or stage_result.get("manifest_sha256") != sha(manifest)
            or stage_result.get("id") != "06.p13_capture"):
        raise ValueError("P13 capture admission or Nsight subprocess result unverified")
    # Compare each capture artifact with the hashes sealed in the worker's
    # terminal result and retained again in the controller stage-results row.
    refs = {r["path"]: r["sha256"] for r in stage_result.get("evidence", [])}
    for artifact in ("p13-worker.json", "p13-input.npy", "p13-capture.nsys-rep",
                     "p13-export.sqlite", "p13-capture-configuration-preliminary.json",
                     "p13-capture-analysis-preliminary.json"):
        path = stage / artifact
        if not path.is_file() or refs.get(str(path)) != sha(path):
            raise ValueError("P13 terminal capture evidence hash mismatch: " + artifact)
    recorded = {r["path"]: r["sha256"] for r in row.get("evidence", [])}
    if refs != recorded:
        raise ValueError("P13 controller/worker capture evidence mismatch")
    worker = read(stage / "p13-worker.json")
    if (worker.get("status") != "completed" or type(worker.get("pid")) is not int
            or worker.get("output_equal") != [True, True, True]
            or worker.get("forward_labels") != LABELS
            or worker.get("cudnn_tf32_disabled") is not True
            or worker.get("checkpoint_sha256") != m["assets"]["models"]["B"]["sha256"]
            or worker.get("gpu_uuid") != m["hardware"]["gpu_uuid"]):
        raise ValueError("P13 native worker identity/output unverified")
    source = Path(m["source_snapshot"]["root"]) / "tools/full_acceptance/p13_capture_worker.py"
    if (sha(source) != worker.get("worker_source_sha256")
            or worker.get("input_sha256") != sha(stage / "p13-input.npy")):
        raise ValueError("P13 worker/input artifacts changed")
    cfg = read(stage / "p13-capture-configuration-preliminary.json")
    if (cfg.get("forward_labels") != LABELS
            or cfg.get("sqlite", {}).get("path") != str(stage / "p13-export.sqlite")
            or cfg.get("report", {}).get("path") != str(stage / "p13-capture.nsys-rep")):
        raise ValueError("P13 preliminary configuration routing mismatch")
    gp = cfg["global_pid"]
    if type(gp) is not int or ((gp >> 24) & 0xffffff) != worker["pid"]:
        raise ValueError("P13 encoded process/worker identity mismatch")
    # Revalidate raw data independently before binding new supervisor evidence.
    from hyso.performance import analyze_nsys_modules
    preliminary = analyze_nsys_modules(cfg)
    if preliminary.get("attribution_complete") is not True:
        raise ValueError("P13 incomplete measured kernel attribution")
    receipt = dict(format="hyso.performance.p13.nsys_capture", version=1,
        status="completed", global_pid=gp, worker_pid=worker["pid"],
        forward_labels=LABELS,
        sqlite_sha256=cfg["sqlite"]["sha256"], report_sha256=cfg["report"]["sha256"],
        nsys_exit_code=stage_result["nsys_exit_code"], output_equal=worker["output_equal"],
        cudnn_tf32_disabled=True, gpu_uuid=worker["gpu_uuid"],
        authorization=dict(approved=True, basis=authorization["basis"]),
        supervisor=dict(owned_process_tree=True, process_terminated=True,
                        cleanup=dict(active_processes=cleanup["active_processes"])),
        checkpoint=record(m["assets"]["models"]["B"]["path"]),
        input=record(stage / "p13-input.npy"), worker_source=record(source))
    if receipt["checkpoint"]["sha256"] != m["assets"]["models"]["B"]["sha256"]:
        raise ValueError("P13 checkpoint unexpectedly changed")
    output.mkdir(parents=True)
    publish(output / "p13-capture-receipt.json", receipt, new=True)
    final_cfg = dict(cfg, capture=record(output / "p13-capture-receipt.json"))
    complete = analyze_nsys_modules(final_cfg)
    if complete.get("status") != "complete" or complete.get("provenance_verified") is not True:
        raise ValueError("P13 final capture evidence rejected; do not reuse")
    publish(output / "p13-evidence-configuration.json", final_cfg, new=True)
    publish(output / "p13-finalization.json", dict(status="PROVENANCE_VALIDATED_NOT_06_LAYER_ACCEPTANCE",
        manifest_sha256=sha(manifest), supervisor_receipt=record(stage / "supervisor-receipt.json"),
        capture_receipt=record(output / "p13-capture-receipt.json"),
        configuration=record(output / "p13-evidence-configuration.json"),
        mapped_kernel_count=complete["mapped_kernel_count"],
        measured_kernel_count=complete["measured_kernel_count"]), new=True)
    return final_cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--approved-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = build(args.manifest, args.approved_sha256, args.output)
    print(json.dumps(dict(status="P13_CAPTURE_PROVENANCE_VALIDATED",
        p13_acceptance="PENDING_06_LAYER",
        p13_evidence=str(args.output.resolve() / "p13-evidence-configuration.json"),
        worker_pid=(cfg["global_pid"] >> 24) & 0xffffff)))


if __name__ == "__main__":
    main()
