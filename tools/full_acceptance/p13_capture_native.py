"""P13 controlled Nsight Systems producer, admitted only by a sealed Job worker.

The result of this stage is a *capture*, not a P13 attribution PASS.  Its
post-run supervisor receipt must be verified by the separate offline finalizer.
"""
from pathlib import Path
import os
import subprocess

from .core import record, read, sha


LABELS = [f"hyso.p13/forward/{i}" for i in range(3)]


def capture_command(nsys, python, worker, manifest, root):
    """Fixed argv; no user-supplied command, shell or script entrypoint."""
    return [str(nsys), "profile", "--trace=cuda,nvtx", "--sample=none",
            "--cpuctxsw=none", "--force-overwrite=false",
            "--output=" + str(root / "p13-capture"),
            str(python), "-I", "-S", "-B", str(worker),
            "--manifest", str(manifest), "--output", str(root)]


def export_command(nsys, root):
    return [str(nsys), "export", "--type=sqlite",
            "--output=" + str(root / "p13-export.sqlite"),
            str(root / "p13-capture.nsys-rep")]


def _execute(argv, root, name, seconds):
    with (root / (name + ".stdout.log")).open("xb") as out, (root / (name + ".stderr.log")).open("xb") as err:
        try:
            completed = subprocess.run(argv, cwd=root, stdout=out, stderr=err,
                                       timeout=seconds, check=False)
        except subprocess.TimeoutExpired as exc:
            from .handlers import UnsafeCaptureError
            raise UnsafeCaptureError("P13_nested_nsys_timeout") from exc
        from .safety import native_exit
        if native_exit(completed.returncode):
            from .handlers import UnsafeCaptureError
            raise UnsafeCaptureError("nested_native_crash")
        return completed.returncode


def run(m, root):
    """Only the acceptance-owned stage worker may invoke this function."""
    from .handlers import result, save
    import os
    if os.environ.get("HYSO_FULL_ACCEPTANCE_ADMITTED") != sha(Path(m["output"]) / "manifest.json"):
        raise PermissionError("P13 worker requires sealed admission")
    e = next(x for x in m["experiments"] if x["id"] == "06.p13_capture")
    if e["authorization"]["approved"] is not True:
        raise PermissionError("P13 native experiment NOT_AUTHORIZED")
    plan = m.get("optional", {}).get("p13_nsys_plan", {}).get("configuration", {})
    tool = plan.get("tool")
    if not isinstance(tool, dict) or not tool.get("path") or not tool.get("sha256"):
        raise ValueError("missing pinned P13 Nsight Systems")
    nsys = Path(tool["path"])
    if not nsys.is_file() or sha(nsys) != tool["sha256"] or "Nsight Systems" not in str(nsys) or nsys.name.lower() != "nsys.exe":
        raise ValueError("Nsight Systems tool mismatch")
    if plan.get("protocol") != "trained_"+m.get("model_run",{}).get("arm","B")+"_3_forward_module_nvtx":
        raise ValueError("unrecognized P13 native protocol")
    manifest = Path(m["output"]) / "manifest.json"
    worker = Path(m["source_snapshot"]["root"]) / "tools/full_acceptance/p13_capture_worker.py"
    if not worker.is_file() or sha(worker) != m["source_snapshot"]["files"].get("tools/full_acceptance/p13_capture_worker.py"):
        raise ValueError("P13 worker source not pinned")
    if not m["hardware"].get("gpu_uuid"):
        raise ValueError("P13 explicit physical GPU UUID missing")
    root = Path(root)
    profile = capture_command(nsys, m["runtime"]["python"]["path"], worker, manifest, root)
    code = _execute(profile, root, "nsys-profile", 480)
    if code != 0:
        from .handlers import UnsafeCaptureError
        raise UnsafeCaptureError(f"P13_nsys_profile_nonzero_exit_{code}")
    report = root / "p13-capture.nsys-rep"
    observed = root / "p13-worker.json"
    if not report.is_file() or not observed.is_file():
        raise ValueError("P13 native report or worker result missing")
    worker_row = read(observed)
    if (worker_row.get("status") != "completed" or worker_row.get("forward_labels") != LABELS
            or worker_row.get("output_equal") != [True] * 3
            or worker_row.get("gpu_uuid") != m["hardware"]["gpu_uuid"]
            or worker_row.get("checkpoint_sha256") != m["assets"]["models"][m.get("model_run",{}).get("arm","B")]["sha256"]
            or worker_row.get("worker_source_sha256") != sha(worker)
            or worker_row.get("cudnn_tf32_disabled") is not True):
        raise ValueError("P13 worker identity/output validation failed")
    export = export_command(nsys, root)
    export_code = _execute(export, root, "nsys-export", 90)
    if export_code != 0 or not (root / "p13-export.sqlite").is_file():
        raise RuntimeError(f"P13 SQLite export failed: exit {export_code}")
    from hyso.performance import analyze_nsys_modules
    import sqlite3
    sqlite = root / "p13-export.sqlite"
    if sqlite.stat().st_size > 256 * 1024**2:
        raise ValueError("P13 SQLite export exceeds reader budget")
    # The encoded process identity must come from the exported NVTX activity.
    with sqlite3.connect(sqlite.as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        matches = db.execute("""SELECT DISTINCT globalTid FROM NVTX_EVENTS
                              WHERE text IN (?, ?, ?)""", LABELS).fetchall()
        if not matches:  # Some Nsight exports store NVTX strings via textId.
            matches = db.execute("""SELECT DISTINCT n.globalTid FROM NVTX_EVENTS n
                             JOIN StringIds s ON s.id=n.textId
                             WHERE s.value IN (?, ?, ?)""", LABELS).fetchall()
    globals_ = {int(row[0]) & ~0xffffff for row in matches if row[0] is not None}
    if len(globals_) != 1:
        raise ValueError("P13 forward process identity absent/ambiguous")
    gp = globals_.pop()
    if ((gp >> 24) & 0xffffff) != worker_row.get("pid"):
        raise ValueError("P13 SQLite process identity differs from worker")
    cfg = {"sqlite": record(sqlite), "report": record(report),
           "global_pid": gp, "forward_labels": LABELS}
    analysed = analyze_nsys_modules(cfg)
    if analysed["attribution_complete"] is not True:
        raise ValueError("P13 capture attribution incomplete; no supervisor receipt")
    evidence = [record(observed), record(root / "p13-input.npy"), record(report), record(sqlite),
                save(root, "p13-capture-configuration-preliminary", cfg),
                save(root, "p13-capture-analysis-preliminary", {
                    "attribution_complete": analysed["attribution_complete"],
                    "mapped_kernel_count": analysed["mapped_kernel_count"],
                    "measured_kernel_count": analysed["measured_kernel_count"],
                    "status": analysed["status"], "capture_provenance": "pending_outer_supervisor_cleanup"
                })]
    return result("PASS", criteria={"P13_native_capture_only": True,
                                     "P13_verified_module_attribution": False},
                  reason="capture_complete; final supervisor provenance and 06.layer acceptance pending",
                  evidence=evidence, nsys_exit_code=code, nsys_export_exit_code=export_code)
