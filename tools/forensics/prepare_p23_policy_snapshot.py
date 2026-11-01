"""Prepare a separate, unapproved v3 diagnostic snapshot. Never execute it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import time
import uuid


R2_HASH = "2be2ebe1562bfc0863b16dc6d1bb093f91f32f1dd9ab8b456ed94387cedbd78d"
OLD_GATE = """assert preflight['run_id']==seal['run_id'] and preflight['passed'] is True
assert 0<=time.time_ns()-preflight['checked_time_ns']<=30*10**9
assert preflight['host_free_bytes']>=4*1024**3 and preflight['disk_free_bytes']>=12*1024**3
assert len(preflight['gpu_utilization_percent_samples'])==3 and max(preflight['gpu_utilization_percent_samples'])<35
assert preflight['gpu_free_mib']>=2048 and preflight['conflicting_workloads']==[]"""
NEW_GATE = """# Bind the new policy module before importing it. Utilization is telemetry.
manifest_bytes=(RUN/'manifest.json').read_bytes()
assert hashlib.sha256(manifest_bytes).hexdigest()==seal['manifest_sha256']
manifest=json.loads(manifest_bytes)
assert hashlib.sha256((ROOT/'p23_preflight.py').read_bytes()).hexdigest()==manifest['source_files']['p23_preflight.py']
from p23_preflight import evaluate
policy_decision=evaluate(preflight,run_id=seal['run_id'],manifest_sha256=seal['manifest_sha256'])
assert policy_decision['eligible'],policy_decision['blockers']"""


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def verify_r2(snapshot):
    if digest(snapshot / "manifest.json") != R2_HASH:
        raise ValueError("r2 manifest identity mismatch")
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    seal = json.loads((snapshot / "seal.json").read_text(encoding="utf-8"))
    if (seal["manifest_sha256"] != R2_HASH or seal["run_id"] != manifest["run_id"]
            or seal["interpreter"] != "C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe"):
        raise ValueError("r2 seal identity mismatch")
    for relative, expected in manifest["source_files"].items():
        path = snapshot / "source" / relative
        if path.is_symlink() or not path.resolve().is_relative_to((snapshot / "source").resolve()):
            raise ValueError("source escaped snapshot")
        if digest(path) != expected:
            raise ValueError("r2 source identity mismatch: " + relative)
    for filename, expected in manifest["runtime_files"].items():
        if digest(Path(filename)) != expected:
            raise ValueError("runtime identity mismatch: " + filename)
    return manifest


def prepare(snapshot, output):
    snapshot, output = snapshot.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(snapshot):
        raise ValueError("new output must be absent and outside r2")
    original = verify_r2(snapshot)
    controller = (snapshot / "source/forensic_controller.py").read_text(encoding="utf-8")
    if controller.count(OLD_GATE) != 1:
        raise ValueError("unexpected sealed controller shape")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    # Copy only declared sources; never inherit evidence, approval or attempt markers.
    for relative in original["source_files"]:
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(snapshot / "source" / relative, destination)
    shutil.copyfile(Path(__file__).with_name("p23_preflight.py"), source / "p23_preflight.py")
    controller = controller.replace(OLD_GATE, NEW_GATE)
    anchor = "diag.verify('prelaunch')"
    if controller.count(anchor) != 1:
        raise ValueError("unexpected identity checkpoint")
    controller = controller.replace(anchor, anchor + "\ndiag.publish('preflight-policy-decision.json',policy_decision)")
    (source / "forensic_controller.py").write_text(controller, encoding="utf-8", newline="\n")
    proposal = json.loads((source / "proposal.json").read_text(encoding="utf-8"))
    proposal.update(preflight_policy="hyso.b4.diagnostic-preflight.v3",
                    utilization_role="telemetry_only_no_admission_threshold",
                    approval_state="NOT_APPROVED", execution_state="NOT_EXECUTED")
    (source / "proposal.json").write_text(json.dumps(proposal, indent=2), encoding="utf-8")
    (source / "APPROVAL-AND-PREFLIGHT.md").write_text(
        "# Prepared v3; NOT APPROVED / NOT EXECUTED\n\n"
        "Separate approval for this new run-ID/manifest is mandatory. r2 approval does not transfer.\n"
        "Collect once: three ordered GPU samples and host RAM/disk plus raw process/compute inventory.\n"
        "Use p23_preflight.py's evidence contract; review every compute PID with a reason.\n"
        "Utilization is telemetry only; this is not a performance comparison.\n"
        "Host >=4GiB, disk >=12GiB, each GPU sample free >=2048MiB, age <=30s from capture start.\n"
        "Unknown/conflicting workload or identity/Job failure: STOP. Never poll for admission.\n"
        "All r2 recipe, Job, memory, timeout, output, reset, guard and single-attempt limits remain.\n"
        "No baseline/prefix/retry. Native crash: STOP, preserve receipts/WER/dumps and verify owned cleanup.\n"
        "Post-crash observation bounded30s; no policy/registry edit or dump upload.\n"
        "Operator-supplied evidence is trusted local input, not adversarial attestation.\n"
        "Do not run the controller until the user explicitly authorizes this new snapshot.\n",
        encoding="utf-8")
    manifest = dict(original, run_id=str(uuid.uuid4()), created_time_ns=time.time_ns(),
                    parent_manifest_sha256=R2_HASH, proposal=proposal)
    manifest["source_files"] = {p.relative_to(source).as_posix(): digest(p)
                                for p in sorted(source.rglob("*")) if p.is_file()}
    payload = json.dumps(manifest, indent=2).encode("utf-8")
    (output / "manifest.json").write_bytes(payload)
    seal = dict(run_id=manifest["run_id"], manifest_sha256=hashlib.sha256(payload).hexdigest(),
                interpreter=json.loads((snapshot / "seal.json").read_text())["interpreter"])
    (output / "seal.json").write_text(json.dumps(seal, indent=2), encoding="utf-8")
    for name in ("worker", "events"):
        (output / name).mkdir()
    for path in list(source.rglob("*")) + [output / "manifest.json", output / "seal.json"]:
        if path.is_file():
            path.chmod(stat.S_IREAD)
    verify_r2(snapshot)
    return seal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.snapshot, args.output), indent=2))
