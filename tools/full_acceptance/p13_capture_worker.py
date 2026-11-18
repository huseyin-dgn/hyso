"""Exactly three trained Model B CUDA forwards; invoked only under Nsight Systems.

Never modifies the checkpoint or source model. Observation hooks are transient.
"""
from pathlib import Path
import argparse
import hashlib
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    root = args.output.resolve()
    snapshot = Path(__file__).resolve().parents[2]
    # -I -S prevents ambient PYTHONPATH/site injection; use only manifest site paths.
    import json
    m = json.loads(manifest.read_text(encoding="utf-8"))
    if Path(m["source_snapshot"]["root"]).resolve() != snapshot:
        raise PermissionError("P13 capture worker source snapshot mismatch")
    sys.path[:0] = [str(snapshot), str(snapshot / "tools"), *m["runtime"]["site_paths"]]
    from tools.full_acceptance.core import read, publish, sha, selected_arm
    if (os.environ.get("HYSO_FULL_ACCEPTANCE_ADMITTED") != sha(manifest)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != m["hardware"]["gpu_uuid"]):
        raise PermissionError("P13 capture worker not GPU-admitted")
    request = read(root / "request.json")
    ready = read(root / "ready.json")
    if (request.get("experiment_id") != "06.p13_capture"
            or request.get("operation") != "p13_capture"
            or request.get("manifest_sha256") != sha(manifest)
            or ready.get("gate", {}).get("exact_job_membership") is not True
            or ready.get("gate", {}).get("kill_on_close") is not True):
        raise PermissionError("P13 stage Job admission unverified")
    if os.name != "nt":
        raise PermissionError("P13 native capture requires Windows owned Job")
    import ctypes as c
    from ctypes import wintypes as w
    api = c.WinDLL("kernel32", use_last_error=True)
    api.GetCurrentProcess.restype = w.HANDLE
    api.IsProcessInJob.argtypes = [w.HANDLE, w.HANDLE, c.POINTER(w.BOOL)]
    api.IsProcessInJob.restype = w.BOOL
    member = w.BOOL()
    if not api.IsProcessInJob(api.GetCurrentProcess(), None, c.byref(member)) or not member.value:
        raise PermissionError("P13 Nsight child is outside Windows Job")
    # Only now import Torch/CUDA; the outer worker verified exact Job identity.
    from tools.full_acceptance.workloads import model_pair
    from tools.full_acceptance.p13_capture_scopes import ModuleNvtxScopes
    from hyso._evaluation_evidence import model_hash
    from hyso.performance.devices import _cuda_hardware
    import torch
    import numpy as np
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU fallback")
    torch.cuda.set_device(0)
    hardware = _cuda_hardware(torch, m["hardware"]["gpu_uuid"])
    # GPU identity is checked by _cuda_hardware; no inferred device equivalence.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    arm = selected_arm(m)
    model, sample, origin = model_pair(manifest, arm, batch=1)
    initial_hash = model_hash(model)
    input_path = root / "p13-input.npy"
    if input_path.exists():
        raise FileExistsError(input_path)
    np.save(input_path, sample.numpy(), allow_pickle=False)
    import copy
    model_gpu = copy.deepcopy(model).to("cuda:0").eval()
    input_gpu = sample.to("cuda:0")
    with torch.inference_mode():
        for _ in range(2):
            baseline = model_gpu(input_gpu)
            torch.cuda.synchronize()
        baseline = baseline.detach().cpu().clone()
        results = []
        with ModuleNvtxScopes(model_gpu, torch.cuda.nvtx) as scopes:
            for index in range(3):
                value = scopes.measured_forward(index, lambda: model_gpu(input_gpu), torch.cuda.synchronize)
                equal = torch.equal(value.detach().cpu(), baseline)
                if not equal:
                    raise ValueError(f"P13 output drift on forward {index}")
                results.append(equal)
    if model_hash(model) != initial_hash:
        raise ValueError("P13 original CPU master state changed")
    cp = m["assets"]["models"][arm]
    if sha(cp["path"]) != cp["sha256"]:
        raise ValueError("P13 checkpoint modified")
    publish(root / "p13-worker.json", dict(status="completed", pid=os.getpid(),
        forward_labels=[f"hyso.p13/forward/{i}" for i in range(3)],
        output_equal=results, cudnn_tf32_disabled=True,
        gpu_uuid=m["hardware"]["gpu_uuid"], checkpoint_sha256=cp["sha256"],
        input_sha256=sha(input_path), worker_source_sha256=sha(__file__),
        input_shape=list(sample.shape), input_dtype=str(sample.dtype),
        original_state_sha256=initial_hash, preprocessing=m["assets"]["dataset"]["preprocessing"],
        captured_gpu_identity=hardware, module_paths=list(scopes._paths or [])),new=True)


if __name__ == "__main__":
    main()
