"""Test-only, bounded durable phases around the real CPU worker; no clock reset.

The production Job/launcher/deadline remain unchanged. This wrapper adds I/O
and import/function observation, so its timings are instrumented observations.
"""
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
import time


def append(path, phase, **fields):
    row = dict(phase=phase, pid=os.getpid(), monotonic_ns=time.monotonic_ns(), **fields)
    payload = (json.dumps(row, allow_nan=False) + "\n").encode("utf-8")
    # Each process owns one file; append/fsync completes before the observed call.
    if path.exists() and path.stat().st_size + len(payload) > 65536:
        raise OverflowError("test phase evidence exceeds 64KiB")
    with path.open("ab", buffering=0) as stream:
        if stream.write(payload) != len(payload):
            raise OSError("incomplete test phase write")
        os.fsync(stream.fileno())


def capture_launch(monkeypatch, output):
    """Observe one real launcher without changing protocol budgets or its timer."""
    from hyso.performance import _training_process as process
    from hyso.performance import isolated_training_memory as isolated
    original_popen = process.subprocess.Popen
    original_assign = process._Job.assign
    original_launch = isolated._launch
    parent = output.with_suffix(".parent.jsonl")
    child = output.with_suffix(".child.jsonl")
    launches = []

    def popen(argv, *args, **kwargs):
        if len(argv) >= 6 and Path(argv[-1]).is_dir() and Path(argv[-2]).name == "_training_memory_worker.py":
            append(parent, "process_launch_enter", worker=str(argv[-2]))
            routed = [*argv[:-2], str(Path(__file__).resolve()), argv[-1], argv[-2], str(child)]
            worker = original_popen(routed, *args, **kwargs)
            launches.append(worker.pid)
            append(parent, "process_launch_returned", worker_pid=worker.pid)
            return worker
        return original_popen(argv, *args, **kwargs)

    def assign(job, worker):
        original_assign(job, worker)
        if worker.pid in launches:
            append(parent, "job_assignment_completed", worker_pid=worker.pid)

    evidence = dict(parent=parent, child=child, launches=launches, ready=None)

    def launch(*args, **kwargs):
        receipt, paths = original_launch(*args, **kwargs)
        if paths["ready"].exists():
            evidence["ready"] = process._bounded_read(paths["ready"], 4096)
        append(parent, "launcher_receipt_observed", worker_pid=receipt["launcher_pid"],
               monitor_outcome=receipt["monitor_outcome"], active_processes=receipt["job"]["active_processes"])
        return receipt, paths

    monkeypatch.setattr(isolated, "_launch", launch)
    monkeypatch.setattr(process.subprocess, "Popen", popen)
    monkeypatch.setattr(process._Job, "assign", assign)
    return evidence


def rows(path):
    if not path.exists():
        return []
    data = path.read_bytes()
    if len(data) > 65536:
        raise ValueError("oversized phase evidence")
    # A killed partial last write is not a completed durable event.
    return [json.loads(line) for line in data.splitlines(keepends=True) if line.endswith(b"\n")]


def main():
    root, actual_worker, output = map(Path, sys.argv[1:4])
    append(output, "python_bootstrap_enter")
    request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    assert request["recipe"]["device"] == "cpu"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    deadline = time.monotonic() + 15
    while not (root / "admit.json").exists() and time.monotonic() < deadline:
        time.sleep(.01)
    admit = json.loads((root / "admit.json").read_text(encoding="utf-8"))
    assert admit == dict(token=request["token"], pid=os.getpid())
    append(output, "job_admission_observed")
    original_import = importlib.import_module
    records = []

    def profile(frame, event, arg):
        code = frame.f_code
        if event not in ("call", "return"):
            return
        if code.co_name not in ("tiny", "__init__", "fit", "forward_hook"):
            return
        filename = code.co_filename.replace("\\", "/").rsplit("/", 1)[-1]
        key = (filename, code.co_name)
        names = {("training_memory_factories.py", "tiny"): "factory",
                 ("trainer.py", "__init__"): "trainer_initialization",
                 ("dataloader.py", "__init__"): "loader_initialization",
                 ("trainer.py", "fit"): "trainer_fit",
                 ("training_memory_factories.py", "forward_hook"): "forward_hook"}
        if key in names:
            if len(records) >= 32:
                raise OverflowError("test phase record budget exceeded")
            records.append(key)
            append(output, names[key] + ("_enter" if event == "call" else "_returned"))

    def observed_import(name, *args, **kwargs):
        if name != "hyso.performance.isolated_training_memory":
            return original_import(name, *args, **kwargs)
        append(output, "hyso_torch_import_enter")
        module = original_import(name, *args, **kwargs)
        torch = sys.modules["torch"]
        assert str(torch.__version__) == request["torch_version"]
        assert not torch.cuda.is_initialized()
        append(output, "hyso_torch_import_completed", torch=str(torch.__version__), cuda_build=torch.version.cuda)
        sys.setprofile(profile)
        return module

    importlib.import_module = observed_import
    sys.argv = [str(actual_worker), str(root)]
    try:
        runpy.run_path(str(actual_worker), run_name="__main__")
    finally:
        sys.setprofile(None)
        importlib.import_module = original_import


if __name__ == "__main__":
    main()
