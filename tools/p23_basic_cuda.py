"""Owned basic-only CUDA acceptance. No forensic permission/PID/freshness gates.

Controller is stdlib-only. Each explicit invocation validates one model once;
normal replay, isolated peak validation and controlled Python failure are finite
acceptance cases, not retry or native crash reproduction.
"""
import argparse
import ctypes as c
from ctypes import wintypes as w
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p23_basic_resources as resources


def publish(path, data):
    raw = json.dumps(data, indent=2, allow_nan=False).encode()
    if len(raw) > 2*1024**2:
        raise ValueError("bounded controller artifact exceeded")
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def job_module(repo):
    spec = importlib.util.spec_from_file_location("basic_owned_job", repo/"hyso/performance/_training_process.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def limits(handle):
    class IO(c.Structure):
        _fields_ = [(n, c.c_ulonglong) for n in ("ro", "wo", "oo", "rb", "wb", "ob")]
    class Basic(c.Structure):
        _fields_ = [("pt", c.c_longlong), ("jt", c.c_longlong), ("flags", w.DWORD),
            ("minws", c.c_size_t), ("maxws", c.c_size_t), ("processes", w.DWORD),
            ("affinity", c.c_size_t), ("priority", w.DWORD), ("schedule", w.DWORD)]
    class Extended(c.Structure):
        _fields_ = [("basic", Basic), ("io", IO), ("process_memory", c.c_size_t),
            ("job_memory", c.c_size_t), ("peak_process", c.c_size_t), ("peak_job", c.c_size_t)]
    api = c.WinDLL("kernel32", use_last_error=True)
    api.QueryInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p]
    value = Extended()
    if not api.QueryInformationJobObject(handle, 9, c.byref(value), c.sizeof(value), None):
        raise OSError(c.get_last_error(), "basic Job limit readback")
    return dict(flags=value.basic.flags, process_bytes=value.process_memory,
                job_bytes=value.job_memory, processes=value.basic.processes)


def worker(config_path):
    config = read(config_path)
    output = Path(config["output"])
    end = time.monotonic()+15
    while not (output/"admit.json").exists():
        if time.monotonic() > end:
            raise TimeoutError("owned Job admission not received")
        time.sleep(.01)
    admit = read(output/"admit.json")
    assert (admit["pid"], admit["parent_pid"], admit["token"]) == (os.getpid(), os.getppid(), config["token"])
    api = c.WinDLL("kernel32", use_last_error=True)
    api.GetCurrentProcess.restype = w.HANDLE
    api.IsProcessInJob.argtypes = [w.HANDLE, w.HANDLE, c.POINTER(w.BOOL)]
    api.CloseHandle.argtypes = [w.HANDLE]
    member = w.BOOL()
    assert api.IsProcessInJob(api.GetCurrentProcess(), admit["query_handle"], c.byref(member)) and member.value
    assert limits(admit["query_handle"]) == admit["limits"]
    api.CloseHandle(admit["query_handle"])
    sys.path[:0] = [config["repository"], str(Path(config["repository"])/"tests"), config["site_packages"]]
    outcome = dict(arm=config["arm"], pid=os.getpid(), token=config["token"], status="FAIL", error=None)
    try:
        import torch
        assert not torch.cuda.is_initialized()
        resident = resources.current_process_memory()
        observation = resources.capture()
        decision = resources.evaluate(observation, config["budget"], resident=resident)
        publish(output/"post-import-resources.json", dict(observation=observation, resident=resident, decision=decision,
            cuda_initialized=False, torch=str(torch.__version__), torch_source=torch.__file__))
        if not decision["eligible"]:
            raise RuntimeError("basic post-import resource stop: "+",".join(decision["blockers"]))
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from p23_basic_validation import validate_pair, validate_failure
        budget = config["budget"]
        child = dict(process_memory_bytes=budget["process_limit_bytes"],
            job_memory_bytes=budget["child_job_limit_bytes"], cuda_memory_fraction=None,
            timeout_seconds=budget["child_timeout_seconds"])
        pair = validate_pair(config["arm"], config["assets"], output/"success", "cuda:0", isolated_limits=child)
        failure = validate_failure(config["arm"], config["assets"], output/"controlled-failure", "cuda:0", isolated_limits=child)
        outcome.update(status="PASS", validation=pair, failure_cleanup=failure,
                       cuda_initialized=torch.cuda.is_initialized(), saved_activation_enabled=False)
    except BaseException as error:
        outcome.update(error=type(error).__name__+": "+str(error)[:4000], traceback=traceback.format_exc()[-16000:])
    finally:
        publish(output/"worker-result.json", outcome)
    return 0 if outcome["status"] == "PASS" else 1


def controller(args):
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir()
    assets = read(args.assets)
    expected = resources.budget(read(args.cpu_profile), assets["models"][args.arm]["parameters"])
    observation = resources.capture()
    decision = resources.evaluate(observation, expected)
    publish(output/"initial-resources.json", dict(observation=observation, budget=expected, decision=decision))
    if not decision["eligible"]:
        publish(output/"controller-result.json", dict(status="RESOURCE_STOP", native_worker_started=False, decision=decision))
        print(json.dumps(dict(arm=args.arm, status="RESOURCE_STOP", decision=decision)), flush=True)
        return 2
    for item in [assets["models"][args.arm], assets["dataset"], assets["split_indices"]]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("basic asset changed: "+item["path"])
    sources = [*sorted((repo/"hyso").rglob("*.py")), Path(__file__), Path(resources.__file__),
               repo/"tests/p23_basic_validation.py", repo/"tests/p23_basic_factories.py"]
    config = dict(arm=args.arm, repository=str(repo), output=str(output), assets=str(args.assets.resolve()),
        budget=expected, token=str(uuid.uuid4()), site_packages=str(Path(sys.executable).parent/"Lib/site-packages"),
        source_hashes={str(p):sha(p) for p in sources}, interpreter=dict(path=sys.executable, sha256=sha(sys.executable)),
        scope="P23 Basic production CUDA only", saved_activation_lifetime=False, automatic_retry=False)
    publish(output/"request.json", config)
    module = job_module(repo)
    job = module._Job(expected["process_limit_bytes"], expected["outer_job_limit_bytes"], 8)
    process, readers = None, []
    logs = dict(stdout=bytearray(), stderr=bytearray())
    counts = dict(stdout=0, stderr=0)
    overflow = threading.Event()
    start = time.monotonic()
    stop, cleanup = None, None
    actual, controller_error = None, None
    assigned = False
    minimum = dict(available_physical=observation["host"]["available_physical"],
                   commit_headroom=observation["host"]["commit_headroom"])
    samples = 0
    def drain(pipe, name):
        with pipe:
            while True:
                block = pipe.read(8192)
                if not block:
                    return
                counts[name] += len(block)
                logs[name].extend(block[:max(0, 65536-len(logs[name]))])
                if counts[name] > 65536:
                    overflow.set()
    try:
        actual = limits(job.handle)
        assert actual == dict(flags=0x2308, process_bytes=expected["process_limit_bytes"],
                              job_bytes=expected["outer_job_limit_bytes"], processes=8)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg",
                   CUDA_VISIBLE_DEVICES="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        process = subprocess.Popen([sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()),
            "--worker", str(output/"request.json")], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW)
        job.assign(process)
        assigned = True
        api = job.api
        api.GetCurrentProcess.restype = w.HANDLE
        api.DuplicateHandle.argtypes = [w.HANDLE, w.HANDLE, w.HANDLE, c.POINTER(w.HANDLE), w.DWORD, w.BOOL, w.DWORD]
        remote = w.HANDLE()
        if not api.DuplicateHandle(api.GetCurrentProcess(), job.handle, int(process._handle), c.byref(remote), 4, False, 0):
            raise OSError(c.get_last_error(), "owned Job query handle")
        for name in logs:
            thread = threading.Thread(target=drain, args=(getattr(process, name), name), daemon=True)
            thread.start()
            readers.append(thread)
        publish(output/"admit.json", dict(pid=process.pid, parent_pid=os.getpid(), token=config["token"],
            limits=actual, query_handle=remote.value))
        publish(output/"execution-started.json", dict(pid=process.pid, time_ns=time.time_ns(), automatic_retry=False))
        while process.poll() is None:
            current = resources.host_snapshot()
            samples += 1
            for name in minimum:
                minimum[name] = min(minimum[name], current[name])
            if current["low_memory"] or current["available_physical"] < decision["host_margin_bytes"] or current["commit_headroom"] < decision["commit_margin_bytes"]:
                stop = "OS_memory_safety_margin_reached"
            if time.monotonic()-start >= expected["outer_timeout_seconds"]:
                stop = "outer_timeout"
            if overflow.is_set():
                stop = "bounded_log_limit"
            if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > 64*1024**2:
                stop = "bounded_artifact_limit"
            if stop:
                break
            time.sleep(.1)
    except BaseException as error:
        stop = "controller_error"
        controller_error = type(error).__name__+": "+str(error)
    finally:
        try:
            job.terminate()
            if process is not None:
                if not assigned and process.poll() is None:
                    process.kill()  # Only our Popen if Job assignment failed.
                process.wait(timeout=5)
            end = time.monotonic()+5
            while job.observation()["active_processes"] and time.monotonic() < end:
                time.sleep(.01)
            cleanup = job.observation()
            for thread in readers:
                thread.join(timeout=3)
            if process is not None and not readers:
                process.stdout.close()
                process.stderr.close()
            assert cleanup["active_processes"] == 0 and not any(t.is_alive() for t in readers)
        finally:
            job.close()
    result = read(output/"worker-result.json") if (output/"worker-result.json").exists() else None
    preserved = all(sha(p) == value for p, value in config["source_hashes"].items())
    exit_code = process.returncode if process is not None else None
    passed = stop is None and exit_code == 0 and result is not None and result["status"] == "PASS" and preserved
    receipt = dict(arm=args.arm, status="PASS" if passed else "STOP", monitor_stop=stop,
        exit_code=exit_code, worker=result, cleanup=cleanup, limits=actual, controller_error=controller_error,
        source_preservation=preserved, elapsed_seconds=time.monotonic()-start,
        memory_monitor=dict(samples=samples, minimum=minimum), log_bytes=counts,
        logs={k:bytes(v).decode(errors="replace") for k,v in logs.items()}, automatic_retry=False)
    publish(output/"controller-result.json", receipt)
    print(json.dumps({k:receipt[k] for k in ("arm", "status", "monitor_stop", "exit_code", "cleanup", "elapsed_seconds")}), flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--arm", choices=("A", "B"))
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--cpu-profile", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.worker and any(getattr(args, name) is None for name in ("arm", "assets", "cpu_profile", "output")):
        parser.error("--arm, --assets, --cpu-profile and --output are required")
    return worker(args.worker) if args.worker else controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
