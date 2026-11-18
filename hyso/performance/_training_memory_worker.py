"""Stdlib bootstrap; no HYSo/Torch import until parent Job admission."""
import importlib
import json
import os
from pathlib import Path
import sys
import time
import uuid


class _PhaseTrace:
    """Private owned-worker CPU breadcrumbs, never measurement evidence.

    Enabled only by a private launcher request. Each bounded record is flushed
    and fsynced; a native fail-fast can still lose the last record. This changes
    timing. I/O faults are latched so tracing cannot interrupt workload cleanup;
    check() prevents a failed diagnostic trace from publishing success.
    """
    _phases = frozenset(("worker_started", "job_admitted", "torch_import", "cuda_setup",
        "factory", "saved_hook_context", "saved_hook_exit", "recorder_cleanup", "activation_cleanup",
        "workload", "result_publication", "forward", "loss_metrics", "backward",
        "scaler_step_attempt", "validation"))

    def __init__(self, config, token, package_root, worker_root):
        if type(config) is not dict or set(config)-{"path", "max_records", "max_bytes"}:
            raise ValueError("invalid private phase trace configuration")
        uuid.UUID(token)
        path = Path(config["path"])
        if not path.is_absolute() or path.is_symlink():
            raise ValueError("phase trace requires an absolute unlinked destination")
        path = path.parent.resolve(strict=True) / path.name
        if any(path.is_relative_to(Path(p).resolve()) for p in (package_root, worker_root)):
            raise ValueError("phase trace must persist outside source and transient worker roots")
        self.max_records = config.get("max_records", 512)
        self.max_bytes = config.get("max_bytes", 65536)
        for value, ceiling in ((self.max_records, 4096), (self.max_bytes, 1048576)):
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError("invalid phase trace budget")
        self.pid, self.token = os.getpid(), token
        self.records = self.bytes = 0
        self.fault = None
        self.handle = path.open("xb")  # Never overwrite prior evidence.

    def emit(self, phase, status, *, timestamp_ns=None):
        if self.fault is not None:
            return
        try:
            if os.getpid() != self.pid or phase not in self._phases or status not in ("begin", "end", "error", "skipped"):
                raise ValueError("invalid CPU phase event/owner")
            row = dict(phase=phase, monotonic_ns=time.monotonic_ns() if timestamp_ns is None else timestamp_ns,
                       pid=self.pid, run_id=self.token, cpu_status=status)
            payload = (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
            if self.records >= self.max_records or self.bytes + len(payload) > self.max_bytes:
                raise OverflowError("phase trace budget exhausted")
            if self.handle.write(payload) != len(payload):
                raise OSError("incomplete diagnostic phase write")
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.records += 1
            self.bytes += len(payload)
        except Exception as error:
            self.fault = type(error)  # No exception/tensor/traceback retention.

    def check(self):
        if self.fault is not None:
            if self.fault is OverflowError:
                raise OverflowError("bounded diagnostic phase trace incomplete")
            raise RuntimeError("diagnostic phase trace I/O or ownership failure")

    def close(self):
        try:
            self.handle.close()
        except Exception as error:
            self.fault = type(error)


def _phase(trace, name, status):
    if trace is not None:
        trace.emit(name, status)


def _trace_hooks(trace, context, enabled):
    if trace is None:
        return context

    class Scope:
        def __enter__(self):
            _phase(trace,"saved_hook_context","begin" if enabled else "skipped")
            try:
                value=context.__enter__()
            except BaseException:
                _phase(trace,"saved_hook_context","error")
                raise
            _phase(trace,"saved_hook_context","end" if enabled else "skipped")
            return value

        def __exit__(self,*args):
            _phase(trace,"saved_hook_exit","begin" if enabled else "skipped")
            try:
                value=context.__exit__(*args)
            except BaseException:
                _phase(trace,"saved_hook_exit","error")
                raise
            _phase(trace,"saved_hook_exit","end" if enabled else "skipped")
            return value

    return Scope()


def _publish(path,data,limit):
    payload=json.dumps(data,allow_nan=False,separators=(",",":")).encode("utf-8")
    if len(payload)>limit:raise OverflowError("worker result exceeds publication budget")
    temporary=path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def main():
    started_ns=time.monotonic_ns()
    root=Path(sys.argv[1]);request=json.loads((root/"request.json").read_text(encoding="utf-8"))
    deadline=time.monotonic()+15
    while not (root/"admit.json").exists() and time.monotonic()<deadline:time.sleep(.01)
    admit=json.loads((root/"admit.json").read_text(encoding="utf-8"))
    if admit!={"token":request["token"],"pid":os.getpid()}:raise RuntimeError("invalid OS admission")
    trace=None
    result=dict(token=request["token"],pid=os.getpid(),status="workload_exception",reason=None,measurement=None)
    _publish(root/"ready.json",dict(token=request["token"],pid=os.getpid(),phase="job_admitted_before_import"),4096)
    sys.path[:0]=[request["package_root"],*request["runtime_paths"]]
    try:
        if "_phase_trace" in request:
            trace=_PhaseTrace(request["_phase_trace"],request["token"],request["package_root"],root)
            # Capture bootstrap time before waiting, persist only after Job admission.
            trace.emit("worker_started","end",timestamp_ns=started_ns)
            trace.emit("job_admitted","end")
        _phase(trace,"torch_import","begin")
        module=importlib.import_module("hyso.performance.isolated_training_memory")
        _phase(trace,"torch_import","end")
        _phase(trace,"workload","begin")
        result["measurement"]=module._run_recipe(request,_phase_trace=trace)
        _phase(trace,"workload","end")
        if trace is not None:trace.check()
        result["status"]="success"
    except BaseException as error:
        _phase(trace,"workload","error")
        result["measurement"]=None
        torch=sys.modules.get("torch")
        if isinstance(error,NotImplementedError):result["status"]="unsupported"
        elif isinstance(error,OverflowError):result["status"]="resource_limit"
        elif isinstance(error,MemoryError) or (torch is not None and isinstance(error,torch.cuda.OutOfMemoryError)):
            result["status"]="oom"
        elif (torch is not None and isinstance(error,RuntimeError) and "DefaultCPUAllocator: not enough memory" in str(error)):
            result["status"]="oom"
        elif type(error).__name__=="ResourceLimitError" and type(error).__module__=="hyso.performance._training_process":
            result["status"]="resource_limit"
        elif type(error).__name__=="TrainingMemoryError" and type(error).__module__=="hyso.performance.training_memory":
            result["status"]="measurement_error"
        result["reason"]=(type(error).__name__+": "+str(error))[:2000]
    try:
        _phase(trace,"result_publication","begin")
        if trace is not None and result["status"]=="success":trace.check()
        try:
            _publish(root/"result.json",result,request["protocol"]["max_result_bytes"])
        except OverflowError:
            result.update(status="resource_limit",reason="bounded result publication rejected",measurement=None)
            _publish(root/"result.json",result,request["protocol"]["max_result_bytes"])
        _phase(trace,"result_publication","end")
    finally:
        if trace is not None:
            trace.close()
            # A post-publication fault must produce a nonzero OS exit, even if
            # result.json already says success. Parent exit classification wins.
            if result["status"]=="success":trace.check()


if __name__=="__main__":main()
