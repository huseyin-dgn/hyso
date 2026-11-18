"""Stdlib bootstrap; no HYSo/Torch import until parent Job admission."""
import importlib
import json
import os
from pathlib import Path
import sys
import time


def _publish(path,data,limit):
    payload=json.dumps(data,allow_nan=False,separators=(",",":")).encode("utf-8")
    if len(payload)>limit:raise OverflowError("worker result exceeds publication budget")
    temporary=path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def main():
    root=Path(sys.argv[1]);request=json.loads((root/"request.json").read_text(encoding="utf-8"))
    deadline=time.monotonic()+15
    while not (root/"admit.json").exists() and time.monotonic()<deadline:time.sleep(.01)
    admit=json.loads((root/"admit.json").read_text(encoding="utf-8"))
    if admit!={"token":request["token"],"pid":os.getpid()}:raise RuntimeError("invalid OS admission")
    result=dict(token=request["token"],pid=os.getpid(),status="workload_exception",reason=None,measurement=None)
    _publish(root/"ready.json",dict(token=request["token"],pid=os.getpid(),phase="job_admitted_before_import"),4096)
    sys.path[:0]=[request["package_root"],*request["runtime_paths"]]
    try:
        module=importlib.import_module("hyso.performance.isolated_training_memory")
        result["measurement"]=module._run_recipe(request)
        result["status"]="success"
    except BaseException as error:
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
        _publish(root/"result.json",result,request["protocol"]["max_result_bytes"])
    except OverflowError:
        result.update(status="resource_limit",reason="bounded result publication rejected",measurement=None)
        _publish(root/"result.json",result,request["protocol"]["max_result_bytes"])


if __name__=="__main__":main()
