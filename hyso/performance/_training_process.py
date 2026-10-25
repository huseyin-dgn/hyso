"""Windows stdlib process ownership for trusted training recipes, not a sandbox."""
import ctypes as c
from ctypes import wintypes as w
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


class ResourceLimitError(RuntimeError):
    pass


class _Job:
    def __init__(self, process_bytes, job_bytes, processes):
        if os.name != "nt":
            raise NotImplementedError("native isolation currently verified on Windows only")
        class IO(c.Structure):
            _fields_ = [(n,c.c_ulonglong) for n in ("ro","wo","oo","rb","wb","ob")]
        class Basic(c.Structure):
            _fields_ = [("pt",c.c_longlong),("jt",c.c_longlong),("flags",w.DWORD),
                ("minws",c.c_size_t),("maxws",c.c_size_t),("processes",w.DWORD),
                ("affinity",c.c_size_t),("priority",w.DWORD),("schedule",w.DWORD)]
        class Extended(c.Structure):
            _fields_ = [("basic",Basic),("io",IO),("process_memory",c.c_size_t),
                ("job_memory",c.c_size_t),("peak_process",c.c_size_t),("peak_job",c.c_size_t)]
        class Accounting(c.Structure):
            _fields_ = [(n,c.c_longlong) for n in ("user","kernel","period_user","period_kernel")] + [
                (n,w.DWORD) for n in ("page_faults","total","active","terminated")]
        api = c.WinDLL("kernel32",use_last_error=True)
        api.CreateJobObjectW.argtypes=[c.c_void_p,w.LPCWSTR];api.CreateJobObjectW.restype=w.HANDLE
        api.SetInformationJobObject.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD];api.SetInformationJobObject.restype=w.BOOL
        api.QueryInformationJobObject.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD,c.c_void_p];api.QueryInformationJobObject.restype=w.BOOL
        api.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE];api.AssignProcessToJobObject.restype=w.BOOL
        api.TerminateJobObject.argtypes=[w.HANDLE,w.UINT];api.TerminateJobObject.restype=w.BOOL
        api.CloseHandle.argtypes=[w.HANDLE];api.CloseHandle.restype=w.BOOL
        self.api,self.Accounting,self.Extended = api,Accounting,Extended
        self.handle = api.CreateJobObjectW(None,None)
        if not self.handle:raise OSError(c.get_last_error(),"CreateJobObjectW")
        info=Extended();info.basic.flags=0x2000|0x8
        info.basic.processes=processes
        # Ownership/cleanup remain mandatory; memory quotas are caller opt-in.
        if process_bytes is not None:
            info.basic.flags|=0x100;info.process_memory=process_bytes
        if job_bytes is not None:
            info.basic.flags|=0x200;info.job_memory=job_bytes
        if not api.SetInformationJobObject(self.handle,9,c.byref(info),c.sizeof(info)):
            error=c.get_last_error();self.close();raise OSError(error,"SetInformationJobObject")

    def assign(self, process):
        if not self.api.AssignProcessToJobObject(self.handle,w.HANDLE(int(process._handle))):
            raise OSError(c.get_last_error(),"AssignProcessToJobObject")

    def observation(self):
        accounting=self.Accounting();memory=self.Extended()
        for kind,value in ((1,accounting),(9,memory)):
            if not self.api.QueryInformationJobObject(self.handle,kind,c.byref(value),c.sizeof(value),None):
                raise OSError(c.get_last_error(),"QueryInformationJobObject")
        return dict(total_processes=accounting.total,active_processes=accounting.active,
                    peak_process_commit_bytes=memory.peak_process,peak_job_commit_bytes=memory.peak_job)

    def terminate(self):
        if not self.api.TerminateJobObject(self.handle,0):
            raise OSError(c.get_last_error(),"TerminateJobObject")

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle);self.handle=None


def _bounded_read(path, limit):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>limit:
        raise ValueError("missing/oversized/linked worker receipt")
    with path.open("rb") as handle:
        data=handle.read(limit+1)
    if len(data)>limit:raise ValueError("worker receipt exceeds byte limit")
    return json.loads(data.decode("utf-8"))


def _launch(root, request, protocol):
    """One spawned interpreter, OS admission before bootstrap imports/workload.

    -S disables implicit site imports. Parent runtime site paths are explicitly
    added by the admitted bootstrap, preserving venv Torch without its redirector.
    Pipes are continuously drained; only capped prefixes are retained.
    """
    root=Path(root);config=dict(request)
    config["runtime_paths"]=[p for p in sys.path if Path(p).is_dir() and
                             Path(p).name in ("site-packages","dist-packages")]
    if not config["runtime_paths"]:raise RuntimeError("cannot reproduce parent runtime site paths")
    paths={k:root/(k+".json") for k in ("request","admit","ready","result")}
    paths["request"].write_text(json.dumps(config,allow_nan=False),encoding="utf-8")
    argv=[getattr(sys,"_base_executable",sys.executable),"-I","-S","-B",
          str(Path(__file__).with_name("_training_memory_worker.py")),str(root)]
    job=_Job(protocol.process_memory_bytes,protocol.job_memory_bytes,protocol.max_processes)
    process=None;threads=[];exceeded=threading.Event();logs={k:bytearray() for k in ("stdout","stderr")}
    counts={k:0 for k in logs};monitor_reason=None;cleanup=None;start=time.monotonic()
    def drain(pipe,key):
        try:
            while True:
                chunk=pipe.read(8192)
                if not chunk:break
                counts[key]+=len(chunk)
                room=protocol.max_log_bytes-len(logs[key]);logs[key].extend(chunk[:max(0,room)])
                if counts[key]>protocol.max_log_bytes:exceeded.set()
        finally:pipe.close()
    try:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONIOENCODING="utf-8")
        if protocol.deterministic:env["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
        process=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        job.assign(process)
        for key in logs:
            thread=threading.Thread(target=drain,args=(getattr(process,key),key),daemon=True)
            threads.append(thread);thread.start()
        paths["admit"].write_text(json.dumps(dict(token=request["token"],pid=process.pid)),encoding="utf-8")
        while process.poll() is None:
            if time.monotonic()-start>=protocol.timeout_seconds:
                monitor_reason="timeout";break
            if exceeded.is_set():monitor_reason="resource_limit";break
            if any(p.exists() and p.stat().st_size>protocol.max_result_bytes for p in
                   (paths["result"],root/"result.tmp")):
                monitor_reason="resource_limit";break
            time.sleep(.01)
    finally:
        try:
            # Terminate the whole job even after a successful leader exit: a
            # trusted factory can otherwise leave child/grandchild processes.
            job.terminate()
            if process is not None:
                if process.poll() is None:
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
                deadline=time.monotonic()+5
                while True:
                    cleanup=job.observation()
                    if cleanup["active_processes"]==0 or time.monotonic()>=deadline:break
                    time.sleep(.01)
            for thread in threads:thread.join(timeout=5)
        finally:
            job.close()
            if process is not None and not threads:
                process.stdout.close();process.stderr.close()
    if any(thread.is_alive() for thread in threads):raise RuntimeError("owned pipe reader did not terminate")
    if cleanup is None or cleanup["active_processes"]!=0:raise RuntimeError("owned process tree did not terminate")
    if monitor_reason is None and exceeded.is_set():monitor_reason="resource_limit"
    receipt=dict(launcher_pid=process.pid,exit_code=process.returncode,elapsed_seconds=time.monotonic()-start,
        monitor_outcome=monitor_reason,job=cleanup,kill_on_close=True,admitted=True,
        limits=dict(process_memory_bytes=protocol.process_memory_bytes,job_memory_bytes=protocol.job_memory_bytes,
                    max_processes=protocol.max_processes),
        stdout=bytes(logs["stdout"]).decode("utf-8",errors="replace"),
        stderr=bytes(logs["stderr"]).decode("utf-8",errors="replace"),log_bytes=counts)
    return receipt,paths
