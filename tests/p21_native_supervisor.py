"""Bounded Windows process-tree runner for P21 spawn and CUDA test scripts.

Job admission precedes any test workload. Closing the sole Job handle kills all
owned descendants on success, timeout, Python failure or native crash.
"""
import argparse
import ctypes as c
from ctypes import wintypes as w
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def job():
    if os.name!='nt':raise RuntimeError('P21 native supervisor currently requires Windows')
    class IO(c.Structure):
        _fields_=[(n,c.c_ulonglong) for n in ('ro','wo','oo','rb','wb','ob')]
    class Basic(c.Structure):
        _fields_=[('pt',c.c_longlong),('jt',c.c_longlong),('flags',w.DWORD),('minws',c.c_size_t),
                  ('maxws',c.c_size_t),('processes',w.DWORD),('affinity',c.c_size_t),('priority',w.DWORD),('schedule',w.DWORD)]
    class Extended(c.Structure):
        _fields_=[('basic',Basic),('io',IO),('process_memory',c.c_size_t),('job_memory',c.c_size_t),('peak_process',c.c_size_t),('peak_job',c.c_size_t)]
    api=c.WinDLL('kernel32',use_last_error=True)
    api.CreateJobObjectW.argtypes=[c.c_void_p,w.LPCWSTR];api.CreateJobObjectW.restype=w.HANDLE
    api.SetInformationJobObject.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD];api.SetInformationJobObject.restype=w.BOOL
    api.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE];api.AssignProcessToJobObject.restype=w.BOOL
    api.CloseHandle.argtypes=[w.HANDLE];api.CloseHandle.restype=w.BOOL
    handle=api.CreateJobObjectW(None,None)
    if not handle:raise OSError(c.get_last_error(),'CreateJobObjectW')
    info=Extended();info.basic.flags=0x2000|0x100|0x200|0x8
    info.basic.processes=8;info.process_memory=4*1024**3;info.job_memory=12*1024**3
    if not api.SetInformationJobObject(handle,9,c.byref(info),c.sizeof(info)):
        error=c.get_last_error();api.CloseHandle(handle);raise OSError(error,'SetInformationJobObject')
    return api,handle


def main():
    p=argparse.ArgumentParser();p.add_argument('--script',required=True);p.add_argument('--case');p.add_argument('--output',required=True)
    p.add_argument('--timeout',type=float,default=90);p.add_argument('--package-root');p.add_argument('--cuda',action='store_true')
    args=p.parse_args();output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    admit=output.with_suffix('.admit');assert not admit.exists(),'use a fresh output name'
    env=dict(os.environ,PYTHONIOENCODING='utf-8',MPLBACKEND='Agg')
    if args.cuda:env.pop('CUDA_VISIBLE_DEVICES',None)
    else:env['CUDA_VISIBLE_DEVICES']='-1'
    if args.package_root:env['HYSO_TEST_PACKAGE_ROOT']=str(Path(args.package_root).resolve())
    # Launch the actual interpreter, avoiding a Windows venv redirector that
    # could create its real child before job assignment. Preserve its packages.
    runtime=getattr(sys,'_base_executable',sys.executable)
    site=str(Path(sys.prefix)/'Lib/site-packages')
    env['PYTHONPATH']=site+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    argv=[runtime,'-B',str(Path(args.script).resolve()),'--output',str(output),'--admit',str(admit)]
    if args.case:argv+=['--case',args.case]
    api,handle=job();worker=None;start=time.monotonic();receipt=dict(argv=argv,timeout_s=args.timeout,
        process_limit=8,process_commit_bytes=4*1024**3,job_commit_bytes=12*1024**3,kill_on_job_close=True)
    try:
        with output.with_suffix('.stdout.log').open('wb') as stdout,output.with_suffix('.stderr.log').open('wb') as stderr:
            worker=subprocess.Popen(argv,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
            if not api.AssignProcessToJobObject(handle,w.HANDLE(int(worker._handle))):
                raise OSError(c.get_last_error(),'AssignProcessToJobObject')
            admit.write_text('admitted')
            try:worker.wait(timeout=args.timeout);receipt['timed_out']=False
            except subprocess.TimeoutExpired:receipt['timed_out']=True
    finally:
        api.CloseHandle(handle)
        if worker is not None:
            if worker.poll() is None:worker.kill()
            worker.wait()
            receipt.update(exit_code=worker.returncode,launcher_pid=worker.pid)
        receipt['elapsed_s']=time.monotonic()-start
        output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)
    if receipt.get('timed_out') or receipt.get('exit_code')!=0:raise SystemExit(1)


if __name__=='__main__':main()
