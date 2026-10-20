"""Read-only Windows counters and optional bounded installed NVIDIA CLI queries."""
import csv
from functools import lru_cache
import math
import os
import shutil
import subprocess
import time


@lru_cache(maxsize=1)
def _windows_times_api():
    import ctypes as c
    from ctypes import wintypes as w
    api=c.WinDLL('kernel32',use_last_error=True)
    api.GetCurrentProcess.argtypes=[]; api.GetCurrentProcess.restype=w.HANDLE
    api.GetProcessTimes.argtypes=[w.HANDLE]+[c.POINTER(w.FILETIME)]*4
    api.GetProcessTimes.restype=w.BOOL
    api.GetSystemTimes.argtypes=[c.POINTER(w.FILETIME)]*3; api.GetSystemTimes.restype=w.BOOL
    return api,w.FILETIME


def cpu_observe():
    start=time.perf_counter_ns()
    if os.name!='nt': raise NotImplementedError('native CPU counter adapter verified on Windows only')
    import ctypes as c
    api,Filetime=_windows_times_api()
    values=[Filetime() for _ in range(4)]
    if not api.GetProcessTimes(api.GetCurrentProcess(),*(c.byref(v) for v in values)):
        raise OSError(c.get_last_error(),'GetProcessTimes failed')
    ns=lambda v: ((v.dwHighDateTime<<32)|v.dwLowDateTime)*100
    system=[Filetime() for _ in range(3)]
    system_ok=bool(api.GetSystemTimes(*(c.byref(v) for v in system)))
    end=time.perf_counter_ns()
    return dict(timestamp_ns=(start+end)//2,query_start_ns=start,query_end_ns=end,
        kernel_ns=ns(values[2]),user_ns=ns(values[3]),status='measured',reason=None,
        system_idle_ns=ns(system[0]) if system_ok else None,
        system_kernel_ns=ns(system[1]) if system_ok else None,
        system_user_ns=ns(system[2]) if system_ok else None,
        system_status='measured' if system_ok else 'unsupported')


GPU_FIELDS='uuid,pci.bus_id,name,driver_version,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total,timestamp'


def _number(text, scale=1, percent=False):
    if text.strip() in ('N/A','[N/A]','[Not Supported]','Not Supported','-',''): return None
    try: value=float(text)*scale
    except ValueError: raise ValueError('invalid NVIDIA numeric field') from None
    if not math.isfinite(value) or value<0 or (percent and value>100): raise ValueError('invalid NVIDIA sensor range')
    return value


def parse_gpu_csv(text):
    rows=list(csv.reader(text.strip().splitlines(),skipinitialspace=True))
    if len(rows)!=1 or len(rows[0])!=10: raise ValueError('NVIDIA query must contain exactly one device/10 fields')
    uuid,pci,name,driver,busy,controller,used,free,total,stamp=[s.strip() for s in rows[0]]
    if not uuid.startswith('GPU-') or not pci or not name: raise ValueError('missing physical GPU identity')
    return dict(uuid=uuid,pci_bus_id=pci,name=name,driver=driver,gpu_busy_percent=_number(busy,percent=True),
        memory_controller_busy_percent=_number(controller,percent=True),
        memory_used_bytes=_number(used,1024**2),memory_free_bytes=_number(free,1024**2),
        memory_total_bytes=_number(total,1024**2),sensor_wall_timestamp=stamp)


def gpu_observe(selector,timeout):
    start=time.perf_counter_ns()
    result=dict(query_start_ns=start,query_end_ns=None,status='unsupported',reason=None,values=None,
                backend='nvidia-smi.NVML',domain='physical_device_wide',tool_path=None)
    tool=shutil.which('nvidia-smi')
    if tool is None: result['reason']='installed nvidia-smi unavailable; no dependency installed'
    else:
        result['tool_path']=tool
        try:
            p=subprocess.run([tool,'--id='+selector,'--query-gpu='+GPU_FIELDS,'--format=csv,noheader,nounits'],
                capture_output=True,text=True,timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            if p.returncode and (p.returncode<0 or (p.returncode & 0xffffffff)>=0x80000000):
                raise RuntimeError('unexpected native sensor process crash; stop collection: '+str(p.returncode))
            if p.returncode:
                result.update(status='unavailable',reason='NVIDIA query exit '+str(p.returncode)+': '+p.stderr[-512:])
            else: result.update(status='measured',reason=None,values=parse_gpu_csv(p.stdout))
        except subprocess.TimeoutExpired: result.update(status='timed_out',reason='sensor query deadline')
        except (OSError,ValueError) as error: result.update(status='unavailable',reason=type(error).__name__+': '+str(error))
    result['query_end_ns']=time.perf_counter_ns()
    return result
