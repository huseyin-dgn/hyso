"""P30 OS low-memory notification and live owned Job observations, without quotas."""
import ctypes as c
from ctypes import wintypes as w
import time


class _StreamResources:
    def __init__(self, token):
        self.api=c.WinDLL('kernel32',use_last_error=True)
        self.api.CreateMemoryResourceNotification.argtypes=[c.c_int]
        self.api.CreateMemoryResourceNotification.restype=w.HANDLE
        self.api.QueryMemoryResourceNotification.argtypes=[w.HANDLE,c.POINTER(w.BOOL)]
        self.api.QueryMemoryResourceNotification.restype=w.BOOL
        self.api.CloseHandle.argtypes=[w.HANDLE];self.api.CloseHandle.restype=w.BOOL
        self.handle=self.api.CreateMemoryResourceNotification(0)  # OS LowMemoryResourceNotification
        if not self.handle:raise OSError(c.get_last_error(),'CreateMemoryResourceNotification')
        self.summary=dict(clock_id=token+':controller_perf_counter_ns',samples=0,first_timestamp_ns=None,
            last_timestamp_ns=None,peak_process_commit_bytes=0,peak_job_commit_bytes=0,
            low_memory_observed=False,policy='stop_on_Windows_low_memory_notification_no_RAM_quota')

    def sample(self, job):
        low=w.BOOL()
        if not self.api.QueryMemoryResourceNotification(self.handle,c.byref(low)):
            raise OSError(c.get_last_error(),'QueryMemoryResourceNotification')
        observation=job.observation();stamp=time.perf_counter_ns();s=self.summary
        s['samples']+=1;s['last_timestamp_ns']=stamp
        if s['first_timestamp_ns'] is None:s['first_timestamp_ns']=stamp
        for field in ('peak_process_commit_bytes','peak_job_commit_bytes'):
            s[field]=max(s[field],observation[field])
        s['low_memory_observed']|=bool(low.value)
        return bool(low.value)

    def close(self):
        self.api.CloseHandle(self.handle)
