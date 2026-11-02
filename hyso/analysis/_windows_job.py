"""Private G3 Windows-only suspended child/Job launcher.

A child cannot import Torch or caller code before assignment to an owned Job.
This is process/resource containment, NOT a security sandbox for hostile Python.
No PID-name killing or implicit breakaway is used.
"""
import ctypes as C
from ctypes import wintypes as W
import os
import subprocess
import sys
import time


class JobLaunchError(RuntimeError):
    pass


class _STARTUPINFO(C.Structure):
    _fields_ = [('cb', W.DWORD), ('lpReserved', W.LPWSTR), ('lpDesktop', W.LPWSTR),
                ('lpTitle', W.LPWSTR), ('dwX', W.DWORD), ('dwY', W.DWORD),
                ('dwXSize', W.DWORD), ('dwYSize', W.DWORD), ('dwXCountChars', W.DWORD),
                ('dwYCountChars', W.DWORD), ('dwFillAttribute', W.DWORD),
                ('dwFlags', W.DWORD), ('wShowWindow', W.WORD), ('cbReserved2', W.WORD),
                ('lpReserved2', C.POINTER(C.c_ubyte)), ('hStdInput', W.HANDLE),
                ('hStdOutput', W.HANDLE), ('hStdError', W.HANDLE)]


class _PROCESS_INFORMATION(C.Structure):
    _fields_ = [('hProcess', W.HANDLE), ('hThread', W.HANDLE),
                ('dwProcessId', W.DWORD), ('dwThreadId', W.DWORD)]


class _BASIC_LIMIT(C.Structure):
    _fields_ = [('PerProcessUserTimeLimit', C.c_int64),
                ('PerJobUserTimeLimit', C.c_int64), ('LimitFlags', W.DWORD),
                ('MinimumWorkingSetSize', C.c_size_t), ('MaximumWorkingSetSize', C.c_size_t),
                ('ActiveProcessLimit', W.DWORD), ('Affinity', C.c_size_t),
                ('PriorityClass', W.DWORD), ('SchedulingClass', W.DWORD)]


class _IO_COUNTERS(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in (
        'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
        'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class _EXTENDED_LIMIT(C.Structure):
    _fields_ = [('BasicLimitInformation', _BASIC_LIMIT), ('IoInfo', _IO_COUNTERS),
                ('ProcessMemoryLimit', C.c_size_t), ('JobMemoryLimit', C.c_size_t),
                ('PeakProcessMemoryUsed', C.c_size_t), ('PeakJobMemoryUsed', C.c_size_t)]


_KILL_ON_CLOSE = 0x2000
_JOB_MEMORY = 0x200
_PROCESS_MEMORY = 0x100
_ACTIVE_PROCESS = 0x8
_EXTENDED_LIMIT_INFORMATION = 9
_CREATE_SUSPENDED = 0x00000004
_CREATE_NO_WINDOW = 0x08000000
_WAIT_OBJECT_0 = 0
_WAIT_TIMEOUT = 0x102
_WAIT_FAILED = 0xFFFFFFFF


def _api():
    if os.name != 'nt':
        raise JobLaunchError('windows_job_only_no_unbounded_fallback')
    k = C.WinDLL('kernel32', use_last_error=True)
    k.CreateJobObjectW.argtypes = [C.c_void_p, W.LPCWSTR]
    k.CreateJobObjectW.restype = W.HANDLE
    k.SetInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD]
    k.SetInformationJobObject.restype = W.BOOL
    k.CreateProcessW.argtypes = [W.LPCWSTR, W.LPWSTR, C.c_void_p, C.c_void_p,
                                  W.BOOL, W.DWORD, C.c_void_p, W.LPCWSTR,
                                  C.POINTER(_STARTUPINFO), C.POINTER(_PROCESS_INFORMATION)]
    k.CreateProcessW.restype = W.BOOL
    k.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
    k.AssignProcessToJobObject.restype = W.BOOL
    k.ResumeThread.argtypes = [W.HANDLE]
    k.ResumeThread.restype = W.DWORD
    k.WaitForSingleObject.argtypes = [W.HANDLE, W.DWORD]
    k.WaitForSingleObject.restype = W.DWORD
    k.TerminateJobObject.argtypes = [W.HANDLE, W.UINT]
    k.TerminateJobObject.restype = W.BOOL
    k.TerminateProcess.argtypes = [W.HANDLE, W.UINT]
    k.TerminateProcess.restype = W.BOOL
    k.GetExitCodeProcess.argtypes = [W.HANDLE, C.POINTER(W.DWORD)]
    k.GetExitCodeProcess.restype = W.BOOL
    k.CloseHandle.argtypes = [W.HANDLE]
    k.CloseHandle.restype = W.BOOL
    return k


def _check(ok, label):
    if not ok:
        raise JobLaunchError(label + '_winerror_' + str(C.get_last_error()))


def run_bounded_windows(command, directory, *, timeout_seconds, job_memory_limit_bytes):
    """Spawn suspended, assign job+hard commit limits, resume, enforce deadline.

    Returns only after process termination (including failure cleanup). A failed
    cleanup is a hard exception, never a completed/cleaned receipt.
    """
    k = _api()
    if (type(command) not in (tuple, list) or len(command) < 2 or
            any(type(x) is not str or not x for x in command)):
        raise JobLaunchError('invalid_command')
    if type(timeout_seconds) not in (int, float) or timeout_seconds <= 0:
        raise JobLaunchError('invalid_timeout')
    if type(job_memory_limit_bytes) is not int or job_memory_limit_bytes <= 0:
        raise JobLaunchError('invalid_memory_limit')
    job = k.CreateJobObjectW(None, None)
    _check(job, 'create_job')
    proc = _PROCESS_INFORMATION()
    child_created = False
    job_assigned = False
    terminated = False
    reason = 'not_started'
    try:
        limits = _EXTENDED_LIMIT()
        limits.BasicLimitInformation.LimitFlags = (
            _KILL_ON_CLOSE | _JOB_MEMORY | _PROCESS_MEMORY | _ACTIVE_PROCESS)
        limits.BasicLimitInformation.ActiveProcessLimit = 1
        limits.ProcessMemoryLimit = job_memory_limit_bytes
        limits.JobMemoryLimit = job_memory_limit_bytes
        _check(k.SetInformationJobObject(job, _EXTENDED_LIMIT_INFORMATION,
                                         C.byref(limits), C.sizeof(limits)), 'set_hard_job_limits')
        si = _STARTUPINFO()
        si.cb = C.sizeof(si)
        argv = subprocess.list2cmdline(list(command))
        line = C.create_unicode_buffer(argv)
        _check(k.CreateProcessW(command[0], line, None, None, False,
                                _CREATE_SUSPENDED | _CREATE_NO_WINDOW,
                                None, str(directory), C.byref(si), C.byref(proc)), 'create_suspended')
        child_created = True
        _check(k.AssignProcessToJobObject(job, proc.hProcess), 'assign_before_resume')
        job_assigned = True
        if k.ResumeThread(proc.hThread) == 0xFFFFFFFF:
            raise JobLaunchError('resume_worker_winerror_' + str(C.get_last_error()))
        began = time.monotonic()
        wait = k.WaitForSingleObject(proc.hProcess, max(1, int(timeout_seconds * 1000)))
        if wait == _WAIT_TIMEOUT:
            reason = 'timeout'
            _check(k.TerminateJobObject(job, 1), 'terminate_job_on_timeout')
        elif wait == _WAIT_OBJECT_0:
            reason = 'process_exited'
        else:
            raise JobLaunchError('wait_worker_failed_' + str(wait))
        done = k.WaitForSingleObject(proc.hProcess, 5000)
        if done != _WAIT_OBJECT_0:
            raise JobLaunchError('worker_cleanup_unconfirmed_' + str(done))
        terminated = True
        code = W.DWORD()
        _check(k.GetExitCodeProcess(proc.hProcess, C.byref(code)), 'get_exit_code')
        return dict(status=reason, exit_code=int(code.value), assigned_before_resume=job_assigned,
                    hard_job_memory_limit_bytes=job_memory_limit_bytes, active_owned_processes=0,
                    elapsed_seconds=time.monotonic()-began)
    finally:
        if child_created and not terminated:
            # Even assignment failure cannot leave a suspended orphan behind.
            if job_assigned:
                k.TerminateJobObject(job, 1)
            else:
                k.TerminateProcess(proc.hProcess, 1)
            terminated = k.WaitForSingleObject(proc.hProcess, 5000) == _WAIT_OBJECT_0
        if proc.hThread:
            k.CloseHandle(proc.hThread)
        if proc.hProcess:
            k.CloseHandle(proc.hProcess)
        k.CloseHandle(job)
        if child_created and not terminated and sys.exc_info()[0] is None:
            raise JobLaunchError('owned_child_termination_not_confirmed')
