"""Basic CUDA validation budgets: stdlib telemetry, no forensic admission rules."""
import ctypes as c
from ctypes import wintypes as w
import json
from pathlib import Path
import shutil
import subprocess
import time

MiB = 1024**2


def host_snapshot():
    class Status(c.Structure):
        _fields_ = [("length", w.DWORD), ("load", w.DWORD)] + [
            (name, c.c_ulonglong) for name in ("total_physical", "available_physical", "commit_limit",
                "commit_headroom", "total_virtual", "available_virtual", "extended")]
    api = c.WinDLL("kernel32", use_last_error=True)
    status = Status(length=c.sizeof(Status))
    if not api.GlobalMemoryStatusEx(c.byref(status)):
        raise OSError(c.get_last_error(), "GlobalMemoryStatusEx")
    api.CreateMemoryResourceNotification.argtypes = [c.c_int]
    api.CreateMemoryResourceNotification.restype = w.HANDLE
    api.QueryMemoryResourceNotification.argtypes = [w.HANDLE, c.POINTER(w.BOOL)]
    api.CloseHandle.argtypes = [w.HANDLE]
    handle = api.CreateMemoryResourceNotification(0)
    if not handle:
        raise OSError(c.get_last_error(), "low-memory notification unavailable")
    try:
        low = w.BOOL()
        if not api.QueryMemoryResourceNotification(handle, c.byref(low)):
            raise OSError(c.get_last_error(), "low-memory notification query")
    finally:
        api.CloseHandle(handle)
    return dict(time_ns=time.time_ns(), total_physical=status.total_physical,
        available_physical=status.available_physical, commit_limit=status.commit_limit,
        commit_headroom=status.commit_headroom, low_memory=bool(low.value))


def current_process_memory():
    class Counters(c.Structure):
        _fields_ = [("cb", w.DWORD), ("faults", w.DWORD)] + [
            (name, c.c_size_t) for name in ("peak_ws", "ws", "peak_paged", "paged", "peak_nonpaged",
                                          "nonpaged", "pagefile", "peak_pagefile", "private")]
    api = c.WinDLL("kernel32", use_last_error=True)
    api.GetCurrentProcess.restype = w.HANDLE
    psapi = c.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, c.POINTER(Counters), w.DWORD]
    value = Counters(cb=c.sizeof(Counters))
    if not psapi.GetProcessMemoryInfo(api.GetCurrentProcess(), c.byref(value), value.cb):
        raise OSError(c.get_last_error(), "GetProcessMemoryInfo")
    return dict(rss_bytes=value.ws, private_commit_bytes=value.private)


def capture():
    host = host_snapshot()
    tool = shutil.which("nvidia-smi")
    if not tool:
        raise RuntimeError("installed GPU telemetry unavailable")
    result = subprocess.run([tool, "--query-gpu=index,uuid,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
        creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode or len(result.stdout) > 65536:
        raise RuntimeError("GPU telemetry failed")
    rows = [r.strip().split(", ") for r in result.stdout.strip().splitlines()]
    if len(rows) != 1 or len(rows[0]) != 6:
        raise RuntimeError("one explicit GPU telemetry row required")
    index, uuid, name, total, free, driver = rows[0]
    return dict(host=host, gpu=dict(index=int(index), uuid=uuid, name=name,
        total_bytes=int(total)*MiB, free_bytes=int(free)*MiB, driver=driver),
        disk_free_bytes=shutil.disk_usage(Path(__file__).anchor).free)


def budget(cpu_result, parameters):
    """Two overlapping interpreters: observer parent plus one isolated worker.

    Native CPU HWM/Job commit are measured inputs. Additional CUDA host/context
    budgets are conservative planning allowances, not measured CUDA values.
    """
    measurement = cpu_result["measurement"]
    hwm = measurement["rss"]["summary"]["current_lifetime_hwm_bytes"]["value"]
    commit = cpu_result["receipt"]["job"]["peak_job_commit_bytes"]
    if not isinstance(hwm, int) or not isinstance(commit, int):
        raise ValueError("measured CPU resource profile required")
    host_per_process = hwm + 128*MiB
    commit_per_process = commit + 512*MiB
    allocator = max(256*MiB, parameters*4*16)
    process_limit = ((commit + 1024*MiB + 128*MiB-1)//(128*MiB))*(128*MiB)
    return dict(concurrent_interpreters=2, cpu_lifetime_hwm_bytes=hwm, cpu_job_commit_bytes=commit,
        cuda_host_allowance_per_process=128*MiB, cuda_commit_allowance_per_process=512*MiB,
        expected_host_bytes=2*host_per_process, expected_commit_bytes=2*commit_per_process,
        expected_gpu_bytes=2*(384*MiB+allocator), expected_allocator_bytes_per_process=allocator,
        process_limit_bytes=process_limit, outer_job_limit_bytes=2*process_limit,
        child_job_limit_bytes=process_limit, outer_timeout_seconds=120,
        child_timeout_seconds=60, cuda_memory_fraction=None)


def evaluate(observation, expected, *, resident=None):
    host, gpu = observation["host"], observation["gpu"]
    resident = resident or dict(rss_bytes=0, private_commit_bytes=0)
    host_margin = max(512*MiB, host["total_physical"]//20)
    commit_margin = max(512*MiB, host["commit_limit"]//20)
    gpu_margin = max(256*MiB, gpu["total_bytes"]//20)
    remaining_host = max(0, expected["expected_host_bytes"]-resident["rss_bytes"])
    remaining_commit = max(0, expected["expected_commit_bytes"]-resident["private_commit_bytes"])
    blockers = []
    if host["low_memory"]:
        blockers.append("Windows_low_memory_notification")
    if host["available_physical"] < remaining_host+host_margin:
        blockers.append("RAM_below_expected_remaining_workload_plus_OS_margin")
    if host["commit_headroom"] < remaining_commit+commit_margin:
        blockers.append("commit_headroom_below_expected_remaining_workload_plus_OS_margin")
    if gpu["free_bytes"] < expected["expected_gpu_bytes"]+gpu_margin:
        blockers.append("VRAM_below_expected_workload_plus_display_margin")
    if observation["disk_free_bytes"] < 512*MiB:
        blockers.append("disk_below_bounded_evidence_margin")
    return dict(eligible=not blockers, blockers=blockers, remaining_host_bytes=remaining_host,
        remaining_commit_bytes=remaining_commit, host_margin_bytes=host_margin,
        commit_margin_bytes=commit_margin, gpu_margin_bytes=gpu_margin,
        policy="workload_estimate_plus_OS_margin_v1", fixed_host_free_threshold=None,
        forensic_freshness_threshold=None, forensic_PID_classification_required=False)
