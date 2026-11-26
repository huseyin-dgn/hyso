"""Read-only host/runtime checks and exact Windows process identities."""
from __future__ import annotations

import ctypes as c
import importlib.metadata
import os
import shutil
import sys
from pathlib import Path

from common import LEGACY, ROOT, REPO, file_hash, read_json, utc


class Memory(c.Structure):
    _fields_ = [('length', c.c_ulong), ('load', c.c_ulong)] + [
        (n, c.c_ulonglong) for n in ('physical_total', 'physical_free', 'commit_total',
        'commit_free', 'virtual_total', 'virtual_free', 'extended')]


def resources():
    if os.name != 'nt':
        raise RuntimeError('UNSUPPORTED_RUNTIME: Windows owned Job verification required')
    value = Memory()
    value.length = c.sizeof(value)
    if not c.windll.kernel32.GlobalMemoryStatusEx(c.byref(value)):
        raise OSError('GlobalMemoryStatusEx failed; no admission')
    import tempfile
    return dict(time_utc=utc(), physical_free_bytes=value.physical_free,
        available_commit_bytes=value.commit_free, commit_limit_bytes=value.commit_total,
        disk_free_bytes=min(shutil.disk_usage(ROOT).free, shutil.disk_usage(tempfile.gettempdir()).free))


def runtime_plan():
    retained = read_json(LEGACY / 'source-runtime.json', {})
    python = Path(retained.get('python', 'missing-python'))
    cpu_site = python.parent.parent / 'Lib/site-packages'
    fallback = retained.get('fallback_test_dependencies', [])
    paths = [str(cpu_site), *fallback]
    versions = {}
    for directory in reversed(paths):
        if Path(directory).is_dir():
            for dist in importlib.metadata.distributions(path=[directory]):
                name = (dist.metadata.get('Name') or '').lower().replace('_', '-')
                versions[name] = dist.version
    requirements = ['torch', 'pytest', 'pygments', 'build', 'setuptools', 'wheel', 'numpy']
    missing = [name for name in requirements if name not in versions]
    cpu_meta = {d.metadata.get('Name', '').lower(): d.version
                for d in importlib.metadata.distributions(path=[str(cpu_site)])} if cpu_site.is_dir() else {}
    if not python.is_file():
        missing.append('CPU Python executable')
    if cpu_meta.get('torch') != '2.5.1+cpu':
        missing.append('verified CPU torch 2.5.1+cpu distribution')
    package = read_json(LEGACY / 'package-config.json', {})
    wheelhouse = Path(package.get('wheelhouse', 'missing-wheelhouse'))
    if not wheelhouse.is_dir():
        missing.append('offline dependency wheelhouse')
    required_paths = [LEGACY / n for n in ('bootstrap.py', 'readonly-test-environment.json',
        'blocked-nodeids.json', 'collected-tests.json', 'complete-static-inventory.json')]
    required_paths += [REPO / 'outputs/package-clean-cpu-54e64e9-20261125' / n
                       for n in ('verify_installed.py', 'clean_fixture.py', 'requirements.txt')]
    missing.extend(str(p) for p in required_paths if not p.is_file())
    # The legacy CPU forensic fixture is immutable and separately pins its runtime.
    base_candidates = [Path(p).parents[1] / 'python.exe' for p in fallback
                       if Path(p).name == 'site-packages' and Path(p).parent.name.lower() == 'lib']
    base = next((p for p in base_candidates if p.is_file()), None)
    return dict(cpu_python=str(python.resolve()), cpu_site=str(cpu_site.resolve()),
        runtime_paths=paths, fallback_test_dependencies=fallback, versions=versions,
        torch_cpu_distribution=cpu_meta.get('torch'), missing=missing,
        required_paths={str(p): p.is_file() for p in required_paths},
        offline_wheel_inventory=[dict(name=p.name, bytes=p.stat().st_size) for p in sorted(wheelhouse.glob('*.whl'))],
        wheelhouse=str(wheelhouse.resolve()), forensic_python=str(base) if base else None,
        forensic_profile='Torch-free parent; only existing CPU recipe; legacy worker build pin is preserved, never CUDA execution',
        runtime_import_validation='DEFERRED_TO_USER_STARTED_OWNED_WORKER')


def process_identity(pid):
    """PID reuse is distinguished by kernel creation time and actual image path."""
    if os.name != 'nt':
        return None
    api = c.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = [c.c_ulong, c.c_int, c.c_ulong]
    api.OpenProcess.restype = c.c_void_p
    api.CloseHandle.argtypes = [c.c_void_p]
    api.QueryFullProcessImageNameW.argtypes = [c.c_void_p, c.c_ulong, c.c_wchar_p, c.POINTER(c.c_ulong)]
    api.GetProcessTimes.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p]
    handle = api.OpenProcess(0x1000, False, int(pid))
    if not handle:
        error=c.get_last_error()
        if error in (87,1168):  # OS says no such PID; access denied is not absence.
            return None
        raise OSError(error,'process identity unavailable; absence cannot be proved')
    try:
        created, exited, kernel, user = (c.c_ulonglong() for _ in range(4))
        if not api.GetProcessTimes(handle, c.byref(created), c.byref(exited), c.byref(kernel), c.byref(user)):
            raise OSError('process creation identity unavailable')
        buffer = c.create_unicode_buffer(32768)
        length = c.c_ulong(len(buffer))
        if not api.QueryFullProcessImageNameW(handle, 0, buffer, c.byref(length)):
            raise OSError('process image identity unavailable')
        return dict(pid=int(pid), creation_filetime=created.value, image=buffer.value)
    finally:
        api.CloseHandle(handle)


def same_process(record):
    actual = process_identity(record['pid'])
    return actual is not None and actual == {k: record[k] for k in ('pid', 'creation_filetime', 'image')}


class ExecutionState:
    def __enter__(self):
        self.api = c.WinDLL('kernel32')
        self.api.SetThreadExecutionState.argtypes = [c.c_ulong]
        self.api.SetThreadExecutionState.restype = c.c_ulong
        self.previous = self.api.SetThreadExecutionState(0x80000000 | 1)
        if not self.previous:
            raise OSError('temporary sleep inhibition unavailable')
        return self

    def __exit__(self, *exc):
        self.api.SetThreadExecutionState(self.previous | 0x80000000)


def check_config(config):
    fixed = dict(start_available_commit_bytes=4 * 1024**3,
        stop_available_commit_bytes=1024**3, process_memory_bytes=3 * 1024**3,
        job_memory_bytes=6 * 1024**3, max_processes=48)
    if config.get('schema_version') != 1 or any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError('mandatory campaign resource limits changed')
    for name in ('allow_real_cuda', 'allow_native_failfast', 'allow_allocation_stress',
                 'allow_p31_long_run', 'automatic_source_fixes', 'network_package_install'):
        if config.get(name) is not False:
            raise ValueError('unsafe campaign option: ' + name)
    if config['max_forensic_cpu_verification_attempts'] != 1:
        raise ValueError('forensic CPU verification must remain one-shot')
    if file_hash(LEGACY/'blocked-nodeids.json') != config.get('risk_exclusion_manifest_sha256'):
        raise ValueError('risk exclusion manifest changed; no fresh admission is implied')
    if len(read_json(LEGACY/'blocked-nodeids.json',{}))!=13:
        raise ValueError('all thirteen recorded risky test nodes must remain blocked')
    return True
