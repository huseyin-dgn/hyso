"""Small OS-only availability preflight; no accelerator, factory or process launch."""
import os
from pathlib import Path
import sys


def memory_availability() -> dict:
    """Return native availability estimates, not reservations or hard limits."""
    try:
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            class Status(ctypes.Structure):
                _fields_ = [('length',wintypes.DWORD),('load',wintypes.DWORD)] + [
                    (name,ctypes.c_ulonglong) for name in (
                        'total_phys','avail_phys','total_commit','avail_commit',
                        'total_virtual','avail_virtual','extended_virtual')]
            status = Status(); status.length = ctypes.sizeof(Status)
            query = ctypes.WinDLL('kernel32',use_last_error=True).GlobalMemoryStatusEx
            query.argtypes = [ctypes.POINTER(Status)]; query.restype = wintypes.BOOL
            if not query(ctypes.byref(status)):
                raise OSError('GlobalMemoryStatusEx failed')
            physical, commit = int(status.avail_phys), int(status.avail_commit)
            source = 'Windows.GlobalMemoryStatusEx'
        elif sys.platform.startswith('linux'):
            values = {}
            for line in Path('/proc/meminfo').read_text(encoding='ascii').splitlines():
                key, rest = line.split(':',1)
                if key in ('MemAvailable','CommitLimit','Committed_AS'):
                    number, unit = rest.split()
                    if unit != 'kB': raise ValueError('unexpected memory unit')
                    values[key] = int(number)*1024
            physical = values['MemAvailable']
            commit = max(0,values['CommitLimit']-values['Committed_AS'])
            source = 'Linux.proc_meminfo;commit_headroom_estimate'
        else:
            raise OSError('availability backend unavailable on this platform')
        return dict(status='observed',available_physical_bytes=physical,
                    available_commit_bytes=commit,source=source,
                    scope='instantaneous estimates; no memory reservation or activation-memory prediction')
    except (OSError,ValueError,KeyError) as error:
        return dict(status='unavailable',reason=str(error),available_physical_bytes=None,
                    available_commit_bytes=None,source=None)
