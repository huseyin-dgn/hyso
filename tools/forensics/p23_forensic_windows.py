"""P23-only Windows debug events and bounded, handle-owned host dumps.

No PID attach API exists here. The supervisor creates the debuggee; the dump
helper receives a duplicate of that Popen handle, never opens an inventory PID.
"""
import ctypes as c
from ctypes import wintypes as w
import os
from pathlib import Path
import struct


class ExceptionRecord(c.Structure):
    _fields_ = [('code', w.DWORD), ('flags', w.DWORD), ('nested', c.c_void_p),
                ('address', c.c_void_p), ('count', w.DWORD), ('information', c.c_size_t * 15)]


class ExceptionInfo(c.Structure):
    _fields_ = [('record', ExceptionRecord), ('first_chance', w.DWORD)]


class CreateProcessInfo(c.Structure):
    _fields_ = [('file', w.HANDLE), ('process', w.HANDLE), ('thread', w.HANDLE),
                ('base', c.c_void_p), ('debug_offset', w.DWORD), ('debug_size', w.DWORD),
                ('tls', c.c_void_p), ('start', c.c_void_p), ('image_name', c.c_void_p),
                ('unicode', w.WORD)]


class EventData(c.Union):
    _fields_ = [('exception', ExceptionInfo), ('created', CreateProcessInfo),
                ('raw', c.c_byte * 160)]


class DebugEvent(c.Structure):
    _fields_ = [('kind', w.DWORD), ('pid', w.DWORD), ('tid', w.DWORD), ('data', EventData)]


class DumpExceptionInformation(c.Structure):
    # minidumpapiset.h includes pshpack4.h, including this API argument.
    # EXCEPTION_POINTERS/CONTEXT themselves retain their native ABI alignment.
    _pack_ = 4
    _fields_ = [('tid', w.DWORD), ('pointers', c.c_void_p), ('client', w.BOOL)]


def kernel():
    if os.name != 'nt' or c.sizeof(c.c_void_p) != 8:
        raise NotImplementedError('reviewed Windows x64 only')
    api = c.WinDLL('kernel32', use_last_error=True)
    signatures = {
        'WaitForDebugEvent': ([c.POINTER(DebugEvent), w.DWORD], w.BOOL),
        'ContinueDebugEvent': ([w.DWORD, w.DWORD, w.DWORD], w.BOOL),
        'GetProcessId': ([w.HANDLE], w.DWORD),
        'GetProcessIdOfThread': ([w.HANDLE], w.DWORD),
        'GetCurrentProcess': ([], w.HANDLE),
        'CloseHandle': ([w.HANDLE], w.BOOL),
        'OpenThread': ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
        'GetThreadContext': ([w.HANDLE, c.c_void_p], w.BOOL),
        'IsProcessInJob': ([w.HANDLE, w.HANDLE, c.POINTER(w.BOOL)], w.BOOL),
        'DuplicateHandle': ([w.HANDLE, w.HANDLE, w.HANDLE, c.POINTER(w.HANDLE),
                             w.DWORD, w.BOOL, w.DWORD], w.BOOL),
    }
    for name, (args, result) in signatures.items():
        getattr(api, name).argtypes, getattr(api, name).restype = args, result
    return api


def duplicate_to(process, handle):
    api, remote = kernel(), w.HANDLE()
    if not api.DuplicateHandle(api.GetCurrentProcess(), w.HANDLE(int(handle)),
                              w.HANDLE(int(process._handle)), c.byref(remote), 0, False, 2):
        raise OSError(c.get_last_error(), 'duplicate owned handle')
    return remote.value


def require_owned(handle, job_handle, pid):
    api, member = kernel(), w.BOOL()
    if api.GetProcessId(w.HANDLE(handle)) != pid or pid == os.getpid():
        raise ValueError('owned process handle/PID mismatch')
    if not api.IsProcessInJob(w.HANDLE(handle), w.HANDLE(job_handle), c.byref(member)) or not member.value:
        raise ValueError('target outside exact owned Job')


def next_event(pid, milliseconds=10):
    event, api = DebugEvent(), kernel()
    if not api.WaitForDebugEvent(c.byref(event), milliseconds):
        error = c.get_last_error()
        if error in (121, 258):
            return None
        raise OSError(error, 'WaitForDebugEvent')
    if event.pid != pid:
        raise ValueError('unexpected debug target; DEBUG_ONLY_THIS_PROCESS required')
    return event


def resume(event, handled=False):
    api = kernel()
    if not api.ContinueDebugEvent(event.pid, event.tid, 0x10002 if handled else 0x80010001):
        raise OSError(c.get_last_error(), 'ContinueDebugEvent')


def close_event_handles(event):
    api = kernel()
    # CREATE_THREAD, CREATE_PROCESS, LOAD_DLL include debugger-owned handles.
    if event.kind == 3:
        values = [event.data.created.file, event.data.created.thread, event.data.created.process]
    elif event.kind in (2, 6):
        values = [c.cast(c.byref(event.data), c.POINTER(w.HANDLE))[0]]
    else:
        values = []
    for value in values:
        if value:
            api.CloseHandle(w.HANDLE(value))


def exception_row(event):
    info = event.data.exception
    if info.record.count > 15:
        raise ValueError('invalid native exception parameters')
    return dict(pid=event.pid, tid=event.tid, code=info.record.code,
                flags=info.record.flags, address=info.record.address,
                first_chance=bool(info.first_chance),
                information=list(info.record.information[:info.record.count]))


def validate_dump(path):
    """Check stream bounds and actual memory/exception/context, not just MDMP."""
    path = Path(path)
    size = path.stat().st_size
    with path.open('rb') as stream:
        header = stream.read(32)
        if len(header) != 32 or header[:4] != b'MDMP':
            raise ValueError('missing complete minidump header')
        _, _, count, rva, _, _, flags = struct.unpack('<4sIIIIIQ', header)
        if count > 256 or rva + 12 * count > size or not flags & 2:
            raise ValueError('invalid full-memory dump directory')
        stream.seek(rva)
        entries = [struct.unpack('<III', stream.read(12)) for _ in range(count)]
        table = {kind: (length, offset) for kind, length, offset in entries}
        for length, offset in table.values():
            if offset + length > size:
                raise ValueError('incomplete dump stream')
        if not {3, 6, 9} <= table.keys():
            raise ValueError('thread/exception/full-memory streams required')
        length, offset = table[9]
        stream.seek(offset)
        ranges, base = struct.unpack('<QQ', stream.read(16))
        if ranges > 100000 or 16 + ranges * 16 > length:
            raise ValueError('invalid memory range table')
        total = sum(struct.unpack('<QQ', stream.read(16))[1] for _ in range(ranges))
        if base + total > size or not total:
            raise ValueError('incomplete full-memory pages')
        stream.seek(table[6][1])
        exception = stream.read(168)
        if len(exception) != 168:
            raise ValueError('incomplete exception stream')
        tid = struct.unpack_from('<I', exception)[0]
        code = struct.unpack_from('<I', exception, 8)[0]
        context_size, context_rva = struct.unpack_from('<II', exception, 160)
        if context_size < 1232 or context_rva + context_size > size:
            raise ValueError('missing exception context')
    return dict(bytes=size, memory_bytes=total, ranges=ranges, exception_code=code,
                exception_tid=tid, full_memory=True, streams=sorted(table))


def write_owned_dump(request, destination, dbghelp):
    """External helper; caller separately enforces time/file/process budgets."""
    api = kernel()
    handle, job = request['process_handle'], request['job_handle']
    require_owned(handle, job, request['pid'])
    thread = api.OpenThread(0x48, False, request['exception']['tid'])  # CONTEXT + QUERY
    if not thread:
        raise OSError(c.get_last_error(), 'owned exception thread context')
    try:
        if api.GetProcessIdOfThread(thread) != request['pid']:
            raise ValueError('exception thread outside owned worker')
        # AMD64 CONTEXT has 16-byte alignment, ContextFlags at offset 48.
        storage = c.create_string_buffer(1232 + 15)
        aligned = (c.addressof(storage) + 15) & ~15
        c.c_uint32.from_address(aligned + 48).value = 0x10001f
        if not api.GetThreadContext(thread, aligned):
            raise OSError(c.get_last_error(), 'GetThreadContext while debug-stopped')
        row = request['exception']
        record = ExceptionRecord(code=row['code'], flags=row['flags'], address=row['address'],
                                 count=len(row['information']))
        record.information[:len(row['information'])] = row['information']
        class Pointers(c.Structure):
            _fields_ = [('record', c.POINTER(ExceptionRecord)), ('context', c.c_void_p)]
        pointers = Pointers(c.pointer(record), aligned)
        info = DumpExceptionInformation(row['tid'], c.addressof(pointers), False)
        library = c.WinDLL(str(dbghelp), use_last_error=True)
        library.MiniDumpWriteDump.argtypes = [w.HANDLE, w.DWORD, w.HANDLE, w.DWORD,
                                              c.c_void_p, c.c_void_p, c.c_void_p]
        library.MiniDumpWriteDump.restype = w.BOOL
        import msvcrt
        with Path(destination).open('xb') as stream:
            # Full host memory, unloaded modules, memory map, thread info.
            if not library.MiniDumpWriteDump(w.HANDLE(handle), request['pid'],
                    w.HANDLE(msvcrt.get_osfhandle(stream.fileno())), 0x1822,
                    c.byref(info), None, None):
                raise OSError(c.get_last_error(), 'MiniDumpWriteDump')
            stream.flush()
            os.fsync(stream.fileno())
        checked = validate_dump(destination)
        if checked['exception_code'] != row['code'] or checked['exception_tid'] != row['tid']:
            raise ValueError('dump exception identity mismatch')
        return checked
    finally:
        api.CloseHandle(thread)
        api.CloseHandle(w.HANDLE(handle))
        api.CloseHandle(w.HANDLE(job))
