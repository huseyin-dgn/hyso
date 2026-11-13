"""DirectShow device metadata only. Never binds a capture filter or opens video."""
import ctypes
import os
import uuid


def enumerate_cameras():
    if os.name != 'nt':
        raise OSError('DirectShow camera identity requires Windows')
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [('data', ctypes.c_ubyte * 16)]

    class VARIANT(ctypes.Structure):
        _fields_ = [('vt', ctypes.c_ushort), ('reserved', ctypes.c_ushort * 3),
                    ('value', ctypes.c_void_p), ('padding', ctypes.c_void_p)]

    def guid(value):
        return GUID((ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID(value).bytes_le))

    def call(pointer, slot, result, types, *args):
        table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *types)(table[slot])(pointer, *args)

    def checked(hr):
        if hr < 0:
            raise OSError(f'DirectShow metadata HRESULT 0x{hr & 0xffffffff:08x}')

    def release(pointer):
        if pointer:
            call(pointer, 2, wintypes.ULONG, [])

    ole = ctypes.OleDLL('ole32')
    auto = ctypes.OleDLL('oleaut32')
    ole.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    ole.CoInitializeEx.restype = ctypes.c_long
    ole.CoCreateInstance.argtypes = [ctypes.POINTER(GUID), ctypes.c_void_p, wintypes.DWORD,
                                   ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
    ole.CoCreateInstance.restype = ctypes.c_long
    auto.VariantClear.argtypes = [ctypes.POINTER(VARIANT)]
    hr = ole.CoInitializeEx(None, 2)
    initialized = hr >= 0
    if not initialized and (hr & 0xffffffff) != 0x80010106:
        checked(hr)
    device_enum = ctypes.c_void_p()
    monikers = ctypes.c_void_p()
    devices = []
    try:
        checked(ole.CoCreateInstance(ctypes.byref(guid('62be5d10-60eb-11d0-bd3b-00a0c911ce86')),
            None, 1, ctypes.byref(guid('29840822-5b84-11d0-bd3b-00a0c911ce86')),
            ctypes.byref(device_enum)))
        hr = call(device_enum, 3, ctypes.c_long,
            [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p), wintypes.DWORD],
            ctypes.byref(guid('860bb310-5d01-11d0-bd3b-00a0c911ce86')), ctypes.byref(monikers), 0)
        checked(hr)
        if hr == 1:
            return []
        while True:
            moniker = ctypes.c_void_p()
            fetched = wintypes.ULONG()
            hr = call(monikers, 3, ctypes.c_long,
                [wintypes.ULONG, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.ULONG)],
                1, ctypes.byref(moniker), ctypes.byref(fetched))
            checked(hr)
            if hr == 1:
                break
            bag = ctypes.c_void_p()
            try:
                checked(call(moniker, 9, ctypes.c_long,
                    [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)],
                    None, None, ctypes.byref(guid('55272a00-42cb-11ce-8135-00aa004bb851')), ctypes.byref(bag)))
                row = dict(index=len(devices), backend='DSHOW')
                for name, key in (('FriendlyName', 'friendly_name'), ('DevicePath', 'device_path')):
                    variant = VARIANT()
                    try:
                        hr = call(bag, 3, ctypes.c_long,
                            [ctypes.c_wchar_p, ctypes.POINTER(VARIANT), ctypes.c_void_p],
                            name, ctypes.byref(variant), None)
                        if name == 'DevicePath' and (hr & 0xffffffff) == 0x80070002:
                            row[key] = None  # Virtual filters may have no physical device path.
                            continue
                        checked(hr)
                        if variant.vt != 8 or not variant.value:
                            raise ValueError('camera metadata requires a BSTR ' + name)
                        row[key] = ctypes.wstring_at(variant.value)
                    finally:
                        auto.VariantClear(ctypes.byref(variant))
                devices.append(row)
            finally:
                release(bag)
                release(moniker)
        return devices
    finally:
        release(monikers)
        release(device_enum)
        if initialized:
            ole.CoUninitialize()


def validate_binding(binding, index):
    if (type(binding) is not dict or binding.get('backend') != 'DSHOW'
            or type(index) is not int or binding.get('index') != index
            or not all(type(binding.get(k)) is str and binding[k]
                       for k in ('device_path', 'friendly_name'))):
        raise ValueError('explicit DirectShow index/device path/name binding required')


def verify_binding(binding, index):
    validate_binding(binding, index)
    devices = enumerate_cameras()
    if index >= len(devices) or devices[index] != binding:
        raise ValueError('physical camera identity/order changed; prepare a new manifest')
    return dict(devices[index])
