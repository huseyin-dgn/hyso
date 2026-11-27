"""Shared stdlib-only Windows loaded-module identity primitives.

No runtime import, provider policy, allowlisting or CUDA initialization.
"""
import ctypes as c
import hashlib
import os
from pathlib import Path

EXPORT_COMPILER_PREFIXES=('nvrtc','nvjit')


def export_selected(name):
    return name.lower().startswith(('nvcuda','cudart','cublas','cudnn',
        *EXPORT_COMPILER_PREFIXES,'nvinfer','nvonnxparser','tensorrt','torch_cuda','onnxruntime'))


def torch_compiler_inventory(site, version, torch_cuda):
    """Pre-admission RECORD-backed compiler DLLs from the admitted Torch only.

    No loaded-module observations, PATH search or directory-wide allowlisting.
    """
    import base64
    import importlib.metadata
    site=Path(site).resolve();lib=site/'torch/lib'
    def ordinary(path):
        for value in (path,*path.parents):
            if value==site:break
            if value.is_symlink() or os.name=='nt' and value.lstat().st_file_attributes & 0x400:
                raise ValueError('Torch dependency reparse path rejected: '+str(value))
        if not path.resolve().is_relative_to(site):raise ValueError('Torch dependency escaped runtime')
    ordinary(lib)
    anchor=lib/'torch_cuda.dll';ordinary(anchor)
    if canonical_path(anchor)!=canonical_path(torch_cuda['path']) or file_hash(anchor)!=torch_cuda['sha256']:
        raise ValueError('Unverified Torch runtime anchor')
    distributions=[d for d in importlib.metadata.distributions(path=[str(site)]) if d.metadata.get('Name','').lower()=='torch']
    if len(distributions)!=1 or distributions[0].version!=version:raise ValueError('Unverified Torch distribution/version')
    dist=distributions[0];rows={}
    for member in dist.files or []:
        name=Path(str(member)).name.lower()
        if not name.endswith('.dll') or not name.startswith(EXPORT_COMPILER_PREFIXES):continue
        # Only direct torch/lib members; no ../, external NVIDIA trees or PATH.
        if Path(str(member)).parts[:2]!=('torch','lib') or len(Path(str(member)).parts)!=3:
            raise ValueError('Compiler DLL RECORD provenance is outside torch/lib')
        path=Path(dist.locate_file(member));ordinary(path)
        if path.resolve().parent!=lib.resolve():raise ValueError('Compiler DLL provenance escaped torch/lib')
        if not export_selected(path.name):raise ValueError('Compiler DLL policy mismatch')
        before=(file_identity(path),path.stat().st_mtime_ns)
        sha=file_hash(path);entry=member.hash
        if entry is None or entry.mode!='sha256' or base64.urlsafe_b64encode(bytes.fromhex(sha)).decode().rstrip('=')!=entry.value:
            raise ValueError('Compiler DLL differs from Torch distribution RECORD: '+path.name)
        if member.size is None or member.size!=path.stat().st_size:raise ValueError('Compiler DLL RECORD size mismatch')
        if before!=(file_identity(path),path.stat().st_mtime_ns):raise ValueError('Compiler DLL changed during inventory')
        key=canonical_path(path)
        if key in rows:raise ValueError('Duplicate compiler DLL RECORD member')
        rows[key]=dict(sha256=sha,file_id=before[0]['file_id'],bytes=before[0]['size'],
            provenance=dict(distribution='torch',version=dist.version,site=str(site),member=str(member),record_sha256=entry.value))
    observed={canonical_path(p) for p in lib.iterdir() if p.name.lower().endswith('.dll') and p.name.lower().startswith(EXPORT_COMPILER_PREFIXES)}
    if set(rows)!=observed or not rows:raise ValueError('Compiler DLLs missing from distribution RECORD/inventory')
    return dict(schema_version=1,source='admitted Torch distribution RECORD',version=version,files=rows,
        allowed_binaries={p:r['sha256'] for p,r in rows.items()},
        binary_identities={p:{k:r[k] for k in ('file_id','bytes')} for p,r in rows.items()})


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for slab in iter(lambda:stream.read(1024**2),b''):h.update(slab)
    return h.hexdigest()


def canonical_path(path):return os.path.normcase(str(Path(path).resolve()))


def file_identity(path):
    value=Path(path).stat()
    return dict(device=value.st_dev,file_id=value.st_ino,size=value.st_size)


def loaded_binaries(*, selected):
    api=c.WinDLL('kernel32',use_last_error=True)
    api.GetCurrentProcess.restype=c.c_void_p
    api.K32EnumProcessModules.argtypes=[c.c_void_p,c.c_void_p,c.c_ulong,c.POINTER(c.c_ulong)]
    api.K32EnumProcessModules.restype=c.c_int
    api.GetModuleFileNameW.argtypes=[c.c_void_p,c.c_wchar_p,c.c_ulong]
    handles=(c.c_void_p*2048)();needed=c.c_ulong()
    if not api.K32EnumProcessModules(api.GetCurrentProcess(),handles,c.sizeof(handles),c.byref(needed)) or needed.value>c.sizeof(handles):
        raise RuntimeError('loaded module inventory unavailable/oversized')
    result={}
    for handle in handles[:needed.value//c.sizeof(c.c_void_p)]:
        buf=c.create_unicode_buffer(32768)
        if not api.GetModuleFileNameW(handle,buf,len(buf)):raise OSError('loaded module path unavailable')
        path=Path(buf.value)
        if selected(path.name):result[canonical_path(path)]=file_hash(path)
    return result

