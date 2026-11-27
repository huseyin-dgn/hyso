"""Stdlib-only runtime discovery and RECORD-verified native package provenance."""
from __future__ import annotations
import base64
import importlib.metadata
import os
from pathlib import Path
import hashlib

class ExportError(ValueError):
    """Inventory-only error; no optional framework import in the controller."""
    def __init__(self,code,reason):
        self.code=code;super().__init__(code+': '+reason)

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def runtime_inventory(python: str | Path) -> dict:
    """Inspect selected Windows interpreter without importing Torch/TRT/CUDA."""
    python=Path(python).resolve(strict=True)
    if not python.is_file() or python.name.lower()!='python.exe':raise ExportError('RUNTIME_SPEC','Windows python.exe required')
    candidates=(python.parent/'Lib/site-packages',python.parent.parent/'Lib/site-packages')
    site=next((p.resolve() for p in candidates if p.is_dir()),None)
    if site is None:raise ExportError('RUNTIME_SPEC','selected interpreter site-packages missing')
    versions={};files={};identities={};provenance={};issues=[]
    for dist in importlib.metadata.distributions(path=[str(site)]):
        name=dist.metadata.get('Name','').lower().replace('_','-')
        if name in versions:issues.append('duplicate distribution: '+name)
        versions[name]=dist.version
        if not name.startswith('tensorrt'):continue
        for member in dist.files or []:
            path=Path(dist.locate_file(member))
            if path.suffix.lower() not in ('.dll','.pyd'):continue
            resolved=path.resolve(strict=True)
            if not resolved.is_relative_to(site):raise ExportError('DLL_INTEGRITY','TensorRT RECORD escapes selected site')
            for part in (path,*path.parents):
                if part==site:break
                if part.is_symlink() or os.name=='nt' and part.lstat().st_file_attributes&0x400:
                    raise ExportError('DLL_INTEGRITY','TensorRT reparse member rejected')
            before=resolved.stat();entry=member.hash
            if entry is None or entry.mode!='sha256' or member.size!=before.st_size:
                raise ExportError('DLL_INTEGRITY','TensorRT file differs from distribution RECORD: '+str(member)+
                    ' (record_bytes='+str(member.size)+', actual_bytes='+str(before.st_size)+')')
            sha=file_hash(resolved)
            if base64.urlsafe_b64encode(bytes.fromhex(sha)).decode().rstrip('=')!=entry.value:
                raise ExportError('DLL_INTEGRITY','TensorRT SHA256 differs from distribution RECORD: '+str(member))
            after=resolved.stat()
            if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):raise ExportError('DLL_INTEGRITY','TensorRT inventory changed during inspection')
            key=os.path.normcase(str(resolved))
            if key in files:raise ExportError('DLL_INTEGRITY','duplicate TensorRT native member')
            files[key]=sha;identities[key]=dict(file_id=before.st_ino,bytes=before.st_size)
            provenance[key]=dict(distribution=name,version=dist.version,member=str(member),record_sha256=entry.value)
    invalid=[p.name for p in site.iterdir() if p.name.startswith(('~umpy','-umpy','~numpy'))]
    issues+=['invalid NumPy remnant: '+p for p in invalid]
    if 'onnxruntime' in versions and 'onnxruntime-gpu' in versions:issues.append('conflicting ORT CPU/GPU distributions')
    trt={n:v for n,v in versions.items() if n.startswith('tensorrt')}
    required={'tensorrt-cu12','tensorrt-cu12-bindings','tensorrt-cu12-libs'}
    missing=sorted(required-set(trt))
    if missing:issues.append('missing TensorRT distributions: '+', '.join(missing))
    if trt and len(set(trt.values()))!=1:issues.append('TensorRT distribution versions disagree')
    ready=bool(files) and not missing and all(v.startswith('10.') for v in trt.values()) and not issues
    return dict(python=str(python),python_sha256=file_hash(python),site=str(site),versions=versions,issues=issues,
        tensorrt=dict(distributions=trt,allowed_binaries=files,binary_identities=identities,provenance=provenance,
                      status='READY_FOR_ADMISSION' if ready else 'BLOCKED',runtime_verified=False))
