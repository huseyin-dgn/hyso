"""Retain actual loaded ORT/CUDA DLL identities in the admitted native worker."""
import atexit
import ctypes as c
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from tools.final_acceptance.common import file_hash,new_json
from .feature_evidence import SafetyViolation


class DLLIntegrityError(ValueError,SafetyViolation):pass


def canonical_path(path):return os.path.normcase(str(Path(path).resolve()))


def file_identity(path):
    value=Path(path).stat()
    return dict(device=value.st_dev,file_id=value.st_ino,size=value.st_size)


def driver_dependency():
    """Resolve the installed driver package before admission; never use loaded paths."""
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'SYSTEM\CurrentControlSet\Services\nvlddmkm') as key:
        image=winreg.QueryValueEx(key,'ImagePath')[0]
    system=Path(os.environ['SYSTEMROOT'])
    expanded=str(system)+image[len('\\SystemRoot'):] if image.lower().startswith('\\systemroot\\') else os.path.expandvars(image)
    driver=Path(expanded.removeprefix('\\??\\')).resolve();store=(system/'System32/DriverStore/FileRepository').resolve()
    if not driver.is_relative_to(store) or driver.name.lower()!='nvlddmkm.sys':raise DLLIntegrityError('unverified installed NVIDIA driver service image')
    wrapper=system/'System32/nvcuda.dll';loader=driver.parent/'nvcuda_loader64.dll';backend=driver.parent/'nvcuda64.dll'
    if not all(p.is_file() and p.resolve().is_relative_to(store) for p in (loader,backend,driver)):
        raise DLLIntegrityError('installed NVIDIA driver dependency escaped DriverStore or is missing')
    if not os.path.samefile(wrapper,loader) or file_hash(wrapper)!=file_hash(loader):raise DLLIntegrityError('NVIDIA loader hardlink identity mismatch')
    if 'nvcuda64.dll'.encode('utf-16-le') not in loader.read_bytes():raise DLLIntegrityError('NVIDIA loader does not reference registered package backend')
    paths=[wrapper,loader,backend,driver]
    before={canonical_path(p):(file_hash(p),file_identity(p),p.stat().st_mtime_ns) for p in paths}
    script=r'''$ErrorActionPreference='Stop'
[Console]::InputEncoding=[Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$paths=ConvertFrom-Json -InputObject ([Console]::In.ReadToEnd())
@($paths | ForEach-Object {
    $file=Get-Item -LiteralPath $_
    $signature=Get-AuthenticodeSignature -LiteralPath $_
    [pscustomobject]@{path=$file.FullName;version=$file.VersionInfo.FileVersion;company=$file.VersionInfo.CompanyName;
        signature=[string]$signature.Status;signer=$signature.SignerCertificate.Subject;thumbprint=$signature.SignerCertificate.Thumbprint}
}) | ConvertTo-Json -Compress -Depth 4
'''
    powershell=system/'System32/WindowsPowerShell/v1.0/powershell.exe'
    result=subprocess.run([str(powershell),'-NoLogo','-NoProfile','-NonInteractive','-Command',script],
        input=json.dumps([str(p) for p in paths]),capture_output=True,text=True,encoding='utf-8',timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW,check=True)
    metadata=json.loads(result.stdout);versions={row['version'] for row in metadata}
    if len(metadata)!=4 or len(versions)!=1 or any(row['signature']!='Valid' or row['company']!='NVIDIA Corporation' or
        not any(publisher in row['signer'] for publisher in ('Microsoft Windows Hardware Compatibility Publisher','NVIDIA')) for row in metadata):
        raise DLLIntegrityError('NVIDIA driver signature/version/company verification failed')
    if before!={canonical_path(p):(file_hash(p),file_identity(p),p.stat().st_mtime_ns) for p in paths}:
        raise DLLIntegrityError('NVIDIA dependency changed during signature verification')
    files={role:dict(row,path=canonical_path(path),sha256=file_hash(path),identity=file_identity(path))
        for role,path,row in zip(('wrapper','loader','backend','service_driver'),paths,metadata)}
    # Keep canonical paths, independent of the display spelling returned by PowerShell.
    for value in files.values():value['path']=canonical_path(value['path'])
    return dict(source='HKLM/SYSTEM/CurrentControlSet/Services/nvlddmkm/ImagePath',registry_image=image,
        package_directory=canonical_path(driver.parent),loader_backend_reference='nvcuda64.dll',files=files)


def _selected(name):
    name=name.lower()
    return name.endswith(('.dll','.pyd')) and name.startswith(('onnxruntime','cud','cublas','nvcuda','torch_cuda'))


def ort_inventory(site,torch_site):
    site=Path(site);torch_site=Path(torch_site)
    system=Path(os.environ['SYSTEMROOT'])/'System32'
    directories={site/'onnxruntime/capi',torch_site/'torch/lib',system}
    for base in (site,torch_site):
        directories.update((base/'nvidia').glob('*/bin'))
    paths={p.resolve() for directory in directories if directory.is_dir() for p in directory.iterdir()
        if _selected(p.name) and p.is_file() and (directory!=system or p.name.lower().startswith('nvcuda'))}
    dependency=driver_dependency()
    paths.add(Path(dependency['files']['backend']['path']))
    versions={d.metadata.get('Name','').lower().replace('_','-'):d.version for d in importlib.metadata.distributions(path=[str(site)])}
    return dict(site=str(site.resolve()),version=versions.get('onnxruntime-gpu'),
        allowed_binaries={canonical_path(p):file_hash(p) for p in sorted(paths)},driver_dependency=dependency)


def loaded_binaries():
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
        if _selected(path.name):result[os.path.normcase(str(path.resolve()))]=file_hash(path)
    return result


def register_ort_identity(path):
    import onnxruntime as ort
    path=Path(path)
    if os.environ.get('HYSO_RUNTIME_ADMITTED_PID')!=str(os.getpid()):raise RuntimeError('ORT identity requires admitted worker')
    def capture():
        try:
            value=dict(pid=os.getpid(),admitted_pid=os.environ['HYSO_RUNTIME_ADMITTED_PID'],python=sys.executable,
                source=str(Path(ort.__file__).resolve()),version=ort.__version__,
                visible_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),loaded_binaries=loaded_binaries())
        except Exception as exc:value=dict(pid=os.getpid(),error=repr(exc))
        new_json(path,value)
    atexit.register(capture)


def validate_ort_identity(identity,result,expected):
    if not isinstance(identity,dict) or identity.get('error'):raise ValueError('ORT loaded identity unavailable: '+str(identity))
    e=result['evidence'];pid=e['controller']['launcher_pid']
    if identity['pid']!=pid or identity['admitted_pid']!=str(pid):raise ValueError('ORT worker identity mismatch')
    if Path(identity['source']).resolve()!=(Path(expected['site'])/'onnxruntime/__init__.py').resolve():raise ValueError('ORT runtime source mismatch')
    if identity['version']!=expected['version'] or identity['version']!=e['report']['runtime']['version']:raise ValueError('ORT runtime version mismatch')
    if identity['visible_uuid']!=e['recipe']['gpu_uuid']:raise ValueError('ORT visibility mismatch')
    def canonical_map(values):
        result={}
        for path,sha in values.items():
            key=canonical_path(path)
            if key in result:raise DLLIntegrityError('ORT ambiguous duplicate canonical DLL path')
            result[key]=sha
        return result
    actual=canonical_map(identity['loaded_binaries']);allowed=canonical_map(expected['allowed_binaries'])
    if not actual:raise DLLIntegrityError('ORT loaded DLL inventory empty')
    for path,sha in actual.items():
        if path not in allowed:raise DLLIntegrityError('ORT loaded DLL path is not admitted: '+path)
        if allowed[path]!=sha:raise DLLIntegrityError('ORT loaded DLL hash mismatch: '+path+' expected='+allowed[path]+' actual='+sha)
    names={Path(path).name.lower() for path in actual}
    if not {'onnxruntime_providers_cuda.dll','onnxruntime_pybind11_state.pyd','nvcuda.dll'}<=names:
        raise ValueError('ORT CUDA provider/driver/extension not actually loaded')
    if not any(n.startswith('cudnn') for n in names) or not any(n.startswith('cudart') for n in names):
        raise ValueError('ORT CUDA/cuDNN dependency identity unavailable')
