"""Versioned export records: conversion is distinct from deployment validation."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import importlib.metadata
import importlib.util
import json
import math
import platform
import shutil
from pathlib import Path
from typing import Any, Literal
from ..analysis._codec import digest

STAGES=('dependencies','admission','conversion','artifact','runtime_loading','runtime_execution','equivalence','preservation','cleanup')

class ExportError(ValueError):
    """Export failure with a stable code/message key and retained partial stages.

    Args:
        code (str): Stable machine-readable diagnostic code; distinct from the human-readable reason.
            Required.
        reason (str): Recorded explanation for rejection, absence or incomplete evidence; None only
            where the schema allows it. Required.
        result (dict[str, Any] | None): Retained HYSo result/evidence record; rendering/advice does not
            run measurements. Default: None.
    """
    def __init__(self, code: str, reason: str, *, result: dict[str,Any] | None=None) -> None:
        """Initialize ExportError from the supplied settings or evidence.

        Args:
            code (str): Stable machine-readable diagnostic code; distinct from the human-readable reason.
                Required.
            reason (str): Recorded explanation for rejection, absence or incomplete evidence; None only
                where the schema allows it. Required.
            result (dict[str, Any] | None): Retained HYSo result/evidence record; rendering/advice does not
                run measurements. Default: None.
        """
        self.code=code;self.message_key='export.'+code.lower()
        self.result=dict(result or {})
        if 'stages' not in self.result:
            stage='cleanup' if code=='OWNERSHIP' else 'equivalence' if code in ('OUTPUT_MISMATCH','OUTPUT_INTEGRITY') else 'preservation' if code=='MODEL_PRESERVATION' else 'conversion' if code in ('ENGINE_BUILD','ONNX_PARSER','ENGINE_LOAD','ENGINE_EXECUTION','NATIVE_SAFETY_STOP') else 'admission'
            self.result['stages']={n:dict(status='NOT_RUN') for n in STAGES}
            self.result['stages'][stage]=dict(status='FAIL',reason=reason)
            self.result['deployment_verified']=False
        super().__init__(code+': '+reason)
    def to_dict(self) -> dict[str,Any]:
        """Stable machine fields; technical reason stays English until E12."""
        return dict(code=self.code,message_key=self.message_key,reason=str(self),result=self.result)

@dataclass(frozen=True)
class ExportConfig:
    """Bounded inference export. No implicit overwrite, dtype/device move or retry.

    atol/rtol are explicit elementwise equivalence bounds. max_model_bytes bounds
    the supported resident profile before copying; it is not an OS memory quota.
    Native work additionally requires an owned worker and NativeConfig.

    Args:
        overwrite (bool): Permit replacement of an existing destination when True; False retains
            no-clobber publication. Default: False.
        validate_runtime (bool): Request runtime loading, execution and equivalence checking. False can
            produce an artifact but cannot yield deployment_verified=True. Default: True.
        atol (float): Finite nonnegative absolute output tolerance. Default: 1e-05.
        rtol (float): Finite nonnegative relative output tolerance; relative comparison uses a small
            denominator floor. Default: 0.0001.
        max_model_bytes (int): Resident model byte budget before conversion; positive integer capped at
            256 MiB. Default: 64 * 1024 ** 2.
        max_input_bytes (int): Example-input byte budget; positive integer capped at 64 MiB. Default: 16
            * 1024 ** 2.
        max_artifact_bytes (int): Total artifact/sidecar byte budget; positive integer capped at 512
            MiB. Default: 256 * 1024 ** 2.
    """
    overwrite: bool=False
    validate_runtime: bool=True
    atol: float=1e-5
    rtol: float=1e-4
    max_model_bytes: int=64*1024**2
    max_input_bytes: int=16*1024**2
    max_artifact_bytes: int=256*1024**2
    def __post_init__(self) -> None:
        if type(self.overwrite) is not bool or type(self.validate_runtime) is not bool:raise ExportError('INVALID_CONFIG','boolean flags required')
        for v in (self.atol,self.rtol):
            if type(v) not in (int,float) or not math.isfinite(v) or v<0:raise ExportError('INVALID_TOLERANCE','finite nonnegative atol/rtol required')
        for v,cap in ((self.max_model_bytes,256*1024**2),(self.max_input_bytes,64*1024**2),(self.max_artifact_bytes,512*1024**2)):
            if type(v) is not int or not 0<v<=cap:raise ExportError('RESOURCE_BUDGET','bounded positive byte limits required')

@dataclass(frozen=True)
class NativeConfig:
    """Explicit bounded native request; workspace is not total VRAM.

    GPU execution requires physical UUID, allocator cap and predeclared DLL hashes.
    Known compiler/SDK/platform gaps remain BLOCKED, never a fallback backend.

    Args:
        gpu_uuid (str): Canonical physical GPU-<UUID>, not an ordinal index. Required.
        allowed_binaries (dict[str, str]): Nonempty mapping of absolute admitted DLL paths to lowercase
            64-character SHA256 digests. Required.
        workspace_bytes (int): Native build workspace budget, at most 256 MiB and strictly below
            max_gpu_bytes; not total VRAM. Default: 64 * 1024 ** 2.
        max_gpu_bytes (int): Native device allocation budget, at most 512 MiB. Default: 384 * 1024 ** 2.
        allocator_fraction (float): Positive CUDA allocator fraction, at most 0.15. Default: 0.15.
        timeout_seconds (float): Positive native worker deadline, at most 120 seconds. Default: 90.0.
        process_memory_bytes (int): Per-process host-memory budget in bytes, at most 3 GiB for
            NativeConfig. Default: 3 * 1024 ** 3.
        job_memory_bytes (int): Whole-job host-memory budget in bytes, at most 6 GiB for NativeConfig;
            must cover the process budget. Default: 6 * 1024 ** 3.
        max_processes (int): Maximum processes in the owned native job, at most 4 for NativeConfig.
            Default: 4.
        runtime_search_path (str | None): Explicit native runtime directory used only within the
            admitted loading scope; no global PATH modification. Default: None.
        dependency_identities (dict[str, dict]): Admitted binary path to {file_id, bytes} metadata; keys
            must belong to allowed_binaries and values are nonnegative integers. Default:
            field(default_factory=dict).
    """
    gpu_uuid: str
    allowed_binaries: dict[str,str]
    workspace_bytes: int=64*1024**2
    max_gpu_bytes: int=384*1024**2
    allocator_fraction: float=.15
    timeout_seconds: float=90.
    process_memory_bytes: int=3*1024**3
    job_memory_bytes: int=6*1024**3
    max_processes: int=4
    runtime_search_path: str | None=None
    dependency_identities: dict[str,dict]=field(default_factory=dict)
    def __post_init__(self) -> None:
        from uuid import UUID
        if type(self.dependency_identities) is not dict or any(p not in self.allowed_binaries or
            type(row) is not dict or set(row)!={'file_id','bytes'} or any(type(v) is not int or v<0 for v in row.values())
            for p,row in self.dependency_identities.items()):raise ExportError('NATIVE_ADMISSION','invalid pre-admission dependency identities')
        import re
        try:valid=type(self.gpu_uuid) is str and self.gpu_uuid=='GPU-'+str(UUID(self.gpu_uuid.removeprefix('GPU-')))
        except ValueError:valid=False
        if not valid or type(self.allowed_binaries) is not dict or not self.allowed_binaries:raise ExportError('NATIVE_ADMISSION','physical GPU UUID and pre-admission binary hashes required')
        if any(type(p) is not str or not Path(p).is_absolute() or type(s) is not str or not re.fullmatch('[0-9a-f]{64}',s) for p,s in self.allowed_binaries.items()):raise ExportError('NATIVE_ADMISSION','absolute DLL paths and SHA256 required')
        for v,cap in ((self.workspace_bytes,256*1024**2),(self.max_gpu_bytes,512*1024**2),(self.process_memory_bytes,3*1024**3),(self.job_memory_bytes,6*1024**3),(self.max_processes,4)):
            if type(v) is not int or not 0<v<=cap:raise ExportError('RESOURCE_BUDGET','native request exceeds existing campaign ceilings')
        if self.workspace_bytes>=self.max_gpu_bytes or self.process_memory_bytes>self.job_memory_bytes:raise ExportError('RESOURCE_BUDGET','workspace and host budgets inconsistent')
        if not 0<self.allocator_fraction<=.15 or not 0<self.timeout_seconds<=120:raise ExportError('RESOURCE_BUDGET','native fraction/timeout exceeds bounded profile')

def availability() -> dict[str,Any]:
    """Detect sources without importing optional runtimes or initializing devices.

    Importable means only Python discovery, never working provider/compiler proof.
    No dependency is installed. Versions refer to the current interpreter paths.

    Returns:
        Discovered module paths/versions and runtime_verified=False; discovery is not backend admission.
    """
    rows={}
    for name,dist in (('torch','torch'),('onnx','onnx'),('onnxruntime','onnxruntime'),('onnxscript','onnxscript'),('tensorrt','tensorrt'),('torch_tensorrt','torch-tensorrt')):
        try:spec=importlib.util.find_spec(name)
        except (ValueError,ImportError):spec=None
        try:version=importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            try:version=importlib.metadata.version('onnxruntime-gpu') if name=='onnxruntime' else importlib.metadata.version('tensorrt-cu12') if name=='tensorrt' else None
            except importlib.metadata.PackageNotFoundError:version=None
        rows[name]=dict(discoverable=spec is not None,source=spec.origin if spec else None,version=version,runtime_verified=False)
    return dict(dependencies=rows,platform=platform.platform(),compilers={n:shutil.which(n) for n in ('cmake','cl','g++','nvcc')},
        aotinductor=dict(available=False,platform=platform.system(),reason='Windows/Torch 2.5 AOT deployment is unsupported; this Windows Job adapter does not admit Linux compilation either',runtime_verified=False))

@dataclass(frozen=True,init=False)
class ExportResult:
    """Immutable metadata, independent stages and verifiable multi-file artifact.

    from_json verifies its digest and every artifact hash before returning. Loading
    metadata never executes a model; runtime loading is a separate explicit action.

    Args:
        data (dict[str, Any]): Detached schema-compatible result mapping, including recorded status and
            integrity fields. Required.
    """
    __slots__=('_payload',)
    _payload: str
    def __init__(self, data: dict[str,Any]) -> None:
        """Initialize ExportResult from the supplied settings or evidence.

        Args:
            data (dict[str, Any]): Detached schema-compatible result mapping, including recorded status and
                integrity fields. Required.
        """
        body=json.loads(json.dumps(data,allow_nan=False))
        if body.get('schema_version')!=1 or body.get('format')!='hyso.export' or set(body.get('stages',{}))!=set(STAGES):raise ExportError('RESULT_SCHEMA','invalid export metadata')
        if body.get('content_sha256')!=digest({k:v for k,v in body.items() if k!='content_sha256'}):raise ExportError('RESULT_INTEGRITY','metadata digest mismatch')
        for row in body['stages'].values():
            if row['status'] not in ('PASS','FAIL','BLOCKED','NOT_RUN'):raise ExportError('RESULT_SCHEMA','invalid stage status')
        completed=all(body['stages'][s]['status']=='PASS' for s in STAGES)
        if body.get('deployment_verified')!=completed:raise ExportError('RESULT_SCHEMA','deployment claim contradicts stages')
        if completed:
            rows=body['stages']['equivalence'].get('comparisons',[])
            if not rows or not all(r.get('matched') is True and r.get('finite') is True for r in rows):raise ExportError('RESULT_SCHEMA','actual finite output comparison evidence required')
            from ._evidence import verify_comparisons
            try:verify_comparisons(body)
            except (ValueError,KeyError,TypeError) as exc:raise ExportError('RESULT_INTEGRITY','inconsistent numerical evidence: '+str(exc)) from exc
        if body.get('artifact_format')=='onnx' and body.get('runtime'):
            runtime=body['runtime']
            if 'controller' in runtime and 'report' in runtime:runtime=runtime['report']['runtime']
            binding=runtime.get('artifact_binding',{})
            if binding.get('main_sha256')!=body.get('artifact_sha256'):raise ExportError('RESULT_INTEGRITY','runtime ONNX artifact binding differs')
            files={r['path']:r['sha256'] for r in body['artifact_files']}
            if any(files.get(p)!=sha for p,sha in binding.get('external_files',{}).items()):raise ExportError('RESULT_INTEGRITY','runtime external-data binding differs')
        payload=json.dumps(body,allow_nan=False,sort_keys=True)
        if max(len(payload.encode()),len(json.dumps(body,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8'))+1)>16*1024**2:raise ExportError('RESOURCE_BUDGET','complete metadata/evidence exceeds 16 MiB reload budget')
        object.__setattr__(self,'_payload',payload)
    def to_dict(self) -> dict[str,Any]:
        """Return detached metadata; does not execute or revalidate runtime."""
        return json.loads(self._payload)
    def verify(self) -> None:
        """Verify complete artifact manifest and its canonical path/hash bounds."""
        from .artifacts import verify_files
        verify_files(self.to_dict())
    def to_json(self, path: str | Path | None=None, *, overwrite: bool=False) -> str:
        """Save detached metadata, refusing existing paths unless explicitly chosen.

        Args:
            path (str | Path | None): Filesystem location read or written by this explicit operation; output
                parents must already exist. Default: None.
            overwrite (bool): Permit replacement of an existing destination when True; False retains
                no-clobber publication. Default: False.
        """
        from ..benchmark.result import _write_export
        text=json.dumps(self.to_dict(),ensure_ascii=False,allow_nan=False,indent=2)+'\n'
        if path is not None:_write_export(path,text,overwrite)
        return text
    @classmethod
    def from_json(cls, path: str | Path) -> ExportResult:
        """Read metadata and require intact artifact files; no native deserialization.

        Args:
            path (str | Path): Filesystem location read or written by this explicit operation; output
                parents must already exist. Required.
        """
        p=Path(path)
        if p.is_symlink() or p.stat().st_size>16*1024**2:raise ExportError('RESULT_INTEGRITY','linked/oversized metadata')
        result=cls(json.loads(p.read_text(encoding='utf-8')));result.verify();return result

def seal(body: dict[str,Any]) -> ExportResult:
    body=json.loads(json.dumps(body,allow_nan=False))
    body['deployment_verified']=all(body['stages'][s]['status']=='PASS' for s in STAGES)
    body['content_sha256']=digest({k:v for k,v in body.items() if k!='content_sha256'})
    return ExportResult(body)
