"""Typed resources for explicitly selected native operations.

Constructing these objects performs metadata validation. Native producers are
called by the executor after the dry-run and explicit feature authorization.
Live resources remain caller-owned. Recipe resources use the native launcher.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from ._codec import canonical, digest, fingerprint


@dataclass(frozen=True)
class ArtifactSource:
    """Size/SHA-pinned file dependency. Paths are locators, hashes are identity."""
    path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        fingerprint(self.sha256)
        if type(self.path) is not str or not Path(self.path).is_absolute():
            raise ValueError('absolute_artifact_path_required')
        if type(self.bytes) is not int or self.bytes < 0:
            raise ValueError('invalid_artifact_size')

    @classmethod
    def from_path(cls, path: str | Path) -> ArtifactSource:
        """Explicitly fingerprint an existing file without deserializing it."""
        path = Path(path).resolve()
        checksum = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                checksum.update(chunk)
        return cls(str(path), checksum.hexdigest(), path.stat().st_size)

    def verify(self) -> dict[str, Any]:
        """Recheck a locator and preserve distinct missing/mismatch states."""
        try:
            actual = self.from_path(self.path)
        except FileNotFoundError:
            return dict(status='missing', path=self.path, sha256=self.sha256)
        except OSError:
            return dict(status='unreadable', path=self.path, sha256=self.sha256)
        return dict(status='verified' if actual == self else 'mismatch',
                    path=self.path, sha256=self.sha256, bytes=self.bytes)


@dataclass(frozen=True)
class OperationPolicy:
    """Per-operation consent, separate from CPUAnalysisPolicy and feature consent.

    Bounds come from the supplied native protocols. Live trusted Python retains
    its native in-process timeout and resource semantics. peak_reset_pid names
    the caller-owned allocator process whose counters may be changed.
    """
    allow_capture: bool = False
    allow_training: bool = False
    allow_cuda: bool = False
    allow_camera: bool = False
    allow_long_run: bool = False
    allow_power_sampling: bool = False
    allow_file_writes: bool = False
    peak_reset_pid: int | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == 'peak_reset_pid':
                if value is not None and (type(value) is not int or value < 1):
                    raise ValueError('invalid_peak_reset_pid')
            elif type(value) is not bool:
                raise TypeError('operation_permission_must_be_bool')


@dataclass(frozen=True)
class RetainedSource:
    """Native payload snapshot, optionally derived through a closed reader path.

    Examples include P03 -> P04, P14 -> P12/P13/P18, P19 -> P22 and
    P31 -> P32/P33. Companions stay separate from the base P21 payload.
    """
    payload: dict[str, Any]
    options: dict[str, Any] = field(default_factory=dict)
    companions: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class InferenceSource:
    """Resident model/input or CPU master/input factory for explicit sweeps."""
    model: Any
    input_tensor: Any = None
    input_factory: Any = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineSource:
    """Trusted sample source and explicit native pipeline stages."""
    model: Any
    source: Any
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecipeSource:
    """Native ColdStartRecipe or TrainingMemoryRecipe; factory import is deferred."""
    recipe: Any
    options: dict[str, Any] = field(default_factory=dict)
    analysis_options: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationSource:
    """Classification model/loader or retained ordered targets/predictions."""
    model: Any = None
    loader: Any = None
    targets: Any = None
    predictions: Any = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualitySource:
    """Explicit cohort, models and task-specific perturbation or trainer factory."""
    model: Any
    cohort: Any
    candidate: Any = None
    perturbation: Any = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingSource:
    """One fresh Trainer or TrainingSession episode, with explicit native options.

    One binding selects a compatible recorder combination. Distinct bindings
    represent distinct episodes and must use distinct Trainer objects.
    """
    trainer: Any
    train_loader: Any = None
    val_loader: Any = None
    session: Any = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_session(cls, session: Any, *, options: dict[str, Any] | None = None) -> TrainingSource:
        """Bind a prepared TrainingSession using its existing fit configuration."""
        from hyso.training import TrainingSession
        if type(session) is not TrainingSession:
            raise TypeError('TrainingSession_required')
        settings = session.settings()['requested']
        native = dict(options or {})
        for key in ('epochs', 'save_last_to', 'resume_from'):
            if key in native and native[key] != settings[key]:
                raise ValueError('session_fit_option_mismatch:' + key)
            native[key] = settings[key]
        return cls(session.trainer, session.train_loader, session.val_loader, session, native)


@dataclass(frozen=True)
class ProtocolSource:
    """Model-free TransferProtocol or UtilizationProtocol resource."""
    protocol: Any


@dataclass(frozen=True)
class RooflineSource:
    """Existing counter sessions or hash-pinned NSYS files for offline analysis.

    kind is counter, nsys_operators or nsys_modules. Native options preserve
    session/target/unit and capture receipts. New profiler capture uses RecipeSource.
    """
    kind: str
    source: Any
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeSource:
    """Owned native runtime recipe and output directory, or retained P31 directory.

    StreamSource, distinct device targets and ORT options remain explicit native
    inputs. Creating this binding never opens the camera or imports the factory.
    """
    recipe: Any = None
    directory: str | None = None
    stream: Any = None
    devices: tuple[Any, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)


Source = (RetainedSource | InferenceSource | PipelineSource | RecipeSource |
          EvaluationSource | QualitySource | TrainingSource | ProtocolSource |
          RooflineSource | RuntimeSource)
SOURCE_TYPES = (RetainedSource, InferenceSource, PipelineSource, RecipeSource,
                EvaluationSource, QualitySource, TrainingSource, ProtocolSource,
                RooflineSource, RuntimeSource)


@dataclass(frozen=True)
class NativeBinding:
    """Bind selected features to one native operation and one or two model labels.

    operation_id is unique within a dry-run. Reusing this binding for several
    selected features shares the producer. Model labels are caller associations.
    Device describes execution, including cpu for offline CUDA artifact analysis.
    """
    operation_id: str
    model_ids: tuple[str, ...]
    features: tuple[str, ...]
    source: Source
    device: str = 'cpu'
    policy: OperationPolicy = OperationPolicy()
    artifacts: tuple[ArtifactSource, ...] = ()

    def __post_init__(self) -> None:
        if type(self.operation_id) is not str or not self.operation_id.strip():
            raise ValueError('operation_id_required')
        for name, allowed_count in (('model_ids', (1, 2)), ('features', range(1, 36))):
            values = getattr(self, name)
            if (type(values) not in (list, tuple) or len(values) not in allowed_count
                    or any(type(x) is not str or not x.strip() for x in values)
                    or len(set(values)) != len(values)):
                raise ValueError('invalid_operation_' + name)
            object.__setattr__(self, name, tuple(values))
        if any(not re.fullmatch(r'P(?:0[1-9]|[12][0-9]|3[0-5])', f) for f in self.features):
            raise ValueError('invalid_operation_feature')
        if type(self.source) not in SOURCE_TYPES or type(self.policy) is not OperationPolicy:
            raise TypeError('typed_source_and_operation_policy_required')
        if type(self.device) is not str or not re.fullmatch(r'cpu|cuda:(0|[1-9][0-9]*)', self.device):
            raise ValueError('explicit_operation_device_required')
        if type(self.artifacts) not in (tuple, list) or any(type(a) is not ArtifactSource for a in self.artifacts):
            raise TypeError('typed_artifact_sources_required')
        object.__setattr__(self, 'artifacts', tuple(self.artifacts))

    def metadata(self) -> dict[str, Any]:
        """Snapshot live identities and metadata without iterating a data source."""
        return dict(operation_id=self.operation_id, model_ids=list(self.model_ids),
                    features=list(self.features), device=self.device,
                    policy=_metadata(self.policy), source=_metadata(self.source),
                    artifacts=[a.verify() for a in self.artifacts])


def _metadata(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int, float):
        return value
    if type(value) in (tuple, list):
        return [_metadata(item) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError('metadata_keys_must_be_strings')
        return {key: _metadata(item) for key, item in value.items()}
    if type(value) is RetainedSource:
        return dict(type='RetainedSource', payload_sha256=digest(value.payload),
                    format=value.payload.get('format'), schema_version=value.payload.get('schema_version'),
                    options=_metadata(value.options), companions=[digest(item) for item in value.companions])
    name = type(value).__module__ + '.' + type(value).__qualname__
    if isinstance(value, Path):
        return str(value)
    import torch
    if isinstance(value, torch.Tensor):
        try:
            version = value._version
        except RuntimeError:
            version = None
        return dict(type=name, identity=id(value), shape=list(value.shape), device=str(value.device),
                    dtype=str(value.dtype), version=version, requires_grad=value.requires_grad)
    if isinstance(value, torch.nn.Module):
        return dict(type=name, identity=id(value), training=value.training,
                    tensors={key: _metadata(tensor) for key, tensor in
                             (*value.named_parameters(), *value.named_buffers())})
    if is_dataclass(value) and type(value).__module__.startswith('hyso.'):
        if hasattr(value, '_payload'):
            raw = value._payload
            if type(value).__name__ in ('ColdStartRecipe', 'TrainingMemoryRecipe'):
                return dict(type=name, configuration=json.loads(raw))
            return dict(type=name, payload_sha256=hashlib.sha256(raw.encode('utf-8')).hexdigest(),
                        bytes=len(raw.encode('utf-8')))
        return dict(type=name, fields={item.name: _metadata(getattr(value, item.name)) for item in fields(value)})
    from hyso.training import Trainer, TrainingSession
    if type(value) is Trainer:
        return dict(type=name, identity=id(value), device=str(value.device),
                    model=_metadata(value.model), optimizer=id(value.optimizer),
                    epochs=value._epochs, updates=value.update_count,
                    callbacks=[id(callback) for callback in value.callbacks])
    if type(value) is TrainingSession:
        return dict(type=name, identity=id(value), settings=value.settings(), trainer=_metadata(value.trainer))
    if isinstance(value, torch.utils.data.DataLoader):
        return dict(type=name, identity=id(value), dataset=id(value.dataset),
                    batch_size=value.batch_size, num_workers=value.num_workers,
                    sampler=id(value.sampler), drop_last=value.drop_last)
    # Callback, cohort or custom loader identity is reviewable without invoking
    # caller code. Native validation remains authoritative during execution.
    return dict(type=name, identity=id(value))
