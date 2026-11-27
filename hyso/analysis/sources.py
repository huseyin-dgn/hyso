"""Typed resources for explicitly selected native operations.

Constructing these objects performs metadata validation. Native producers are
called by the executor after the dry-run and explicit feature authorization.
Live resources remain caller-owned. Recipe resources use the native launcher.
"""
from __future__ import annotations
import typing as _typing

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
    """Size/SHA-pinned file dependency. Paths are locators, hashes are identity.

    Args:
        path (str): Filesystem path for the specific reader/exporter; no configuration code is executed.
            Required.
        sha256 (str): Exact canonical SHA-256 content digest. Required.
        bytes (int): Observed byte count under the declared metric scope. Required.

    Returns:
        An instance of ArtifactSource with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        ArtifactSource(path=path, sha256=sha256, bytes=bytes)

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """
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
        """Explicitly fingerprint an existing file without deserializing it.

        Args:
            path (str | Path): Filesystem path for the specific reader/exporter; no configuration code is
                executed. Required.

        Returns:
            A validated ArtifactSource instance reconstructed from the supplied data.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            ArtifactSource.from_path(path)

        HYSo Guide: analyzer.workflow
        Offline: hyso docs show analyzer.workflow --lang en
        Open: hyso docs open analyzer.workflow --lang en
        """
        path = Path(path).resolve()
        checksum = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                checksum.update(chunk)
        return cls(str(path), checksum.hexdigest(), path.stat().st_size)

    def verify(self) -> dict[str, Any]:
        """Recheck a locator and preserve distinct missing/mismatch states.

        Returns:
            The retained verify value, without recomputing measurements.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.verify()

        HYSo Guide: analyzer.workflow
        Offline: hyso docs show analyzer.workflow --lang en
        Open: hyso docs open analyzer.workflow --lang en
        """
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

    Args:
        allow_capture (bool): Explicit permission for the supported capture operation. Default: False.
        allow_training (bool): Explicit permission for the selected native training operation. Default:
            False.
        allow_cuda (bool): Explicit CUDA permission flag; this alone does not satisfy GPU admission.
            Default: False.
        allow_camera (bool): Explicit permission for the supported camera operation; source binding and
            guards still apply. Default: False.
        allow_long_run (bool): Explicit permission for the bounded long-run operation. Default: False.
        allow_power_sampling (bool): Explicit permission for supported power-sensor sampling. Default:
            False.
        allow_file_writes (bool): Explicit permission for native output file writes within the operation
            scope. Default: False.
        peak_reset_pid (int | None): Process identity associated with peak-memory reset evidence.
            Default: None.

    Returns:
        An instance of OperationPolicy with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        OperationPolicy()

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
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

    Args:
        payload (dict[str, Any]): Plain retained evidence payload. Required.
        options (dict[str, Any]): Existing keyword options of the delegated API; unknown options are not
            silently inferred. Default: field(default_factory=dict).
        companions (tuple[dict[str, Any], ...]): Related retained records required to interpret this
            evidence. Default: ().

    Returns:
        An instance of RetainedSource with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        RetainedSource(payload=payload)

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """
    payload: dict[str, Any]
    options: dict[str, Any] = field(default_factory=dict)
    companions: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class InferenceSource:
    """Resident model/input or CPU master/input factory for explicit sweeps.

    Args:
        model (Any): Caller-owned model; device, dtype and task must match the native operation.
            Required.
        input_tensor (Any): Caller-supplied input Tensor with the declared shape/device/dtype. Default:
            None.
        input_factory (Any): Trusted input factory bound to this measurement. Default: None.
        options (dict[str, Any]): Existing keyword options of the delegated API; unknown options are not
            silently inferred. Default: field(default_factory=dict).

    Returns:
        An instance of InferenceSource with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        InferenceSource(model=model)

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """
    model: Any
    input_tensor: Any = None
    input_factory: Any = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineSource:
    """Trusted sample source and explicit native pipeline stages.

    Args:
        model (Any): Caller-owned model; device, dtype and task must match the native operation.
            Required.
        source (Any): Typed native/retained source required by this API, not a declaration of producer
            authority. Required.
        options (dict[str, Any]): Existing keyword options of the delegated API; unknown options are not
            silently inferred. Default: field(default_factory=dict).

    Returns:
        An instance of PipelineSource with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        PipelineSource(model=model, source=source)

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """
    model: Any
    source: Any
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecipeSource:
    """Native ColdStartRecipe or TrainingMemoryRecipe; factory import is deferred.

    Args:
        recipe (Any): Trusted construction recipe; factory import/execution is governed by the existing
            operation contract. Required.
        options (dict[str, Any]): Existing keyword options of the delegated API; unknown options are not
            silently inferred. Default: field(default_factory=dict).
        analysis_options (dict[str, dict[str, Any]]): Closed options for the selected native analysis.
            Default: field(default_factory=dict).

    Returns:
        An instance of RecipeSource with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        RecipeSource(recipe=recipe)

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """
    recipe: Any
    options: dict[str, Any] = field(default_factory=dict)
    analysis_options: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationSource:
    """Classification model/loader or retained ordered targets/predictions.

    Args:
        model (Any): Caller-owned model; device, dtype and task must match the native operation.
            Default: None.
        loader (Any): Caller-owned finite DataLoader with the required input/target batch format.
            Default: None.
        targets (Any): Ground-truth targets with the task’s required shape, dtype and label space.
            Default: None.
        predictions (Any): Caller-supplied predictions with the declared label space and sample order.
            Default: None.
        options (dict[str, Any]): Existing keyword options of the delegated API; unknown options are not
            silently inferred. Default: field(default_factory=dict).

    Returns:
        An instance of EvaluationSource with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        EvaluationSource()

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """
    model: Any = None
    loader: Any = None
    targets: Any = None
    predictions: Any = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualitySource:
    """Explicit cohort, models and task-specific perturbation or trainer factory.

    Args:
        model (Any): Caller-owned model; device, dtype and task must match the native operation.
            Required.
        cohort (Any): Fixed evaluation inputs, targets, labels and sample identity. Required.
        candidate (Any): Validated retained candidate source or predictions required by this comparison.
            Default: None.
        perturbation (Any): Explicit perturbed prediction/cohort evidence; no perturbation is
            synthesized implicitly. Default: None.
        options (dict[str, Any]): Existing keyword options of the delegated API; unknown options are not
            silently inferred. Default: field(default_factory=dict).

    Returns:
        An instance of QualitySource with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        QualitySource(model=model, cohort=cohort)

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """
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

    Args:
        trainer (Any): Caller-owned standard Trainer; a measurement may execute its fit lifecycle only
            when called explicitly. Required.
        train_loader (Any): Finite training loader; batches must follow the task/model input contract.
            Default: None.
        val_loader (Any): Optional finite validation loader; must agree with the explicit validation
            setting. Default: None.
        session (Any): Prepared caller-owned training session; binding it does not call fit(). Default:
            None.
        options (dict[str, Any]): Existing keyword options of the delegated API; unknown options are not
            silently inferred. Default: field(default_factory=dict).

    Returns:
        An instance of TrainingSource with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        TrainingSource(trainer=trainer)

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """
    trainer: Any
    train_loader: Any = None
    val_loader: Any = None
    session: Any = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_session(cls, session: Any, *, options: dict[str, Any] | None = None) -> TrainingSource:
        """Bind a prepared TrainingSession using its existing fit configuration.

        Args:
            session (Any): Prepared caller-owned training session; binding it does not call fit(). Required.
            options (dict[str, Any] | None): Existing keyword options of the delegated API; unknown options
                are not silently inferred. Default: None.

        Returns:
            A validated TrainingSource instance reconstructed from the supplied data.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            TrainingSource.from_session(session)

        HYSo Guide: analyzer.workflow
        Offline: hyso docs show analyzer.workflow --lang en
        Open: hyso docs open analyzer.workflow --lang en
        """
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
    """Model-free TransferProtocol or UtilizationProtocol resource.

    Args:
        protocol (Any): Typed native measurement protocol; scope, budgets and device guards remain
            unchanged. Required.

    Returns:
        An instance of ProtocolSource with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        ProtocolSource(protocol=protocol)

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """
    protocol: Any


@dataclass(frozen=True)
class RooflineSource:
    """Existing counter sessions or hash-pinned NSYS files for offline analysis.

    kind is counter, nsys_operators or nsys_modules. Native options preserve
    session/target/unit and capture receipts. New profiler capture uses RecipeSource.

    Args:
        kind (str): Existing run or artifact kind accepted by this operation. Required.
        source (Any): Typed native/retained source required by this API, not a declaration of producer
            authority. Required.
        options (dict[str, Any]): Existing keyword options of the delegated API; unknown options are not
            silently inferred. Default: field(default_factory=dict).

    Returns:
        An instance of RooflineSource with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        RooflineSource(kind=kind, source=source)

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """
    kind: _typing.Literal['counter', 'nsys_operators', 'nsys_modules']
    source: Any
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeSource:
    """Owned native runtime recipe and output directory, or retained P31 directory.

    StreamSource, distinct device targets and ORT options remain explicit native
    inputs. Creating this binding never opens the camera or imports the factory.

    Args:
        recipe (Any): Trusted construction recipe; factory import/execution is governed by the existing
            operation contract. Default: None.
        directory (str | None): Dedicated output/owned-run directory; existing overwrite and ownership
            rules apply. Default: None.
        stream (Any): Explicit native stream selection; no stream or context is created by a
            declaration. Default: None.
        devices (tuple[Any, ...]): Explicit device identities to compare. Default: ().
        options (dict[str, Any]): Existing keyword options of the delegated API; unknown options are not
            silently inferred. Default: field(default_factory=dict).

    Returns:
        An instance of RuntimeSource with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        RuntimeSource()

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
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

    Args:
        operation_id (str): Exact operation identity. Required.
        model_ids (tuple[str, ...]): Explicit ordered model identities for the recorded workflow.
            Required.
        features (tuple[str, ...]): Explicit feature IDs to compare/plan; prerequisites are never
            silently executed. Required.
        source (Source): Typed native/retained source required by this API, not a declaration of
            producer authority. Required.
        device (str): Explicit cpu or cuda:index device; this setting grants no GPU admission. Default:
            'cpu'.
        policy (OperationPolicy): Existing protocol/resource policy; defaults do not grant additional
            consent. Default: OperationPolicy().
        artifacts (tuple[ArtifactSource, ...]): Explicit retained artifact references. Default: ().

    Returns:
        An instance of NativeBinding with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        NativeBinding(operation_id=operation_id, model_ids=model_ids, features=features, source=source)

    HYSo Guide: analyzer.workflow
    Offline: hyso docs show analyzer.workflow --lang en
    Open: hyso docs open analyzer.workflow --lang en
    """
    operation_id: str
    model_ids: tuple[str, ...]
    features: tuple[_typing.Literal['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35'], ...]
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
        """Snapshot live identities and metadata without iterating a data source.

        Returns:
            The retained metadata value, without recomputing measurements.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.metadata()

        HYSo Guide: analyzer.workflow
        Offline: hyso docs show analyzer.workflow --lang en
        Open: hyso docs open analyzer.workflow --lang en
        """
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
