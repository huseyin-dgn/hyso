"""Opt-in training-state serialization, separate from legacy model-only files.

Create the model, optimizer and scheduler before loading. Pass the scheduler
itself (not LRSchedulerCallback). Only state_dict data and plain metadata are
stored. Modes, gradients, loaders and callback state are caller responsibilities.
"""

from collections import OrderedDict
import copy
import os
from pathlib import Path
import pickle
import random
import tempfile

import torch

__all__ = ["save_training_checkpoint", "load_training_checkpoint", "TrainingCheckpointError"]
FORMAT_VERSION = 1
_FIELDS = {"format", "format_version", "model", "optimizer", "scheduler", "scaler",
           "ema", "completed_epochs", "history", "metrics", "rng_state"}


class TrainingCheckpointError(ValueError):
    """Invalid, unsupported or incompatible training-state checkpoint."""


def _type_name(obj):
    return type(obj).__module__ + "." + type(obj).__qualname__


def _safe(value, path="state"):
    # Reject custom pickle globals, including subclasses of plain scalar types.
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) is torch.Tensor:
        return value.detach().clone()
    if type(value) in (dict, OrderedDict):
        result = OrderedDict() if type(value) is OrderedDict else {}
        for key, item in value.items():
            if type(key) not in (str, int):
                raise TrainingCheckpointError(path + ": keys must be strings or integers")
            result[key] = _safe(item, path + "." + str(key))
        if hasattr(value, "_metadata"):
            result._metadata = _safe(value._metadata, path + "._metadata")
        return result
    if type(value) in (list, tuple):
        return type(value)(_safe(item, path) for item in value)
    raise TrainingCheckpointError(path + ": unsupported value type " + _type_name(value))


def _component(obj, name):
    if not callable(getattr(obj, "state_dict", None)) or not callable(getattr(obj, "load_state_dict", None)):
        raise TrainingCheckpointError(name + ": provide an initialized state_dict component")
    state = _safe(obj.state_dict(), name)
    if not isinstance(state, dict):
        raise TrainingCheckpointError(name + ": state_dict must be a mapping")
    return {"type": _type_name(obj), "state_dict": state}


def _param_names(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    try:
        return [[names[id(parameter)] for parameter in group["params"]]
                for group in optimizer.param_groups]
    except (KeyError, AttributeError) as error:
        raise TrainingCheckpointError("optimizer parameters must belong to the supplied model") from error


def _rng_state():
    import numpy as np
    kind, keys, position, has_gauss, cached = np.random.get_state()
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
            "numpy": (kind, keys.tolist(), position, has_gauss, cached),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def _restore_rng(state):
    import numpy as np
    kind, keys, position, has_gauss, cached = state["numpy"]
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def _validate_rng(state, restore):
    if state is None:
        if restore:
            raise TrainingCheckpointError("rng_state is absent; cannot restore RNG")
        return
    if type(state) is not dict or set(state) != {"python", "torch_cpu", "numpy", "cuda"}:
        raise TrainingCheckpointError("rng_state: invalid fields")
    try:
        import numpy as np
        random.Random().setstate(state["python"])
        torch.Generator(device="cpu").set_state(state["torch_cpu"].cpu())
        kind, keys, position, has_gauss, cached = state["numpy"]
        np.random.RandomState().set_state((kind, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
        if state["cuda"] is not None:
            if type(state["cuda"]) is not list or not all(
                    type(item) is torch.Tensor and item.dtype == torch.uint8 and item.ndim == 1
                    for item in state["cuda"]):
                raise ValueError("invalid CUDA RNG tensors")
            if restore and (not torch.cuda.is_initialized() or len(state["cuda"]) != torch.cuda.device_count()):
                raise ValueError("CUDA RNG restoration requires initialized matching CUDA devices")
    except Exception as error:
        raise TrainingCheckpointError("rng_state: " + str(error)) from error


def _model_state(target, state, name):
    expected = target.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise TrainingCheckpointError(name + ": state_dict keys differ")
    for key, value in expected.items():
        saved = state[key]
        if isinstance(value, torch.Tensor):
            if type(saved) is not torch.Tensor or saved.shape != value.shape or saved.dtype != value.dtype:
                raise TrainingCheckpointError(name + ": incompatible tensor " + key)


def _validate(payload, model, optimizer, scheduler, scaler, ema, restore_rng):
    if type(payload) is not dict or payload.get("format") != "hyso.training":
        raise TrainingCheckpointError("not a HYSo full training checkpoint (model-only files are separate)")
    if type(payload.get("format_version")) is not int or payload["format_version"] != FORMAT_VERSION:
        raise TrainingCheckpointError("unsupported training checkpoint format_version")
    if set(payload) != _FIELDS:
        raise TrainingCheckpointError("invalid checkpoint fields; missing=" + str(sorted(_FIELDS - set(payload)))
                                      + "; unexpected=" + str(list(set(payload) - _FIELDS)))
    if type(payload["completed_epochs"]) is not int or payload["completed_epochs"] < 0:
        raise TrainingCheckpointError("completed_epochs must be a non-negative integer")
    for name in ("history", "metrics"):
        if payload[name] is not None and type(payload[name]) is not dict:
            raise TrainingCheckpointError(name + " must be a plain dictionary or None")
    if not isinstance(model, torch.nn.Module):
        raise TrainingCheckpointError("model must be a torch.nn.Module")
    if scheduler is not None and (optimizer is None or getattr(scheduler, "optimizer", None) is not optimizer):
        raise TrainingCheckpointError("scheduler must reference the supplied optimizer")
    for name, target in (("model", model), ("optimizer", optimizer), ("scheduler", scheduler),
                         ("scaler", scaler), ("ema", ema)):
        saved = payload[name]
        if (saved is None) != (target is None):
            raise TrainingCheckpointError(name + ": checkpoint/target presence differs; no state is silently skipped")
        if target is None:
            continue
        fields = {"type", "state_dict"}
        if name == "optimizer":
            fields.add("param_names")
        if name == "ema":
            fields.update(("decay", "model_type"))
        if type(saved) is not dict or set(saved) != fields or saved["type"] != _type_name(target):
            raise TrainingCheckpointError(name + ": incompatible component type or fields")
        if not isinstance(saved["state_dict"], dict):
            raise TrainingCheckpointError(name + ": invalid state_dict")
        if name == "model":
            _model_state(target, saved["state_dict"], name)
        elif name == "ema":
            if (not isinstance(getattr(target, "ema_model", None), torch.nn.Module)
                    or saved["model_type"] != _type_name(target.ema_model)
                    or type(saved["decay"]) not in (int, float) or not 0 <= saved["decay"] <= 1):
                raise TrainingCheckpointError("ema: incompatible shadow or decay")
            _model_state(target.ema_model, saved["state_dict"], "ema")
        elif name == "optimizer":
            state = saved["state_dict"]
            if (set(state) != {"state", "param_groups"} or type(state["state"]) is not dict
                    or type(state["param_groups"]) is not list
                    or saved["param_names"] != _param_names(model, target)):
                raise TrainingCheckpointError("optimizer: incompatible state or parameter ordering")
            groups = target.state_dict()["param_groups"]
            if len(groups) != len(state["param_groups"]) or any(
                    type(a) is not dict or "params" not in a or len(a["params"]) != len(b["params"])
                    for a, b in zip(state["param_groups"], groups)):
                raise TrainingCheckpointError("optimizer: incompatible parameter groups")
            ids = [item for group in state["param_groups"] for item in group["params"]]
            parameters = [item for group in target.param_groups for item in group["params"]]
            if (not all(type(item) is int for item in ids) or len(set(ids)) != len(ids)
                    or not set(state["state"]).issubset(ids)):
                raise TrainingCheckpointError("optimizer: invalid parameter state identifiers")
            for identifier, parameter in zip(ids, parameters):
                entry = state["state"].get(identifier, {})
                if type(entry) is not dict:
                    raise TrainingCheckpointError("optimizer: parameter state must be a dictionary")
                if isinstance(target, (torch.optim.Adam, torch.optim.AdamW)) and entry:
                    required = {"step", "exp_avg", "exp_avg_sq"}
                    if not required.issubset(entry) or any(type(entry[key]) is not torch.Tensor
                                                         for key in ("exp_avg", "exp_avg_sq")):
                        raise TrainingCheckpointError("optimizer: missing or invalid Adam moments")
                for key, value in entry.items():
                    if type(value) is torch.Tensor:
                        if (key == "step" and value.numel() != 1) or (key != "step" and value.shape != parameter.shape):
                            raise TrainingCheckpointError("optimizer: incompatible state tensor " + str(key))
        else:
            current = _component(target, name)["state_dict"]
            if set(current) != set(saved["state_dict"]):
                raise TrainingCheckpointError(name + ": state_dict fields differ")
            for key, value in current.items():
                restored = saved["state_dict"][key]
                if type(value) in (bool, int, str) and type(restored) is not type(value):
                    raise TrainingCheckpointError(name + ": incompatible scalar field " + key)
                if type(value) is float and type(restored) not in (int, float):
                    raise TrainingCheckpointError(name + ": incompatible numeric field " + key)
    _validate_rng(payload["rng_state"], restore_rng)


def save_training_checkpoint(path, model, *, optimizer=None, scheduler=None,
                             scaler=None, ema=None, completed_epochs=0,
                             history=None, metrics=None, include_rng=False):
    """Atomically save version 1 state. Omitted components are recorded as None.

    RNG capture is opt-in (Python, NumPy, CPU torch, and already-initialized CUDA).
    Call at a completed optimizer/epoch boundary; in-flight gradients are not saved.
    """
    if not isinstance(model, torch.nn.Module):
        raise TrainingCheckpointError("model must be a torch.nn.Module")
    payload = {"format": "hyso.training", "format_version": FORMAT_VERSION,
               "model": _component(model, "model"), "optimizer": None, "scheduler": None,
               "scaler": None, "ema": None, "completed_epochs": completed_epochs,
               "history": _safe(history, "history"), "metrics": _safe(metrics, "metrics"),
               "rng_state": _safe(_rng_state()) if include_rng else None}
    for name, obj in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        if obj is not None:
            payload[name] = _component(obj, name)
    if optimizer is not None:
        payload["optimizer"]["param_names"] = _param_names(model, optimizer)
    if ema is not None:
        if not isinstance(getattr(ema, "ema_model", None), torch.nn.Module) or not hasattr(ema, "decay"):
            raise TrainingCheckpointError("ema: provide an initialized EMA shadow and decay")
        payload["ema"] = _component(ema.ema_model, "ema")
        payload["ema"].update(type=_type_name(ema), model_type=_type_name(ema.ema_model), decay=_safe(ema.decay))
    _validate(payload, model, optimizer, scheduler, scaler, ema, False)
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=str(destination.parent),
                                         prefix="." + destination.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(destination))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_training_checkpoint(path, model, *, optimizer=None, scheduler=None,
                             scaler=None, ema=None, map_location="cpu", restore_rng=False,
                             validate_metadata=None):
    """Strictly restore caller-created components and return epoch/history/metrics.

    Presence, component types, model tensor schemas and optimizer parameter order
    must match. RNG restoration is opt-in. Never retries with unsafe pickle.
    An optional caller validator receives epoch/history/metrics and the scheduler
    record before any target/RNG mutation.
    """
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
        payload = _safe(payload, "checkpoint")
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as error:
        raise TrainingCheckpointError("cannot read weights-only training checkpoint: " + str(error)) from error
    try:
        _validate(payload, model, optimizer, scheduler, scaler, ema, restore_rng)
        if validate_metadata is not None:
            validate_metadata(copy.deepcopy({key: payload[key] for key in
                                             ("completed_epochs", "history", "metrics", "scheduler")}))
    except TrainingCheckpointError:
        raise
    except (TypeError, KeyError, AttributeError, ValueError) as error:
        raise TrainingCheckpointError("malformed training checkpoint state: " + str(error)) from error
    targets = [("model", model)]
    targets.extend((name, obj) for name, obj in
                   (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)) if obj is not None)
    if ema is not None:
        targets.append(("ema", ema.ema_model))
    backups = [(name, target, copy.deepcopy(target.state_dict())) for name, target in targets]
    decay_before = ema.decay if ema is not None else None
    rng_before = _rng_state() if restore_rng else None
    try:
        for name, target in targets:
            target.load_state_dict(payload[name]["state_dict"])
        if ema is not None:
            ema.decay = payload["ema"]["decay"]
        if restore_rng:
            _restore_rng(payload["rng_state"])
    except Exception as error:
        try:
            for _, target, state in backups:
                target.load_state_dict(state)
            if ema is not None:
                ema.decay = decay_before
            if restore_rng:
                _restore_rng(rng_before)
        except Exception as rollback_error:
            raise TrainingCheckpointError("load failed and rollback failed; targets may be partially restored") from rollback_error
        raise TrainingCheckpointError("component restore failed; previous state restored: " + str(error)) from error
    return {"completed_epochs": payload["completed_epochs"], "history": payload["history"],
            "metrics": payload["metrics"], "rng_state": payload["rng_state"], "rng_restored": bool(restore_rng)}

