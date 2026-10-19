"""Bounded Generic Trainer checkpoint plans, save and transactional restore.

The legacy v1 serializer owns component capture/load and atomic replacement. Private coupling
to its canonical scheduler state and the existing loader descriptor is explicit.
"""

import hashlib
import inspect
import json
import math
import sys
import copy
import io
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hyso import checkpoint
from hyso.models.bcam import _resume_loader_plan
from hyso.models.modular import ModularCNN
from hyso.models.WithSE import EMA
from hyso.callbacks.callbacks import EarlyStopping


def _plain(value):
    """Canonical finite configuration: no scalar subclasses, tensors or pickles."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (tuple, list):
        return [_plain(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _plain(value[key]) for key in sorted(value)}
    raise checkpoint.TrainingCheckpointError("checkpoint configuration requires plain finite values")


def _type(obj):
    return type(obj).__module__ + "." + type(obj).__qualname__


def _structure(model):
    result = []
    for name, module in model.named_modules():
        if any(key in vars(module) for key in ("forward", "forward_features", "extra_repr")) or any(
                getattr(module, key, {}) for key in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")):
            raise checkpoint.TrainingCheckpointError("checkpoint model must not have instance overrides or hooks")
        result.append({"name": name, "type": _type(module), "config": module.extra_repr()})
    return result


def _model_plan(trainer):
    model = trainer.model
    if type(model) is not ModularCNN:
        # TypeError also makes the unsupported save_last_to model boundary clear.
        raise TypeError("save_last_to requires the standard ModularCNN checkpoint profile")
    if trainer.model_config is None:
        raise checkpoint.TrainingCheckpointError("checkpoint save requires caller-provided model_config")
    actual = dict(input_channels=model.input_channels, num_classes=model.num_classes,
                  widths=model.widths, repeats=model.repeats, strides=model.strides,
                  attention=model.attention_name, norm=model.norm_name)
    config = _plain(trainer.model_config)
    if json.dumps(config, sort_keys=True) != json.dumps(_plain(actual), sort_keys=True):
        raise checkpoint.TrainingCheckpointError("model_config must exactly describe all ModularCNN constructor fields")
    # Build only a structural reference; do not consume caller RNG or move the
    # live model. This detects same-weight-shape stride/activation alterations.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        reference = ModularCNN(**config)
    structure = _structure(model)
    if structure != _structure(reference):
        raise checkpoint.TrainingCheckpointError("model structure differs from the declared ModularCNN config")
    tensors = []
    for kind, items, expected in (
            ("parameter", model.named_parameters(), dict(reference.named_parameters())),
            ("buffer", model.named_buffers(), dict(reference.named_buffers()))):
        names = []
        for name, tensor in items:
            names.append(name)
            if getattr(tensor, "_backward_hooks", None):
                raise checkpoint.TrainingCheckpointError("checkpoint model tensors must not have custom gradient hooks")
            if name not in expected or tensor.shape != expected[name].shape:
                raise checkpoint.TrainingCheckpointError("model tensor structure differs from config")
            tensors.append(dict(name=name, kind=kind, shape=list(tensor.shape), dtype=str(tensor.dtype),
                                device=str(tensor.device), requires_grad=tensor.requires_grad))
        if names != list(expected):
            raise checkpoint.TrainingCheckpointError("model tensor names/order differ from config")
    code = {}
    for module_name in sorted({type(m).__module__ for m in model.modules() if type(m).__module__.startswith("hyso.")}):
        code[module_name] = hashlib.sha256(inspect.getsource(sys.modules[module_name]).encode("utf-8")).hexdigest()
    convs = [{"name": name, "kernel_size": list(m.kernel_size), "stride": list(m.stride),
              "padding": list(m.padding), "dilation": list(m.dilation), "groups": m.groups,
              "bias": m.bias is not None, "padding_mode": m.padding_mode}
             for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    return _plain(dict(type=_type(model), config=config, modules=structure, convolutions=convs,
                       tensors=tensors, source_sha256=code))


def _data_plan(trainer, loader):
    if loader is None:
        return None
    if type(loader) is not DataLoader or type(loader.dataset) is not TensorDataset:
        raise checkpoint.TrainingCheckpointError("checkpoint save requires standard DataLoader(TensorDataset)")
    if (set(vars(loader.dataset)) != {"tensors"} or loader.worker_init_fn is not None
            or any(callable(value) for key, value in vars(loader).items() if key != "collate_fn")):
        raise checkpoint.TrainingCheckpointError("checkpoint loader/dataset must not have custom methods or worker initialization")
    if any(type(tensor) is not torch.Tensor for tensor in loader.dataset.tensors):
        raise checkpoint.TrainingCheckpointError("checkpoint dataset requires standard dense CPU tensors")
    plan = _resume_loader_plan(loader)
    if (loader.batch_sampler.sampler is not loader.sampler
            or loader.batch_sampler.batch_size != loader.batch_size
            or loader.batch_sampler.drop_last != loader.drop_last):
        raise checkpoint.TrainingCheckpointError("checkpoint loader batch/sampler configuration is inconsistent")
    if len(loader.dataset.tensors) != 2:
        raise checkpoint.TrainingCheckpointError("checkpoint TensorDataset requires inputs and targets")
    inputs, targets = loader.dataset.tensors
    if (inputs.ndim != 4 or inputs.shape[1] != trainer.model.input_channels
            or any(size <= 0 for size in inputs.shape)
            or inputs.dtype not in (torch.float32, torch.float64) or not torch.isfinite(inputs).all()):
        raise checkpoint.TrainingCheckpointError("checkpoint inputs must be finite CPU classification NCHW tensors")
    count = len(inputs)
    if trainer.task == "multiclass":
        if (targets.shape != (count,) or targets.dtype != torch.long
                or (targets == trainer.loss_fn.ignore_index).any()
                or ((targets < 0) | (targets >= trainer.model.num_classes)).any()):
            raise checkpoint.TrainingCheckpointError("checkpoint multiclass targets must be valid hard class indices")
    elif (targets.shape not in ((count,), (count, 1)) or not targets.is_floating_point()
          or not torch.isfinite(targets).all() or not ((targets == 0) | (targets == 1)).all()):
        raise checkpoint.TrainingCheckpointError("checkpoint binary targets must be finite floating hard labels")
    if trainer.amp and inputs.dtype != torch.float32:
        raise checkpoint.TrainingCheckpointError("checkpoint AMP inputs must be float32")
    if inputs.dtype != next(trainer.model.parameters()).dtype:
        raise checkpoint.TrainingCheckpointError("checkpoint input dtype must match model master parameters")
    return _plain(dict(type=_type(loader), dataset_type=_type(loader.dataset),
                       num_workers=loader.num_workers, pin_memory=loader.pin_memory,
                       pin_memory_device=getattr(loader, "pin_memory_device", ""), **plan))


def _finite_optimizer(value):
    """The native serializer checks shape/type; reject nonfinite optimizer state too."""
    if type(value) is torch.Tensor:
        if not torch.isfinite(value).all():
            raise checkpoint.TrainingCheckpointError("checkpoint optimizer state must be finite")
    elif type(value) is float and not math.isfinite(value):
        raise checkpoint.TrainingCheckpointError("checkpoint optimizer state must be finite")
    elif isinstance(value, dict):
        for item in value.values():
            _finite_optimizer(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_optimizer(item)


def _optimizer_plan(trainer):
    names = {id(p): name for name, p in trainer.model.named_parameters()}
    groups = []
    for group in trainer.optimizer.param_groups:
        try:
            group_names = [names[id(p)] for p in group["params"]]
        except KeyError as exc:
            raise checkpoint.TrainingCheckpointError("optimizer contains foreign parameters") from exc
        groups.append(dict(names=group_names, options=_plain({key: value for key, value in group.items() if key != "params"})))
    return dict(type=_type(trainer.optimizer), groups=groups)


def _scheduler_plan(trainer, state=None):
    scheduler = trainer.scheduler
    if scheduler is None:
        return None
    cls = torch.optim.lr_scheduler
    state = checkpoint._scheduler_state(scheduler) if state is None else state
    if type(scheduler) is cls.OneCycleLR:
        checkpoint._check_onecycle_state(state)
        dynamic = {"last_epoch", "_step_count", "_get_lr_called_within_step", "_last_lr"}
        config = {key: value for key, value in state.items() if key not in dynamic}
    else:
        fields = {cls.StepLR: ("step_size", "gamma", "base_lrs"),
                  cls.CosineAnnealingLR: ("T_max", "eta_min", "base_lrs"),
                  cls.ReduceLROnPlateau: ("mode", "factor", "patience", "threshold", "threshold_mode", "cooldown", "min_lrs", "eps")}
        if type(scheduler) is cls.ReduceLROnPlateau:
            expected = {"factor", "min_lrs", "patience", "verbose", "cooldown", "cooldown_counter", "mode",
                        "threshold", "threshold_mode", "best", "num_bad_epochs", "mode_worse", "eps", "last_epoch"}
            # _last_lr is absent in fresh Torch 2.2.2 Plateau and present after
            # its first step (always present in the tested 2.5.1 runtime).
            if "_last_lr" in state:
                expected.add("_last_lr")
        else:
            expected = {"base_lrs", "last_epoch", "verbose", "_step_count", "_get_lr_called_within_step", "_last_lr"}
            expected.update(("step_size", "gamma") if type(scheduler) is cls.StepLR else ("T_max", "eta_min"))
        if set(state) != expected:
            raise checkpoint.TrainingCheckpointError("scheduler contains unsupported state fields")
        config = {key: state[key] for key in fields[type(scheduler)]}
    return dict(type=_type(scheduler), config=_plain(config))


def _early_config(trainer):
    es = trainer.early_stopping
    if es is None:
        return None
    if (type(es) is not EarlyStopping or type(es.patience) is not int or es.patience <= 0
            or type(es.delta) not in (int, float) or not math.isfinite(es.delta) or es.delta < 0
            or type(es.restore_best_weights) is not bool or type(es.verbose) is not bool):
        raise checkpoint.TrainingCheckpointError("invalid typed EarlyStopping configuration")
    return _plain(dict(type=_type(es), patience=es.patience, delta=es.delta,
                       restore_best_weights=es.restore_best_weights, verbose=es.verbose,
                       target="ema" if trainer.ema is not None else "live"))


def _early_state(trainer):
    es = trainer.early_stopping
    if es is None:
        return None
    config = _early_config(trainer)
    if type(es.counter) is not int or es.counter < 0 or type(es.early_stop) is not bool:
        raise checkpoint.TrainingCheckpointError("invalid EarlyStopping counter/terminal state")
    if es.early_stop != (es.counter >= es.patience):
        raise checkpoint.TrainingCheckpointError("EarlyStopping terminal flag and patience counter disagree")
    if es.best_state is None:
        if es.best_loss != math.inf or es.counter != 0 or es.early_stop:
            raise checkpoint.TrainingCheckpointError("invalid initial EarlyStopping state")
        best_loss, best_state = None, None
    else:
        if type(es.best_loss) not in (int, float) or not math.isfinite(es.best_loss):
            raise checkpoint.TrainingCheckpointError("EarlyStopping best_loss must be finite")
        expected = trainer.evaluation_model.state_dict()
        if not isinstance(es.best_state, dict) or list(es.best_state) != list(expected):
            raise checkpoint.TrainingCheckpointError("EarlyStopping best snapshot keys/order differ")
        for name, value in es.best_state.items():
            if (type(value) is not torch.Tensor or value.shape != expected[name].shape
                    or value.dtype != expected[name].dtype or not torch.isfinite(value).all()):
                raise checkpoint.TrainingCheckpointError("EarlyStopping best snapshot tensor schema is invalid")
        best_loss, best_state = es.best_loss, checkpoint._safe(es.best_state, "early_stopping.best_state")
    return dict(**config, counter=es.counter, early_stop=es.early_stop,
                best_loss=best_loss, best_state=best_state)


def prepare_save(trainer, train_loader, val_loader, epochs, path):
    """Validate the bounded save profile before iterator creation or updates."""
    if not isinstance(path, (str, Path)):
        raise TypeError("save_last_to must be a filesystem path")
    destination = Path(path)
    if not destination.name or not destination.parent.is_dir() or destination.is_dir():
        raise checkpoint.TrainingCheckpointError("save_last_to requires an existing parent directory and a file path")
    if trainer.callbacks:
        raise checkpoint.TrainingCheckpointError("checkpoint save requires callbacks=()")
    if trainer.optimizer._optimizer_step_pre_hooks or trainer.optimizer._optimizer_step_post_hooks:
        raise checkpoint.TrainingCheckpointError("checkpoint save does not serialize custom optimizer hooks")
    model = _model_plan(trainer)
    if type(trainer.scaler) is not torch.cuda.amp.GradScaler or trainer.scaler.is_enabled() != trainer.amp:
        raise checkpoint.TrainingCheckpointError("checkpoint save requires the native matching GradScaler")
    if trainer.ema is not None and type(trainer.ema) is not EMA:
        raise checkpoint.TrainingCheckpointError("checkpoint save requires the native EMA")
    scaler = trainer.scaler.state_dict()
    plan = _plain(dict(profile="modular-tensor-v1", epochs=epochs, task=trainer.task, model=model,
                       loss=dict(type=_type(trainer.loss_fn), reduction=trainer.loss_fn.reduction,
                                 ignore_index=getattr(trainer.loss_fn, "ignore_index", None),
                                 label_smoothing=getattr(trainer.loss_fn, "label_smoothing", None)),
                       optimizer=_optimizer_plan(trainer), scheduler=_scheduler_plan(trainer),
                       device=str(trainer.device), amp=trainer.amp, accum_steps=trainer.accum_steps,
                       grad_clip=trainer.grad_clip, scaler=dict(type=_type(trainer.scaler), initial_state=scaler),
                       ema=None if trainer.ema is None else dict(type=_type(trainer.ema), decay=trainer.ema.decay, cadence="applied_update"),
                       early_stopping=_early_config(trainer), train_loader=_data_plan(trainer, train_loader),
                       val_loader=_data_plan(trainer, val_loader), callbacks=[],
                       runtime=dict(torch=str(torch.__version__), python=list(sys.version_info[:2]))))
    _early_state(trainer)  # Validate typed state, including initial infinity normalization.
    # Reject arbitrary state added to supported components before any updates.
    _finite_optimizer(checkpoint._component(trainer.optimizer, "optimizer")["state_dict"])
    if trainer.scheduler is not None:
        checkpoint._component(trainer.scheduler, "scheduler")
    return destination, plan


def save_last(trainer, destination, plan, train_loader, val_loader):
    """Save a coherent completed epoch, before best-weight finalization."""
    trainer._validate_setup()
    if _model_plan(trainer) != plan["model"] or _data_plan(trainer, train_loader) != plan["train_loader"] or _data_plan(trainer, val_loader) != plan["val_loader"]:
        raise checkpoint.TrainingCheckpointError("model or data plan changed during training")
    current_optimizer = _optimizer_plan(trainer)
    _finite_optimizer(trainer.optimizer.state_dict())
    _check_optimizer_plan(current_optimizer, plan["optimizer"], trainer.scheduler)
    if _scheduler_plan(trainer) != plan["scheduler"] or _early_config(trainer) != plan["early_stopping"]:
        raise checkpoint.TrainingCheckpointError("scheduler/EarlyStopping plan changed during training")
    completed = len(trainer.history["train_loss"])
    _plain(trainer.history)
    attempts = completed * ((len(train_loader) + trainer.accum_steps - 1) // trainer.accum_steps)
    if (not 0 < completed <= plan["epochs"]
            or any(type(value) is not int for value in (trainer.attempt_count, trainer.update_count, trainer.microbatch_count, trainer.ema_update_count))
            or any(len(values) != completed for values in trainer.history.values()) or trainer.attempt_count != attempts
            or trainer.microbatch_count != completed * len(train_loader) or not 0 <= trainer.update_count <= attempts
            or trainer.ema_update_count != (trainer.update_count if trainer.ema is not None else 0)):
        raise checkpoint.TrainingCheckpointError("incoherent Trainer progress/history")
    if trainer.scheduler is not None:
        expected = trainer.update_count if type(trainer.scheduler) is torch.optim.lr_scheduler.OneCycleLR else completed
        if trainer.scheduler.last_epoch != expected:
            raise checkpoint.TrainingCheckpointError("scheduler counter differs from Trainer progress")
    for tensor in list(trainer.model.state_dict().values()) + ([] if trainer.ema is None else list(trainer.ema.ema_model.state_dict().values())):
        if type(tensor) is not torch.Tensor or not torch.isfinite(tensor).all():
            raise checkpoint.TrainingCheckpointError("checkpoint model state must contain finite tensors")
    early = _early_state(trainer)
    progress = dict(completed_epochs=completed, microbatches=trainer.microbatch_count,
                    attempts=trainer.attempt_count, updates=trainer.update_count, ema_updates=trainer.ema_update_count,
                    early_stop=False if early is None else early["early_stop"],
                    model_modes={name: module.training for name, module in trainer.model.named_modules()},
                    ema_modes=None if trainer.ema is None else {name: module.training for name, module in trainer.ema.ema_model.named_modules()})
    if any(type(flag) is not bool for flag in progress["model_modes"].values()) or (
            progress["ema_modes"] is not None and any(type(flag) is not bool for flag in progress["ema_modes"].values())):
        raise checkpoint.TrainingCheckpointError("model mode maps must contain bool flags")
    checkpoint.save_training_checkpoint(destination, trainer.model, optimizer=trainer.optimizer,
                                        scheduler=trainer.scheduler, scaler=trainer.scaler, ema=trainer.ema,
                                        completed_epochs=completed, history=trainer.history,
                                        metrics={"generic_trainer": dict(schema_version=1, plan=plan,
                                                                         progress=_plain(progress), early_stopping=early)},
                                        include_rng=True)


def _check_optimizer_plan(current, original, scheduler):
    if current["type"] != original["type"] or [g["names"] for g in current["groups"]] != [g["names"] for g in original["groups"]]:
        raise checkpoint.TrainingCheckpointError("optimizer parameter names/order/groups changed")
    for actual, expected in zip(current["groups"], original["groups"]):
        before, after = dict(expected["options"]), dict(actual["options"])
        # Native scheduler-owned LR/momentum evolve; all other hyperparameters
        # must still agree with the original caller-created training plan.
        if scheduler is not None:
            before.pop("lr", None)
            after.pop("lr", None)
        if type(scheduler) is torch.optim.lr_scheduler.OneCycleLR and scheduler.cycle_momentum:
            if "betas" in before:
                before["betas"] = before["betas"][1:]
                after["betas"] = after["betas"][1:]
            else:
                before.pop("momentum", None)
                after.pop("momentum", None)
        if json.dumps(before, sort_keys=True) != json.dumps(after, sort_keys=True):
            raise checkpoint.TrainingCheckpointError("optimizer hyperparameter plan changed during training")


def _require(condition, message):
    if not condition:
        raise checkpoint.TrainingCheckpointError(message)


def _mode_map(model, saved):
    _require(type(saved) is dict and set(saved) == set(dict(model.named_modules()))
             and all(type(flag) is bool for flag in saved.values()), "invalid model/EMA mode map")


def _stage_early(trainer, saved, losses):
    expected = _early_config(trainer)
    if expected is None:
        _require(saved is None, "unexpected EarlyStopping state")
        return None
    fields = set(expected) | {"counter", "early_stop", "best_loss", "best_state"}
    _require(type(saved) is dict and set(saved) == fields, "invalid EarlyStopping metadata fields")
    _require(json.dumps(_plain({key: saved[key] for key in expected}), sort_keys=True) == json.dumps(expected, sort_keys=True),
             "EarlyStopping config/target mismatch")
    es = copy.copy(trainer.early_stopping)
    for key in ("counter", "early_stop", "best_loss", "best_state"):
        setattr(es, key, copy.deepcopy(saved[key]))
    if es.best_loss is None:
        es.best_loss = math.inf
    proxy = copy.copy(trainer)
    proxy.early_stopping = es
    _early_state(proxy)  # Reuse the B4-A typed serializer's schema checks.
    replay = EarlyStopping(patience=es.patience, delta=es.delta, verbose=False,
                           restore_best_weights=es.restore_best_weights)
    for index, loss in enumerate(losses):
        replay(loss, trainer.evaluation_model)
        _require(not replay.early_stop or index == len(losses) - 1, "history continued beyond terminal EarlyStopping")
    _require((es.best_loss, es.counter, es.early_stop) == (replay.best_loss, replay.counter, replay.early_stop),
             "EarlyStopping counters/best loss disagree with validation history")
    return es


def _stage_restore(trainer, payload, expected_plan):
    """Validate every native component and the complete Trainer namespace before apply."""
    scheduler = trainer.scheduler
    # Native 2.2.2 Plateau adds _last_lr only on its first step. Stage the
    # equivalent initialized schema on a copy; never mutate the live preflight.
    if (type(scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau and not hasattr(scheduler, "_last_lr")
            and payload.get("scheduler") is not None and "_last_lr" in payload["scheduler"].get("state_dict", {})):
        scheduler = copy.copy(scheduler)
        scheduler._last_lr = [group["lr"] for group in trainer.optimizer.param_groups]
    checkpoint._validate(payload, trainer.model, trainer.optimizer, scheduler, trainer.scaler, trainer.ema, True)
    metrics = payload["metrics"]
    _require(type(metrics) is dict and set(metrics) == {"generic_trainer"}, "missing/invalid generic_trainer namespace")
    metadata = metrics["generic_trainer"]
    _require(type(metadata) is dict and set(metadata) == {"schema_version", "plan", "progress", "early_stopping"},
             "invalid generic_trainer fields")
    _require(type(metadata["schema_version"]) is int and metadata["schema_version"] == 1, "unsupported generic_trainer schema_version")
    plan = _plain(metadata["plan"])
    _require(json.dumps(plan, sort_keys=True) == json.dumps(expected_plan, sort_keys=True), "Trainer model/training/data plan mismatch")
    completed, history, progress = payload["completed_epochs"], payload["history"], metadata["progress"]
    _require(type(completed) is int and 0 < completed <= plan["epochs"], "invalid completed_epochs")
    _require(type(history) is dict and set(history) == {"train_loss", "train_acc", "val_loss", "val_acc"}, "invalid history fields")
    for key, values in history.items():
        _require(type(values) is list and len(values) == completed, "invalid history length")
        for value in values:
            if key.startswith("val_") and plan["val_loader"] is None:
                _require(value is None, "validation history must be None without validation")
            else:
                _require(type(value) in (int, float) and math.isfinite(value) and value >= 0
                         and (not key.endswith("acc") or value <= 100), "invalid finite history metric")
    fields = {"completed_epochs", "microbatches", "attempts", "updates", "ema_updates", "early_stop", "model_modes", "ema_modes"}
    _require(type(progress) is dict and set(progress) == fields, "invalid progress fields")
    for key in ("completed_epochs", "microbatches", "attempts", "updates", "ema_updates"):
        _require(type(progress[key]) is int and progress[key] >= 0, "progress counters must be nonnegative integers")
    batches = plan["train_loader"]["batches"]
    attempts = completed * ((batches + plan["accum_steps"] - 1) // plan["accum_steps"])
    _require(progress["completed_epochs"] == completed and progress["microbatches"] == completed * batches
             and progress["attempts"] == attempts and 0 <= progress["updates"] <= attempts, "incoherent progress counters")
    _require(plan["amp"] or progress["updates"] == attempts, "non-AMP updates must equal attempts")
    _require(progress["ema_updates"] == (progress["updates"] if trainer.ema is not None else 0), "invalid EMA update counter")
    _mode_map(trainer.model, progress["model_modes"])
    if trainer.ema is None:
        _require(progress["ema_modes"] is None, "unexpected EMA mode map")
    else:
        _mode_map(trainer.ema.ema_model, progress["ema_modes"])
        _require(payload["ema"]["decay"] == plan["ema"]["decay"], "EMA decay mismatch")
    early = _stage_early(trainer, metadata["early_stopping"], history["val_loss"])
    _require(type(progress["early_stop"]) is bool and progress["early_stop"] == (early.early_stop if early else False), "invalid terminal flag")
    optimizer = payload["optimizer"]
    current_groups = [dict(names=names, options=_plain({key: value for key, value in group.items() if key != "params"}))
                      for names, group in zip(optimizer["param_names"], optimizer["state_dict"]["param_groups"])]
    _check_optimizer_plan(dict(type=optimizer["type"], groups=current_groups), plan["optimizer"], trainer.scheduler)
    _finite_optimizer(optimizer["state_dict"])
    if trainer.scheduler is not None:
        state = payload["scheduler"]["state_dict"]
        _require(_scheduler_plan(trainer, state) == plan["scheduler"], "scheduler configuration mismatch")
        expected = progress["updates"] if type(trainer.scheduler) is torch.optim.lr_scheduler.OneCycleLR else completed
        _require(type(state["last_epoch"]) is int and state["last_epoch"] == expected, "scheduler counter mismatch")
        if type(trainer.scheduler) is not torch.optim.lr_scheduler.ReduceLROnPlateau:
            _require(type(state["_step_count"]) is int and state["_step_count"] == expected + 1, "scheduler step counter mismatch")
        _require(state["_last_lr"] == [g["lr"] for g in optimizer["state_dict"]["param_groups"]], "scheduler/optimizer LR mismatch")
    state = payload["scaler"]["state_dict"]
    if not plan["amp"]:
        _require(state == {}, "disabled scaler must have empty state")
    else:
        _require(set(state) == {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}, "invalid scaler fields")
        _plain(state)
        _require(type(state["scale"]) in (int, float) and state["scale"] > 0, "invalid scaler scale")
        initial = plan["scaler"]["initial_state"]
        _require(all(state[key] == initial[key] for key in ("growth_factor", "backoff_factor", "growth_interval")), "scaler policy mismatch")
        _require(type(state["_growth_tracker"]) is int and 0 <= state["_growth_tracker"] < state["growth_interval"], "invalid scaler growth tracker")
    for name in ("model", "ema"):
        if payload[name] is not None:
            _require(all(type(t) is torch.Tensor and torch.isfinite(t).all() for t in payload[name]["state_dict"].values()), "nonfinite model/EMA state")
    return dict(completed=completed, history=copy.deepcopy(history), progress=copy.deepcopy(progress), early=early)


_TRAINER_FIELDS = ("history", "microbatch_count", "attempt_count", "update_count", "ema_update_count", "_epochs", "_finished", "_finalized")


def _snapshot_runtime(trainer):
    components = [("model", trainer.model), ("optimizer", trainer.optimizer), ("scaler", trainer.scaler)]
    if trainer.scheduler is not None:
        components.append(("scheduler", trainer.scheduler))
    if trainer.ema is not None:
        components.append(("ema", trainer.ema.ema_model))
    return dict(components=[(name, target, copy.deepcopy(checkpoint._scheduler_state(target) if name == "scheduler" else target.state_dict()))
                            for name, target in components],
                trainer={key: copy.deepcopy(getattr(trainer, key)) for key in _TRAINER_FIELDS},
                early=None if trainer.early_stopping is None else copy.deepcopy(vars(trainer.early_stopping)),
                decay=None if trainer.ema is None else trainer.ema.decay,
                modes=[(module, module.training) for _, target in components if isinstance(target, nn.Module) for module in target.modules()],
                gradients=[(p, None if p.grad is None else p.grad.detach().clone()) for p in trainer.model.parameters()],
                scheduler_fields=None if trainer.scheduler is None else set(trainer.scheduler.state_dict()),
                scaler_runtime=copy.deepcopy(vars(trainer.scaler)),
                rng=checkpoint._rng_state())


def _apply_early(trainer, staged):
    if staged is None:
        return
    state = copy.deepcopy(vars(staged))
    if state["best_state"] is not None:
        expected = trainer.evaluation_model.state_dict()
        for key, value in state["best_state"].items():
            state["best_state"][key] = value.to(device=expected[key].device).clone()
    vars(trainer.early_stopping).clear()
    vars(trainer.early_stopping).update(state)


def _apply_progress(trainer, staged, epochs):
    trainer.history = copy.deepcopy(staged["history"])
    for attribute, key in (("microbatch_count", "microbatches"), ("attempt_count", "attempts"),
                           ("update_count", "updates"), ("ema_update_count", "ema_updates")):
        setattr(trainer, attribute, staged["progress"][key])
    trainer._epochs, trainer._finished, trainer._finalized = epochs, False, False
    for parameter in trainer.model.parameters():
        parameter.grad = None


def _apply_modes(trainer, progress):
    for model, key in ((trainer.model, "model_modes"), (None if trainer.ema is None else trainer.ema.ema_model, "ema_modes")):
        if model is not None:
            for name, module in model.named_modules():
                module.training = progress[key][name]


def _rollback(trainer, snapshot):
    for name, target, state in snapshot["components"]:
        if name == "scheduler":
            checkpoint._scheduler_load_state(target, state)
            for key in set(target.state_dict()) - snapshot["scheduler_fields"]:
                vars(target).pop(key, None)
        else:
            target.load_state_dict(state)
    if trainer.ema is not None:
        trainer.ema.decay = snapshot["decay"]
    vars(trainer.scaler).clear()
    vars(trainer.scaler).update(snapshot["scaler_runtime"])
    if trainer.early_stopping is not None:
        vars(trainer.early_stopping).clear()
        vars(trainer.early_stopping).update(snapshot["early"])
    for key, value in snapshot["trainer"].items():
        setattr(trainer, key, value)
    for module, training in snapshot["modes"]:
        module.training = training
    for parameter, gradient in snapshot["gradients"]:
        parameter.grad = gradient
    checkpoint._restore_rng(snapshot["rng"])


def restore_checkpoint(trainer, path, plan):
    """Restore one immutable weights-only snapshot, with an outer runtime transaction."""
    try:
        blob = Path(path).read_bytes()
        payload = checkpoint._safe(torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True), "checkpoint")
        staged = _stage_restore(trainer, payload, plan)
    except checkpoint.TrainingCheckpointError:
        raise
    except Exception as error:
        raise checkpoint.TrainingCheckpointError("resume_from: invalid weights-only checkpoint: " + str(error)) from error
    snapshot = _snapshot_runtime(trainer)
    try:
        if (type(trainer.scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau
                and not hasattr(trainer.scheduler, "_last_lr")):
            trainer.scheduler._last_lr = [group["lr"] for group in trainer.optimizer.param_groups]
        checkpoint.load_training_checkpoint(io.BytesIO(blob), trainer.model, optimizer=trainer.optimizer,
                                            scheduler=trainer.scheduler, scaler=trainer.scaler, ema=trainer.ema,
                                            map_location="cpu", restore_rng=False)
        if trainer.scaler.is_enabled():
            # Native state_dict omits in-flight per-optimizer bookkeeping.
            # A completed-epoch restore starts with no pending unscale/step.
            trainer.scaler._per_optimizer_states.clear()
        _apply_early(trainer, staged["early"])
        _apply_progress(trainer, staged, plan["epochs"])
        _apply_modes(trainer, staged["progress"])
        terminal = staged["progress"]["early_stop"]
        if terminal or staged["completed"] == plan["epochs"]:
            es = trainer.early_stopping
            if es is not None and es.restore_best_weights:
                trainer._finalized = True
                es.restore_weights(trainer.evaluation_model)
            trainer._finalized = trainer._finalized or terminal
            trainer._finished = True
        # Nothing after this consumes training RNG or opens a loader iterator.
        checkpoint._restore_rng(payload["rng_state"])
    except Exception as error:
        try:
            _rollback(trainer, snapshot)
        except Exception as rollback_error:
            raise checkpoint.TrainingCheckpointError("restore failed and rollback failed; runtime may be partially restored") from rollback_error
        raise checkpoint.TrainingCheckpointError("Trainer restore failed; previous runtime restored: " + str(error)) from error
    return staged["completed"]
