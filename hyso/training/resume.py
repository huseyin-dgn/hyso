"""Bounded Generic Trainer checkpoint plans and save-only metadata.

No deserialization or Trainer epoch-resume is implemented here. The legacy v1
serializer owns component/RNG capture and atomic replacement. Private coupling
to its canonical scheduler state and the existing loader descriptor is explicit.
"""

import hashlib
import inspect
import json
import math
import sys
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


def _scheduler_plan(trainer):
    scheduler = trainer.scheduler
    if scheduler is None:
        return None
    cls = torch.optim.lr_scheduler
    if type(scheduler) is cls.OneCycleLR:
        dynamic = {"last_epoch", "_step_count", "_get_lr_called_within_step", "_last_lr"}
        config = {key: value for key, value in checkpoint._scheduler_state(scheduler).items() if key not in dynamic}
    else:
        fields = {cls.StepLR: ("step_size", "gamma", "base_lrs"),
                  cls.CosineAnnealingLR: ("T_max", "eta_min", "base_lrs"),
                  cls.ReduceLROnPlateau: ("mode", "factor", "patience", "threshold", "threshold_mode", "cooldown", "min_lrs", "eps")}
        state = checkpoint._scheduler_state(scheduler)
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
        config = {key: getattr(scheduler, key) for key in fields[type(scheduler)]}
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
    if [g["names"] for g in current_optimizer["groups"]] != [g["names"] for g in plan["optimizer"]["groups"]]:
        raise checkpoint.TrainingCheckpointError("optimizer parameter names/order/groups changed")
    for current, original in zip(current_optimizer["groups"], plan["optimizer"]["groups"]):
        before, after = dict(original["options"]), dict(current["options"])
        # Native scheduler-owned LR/momentum evolve; all other hyperparameters
        # must still agree with the original caller-created training plan.
        if trainer.scheduler is not None:
            before.pop("lr", None)
            after.pop("lr", None)
        if type(trainer.scheduler) is torch.optim.lr_scheduler.OneCycleLR and trainer.scheduler.cycle_momentum:
            if "betas" in before:
                before["betas"] = before["betas"][1:]
                after["betas"] = after["betas"][1:]
            else:
                before.pop("momentum", None)
                after.pop("momentum", None)
        if before != after:
            raise checkpoint.TrainingCheckpointError("optimizer hyperparameter plan changed during training")
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
