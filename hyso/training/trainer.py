"""Classification updates, independent of model-specific fit APIs."""

import math
import copy
import inspect
from numbers import Real

import torch
from torch import nn

from hyso.models.WithSE import EMA
from hyso.callbacks.callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback


class Trainer:
    """Train a caller-owned model, optimizer and mean classification loss.

    Supports CPU/CUDA float32/float64 and CUDA FP16 AMP with FP32 master weights.
    Move the model before creating the optimizer; this class never moves it.
    Loaders must be finite, sized iterables of (input_tensor, target_tensor).
    Inputs retain their dtype; binary floating targets are cast to logits dtype.
    History describes predictions before each batch's optimizer update, with
    loss weighted by actual sample count and accuracy expressed as a percentage.

    A Trainer owns one fit lifecycle. Repeating its completed epoch budget is
    a no-op; restarting or retrying a failed run requires a fresh Trainer.
    Validation evaluates the EMA shadow when enabled, otherwise the live model.
    EarlyStopping uses that same target and optionally restores its best weights
    after the final coherent epoch boundary. Opt-in save_last_to writes a v1
    checkpoint before that restore, for standard ModularCNN with a complete
    model_config and bounded TensorDataset loaders. resume_from restores that
    same original epoch budget transactionally before opening a new iterator.
    Callbacks are observers: callback(event, independent_plain_logs).
    Optimizers are
    standard non-fused SGD/Adam/AdamW; schedulers are bare OneCycleLR, StepLR
    or CosineAnnealingLR instances. The Trainer owns their stepping.
    """

    def __init__(self, model, *, optimizer, loss_fn, device="cpu", task="multiclass",
                 scheduler=None, amp=False, accum_steps=1, grad_clip=None,
                 ema_decay=None, early_stopping=None, callbacks=(), model_config=None):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an nn.Module")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        if task not in ("multiclass", "binary"):
            raise ValueError("task must be 'multiclass' or 'binary'")
        try:
            requested = torch.device(device)
        except (TypeError, RuntimeError, ValueError) as exc:
            raise ValueError("device must be a valid PyTorch device") from exc
        if requested.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is unavailable")
            index = torch.cuda.current_device() if requested.index is None else requested.index
            if index >= torch.cuda.device_count():
                raise ValueError("CUDA device index is unavailable")
            requested = torch.device("cuda", index)
        elif requested.type == "cpu":
            requested = torch.device("cpu")
        else:
            raise NotImplementedError("only CPU and CUDA devices are supported")
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = requested
        self.task = task
        self.scheduler = scheduler
        self.amp = amp
        self.accum_steps = accum_steps
        self.grad_clip = grad_clip
        if model_config is not None:
            if type(model_config) is not dict:
                raise TypeError("model_config must be a plain dictionary or None")
            from .resume import _plain
            model_config = _plain(model_config)
        self.model_config = model_config
        if ema_decay is not None:
            if isinstance(ema_decay, bool) or not isinstance(ema_decay, Real):
                raise TypeError("ema_decay must be a finite number in [0,1] or None")
            if not math.isfinite(ema_decay) or not 0 <= ema_decay <= 1:
                raise ValueError("ema_decay must be finite and in [0,1]")
        if early_stopping is not None:
            if type(early_stopping) is not EarlyStopping:
                raise TypeError("early_stopping must be a standard EarlyStopping instance")
            if type(early_stopping.patience) is not int or early_stopping.patience <= 0:
                raise ValueError("EarlyStopping patience must be a positive Python int")
            if isinstance(early_stopping.delta, bool) or not isinstance(early_stopping.delta, Real) or not math.isfinite(early_stopping.delta) or early_stopping.delta < 0:
                raise ValueError("EarlyStopping delta must be finite and nonnegative")
            if type(early_stopping.restore_best_weights) is not bool or type(early_stopping.verbose) is not bool:
                raise TypeError("EarlyStopping restore_best_weights and verbose must be bool")
        if not isinstance(callbacks, (tuple, list)):
            raise TypeError("callbacks must be a tuple/list of observer callables")
        for callback in callbacks:
            if isinstance(callback, (EarlyStopping, ModelCheckpoint, LRSchedulerCallback)) or not callable(callback):
                raise TypeError("callbacks must use callback(event, logs), not legacy training callbacks")
            if inspect.iscoroutinefunction(callback) or inspect.iscoroutinefunction(getattr(callback, "__call__", None)):
                raise TypeError("callbacks must be synchronous observers")
            try:
                inspect.signature(callback).bind("epoch_end", {})
            except (ValueError, TypeError) as exc:
                raise TypeError("callbacks must accept (event, logs)") from exc
        self.callbacks = tuple(callbacks)
        self.early_stopping = early_stopping
        self.ema_decay = ema_decay
        self.history = {key: [] for key in ("train_loss", "train_acc", "val_loss", "val_acc")}
        self._epochs = None
        self.microbatch_count = 0
        self.attempt_count = 0
        self.update_count = 0
        self.ema_update_count = 0
        self._finalized = False
        self._finished = False
        self._validate_setup()
        # This API exists in the minimum supported Torch 2.2.2 runtime.
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)
        self.ema = None if ema_decay is None else EMA(model, decay=ema_decay, device=self.device)
        if self.ema is not None:
            self.ema.ema_model.eval()
            self._validate_ema()

    @property
    def evaluation_model(self):
        """Inference/best-weight target; EMA weights are never loaded into live model."""
        return self.model if self.ema is None else self.ema.ema_model

    def _validate_ema(self):
        if self.ema is None:
            return
        for live, shadow in ((list(self.model.named_parameters()), list(self.ema.ema_model.named_parameters())),
                             (list(self.model.named_buffers()), list(self.ema.ema_model.named_buffers()))):
            if [name for name, _ in live] != [name for name, _ in shadow]:
                raise ValueError("EMA parameter/buffer names and order must match live model")
            for (name, source), (_, target) in zip(live, shadow):
                if source.shape != target.shape or source.dtype != target.dtype or source.device != target.device:
                    raise ValueError(f"EMA tensor schema/device mismatch: {name}")
                if source is target or (source.numel() and source.data_ptr() == target.data_ptr()):
                    raise ValueError(f"EMA tensors must be independent: {name}")
        if any(p.requires_grad for p in self.ema.ema_model.parameters()):
            raise ValueError("EMA shadow parameters must have gradients disabled")

    def _emit(self, event, epoch):
        logs = {key: (values[-1] if values else None) for key, values in self.history.items()}
        logs.update(epoch=epoch, applied_updates=self.update_count, optimizer_attempts=self.attempt_count,
                    ema_updates=self.ema_update_count,
                    learning_rate=[float(group["lr"]) for group in self.optimizer.param_groups],
                    early_stop=self.early_stopping.early_stop if self.early_stopping is not None else False)
        for callback in self.callbacks:
            callback(event, copy.deepcopy(logs))

    @staticmethod
    def _loader_length(loader, name):
        try:
            count = len(loader)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite, sized iterable") from exc
        if count <= 0 or not callable(getattr(loader, "__iter__", None)):
            raise ValueError(f"{name} must be nonempty and iterable")
        return count

    def _validate(self, loader):
        """Pure metric phase with exact per-submodule mode restoration, even on error."""
        count = self._loader_length(loader, "val_loader")
        self._validate_ema()
        target = self.evaluation_model
        modes = [(module, module.training) for module in target.modules()]
        try:
            target.eval()
            summed, correct, samples, seen = 0., 0, 0, 0
            with torch.no_grad():
                for batch in loader:
                    seen += 1
                    if seen > count:
                        raise ValueError("val_loader yielded more batches than len()")
                    inputs, labels, size = self._batch(batch)
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                        loss, batch_correct = self._loss_and_correct(target(inputs), labels, size)
                    summed += loss.item() * size
                    correct += batch_correct
                    samples += size
            if seen != count or not samples:
                raise ValueError("val_loader yielded fewer batches than len() or no samples")
            loss = summed / samples
            if not math.isfinite(loss):
                raise FloatingPointError("nonfinite validation loss encountered")
            return loss, 100. * correct / samples
        finally:
            for module, training in modes:
                module.training = training

    def _validate_setup(self):
        if type(self.amp) is not bool:
            raise TypeError("amp must be a bool")
        if type(self.accum_steps) is not int:
            raise TypeError("accum_steps must be a positive Python int, not bool")
        if self.accum_steps <= 0:
            raise ValueError("accum_steps must be positive")
        if self.grad_clip is not None:
            if isinstance(self.grad_clip, bool) or not isinstance(self.grad_clip, Real):
                raise TypeError("grad_clip must be a positive finite number")
            if not math.isfinite(self.grad_clip) or self.grad_clip <= 0:
                raise ValueError("grad_clip must be positive and finite")
        if self.amp and self.device.type != "cuda":
            raise TypeError("amp=True requires CUDA FP16; CPU amp is unsupported")
        expected = nn.CrossEntropyLoss if self.task == "multiclass" else nn.BCEWithLogitsLoss
        # Custom forwards can change the denominator or target semantics.
        if type(self.loss_fn) is not expected:
            raise TypeError(f"{self.task} requires standard {expected.__name__}")
        if self.loss_fn.reduction != "mean" or self.loss_fn.weight is not None:
            raise ValueError("only unweighted reduction='mean' loss is supported")
        if self.task == "multiclass" and self.loss_fn.label_smoothing != 0:
            raise ValueError("label smoothing is unsupported in M2-B1")
        if self.task == "binary" and self.loss_fn.pos_weight is not None:
            raise ValueError("pos_weight is unsupported in M2-B1")
        parameters = list(self.model.parameters())
        for tensor in [*parameters, *self.model.buffers()]:
            if tensor.device != self.device:
                raise ValueError(f"model parameters and buffers must already be on {self.device}")
            if (tensor.is_floating_point() and tensor.dtype not in (torch.float32, torch.float64)) or tensor.is_complex():
                raise ValueError("model supports only float32/float64 floating tensors")
            if self.amp and tensor.is_floating_point() and tensor.dtype != torch.float32:
                raise ValueError("amp requires FP32 model parameters and floating buffers")
        owned = {id(p) for p in parameters}
        trainable = {id(p) for p in parameters if p.requires_grad}
        supplied = [id(p) for group in self.optimizer.param_groups for p in group["params"]]
        if len(supplied) != len(set(supplied)):
            raise ValueError("optimizer contains duplicate parameters")
        if not set(supplied) <= owned:
            raise ValueError("optimizer contains foreign model parameters")
        if not trainable or not trainable <= set(supplied):
            raise ValueError("optimizer must cover every trainable model parameter")
        if isinstance(self.optimizer, torch.optim.LBFGS):
            raise NotImplementedError("closure-based optimizers are unsupported in M2-B1")
        if type(self.optimizer) not in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
            raise NotImplementedError("only standard non-fused SGD/Adam/AdamW optimizers are supported")
        if any(group.get("fused", False) or group.get("differentiable", False)
               for group in self.optimizer.param_groups):
            raise NotImplementedError("fused/differentiable optimizers are unsupported")
        self._validate_scheduler()
        if hasattr(self, "ema"):
            self._validate_ema()

    def _validate_scheduler(self, *, fresh=False, total_steps=None):
        scheduler = self.scheduler
        if scheduler is None:
            return
        schedulers = torch.optim.lr_scheduler
        plateau = type(scheduler) is schedulers.ReduceLROnPlateau
        if type(scheduler) not in (schedulers.OneCycleLR, schedulers.StepLR, schedulers.CosineAnnealingLR, schedulers.ReduceLROnPlateau):
            if not isinstance(scheduler, schedulers.LRScheduler):
                raise TypeError("scheduler must be a supported bare PyTorch scheduler")
            raise NotImplementedError("custom scheduler is unsupported")
        if scheduler.optimizer is not self.optimizer:
            raise ValueError("scheduler must be bound to the Trainer optimizer")
        if fresh and (scheduler.last_epoch != 0 or (not plateau and scheduler._step_count != 1)):
            raise ValueError("scheduler must have fresh state (last_epoch=0)")
        # Torch 2.2.2 Plateau has neither get_last_lr nor _last_lr before
        # its first step. Its initial LR state is the optimizer group state.
        lrs = getattr(scheduler, "_last_lr", [g["lr"] for g in self.optimizer.param_groups]) if plateau else scheduler.get_last_lr()
        if len(lrs) != len(self.optimizer.param_groups):
            raise ValueError("scheduler LR state does not match optimizer groups")
        for lr, group in zip(lrs, self.optimizer.param_groups):
            if not isinstance(lr, Real) or not math.isfinite(lr) or lr < 0 or group["lr"] != lr:
                raise ValueError("scheduler LR state must be finite and match optimizer")
        if plateau:
            if scheduler.mode != "min":
                raise ValueError("Plateau must minimize validation loss (mode='min')")
            if not isinstance(scheduler.factor, Real) or not math.isfinite(scheduler.factor) or not 0 <= scheduler.factor < 1:
                raise ValueError("Plateau factor must be finite and in [0,1)")
            for key in ("patience", "cooldown"):
                if type(getattr(scheduler, key)) is not int or getattr(scheduler, key) < 0:
                    raise ValueError(f"Plateau {key} must be a nonnegative int")
            for key in ("threshold", "eps"):
                value = getattr(scheduler, key)
                if not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Plateau {key} must be finite and nonnegative")
            if fresh and (scheduler.best != math.inf or scheduler.num_bad_epochs != 0 or scheduler.cooldown_counter != 0):
                raise ValueError("Plateau must have fresh state")
        elif type(scheduler) is schedulers.OneCycleLR:
            if type(scheduler.total_steps) is not int or scheduler.total_steps <= 0:
                raise ValueError("OneCycle total_steps must be a positive int")
            if total_steps is not None and scheduler.total_steps != total_steps:
                raise ValueError("OneCycle total_steps does not match epochs * ceil(len(loader)/accum_steps)")
            phases = scheduler._schedule_phases
            if not isinstance(phases, list) or len(phases) not in (2, 3):
                raise ValueError("invalid OneCycle schedule phases")
            end = -math.inf
            for phase in phases:
                if not isinstance(phase, dict) or not isinstance(phase.get("end_step"), Real):
                    raise ValueError("invalid OneCycle schedule phase")
                if not math.isfinite(phase["end_step"]) or phase["end_step"] <= end:
                    raise ValueError("invalid OneCycle schedule phase boundaries")
                end = phase["end_step"]
                for key in ("start_lr", "end_lr"):
                    if phase.get(key) not in ("initial_lr", "max_lr", "min_lr"):
                        raise ValueError("invalid OneCycle LR plan")
            if end != scheduler.total_steps - 1:
                raise ValueError("invalid OneCycle final schedule boundary")
            for group in self.optimizer.param_groups:
                for key in ("initial_lr", "max_lr", "min_lr"):
                    value = group.get(key)
                    if not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                        raise ValueError("invalid OneCycle optimizer LR plan")
        elif type(scheduler) is schedulers.StepLR:
            if type(scheduler.step_size) is not int or scheduler.step_size <= 0:
                raise ValueError("StepLR step_size must be a positive int")
            if not isinstance(scheduler.gamma, Real) or not math.isfinite(scheduler.gamma) or scheduler.gamma < 0:
                raise ValueError("StepLR gamma must be finite and nonnegative")
        else:
            if type(scheduler.T_max) is not int or scheduler.T_max <= 0:
                raise ValueError("CosineAnnealingLR T_max must be a positive int")
            if not isinstance(scheduler.eta_min, Real) or not math.isfinite(scheduler.eta_min) or scheduler.eta_min < 0:
                raise ValueError("CosineAnnealingLR eta_min must be finite and nonnegative")

    def _update(self, samples):
        """One window: unscale, sample normalization, clip, native step/update."""
        stage = getattr(self, "_stage_recorder", None)
        parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
        try:
            if stage is not None:
                stage.stage_begin("unscale")
            self.scaler.unscale_(self.optimizer)
            if stage is not None:
                stage.stage_end("unscale")
                stage.stage_begin("gradient_check_normalization")
            finite = True
            for parameter in parameters:
                if parameter.grad is not None:
                    values = parameter.grad.coalesce().values() if parameter.grad.is_sparse else parameter.grad
                    finite = bool(torch.isfinite(values).all()) and finite
                    parameter.grad.div_(samples)
            if not finite and not self.amp:
                raise FloatingPointError("nonfinite gradient encountered before optimizer.step")
            if stage is not None:
                stage.stage_end("gradient_check_normalization")
            # On AMP overflow unscale_ has recorded found_inf. Let the native
            # scaler skip; clipping an inf gradient would only produce NaNs.
            if finite and self.grad_clip is not None:
                if stage is not None:
                    stage.stage_begin("gradient_clipping")
                torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip,
                                              error_if_nonfinite=True)
                if stage is not None:
                    stage.stage_end("gradient_clipping")
            elif stage is not None:
                stage.not_executed("gradient_clipping", "disabled_or_nonfinite_gradient")
            applied = []
            before = self.scaler.get_scale()
            handle = self.optimizer.register_step_post_hook(lambda *args: applied.append(True))
            try:
                self.attempt_count += 1
                if stage is not None:
                    stage.stage_begin("scaler_step_attempt")
                self.scaler.step(self.optimizer)
                if stage is not None:
                    stage.stage_end("scaler_step_attempt")
            finally:
                handle.remove()
                if stage is not None:
                    stage.remove_optimizer_hooks()
            if stage is not None:
                stage.stage_begin("scaler_update")
            self.scaler.update()
            if stage is not None:
                stage.stage_end("scaler_update")
            if applied and self.scaler.get_scale() >= before:
                self.update_count += 1
                if self.ema is not None:
                    if stage is not None:
                        stage.stage_begin("ema_update")
                    self._validate_ema()
                    self.ema.update(self.model)
                    self.ema_update_count += 1
                    if stage is not None:
                        stage.stage_end("ema_update")
                elif stage is not None:
                    stage.not_executed("ema_update", "disabled")
                if type(self.scheduler) is torch.optim.lr_scheduler.OneCycleLR:
                    if stage is not None:
                        stage.stage_begin("onecycle_scheduler")
                    self.scheduler.step()
                    if stage is not None:
                        stage.stage_end("onecycle_scheduler")
                elif stage is not None:
                    stage.not_executed("onecycle_scheduler", "not_onecycle")
            elif stage is not None:
                stage.not_executed("ema_update", "optimizer_update_skipped")
                stage.not_executed("onecycle_scheduler", "optimizer_update_skipped")
        finally:
            if stage is None:
                self.optimizer.zero_grad()
            else:
                stage.zero_grad(self.optimizer)

    def _batch(self, batch):
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ValueError("each batch must contain (inputs, targets)")
        inputs, targets = batch
        if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise TypeError("inputs and targets must be tensors")
        if inputs.ndim == 0 or inputs.shape[0] == 0:
            raise ValueError("inputs must have a nonempty batch dimension")
        if inputs.dtype not in (torch.float32, torch.float64):
            raise TypeError("inputs must have float32 or float64 dtype")
        if self.amp and inputs.dtype != torch.float32:
            raise TypeError("amp requires float32 inputs")
        size = inputs.shape[0]
        if self.task == "multiclass":
            if targets.shape != (size,):
                raise ValueError("multiclass targets must have shape (B,)")
            if targets.dtype != torch.long:
                raise TypeError("multiclass targets must have torch.long dtype")
            if (targets == self.loss_fn.ignore_index).any():
                raise ValueError("ignored targets are unsupported in M2-B1")
        else:
            if targets.shape not in ((size,), (size, 1)):
                raise ValueError("binary targets must have shape (B,) or (B,1)")
            if not targets.is_floating_point():
                raise TypeError("binary targets must have floating dtype")
            if not torch.isfinite(targets).all() or not ((targets == 0) | (targets == 1)).all():
                raise ValueError("binary targets must be finite hard labels 0 or 1")
        return inputs.to(self.device), targets.to(self.device), size

    def _loss_and_correct(self, logits, targets, size):
        if not isinstance(logits, torch.Tensor):
            raise TypeError("model forward must return tensor logits")
        allowed = (torch.float16, torch.float32) if self.amp else (torch.float32, torch.float64)
        if logits.device != self.device or logits.dtype not in allowed:
            raise ValueError("logits device/dtype must match the requested training precision")
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite logits/loss encountered")
        if self.task == "multiclass":
            if logits.ndim != 2 or logits.shape[0] != size or logits.shape[1] < 2:
                raise ValueError("multiclass logits must have shape (B,C), C >= 2")
            if ((targets < 0) | (targets >= logits.shape[1])).any():
                raise ValueError("class indices are outside the logits class range")
            correct = (logits.detach().argmax(dim=1) == targets).sum().item()
        else:
            if logits.shape not in ((size,), (size, 1)):
                raise ValueError("binary logits must have shape (B,) or (B,1)")
            logits = logits.reshape(size)
            targets = targets.reshape(size).to(dtype=logits.dtype)
            correct = ((logits.detach() >= 0) == (targets == 1)).sum().item()
        loss = self.loss_fn(logits, targets)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError("nonfinite scalar loss encountered")
        return loss, correct

    def fit(self, train_loader, val_loader=None, *, epochs=5, save_last_to=None, resume_from=None):
        """Run train→validation→scheduler→ES→observer→optional last save→finalization.

        Save requires callbacks=(), standard num_workers=0 TensorDataset loaders
        and all seven ModularCNN constructor fields in model_config. It saves
        continuing state before best restoration. Resume requires the same plan
        and total epochs; completed/terminal checkpoints do not train again.
        """
        if type(epochs) is not int or epochs <= 0:
            raise ValueError("epochs must be a positive Python int")
        self._validate_setup()
        count = self._loader_length(train_loader, "train_loader")
        if val_loader is not None:
            self._loader_length(val_loader, "val_loader")
        elif self.early_stopping is not None or type(self.scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau:
            raise ValueError("EarlyStopping and ReduceLROnPlateau require val_loader")
        if self._finalized:
            raise RuntimeError("Trainer is terminal after best-weight finalization/early stop; create a fresh Trainer")
        if self._epochs is not None:
            if self._finished and self._epochs == epochs and len(self.history["train_loss"]) == epochs:
                if save_last_to is not None:
                    raise RuntimeError("cannot write save_last_to after a completed fit; enable saving on the initial fit")
                return {key: list(values) for key, values in self.history.items()}
            raise RuntimeError("fit lifecycle already started; create a fresh Trainer")
        self._validate_scheduler(fresh=True,
                                 total_steps=epochs * ((count + self.accum_steps - 1) // self.accum_steps))
        if self.early_stopping is not None:
            es = self.early_stopping
            if es.best_state is not None or es.best_loss != float("inf") or es.counter != 0 or es.early_stop:
                raise ValueError("EarlyStopping must have fresh state before fit or checkpoint restore")
        save_plan = None
        start_epoch = 0
        if resume_from is not None:
            from .resume import prepare_save, save_last, restore_checkpoint
            try:
                destination, plan = prepare_save(self, train_loader, val_loader, epochs,
                                                 save_last_to if save_last_to is not None else resume_from)
            except TypeError as exc:
                raise TypeError("resume_from: " + str(exc)) from exc
            start_epoch = restore_checkpoint(self, resume_from, plan)
            if self._finished:
                return {key: list(values) for key, values in self.history.items()}
            if save_last_to is not None:
                save_plan = plan
        elif save_last_to is not None:
            from .resume import prepare_save, save_last
            destination, save_plan = prepare_save(self, train_loader, val_loader, epochs, save_last_to)
        self._epochs = epochs
        recorder = getattr(self, "_timing_recorder", None)
        stage = getattr(self, "_stage_recorder", None)
        self._emit("train_begin", 0)
        # Enable autograd explicitly even if the caller has entered no_grad.
        with torch.enable_grad():
            for _ in range(start_epoch, epochs):
                if recorder is not None:
                    recorder.epoch_begin(_ + 1)
                # Keep intentionally eval/frozen children of a training model.
                modes = [(m, m.training) for m in self.model.modules()]
                mixed = modes if self.model.training or any(flag for _, flag in modes[1:]) else []
                self.model.train()
                for module, training in mixed:
                    if module is not self.model:
                        module.training = training
                total_loss, total_correct, total_samples = 0.0, 0, 0
                window_samples = 0
                if stage is not None:
                    stage.stage_begin("zero_grad")
                self.optimizer.zero_grad()
                if stage is not None:
                    stage.stage_end("zero_grad")
                seen = 0
                for batch in train_loader:
                    seen += 1
                    if seen > count:
                        raise ValueError("train_loader yielded more batches than len()")
                    if recorder is not None:
                        recorder.microbatch_begin(self.microbatch_count + 1, window_samples == 0)
                    if stage is not None:
                        stage.stage_begin("input_preparation")
                    inputs, targets, size = self._batch(batch)
                    if stage is not None:
                        stage.stage_end("input_preparation")
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                        if stage is None:
                            loss, correct = self._loss_and_correct(self.model(inputs), targets, size)
                        else:
                            stage.stage_begin("forward")
                            logits = self.model(inputs)
                            stage.stage_end("forward")
                            stage.stage_begin("loss_metrics")
                            loss, correct = self._loss_and_correct(logits, targets, size)
                            del logits
                            stage.stage_end("loss_metrics")
                    if stage is not None:
                        stage.stage_begin("backward")
                    self.scaler.scale(loss * size).backward()
                    if stage is not None:
                        stage.stage_end("backward")
                    self.microbatch_count += 1
                    window_samples += size
                    if recorder is not None:
                        recorder.microbatch_end(size, inputs, targets)
                    if seen % self.accum_steps == 0 or seen == count:
                        if recorder is not None:
                            recorder.attempt_begin(self.attempt_count + 1, self.update_count)
                        self._update(window_samples)
                        if recorder is not None:
                            recorder.attempt_end(self.update_count, window_samples)
                        window_samples = 0
                    total_loss += loss.detach().item() * size
                    total_correct += correct
                    total_samples += size
                if seen != count or total_samples == 0:
                    raise ValueError("train_loader yielded fewer batches than len() or no samples")
                epoch_loss = total_loss / total_samples
                if not math.isfinite(epoch_loss):
                    raise FloatingPointError("nonfinite epoch loss encountered")
                if recorder is not None:
                    recorder.training_end(val_loader is not None)
                val_loss, val_acc = (None, None) if val_loader is None else self._validate(val_loader)
                if recorder is not None:
                    recorder.validation_end(val_loader is not None)
                epoch_scheduler = type(self.scheduler) in (torch.optim.lr_scheduler.StepLR,
                    torch.optim.lr_scheduler.CosineAnnealingLR, torch.optim.lr_scheduler.ReduceLROnPlateau)
                if stage is not None:
                    if epoch_scheduler:
                        stage.stage_begin("epoch_scheduler")
                    else:
                        stage.not_executed("epoch_scheduler", "no_epoch_scheduler")
                if type(self.scheduler) in (torch.optim.lr_scheduler.StepLR,
                                           torch.optim.lr_scheduler.CosineAnnealingLR):
                    self.scheduler.step()
                elif type(self.scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau:
                    self.scheduler.step(val_loss)
                if stage is not None and epoch_scheduler:
                    stage.stage_end("epoch_scheduler")
                if self.early_stopping is not None:
                    self.early_stopping(val_loss, self.evaluation_model)
                self.history["train_loss"].append(epoch_loss)
                self.history["train_acc"].append(100.0 * total_correct / total_samples)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                self._emit("epoch_end", len(self.history["train_loss"]))
                if recorder is not None:
                    recorder.epoch_end()
                # Coherent last-state boundary, before the stop decision and
                # any best restore. The v1 serializer owns atomic replacement.
                if save_plan is not None:
                    save_last(self, destination, save_plan, train_loader, val_loader)
                if self.early_stopping is not None and self.early_stopping.early_stop:
                    break
        if self.early_stopping is not None:
            es = self.early_stopping
            if es.restore_best_weights:
                self._finalized = True
                es.restore_weights(self.evaluation_model)
            self._finalized = self._finalized or es.early_stop
        self._emit("train_end", len(self.history["train_loss"]))
        self._finished = True
        return {key: list(values) for key, values in self.history.items()}
