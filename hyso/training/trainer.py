"""Classification updates, independent of model-specific fit APIs."""

import math
from numbers import Real

import torch
from torch import nn


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
    Validation and M2-B3/B4 features are not implemented yet. Optimizers are
    standard non-fused SGD/Adam/AdamW; schedulers are bare OneCycleLR, StepLR
    or CosineAnnealingLR instances. The Trainer owns their stepping.
    """

    def __init__(self, model, *, optimizer, loss_fn, device="cpu", task="multiclass",
                 scheduler=None, amp=False, accum_steps=1, grad_clip=None):
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
        self.history = {key: [] for key in ("train_loss", "train_acc", "val_loss", "val_acc")}
        self._epochs = None
        self.microbatch_count = 0
        self.attempt_count = 0
        self.update_count = 0
        self._validate_setup()
        # This API exists in the minimum supported Torch 2.2.2 runtime.
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)

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

    def _validate_scheduler(self, *, fresh=False, total_steps=None):
        scheduler = self.scheduler
        if scheduler is None:
            return
        schedulers = torch.optim.lr_scheduler
        if isinstance(scheduler, schedulers.ReduceLROnPlateau):
            raise NotImplementedError("ReduceLROnPlateau requires validation in M2-B3")
        if type(scheduler) not in (schedulers.OneCycleLR, schedulers.StepLR, schedulers.CosineAnnealingLR):
            if not isinstance(scheduler, schedulers.LRScheduler):
                raise TypeError("scheduler must be a supported bare PyTorch scheduler")
            raise NotImplementedError("custom scheduler is unsupported")
        if scheduler.optimizer is not self.optimizer:
            raise ValueError("scheduler must be bound to the Trainer optimizer")
        if fresh and (scheduler.last_epoch != 0 or scheduler._step_count != 1):
            raise ValueError("scheduler must have fresh state (last_epoch=0)")
        lrs = scheduler.get_last_lr()
        if len(lrs) != len(self.optimizer.param_groups):
            raise ValueError("scheduler LR state does not match optimizer groups")
        for lr, group in zip(lrs, self.optimizer.param_groups):
            if not isinstance(lr, Real) or not math.isfinite(lr) or lr < 0 or group["lr"] != lr:
                raise ValueError("scheduler LR state must be finite and match optimizer")
        if type(scheduler) is schedulers.OneCycleLR:
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
        parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
        try:
            self.scaler.unscale_(self.optimizer)
            finite = True
            for parameter in parameters:
                if parameter.grad is not None:
                    values = parameter.grad.coalesce().values() if parameter.grad.is_sparse else parameter.grad
                    finite = bool(torch.isfinite(values).all()) and finite
                    parameter.grad.div_(samples)
            if not finite and not self.amp:
                raise FloatingPointError("nonfinite gradient encountered before optimizer.step")
            # On AMP overflow unscale_ has recorded found_inf. Let the native
            # scaler skip; clipping an inf gradient would only produce NaNs.
            if finite and self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip,
                                              error_if_nonfinite=True)
            applied = []
            before = self.scaler.get_scale()
            handle = self.optimizer.register_step_post_hook(lambda *args: applied.append(True))
            try:
                self.attempt_count += 1
                self.scaler.step(self.optimizer)
            finally:
                handle.remove()
            self.scaler.update()
            if applied and self.scaler.get_scale() >= before:
                self.update_count += 1
                if type(self.scheduler) is torch.optim.lr_scheduler.OneCycleLR:
                    self.scheduler.step()
        finally:
            self.optimizer.zero_grad()

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

    def fit(self, train_loader, val_loader=None, *, epochs=5):
        """Run one training budget; validation is unsupported until M2-B3."""
        if val_loader is not None:
            raise NotImplementedError("validation is unsupported until M2-B3")
        if type(epochs) is not int or epochs <= 0:
            raise ValueError("epochs must be a positive Python int")
        self._validate_setup()
        try:
            count = len(train_loader)
        except (TypeError, ValueError) as exc:
            raise ValueError("train_loader must be a finite, sized iterable") from exc
        if count <= 0 or not callable(getattr(train_loader, "__iter__", None)):
            raise ValueError("train_loader must be nonempty and iterable")
        if self._epochs is not None:
            if self._epochs == epochs and len(self.history["train_loss"]) == epochs:
                return {key: list(values) for key, values in self.history.items()}
            raise RuntimeError("fit lifecycle already started; create a fresh Trainer")
        self._validate_scheduler(fresh=True,
                                 total_steps=epochs * ((count + self.accum_steps - 1) // self.accum_steps))
        self._epochs = epochs
        # Enable autograd explicitly even if the caller has entered no_grad.
        with torch.enable_grad():
            for _ in range(epochs):
                self.model.train()
                total_loss, total_correct, total_samples = 0.0, 0, 0
                window_samples = 0
                self.optimizer.zero_grad()
                seen = 0
                for batch in train_loader:
                    seen += 1
                    if seen > count:
                        raise ValueError("train_loader yielded more batches than len()")
                    inputs, targets, size = self._batch(batch)
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                        loss, correct = self._loss_and_correct(self.model(inputs), targets, size)
                    self.scaler.scale(loss * size).backward()
                    self.microbatch_count += 1
                    window_samples += size
                    if seen % self.accum_steps == 0 or seen == count:
                        self._update(window_samples)
                        window_samples = 0
                    total_loss += loss.detach().item() * size
                    total_correct += correct
                    total_samples += size
                if seen != count or total_samples == 0:
                    raise ValueError("train_loader yielded fewer batches than len() or no samples")
                epoch_loss = total_loss / total_samples
                if not math.isfinite(epoch_loss):
                    raise FloatingPointError("nonfinite epoch loss encountered")
                if type(self.scheduler) in (torch.optim.lr_scheduler.StepLR,
                                           torch.optim.lr_scheduler.CosineAnnealingLR):
                    self.scheduler.step()
                self.history["train_loss"].append(epoch_loss)
                self.history["train_acc"].append(100.0 * total_correct / total_samples)
                self.history["val_loss"].append(None)
                self.history["val_acc"].append(None)
        return {key: list(values) for key, values in self.history.items()}
