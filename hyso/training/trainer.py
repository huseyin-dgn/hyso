"""Minimal CPU classification lifecycle, independent of model-specific fit APIs."""

import math

import torch
from torch import nn


class Trainer:
    """Train a caller-owned model, optimizer and mean classification loss.

    B1 supports CPU float32/float64, hard-label multiclass and binary tasks.
    Move the model before creating the optimizer; this class never moves it.
    Loaders must be finite, sized iterables of (input_tensor, target_tensor).
    Inputs retain their dtype; binary floating targets are cast to logits dtype.
    History describes predictions before each batch's optimizer update, with
    loss weighted by actual sample count and accuracy expressed as a percentage.

    A Trainer owns one fit lifecycle. Repeating its completed epoch budget is
    a no-op; restarting or retrying a failed run requires a fresh Trainer.
    Validation and other M2-B2/B3/B4 features are not implemented yet.
    """

    def __init__(self, model, *, optimizer, loss_fn, device="cpu", task="multiclass"):
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
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if requested.type != "cpu":
            raise NotImplementedError("M2-B1 supports CPU training only")
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device("cpu")
        self.task = task
        self.history = {key: [] for key in ("train_loss", "train_acc", "val_loss", "val_acc")}
        self._epochs = None
        self._validate_setup()

    def _validate_setup(self):
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
                raise ValueError("model parameters and buffers must already be on CPU")
            if (tensor.is_floating_point() and tensor.dtype not in (torch.float32, torch.float64)) or tensor.is_complex():
                raise ValueError("model supports only float32/float64 floating tensors")
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
        if logits.device != self.device or logits.dtype not in (torch.float32, torch.float64):
            raise ValueError("logits must be CPU float32/float64 tensors")
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
        """Run one training budget; val_loader is explicitly unsupported in B1."""
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
        self._epochs = epochs
        # Enable autograd explicitly even if the caller has entered no_grad.
        with torch.enable_grad():
            for _ in range(epochs):
                self.model.train()
                total_loss, total_correct, total_samples = 0.0, 0, 0
                seen = 0
                for batch in train_loader:
                    seen += 1
                    if seen > count:
                        raise ValueError("train_loader yielded more batches than len()")
                    inputs, targets, size = self._batch(batch)
                    self.optimizer.zero_grad()
                    loss, correct = self._loss_and_correct(self.model(inputs), targets, size)
                    loss.backward()
                    for parameter in self.model.parameters():
                        gradient = parameter.grad
                        if gradient is not None:
                            values = gradient.coalesce().values() if gradient.is_sparse else gradient
                            if not torch.isfinite(values).all():
                                raise FloatingPointError("nonfinite gradient encountered before optimizer.step")
                    self.optimizer.step()
                    total_loss += loss.detach().item() * size
                    total_correct += correct
                    total_samples += size
                if seen != count or total_samples == 0:
                    raise ValueError("train_loader yielded fewer batches than len() or no samples")
                epoch_loss = total_loss / total_samples
                if not math.isfinite(epoch_loss):
                    raise FloatingPointError("nonfinite epoch loss encountered")
                self.history["train_loss"].append(epoch_loss)
                self.history["train_acc"].append(100.0 * total_correct / total_samples)
                self.history["val_loss"].append(None)
                self.history["val_acc"].append(None)
        return {key: list(values) for key, values in self.history.items()}
