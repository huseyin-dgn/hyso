"""Real CPU training probes shared by the P0.1-C tests."""

import copy
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hyso.models import CEMO, EBO


def tiny_ebo(num_classes=2):
    return EBO(
        input_channels=3, num_classes=num_classes, conv_channels=[4, 8],
        residual_per_block=1, p_drop=0.0, att_reduction=2,
        norm_type="auto", device="cpu", use_depthwise=False,
    )


def tiny_cemo(use_ema=False):
    return CEMO(
        input_channels=3, num_classes=2, conv_channels=[4, 8],
        residuals_per_stage=1, se_reduction=2, p_drop=0.0,
        dropout_fc=0.0, use_ema=use_ema, ema_decay=0.5, device="cpu",
    )


def synthetic_loader(batch_sizes=(2, 2, 2), *, dtype=torch.float32):
    """Explicit microbatch sizes; iteration never consumes the model RNG."""
    count = sum(batch_sizes)
    generator = torch.Generator(device="cpu").manual_seed(23)
    x = torch.randn(count, 3, 16, 16, generator=generator).to(dtype=dtype)
    y = torch.arange(count, dtype=torch.long) % 2
    batches, start = [], 0
    for size in batch_sizes:
        batches.append(list(range(start, start + size)))
        start += size
    return DataLoader(
        TensorDataset(x, y), batch_sampler=batches,
        generator=torch.Generator(device="cpu").manual_seed(29),
    )


def observe_optimizer(optimizer):
    trace = SimpleNamespace(before=[], after=[])

    def before_step(opt, args, kwargs):
        trace.before.append({
            "lr": opt.param_groups[0]["lr"],
            "gradients": [
                None if p.grad is None else p.grad.detach().clone()
                for group in opt.param_groups for p in group["params"]
            ],
        })

    def after_step(opt, args, kwargs):
        trace.after.append([p.detach().clone() for g in opt.param_groups for p in g["params"]])

    optimizer.register_step_pre_hook(before_step)
    optimizer.register_step_post_hook(after_step)
    return trace


def attach_sgd(model, lr=0.05):
    parameters = model.model.parameters() if isinstance(model, EBO) else model.parameters()
    model.optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.0, weight_decay=0.0)
    return observe_optimizer(model.optimizer)


def run_ebo(model, loader, *, validation=False, **kwargs):
    options = dict(epochs=1, use_amp=False, accum_steps=1)
    options.update(kwargs)
    return model.fit(loader, val_loader=loader if validation else None, **options)


def run_with_sample_weighted_reference(batch_sizes, accum_steps, *, epochs=1,
                                       grad_clip=None, num_classes=2, dtype=torch.float32):
    """Oracle uses CE mean * n_microbatch / n_window, including the tail."""
    model = tiny_ebo(num_classes=num_classes).to(dtype=dtype)
    reference = copy.deepcopy(model.model)
    trace = attach_sgd(model)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    reference_trace = observe_optimizer(reference_optimizer)
    loader = synthetic_loader(batch_sizes, dtype=dtype)
    torch.random.default_generator.manual_seed(101)
    history = run_ebo(model, loader, accum_steps=accum_steps, epochs=epochs, grad_clip=grad_clip)
    torch.random.default_generator.manual_seed(101)
    reference.train()
    batches = list(synthetic_loader(batch_sizes, dtype=dtype))
    reference_trace.epoch_losses = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for offset in range(0, len(batches), accum_steps):
            window = batches[offset:offset + accum_steps]
            samples = sum(y.numel() for _, y in window)
            reference_optimizer.zero_grad()
            for x, y in window:
                output = reference(x)
                if num_classes == 1:
                    loss = nn.functional.binary_cross_entropy_with_logits(output.view(-1), y.float())
                else:
                    loss = nn.functional.cross_entropy(output, y, reduction="mean")
                epoch_loss += loss.item() * y.numel()
                (loss * y.numel() / samples).backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(reference.parameters(), grad_clip)
            reference_optimizer.step()
        reference_trace.epoch_losses.append(epoch_loss / sum(batch_sizes))
    return model, reference, trace, reference_trace, history


class SchedulerProbe:
    """Wrap real post-constructor step calls; preserve class/isinstance behavior."""

    def __init__(self, monkeypatch, scheduler_class):
        self.records = []
        original_init = scheduler_class.__init__

        def observed_init(scheduler, *args, **kwargs):
            original_init(scheduler, *args, **kwargs)
            record = SimpleNamespace(
                scheduler=scheduler, initial_lr=scheduler.optimizer.param_groups[0]["lr"],
                calls=[],
            )
            self.records.append(record)
            real_step = scheduler.step

            def observed_step(*step_args, **step_kwargs):
                event = {"before": scheduler.optimizer.param_groups[0]["lr"],
                         "args": step_args, "error": None, "error_message": None,
                         "last_epoch_before": scheduler.last_epoch}
                try:
                    return real_step(*step_args, **step_kwargs)
                except Exception as error:
                    event["error"] = type(error).__name__
                    event["error_message"] = str(error)
                    raise
                finally:
                    event["after"] = scheduler.optimizer.param_groups[0]["lr"]
                    event["last_epoch_after"] = scheduler.last_epoch
                    record.calls.append(event)

            scheduler.step = observed_step

        monkeypatch.setattr(scheduler_class, "__init__", observed_init)
