"""Explicit GPU validation, outside the CPU-only pytest conftest.

Run directly with a CUDA-visible interpreter; missing CUDA is an error, never a skip.
No production math, scaler outcomes or optimizer updates are mocked.
"""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import warnings

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torchvision
from unittest.mock import patch

from hyso.models import bcam
from hyso.models.WithSE import EMA
from hyso.checkpoint import load_training_checkpoint
from test_training_checkpoint import assert_state
from test_ebo_training_resume import loader, options, seed as cpu_seed
from training_helpers import tiny_ebo


def seed(value):
    cpu_seed(value)
    torch.cuda.manual_seed_all(value)


def model():
    return bcam.EBO(input_channels=3, num_classes=2, conv_channels=[4, 8],
                    residual_per_block=1, p_drop=0.0, att_reduction=2,
                    norm_type="auto", device="cuda", use_depthwise=False)


def no_plot(*args, **kwargs):
    raise RuntimeError("presentation disabled during numerical verification")


def resume_case(directory, optimizer, scheduler):
    opts = options(optimizer=optimizer, scheduler=scheduler, ema=True)
    full_path, cut_path, last_path = [directory / name for name in ("full.pth", "cut.pth", "last.pth")]
    seed(19)
    full = model()
    before = [p.detach().clone() for p in full.parameters()]
    history = full.fit(loader(), val_loader=loader(False), save_last_to=full_path, **opts)
    expected = torch.load(full_path, weights_only=True, map_location="cpu")
    assert any(not torch.equal(p, previous) for p, previous in zip(full.parameters(), before))
    assert expected["scaler"]["state_dict"]["scale"] > 0
    assert expected["rng_state"]["cuda"] is not None
    seed(19)
    first = model()
    real_save = bcam.save_training_checkpoint

    class Stop(Exception):
        pass

    def save(*args, **kwargs):
        real_save(*args, **kwargs)
        if kwargs["completed_epochs"] == 3:
            raise Stop

    with patch.object(bcam, "save_training_checkpoint", save):
        try:
            first.fit(loader(), val_loader=loader(False), save_last_to=cut_path, **opts)
        except Stop:
            pass
        else:
            raise AssertionError("controlled interruption did not occur")
    cut = torch.load(cut_path, weights_only=True, map_location="cpu")
    assert cut["completed_epochs"] == 3
    seed(999)
    resumed = model()
    actual_history = resumed.fit(loader(), val_loader=loader(False), resume_from=cut_path,
                                 save_last_to=last_path, **opts)
    actual = torch.load(last_path, weights_only=True, map_location="cpu")
    assert_state(actual, expected)  # Exact, including CPU/CUDA/Python/NumPy RNG.
    assert_state(resumed.state_dict(), full.state_dict())
    assert actual_history == history
    assert all(np.isfinite(values).all() for values in history.values())
    updates = {int(entry["step"].item()) for entry in resumed.optimizer.state.values()}
    assert len(updates) == 1 and 0 < next(iter(updates)) <= 10
    updates = next(iter(updates))
    counter = resumed.lr_scheduler.scheduler.last_epoch
    assert counter == (updates if scheduler == "onecycle" else 5)
    if scheduler == "onecycle":
        assert actual["scheduler"]["state_dict"]["total_steps"] == 10
    for name, buffer in resumed.model.named_buffers():
        assert_state(dict(resumed.ema.ema_model.named_buffers())[name], buffer)

    # Exercise the actual public loader on CPU, not only raw torch.load.
    cpu = tiny_ebo()
    opt = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[optimizer](
        cpu.model.parameters(), lr=0.005, weight_decay=0.001)
    sched = (torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5) if scheduler == "steplr"
             else torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.03, total_steps=10, cycle_momentum=False))
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    ema = EMA(cpu.model)
    progress = load_training_checkpoint(full_path, cpu, optimizer=opt, scheduler=sched,
                                        scaler=scaler, ema=ema, map_location="cpu")
    assert progress["completed_epochs"] == 5
    assert all(p.device.type == "cpu" for p in cpu.parameters())
    assert_state(cpu.state_dict(), expected["model"]["state_dict"])
    assert_state(opt.state_dict(), expected["optimizer"]["state_dict"])
    assert_state(ema.ema_model.state_dict(), expected["ema"]["state_dict"])
    return dict(optimizer=optimizer, scheduler=scheduler, exact_resume=True,
                updates=updates, scheduler_counter=counter,
                scale=actual["scaler"]["state_dict"]["scale"], cpu_load=True)


def overflow_case(directory, scheduler):
    seed(73)
    net = model()
    net.optimizer = torch.optim.AdamW(net.model.parameters(), lr=0.005, weight_decay=0.001)
    applied, clipped, scalers, seen_dtype, hook_calls = [], [], [], [], []
    net.optimizer.register_step_post_hook(lambda *args: applied.append(True))
    parameter = next(net.model.parameters())

    def overflow(gradient):
        hook_calls.append(True)
        return torch.full_like(gradient, float("inf")) if len(hook_calls) == 1 else gradient

    parameter.register_hook(overflow)
    net.model.register_forward_hook(lambda module, inputs, output: seen_dtype.append(str(output.dtype)))
    real_scaler = torch.cuda.amp.GradScaler
    real_clip = torch.nn.utils.clip_grad_norm_

    def scaler(**kwargs):
        value = real_scaler(init_scale=128.0, **kwargs)
        assert value.is_enabled()
        scalers.append(value)
        return value

    def clip(parameters, max_norm, **kwargs):
        parameters = list(parameters)
        result = real_clip(parameters, max_norm, **kwargs)
        if torch.isfinite(result):
            norm = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(p.grad.detach()) for p in parameters if p.grad is not None]))
            assert norm.item() <= max_norm + 1e-6
        clipped.append(float(result))
        return result

    opts = options(ema=True, scheduler=scheduler)
    opts["epochs"] = 1
    path = directory / "overflow.pth"
    with patch.object(torch.cuda.amp, "GradScaler", scaler), patch.object(torch.nn.utils, "clip_grad_norm_", clip):
        net.fit(loader(), save_last_to=path, **opts)
    state = torch.load(path, weights_only=True, map_location="cpu")
    assert len(hook_calls) == 3 and len(clipped) == 2  # 3 microbatches, 2 update attempts: tail flushed.
    assert np.isinf(clipped[0]) and np.isfinite(clipped[1])
    assert len(applied) == 1 and net.lr_scheduler.scheduler.last_epoch == 1
    assert scalers[0].get_scale() == 64.0
    assert all(int(entry["step"].item()) == 1 for entry in net.optimizer.state.values())
    assert seen_dtype == ["torch.float16"] * 3
    assert state["scaler"]["state_dict"]["scale"] == 64.0
    return dict(scheduler=scheduler, real_inf_injected=True, microbatches=3, update_attempts=2, applied=1,
                scheduler_counter=1, initial_scale=128, final_scale=64,
                clip_norms=["inf", clipped[1]], finite_post_clip_norm_at_most=0.5)


def device_ema_check():
    seed(7)
    net = tiny_ebo().to("cuda")
    assert all(p.device.type == "cuda" for p in net.parameters())
    x = torch.randn(3, 3, 16, 16, device="cuda")
    ema = EMA(net.model, decay=0.6)
    previous = {name: p.detach().clone() for name, p in ema.ema_model.named_parameters()}
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    with torch.amp.autocast("cuda"):
        output = net(x)
        loss = output.square().mean()
    scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=128)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    assert output.dtype == torch.float16
    ema.update(net.model)
    for name, current in net.model.named_parameters():
        expected = previous[name].mul(0.6).add(current.detach(), alpha=0.4)
        assert_state(dict(ema.ema_model.named_parameters())[name], expected)
    for name, buffer in net.model.named_buffers():
        assert_state(dict(ema.ema_model.named_buffers())[name], buffer)
    assert next(iter(opt.state.values()))["step"].item() == 1
    return dict(cpu_to_cuda=True, forward_backward=True, autocast="float16", ema_exact=True)


def main():
    assert torch.cuda.is_available(), "Real CUDA required; CPU results are not a GPU pass"
    torch.set_num_threads(1)
    # CBAM AdaptiveMaxPool2d backward has no deterministic CUDA implementation.
    # Keep exact comparisons; permit the real kernel and record this limitation.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = dict(python=sys.version.split()[0], torch=torch.__version__, torchvision=torchvision.__version__,
                   cuda_available=torch.cuda.is_available(), cuda_build=torch.version.cuda,
                   cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
                   total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    # Presentation only: no training/scaler/optimizer/scheduler logic is replaced.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
            patch.object(bcam, "summary", lambda *args: None), \
            patch.object(bcam.plt, "figure", no_plot), \
            patch.object(bcam, "classification_report", lambda *args, **kwargs: ""), \
            tempfile.TemporaryDirectory(prefix="hyso-cuda-") as temporary, \
            warnings.catch_warnings(record=True) as observed_warnings:
        directory = Path(temporary)
        results["device_ema"] = device_ema_check()
        results["overflow"] = [overflow_case(directory, scheduler) for scheduler in ("steplr", "onecycle")]
        results["resume"] = [resume_case(directory, optimizer, scheduler)
                             for optimizer in ("adam", "adamw") for scheduler in ("steplr", "onecycle")]
    results["warnings"] = sorted(set(str(item.message) for item in observed_warnings))
    torch.cuda.synchronize()
    results["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    print("CUDA_VALIDATION=" + json.dumps(results))


if __name__ == "__main__":
    main()
