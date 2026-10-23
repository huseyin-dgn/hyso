"""Real M1 CUDA/FP16 and component checkpoint probe; run outside pytest fixtures.

No CUDA means failure, never skip. Generic checkpoint round-trips here do not
provide ModularCNN with EBO.fit or epoch-boundary resume support.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchvision
from torch import nn

from hyso.models import modular
from hyso.models.modular import ModularCNN
from hyso.models.blocks import ConvNormAct, ResidualBlock
from hyso.models.attention import ECAAttention, CoordinateAttention
from hyso.models.WithSE import EMA, SEBlock
from hyso.checkpoint import (save_training_checkpoint, load_training_checkpoint,
                             TrainingCheckpointError, _rng_state)


def exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual.detach().cpu(), expected.detach().cpu())
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            exact(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            exact(a, b)
    else:
        assert actual == expected


def train_step(model, optimizer, scaler, x, amp, classification=False):
    model.train()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    applied, conv_dtypes = [], []
    handles = [optimizer.register_step_post_hook(lambda *args: applied.append(True))]
    for layer in model.modules():
        if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
            handles.append(layer.register_forward_hook(
                lambda module, inputs, output: conv_dtypes.append(str(output.dtype))))
    try:
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp):
            out = model(x)
            loss = (nn.functional.cross_entropy(out, torch.tensor([0, 2], device=x.device))
                    if classification else out.float().square().mean())
        assert torch.isfinite(out).all() and torch.isfinite(loss)
        if classification:
            assert out.shape == (2, 3)
        if amp:
            assert scaler.is_enabled() and conv_dtypes and set(conv_dtypes) == {"torch.float16"}
            if classification:
                assert out.dtype == torch.float16
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert x.grad is not None and torch.isfinite(x.grad).all()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) > 0 for g in grads)
        scaler.step(optimizer)
        scaler.update()
        assert applied == [True]
        assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), before))
    finally:
        for handle in handles:
            handle.remove()
        x.grad = None
    return {"loss": loss.item(), "output_dtype": str(out.dtype),
            "conv_dtypes": sorted(set(conv_dtypes)), "optimizer_updates": 1}


def model_components(attention, device, amp, classes=3):
    model = ModularCNN(num_classes=classes, widths=(4, 6), repeats=(2, 1),
                       strides=(1, 2), attention=attention).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.7)
    scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=128.)
    ema = EMA(model, decay=.7, device=device)
    return model, optimizer, scheduler, scaler, ema


def model_case(directory, attention, device, amp):
    directory.mkdir(parents=True, exist_ok=True)
    model, opt, sched, scaler, ema = model_components(attention, device, amp)
    ids = [(name, id(p)) for name, p in model.named_parameters()]
    blocks = [block for stage in model.stages for block in stage]
    assert len({id(block.attention) for block in blocks}) == 3
    attention_parameters = [p for block in blocks for p in block.attention.parameters()]
    assert len({id(p) for p in attention_parameters}) == len(attention_parameters)
    assert all(any(p is q for q in model.parameters()) for p in attention_parameters)
    assert all(p.device.type == device.type for p in model.parameters())
    assert all(b.device.type == device.type for b in model.buffers())
    x = torch.randn(2, 3, 9, 13, generator=torch.Generator().manual_seed(17)).to(device).requires_grad_()
    history, steps = {"loss": []}, []
    for _ in range(2):
        row = train_step(model, opt, scaler, x, amp, classification=True)
        sched.step(); ema.update(model)
        history["loss"].append(row["loss"]); steps.append(row)
    assert ids == [(name, id(p)) for name, p in model.named_parameters()]
    assert all(b.item() == 2 for n, b in model.named_buffers() if n.endswith("num_batches_tracked"))
    path = directory/"full.pth"
    metrics = {"attention": attention, "probe": "M1-E component round-trip"}
    def save(destination, net, optimizer, scheduler, scale, shadow, losses):
        save_training_checkpoint(destination, net, optimizer=optimizer, scheduler=scheduler,
                                 scaler=scale, ema=shadow, history=losses, metrics=metrics, include_rng=True)
    save(path, model, opt, sched, scaler, ema, history)
    raw = torch.load(path, weights_only=True, map_location="cpu")
    assert raw["format_version"] == 1 and raw["format"] == "hyso.training"
    assert (raw["rng_state"]["cuda"] is not None) == (device.type == "cuda")
    fresh, fresh_opt, fresh_sched, fresh_scaler, fresh_ema = model_components(attention, device, amp)
    strict = fresh.load_state_dict(raw["model"]["state_dict"], strict=True)
    assert strict.missing_keys == strict.unexpected_keys == []
    progress = load_training_checkpoint(path, fresh, optimizer=fresh_opt, scheduler=fresh_sched,
                                       scaler=fresh_scaler, ema=fresh_ema, restore_rng=True, map_location="cpu")
    exact(progress["history"], history); exact(_rng_state(), raw["rng_state"])
    for actual, expected in [(fresh.state_dict(), model.state_dict()),
                             (fresh_opt.state_dict(), opt.state_dict()),
                             (fresh_sched.state_dict(), sched.state_dict()),
                             (fresh_scaler.state_dict(), scaler.state_dict()),
                             (fresh_ema.ema_model.state_dict(), ema.ema_model.state_dict())]:
        exact(actual, expected)
    assert fresh_ema.decay == ema.decay
    model.eval(); fresh.eval()
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        exact(fresh(x), model(x))
        assert model.forward_features(x).shape == (2, 6, 5, 7)
    exact(dict(model.named_buffers()), buffers)
    # Same saved components, then a real additional update on each independent model.
    continued = []
    for net, optimizer, scheduler, scale, shadow in (
            (model, opt, sched, scaler, ema), (fresh, fresh_opt, fresh_sched, fresh_scaler, fresh_ema)):
        row = train_step(net, optimizer, scale, x, amp, classification=True)
        scheduler.step(); shadow.update(net); continued.append(row["loss"])
    assert continued[0] == continued[1]
    history["loss"].append(continued[0])
    save(directory/"next-source.pth", model, opt, sched, scaler, ema, history)
    save(directory/"next-restored.pth", fresh, fresh_opt, fresh_sched, fresh_scaler, fresh_ema, history)
    exact(torch.load(directory/"next-source.pth", weights_only=True, map_location="cpu"),
          torch.load(directory/"next-restored.pth", weights_only=True, map_location="cpu"))
    wrong, wrong_opt, wrong_sched, wrong_scaler, wrong_ema = model_components(attention, device, amp, classes=4)
    before = {n: t.clone() for n, t in wrong.state_dict().items()}
    try:
        load_training_checkpoint(path, wrong, optimizer=wrong_opt, scheduler=wrong_sched,
                                 scaler=wrong_scaler, ema=wrong_ema, map_location="cpu")
    except TrainingCheckpointError as error:
        rejection = str(error)
    else:
        raise AssertionError("incompatible model schema accepted")
    exact(wrong.state_dict(), before)
    assert wrong_opt.state_dict()["state"] == {}
    return {"attention": attention, "amp": amp, "steps_before_save": steps,
            "strict_load": True, "exact_eval": True, "exact_next_full_state": True,
            "optimizer_scheduler_scaler_ema_rng": True, "negative_schema": rejection,
            "parameters": sum(p.numel() for p in model.parameters()),
            "source": modular.__file__, "device": str(device)}


def primitive_case(directory, kind, amp):
    directory.mkdir(parents=True, exist_ok=True)
    factories = {"conv": lambda: ConvNormAct(3, 6, stride=2, norm="group"),
                 "residual": lambda: ResidualBlock(3, 6, stride=2, attention=SEBlock(6)),
                 "eca": lambda: ECAAttention(6), "coordinate": lambda: CoordinateAttention(6)}
    model = factories[kind]().cuda()
    channels = 3 if kind in ("conv", "residual") else 6
    x = torch.randn(2, channels, 9, 13, device="cuda", requires_grad=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=128.)
    ids = [(n, id(p)) for n, p in model.named_parameters()]
    row = train_step(model, optimizer, scaler, x, amp)
    assert ids == [(n, id(p)) for n, p in model.named_parameters()]
    path = directory/"weights.pth"; torch.save(model.state_dict(), path)
    fresh = factories[kind]().cuda()
    result = fresh.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    exact(fresh.state_dict(), model.state_dict())
    model.eval(); fresh.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        actual = model(x); exact(fresh(x), actual)
    expected = (2, 6, 5, 7) if kind in ("conv", "residual") else (2, 6, 9, 13)
    assert actual.shape == expected
    return {"component": kind, "amp": amp, "shape": list(actual.shape), "step": row,
            "strict_load": True, "exact_eval": True, "device": str(next(model.parameters()).device)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "Real CUDA required; no CPU fallback or skip"
    assert Path(modular.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    with warnings.catch_warnings(record=True) as observed:
        primitives = [primitive_case(args.output/(kind+str(amp)), kind, amp)
                      for kind in ("conv", "residual", "eca", "coordinate") for amp in (False, True)]
        models = [model_case(args.output/(name+str(amp)), name, torch.device("cuda"), amp)
                  for name in ("none", "se", "cbam", "eca", "coordinate") for amp in (False, True)]
    torch.cuda.synchronize()
    result = {"python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__,
              "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability()),
              "cuda": torch.version.cuda, "source": modular.__file__, "primitives": primitives, "models": models,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "warnings": sorted(set(str(w.message) for w in observed))}
    (args.output/"summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("CUDA_MODULAR=" + json.dumps(result))


if __name__ == "__main__":
    main()
