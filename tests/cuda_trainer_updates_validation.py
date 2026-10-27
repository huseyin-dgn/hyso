"""Standalone native CUDA B2 gate; fails explicitly when CUDA is unavailable.

Run outside the CPU pytest fixture:
  python -B tests/cuda_trainer_updates_validation.py --output <result.json>
No datasets, weights or dependencies are downloaded.
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import warnings
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hyso.training import Trainer
from hyso.models.modular import ModularCNN
import hyso.training.trainer as trainer_module


OPTIMIZERS = (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW)


def data(task="multiclass"):
    x = torch.randn(7, 3) * .1
    y = torch.tensor([0, 1, 0, 1, 1, 0, 1])
    return list(zip(x.split([2, 1, 3, 1]), (y if task == "multiclass" else y.float()).split([2, 1, 3, 1])))


def configure_scale(instance):
    assert type(instance.scaler) is torch.cuda.amp.GradScaler and instance.scaler.is_enabled()
    state = instance.scaler.state_dict()
    state.update(scale=128., growth_interval=1000)
    instance.scaler.load_state_dict(state)


def normal_case(optimizer_type, amp, task):
    torch.manual_seed(51)
    model = nn.Linear(3, 2 if task == "multiclass" else 1).cuda()
    identities = [id(p) for p in model.parameters()]
    before = [p.detach().clone() for p in model.parameters()]
    opt = optimizer_type(model.parameters(), lr=.03)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=.05, total_steps=4,
                                                   cycle_momentum=False)
    instance = Trainer(model, optimizer=opt,
                       loss_fn=nn.CrossEntropyLoss() if task == "multiclass" else nn.BCEWithLogitsLoss(),
                       device="cuda", amp=amp, task=task, accum_steps=2, grad_clip=.1, scheduler=scheduler)
    if amp:
        configure_scale(instance)
    dtypes, norms = [], []
    model.register_forward_hook(lambda module, args, out: dtypes.append(str(out.dtype)))
    opt.register_step_pre_hook(lambda *args: norms.append(float(torch.linalg.vector_norm(
        torch.cat([p.grad.flatten() for p in model.parameters()])))))
    history = instance.fit(data(task), epochs=2)
    torch.cuda.synchronize()
    assert instance.update_count == instance.attempt_count == scheduler.last_epoch == 4
    assert instance.microbatch_count == 8
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert identities == [id(p) for p in model.parameters()]
    assert all(p.dtype == torch.float32 and p.device.type == "cuda" for p in model.parameters())
    assert dtypes == ["torch.float16" if amp else "torch.float32"] * 8
    assert all(norm <= .100001 for norm in norms)
    assert all(p.grad is None for p in model.parameters())
    assert all(0 <= value <= 100 for value in history["train_acc"])
    assert torch.isfinite(torch.tensor(history["train_loss"])).all()
    assert history["val_loss"] == history["val_acc"] == [None, None]
    return {"id": f"normal-{optimizer_type.__name__}-{amp}-{task}", "updates": instance.update_count,
            "dtype": dtypes[0], "clip_norms": norms, "amp": amp}


def overflow_case(optimizer_type, kind, all_skipped):
    torch.manual_seed(93)
    model = nn.Linear(3, 2).cuda()
    opt = optimizer_type(model.parameters(), lr=.02)
    if kind == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, .03, total_steps=4, cycle_momentum=False)
    elif kind == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=.5)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2)
    instance = Trainer(model, optimizer=opt, loss_fn=nn.CrossEntropyLoss(), device="cuda",
                       amp=True, accum_steps=3, grad_clip=.2, scheduler=scheduler)
    configure_scale(instance)
    backward_calls, calls, snapshots, returns, scales, applied_lrs = [], [], [], [], [], []
    real_step, real_update = instance.scaler.step, instance.scaler.update
    hook_ids = set(opt._optimizer_step_post_hooks)
    external = opt.register_step_post_hook(lambda *args: calls.append(True))
    hooks_with_external = set(opt._optimizer_step_post_hooks)
    lr_handle = opt.register_step_pre_hook(lambda optimizer, *args: applied_lrs.append(optimizer.param_groups[0]["lr"]))

    def poison(gradient):
        backward_calls.append(True)
        return torch.full_like(gradient, float("inf")) if all_skipped or len(backward_calls) == 1 else gradient

    poison_handle = model.weight.register_hook(poison)
    def step(optimizer):
        before = [p.detach().clone() for p in model.parameters()]
        result = real_step(optimizer)
        snapshots.append(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
        returns.append(result)
        return result

    def update():
        before = instance.scaler.get_scale()
        result = real_update()
        scales.append([before, instance.scaler.get_scale()])
        return result

    try:
        with patch.object(instance.scaler, "step", step), patch.object(instance.scaler, "update", update):
            with warnings.catch_warnings(record=True) as observed_warnings:
                warnings.simplefilter("always")
                history = instance.fit(data(), epochs=2)
    finally:
        poison_handle.remove()
        assert set(opt._optimizer_step_post_hooks) == hooks_with_external
        external.remove()
        lr_handle.remove()
    assert set(opt._optimizer_step_post_hooks) == hook_ids
    expected = 0 if all_skipped else 3
    assert len(backward_calls) == 8
    assert instance.attempt_count == 4 and instance.update_count == len(calls) == expected
    assert snapshots == ([False] * 4 if all_skipped else [False, True, True, True])
    assert returns == [None] * 4  # None occurs on both real success and real skip.
    assert scales[0] == [128., 64.]
    assert instance.scaler.get_scale() == (8. if all_skipped else 64.)
    assert scheduler.last_epoch == (expected if kind == "onecycle" else 2)
    if kind == "onecycle":
        assert scheduler.total_steps == 4
        ref_opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=.02)
        ref_schedule = torch.optim.lr_scheduler.OneCycleLR(ref_opt, .03, total_steps=4, cycle_momentum=False)
        expected_lrs = []
        for _ in range(expected):
            expected_lrs.append(ref_opt.param_groups[0]["lr"])
            ref_opt.step()
            ref_schedule.step()
        assert applied_lrs == expected_lrs
    if optimizer_type in (torch.optim.Adam, torch.optim.AdamW) and not all_skipped:
        assert all(int(state["step"].item()) == expected for state in opt.state.values())
    if all_skipped:
        assert not opt.state
    assert all(p.grad is None and torch.isfinite(p).all() for p in model.parameters())
    assert len(history["train_loss"]) == 2
    return {"id": f"overflow-{optimizer_type.__name__}-{kind}-{all_skipped}",
            "attempts": 4, "updates": expected, "scheduler_last_epoch": scheduler.last_epoch,
            "changed_per_attempt": snapshots, "scale_transitions": scales,
            "applied_lrs": applied_lrs,
            "warnings": [str(item.message) for item in observed_warnings]}


def order_case():
    torch.manual_seed(21)
    model = nn.Linear(3, 2).cuda()
    reference = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=.01)
    instance = Trainer(model, optimizer=opt, loss_fn=nn.CrossEntropyLoss(), device="cuda",
                       amp=True, accum_steps=8, grad_clip=.01)
    configure_scale(instance)
    dataset = data()
    # Independent native autocast gradient reference; no scaler or Trainer.
    for x, y in dataset:
        with torch.autocast("cuda", dtype=torch.float16):
            loss = nn.functional.cross_entropy(reference(x.cuda()), y.cuda())
        (loss * (len(x) / 7)).backward()
    expected = [p.grad.clone() for p in reference.parameters()]
    events, unscaled, post_norm = [], [], []
    real_unscale = instance.scaler.unscale_
    real_clip = torch.nn.utils.clip_grad_norm_
    def unscale(optimizer):
        events.append("unscale")
        result = real_unscale(optimizer)
        unscaled.extend(p.grad.detach().clone() for p in model.parameters())
        return result
    def clip(parameters, maximum, **kwargs):
        events.append("clip")
        parameters = list(parameters)
        for p, summed, reference_gradient in zip(parameters, unscaled, expected):
            torch.testing.assert_close(p.grad, summed / 7, rtol=0, atol=0)
            torch.testing.assert_close(p.grad, reference_gradient, rtol=.004, atol=.0001)
        result = real_clip(parameters, maximum, **kwargs)
        post_norm.append(float(torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in parameters]))))
        assert post_norm[-1] <= maximum + 1e-6
        return result
    with patch.object(instance.scaler, "unscale_", unscale), patch.object(torch.nn.utils, "clip_grad_norm_", clip):
        instance.fit(dataset, epochs=1)
    assert events == ["unscale", "clip"] and instance.update_count == 1
    return {"id": "native-amp-unscale-normalize-clip", "events": events, "post_clip_norm": post_norm[0]}


def modular_case():
    torch.manual_seed(4)
    model = ModularCNN(input_channels=1, num_classes=2, widths=(3,), repeats=(1,),
                       strides=(1,), norm=None, attention="eca").cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    instance = Trainer(model, optimizer=opt, loss_fn=nn.CrossEntropyLoss(), device="cuda",
                       amp=True, accum_steps=2)
    configure_scale(instance)
    before = model.classifier.weight.detach().clone()
    dataset = [(torch.randn(2, 1, 5, 7) * .1, torch.tensor([0, 1])) for _ in range(3)]
    history = instance.fit(dataset, epochs=2)
    assert instance.update_count == 4
    assert not torch.equal(before, model.classifier.weight)
    assert torch.isfinite(torch.tensor(history["train_loss"])).all()
    return {"id": "modular-cuda-amp", "updates": instance.update_count}


def growth_case():
    torch.manual_seed(81)
    model = nn.Linear(3, 2).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=.01)
    schedule = torch.optim.lr_scheduler.OneCycleLR(opt, .02, total_steps=4, cycle_momentum=False)
    instance = Trainer(model, optimizer=opt, loss_fn=nn.CrossEntropyLoss(), device="cuda",
                       amp=True, accum_steps=2, scheduler=schedule)
    configure_scale(instance)
    state = instance.scaler.state_dict()
    state["growth_interval"] = 1
    instance.scaler.load_state_dict(state)
    instance.fit(data(), epochs=2)
    assert instance.update_count == instance.attempt_count == schedule.last_epoch == 4
    assert instance.scaler.get_scale() == 2048.
    return {"id": "native-scaler-growth", "initial_scale": 128., "final_scale": 2048., "updates": 4}


def boundary_cases():
    rows = []
    for kind in ("model_device", "amp_double", "amp_input_double", "device_index", "fused_SGD", "fused_Adam", "fused_AdamW"):
        model = nn.Linear(3, 2)
        if kind != "model_device":
            model.cuda()
        if kind == "amp_double":
            model.double()
        opt_type = getattr(torch.optim, kind.split("_")[1]) if kind.startswith("fused_") else torch.optim.SGD
        opt = opt_type(model.parameters(), lr=.01, **({"fused": True} if kind.startswith("fused_") else {}))
        try:
            instance = Trainer(model, optimizer=opt, loss_fn=nn.CrossEntropyLoss(), amp=True,
                               device=f"cuda:{torch.cuda.device_count()}" if kind == "device_index" else "cuda")
            if kind == "amp_input_double":
                instance.fit([(torch.ones(2, 3, dtype=torch.float64), torch.zeros(2, dtype=torch.long))])
        except (ValueError, TypeError, NotImplementedError) as exc:
            rows.append({"id": "reject-" + kind, "error": type(exc).__name__, "message": str(exc)})
        else:
            raise AssertionError("invalid CUDA profile accepted: " + kind)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("native CUDA required: this gate cannot report PASS without a GPU")
    assert Path(trainer_module.__file__).resolve().is_relative_to(ROOT)
    torch.set_num_threads(1)
    rows = [normal_case(opt, amp, task) for opt in OPTIMIZERS
            for amp in (False, True) for task in ("multiclass", "binary")]
    rows += [overflow_case(opt, kind, all_skipped) for opt in OPTIMIZERS
             for kind in ("onecycle", "step", "cosine") for all_skipped in (False, True)]
    rows += [order_case(), modular_case(), growth_case(), *boundary_cases()]
    assert len(rows) == len({row["id"] for row in rows}) == 40
    result = {"torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "trainer_source": trainer_module.__file__,
              "passed": len(rows), "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}))


if __name__ == "__main__":
    main()
