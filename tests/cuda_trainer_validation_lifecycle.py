"""Standalone native CUDA B3 validation/EMA lifecycle gate, never a CPU fake PASS."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hyso.training import Trainer
from hyso.models.WithSE import EMA
from hyso.callbacks.callbacks import EarlyStopping
import hyso.training.trainer as trainer_module


def scale(instance):
    assert type(instance.scaler) is torch.cuda.amp.GradScaler and instance.scaler.is_enabled()
    state = instance.scaler.state_dict()
    state.update(scale=128., growth_interval=1000)
    instance.scaler.load_state_dict(state)


def lifecycle_case(amp, ema, root_training):
    torch.manual_seed(51)
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2)).cuda()
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.zero_()
    # Preserve a deliberate eval child through training and validation.
    model.train(root_training)
    model[1].train()
    model[0].eval()
    model[0].requires_grad_(False)
    opt = torch.optim.SGD(model.parameters(), lr=.5)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=0, factor=.5)
    es = EarlyStopping(patience=1, verbose=False, restore_best_weights=True)
    instance = Trainer(model, optimizer=opt, loss_fn=nn.CrossEntropyLoss(), device="cuda",
                       amp=amp, ema_decay=.8 if ema else None, early_stopping=es, scheduler=plateau)
    if amp:
        scale(instance)
    target = instance.evaluation_model
    if ema:
        target.train()
        target[0].eval()
    prior_modes = [m.training for m in target.modules()]
    if not ema:
        prior_modes[0] = True  # entering training enables root, preserving mixed children
    buffers = {name: value.clone() for name, value in target.named_buffers()}
    observations, snapshots, live_snapshots, events = [], [], [], []
    def forward(module, args, out):
        if not torch.is_grad_enabled():
            observations.append((str(out.dtype), all(not m.training for m in module.modules())))
    handle = target.register_forward_hook(forward)
    def observe(event, logs):
        events.append(event)
        if event == "epoch_end":
            assert plateau.last_epoch == logs["epoch"]
            assert es.best_state is not None
            assert [m.training for m in target.modules()] == prior_modes
            snapshots.append(copy.deepcopy(target.state_dict()))
            live_snapshots.append(copy.deepcopy(model.state_dict()))
    instance.callbacks = (observe,)
    train = [(torch.zeros(2, 2), torch.ones(2, dtype=torch.long))]
    val = [(torch.zeros(size, 2), torch.zeros(size, dtype=torch.long)) for size in (2, 1, 1)]
    try:
        history = instance.fit(train, val, epochs=8)
    finally:
        handle.remove()
    assert len(history["val_loss"]) == 2 and es.early_stop
    assert history["val_loss"][0] < history["val_loss"][1]
    assert instance.update_count == instance.attempt_count == plateau.last_epoch == 2
    assert instance.ema_update_count == (2 if ema else 0)
    assert observations == [("torch.float16" if amp else "torch.float32", True)] * 6
    assert events == ["train_begin", "epoch_end", "epoch_end", "train_end"]
    assert [m.training for m in target.modules()] == prior_modes
    for name, value in target.state_dict().items():
        assert torch.equal(value, snapshots[0][name])
    for name, value in buffers.items():
        assert torch.equal(dict(target.named_buffers())[name], value)
    if ema:
        assert all(torch.equal(value, live_snapshots[-1][name]) for name, value in model.state_dict().items())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    try:
        instance.fit(train, val, epochs=8)
    except RuntimeError:
        pass
    else:
        raise AssertionError("best-restored Trainer must be terminal")
    return {"id": f"lifecycle-{amp}-{ema}-{root_training}", "epochs": 2, "updates": 2,
            "ema_updates": instance.ema_update_count, "val_loss": history["val_loss"],
            "lr": opt.param_groups[0]["lr"], "events": events}


def overflow_case(opt_type, all_skipped):
    torch.manual_seed(8)
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2)).cuda()
    opt = opt_type(model.parameters(), lr=.01)
    onecycle = torch.optim.lr_scheduler.OneCycleLR(opt, .02, total_steps=4, cycle_momentum=False)
    instance = Trainer(model, optimizer=opt, loss_fn=nn.CrossEntropyLoss(), device="cuda",
                       amp=True, accum_steps=2, ema_decay=.8, scheduler=onecycle)
    scale(instance)
    reference = EMA(model, decay=.8)
    calls, changes, buffer_pairs = [], [], []
    def poison(gradient):
        calls.append(True)
        return torch.full_like(gradient, float("inf")) if all_skipped or len(calls) == 1 else gradient
    handle = model[1].weight.register_hook(poison)
    reference_handle = opt.register_step_post_hook(lambda *args: reference.update(model))
    real_step = instance.scaler.step
    def observed_step(optimizer):
        before = [p.clone() for p in model.parameters()]
        result = real_step(optimizer)
        changes.append(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
        return result
    instance.scaler.step = observed_step
    val = [(torch.zeros(3, 2), torch.tensor([0, 1, 0]))]
    train = [(torch.randn(2, 2) * .1, torch.tensor([0, 1])) for _ in range(3)]
    def observer(event, logs):
        if event == "epoch_end":
            # Pure validation cannot advance buffers beyond the last native EMA update.
            for name, value in instance.ema.ema_model.state_dict().items():
                torch.testing.assert_close(value, reference.ema_model.state_dict()[name], rtol=0, atol=0)
            buffer_pairs.append([int(model[0].num_batches_tracked), int(instance.ema.ema_model[0].num_batches_tracked)])
    instance.callbacks = (observer,)
    try:
        history = instance.fit(train, val, epochs=2)
    finally:
        handle.remove()
        reference_handle.remove()
    expected = 0 if all_skipped else 3
    assert instance.attempt_count == 4
    assert instance.update_count == instance.ema_update_count == onecycle.last_epoch == expected
    assert changes == ([False] * 4 if all_skipped else [False, True, True, True])
    assert instance.scaler.get_scale() == (8. if all_skipped else 64.)
    assert len(history["val_loss"]) == 2
    assert buffer_pairs == ([[3, 0], [6, 0]] if all_skipped else [[3, 3], [6, 6]])
    return {"id": f"overflow-ema-{opt_type.__name__}-{all_skipped}", "attempts": 4,
            "updates": expected, "ema_updates": instance.ema_update_count,
            "scale": instance.scaler.get_scale(), "bn_live_shadow_counts": buffer_pairs}


def failure_case(amp):
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2)).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=.01)
    instance = Trainer(model, optimizer=opt, loss_fn=nn.CrossEntropyLoss(), device="cuda", amp=amp, ema_decay=.9)
    if amp:
        scale(instance)
    target = instance.evaluation_model
    target.train()
    target[0].eval()
    modes = [m.training for m in target.modules()]
    before = copy.deepcopy(target.state_dict())
    def fail(*args):
        raise RuntimeError("EMA validation failed")
    target.register_forward_pre_hook(fail)
    try:
        instance._validate([(torch.zeros(2, 2), torch.tensor([0, 1]))])
    except RuntimeError as exc:
        assert "EMA validation failed" in str(exc)
    else:
        raise AssertionError("EMA failure must not fall back to live model")
    assert [m.training for m in target.modules()] == modes
    assert all(torch.equal(value, before[name]) for name, value in target.state_dict().items())
    return {"id": f"ema-validation-exception-{amp}", "mode_restored": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: native B3 gate cannot PASS")
    assert Path(trainer_module.__file__).resolve().is_relative_to(ROOT)
    torch.set_num_threads(1)
    rows = [lifecycle_case(amp, ema, root) for amp in (False, True) for ema in (False, True) for root in (False, True)]
    rows += [overflow_case(opt, all_skipped) for opt in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW)
             for all_skipped in (False, True)]
    rows += [failure_case(amp) for amp in (False, True)]
    assert len(rows) == len({row["id"] for row in rows}) == 16
    torch.cuda.synchronize()
    result = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "trainer_source": trainer_module.__file__, "passed": len(rows), "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}))


if __name__ == "__main__":
    main()
