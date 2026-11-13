"""Small native CUDA restore/continuation/rollback gate; B4-C/B5 remain separate."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hyso.training import Trainer, resume
from hyso.models.modular import ModularCNN
from hyso import checkpoint


def equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equal(a, b)
    else:
        assert actual == expected


def build(amp, overflow=False):
    cfg = dict(input_channels=1, num_classes=2, widths=[3], repeats=[1], strides=[1], attention="eca", norm="batch")
    model = ModularCNN(**cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.03)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, .05, total_steps=10)
    trainer = Trainer(model, optimizer=opt, scheduler=scheduler, model_config=cfg, loss_fn=nn.CrossEntropyLoss(),
                      device="cuda", amp=amp, accum_steps=2, ema_decay=.8)
    if amp:
        state = trainer.scaler.state_dict()
        state.update(scale=1e35 if overflow else 128., growth_interval=1000)
        trainer.scaler.load_state_dict(state)
    generator = torch.Generator().manual_seed(932)
    ds = torch.utils.data.TensorDataset(torch.randn(5, 1, 5, 5, generator=generator) * .1, torch.tensor([0, 1, 1, 0, 1]))
    return trainer, torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True), torch.utils.data.DataLoader(ds, batch_size=2)


def interrupted(path, amp, overflow=False):
    trainer, train, val = build(amp, overflow)
    native = resume.save_last
    def save_then_interrupt(*args):
        native(*args)
        if len(args[0].history["train_loss"]) == 3:
            torch.rand(7, device="cuda")  # deliberately disturb post-checkpoint GPU RNG
            raise InterruptedError("controlled epoch three interruption")
    resume.save_last = save_then_interrupt
    try:
        trainer.fit(train, val, epochs=5, save_last_to=path)
    except InterruptedError:
        pass
    else:
        raise AssertionError("expected controlled interruption")
    finally:
        resume.save_last = native
    assert torch.load(path, weights_only=True)["completed_epochs"] == 3
    return trainer


def continuation(directory, amp, overflow=False):
    torch.manual_seed(73)
    baseline, train, val = build(amp, overflow)
    baseline.fit(train, val, epochs=5)
    expected_rng = checkpoint._rng_state()
    torch.manual_seed(73)
    path = directory / f"cuda-{amp}-{overflow}.pth"
    interrupted(path, amp, overflow)
    target, train, val = build(amp, overflow)
    torch.cuda.manual_seed_all(777)
    target.fit(train, val, epochs=5, resume_from=path, save_last_to=path)
    equal(target.model.state_dict(), baseline.model.state_dict())
    equal(target.optimizer.state_dict(), baseline.optimizer.state_dict())
    equal(target.scheduler.state_dict(), baseline.scheduler.state_dict())
    equal(target.scaler.state_dict(), baseline.scaler.state_dict())
    equal(target.ema.ema_model.state_dict(), baseline.ema.ema_model.state_dict())
    equal(target.history, baseline.history)
    equal(checkpoint._rng_state(), expected_rng)
    assert target.attempt_count == 10
    assert target.update_count == target.ema_update_count == target.scheduler.last_epoch == (0 if overflow else 10)
    assert torch.load(path, weights_only=True)["completed_epochs"] == 5
    return {"id": f"continuation-{amp}-{overflow}", "attempts": target.attempt_count,
            "updates": target.update_count, "scaler": target.scaler.state_dict()}


def state(trainer):
    return dict(model=copy.deepcopy(trainer.model.state_dict()), optimizer=copy.deepcopy(trainer.optimizer.state_dict()),
                scheduler=copy.deepcopy(trainer.scheduler.state_dict()), scaler=copy.deepcopy(vars(trainer.scaler)),
                ema=copy.deepcopy(trainer.ema.ema_model.state_dict()), history=copy.deepcopy(trainer.history),
                fields={key: getattr(trainer, key) for key in resume._TRAINER_FIELDS if key != "history"},
                modes=[m.training for model in (trainer.model, trainer.evaluation_model) for m in model.modules()],
                gradients=[None if p.grad is None else p.grad.clone() for p in trainer.model.parameters()],
                rng=checkpoint._rng_state())


def negative_rollback(directory, amp):
    torch.manual_seed(22)
    path = directory / f"rollback-{amp}.pth"
    interrupted(path, amp)
    target, train, val = build(amp)
    x, y = train.dataset.tensors
    # Real warm optimizer state, then (for AMP) a real in-flight unscale stage.
    nn.functional.cross_entropy(target.model(x[:2].cuda()), y[:2].cuda()).backward()
    target.optimizer.step()
    target.optimizer.zero_grad()
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        loss = nn.functional.cross_entropy(target.model(x[:2].cuda()), y[:2].cuda())
    target.scaler.scale(loss).backward()
    if amp:
        target.scaler.unscale_(target.optimizer)
    target.model.eval()
    target.model.classifier.train()
    target.evaluation_model.train()
    target.evaluation_model.classifier.eval()
    random.seed(121)
    np.random.seed(122)
    torch.manual_seed(123)
    before = state(target)
    native = resume._apply_progress
    def fail(*args):
        native(*args)
        raise RuntimeError("CUDA late metadata commit failure")
    resume._apply_progress = fail
    try:
        target.fit(train, val, epochs=5, resume_from=path)
    except checkpoint.TrainingCheckpointError as error:
        assert "previous runtime restored" in str(error)
    else:
        raise AssertionError("expected late metadata failure")
    finally:
        resume._apply_progress = native
    equal(state(target), before)
    # A second, successful restore also clears the old UNSCALED bookkeeping.
    target.fit(train, val, epochs=5, resume_from=path)
    assert target.update_count == 10 and target._finished
    return {"id": f"rollback-{amp}", "all_runtime_restored": True, "retry_updates": target.update_count,
            "inflight_scaler_restored_then_cleared": amp}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("native CUDA required: no simulated restore PASS")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = [continuation(args.output.parent, amp) for amp in (False, True)]
    rows += [continuation(args.output.parent, True, True)]
    rows += [negative_rollback(args.output.parent, amp) for amp in (False, True)]
    torch.cuda.synchronize()
    assert len(rows) == len({row["id"] for row in rows}) == 5
    result = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "passed": 5, "cases": rows}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
