"""Bounded standalone P19 native gate; invoke through a timed supervisor.

No pytest conftest (which intentionally disables CUDA), downloads or profiler.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn

sys.path.insert(0, os.environ.get("HYSO_TEST_PACKAGE_ROOT", str(Path(__file__).resolve().parents[1])))
from hyso.training import Trainer
from hyso.performance import measure_training, TrainingTimingProtocol
import hyso.performance.training as timing_module
from trainer_replay_support import equal, seed, snapshot, build, CRITICAL


def run(optimizer_type, amp, nondefault, overflow):
    states, timing = [], None
    stream = torch.cuda.Stream() if nondefault else torch.cuda.default_stream()
    for observed in (False, True):
        seed(192)
        with torch.cuda.stream(stream):
            model = nn.Sequential(nn.Linear(3, 4), nn.Dropout(.2), nn.Linear(4, 2)).cuda()
            opt = optimizer_type(model.parameters(), lr=.01)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, .02, total_steps=4, cycle_momentum=False)
            t = Trainer(model, optimizer=opt, loss_fn=nn.CrossEntropyLoss(), device="cuda",
                        amp=amp, accum_steps=2, scheduler=scheduler, ema_decay=.8)
            if amp:
                state = t.scaler.state_dict()
                state.update(scale=128., growth_interval=1000)
                t.scaler.load_state_dict(state)
            calls = []
            def poison(gradient):
                calls.append(True)
                return torch.full_like(gradient, float("inf")) if len(calls) == 1 else gradient
            handle = model[0].weight.register_hook(poison) if overflow else None
            train = list(zip(torch.linspace(-1, 1, 15).reshape(5, 3).split(2),
                             torch.tensor([0, 1, 0, 1, 0]).split(2)))
            try:
                if observed:
                    timing = measure_training(t, train, train, epochs=2,
                        protocol=TrainingTimingProtocol(event_buffer=2)).to_dict()
                else:
                    t.fit(train, train, epochs=2)
            finally:
                if handle is not None:
                    handle.remove()
            stream.synchronize()
            assert not hasattr(t, "_timing_recorder") and not opt._optimizer_step_post_hooks
            assert t.attempt_count == 4 and t.microbatch_count == 6
            assert t.update_count == t.ema_update_count == scheduler.last_epoch == 4 - int(overflow)
            states.append(snapshot(t))
    equal(*states)
    attempts = [r for r in timing["observations"] if r["kind"] == "attempt"]
    assert sum(r["skipped"] for r in attempts) == int(overflow)
    assert [r["applied"] for r in attempts] == ([False, True, True, True] if overflow else [True] * 4)
    assert timing["instrumentation"]["max_pending_event_pairs"] == 2
    assert timing["instrumentation"]["pending_event_pairs"] == 0
    assert timing["instrumentation"]["stream_id"] == int(stream.cuda_stream)
    assert all(r["cuda_event_ms"] is not None and r["cuda_event_ms"] >= 0 for r in timing["observations"])
    return dict(id=f"{optimizer_type.__name__}-amp{amp}-stream{nondefault}-overflow{overflow}",
                timing=timing, exact_state_rng_history=True)


def checkpoint_case(amp):
    import hyso.training.resume as resume
    states = []
    with tempfile.TemporaryDirectory(prefix="hyso-p19-native-resume-") as directory:
        for observed in (False, True):
            seed(35)
            t, train, val = build(CRITICAL, device="cuda", amp=amp)
            path = Path(directory) / (str(observed) + ".pth")
            original = resume.save_last
            class Stop(Exception):
                pass
            def cut(*args):
                original(*args)
                if len(args[0].history["train_loss"]) == 2:
                    raise Stop()
            try:
                with patch.object(resume, "save_last", cut):
                    if observed:
                        measure_training(t, train, val, epochs=5, save_last_to=path)
                    else:
                        t.fit(train, val, epochs=5, save_last_to=path)
            except Stop:
                pass
            else:
                raise AssertionError("checkpoint boundary not reached")
            assert not hasattr(t, "_timing_recorder")
            seed(98)
            t, train, val = build(CRITICAL, device="cuda", amp=amp)
            if observed:
                result = measure_training(t, train, val, epochs=5, resume_from=path).to_dict()
                assert [e["epoch_id"] for e in result["epochs"]] == [3, 4, 5]
            else:
                t.fit(train, val, epochs=5, resume_from=path)
            torch.cuda.current_stream().synchronize()
            states.append(snapshot(t))
    equal(*states)
    return dict(id=f"checkpoint-resume-amp{amp}", exact_state_rng_history=True, timing=result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("native CUDA gate requires an available device")
    torch.set_num_threads(1)
    # Same exact-replay preconditions as cuda_trainer_replay_conformance.py.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.15)
    rows = []
    for opt in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
        for amp in (False, True):
            for stream in (False, True):
                for overflow in ((False, True) if amp else (False,)):
                    rows.append(run(opt, amp, stream, overflow))
    rows.extend(checkpoint_case(amp) for amp in (False, True))
    if os.environ.get("HYSO_TEST_PACKAGE_ROOT"):
        assert Path(timing_module.__file__).resolve().is_relative_to(Path(os.environ["HYSO_TEST_PACKAGE_ROOT"]).resolve())
    Path(args.output).write_text(json.dumps(dict(status="passed", package_source=timing_module.__file__,
                                               scenarios=rows), indent=2), encoding="utf-8")
    print(f"P19 native: {len(rows)} scenarios passed")


if __name__ == "__main__":
    main()
