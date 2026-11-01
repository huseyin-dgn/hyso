"""Native CUDA replay gate. No CPU fallback, skips, or approximate assertions."""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import torch
from torch.utils.data import DataLoader
from hyso import checkpoint
from trainer_replay_support import CRITICAL, run_replay, equal
# Reuse real warm-runtime rollback probes rather than duplicating fault injection.
from cuda_trainer_checkpoint_restore import build, interrupted, negative_rollback, state


def negative_plan(directory, amp):
    path = directory / f"mismatch-{amp}.pth"
    interrupted(path, amp)
    target, train, val = build(amp)
    train.dataset.tensors[0][0, 0, 0, 0] += 1
    before = state(target)
    def forbidden(loader):
        raise AssertionError("mismatch restore must not open a loader iterator")
    with patch.object(DataLoader, "__iter__", forbidden):
        try:
            target.fit(train, val, epochs=5, resume_from=path)
        except checkpoint.TrainingCheckpointError as error:
            assert "plan mismatch" in str(error)
        else:
            raise AssertionError("mismatched dataset accepted")
    equal(state(target), before)
    return dict(id=f"negative-plan-{amp}", runtime_unchanged=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("real NVIDIA CUDA required; no CPU fallback")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    for amp in (False, True):
        for ema in (False, True):
            name = f"fp{16 if amp else 32}-ema-{ema}"
            case = dict(CRITICAL, ema=ema)
            result = run_replay(case, args.output.parent / name, device="cuda", amp=amp)
            cases.append(dict(id=name, **result))
    for ema in (False, True):
        name = f"native-overflow-recovery-ema-{ema}"
        result = run_replay(dict(CRITICAL, ema=ema), args.output.parent / name,
                            device="cuda", amp=True, overflow=True)
        cases.append(dict(id=name, **result))
    for amp in (False, True):
        cases.append(negative_plan(args.output.parent, amp))
        cases.append(negative_rollback(args.output.parent, amp))
    assert len(cases) == len({case["id"] for case in cases}) == 10
    torch.cuda.synchronize()
    result = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), passed=len(cases),
                  deterministic_algorithms=True, tolerance=0, cases=cases)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
