"""Balanced deterministic CPU replay matrix; every comparison is byte exact."""
import pytest

from trainer_replay_support import CASES, CRITICAL, run_replay


@pytest.mark.parametrize("case", CASES, ids=[
    f'{c["attention"]}-{c["optimizer"]}-{c["scheduler"]}-fp{64 if c["double"] else 32}' for c in CASES])
def test_balanced_cpu_five_epoch_three_plus_two_replay(tmp_path, case):
    run_replay(case, tmp_path)


@pytest.mark.parametrize("boundary", [1, 2, 4, 5])
def test_critical_shuffled_ema_onecycle_boundary_replay(tmp_path, boundary):
    run_replay(CRITICAL, tmp_path, boundary)


@pytest.mark.parametrize("ema", [False, True])
@pytest.mark.parametrize("best", [False, True])
@pytest.mark.parametrize("boundary", [1, 2])
def test_early_stop_prefix_and_terminal_checkpoint_replay(tmp_path, ema, best, boundary):
    case = dict(CRITICAL, scheduler="plateau", early="terminal", ema=ema, best=best)
    run_replay(case, tmp_path, boundary)
