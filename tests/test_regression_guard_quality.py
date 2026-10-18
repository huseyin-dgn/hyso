"""Quality controls for xfail isolation; these do not simulate production fixes."""

from pathlib import Path
import json
import os
import re
import sys


pytest_plugins = ["pytester"]


def run_isolated(pytester, monkeypatch, *args):
    tests = Path(__file__).parent
    paths = [str(tests), str(tests.parent), os.environ.get("PYTHONPATH", "")]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, paths)))
    # Native torch/torchvision registrations outlive pytester's sys.modules reset.
    # A fresh interpreter keeps both controls independent of collection order.
    return pytester.run(sys.executable, "-B", "-m", "pytest", *args, timeout=60)


def test_xfail_markers_use_specific_risk_signals(pytester, monkeypatch):
    # Audit all definitions even when this quality module is selected alone.
    # Collection-only: no production test is run recursively.
    pytester.makepyfile(guard_audit_plugin='''
import json
def pytest_collection_finish(session):
    from regression_guards import KnownRegression
    rows = []
    for item in session.items:
        marker = item.get_closest_marker("xfail")
        if marker is None:
            continue
        allowed = marker.kwargs["raises"]
        allowed = allowed if isinstance(allowed, tuple) else (allowed,)
        rows.append(dict(strict=marker.kwargs.get("strict"), reason=marker.kwargs["reason"],
                         names=[cls.__name__ for cls in allowed],
                         specific=all(isinstance(cls, type) and issubclass(cls, KnownRegression)
                                      and cls is not KnownRegression for cls in allowed)))
    session.config.pluginmanager.getplugin("terminalreporter").write_line(
        "MARKER_JSON=" + json.dumps(rows))
''')
    tests = Path(__file__).parent
    result = run_isolated(pytester, monkeypatch, "-c", str(tests / "pytest.ini"),
                          str(tests), "--collect-only", "-q", "-p", "guard_audit_plugin")
    assert result.ret == 0
    marked = json.loads(next(line.removeprefix("MARKER_JSON=")
                             for line in result.stdout.str().splitlines()
                             if line.startswith("MARKER_JSON=")))
    assert marked
    for marker in marked:
        assert marker["strict"] is True
        assert re.match(r"R\d{2}:", marker["reason"])
        assert marker["specific"]
        risk = marker["reason"].split(":", 1)[0]
        assert all(name.startswith(risk) for name in marker["names"])


def test_guard_errors_fail_and_fixed_boundary_xpasses_strict(pytester, monkeypatch):
    pytester.makeini("[pytest]\naddopts = -p no:cacheprovider\n")
    pytester.makepyfile(test_guard_controls='''
from types import SimpleNamespace
import pytest
import torch
from regression_guards import (
    R03FloatBufferCopy, R14EnableIgnored, expect_tensor, require_optimizer_activity,
)
from test_ema_regression import require_early_ema_rejection

known = pytest.mark.xfail(strict=True, reason="R03: control boundary", raises=R03FloatBufferCopy)

@known
def test_known_value_mismatch():
    expect_tensor(torch.zeros(2), torch.ones(2), R03FloatBufferCopy, "R03: copy")

@known
def test_unrelated_helper_assertion():
    assert False, "unrelated helper failure"

@known
def test_unrelated_tensor_schema():
    expect_tensor(torch.zeros(2), torch.ones(3), R03FloatBufferCopy, "schema is not a copy mismatch")

@known
def test_unexpected_runtime_exception():
    raise RuntimeError("unexpected helper exception")

@known
def test_zero_optimizer_activity():
    require_optimizer_activity(SimpleNamespace(before=[], after=[]), (2,))

@known
def test_fixed_boundary():
    expected = torch.ones(2)
    expect_tensor(expected.clone(), expected, R03FloatBufferCopy, "correct copy")

def observation():
    model = torch.nn.BatchNorm1d(2)
    return SimpleNamespace(
        model=model, error=ValueError("EMA unavailable"), parent_forwards=[],
        shadow_forwards=[], updates=[], optimizer=SimpleNamespace(after=[]),
        before={key: value.clone() for key, value in model.state_dict().items()},
    )

@pytest.mark.parametrize("side_effect", ["forward", "step", "parameter", "floating_buffer", "integer_buffer"])
@pytest.mark.xfail(strict=True, reason="R14: control early rejection", raises=R14EnableIgnored)
def test_late_ema_rejection_is_a_normal_failure(side_effect):
    result = observation()
    with torch.no_grad():
        if side_effect == "forward": result.parent_forwards.append((True, "cpu"))
        elif side_effect == "step": result.optimizer.after.append([])
        elif side_effect == "parameter": result.model.weight.add_(1)
        elif side_effect == "floating_buffer": result.model.running_mean.add_(1)
        else: result.model.num_batches_tracked.add_(1)
    require_early_ema_rejection(result)

@pytest.mark.xfail(strict=True, reason="R14: control early rejection", raises=R14EnableIgnored)
def test_correct_early_ema_rejection():
    require_early_ema_rejection(observation())
''')
    result = run_isolated(pytester, monkeypatch, "-q", "--tb=short", "--show-capture=no")
    # Nested negative controls: nine ordinary failures, two XPASS(strict)
    # failures, one genuine XFAIL. The outer quality test must pass this gate.
    result.assert_outcomes(passed=0, failed=11, xfailed=1, xpassed=0, skipped=0, errors=0)
    assert result.ret == 1
    assert result.stdout.str().count("[XPASS(strict)]") == 2
