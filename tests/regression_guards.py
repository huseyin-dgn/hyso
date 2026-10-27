"""Narrow known-failure signals; ordinary helper errors are never converted."""

import torch


class KnownRegression(AssertionError):
    """Base for auditability only: never use this broad base in an xfail."""


class R01ForwardMissing(KnownRegression): pass
class R02StorageAlias(KnownRegression): pass
class R02SnapshotMutation(KnownRegression): pass
class R02RestoreMismatch(KnownRegression): pass
class R03FloatBufferCopy(KnownRegression): pass
class R03IntegerBufferCopy(KnownRegression): pass
class R06RandomValidation(KnownRegression): pass
class R06ValidationVariation(KnownRegression): pass
class R07SharedTransform(KnownRegression): pass
class R07EmptyValidation(KnownRegression): pass
class R08UnknownClass(KnownRegression): pass
class R08SubsetMapping(KnownRegression): pass
class R09TailStep(KnownRegression): pass
class R09TailWeights(KnownRegression): pass
class R09SampleWeighting(KnownRegression): pass
class R09ZeroAccumulation(KnownRegression): pass
class R09AcceptedAccumulation(KnownRegression): pass
class R09OptimizerUpdates(KnownRegression): pass
class R10EpochSchedule(KnownRegression): pass
class R10OneCycleSchedule(KnownRegression): pass
class R11UpdateBudget(KnownRegression): pass
class R12CPUSelection(KnownRegression): pass
class R14EnableIgnored(KnownRegression): pass
class R14DisableIgnored(KnownRegression): pass


def expect_value(actual, expected, failure, message):
    """Only an explicit value mismatch emits the selected risk signal."""
    assert isinstance(failure, type) and issubclass(failure, KnownRegression)
    assert failure is not KnownRegression
    if actual != expected:
        raise failure(f"{message}: actual={actual!r}, expected={expected!r}")


def expect_tensor(actual, expected, failure, message, *, rtol=0, atol=0):
    # Schema, device and numeric corruption are prerequisites, not known bugs.
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    equal = torch.allclose(actual, expected, rtol=rtol, atol=atol)
    expect_value(equal, True, failure, message)


def expect_state(actual, expected, failure, message, *, rtol=0, atol=0):
    assert list(actual) == list(expected)
    # Validate every prerequisite before emitting any known-mismatch signal.
    for key in expected:
        a, b = actual[key], expected[key]
        assert a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
    for key in expected:
        expect_tensor(actual[key], expected[key], failure, f"{message}: {key}", rtol=rtol, atol=atol)


def require_optimizer_activity(trace, expected_counts):
    """A dead optimizer must FAIL before a scheduler/known-bug comparison."""
    assert expected_counts and all(isinstance(n, int) and n > 0 for n in expected_counts)
    assert len(trace.after) > 0
    assert len(trace.before) == len(trace.after)
    assert len(trace.after) in expected_counts
    for event in trace.before:
        assert event["gradients"]
        assert all(g is not None and torch.isfinite(g).all() for g in event["gradients"])
