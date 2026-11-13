"""CPU-only P13 NVTX scope contract tests: no torch/CUDA/worker execution."""

import pytest
from tools.full_acceptance.p13_capture_scopes import ModuleNvtxScopes


class Recorder:
    def __init__(self):
        self.events = []
        self.stack = []

    def range_push(self, label):
        self.events.append(("push", label))
        self.stack.append(label)

    def range_pop(self):
        self.events.append(("pop", self.stack.pop()))


class Handle:
    def __init__(self, hooks, hook):
        self.hooks, self.hook = hooks, hook

    def remove(self):
        self.hooks.remove(self.hook)


class Module:
    def __init__(self, name, children=()):
        self.name = name
        self.children = children
        self.pres = []
        self.posts = []
        self.parameter = object()

    def named_modules(self, *, remove_duplicate=False):
        yield "", self
        for name, child in self.children:
            yield name, child

    def register_forward_pre_hook(self, hook):
        self.pres.append(hook)
        return Handle(self.pres, hook)

    def register_forward_hook(self, hook, *, always_call=False):
        assert always_call
        self.posts.append(hook)
        return Handle(self.posts, hook)

    def __call__(self, *, fail=False):
        for h in self.pres:
            h(self, ())
        try:
            for _, child in self.children:
                child()
            if fail:
                raise RuntimeError("observed model failed")
            return 7
        finally:
            for h in self.posts:
                h(self, (), None)


def test_nested_real_invocation_order_cleanup_and_no_model_mutation():
    leaf = Module("leaf")
    model = Module("root", [("layer.0", leaf)])
    before = (model.parameter, leaf.parameter)
    nvtx = Recorder()
    with ModuleNvtxScopes(model, nvtx) as scopes:
        assert scopes.module_paths == ("<root>", "layer.0")
        assert scopes.measured_forward(0, model, lambda: None) == 7
        with pytest.raises(ValueError, match="duplicate"):
            scopes.measured_forward(0, model, lambda: None)
        assert scopes.measured_forward(1, model, lambda: None) == 7
    assert (model.parameter, leaf.parameter) == before
    assert not model.pres and not model.posts and not leaf.pres and not leaf.posts
    assert not nvtx.stack
    labels = [label for op, label in nvtx.events if op == "push"]
    assert labels == ["hyso.p13/forward/0", "hyso.p13.module/<root>",
                      "hyso.p13.module/layer.0", "hyso.p13/forward/1",
                      "hyso.p13.module/<root>", "hyso.p13.module/layer.0"]


def test_failed_forward_closes_ranges_and_removes_hooks():
    model = Module("model")
    nvtx = Recorder()
    with pytest.raises(RuntimeError, match="observed model failed"):
        with ModuleNvtxScopes(model, nvtx) as scopes:
            scopes.measured_forward(0, lambda: model(fail=True), lambda: None)
    assert not nvtx.stack and not model.pres and not model.posts


def test_alias_rejected_before_hook_install():
    leaf = Module("shared")
    model = Module("model", [("a", leaf), ("b", leaf)])
    with pytest.raises(ValueError, match="alias"):
        ModuleNvtxScopes(model, Recorder()).__enter__()
    assert not model.pres and not leaf.pres


def test_missing_synchronize_does_not_leave_open_forward_range():
    model, nvtx = Module("model"), Recorder()
    with ModuleNvtxScopes(model, nvtx) as scopes:
        def fail():
            raise RuntimeError("sync failed")
        with pytest.raises(RuntimeError, match="sync"):
            scopes.measured_forward(0, model, fail)
    assert not nvtx.stack


def test_invalid_forward_and_budget_fail_closed():
    model = Module("model")
    with pytest.raises(ValueError, match="budget"):
        ModuleNvtxScopes(model, Recorder(), max_modules=0).__enter__()
    with ModuleNvtxScopes(model, Recorder()) as scopes:
        with pytest.raises(ValueError, match="forward"):
            scopes.measured_forward(-1, model, lambda: None)
