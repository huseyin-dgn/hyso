"""P13 native *instrumentation* primitive. No CUDA launch, capture or attestation.

Apply only within a separately authorized, source-pinned, owned Windows Job.
The worker must bind model/checkpoint/input/output and verify GPU completion;
Nsight capture/export and supervisor receipts are intentionally NOT implemented
here. This file cannot by itself satisfy P13 acceptance.
"""

from contextlib import contextmanager


class ModuleNvtxScopes:
    """Temporarily instrument real PyTorch forward calls with lexical NVTX scopes.

    Does not modify model source code, module parameters, tensors or checkpoint.
    PyTorch hooks are transient observer-only modifications; removed on exit.
    Shared module objects with multiple paths cannot be assigned a unique path,
    so fail closed rather than invent model/module ownership.
    """

    PREFIX = "hyso.p13.module/"

    def __init__(self, model, nvtx, *, max_modules=4096):
        self.model = model
        self.nvtx = nvtx
        self.max_modules = max_modules
        self._handles = []
        self._entered = False
        self._used_forwards = set()
        self._paths = None

    def _catalog(self):
        if type(self.max_modules) is not int or not 1 <= self.max_modules <= 4096:
            raise ValueError("invalid P13 module budget")
        modules = list(self.model.named_modules(remove_duplicate=False))
        if not 1 <= len(modules) <= self.max_modules:
            raise ValueError("P13 module budget exceeded or empty model")
        seen_names = set()
        seen_objects = set()
        for name, module in modules:
            if not isinstance(name, str) or len(name) > 256 or any(c in name for c in "\r\n/"):
                raise ValueError("invalid P13 module path")
            if name in seen_names or id(module) in seen_objects:
                raise ValueError("ambiguous P13 module alias or duplicate name")
            seen_names.add(name)
            seen_objects.add(id(module))
        return modules

    def __enter__(self):
        if self._entered:
            raise RuntimeError("P13 module hooks already active")
        modules = self._catalog()
        try:
            for name, module in modules:
                label = self.PREFIX + (name or "<root>")

                def pre(_mod, _args, *, _label=label):
                    self.nvtx.range_push(_label)

                def post(_mod, _args, _output):
                    self.nvtx.range_pop()

                self._handles.append(module.register_forward_pre_hook(pre))
                # PyTorch >=2.0: always_call ensures correct stack cleanup if
                # an observed forward raises. Older implementations fail closed.
                self._handles.append(module.register_forward_hook(post, always_call=True))
            self._entered = True
            self._paths = [name or "<root>" for name, _ in modules]
            self._used_forwards.clear()
            return self
        except BaseException:
            self._remove_all()
            raise

    def _remove_all(self):
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._entered = False

    def __exit__(self, exc_type, exc, tb):
        self._remove_all()
        return False

    @property
    def module_paths(self):
        if not self._entered:
            raise RuntimeError("P13 module scopes not active")
        return tuple(self._paths)

    def measured_forward(self, index, work, synchronize):
        """Execute exactly one synchronized forward under its named NVTX range.

        ``work`` is supplied by the authorized native worker, not arbitrary
        model/plugin source. No model outputs or equivalence claims are written.
        """
        if not self._entered or type(index) is not int or index < 0 or index >= 100:
            raise ValueError("invalid or inactive P13 forward")
        if index in self._used_forwards:
            raise ValueError("duplicate P13 measured forward")
        self._used_forwards.add(index)
        self.nvtx.range_push(f"hyso.p13/forward/{index}")
        try:
            result = work()
            synchronize()
            return result
        finally:
            self.nvtx.range_pop()
