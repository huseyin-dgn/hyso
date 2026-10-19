"""Caller-owned resident eager inference with host or selected-stream event timing."""
from contextlib import ExitStack
import hashlib
import importlib.metadata
import math
import os
import platform
import random
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn

from .result import BenchmarkResult, _plain, _statistics, _throughput

try:
    _VERSION = importlib.metadata.version("hyso")
except importlib.metadata.PackageNotFoundError:
    _VERSION = None

_RUN_LOCK = threading.Lock()


class StateMutationError(RuntimeError):
    """Unsupported forward mutation; no successful benchmark result is available."""


def _type(value):
    return type(value).__module__ + "." + type(value).__qualname__


def _bits(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def _signature(value):
    return (tuple(value.shape), tuple(value.stride()), value.storage_offset(), value.dtype,
            value.device, value.requires_grad, value.untyped_storage()._cdata)


def _same_bits(value, saved):
    return value.shape == saved.shape and value.dtype == saved.dtype and torch.equal(_bits(value), _bits(saved))


def _shape(value):
    if type(value) not in (tuple, list) or len(value) != 4 or any(type(n) is not int or n <= 0 for n in value):
        raise ValueError("input_shape must contain four positive Python integers (N,C,H,W)")
    return tuple(value)


def _tensor(value, dtype, *, parameter=False, input_tensor=False, device=torch.device("cpu")):
    expected = nn.Parameter if parameter else torch.Tensor
    if type(value) is not expected:
        raise TypeError("benchmark requires standard initialized Tensor/Parameter objects")
    if (value.device != device or value.layout != torch.strided or value.is_quantized
            or value.is_conj() or value.is_neg() or (torch.is_inference(value) and not input_tensor)):
        raise ValueError("benchmark requires dense initialized ordinary tensors on selected device")
    if value.is_floating_point():
        if value.dtype != dtype:
            raise ValueError("floating model/input dtype must match precision")
    elif parameter or input_tensor or value.is_complex():
        raise ValueError("parameters and inputs must be floating; integer/bool buffers are supported")
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite model/input tensor")
    if input_tensor and (value.requires_grad or not value.is_contiguous()):
        raise ValueError("example_input must be contiguous and no-grad")


def _profile(model, dtype, device=torch.device("cpu")):
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    from torch.nn.modules import module as module_api
    if any(name.startswith("_global_") and name.endswith("_hooks") and bool(value)
           for name, value in vars(module_api).items() if isinstance(value, dict)):
        raise ValueError("global Module hooks are outside the safe benchmark profile")
    modules, entries, seen = [], [], set()
    def visit(module, path, ancestors):
        if id(module) in ancestors:
            raise ValueError("cyclic module graph is unsupported")
        if id(module) in seen:
            return
        seen.add(id(module))
        if (isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel, torch.jit.ScriptModule))
                or "_orig_mod" in vars(module) or ".quantized." in type(module).__module__):
            raise ValueError("parallel/compiled/scripted/quantized modules are unsupported")
        for method in ("train", "eval", "children", "named_children"):
            if method in vars(module) or getattr(type(module), method) is not getattr(nn.Module, method):
                raise ValueError("custom train/eval/traversal side effects are unsupported")
        if "forward" in vars(module) or any(name.endswith("_hooks") and bool(value)
                                            for name, value in vars(module).items() if isinstance(value, dict)):
            raise ValueError("instance forward overrides and module hooks are unsupported")
        modules.append((path, module))
        for kind, mapping in (("parameter", module._parameters), ("buffer", module._buffers)):
            for name, value in mapping.items():
                if value is None:
                    continue
                _tensor(value, dtype, parameter=kind == "parameter" or type(value) is nn.Parameter, device=device)
                entries.append((kind, path + "." + name if path else name, value))
                if kind == "parameter" and value.grad is not None:
                    if (type(value.grad) is not torch.Tensor or value.grad.layout != torch.strided
                            or value.grad.device != device or torch.is_inference(value.grad)):
                        raise ValueError("existing gradients must be ordinary dense tensors on selected device")
        for name, child in module._modules.items():
            if child is not None:
                visit(child, path + "." + name if path else name, ancestors | {id(module)})
    visit(model, "", set())
    return modules, entries


class _Snapshot:
    def __init__(self, modules, entries, inputs):
        self.modules = [(m, m.training, {name: (getattr(m, name), dict(getattr(m, name)))
                         for name in ("_parameters", "_buffers", "_modules")}, set(m._non_persistent_buffers_set))
                        for _, m in modules]
        unique = {id(t): t for _, _, t in entries}
        self.tensors = [(t, _signature(t), t.detach().cpu().clone(), t._version) for t in unique.values()]
        parameters = {id(t): t for kind, _, t in entries if kind == "parameter"}
        self.gradients = [(p, p.grad, None if p.grad is None else p.grad.detach().cpu().clone(),
                           None if p.grad is None else _signature(p.grad)) for p in parameters.values()]
        self.inputs = inputs
        self.input_signature, self.input_copy, self.input_version = _signature(inputs), inputs.detach().cpu().clone(), inputs._version

    def check(self, *, full=False):
        try:
            self._check(full=full)
        except StateMutationError:
            raise
        except Exception as error:
            raise StateMutationError("forward made registered state uninspectable") from error

    def _check(self, *, full=False):
        for module, _, maps, nonpersistent in self.modules:
            if module.training is not False or module._non_persistent_buffers_set != nonpersistent:
                raise StateMutationError("forward changed module mode/buffer registration")
            for name, (original, saved) in maps.items():
                current = getattr(module, name)
                if (current is not original or list(current) != list(saved)
                        or any(current[key] is not saved[key] for key in saved)):
                    raise StateMutationError("forward changed registered keys/references; structural rollback is best-effort")
        for tensor, signature, saved, version in self.tensors:
            if (_signature(tensor) != signature or tensor._version != version
                    or (full and not _same_bits(tensor, saved))):
                raise StateMutationError("forward mutated registered tensor state/alias structure")
        for parameter, gradient, saved, signature in self.gradients:
            if (parameter.grad is not gradient or (gradient is not None and (
                    _signature(gradient) != signature or not _same_bits(gradient, saved)))):
                raise StateMutationError("forward mutated an existing gradient")
        if (_signature(self.inputs) != self.input_signature or self.inputs._version != self.input_version
                or (full and not _same_bits(self.inputs, self.input_copy))):
            raise StateMutationError("forward mutated resident input")

    def restore_values(self):
        # Structural changes are unsupported. Repair the original registries
        # where possible, then verify; this is not arbitrary Python rollback.
        for module, _, maps, nonpersistent in self.modules:
            for name, (original, saved) in maps.items():
                original.clear()
                original.update(saved)
                vars(module)[name] = original
            module._non_persistent_buffers_set = set(nonpersistent)
        with torch.no_grad():
            for tensor, signature, saved, _ in self.tensors:
                if _signature(tensor) != signature:
                    raise RuntimeError("cannot safely undo tensor storage/dtype/topology mutation")
                if not _same_bits(tensor, saved):
                    tensor.copy_(saved)
            for parameter, gradient, saved, signature in self.gradients:
                parameter.grad = gradient
                if gradient is not None:
                    if _signature(gradient) != signature:
                        raise RuntimeError("cannot safely undo gradient topology mutation")
                    if not _same_bits(gradient, saved):
                        gradient.copy_(saved)

    def restore_modes(self):
        for module, training, _, _ in self.modules:
            module.training = training


def _descriptor(modules, entries):
    objects, storages, tensors, digest = {}, {}, [], hashlib.sha256()
    for kind, name, tensor in entries:
        object_alias = objects.setdefault(id(tensor), len(objects))
        storage_alias = storages.setdefault(tensor.untyped_storage()._cdata, len(storages))
        item = dict(kind=kind, name=name, shape=list(tensor.shape), stride=list(tensor.stride()),
                    dtype=str(tensor.dtype), requires_grad=tensor.requires_grad,
                    object_alias=object_alias, storage_alias=storage_alias, storage_offset=tensor.storage_offset())
        tensors.append(item)
        digest.update(str((kind, name, tuple(tensor.shape), str(tensor.dtype))).encode("utf-8"))
        digest.update(_bits(tensor).numpy().tobytes())
    module_descriptors = []
    def native_plain(value):
        return [native_plain(v) for v in value] if type(value) in (tuple, list) else _plain(value)
    fields = ("input_channels", "num_classes", "widths", "repeats", "strides", "attention_name", "norm_name",
              "in_channels", "out_channels", "in_features", "out_features", "kernel_size", "stride", "padding",
              "dilation", "groups", "padding_mode", "num_features", "affine", "track_running_stats", "eps",
              "momentum", "p", "inplace", "output_size", "num_groups", "num_channels", "channels", "reduction")
    for name, module in modules:
        native = type(module).__module__.startswith(("torch.nn.", "hyso.models."))
        config = {key: native_plain(vars(module)[key]) for key in fields if native and key in vars(module)}
        module_descriptors.append(dict(name=name, qualified_class=_type(module), native_config=config))
    return dict(modules=module_descriptors,
                tensors=tensors), digest.hexdigest()


def _counts(entries):
    parameters = {id(t): t for kind, _, t in entries if kind == "parameter"}
    buffers = {id(t): t for kind, _, t in entries if kind == "buffer"}
    combined = dict(parameters)
    combined.update(buffers)
    total = sum(t.numel() for t in parameters.values())
    trainable = sum(t.numel() for t in parameters.values() if t.requires_grad)
    return dict(total_parameters=total, trainable_parameters=trainable, frozen_parameters=total-trainable,
                parameter_bytes=sum(t.numel()*t.element_size() for t in parameters.values()),
                buffer_bytes=sum(t.numel()*t.element_size() for t in buffers.values()),
                parameter_buffer_bytes=sum(t.numel()*t.element_size() for t in combined.values()),
                byte_accounting="unique_tensor_object_logical",
                cross_category_object_alias=bool(parameters.keys() & buffers.keys()),
                distinct_objects_share_storage=len({t.untyped_storage()._cdata for t in combined.values()}) < len(combined))


def _output(value, batch, expected=None, device=torch.device("cpu")):
    if (type(value) is not torch.Tensor or value.device != device or value.layout != torch.strided
            or not value.is_floating_point() or value.ndim == 0 or value.shape[0] != batch
            or value.numel() == 0 or not torch.isfinite(value.detach().cpu()).all()):
        raise ValueError("forward must return one finite dense floating Tensor on selected device with matching batch")
    descriptor = dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device))
    if expected is not None and descriptor != expected:
        raise ValueError("dynamic output shape/dtype is unsupported")
    return descriptor


class _CudaTiming:
    """Event wiring only; prepare/drain are outside individual sample scopes."""
    def __init__(self, device, stream):
        self.device, self.stream = device, stream
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record(stream)
        self.end.record(stream)
        self.end.synchronize()

    def repeat_begin(self):
        torch.cuda.synchronize(self.device)

    def measure(self, forward):
        self.start.record(self.stream)
        output = forward()
        self.end.record(self.stream)
        self.end.synchronize()
        elapsed = self.start.elapsed_time(self.end)
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("CUDA event elapsed time must be finite and positive")
        return output, elapsed


_MEMORY_FIELDS = (
    "baseline_allocated_bytes", "baseline_reserved_bytes", "previous_peak_allocated_bytes",
    "previous_peak_reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes", "incremental_peak_bytes",
    "current_allocated_after_probe_bytes", "current_reserved_after_probe_bytes",
    "current_allocated_after_output_release_bytes", "current_reserved_after_output_release_bytes")


def _cuda_memory_probe(device, forward, validate):
    torch.cuda.synchronize(device)
    read = lambda name: int(getattr(torch.cuda, name)(device))
    data = dict(status="measured", value=None, reason=None, scope="separate_single_forward",
                counter_reset_performed=True)
    data.update(baseline_allocated_bytes=read("memory_allocated"), baseline_reserved_bytes=read("memory_reserved"),
                previous_peak_allocated_bytes=read("max_memory_allocated"),
                previous_peak_reserved_bytes=read("max_memory_reserved"))
    torch.cuda.reset_peak_memory_stats(device)
    try:
        output = forward()
        torch.cuda.synchronize(device)
        data.update(peak_allocated_bytes=read("max_memory_allocated"), peak_reserved_bytes=read("max_memory_reserved"),
                    current_allocated_after_probe_bytes=read("memory_allocated"),
                    current_reserved_after_probe_bytes=read("memory_reserved"))
        validate(output)
        del output
        torch.cuda.synchronize(device)
        data.update(current_allocated_after_output_release_bytes=read("memory_allocated"),
                    current_reserved_after_output_release_bytes=read("memory_reserved"))
    except BaseException as error:
        if isinstance(error, StateMutationError):
            raise StateMutationError(str(error) + "; CUDA memory probe reset process/device peak counters irreversibly") from error
        raise RuntimeError("CUDA memory probe failed; process/device peak counters were reset and cannot be restored") from error
    data["incremental_peak_bytes"] = data["peak_allocated_bytes"] - data["baseline_allocated_bytes"]
    data["value"] = data["peak_allocated_bytes"]
    return data


class BenchmarkRunner:
    """Measure resident eager inference in eval/inference_mode.

    CPU uses host timing; CUDA uses events on the selected current stream.
    Modes, registered values, gradients and CPU/selected CUDA RNG are guarded.
    Hidden cross-stream/device work and concurrent use are unsupported. Opt-in
    memory probing irreversibly resets process/device allocator peak counters.
    No model movement, training, file export or comparison occurs.
    """
    def __init__(self, model, *, device="cpu", input_shape=None, example_input=None,
                 precision="fp32", warmup_iterations=10, measure_iterations=50, repeats=3,
                 seed=0, model_name=None, model_config=None, measure_cuda_memory=False):
        if type(measure_cuda_memory) is not bool:
            raise TypeError("measure_cuda_memory must be bool")
        if not isinstance(device, (str, torch.device)):
            raise TypeError("device must be an explicit string or torch.device")
        resolved = torch.device(device)
        if resolved.type not in ("cpu", "cuda"):
            raise ValueError("only CPU and CUDA are supported")
        if resolved.type == "cpu" and resolved.index is not None:
            raise ValueError("use unindexed CPU device")
        if resolved.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable; no CPU fallback")
            index = torch.cuda.current_device() if resolved.index is None else resolved.index
            if not 0 <= index < torch.cuda.device_count():
                raise ValueError("CUDA device index is unavailable")
            resolved = torch.device("cuda", index)
        if measure_cuda_memory and resolved.type == "cpu":
            raise ValueError("CUDA memory measurement is unavailable for CPU")
        if type(precision) is not str or precision not in (("fp32", "fp64") if resolved.type == "cpu" else ("fp32", "amp_fp16")):
            raise ValueError("precision must be CPU fp32/fp64 or CUDA fp32/amp_fp16")
        for key, value, minimum in (("warmup_iterations", warmup_iterations, 0),
                                    ("measure_iterations", measure_iterations, 1), ("repeats", repeats, 1)):
            if type(value) is not int or value < minimum:
                raise ValueError(key + " must be a Python integer >= " + str(minimum))
        if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
            raise ValueError("seed must be a nonnegative Python int <= 2**63-1")
        if (input_shape is None) == (example_input is None):
            raise ValueError("provide exactly one of input_shape or example_input")
        if model_name is not None and type(model_name) is not str:
            raise TypeError("model_name must be a string or None")
        if model_config is not None and type(model_config) is not dict:
            raise TypeError("model_config must be a plain dictionary or None")
        self.model = model
        self._device, self._memory = resolved, measure_cuda_memory
        self._dtype = torch.float64 if precision == "fp64" else torch.float32
        self._example = example_input
        self._shape = _shape(input_shape) if input_shape is not None else None
        self._config = dict(precision=precision, warmup_iterations=warmup_iterations,
                            measure_iterations=measure_iterations, repeats=repeats, seed=seed,
                            preflight_forward_count=1, memory_probe_count=int(measure_cuda_memory), forced_model_mode="eval",
                            inference_mode=True, autocast_enabled=precision == "amp_fp16", protocol_version=1,
                            input_policy="fixed_resident_input")
        self._name, self._model_config = model_name, _plain(model_config)
        self._validate()

    def _validate(self):
        modules, entries = _profile(self.model, self._dtype, self._device)
        if self._example is not None:
            _tensor(self._example, self._dtype, input_tensor=True, device=self._device)
            _shape(tuple(self._example.shape))
        return modules, entries

    def run(self):
        if not _RUN_LOCK.acquire(blocking=False):
            raise RuntimeError("concurrent/reentrant benchmark runs are unsupported")
        rng = None
        snapshot = None
        contexts = ExitStack()
        cuda_rng = None
        cuda = self._device.type == "cuda"
        memory = None
        try:
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            if cuda:
                contexts.enter_context(torch.cuda.device(self._device))
                stream = torch.cuda.current_stream(self._device)
                contexts.enter_context(torch.cuda.stream(stream))
                cuda_rng = torch.cuda.get_rng_state(self._device)
                if self._memory and torch.cuda.get_allocator_backend() != "native":
                    raise ValueError("memory probe requires the native CUDA allocator")
            def context_check():
                if cuda and (torch.cuda.current_device() != self._device.index
                             or torch.cuda.current_stream(self._device) != stream):
                    raise StateMutationError("forward changed selected CUDA device/current stream")
            def forward():
                context_check()
                return self.model(inputs)
            def validate(output):
                context_check()
                return _output(output, inputs.shape[0], expected_output, self._device)
            modules, entries = self._validate()
            # Own input must have a version counter even when the caller is
            # already in inference_mode or supplied an inference Tensor.
            with torch.inference_mode(False), torch.no_grad():
                if self._example is not None:
                    inputs = self._example.detach().clone()
                else:
                    generator = torch.Generator(device="cpu").manual_seed(self._config["seed"])
                    inputs = torch.randn(self._shape, device="cpu", dtype=self._dtype, generator=generator).to(self._device)
            _tensor(inputs, self._dtype, input_tensor=True, device=self._device)
            snapshot = _Snapshot(modules, entries, inputs)
            descriptor, weights_hash = _descriptor(modules, entries)
            counts = _counts(entries)
            input_metadata = dict(shape=list(inputs.shape), stride=list(inputs.stride()), dtype=str(inputs.dtype),
                                  layout="torch.strided", batch_size=inputs.shape[0], device=str(self._device),
                                  source="custom_clone" if self._example is not None else "generated",
                                  seed=None if self._example is not None else self._config["seed"],
                                  content_sha256=hashlib.sha256(_bits(inputs).numpy().tobytes()).hexdigest())
            started = datetime.now(timezone.utc).isoformat()
            samples = []
            try:
                self.model.eval()
                with torch.inference_mode(), torch.autocast(device_type=self._device.type, dtype=torch.float16,
                                                           enabled=self._config["autocast_enabled"]):
                    output = forward()
                    context_check()
                    expected_output = _output(output, inputs.shape[0], device=self._device)
                    del output
                    snapshot.check(full=True)
                    timer = None
                    for _ in range(self._config["repeats"]):
                        for _ in range(self._config["warmup_iterations"]):
                            output = forward()
                            validate(output)
                            del output
                            snapshot.check()
                        repeat = []
                        if cuda:
                            if timer is None:
                                timer = _CudaTiming(self._device, stream)
                            timer.repeat_begin()
                        for _ in range(self._config["measure_iterations"]):
                            if cuda:
                                context_check()
                                output, latency = timer.measure(lambda: self.model(inputs))
                            else:
                                start = time.perf_counter_ns()
                                output = self.model(inputs)
                                end = time.perf_counter_ns()
                                if type(start) is not int or type(end) is not int or end <= start:
                                    raise FloatingPointError("timer readings must give finite positive latency")
                                latency = (end - start) / 1_000_000
                            if not math.isfinite(latency) or latency <= 0:
                                raise FloatingPointError("timer readings must give finite positive latency")
                            repeat.append(latency)
                            validate(output)
                            del output
                            snapshot.check()
                        samples.append(repeat)
                        snapshot.check(full=True)
                    if self._memory:
                        memory = _cuda_memory_probe(self._device, forward, validate)
                snapshot.check(full=True)
            except BaseException as error:
                if cuda:
                    torch.cuda.synchronize(self._device)
                try:
                    snapshot.check(full=True)
                except StateMutationError as mutation:
                    try:
                        snapshot.restore_values()
                    except BaseException as rollback_error:
                        raise RuntimeError("benchmark state rollback failed; caller runtime may be mutated") from rollback_error
                    raise StateMutationError(str(mutation) + "; best-effort rollback applied, arbitrary version/external state is not transactional") from error
                raise
            finally:
                snapshot.restore_modes()
            return self._result(input_metadata, descriptor, weights_hash, counts, samples, expected_output, started, memory)
        finally:
            try:
                if snapshot is not None:
                    snapshot.restore_modes()
            finally:
                try:
                    if rng is not None:
                        random.setstate(rng[0])
                        np.random.set_state(rng[1])
                        torch.set_rng_state(rng[2])
                finally:
                    try:
                        if cuda_rng is not None:
                            torch.cuda.set_rng_state(cuda_rng, self._device)
                    finally:
                        try:
                            contexts.close()
                        finally:
                            _RUN_LOCK.release()

    def _result(self, inputs, descriptor, weights_hash, counts, samples, output, started, memory=None):
        flat = [value for repeat in samples for value in repeat]
        missing = lambda reason, status="unsupported": dict(status=status, value=None, reason=reason)
        warnings = []
        cuda = self._device.type == "cuda"
        method = "cuda_event_elapsed" if cuda else "perf_counter_ns"
        cuda_metadata = missing("cpu_only_b1", "not_applicable")
        if cuda:
            properties = torch.cuda.get_device_properties(self._device)
            cuda_metadata = dict(status="available", device_index=self._device.index, name=properties.name,
                                 capability=list(torch.cuda.get_device_capability(self._device)),
                                 total_device_memory_bytes=properties.total_memory, cuda_build=torch.version.cuda,
                                 cudnn_version=torch.backends.cudnn.version(), allocator_backend=torch.cuda.get_allocator_backend(),
                                 stream_id=int(torch.cuda.current_stream(self._device).cuda_stream),
                                 cudnn_benchmark=torch.backends.cudnn.benchmark,
                                 cudnn_deterministic=torch.backends.cudnn.deterministic,
                                 cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                 matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
            if memory is None:
                memory = dict(status="disabled", value=None, reason="measure_cuda_memory_false", scope="separate_single_forward",
                              counter_reset_performed=False, **dict.fromkeys(_MEMORY_FIELDS))
            else:
                warnings.append(dict(code="cuda_peak_reset", counter_reset_performed=True,
                                     previous_peak_allocated_bytes=memory["previous_peak_allocated_bytes"],
                                     previous_peak_reserved_bytes=memory["previous_peak_reserved_bytes"],
                                     scope="separate_single_forward",
                                     message="Process/device peak counters were irreversibly reset; caller/process exclusivity is required. Allocator counters exclude other processes and non-allocator VRAM."))
        if self._config["warmup_iterations"] == 0:
            warnings.append(dict(code="no_warmup", message="Cold/steady-state effects may be mixed."))
        if counts["distinct_objects_share_storage"] or counts["cross_category_object_alias"]:
            warnings.append(dict(code="tensor_aliasing", message="Logical bytes are not physical storage bytes."))
        if _VERSION is None:
            warnings.append(dict(code="hyso_version_unknown", message="No installed distribution metadata."))
        return BenchmarkResult(dict(
            schema_version=1, benchmark_type="inference",
            identity=dict(run_id=str(uuid.uuid4()), started_at_utc=started,
                          finished_at_utc=datetime.now(timezone.utc).isoformat()),
            model=dict(model_name=self._name or type(self.model).__name__, qualified_class=_type(self.model),
                       model_config=self._model_config, descriptor_version=1, structural_descriptor=descriptor,
                       weights_sha256=weights_hash, source_sha256=None, source_hash_status="not_collected",
                       source_hash_reason="B1_run_does_not_read_source_files", output=output),
            input=inputs, configuration=self._config,
            environment=dict(python_version=platform.python_version(), torch_version=str(torch.__version__),
                             hyso_version=_VERSION, torchvision_version=None,
                             torchvision_status="not_collected", operating_system=platform.system(),
                             platform=platform.platform(), cpu=dict(machine=platform.machine(), processor=platform.processor(),
                                                                  logical_cpu_count=os.cpu_count(), threads=torch.get_num_threads(),
                                                                  interop_threads=torch.get_num_interop_threads()),
                             device=str(self._device), cuda=cuda_metadata,
                             backend=dict(mkldnn_enabled=torch.backends.mkldnn.enabled,
                                          deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                                          deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                                          float32_matmul_precision=torch.get_float32_matmul_precision())),
            provenance=dict(module_source=getattr(sys.modules.get(type(self.model).__module__), "__file__", None),
                            benchmark_source=__file__, source_revision=None, revision_status="not_collected"),
            timing=dict(timing_method=method, timing_scope="serial_current_stream_forward" if cuda else "synchronous_host_forward",
                        synchronization_policy="repeat_device_drain_and_end_event_sync" if cuda else "none",
                        clock_resolution_ns=None if cuda else time.get_clock_info("perf_counter").resolution*1e9,
                        latency_samples_ms=samples),
            stats=dict(per_repeat=[_statistics(row) for row in samples], pooled=_statistics(flat),
                       quantile_method="linear_(n-1)*q", std_ddof=1),
            throughput=dict(images_per_sec=_throughput(flat, inputs["batch_size"]), method="derived_serial_service_rate",
                            source_timing_method=method, per_repeat_images_per_sec=[_throughput(row, inputs["batch_size"]) for row in samples],
                            window_wall_images_per_sec=None, window_status="unsupported", window_reason="separate_protocol_deferred"),
            parameters=counts, cuda_memory=memory if cuda else missing("cpu_only_b1", "not_applicable"),
            optional_metrics={name: missing(reason) for name, reason in dict(
                accuracy="separate_evaluation_protocol", flops="no_counting_method", macs="no_counting_method",
                disk_artifact_bytes="no_artifact_written", cpu_rss_bytes="no_validated_portable_backend",
                optimizer_memory_bytes="inference_only", activation_memory_bytes="not_a_logical_model_size").items()},
            diagnostics=dict(warnings=warnings, state_guard="registered_state_CPU_and_selected_CUDA_RNG" if cuda else "registered_state_and_CPU_RNG",
                             limitations=(["Fixed resident input, eager single-Tensor inference only.",
                                          "CUDA events measure selected-stream elapsed time including host launch gaps, not host wall time; each sample synchronizes its end event.",
                                          "Hidden cross-stream/other-device work is unsupported and cannot be reliably detected. Warmup/autotune affect results.",
                                          "AMP does not make every operation FP16. GPU clocks, thermals and background load affect latency.",
                                          "OOM/arbitrary external state cannot be transactionally restored; allocator peaks cannot be restored after reset."]
                                          if cuda else ["Fixed resident input, CPU eager single-Tensor inference only."]) + [
                                          "Derived service rate is not sustained throughput or camera FPS.",
                                          "Logical model bytes exclude optimizer, activations and process RSS.",
                                          "Unregistered Python state/external I/O and concurrent use are unsupported.",
                                          "Latency varies with threads, clocks, OS and background load.",
                                          "Validation/guard work runs outside timed forwards and may affect cache conditions."])))
