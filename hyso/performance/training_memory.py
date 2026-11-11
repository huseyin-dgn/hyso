"""P23 bounded, in-process training observations. No peak reset or GC control.

Logical tensors, native process RSS and CUDA allocator counters are independent
domains. Activation lifetimes and isolated window peaks remain unsupported.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import sys
import threading
import time

import torch

from hyso.benchmark.runner import _RUN_LOCK
from hyso.benchmark.result import _plain, _csv_text, _write_export
from .memory import _observe, _BackendUnavailable, _memory_metrics
from .training import (_Recorder, TrainingTimingProtocol, TrainingPerformanceResult,
                       _measure_training, _int)
from .training_details import _source


class TrainingMemoryError(RuntimeError):
    """Incomplete observation; training may have advanced. Never retry implicitly."""


@dataclass(frozen=True)
class TrainingMemoryProtocol:
    device: str = "cpu"
    scope: str = "in_process_fit"
    peak_reset_policy: str = "never"
    interval_seconds: float = .01
    max_samples: int = 10000
    max_boundaries: int = 10000
    max_tensor_records: int = 100000

    def __post_init__(self):
        if self.scope != "in_process_fit" or self.peak_reset_policy != "never":
            raise ValueError("only in_process_fit / never-reset is supported; isolated probes are unavailable")
        if type(self.device) is not str or (self.device != "cpu" and
                not (self.device.startswith("cuda:") and self.device[5:].isdigit())):
            raise ValueError("explicit cpu or indexed cuda device required")
        if type(self.interval_seconds) not in (float, int) or not math.isfinite(self.interval_seconds) or self.interval_seconds < 1e-6:
            raise ValueError("interval_seconds must be finite and >= 1 microsecond")
        for name, lower, upper in (("max_samples", 2, 100000), ("max_boundaries", 2, 100000),
                                   ("max_tensor_records", 1, 1000000)):
            _int(getattr(self, name), lower)
            if getattr(self, name) > upper:
                raise ValueError(name + " exceeds hard ceiling")


def _unsupported(reason):
    return dict(value=None, status="unsupported", reason=reason)


def _logical(model, optimizer, budget):
    """One metadata snapshot. Identity/storage group IDs are snapshot-local.

    Count each distinct tensor object once *within each component*. Views remain
    distinct logical tensors; full storage bytes are separately deduplicated.
    No state_dict, tensor copies, value reads, or lazy state initialization.
    """
    rows, identities, storages, metadata = [], {}, {}, []
    visits = 0

    def add(component, name, tensor):
        nonlocal visits
        visits += 1
        if visits > budget:
            raise TrainingMemoryError("logical traversal budget exhausted")
        key = (component, id(tensor))
        if key in identities:
            identities[key]["names"].append(name)
            return
        if len(rows) >= budget:
            raise TrainingMemoryError("logical tensor record budget exhausted")
        row = dict(component=component, names=[name], tensor_id=None, storage_id=None,
                   dtype=str(tensor.dtype), device=str(tensor.device), shape=list(tensor.shape),
                   elements=None, element_size_bytes=None, logical_bytes=None, storage_bytes=None,
                   storage_offset=None, stride=None, frozen=component == "parameter" and not tensor.requires_grad,
                   status="unsupported", reason="requires ordinary dense CPU/CUDA strided tensor")
        identities[key] = row
        rows.append(row)
        if (type(tensor) not in (torch.Tensor, torch.nn.Parameter) or tensor.layout != torch.strided
                or tensor.is_quantized or tensor.device.type not in ("cpu", "cuda")
                or tensor.is_conj() or tensor.is_neg()):
            return
        storage = tensor.untyped_storage()
        storage_key = (str(tensor.device), storage._cdata)
        storage_id = storages.setdefault(storage_key, len(storages) + 1)
        # Cross-component identity evidence is independent of storage aliasing.
        # Explicit map avoids retaining any tensors beyond this function.
        tensor_id = tensor_ids.setdefault(id(tensor), len(tensor_ids) + 1)
        row.update(tensor_id=tensor_id, storage_id=storage_id, elements=tensor.numel(),
                   element_size_bytes=tensor.element_size(), logical_bytes=tensor.numel()*tensor.element_size(),
                   storage_bytes=storage.nbytes(), storage_offset=tensor.storage_offset(),
                   stride=list(tensor.stride()), status="measured", reason=None)

    tensor_ids = {}
    stack = [("", model, frozenset())]
    module_visits = 0
    while stack:
        module_visits += 1
        if module_visits > budget:
            raise TrainingMemoryError("module traversal budget exhausted")
        path, module, ancestors = stack.pop()
        if id(module) in ancestors:
            raise TrainingMemoryError("cyclic module graph unsupported")
        for component, mapping in (("parameter", module._parameters), ("buffer", module._buffers)):
            for name, tensor in mapping.items():
                if tensor is None:
                    continue
                label = path + "." + name if path else name
                add(component, label, tensor)
                if component == "parameter" and tensor.grad is not None:
                    add("gradient", label + ".grad", tensor.grad)
        for name, child in reversed(list(module._modules.items())):
            if child is not None:
                stack.append((path + "." + name if path else name, child, ancestors | {id(module)}))

    supported = type(optimizer) in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW)

    state_stack = ([(value, "state." + str(index), frozenset())
                    for index, (_, value) in reversed(list(enumerate(optimizer.state.items())))]
                   if supported else [])
    state_visits = 0
    while state_stack:
        value, path, ancestors = state_stack.pop()
        state_visits += 1
        if state_visits > budget:
            raise TrainingMemoryError("optimizer state traversal budget exhausted")
        if isinstance(value, torch.Tensor):
            add("optimizer", path, value)
        elif type(value) in (dict, list, tuple):
            if id(value) in ancestors:
                raise TrainingMemoryError("cyclic optimizer state unsupported")
            items = list(value.items()) if type(value) is dict else list(enumerate(value))
            for key, item in reversed(items):
                if type(key) not in (str, int):
                    raise TrainingMemoryError("unsupported optimizer metadata key")
                state_stack.append((item, path + "." + str(key), ancestors | {id(value)}))
        else:
            if len(metadata) + len(rows) >= budget:
                raise TrainingMemoryError("optimizer metadata record budget exhausted")
            valid = value is None or type(value) in (str, bool, int, float)
            if type(value) is float and not math.isfinite(value):
                valid = False
            metadata.append(dict(path=path, value=value if valid else None,
                                 status="observed" if valid else "unsupported",
                                 reason=None if valid else "nonstandard non-tensor state",
                                 python_heap_bytes=None))

    summary = _logical_summary(rows)
    if not supported:
        summary["optimizer"].update(logical_bytes=None, elements=None, status="unsupported",
                                     reason="custom optimizer state traversal is unvalidated")
    return dict(tensors=rows, summary=summary, optimizer_metadata=metadata,
                optimizer_supported=supported,
                convention="distinct_tensor_object_per_component; views_count_logically; storage_ids_snapshot_local",
                unique_storage_bytes=sum({r["storage_id"]: r["storage_bytes"] for r in rows
                                          if r["storage_id"] is not None}.values()))


def _logical_summary(rows):
    result = {}
    for component in ("parameter", "frozen_parameter", "buffer", "gradient", "optimizer"):
        selected = [r for r in rows if (r["component"] == "parameter" and r["frozen"]
                                       if component == "frozen_parameter" else r["component"] == component)]
        valid = all(r["status"] == "measured" for r in selected)
        result[component] = dict(tensor_count=len(selected), elements=sum(r["elements"] for r in selected) if valid else None,
            logical_bytes=sum(r["logical_bytes"] for r in selected) if valid else None,
            status="measured" if valid else "unsupported", reason=None if valid else "unsupported tensor representation")
    return result


def _validate_logical_storage(logical):
    """Validate alias extents as well as sums, including isolated saved results."""
    storages, identities = {}, set()
    for row in logical["tensors"]:
        if row["status"] != "measured":
            if row["status"] != "unsupported" or row["logical_bytes"] is not None or not row["reason"]:
                raise ValueError("unsupported logical tensor needs null + reason")
            continue
        if row["component"] not in ("parameter", "buffer", "gradient", "optimizer"):
            raise ValueError("invalid logical component")
        if row["reason"] is not None or type(row["frozen"]) is not bool:
            raise ValueError("invalid measured logical tensor")
        for key in ("elements", "element_size_bytes", "logical_bytes", "storage_bytes", "storage_offset"):
            _int(row[key])
        for key in ("tensor_id", "storage_id"):
            _int(row[key], 1)
        for value in (*row["shape"], *row["stride"]):
            _int(value)
        if (not row["element_size_bytes"] or len(row["shape"]) != len(row["stride"])
                or math.prod(row["shape"]) != row["elements"]
                or row["logical_bytes"] != row["elements"] * row["element_size_bytes"]):
            raise ValueError("inconsistent logical shape/bytes")
        extent = row["storage_offset"] + sum((d - 1)*s for d, s in zip(row["shape"], row["stride"])) + 1
        if row["elements"] and extent * row["element_size_bytes"] > row["storage_bytes"]:
            raise ValueError("logical tensor exceeds storage extent")
        identity = (row["component"], row["tensor_id"])
        if identity in identities:
            raise ValueError("duplicate logical tensor within component")
        identities.add(identity)
        storage = (row["device"], row["storage_bytes"])
        if storages.setdefault(row["storage_id"], storage) != storage:
            raise ValueError("inconsistent logical storage alias")
    expected = sum(size for _, size in storages.values())
    if type(logical["unique_storage_bytes"]) is not int or logical["unique_storage_bytes"] != expected:
        raise ValueError("inconsistent unique logical storage bytes")


def _validate_cuda_boundary(cuda, device):
    if device == "cpu":
        if cuda is not None:
            raise ValueError("CPU is not CUDA allocator evidence")
        return
    if cuda is None or cuda.get("allocator_backend") != "native":
        raise ValueError("missing/unvalidated CUDA allocator evidence")
    for key in ("allocated_bytes", "reserved_bytes", "lifetime_peak_allocated_bytes", "lifetime_peak_reserved_bytes"):
        _int(cuda[key])
    if (cuda["allocated_bytes"] > cuda["reserved_bytes"]
            or cuda["allocated_bytes"] > cuda["lifetime_peak_allocated_bytes"]
            or cuda["reserved_bytes"] > cuda["lifetime_peak_reserved_bytes"]):
        raise ValueError("inconsistent CUDA counters")


def _cuda(device):
    if device.type != "cuda":
        return None
    backend = torch.cuda.get_allocator_backend()
    if backend != "native":
        raise TrainingMemoryError("non-native CUDA allocator accounting is unvalidated: " + backend)
    stats = torch.cuda.memory_stats(device)
    return dict(allocated_bytes=torch.cuda.memory_allocated(device), reserved_bytes=torch.cuda.memory_reserved(device),
                lifetime_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                lifetime_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                allocation_retries=stats.get("num_alloc_retries"), oom_count=stats.get("num_ooms"),
                allocator_backend=backend, device_name=torch.cuda.get_device_name(device),
                device_capability=list(torch.cuda.get_device_capability(device)))


def _rss_summary(samples, reason):
    return _memory_metrics(samples if reason is None else [], reason)


class _MemoryRecorder(_Recorder):
    def __init__(self, trainer, timing_protocol, protocol):
        super().__init__(trainer, timing_protocol)
        self.memory_protocol = protocol
        self.boundaries, self.rss, self.errors = [], [], []
        self.tensor_records = 0
        self.stop = threading.Event()
        self.thread = None
        self.rss_reason = None
        self.closed = False
        try:
            self.capture("ready")
            self.sample()
            if self.rss_reason is None:
                self.thread = threading.Thread(target=self.sampling, name="hyso-training-rss", daemon=True)
                self.thread.start()
        except BaseException:
            self._stop_sampler()
            self.trainer = None
            raise

    def _stop_sampler(self):
        self.stop.set()
        if self.thread is not None and self.thread.ident is not None:
            self.thread.join(timeout=5.)
            if self.thread.is_alive():
                # A blocked native query cannot be cancelled safely in-process.
                # Retain a failure, release trainer references, never publish a
                # successful measurement. Isolated callers have the outer Job.
                self.errors.append("RSS_sampler_shutdown_timeout; native query not cancelled")

    def sample(self):
        try:
            sample = _observe()
        except _BackendUnavailable as error:
            self.rss_reason = str(error)
            return
        except ValueError as error:
            if str(error) != "inconsistent native memory observation/clock":
                raise
            self.rss_reason = str(error)
            return
        if self.rss and sample["lifetime_hwm_bytes"] < self.rss[-1]["lifetime_hwm_bytes"]:
            self.rss_reason = "native_RSS_lifetime_high_water_mark_decreased"
            return
        self.rss.append(sample)

    def sampling(self):
        try:
            while not self.stop.wait(self.memory_protocol.interval_seconds):
                if len(self.rss) >= self.memory_protocol.max_samples - 1:
                    raise TrainingMemoryError("RSS sample budget exhausted")
                self.sample()
                if self.rss_reason is not None:
                    return
        except Exception as error:
            # Store text only, never a traceback/exception retaining trainer tensors.
            self.errors.append(type(error).__name__ + ": " + str(error))

    def capture(self, boundary, **fields):
        if len(self.boundaries) >= self.memory_protocol.max_boundaries:
            raise TrainingMemoryError("memory boundary budget exhausted")
        start = time.perf_counter_ns()
        logical = _logical(self.trainer.model, self.trainer.optimizer,
                           self.memory_protocol.max_tensor_records - self.tensor_records)
        self.tensor_records += len(logical["tensors"]) + len(logical["optimizer_metadata"])
        if self.tensor_records > self.memory_protocol.max_tensor_records:
            raise TrainingMemoryError("logical record budget exhausted")
        context = dict(epoch_id=None if self.epoch is None else self.epoch["epoch_id"],
                       microbatch_id=None, window_id=None, attempt_id=None)
        for kind in ("window", "microbatch", "attempt"):
            if kind in self.active:
                context.update({k: v for k, v in self.active[kind][0].items() if k in context})
        self._check_stream()
        cuda = _cuda(self.trainer.device)
        self.boundaries.append(dict(boundary=boundary, **context, fields=fields, logical=logical,
                                    cuda=cuda, scope="registered_tensor_state_and_allocator_boundary",
                                    status="measured", reason=None,
                                    query_start_ns=start, query_end_ns=time.perf_counter_ns()))

    def epoch_begin(self, epoch_id):
        super().epoch_begin(epoch_id)
        self.capture("epoch_ready")

    def microbatch_begin(self, microbatch_id, new_window):
        super().microbatch_begin(microbatch_id, new_window)
        self.capture("pre_microbatch")

    def attempt_begin(self, attempt_id, updates):
        super().attempt_begin(attempt_id, updates)
        self.capture("pre_attempt")

    def attempt_end(self, updates, samples):
        before = self.active["attempt"][0]["update_before"]
        self.capture("post_attempt", applied=updates == before + 1, skipped=updates == before, samples=samples)
        super().attempt_end(updates, samples)

    def stage_begin(self, name):
        if name in ("forward", "backward", "scaler_step_attempt", "zero_grad"):
            self.capture({"forward": "pre_forward", "backward": "pre_backward",
                          "scaler_step_attempt": "pre_scaler_step", "zero_grad": "pre_zero_grad"}[name])

    def stage_end(self, name):
        if name in ("forward", "backward", "scaler_step_attempt", "zero_grad"):
            self.capture({"forward": "post_forward", "backward": "post_backward",
                          "scaler_step_attempt": "post_scaler_step", "zero_grad": "post_zero_grad"}[name])

    def not_executed(self, name, reason):
        pass

    def remove_optimizer_hooks(self):
        # Trainer calls this in scaler.step's finally. P23 installs no hooks.
        pass

    def zero_grad(self, optimizer):
        self.capture("pre_zero_grad")
        optimizer.zero_grad()
        self.capture("post_zero_grad")

    def epoch_end(self):
        self.capture("epoch_end")
        super().epoch_end()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self._stop_sampler()
            super().close()
            # On training failure no successful result is constructed. This final
            # snapshot also observes best restore and explicit checkpoint I/O scope.
            if not self.errors and sys.exc_info()[0] is None:
                self.capture("segment_end")
                if self.rss_reason is None:
                    self.sample()
        finally:
            self.trainer = None
            self.thread = None


def _digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _validate_stage_pairs(rows):
    pairs = (("pre_forward", "post_forward"), ("pre_backward", "post_backward"),
             ("pre_scaler_step", "post_scaler_step"), ("pre_zero_grad", "post_zero_grad"))
    if not any(r["boundary"] in {p[0] for p in pairs} for r in rows):
        return  # Legacy v1 artifacts contain only post-stage observations.
    for before, after in pairs:
        pending = None
        for row in rows:
            if row["boundary"] == before:
                if pending is not None:
                    raise ValueError("incomplete memory stage boundary pair")
                pending = row
            elif row["boundary"] == after:
                if (pending is None or pending["query_end_ns"] > row["query_start_ns"] or
                        any(pending[k] != row[k] for k in ("epoch_id", "microbatch_id", "window_id", "attempt_id"))):
                    raise ValueError("incomplete or unordered memory stage boundary pair")
                pending = None
        if pending is not None:
            raise ValueError("incomplete memory stage boundary pair")


def _validate(data):
    required = {"schema_version", "format", "feature_ids", "source_p19", "source_sha256", "source_segment_id",
                "protocol", "status", "reason", "observations", "rss", "cuda", "activation", "worker_rss",
                "instrumentation", "limitations", "sha256"}
    if set(data) != required or data["schema_version"] != 1 or type(data["schema_version"]) is not int:
        raise ValueError("invalid training memory schema")
    if data["format"] != "hyso.performance.training_memory" or data["feature_ids"] != ["P23"]:
        raise ValueError("invalid P23 identity")
    if _digest({k: v for k, v in data.items() if k != "sha256"}) != data["sha256"]:
        raise ValueError("training memory evidence checksum mismatch")
    source, digest = _source(TrainingPerformanceResult(data["source_p19"]))
    if digest != data["source_sha256"] or source["segment_id"] != data["source_segment_id"]:
        raise ValueError("invalid source association")
    protocol = TrainingMemoryProtocol(**data["protocol"])
    source_device = source["provenance"]["runtime"]["device"]
    if (source_device == "cpu") != (protocol.device == "cpu") or (source_device not in ("cpu", "cuda") and source_device != protocol.device):
        raise ValueError("memory/source device mismatch")
    if data["status"] != "partial" or data["reason"] != "activation_lifetime_and_isolated_window_peak_unavailable":
        raise ValueError("unsupported coverage claim")
    if data["activation"] != _unsupported("saved_tensor_lifetime_adapter_unvalidated") or data["worker_rss"] != _unsupported("worker_private_rss_not_observed"):
        raise ValueError("unsupported domains must be null")
    rows = data["observations"]
    if not 2 <= len(rows) <= protocol.max_boundaries or rows[0]["boundary"] != "ready" or rows[-1]["boundary"] != "segment_end":
        raise ValueError("invalid memory boundary ledger")
    source_epochs = {r["epoch_id"]: r for r in source["epochs"]}
    source_micro = {r["microbatch_id"]: r for r in source["observations"] if r["kind"] == "microbatch"}
    source_attempts = {r["attempt_id"]: r for r in source["observations"] if r["kind"] == "attempt"}
    allowed = {"ready", "epoch_ready", "pre_microbatch", "post_forward", "post_backward", "pre_attempt",
               "post_scaler_step", "post_zero_grad", "post_attempt", "epoch_end", "segment_end",
               "pre_forward", "pre_backward", "pre_scaler_step", "pre_zero_grad"}
    count, end, previous_cuda = 0, 0, None
    for row in rows:
        if set(row) != {"boundary", "epoch_id", "microbatch_id", "window_id", "attempt_id", "fields", "logical",
                        "cuda", "scope", "status", "reason", "query_start_ns", "query_end_ns"} or row["status"] != "measured" or row["reason"] is not None or row["scope"] != "registered_tensor_state_and_allocator_boundary":
            raise ValueError("invalid boundary schema/scope")
        if row["boundary"] not in allowed:
            raise ValueError("unknown memory boundary")
        _int(row["query_start_ns"]); _int(row["query_end_ns"])
        if not end <= row["query_start_ns"] <= row["query_end_ns"]:
            raise ValueError("unordered boundary queries")
        end = row["query_end_ns"]
        if row["boundary"] in ("ready", "segment_end"):
            if any(row[k] is not None for k in ("epoch_id", "microbatch_id", "window_id", "attempt_id")):
                raise ValueError("segment snapshots cannot claim epoch IDs")
        else:
            epoch = source_epochs.get(row["epoch_id"])
            if epoch is None or not epoch["start_ns"] <= row["query_start_ns"] <= row["query_end_ns"] <= epoch["end_ns"]:
                raise ValueError("memory boundary outside source epoch")
        for key, mapping in (("microbatch_id", source_micro), ("attempt_id", source_attempts)):
            if row[key] is not None:
                linked = mapping.get(row[key])
                if linked is None or linked["epoch_id"] != row["epoch_id"] or linked["window_id"] != row["window_id"] or not linked["start_ns"] <= row["query_start_ns"] <= row["query_end_ns"] <= linked["end_ns"]:
                    raise ValueError("memory boundary outside linked training scope")
        expected_fields = {}
        if row["boundary"] == "post_attempt":
            linked = source_attempts[row["attempt_id"]]
            expected_fields = {k:linked[k] for k in ("applied", "skipped", "samples")}
        if row["fields"] != expected_fields:
            raise ValueError("memory applied/skipped/sample ledger mismatch")
        logical = row["logical"]
        _validate_logical_storage(logical)
        count += len(logical["tensors"]) + len(logical["optimizer_metadata"])
        for tensor in logical["tensors"]:
            if tensor["component"] not in ("parameter", "buffer", "gradient", "optimizer") or type(tensor["frozen"]) is not bool:
                raise ValueError("invalid logical component")
            if not tensor["names"] or any(type(n) is not str or not n for n in tensor["names"]):
                raise ValueError("invalid tensor names")
            if tensor["status"] == "measured":
                if tensor["reason"] is not None:
                    raise ValueError("measured tensors cannot have unsupported reasons")
                for dimension in tensor["shape"]:
                    _int(dimension)
                for key in ("elements", "element_size_bytes", "logical_bytes", "storage_bytes", "storage_offset"):
                    _int(tensor[key])
                if tensor["logical_bytes"] != tensor["elements"] * tensor["element_size_bytes"] or tensor["elements"] != math.prod(tensor["shape"]):
                    raise ValueError("inconsistent logical bytes")
            elif tensor["status"] != "unsupported" or tensor["logical_bytes"] is not None:
                raise ValueError("invalid unsupported tensor")
        summary = _logical_summary(logical["tensors"])
        if not logical["optimizer_supported"]:
            summary["optimizer"].update(logical_bytes=None, elements=None, status="unsupported",
                reason="custom optimizer state traversal is unvalidated")
        if summary != logical["summary"] or logical["unique_storage_bytes"] != sum({r["storage_id"]:r["storage_bytes"] for r in logical["tensors"] if r["storage_id"] is not None}.values()):
            raise ValueError("inconsistent logical summary/storage evidence")
        cuda = row["cuda"]
        if protocol.device == "cpu":
            if cuda is not None:
                raise ValueError("CPU is not CUDA allocator evidence")
        else:
            if cuda["allocator_backend"] != "native":
                raise ValueError("unvalidated CUDA allocator backend")
            for key in ("allocated_bytes", "reserved_bytes", "lifetime_peak_allocated_bytes", "lifetime_peak_reserved_bytes"):
                _int(cuda[key])
            if cuda["allocated_bytes"] > cuda["reserved_bytes"] or cuda["allocated_bytes"] > cuda["lifetime_peak_allocated_bytes"] or cuda["reserved_bytes"] > cuda["lifetime_peak_reserved_bytes"]:
                raise ValueError("inconsistent CUDA counters")
            if previous_cuda is not None and any(cuda[k] < previous_cuda[k] for k in
                    ("lifetime_peak_allocated_bytes", "lifetime_peak_reserved_bytes")):
                raise ValueError("CUDA peak counters reset during observation")
            previous_cuda = cuda
    if count > protocol.max_tensor_records:
        raise ValueError("unbounded tensor records")
    for boundary, kind in (("pre_microbatch", "microbatch"), ("post_forward", "microbatch"),
                           ("post_backward", "microbatch"), ("pre_attempt", "attempt"), ("post_attempt", "attempt")):
        actual = [r[kind + "_id"] for r in rows if r["boundary"] == boundary]
        expected = [r[kind + "_id"] for r in source["observations"] if r["kind"] == kind]
        if actual != expected:
            raise ValueError("boundary IDs disagree with training ledger")
    _validate_stage_pairs(rows)
    rss = data["rss"]
    samples = rss["observations"]
    if len(samples) > protocol.max_samples:
        raise ValueError("unbounded RSS samples")
    previous, hwm, backend = 0, 0, None
    for sample in samples:
        for key in ("query_start_ns", "query_end_ns", "timestamp_ns", "rss_bytes", "lifetime_hwm_bytes"):
            _int(sample[key])
        if not previous <= sample["query_start_ns"] <= sample["timestamp_ns"] <= sample["query_end_ns"] or sample["timestamp_ns"] != (sample["query_start_ns"]+sample["query_end_ns"])//2 or sample["lifetime_hwm_bytes"] < max(hwm,sample["rss_bytes"]):
            raise ValueError("inconsistent RSS observation")
        if sample["backend"] not in ("windows.GetProcessMemoryInfo.working_set", "linux.proc_self_status") or (backend is not None and backend != sample["backend"]):
            raise ValueError("invalid/changing native RSS backend")
        previous = sample["query_end_ns"]
        hwm, backend = sample["lifetime_hwm_bytes"], sample["backend"]
    if rss["summary"] != _rss_summary(samples, rss["reason"]) or rss["gaps_ns"] != [b["timestamp_ns"]-a["timestamp_ns"] for a,b in zip(samples,samples[1:])]:
        raise ValueError("inconsistent RSS summary")
    if data["cuda"] != _cuda_summary(rows, protocol.device):
        raise ValueError("inconsistent CUDA summary/reset policy")
    ins = data["instrumentation"]
    if ins["isolated"] is not False or ins["workload_memory_budget"] is not None or ins["hard_timeout"] is not None or ins["total_overhead_ns"] is not None:
        raise ValueError("unsupported isolation/budget/overhead claim")
    if ins["boundary_query_ns"] != sum(r["query_end_ns"]-r["query_start_ns"] for r in rows) or ins["rss_query_ns"] != sum(r["query_end_ns"]-r["query_start_ns"] for r in samples) or ins["stream_id"] != source["instrumentation"]["stream_id"]:
        raise ValueError("invalid instrumentation accounting")


def _cuda_summary(rows, device):
    if device == "cpu":
        return _unsupported("CPU training; CUDA not initialized by observer")
    return dict(status="measured", scope="process_device_allocator_boundary_samples_and_lifetime_peaks",
                device=device,
                baseline=rows[0]["cuda"], end=rows[-1]["cuda"], reset_performed=False,
                reset_ownership="none", window_peak=_unsupported("no_reset_no_isolated_worker"),
                incremental_window_peak=_unsupported("lifetime_peak_cannot_be_attributed_to_window"),
                sampled_peak_allocated_bytes=max(r["cuda"]["allocated_bytes"] for r in rows),
                sampled_peak_reserved_bytes=max(r["cuda"]["reserved_bytes"] for r in rows))


def _basic_memory_report(rows, rss, device, *, isolated=False, windows=(), reason=None):
    """Offline basic-scope projection, independent of saved-activation evidence.

    Keep schema1/CSV evidence unchanged. Values are bytes in separate domains;
    a measured empty gradient/state is zero, missing evidence is always null.
    """
    def metric(value, scope, status="measured", reason=None, **fields):
        return dict(value=value, unit="byte", status=status, reason=reason,
                    scope=scope, **fields)

    metrics = {}
    for component, name in (("parameter", "parameter_bytes"), ("gradient", "gradient_bytes"),
                            ("optimizer", "optimizer_state_bytes")):
        summaries = [r["logical"]["summary"][component] for r in rows]
        valid = bool(summaries) and all(r["status"] == "measured" for r in summaries)
        values = [r["logical_bytes"] for r in summaries] if valid else []
        missing = reason or next((r["reason"] for r in summaries if r["status"] != "measured"),
                                 "no_completed_measurement")
        metrics[name] = metric(values[-1] if valid else None, "registered_tensor_logical_boundary",
            "measured" if valid else "unsupported" if summaries else "unavailable",
            None if valid else missing, baseline_bytes=values[0] if valid else None,
            sampled_min_bytes=min(values) if valid else None,
            sampled_max_bytes=max(values) if valid else None)
    for source, name, scope in (
            ("current_rss_bytes", "process_rss_bytes", "current_process_end_observation"),
            ("sampled_peak_rss_bytes", "process_sampled_peak_rss_bytes", "sampled_fit_maximum_not_exact_peak"),
            ("current_lifetime_hwm_bytes", "process_lifetime_peak_rss_bytes", "process_lifetime_including_before_fit")):
        observed = rss["summary"][source] if rss is not None else {}
        valid = observed.get("status") == "measured"
        metrics[name] = metric(observed.get("value") if valid else None, scope,
            "measured" if valid else "unavailable",
            None if valid else observed.get("reason") or reason or "RSS_backend_unavailable")
    reset_windows = [r for r in windows if r["reset_performed"]]
    for kind in ("allocated", "reserved"):
        for peak in (False, True):
            name = "cuda_" + ("peak_" if peak else "") + kind + "_bytes"
            if device == "cpu":
                metrics[name] = metric(None, "CUDA_allocator", "not_applicable", "CPU_training")
            elif not rows:
                metrics[name] = metric(None, "CUDA_allocator", "unavailable", reason or "no_completed_measurement")
            elif peak and reset_windows:
                # Each reset has one owned window; never subtract lifetime peaks.
                valid = all(r["status"] == "measured" for r in reset_windows)
                metrics[name] = metric(max(r["peak_"+kind+"_bytes"] for r in reset_windows) if valid else None,
                    "maximum_of_owned_training_windows", "measured" if valid else "unsupported",
                    None if valid else "incomplete_owned_window_peak_evidence")
            else:
                key = ("lifetime_peak_" if peak else "") + kind + "_bytes"
                metrics[name] = metric(rows[-1]["cuda"][key],
                    "process_device_lifetime" if peak else "process_device_end_boundary")
    required = ("parameter_bytes", "gradient_bytes", "optimizer_state_bytes", "process_rss_bytes")
    if device != "cpu":
        required += tuple(k for k in metrics if k.startswith("cuda_"))
    return dict(format="hyso.performance.p23_basic_memory", schema_version=1,
        status="not_executed" if not rows else "complete" if all(metrics[k]["status"] == "measured" for k in required) else "partial",
        reason=reason, device=device, scope="isolated_recipe_fit" if isolated else "in_process_fit",
        metrics=metrics, reset_performed=bool(reset_windows),
        activation_lifetime_scope="experimental_excluded_from_basic_acceptance",
        limitations=["Logical components can alias and must not be added to RSS or allocator bytes.",
            "Gradient/state values describe the end boundary; sampled maxima show their lifecycle.",
            "RSS samples can miss transients; native RSS high-water marks include process history.",
            "In-process CUDA peaks include pre-fit history; only owned reset windows have fit-window scope."])


def _memory_diagnostics(rows, rss, device, *, isolated=False, windows=(), activation=None):
    """Derive changes within each domain; never add logical/RSS/allocator bytes.

    Sample extrema are lower-coverage observations, never stage/window peaks.
    This function reads verified artifacts only; no live model or device calls.
    """
    def changes(values, reason=None):
        valid = bool(values) and reason is None and all(v is not None for v in values)
        return dict(status="derived" if valid else "unsupported", reason=reason if not valid else None,
            baseline_bytes=values[0] if valid else None, end_bytes=values[-1] if valid else None,
            delta_bytes=values[-1]-values[0] if valid else None,
            sampled_min_bytes=min(values) if valid else None, sampled_max_bytes=max(values) if valid else None,
            sampled_range_bytes=max(values)-min(values) if valid else None)

    logical = {component: changes([r["logical"]["summary"][component]["logical_bytes"] for r in rows],
                    None if all(r["logical"]["summary"][component]["status"] == "measured" for r in rows)
                    else "unsupported_tensor_or_optimizer_representation")
               for component in ("parameter", "frozen_parameter", "buffer", "gradient", "optimizer")}
    rss_change = changes([r["rss_bytes"] for r in rss["observations"]], rss["reason"])
    cuda = (_unsupported("CPU_training") if device == "cpu" else
            {kind: changes([r["cuda"][kind+"_bytes"] for r in rows]) for kind in ("allocated", "reserved")})
    logical_timeline = [dict(boundary=r["boundary"], epoch_id=r["epoch_id"],
        microbatch_id=r["microbatch_id"], window_id=r["window_id"], attempt_id=r["attempt_id"],
        query_start_ns=r["query_start_ns"], query_end_ns=r["query_end_ns"],
        logical_bytes={k:r["logical"]["summary"][k]["logical_bytes"] for k in logical}, cuda=r["cuda"])
        for r in rows]
    return dict(core_status="complete" if rss_change["status"] == "derived" and
                    all(logical[k]["status"] == "derived" for k in ("parameter", "gradient", "optimizer")) else "partial",
        scope="isolated_recipe_fit" if isolated else "in_process_fit",
        resource_policy="advisory_unless_explicit_limits_requested", logical=logical,
        rss=rss_change, cuda_allocator=cuda, boundary_observations=logical_timeline,
        rss_observations=rss["observations"],
        peaks=dict(rss=_unsupported("sampled_RSS_maximum_is_not_a_training_peak"),
            cuda_training_windows=list(windows) if isolated else _unsupported("in_process_peak_reset_not_permitted"),
            stage_peaks=_unsupported("boundary_samples_are_not_stage_peaks")),
        activation=activation if activation is not None else _unsupported("saved_tensor_lifetime_adapter_unvalidated"),
        process_private_commit=_unsupported("not_observed_in_process_RSS_is_not_private_commit"),
        artifact_bytes=_unsupported("serialized_artifact_sizes_are_not_training_memory"),
        limitations=["logical components can alias; frozen parameters are a subset of parameters",
            "logical bytes, storage extents, RSS and CUDA allocator bytes must not be added",
            "RSS includes observer/runtime/main-process activity and excludes DataLoader workers",
            "sampled extrema can miss transients; boundary queries do not synchronize each stage",
            "allocator counters cover the process/device, not physical or other-process VRAM",
            "ready/end exclude construction and include fit finalization; phase scopes differ from P19",
            "query costs are included; total instrumentation overhead is not separately calibrated"])


@dataclass(frozen=True, init=False)
class TrainingMemoryResult:
    """Immutable canonical evidence; constructors and exports validate integrity."""
    __slots__ = ("_payload",)
    _payload: str

    def __init__(self, data):
        copied = _plain(data)
        _validate(copied)
        object.__setattr__(self, "_payload", json.dumps(copied, allow_nan=False, separators=(",", ":")))

    def to_dict(self):
        return json.loads(self._payload)

    def _export_data(self):
        data = _plain(self.to_dict())
        _validate(data)
        return data

    def diagnostics(self):
        """Changes and boundary samples by memory domain; v1 exports unchanged."""
        data = self._export_data()
        return _memory_diagnostics(data["observations"], data["rss"], data["protocol"]["device"],
                                   activation=data["activation"])

    def basic_metrics(self):
        """Basic P23 bytes/status/scope; no execution or saved-activation hooks.

        CPU CUDA metrics use not_applicable. Legacy schema1 exports retain their
        original full-scope coverage labels and are not rewritten by this view.
        """
        data = self._export_data()
        return _basic_memory_report(data["observations"], data["rss"], data["protocol"]["device"])

    def to_json(self, path, *, overwrite=False):
        _write_export(path, json.dumps(self._export_data(), ensure_ascii=False, sort_keys=True,
                                      indent=2, allow_nan=False) + "\n", overwrite)

    def to_csv(self, path, *, overwrite=False):
        data = self._export_data()
        columns = ["source_segment_id", "boundary", "epoch_id", "microbatch_id", "window_id", "attempt_id",
                   "query_start_ns", "query_end_ns", "parameter_bytes", "buffer_bytes", "gradient_bytes", "optimizer_bytes",
                   "allocated_bytes", "reserved_bytes"]
        rows = []
        for row in data["observations"]:
            plain = {k:row.get(k) for k in columns}
            plain["source_segment_id"] = data["source_segment_id"]
            for component in ("parameter", "buffer", "gradient", "optimizer"):
                plain[component+"_bytes"] = row["logical"]["summary"][component]["logical_bytes"]
            plain.update({k:(row["cuda"] or {}).get(k) for k in ("allocated_bytes", "reserved_bytes")})
            rows.append(plain)
        _write_export(path, _csv_text(columns, rows), overwrite)


def measure_training_memory(trainer, train_loader, val_loader=None, *, epochs=5,
                            protocol=None, timing_protocol=None, save_last_to=None, resume_from=None):
    """Mutate one fresh Trainer via one actual fit. No replay, resets or GC calls.

    Bounded instrumentation memory is not a workload RAM/VRAM safety budget. This
    in-process API has no hard timeout/isolation; use only trusted finite workloads.
    Full-fit RSS includes optional checkpoint I/O and finalization. The embedded
    P19 timing retains its narrower epoch scope and unchanged schema.
    """
    protocol = TrainingMemoryProtocol() if protocol is None else protocol
    if type(protocol) is not TrainingMemoryProtocol:
        raise TypeError("protocol must be TrainingMemoryProtocol")
    from hyso.training import Trainer
    if type(trainer) is not Trainer:
        raise TypeError("standard Trainer required")
    device = trainer.device
    if device.type == "cuda" and device.index is None:
        # Trainer has already placed its model; never change selected device.
        device = torch.device("cuda", torch.cuda.current_device())
    if str(device) != protocol.device:
        raise ValueError("protocol device must match Trainer device")
    if not _RUN_LOCK.acquire(blocking=False):
        raise TrainingMemoryError("concurrent/reentrant performance calls unsupported")
    captured = []
    def factory(t, p):
        recorder = _MemoryRecorder(t, p, protocol)
        captured.append(recorder)
        return recorder
    try:
        timing = _measure_training(trainer, train_loader, val_loader, epochs=epochs, protocol=timing_protocol,
            save_last_to=save_last_to, resume_from=resume_from, recorder_factory=factory)
        recorder = captured[0]
        if recorder.errors:
            raise TrainingMemoryError("RSS sampler failed: " + recorder.errors[0])
        source = timing._export_data()
        source["limitations"][1] = "P19 projection of P23 capture; no P20 stage timing result"
        source["instrumentation"]["event_policy"] = "P19 bounded event pairs; memory boundary queries do not record events"
        timing = TrainingPerformanceResult(source)
        source, digest = _source(timing)
        samples = recorder.rss
        data = dict(schema_version=1, format="hyso.performance.training_memory", feature_ids=["P23"],
            source_p19=source, source_sha256=digest, source_segment_id=source["segment_id"], protocol=asdict(protocol),
            status="partial", reason="activation_lifetime_and_isolated_window_peak_unavailable",
            observations=recorder.boundaries,
            rss=dict(observations=samples, reason=recorder.rss_reason, summary=_rss_summary(samples, recorder.rss_reason),
                     gaps_ns=[b["timestamp_ns"]-a["timestamp_ns"] for a,b in zip(samples,samples[1:])]),
            cuda=_cuda_summary(recorder.boundaries, protocol.device),
            activation=_unsupported("saved_tensor_lifetime_adapter_unvalidated"),
            worker_rss=_unsupported("worker_private_rss_not_observed"),
            instrumentation=dict(scope="ready_before_fit_to_after_fit_including_optional_checkpoint_IO_and_finalization",
                source_timing_scope="P19_epoch_training_and_separate_validation",
                stream_id=source["instrumentation"]["stream_id"],
                boundary_query_ns=sum(r["query_end_ns"]-r["query_start_ns"] for r in recorder.boundaries),
                rss_query_ns=sum(r["query_end_ns"]-r["query_start_ns"] for r in samples),
                total_overhead_ns=None, workload_memory_budget=None, hard_timeout=None, isolated=False),
            limitations=["Logical tensor bytes and deduplicated full storage extents are not physical RAM/VRAM.",
                "Frozen parameter bytes are a subset of parameter bytes; components may share storage.",
                "RSS includes runtime, main-process loader, observer and other process activity; workers excluded.",
                "Sampled RSS/allocator maxima may miss short transients; lifetime peaks include pre-fit history.",
                "CUDA allocator counters exclude non-allocator/device-wide/other-process VRAM; no counter reset.",
                "Boundary counters are host-query observations without per-boundary synchronization; P19 drains remain.",
                "No activation hook, isolated worker, hard resource budget, leak attribution or window-peak claim.",
                "Snapshots traverse registered model and native SGD/Adam/AdamW state; EMA/scaler/scheduler tensors excluded.",
                "Instrumentation and GIL scheduling perturb wall time and process memory; total overhead unknown."])
        data["sha256"] = _digest(data)
        return TrainingMemoryResult(data)
    finally:
        try:
            for recorder in captured:
                recorder.close()
        finally:
            captured.clear()
            _RUN_LOCK.release()
