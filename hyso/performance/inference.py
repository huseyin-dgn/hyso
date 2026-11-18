"""P01 sample-only analytics and P02 continuous resident completion windows."""
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import platform
import random
import time
import uuid

import numpy as np
import torch

from hyso.benchmark import BenchmarkResult, BenchmarkRunner
# Share the existing guard and lock, not its timer or allocator probe. These are
# deliberately internal dependencies covered by the legacy regression suite.
from hyso.benchmark.runner import (_Snapshot, _descriptor, _bits, _output, _RUN_LOCK, StateMutationError)
from .result import LatencyAnalysis, ThroughputResult, _integer, _positive, _latency_metrics


@dataclass(frozen=True)
class TailPolicy:
    min_samples: int = 1000
    min_tail_samples: int = 10

    def __post_init__(self):
        _integer(self.min_samples, 1)
        _integer(self.min_tail_samples, 1)


@dataclass(frozen=True)
class ThroughputProtocol:
    device: str = "cpu"
    precision: str = "fp32"
    warmup_iterations: int = 10
    batch_count: int | None = 100
    duration_seconds: float | None = None
    repeats: int = 3
    queue_depth: int = 16
    concurrency: int = 1

    def __post_init__(self):
        if type(self.device) is not str or (self.device != "cpu" and self.device != "cuda" and
                not (self.device.startswith("cuda:") and self.device[5:].isdigit()
                     and str(int(self.device[5:])) == self.device[5:])):
            raise ValueError("unsupported device; CPU fallback is disabled")
        allowed = ("fp32", "fp64") if self.device == "cpu" else ("fp32", "amp_fp16")
        if type(self.precision) is not str or self.precision not in allowed:
            raise ValueError("unsupported precision for selected device")
        _integer(self.warmup_iterations)
        _integer(self.repeats, 1)
        _integer(self.queue_depth, 2)
        if type(self.concurrency) is not int or self.concurrency != 1:
            raise ValueError("only concurrency=1 is supported")
        if (self.batch_count is None) == (self.duration_seconds is None):
            raise ValueError("provide exactly one of batch_count or duration_seconds")
        if self.batch_count is not None:
            _integer(self.batch_count, 1)
        if self.duration_seconds is not None:
            _positive(self.duration_seconds)
            if not math.isfinite(self.duration_seconds * 1e9) or self.duration_seconds * 1e9 < 1:
                raise ValueError("duration must be representable in nanoseconds")


def _base(kind, feature, protocol, provenance):
    return dict(format="hyso.performance." + kind, schema_version=1, result_id=str(uuid.uuid4()),
                feature_ids=[feature], identity=dict(experiment_run_id=None, source_result_id=None),
                protocol=protocol, provenance=provenance,
                diagnostics=dict(warnings=[], limitations=[]))


def analyze_latency(benchmark_result, *, quantiles=(.99,), tail_policy=TailPolicy()):
    """Analyze existing R*N samples; never execute a model or mutate the source.

    TailPolicy may be explicitly relaxed for descriptive exploration; no policy
    establishes independent repeats, effective sample size or a confidence interval.
    """
    if not isinstance(benchmark_result, BenchmarkResult):
        raise TypeError("benchmark_result must be a validated BenchmarkResult")
    if type(tail_policy) is not TailPolicy:
        raise TypeError("tail_policy must be TailPolicy")
    if type(quantiles) not in (tuple, list) or not quantiles:
        raise ValueError("quantiles must be a nonempty sequence")
    if any(type(q) not in (int, float) or not math.isfinite(q) or not 0 < q < 1 for q in quantiles):
        raise ValueError("quantiles must be finite fractions strictly between zero and one")
    if len(set(quantiles)) != len(quantiles):
        raise ValueError("duplicate quantile")
    # Revalidation also rejects a corrupt/deserialized or subclass-supplied source.
    source = BenchmarkResult(benchmark_result.to_dict()).to_dict()
    rows = source["timing"]["latency_samples_ms"]
    n = sum(map(len, rows))
    policy = asdict(tail_policy)
    data = _base("latency_analysis", "P01", dict(id="resident_tail", version=1,
                 scope=source["timing"]["timing_scope"], method="linear_(n-1)*q",
                 quantiles=list(quantiles), tail_policy=policy),
                 {key: source[key] for key in ("model", "input", "environment", "provenance", "configuration")})
    data["identity"]["source_result_id"] = source["identity"]["run_id"]
    data.update(requested=dict(samples=n, repeats=len(rows)),
                actual=dict(samples=n, repeats=len(rows), samples_per_repeat=[len(row) for row in rows]),
                measurements=_latency_metrics(rows, quantiles, policy),
                raw_evidence=dict(source_result_id=source["identity"]["run_id"],
                    source_sha256=hashlib.sha256(json.dumps(source, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                    latency_samples_ms=rows))
    data["diagnostics"]["limitations"] = ["Pooled samples may be correlated; repeats are not asserted independent.",
        "Tail eligibility does not establish a confidence interval or effective sample size."]
    return LatencyAnalysis(data)


class _CompletionLedger:
    """Contiguous IDs can only be committed once, after actual completion."""
    def __init__(self, batch):
        _integer(batch, 1)
        self.batch = batch
        self.completed = 0
        self.groups = 0

    def commit(self, ids):
        if not ids or ids != list(range(self.completed, self.completed + len(ids))):
            raise ValueError("duplicate, negative or out-of-order completion")
        self.completed += len(ids)
        self.groups += 1


def _clock():
    value = time.perf_counter_ns()
    _integer(value)
    return value


def _context_check(device, stream):
    if stream is not None and (torch.cuda.current_device() != device.index or
                               torch.cuda.current_stream(device) != stream):
        raise StateMutationError("forward changed selected CUDA device/current stream")


def _shape_check(output, expected, device):
    if (type(output) is not torch.Tensor or output.device != device or output.layout != torch.strided
            or not output.is_floating_point() or list(output.shape) != expected["shape"]
            or str(output.dtype) != expected["dtype"]):
        raise ValueError("forward changed output shape/dtype/device")


def _window(model, inputs, plan, device, stream, expected):
    cuda = stream is not None
    # All ring events and validity storage are created/initialized before t0.
    events = [torch.cuda.Event() for _ in range(plan.queue_depth)] if cuda else []
    terminal = torch.cuda.Event() if cuda else None
    valid = torch.ones((), dtype=torch.bool, device=device)
    if cuda:
        for event in events:
            event.record(stream)
        terminal.record(stream)
        terminal.synchronize()
    ledger = _CompletionLedger(inputs.shape[0])
    pending = []
    admitted = max_pending = backpressure_syncs = 0
    start = _clock()
    deadline = None if plan.duration_seconds is None else start + int(plan.duration_seconds * 1e9)
    try:
        while (admitted < plan.batch_count if deadline is None else _clock() < deadline):
            _context_check(device, stream)
            output = model(inputs)
            _context_check(device, stream)
            _shape_check(output, expected, device)
            # Device-side finite reductions are included instrumentation. There
            # is no .item()/D2H/per-forward sync; one flag is read after t1.
            valid.logical_and_(torch.isfinite(output).all())
            if cuda:
                event = events[len(pending)]
                event.record(stream)
                pending.append((admitted, output, event))
                max_pending = max(max_pending, len(pending))
                if len(pending) == plan.queue_depth:
                    # One drain per full bounded group, never one per forward.
                    if not pending[-1][2].query():
                        pending[-1][2].synchronize()
                        backpressure_syncs += 1
                    if not all(item[2].query() for item in pending):
                        raise RuntimeError("incomplete CUDA completion group")
                    ledger.commit([item[0] for item in pending])
                    pending.clear()
            else:
                ledger.commit([admitted])
                max_pending = 1
            admitted += 1
            del output
        admission_end = _clock()
        if cuda:
            terminal.record(stream)
            terminal.synchronize()
            if not terminal.query() or not all(item[2].query() for item in pending):
                raise RuntimeError("incomplete CUDA terminal tail")
            if pending:
                ledger.commit([item[0] for item in pending])
            pending.clear()
        end = _clock()
        if not bool(valid.item()):
            raise FloatingPointError("nonfinite forward output")
        if end <= start or not start <= admission_end <= end:
            raise FloatingPointError("nonpositive or inconsistent completion clock")
        if ledger.completed < 1 or ledger.completed != admitted:
            raise RuntimeError("no completed work or incomplete tail")
        elapsed = (end - start) / 1e9
        rate = ledger.completed * ledger.batch / elapsed
        _positive(rate)
        return dict(start_ns=start, admission_end_ns=admission_end, end_ns=end,
                    admitted_batches=admitted, completed_batches=ledger.completed,
                    completed_images=ledger.completed * ledger.batch, pending_batches=admitted-ledger.completed,
                    completed_id_range=[0, ledger.completed-1], completion_groups=ledger.groups,
                    max_pending_batches=max_pending, backpressure_syncs=backpressure_syncs,
                    terminal_drains=int(cuda), elapsed_wall_seconds=elapsed,
                    tail_seconds=(end-admission_end)/1e9, images_per_second=rate)
    finally:
        # Caller handles a failed runtime drain; never retain ring outputs in a
        # logger/closure/result. No allocator reset or global GC operation.
        pending.clear()


def measure_throughput(model, input_source, *, protocol=ThroughputProtocol()):
    """Continuous resident eager inference, completed images / host wall seconds.

    input_source is one resident contiguous NCHW Tensor, defensively cloned.
    Queue capacity bounds CUDA event/output retention; concurrency is one stream.
    Transfers, preflight/warmup, state/hash inspection and exports are outside
    windows. Loop/finite-check/queue instrumentation and terminal tail are inside.
    Hidden cross-stream/device work and concurrent caller activity are unsupported.
    """
    if type(protocol) is not ThroughputProtocol:
        raise TypeError("protocol must be ThroughputProtocol")
    if type(input_source) is not torch.Tensor:
        raise TypeError("input_source must be one resident NCHW Tensor")
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("concurrent/reentrant benchmark/performance runs are unsupported")
    contexts = ExitStack()
    snapshot = None
    rng = cuda_rng = device = None
    try:
        rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        guard = BenchmarkRunner(model, device=protocol.device, precision=protocol.precision,
                                example_input=input_source, warmup_iterations=0, measure_iterations=1, repeats=1)
        device = guard._device
        stream = None
        if device.type == "cuda":
            contexts.enter_context(torch.cuda.device(device))
            stream = torch.cuda.current_stream(device)
            contexts.enter_context(torch.cuda.stream(stream))
            cuda_rng = torch.cuda.get_rng_state(device)
        modules, entries = guard._validate()
        with torch.inference_mode(False), torch.no_grad():
            inputs = input_source.detach().clone()
        snapshot = _Snapshot(modules, entries, inputs)
        descriptor, weights_hash = _descriptor(modules, entries)
        provenance = dict(model=dict(qualified_class=type(model).__module__ + "." + type(model).__qualname__,
                                    structural_descriptor=descriptor, weights_sha256=weights_hash),
            input=dict(shape=list(inputs.shape), dtype=str(inputs.dtype), device=str(device), source="resident_clone",
                       content_sha256=hashlib.sha256(_bits(inputs).numpy().tobytes()).hexdigest()),
            environment=dict(device=str(device), python=platform.python_version(), torch=str(torch.__version__),
                cuda_build=torch.version.cuda, os=platform.platform(), cpu_threads=torch.get_num_threads(),
                interop_threads=torch.get_num_interop_threads(), logical_cpu_count=os.cpu_count(),
                gpu_name=torch.cuda.get_device_name(device) if stream is not None else None,
                gpu_capability=list(torch.cuda.get_device_capability(device)) if stream is not None else None,
                gpu_total_memory_bytes=torch.cuda.get_device_properties(device).total_memory if stream is not None else None,
                allocator_backend=torch.cuda.get_allocator_backend() if stream is not None else None,
                stream_id=int(stream.cuda_stream) if stream is not None else None,
                clock_resolution_ns=time.get_clock_info('perf_counter').resolution*1e9,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                matmul_precision=torch.get_float32_matmul_precision(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),
            precision=dict(policy=protocol.precision, master_dtype=str(guard._dtype), input_dtype=str(inputs.dtype),
                           autocast=protocol.precision == "amp_fp16"),
            source_revision=None, revision_status="not_collected", driver_version=None, driver_status="not_collected",
            thermal_clock_power_status="not_collected", isolation="caller_exclusive_assertion_required")
        windows = []
        try:
            model.eval()
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                       enabled=protocol.precision == "amp_fp16"):
                _context_check(device, stream)
                output = model(inputs)
                _context_check(device, stream)
                expected = _output(output, inputs.shape[0], device=device)
                provenance['model']['output'] = expected
                del output
                snapshot.check(full=True)
                for repeat in range(protocol.repeats):
                    for _ in range(protocol.warmup_iterations):
                        _context_check(device, stream)
                        output = model(inputs)
                        _context_check(device, stream)
                        _output(output, inputs.shape[0], expected, device)
                        del output
                        snapshot.check()
                    if stream is not None:
                        torch.cuda.synchronize(device)
                    row = _window(model, inputs, protocol, device, stream, expected)
                    row["repeat_index"] = repeat
                    windows.append(row)
                    snapshot.check(full=True)
            snapshot.check(full=True)
        except BaseException as error:
            # Even a CUDA asynchronous drain failure must reach outer finally
            # for modes/RNG/stream/lock cleanup. A poisoned context is not repaired.
            if stream is not None:
                torch.cuda.synchronize(device)
            try:
                snapshot.check(full=True)
            except StateMutationError as mutation:
                snapshot.restore_values()
                raise StateMutationError(str(mutation) + "; best-effort value rollback, not an arbitrary transaction") from error
            raise
        total_images = sum(w["completed_images"] for w in windows)
        elapsed = math.fsum(w["elapsed_wall_seconds"] for w in windows)
        data = _base("throughput", "P02", dict(id="continuous_completion_window", version=1,
            scope="resident_forward_completion_wall", configuration=asdict(protocol),
            clock="perf_counter_ns", synchronization="bounded_group_completion_and_terminal_event" if stream is not None else "synchronous_host",
            included=["forward", "finite_check_kernels", "loop_and_queue_bookkeeping", "backpressure", "terminal_tail"],
            excluded=["input_clone_transfer", "preflight", "warmup", "state_checks", "hashing", "export", "tracking"]), provenance)
        data.update(requested=dict(batch_count=protocol.batch_count, duration_seconds=protocol.duration_seconds,
                        repeats=protocol.repeats, warmup_iterations=protocol.warmup_iterations),
                    actual=dict(repeats=len(windows), completed_batches=sum(w["completed_batches"] for w in windows),
                                completed_images=total_images, pending_batches=0),
                    measurements=dict(value=total_images/elapsed, unit="images/s", status="measured", reason=None,
                        elapsed_wall_seconds=elapsed, aggregation="total_completed_images_over_total_wall_seconds"),
                    raw_evidence=dict(windows=windows))
        data["diagnostics"]["limitations"] = ["Resident fixed-input eager completion throughput, not camera/display/end-to-end FPS.",
            "Includes finite-check and bounded queue instrumentation; not bare forward latency.",
            "Hidden cross-stream/device work, arbitrary Python side effects and concurrent callers unsupported.",
            "No hard GPU watchdog; asynchronous runtime failures may leave a poisoned context.",
            "Clocks, thermals and background work are uncontrolled; repeats need not be independent."]
        return ThroughputResult(data)
    finally:
        try:
            if snapshot is not None:
                snapshot.restore_modes()
        finally:
            try:
                if rng is not None:
                    random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
            finally:
                try:
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state(cuda_rng, device)
                finally:
                    try:
                        contexts.close()
                    finally:
                        _RUN_LOCK.release()
