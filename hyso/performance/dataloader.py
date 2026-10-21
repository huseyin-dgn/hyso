"""P21 main-process iterator evidence; worker service clocks are unsupported.

No loader reconstruction, worker IPC, dataset wrapping, or worker ownership.
The observer drops its iterator references; PyTorch retains normal teardown and
persistent-worker ownership. No retrospective inference of wait from P19 clocks.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
import weakref

from torch.utils.data import DataLoader

from hyso.benchmark.result import _plain, _csv_text, _write_export
from .result import _quantile
from .training import (_Recorder, TrainingTimingError, TrainingPerformanceResult,
                       TrainingTimingProtocol, _measure_training, _int, _duration, _type)
from .training_details import (_StageRecorder, TrainingStageProtocol, TrainingStageResult,
    TrainingThroughputResult, _stage_result, _source, analyze_training_throughput)


@dataclass(frozen=True)
class DataLoaderAnalysisProtocol:
    """Hard P21 record ceiling, including iterator init and exhaustion calls."""
    max_records: int = 100000

    def __post_init__(self):
        _int(self.max_records, 1)
        if self.max_records > 1000000:
            raise ValueError("loader record ceiling is 1000000")


class _ObservedIterator:
    def __init__(self, owner, loader, phase):
        self.owner = weakref.ref(owner)
        self.loader, self.iterator, self.phase = loader, None, phase
        self.started = self.closed = False
        self.batch_index = 0

    def __iter__(self):
        if not self.started and not self.closed:
            owner = self.owner()
            owner.reserve_loader()
            start = time.perf_counter_ns()
            try:
                self.iterator = iter(self.loader)
                end = time.perf_counter_ns()
                owner.loader_row("iterator_init", self.phase, start, end, None)
                self.started = True
            except BaseException:
                self.close()
                raise
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        if not self.started:
            self.__iter__()
        owner = self.owner()
        try:
            owner.reserve_loader()
            start = time.perf_counter_ns()
            try:
                batch = next(self.iterator)
            except StopIteration:
                end = time.perf_counter_ns()
                owner.loader_row("exhaustion", self.phase, start, end, None)
                self.close()
                raise
            end = time.perf_counter_ns()
            self.batch_index += 1
            row = owner.loader_row("batch", self.phase, start, end, self.batch_index)
            owner.waiting[self.phase] = row
            return batch
        except BaseException:
            self.close()
            raise

    def close(self):
        # No private _shutdown_workers call, worker kill, or persistent reset.
        self.closed = True
        self.iterator = self.loader = None
        owner = self.owner()
        if owner is not None:
            owner.live.discard(self)


class _LoaderMixin:
    def init_loader(self, protocol):
        self.loader_protocol = protocol
        self.loader_rows, self.live, self.waiting = [], set(), {}
        self.window_batches = []

    def reserve_loader(self):
        if len(self.loader_rows) >= self.loader_protocol.max_records:
            raise TrainingTimingError("P21 record buffer exhausted; incomplete evidence is not a result")

    def loader_row(self, kind, phase, start, end, index):
        row = dict(kind=kind, phase=phase, epoch_id=self.epoch["epoch_id"], batch_index=index,
                   start_ns=start, end_ns=end, host_ms=(end-start)/1e6,
                   delivery_ns=end if kind == "batch" else None, samples=None,
                   microbatch_id=None, window_id=None, attempt_id=None)
        self.loader_rows.append(row)
        return row

    def observe_loader(self, loader, phase):
        iterator = _ObservedIterator(self, loader, phase)
        self.live.add(iterator)
        return iterator

    def microbatch_begin(self, microbatch_id, new_window):
        super().microbatch_begin(microbatch_id, new_window)
        row = self.waiting["training"]
        row.update(microbatch_id=microbatch_id, window_id=self.window_id)
        self.window_batches.append(row)

    def microbatch_end(self, samples, inputs, targets):
        self.waiting["training"]["samples"] = samples
        super().microbatch_end(samples, inputs, targets)

    def attempt_end(self, updates, samples):
        attempt_id = self.active["attempt"][0]["attempt_id"]
        super().attempt_end(updates, samples)
        for row in self.window_batches:
            row["attempt_id"] = attempt_id
        self.window_batches.clear()

    def validation_batch(self, samples):
        self.waiting["validation"]["samples"] = samples

    def close(self):
        try:
            for iterator in tuple(self.live):
                iterator.close()
            self.waiting.clear()
            self.window_batches.clear()
        finally:
            super().close()


class _LoaderRecorder(_LoaderMixin, _Recorder):
    def __init__(self, trainer, timing_protocol, loader_protocol):
        super().__init__(trainer, timing_protocol)
        self.init_loader(loader_protocol)


class _LoaderStageRecorder(_LoaderMixin, _StageRecorder):
    def __init__(self, trainer, timing_protocol, loader_protocol, stage_protocol):
        super().__init__(trainer, timing_protocol, stage_protocol)
        self.init_loader(loader_protocol)


def _configuration(loader):
    if loader is None:
        return None
    if type(loader) is not DataLoader:
        raise TypeError("P21 requires a standard existing torch DataLoader")
    context = loader.multiprocessing_context
    return dict(type=_type(loader), dataset_type=_type(loader.dataset), sampler_type=_type(loader.sampler),
                batch_sampler_type=_type(loader.batch_sampler), batch_size=loader.batch_size,
                num_workers=loader.num_workers, pin_memory=loader.pin_memory,
                pin_memory_device=getattr(loader, "pin_memory_device", ""),
                prefetch_factor=loader.prefetch_factor, persistent_workers=loader.persistent_workers,
                drop_last=loader.drop_last, timeout=loader.timeout,
                multiprocessing_context=None if context is None else context.get_start_method(),
                worker_init_fn_type=None if loader.worker_init_fn is None else _type(loader.worker_init_fn),
                worker_init_fn_name=None if loader.worker_init_fn is None else getattr(loader.worker_init_fn, "__qualname__", _type(loader.worker_init_fn)),
                collate_fn_type=_type(loader.collate_fn), collate_fn_name=getattr(loader.collate_fn, "__qualname__", _type(loader.collate_fn)),
                generator_present=loader.generator is not None)


def _hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _distribution(values):
    if not values:
        return dict(count=0, mean_ms=None, p50_ms=None, p95_ms=None,
                    p99=dict(value=None, unit="ms", status="insufficient_samples", reason="requires_1000_delivered_batches"))
    return dict(count=len(values), mean_ms=sum(values)/len(values), p50_ms=_quantile(values, .5),
                p95_ms=_quantile(values, .95),
                p99=dict(value=_quantile(values, .99) if len(values) >= 1000 else None, unit="ms",
                         status="derived" if len(values) >= 1000 else "insufficient_samples",
                         reason=None if len(values) >= 1000 else "requires_1000_delivered_batches"))


def _summaries(rows, source):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["epoch_id"], row["phase"]), []).append(row)
    result = []
    for epoch in source["epochs"]:
        for phase in ("training", "validation"):
            selected = grouped.get((epoch["epoch_id"], phase), [])
            if not selected:
                continue
            batches = [r for r in selected if r["kind"] == "batch"]
            init = sum(r["host_ms"] for r in selected if r["kind"] == "iterator_init")
            wait_ns = sum(r["end_ns"]-r["start_ns"] for r in selected if r["kind"] != "iterator_init")
            denominator_ns = epoch["train_end_ns"]-epoch["train_start_ns"]
            ratio = wait_ns/denominator_ns if phase == "training" and denominator_ns > 0 else None
            result.append(dict(epoch_id=epoch["epoch_id"], phase=phase, batches=len(batches),
                samples=sum(r["samples"] for r in batches), iterator_init_ms=init,
                next_wait_ms=wait_ns/1e6, delivered_wait=_distribution([r["host_ms"] for r in batches]),
                training_wall_wait_fraction=dict(value=ratio, unit="fraction", status="derived" if ratio is not None else "unavailable",
                    reason=None if ratio is not None else ("validation_not_training" if phase == "validation" else "nonpositive_training_duration"))))
    return result


def _links(rows, stages):
    by_micro = {}
    if stages is not None:
        for r in stages["observations"]:
            if r["scope"] == "microbatch":
                by_micro.setdefault(r["microbatch_id"], {})[r["stage"]] = r
    result = []
    for row in rows:
        if row["kind"] == "batch" and row["phase"] == "training":
            stages_for_batch = by_micro.get(row["microbatch_id"])
            result.append(dict(microbatch_id=row["microbatch_id"], epoch_id=row["epoch_id"],
                status="linked" if stages_for_batch is not None else "unsupported",
                reason=None if stages_for_batch is not None else "P20_not_requested",
                input_preparation=None if stages_for_batch is None else stages_for_batch["input_preparation"],
                compute=None if stages_for_batch is None else {k: stages_for_batch[k] for k in ("forward", "loss_metrics", "backward")}))
    return result


def _validate(data):
    required = {"schema_version", "format", "feature_ids", "protocol", "source_segment_id", "source_sha256",
                "source_p19", "source_p20", "source_p20_sha256", "provenance", "requested", "actual",
                "observations", "summaries", "stage_links", "worker_service", "queue_occupancy",
                "prefetch_efficiency", "h2d_transfer", "status", "reason", "limitations"}
    if set(data) != required or type(data["schema_version"]) is not int or data["schema_version"] != 1 or data["format"] != "hyso.performance.dataloader" or data["feature_ids"] != ["P21"]:
        raise ValueError("invalid P21 identity/schema")
    plan = DataLoaderAnalysisProtocol(**data["protocol"])
    source, digest = _source(TrainingPerformanceResult(data["source_p19"]))
    if data["source_sha256"] != digest or data["source_segment_id"] != source["segment_id"]:
        raise ValueError("invalid P19 source association")
    stages = data["source_p20"]
    if stages is not None:
        TrainingStageResult(stages)
        if stages["source_p19"] != source or _hash(stages) != data["source_p20_sha256"]:
            raise ValueError("invalid P20 source association")
    elif data["source_p20_sha256"] is not None:
        raise ValueError("missing P20 evidence")
    if data["provenance"]["training"] != source["provenance"] or set(data["provenance"]["loaders"]) != {"training", "validation"}:
        raise ValueError("invalid P21 provenance")
    configs = data["provenance"]["loaders"]
    if configs["training"] is None or (configs["validation"] is not None) != source["provenance"]["training"]["validation"]:
        raise ValueError("loader configuration phase mismatch")
    for config in configs.values():
        if config is None:
            continue
        _int(config["num_workers"])
        for name in ("pin_memory", "persistent_workers", "drop_last", "generator_present"):
            if type(config[name]) is not bool:
                raise ValueError("invalid loader configuration flag")
        if config["batch_size"] is not None:
            _int(config["batch_size"], 1)
        if config["num_workers"]:
            _int(config["prefetch_factor"], 1)
        elif config["prefetch_factor"] is not None or config["persistent_workers"]:
            raise ValueError("invalid zero-worker prefetch configuration")
        if type(config["timeout"]) not in (int, float) or not math.isfinite(config["timeout"]) or config["timeout"] < 0:
            raise ValueError("invalid loader timeout")
    if data["requested"] != dict(epochs=source["requested_epochs"]) or data["actual"] != dict(epochs=source["actual_epochs"]):
        raise ValueError("invalid P21 workload")
    if (data["status"], data["reason"]) != ("partial", "worker_service_trace_unsupported"):
        raise ValueError("P21 cannot claim worker service coverage")
    for name in ("worker_service", "queue_occupancy", "prefetch_efficiency"):
        if data[name] != dict(value=None, status="unsupported", reason="no_verified_worker_or_queue_trace"):
            raise ValueError("unmeasured worker/queue data must stay null")
    cuda = source["provenance"]["runtime"]["device"].startswith("cuda")
    if data["h2d_transfer"] != dict(value=None, status="unsupported" if cuda else "not_executed",
                                   reason="no_separate_H2D_clock" if cuda else "CPU_training"):
        raise ValueError("input preparation is not an isolated H2D measurement")
    rows = data["observations"]
    if len(rows) > plan.max_records:
        raise ValueError("P21 raw budget exceeded")
    epochs = {e["epoch_id"]: e for e in source["epochs"]}
    micro = {r["microbatch_id"]: r for r in source["observations"] if r["kind"] == "microbatch"}
    attempts = {r["window_id"]: r for r in source["observations"] if r["kind"] == "attempt"}
    grouped = {}
    observed_micro = []
    for row in rows:
        if set(row) != {"kind", "phase", "epoch_id", "batch_index", "start_ns", "end_ns", "host_ms", "delivery_ns", "samples", "microbatch_id", "window_id", "attempt_id"}:
            raise ValueError("invalid loader row fields")
        if row["kind"] not in ("iterator_init", "batch", "exhaustion") or row["phase"] not in ("training", "validation") or row["epoch_id"] not in epochs:
            raise ValueError("invalid loader row scope")
        _int(row["epoch_id"], 1)
        _duration(row, "start_ns", "end_ns", "host_ms", 1e6)
        e = epochs[row["epoch_id"]]
        start, end = ((e["train_start_ns"], e["train_end_ns"]) if row["phase"] == "training" else (e["validation_start_ns"], e["validation_end_ns"]))
        if start is None or not start <= row["start_ns"] <= row["end_ns"] <= end:
            raise ValueError("loader observation outside parent clock")
        grouped.setdefault((row["epoch_id"], row["phase"]), []).append(row)
        if row["kind"] == "batch":
            _int(row["samples"], 1); _int(row["batch_index"], 1)
            if row["delivery_ns"] != row["end_ns"]:
                raise ValueError("invalid delivery timestamp")
            if row["phase"] == "training":
                for key in ("microbatch_id", "window_id", "attempt_id"):
                    _int(row[key], 1)
                m = micro.get(row["microbatch_id"])
                a = attempts.get(row["window_id"])
                if m is None or a is None or row["samples"] != m["samples"] or row["epoch_id"] != m["epoch_id"] or row["window_id"] != m["window_id"] or row["attempt_id"] != a["attempt_id"] or row["delivery_ns"] > m["start_ns"]:
                    raise ValueError("P21 delivery disagrees with training ledger")
                observed_micro.append(row["microbatch_id"])
            elif any(row[k] is not None for k in ("microbatch_id", "window_id", "attempt_id")):
                raise ValueError("validation is not a training microbatch")
        elif any(row[k] is not None for k in ("samples", "delivery_ns", "batch_index", "microbatch_id", "window_id", "attempt_id")):
            raise ValueError("non-batch row has invented batch data")
    if observed_micro != list(micro):
        raise ValueError("missing/duplicate/out-of-order delivered microbatches")
    for e in source["epochs"]:
        for phase in ("training", "validation"):
            selected = grouped.get((e["epoch_id"], phase), [])
            if phase == "validation" and e["validation_s"] is None:
                if selected:
                    raise ValueError("unexpected validation rows")
                continue
            if len(selected) < 3 or selected[0]["kind"] != "iterator_init" or selected[-1]["kind"] != "exhaustion" or any(r["kind"] != "batch" for r in selected[1:-1]):
                raise ValueError("incomplete iterator lifecycle")
            if [r["batch_index"] for r in selected[1:-1]] != list(range(1, len(selected)-1)) or any(a["end_ns"] > b["start_ns"] for a, b in zip(selected, selected[1:])):
                raise ValueError("invalid iterator ordering")
    if data["summaries"] != _summaries(rows, source) or data["stage_links"] != _links(rows, stages):
        raise ValueError("P21 derived evidence mismatch")


@dataclass(frozen=True, init=False)
class DataLoaderAnalysisResult:
    """Immutable P21 raw host waits plus validated P19/P20 source association."""
    __slots__ = ("_payload",)
    _payload: str

    def __init__(self, data):
        data = _plain(data)
        self._check(data)
        object.__setattr__(self, "_payload", json.dumps(data, allow_nan=False, separators=(",", ":")))

    @staticmethod
    def _check(data):
        try:
            _validate(data)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("malformed P21 payload") from exc

    def to_dict(self):
        return json.loads(self._payload)

    def _export_data(self):
        data = _plain(self.to_dict()); self._check(data)
        return data

    def to_json(self, path, *, overwrite=False):
        _write_export(path, json.dumps(self._export_data(), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", overwrite)

    def to_csv(self, path, *, overwrite=False):
        data = self._export_data()
        columns = ["source_segment_id", "kind", "phase", "epoch_id", "batch_index", "microbatch_id", "window_id", "attempt_id", "samples", "start_ns", "end_ns", "delivery_ns", "host_ms"]
        _write_export(path, _csv_text(columns, [{k: data["source_segment_id"] if k == "source_segment_id" else row[k] for k in columns} for row in data["observations"]]), overwrite)


@dataclass(frozen=True)
class LoaderTrainingResult:
    timing: TrainingPerformanceResult
    loader: DataLoaderAnalysisResult
    stages: TrainingStageResult | None
    throughput: TrainingThroughputResult


def measure_training_with_loader(trainer, train_loader, val_loader=None, *, epochs=5,
        protocol=None, loader_protocol=None, stages=True, stage_protocol=None,
        save_last_to=None, resume_from=None):
    """One actual fit; main wait observed, optional P20, P22 derived, no worker trace.

    DataLoader configuration/ownership stays caller-owned. No strict multiworker
    checkpoint/resume support is added. Worker service/queue metrics stay null.
    """
    loader_protocol = DataLoaderAnalysisProtocol() if loader_protocol is None else loader_protocol
    if type(loader_protocol) is not DataLoaderAnalysisProtocol or type(stages) is not bool:
        raise TypeError("invalid loader protocol/stages option")
    if stage_protocol is not None and not stages:
        raise ValueError("stage_protocol requires stages=True")
    stage_protocol = TrainingStageProtocol() if stage_protocol is None else stage_protocol
    if type(stage_protocol) is not TrainingStageProtocol:
        raise TypeError("stage_protocol must be TrainingStageProtocol")
    configs = dict(training=_configuration(train_loader), validation=_configuration(val_loader))
    captured = []
    def factory(t, p):
        recorder = (_LoaderStageRecorder(t, p, loader_protocol, stage_protocol) if stages else
                    _LoaderRecorder(t, p, loader_protocol))
        captured.append(recorder)
        return recorder
    timing = _measure_training(trainer, train_loader, val_loader, epochs=epochs, protocol=protocol,
        save_last_to=save_last_to, resume_from=resume_from, recorder_factory=factory)
    recorder = captured[0]
    stage_result = _stage_result(timing, recorder, trainer.grad_clip) if stages else None
    source, digest = _source(timing)
    stage_data = None if stage_result is None else stage_result.to_dict()
    data = dict(schema_version=1, format="hyso.performance.dataloader", feature_ids=["P21"],
        protocol=asdict(loader_protocol), source_segment_id=source["segment_id"], source_sha256=digest,
        source_p19=source, source_p20=stage_data, source_p20_sha256=None if stage_data is None else _hash(stage_data),
        provenance=dict(training=source["provenance"], loaders=configs), requested=dict(epochs=epochs), actual=dict(epochs=source["actual_epochs"]),
        observations=recorder.loader_rows, summaries=_summaries(recorder.loader_rows, source), stage_links=_links(recorder.loader_rows, stage_data),
        status="partial", reason="worker_service_trace_unsupported",
        h2d_transfer=dict(value=None, status="unsupported" if trainer.device.type == "cuda" else "not_executed",
                          reason="no_separate_H2D_clock" if trainer.device.type == "cuda" else "CPU_training"),
        limitations=["host next wait is not worker decode/transform/service time", "queue occupancy and prefetch efficiency unmeasured",
                     "instrumented scopes include observer overhead; stage intervals must not be summed",
                     "no strict multiworker resume or hidden-stream/kernel attribution"],
        **{name: dict(value=None, status="unsupported", reason="no_verified_worker_or_queue_trace") for name in ("worker_service", "queue_occupancy", "prefetch_efficiency")})
    return LoaderTrainingResult(timing, DataLoaderAnalysisResult(data), stage_result, analyze_training_throughput(timing))
