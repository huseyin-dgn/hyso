"""P19: opt-in observation of one real Trainer fit, never a training replay.

Host scopes include instrumentation. CUDA intervals belong only to the stream
selected on entry; hidden streams, kernel attribution and stage breakdown are
unsupported. No file/database access occurs in the recorder.
"""
from dataclasses import asdict, dataclass
import json
import math
import platform
import time
import uuid

import torch

from hyso.benchmark.result import _plain, _csv_text, _write_export


@dataclass(frozen=True)
class TrainingTimingProtocol:
    """Bound all retained rows and pending event pairs; no warmup/replay work.

    warmup_epochs labels real initial epochs without discarding their ledger.
    max_records is a hard segment limit, not a lossy sampling policy.
    """
    max_records: int = 100000
    event_buffer: int = 128
    warmup_epochs: int = 0

    def __post_init__(self):
        for name in ("max_records", "event_buffer", "warmup_epochs"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "warmup_epochs" else 1):
                raise ValueError(name + " has an invalid integer budget")
        if self.max_records > 1000000 or self.event_buffer > 4096:
            raise ValueError("record/event budget exceeds supported hard ceiling")


class TrainingTimingError(RuntimeError):
    """Observation failed. Training may have advanced; never retry implicitly."""


def _type(value):
    return type(value).__module__ + "." + type(value).__qualname__


class _Recorder:
    def __init__(self, trainer, protocol):
        self.trainer, self.protocol = trainer, protocol
        self.rows, self.epochs, self.pending, self.active = [], [], [], {}
        self.stream = (torch.cuda.current_stream(trainer.device)
                       if trainer.device.type == "cuda" else None)
        self.drains = self.max_pending = 0
        self.input = None
        self.epoch = None
        self.window_id = 0
        self._epoch_row_start = 0

    def _check_stream(self):
        if self.stream is not None and torch.cuda.current_stream(self.trainer.device) != self.stream:
            raise TrainingTimingError("selected CUDA stream changed during fit")

    def _event(self):
        self._check_stream()
        if self.stream is None:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record(self.stream)
        return event

    def _begin(self, kind, **fields):
        if len(self.rows) + len(self.active) + len(self.epochs) + 1 >= self.protocol.max_records:
            raise TrainingTimingError("max_records exceeded; incomplete timing is not a result")
        self.active[kind] = (dict(kind=kind, epoch_id=self.epoch["epoch_id"],
                                  start_ns=time.perf_counter_ns(), **fields), self._event())

    def _end(self, kind, **fields):
        row, start = self.active.pop(kind)
        end = self._event()
        row.update(end_ns=time.perf_counter_ns(), **fields)
        row["host_ms"] = (row["end_ns"] - row["start_ns"]) / 1e6
        row["cuda_event_ms"] = None
        self.rows.append(row)
        if start is not None:
            self.pending.append((row, start, end))
            self.max_pending = max(self.max_pending, len(self.pending))
            if len(self.pending) >= self.protocol.event_buffer:
                self.drain()
        return row

    def drain(self):
        if self.pending:
            # Synchronize this stream's last closed event, never device-wide.
            self.pending[-1][2].synchronize()
            for row, start, end in self.pending:
                if not end.query():
                    raise TrainingTimingError("CUDA interval is not ready")
                row["cuda_event_ms"] = float(start.elapsed_time(end))
            self.pending.clear()
            self.drains += 1

    def epoch_begin(self, epoch_id):
        self._check_stream()
        if self.stream is not None:
            self.stream.synchronize()
        if len(self.rows) + len(self.epochs) >= self.protocol.max_records:
            raise TrainingTimingError("max_records exceeded")
        self.epoch = dict(epoch_id=epoch_id, start_ns=time.perf_counter_ns(),
                          warmup=len(self.epochs) < self.protocol.warmup_epochs)
        self._epoch_row_start = len(self.rows)
        self.epoch["train_start_ns"] = self.epoch["start_ns"]

    def microbatch_begin(self, microbatch_id, new_window):
        if new_window:
            self.window_id += 1
            self._begin("window", window_id=self.window_id)
        self._begin("microbatch", microbatch_id=microbatch_id, window_id=self.window_id)

    def microbatch_end(self, samples, inputs, targets):
        self._end("microbatch", samples=samples)
        if self.input is None:
            self.input = dict(shape=list(inputs.shape), dtype=str(inputs.dtype),
                              target_dtype=str(targets.dtype), device=str(inputs.device),
                              scope="first_actual_microbatch; ledger records varying batch sizes")

    def attempt_begin(self, attempt_id, updates):
        self._begin("attempt", attempt_id=attempt_id, window_id=self.window_id,
                    update_before=updates)

    def attempt_end(self, updates, samples):
        before = self.active["attempt"][0]["update_before"]
        self._end("attempt", update_after=updates, applied=updates == before + 1,
                  skipped=updates == before, applied_update_id=updates if updates > before else None,
                  samples=samples)
        self._end("window", samples=samples)

    def training_end(self, validation):
        self.drain()
        if self.stream is not None:
            self.stream.synchronize()
        self.epoch["train_end_ns"] = time.perf_counter_ns()
        self.epoch["validation_start_ns"] = time.perf_counter_ns() if validation else None

    def validation_end(self, validation):
        if validation and self.stream is not None:
            self._check_stream()
            self.stream.synchronize()
        self.epoch["validation_end_ns"] = time.perf_counter_ns() if validation else None

    def epoch_end(self):
        e = self.epoch
        e["end_ns"] = time.perf_counter_ns()
        e["training_s"] = (e["train_end_ns"] - e["train_start_ns"]) / 1e9
        e["validation_s"] = (None if e["validation_start_ns"] is None else
                             (e["validation_end_ns"] - e["validation_start_ns"]) / 1e9)
        e["total_s"] = (e["end_ns"] - e["start_ns"]) / 1e9
        rows = self.rows[self._epoch_row_start:]
        micro = [r for r in rows if r["kind"] == "microbatch"]
        attempts = [r for r in rows if r["kind"] == "attempt"]
        e.update(samples=sum(r["samples"] for r in micro), microbatches=len(micro),
                 windows=sum(r["kind"] == "window" for r in rows), attempts=len(attempts),
                 applied=sum(r["applied"] for r in attempts), skipped=sum(r["skipped"] for r in attempts))
        self.epochs.append(e)
        self.epoch = None

    def close(self):
        try:
            # Also complete open intervals on workload/recorder failure.
            if self.stream is not None:
                self.stream.synchronize()
            self.drain()
        finally:
            self.pending.clear()
            self.active.clear()


def _int(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid training timing count/timestamp")


def _duration(row, start, end, field, scale):
    _int(row[start]); _int(row[end])
    if type(row[field]) not in (int, float) or not math.isfinite(row[field]) or row[end] < row[start] or row[field] != (row[end] - row[start]) / scale:
        raise ValueError("inconsistent timing arithmetic/order")


def _validate(data):
    expected = {"schema_version", "feature_ids", "format", "segment_id", "protocol", "provenance",
                "requested_epochs", "actual_epochs", "epochs", "observations", "counters",
                "status", "reason", "limitations", "instrumentation"}
    if set(data) != expected or type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("invalid training result schema")
    if data["feature_ids"] != ["P19"] or data["format"] != "hyso.performance.training_timing":
        raise ValueError("invalid training feature identity")
    uuid.UUID(data["segment_id"])
    protocol = TrainingTimingProtocol(**data["protocol"])
    _int(data["requested_epochs"], 1); _int(data["actual_epochs"])
    if data["status"] != ("measured" if data["actual_epochs"] else "no_new_epochs") or data["reason"] != (None if data["actual_epochs"] else "completed_checkpoint"):
        raise ValueError("invalid training result status")
    if len(data["epochs"]) != data["actual_epochs"] or len(data["observations"]) + len(data["epochs"]) > protocol.max_records:
        raise ValueError("invalid epoch/record ledger")
    p = data["provenance"]
    if set(p) != {"model", "input", "optimizer", "runtime", "training"} or not all(type(p[k]) is dict and p[k] for k in p):
        raise ValueError("missing training provenance")
    cuda = p["runtime"]["device"].startswith("cuda")
    ins = data["instrumentation"]
    if ins["host_clock"] != "perf_counter_ns" or ins["device_clock"] != ("cuda_event_selected_stream" if cuda else None):
        raise ValueError("invalid clock domain")
    if ins["max_pending_event_pairs"] > protocol.event_buffer or ins["pending_event_pairs"] != 0:
        raise ValueError("unbounded/incomplete events")
    rows = data["observations"]
    by_epoch = {}
    for row in rows:
        by_epoch.setdefault(row["epoch_id"], []).append(row)
    all_ids = {"microbatch": [], "attempt": [], "window": []}
    previous_end = 0
    update_cursor = data["counters"]["before"]["applied"]
    for index, e in enumerate(data["epochs"]):
        _int(e["epoch_id"], 1)
        if e["epoch_id"] > data["requested_epochs"] or (index and e["epoch_id"] != data["epochs"][index - 1]["epoch_id"] + 1):
            raise ValueError("invalid epoch IDs")
        _duration(e, "train_start_ns", "train_end_ns", "training_s", 1e9)
        _duration(e, "start_ns", "end_ns", "total_s", 1e9)
        if not previous_end <= e["start_ns"] == e["train_start_ns"] <= e["train_end_ns"] <= e["end_ns"] or type(e["warmup"]) is not bool or e["warmup"] != (index < protocol.warmup_epochs):
            raise ValueError("invalid epoch boundaries/warmup")
        previous_end = e["end_ns"]
        if e["validation_s"] is not None:
            _duration(e, "validation_start_ns", "validation_end_ns", "validation_s", 1e9)
            if not e["train_end_ns"] <= e["validation_start_ns"] <= e["validation_end_ns"] <= e["end_ns"]:
                raise ValueError("invalid validation boundaries")
        elif e["validation_start_ns"] is not None or e["validation_end_ns"] is not None:
            raise ValueError("invalid absent validation")
        selected = by_epoch.pop(e["epoch_id"], [])
        grouped = {k: [r for r in selected if r["kind"] == k] for k in all_ids}
        for r in selected:
            if r["kind"] not in all_ids:
                raise ValueError("invalid observation kind")
            _duration(r, "start_ns", "end_ns", "host_ms", 1e6)
            if not e["start_ns"] <= r["start_ns"] <= r["end_ns"] <= e["train_end_ns"]:
                raise ValueError("observation outside training scope")
            _int(r["samples"], 1)
            value = r["cuda_event_ms"]
            if cuda:
                if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                    raise ValueError("missing/invalid completed device interval")
            elif value is not None:
                raise ValueError("CPU wall time is not CUDA device time")
            key = {"microbatch": "microbatch_id", "attempt": "attempt_id", "window": "window_id"}[r["kind"]]
            _int(r[key], 1)
            all_ids[r["kind"]].append(r[key])
        micro, windows, attempts = (grouped[k] for k in ("microbatch", "window", "attempt"))
        if len(windows) != len(attempts):
            raise ValueError("window/attempt mismatch")
        if sum(w["samples"] for w in windows) != sum(r["samples"] for r in micro):
            raise ValueError("unassigned microbatch samples")
        if any(a["end_ns"] > b["start_ns"] for a, b in zip(micro, micro[1:])):
            raise ValueError("overlapping microbatch host intervals")
        by_window = {}
        for row in micro:
            by_window.setdefault(row["window_id"], []).append(row)
        for w, a in zip(windows, attempts):
            batch = by_window.pop(w["window_id"], [])
            if not batch or a["window_id"] != w["window_id"] or w["samples"] != a["samples"] or w["samples"] != sum(r["samples"] for r in batch):
                raise ValueError("invalid sample/window ledger")
            if not w["start_ns"] <= batch[0]["start_ns"] <= batch[-1]["end_ns"] <= a["start_ns"] <= a["end_ns"] <= w["end_ns"]:
                raise ValueError("invalid accumulation ordering")
            delta = a["update_after"] - a["update_before"]
            _int(a["update_before"]); _int(a["update_after"])
            if a["update_before"] != update_cursor:
                raise ValueError("noncontiguous optimizer update ledger")
            update_cursor = a["update_after"]
            if delta not in (0, 1) or type(a["applied"]) is not bool or type(a["skipped"]) is not bool or a["applied"] != bool(delta) or a["skipped"] == bool(delta) or a["applied_update_id"] != (a["update_after"] if delta else None):
                raise ValueError("invalid applied/skipped ownership")
        if by_window:
            raise ValueError("orphan microbatch window IDs")
        counts = dict(samples=sum(r["samples"] for r in micro), microbatches=len(micro),
                      windows=len(windows), attempts=len(attempts), applied=sum(r["applied"] for r in attempts),
                      skipped=sum(r["skipped"] for r in attempts))
        if any(e[k] != v for k, v in counts.items()):
            raise ValueError("inconsistent epoch counts")
        for k in counts:
            _int(e[k])
    for ids in all_ids.values():
        if ids and ids != list(range(ids[0], ids[0] + len(ids))):
            raise ValueError("duplicate/missing observation IDs")
    for kind, counter in (("microbatch", "microbatches"), ("attempt", "attempts")):
        ids = all_ids[kind]
        before = data["counters"]["before"][counter]
        if ids and ids[0] != before + 1:
            raise ValueError("segment IDs disagree with Trainer counters")
    if by_epoch:
        raise ValueError("orphan observations")
    for key in ("microbatches", "attempts", "applied", "ema_updates"):
        before, after = data["counters"]["before"][key], data["counters"]["after"][key]
        _int(before); _int(after)
        expected_delta = sum(e[key] for e in data["epochs"]) if key != "ema_updates" else (sum(e["applied"] for e in data["epochs"]) if p["training"]["ema"] else 0)
        if after - before != expected_delta:
            raise ValueError("Trainer counters disagree with ledger")


@dataclass(frozen=True, init=False)
class TrainingPerformanceResult:
    """Immutable JSON-backed P19 evidence; exports revalidate against tampering."""
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

    def to_json(self, path, *, overwrite=False):
        _write_export(path, json.dumps(self._export_data(), ensure_ascii=False,
                                      sort_keys=True, indent=2, allow_nan=False) + "\n", overwrite)

    def to_csv(self, path, *, overwrite=False):
        """Raw observations; canonical JSON additionally contains epoch/provenance."""
        data = self._export_data()
        columns = ["segment_id", "epoch_id", "kind", "microbatch_id", "window_id", "attempt_id",
                   "samples", "start_ns", "end_ns", "host_ms", "cuda_event_ms", "applied", "skipped", "applied_update_id"]
        _write_export(path, _csv_text(columns, [{k: data["segment_id"] if k == "segment_id" else r.get(k)
                                               for k in columns} for r in data["observations"]]), overwrite)


def _counters(trainer):
    return dict(microbatches=trainer.microbatch_count, attempts=trainer.attempt_count,
                applied=trainer.update_count, ema_updates=trainer.ema_update_count)


def measure_training(trainer, train_loader, val_loader=None, *, epochs=5, protocol=None,
                     save_last_to=None, resume_from=None):
    """Run the caller's actual fit once and return P19 timing (history on Trainer).

    Training changes model/state/RNG normally. Tracking/export is explicit and
    subsequent; tracking errors never trigger a rerun. Checkpoint I/O, restore,
    train_begin/train_end callbacks and best-weight finalization are excluded.
    """
    from hyso.training import Trainer
    if type(trainer) is not Trainer:
        raise TypeError("measure_training requires the standard Trainer")
    if trainer._epochs is not None or hasattr(trainer, "_timing_recorder"):
        raise ValueError("measure_training requires a fresh, unattached Trainer")
    protocol = TrainingTimingProtocol() if protocol is None else protocol
    if type(protocol) is not TrainingTimingProtocol:
        raise TypeError("protocol must be TrainingTimingProtocol")
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive Python int")
    # Metadata reads never iterate loaders or consume a random sample.
    provenance = dict(model=dict(type=_type(trainer.model), config=trainer.model_config,
                                modules=[dict(name=name, type=_type(module))
                                         for name, module in trainer.model.named_modules()],
                                tensors=[dict(name=name, shape=list(t.shape), dtype=str(t.dtype), device=str(t.device))
                                         for name, t in list(trainer.model.named_parameters()) + list(trainer.model.named_buffers())]),
                      optimizer=dict(type=_type(trainer.optimizer),
                                     groups=[{k: v for k, v in g.items() if k != "params"} for g in trainer.optimizer.param_groups]),
                      runtime=dict(python=platform.python_version(), torch=str(torch.__version__),
                                   platform=platform.platform(), device=str(trainer.device), cuda_build=torch.version.cuda,
                                   cpu_threads=torch.get_num_threads(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                                   cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic),
                      training=dict(amp=trainer.amp, accum_steps=trainer.accum_steps,
                                    ema=trainer.ema is not None, scheduler=_type(trainer.scheduler),
                                    loss_type=_type(trainer.loss_fn), task=trainer.task,
                                    validation=val_loader is not None, loader_type=_type(train_loader)))
    from hyso.training.resume import _plain as _configuration
    provenance = _plain(_configuration(provenance))
    recorder = _Recorder(trainer, protocol)
    before = _counters(trainer)
    trainer._timing_recorder = recorder
    try:
        trainer.fit(train_loader, val_loader, epochs=epochs, save_last_to=save_last_to,
                    resume_from=resume_from)
    finally:
        del trainer._timing_recorder
        recorder.close()
    # On resume the checkpoint supplies the baseline counters; derive it from
    # observed deltas, never manufacture timings for the checkpoint prefix.
    after = _counters(trainer)
    if resume_from is not None:
        before = {k: after[k] - sum(e[k if k != "ema_updates" else "applied"]
                  for e in recorder.epochs) * (int(trainer.ema is not None) if k == "ema_updates" else 1)
                  for k in after}
    provenance["input"] = recorder.input or dict(status="no_new_microbatch")
    data = dict(schema_version=1, feature_ids=["P19"], format="hyso.performance.training_timing",
                segment_id=str(uuid.uuid4()), protocol=asdict(protocol), provenance=provenance,
                requested_epochs=epochs, actual_epochs=len(recorder.epochs), epochs=recorder.epochs,
                observations=recorder.rows, counters=dict(before=before, after=after),
                status="measured" if recorder.epochs else "no_new_epochs",
                reason=None if recorder.epochs else "completed_checkpoint",
                limitations=["instrumented wall time includes timer/event/drain overhead",
                             "no P20 stage decomposition or kernel attribution",
                             "hidden streams and concurrent process attribution unsupported",
                             "checkpoint I/O and fit finalization excluded; no cumulative resume timing"],
                instrumentation=dict(host_clock="perf_counter_ns",
                    device_clock="cuda_event_selected_stream" if recorder.stream is not None else None,
                    stream_id=int(recorder.stream.cuda_stream) if recorder.stream is not None else None,
                    event_drains=recorder.drains, max_pending_event_pairs=recorder.max_pending,
                    pending_event_pairs=len(recorder.pending), warmup_policy="label_real_epochs_no_extra_work",
                    validation_policy="separate_completion_wall_included_in_total_epoch",
                    event_policy="bounded_closed_pairs_plus_at_most_three_open_pairs; chunk_and_epoch_drain"))
    return TrainingPerformanceResult(data)
