"""P20 stage observation and P22 pure analysis, additive to P19 schema1."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

from hyso.benchmark.result import _plain, _csv_text, _write_export
from .training import (_Recorder, TrainingTimingError, TrainingTimingProtocol,
                       TrainingPerformanceResult, _measure_training, _int, _duration)


MICRO_STAGES = ("input_preparation", "forward", "loss_metrics", "backward")
ATTEMPT_STAGES = ("unscale", "gradient_check_normalization", "gradient_clipping",
                  "scaler_step_attempt", "optimizer_execution", "scaler_update",
                  "ema_update", "onecycle_scheduler", "zero_grad")
STAGES = MICRO_STAGES + ATTEMPT_STAGES + ("epoch_scheduler",)


@dataclass(frozen=True)
class TrainingStageProtocol:
    """Hard combined P19+P20 raw-record budget; CUDA queue uses P19 event_buffer."""
    max_records: int = 100000

    def __post_init__(self):
        _int(self.max_records, 1)
        if self.max_records > 1000000:
            raise ValueError("stage record ceiling is 1000000")


class _StageRecorder(_Recorder):
    def __init__(self, trainer, protocol, stage_protocol):
        super().__init__(trainer, protocol)
        self.stage_protocol = stage_protocol
        self.stages = []
        self.handles = []

    def _reserve(self):
        if len(self.rows) + len(self.stages) + len(self.active) + len(self.epochs) + 1 >= self.stage_protocol.max_records:
            raise TrainingTimingError("combined stage record buffer exhausted")

    def _begin(self, kind, **fields):
        self._reserve()
        super()._begin(kind, **fields)

    def _context(self, name):
        if "microbatch" in self.active:
            parent = self.active["microbatch"][0]
            return dict(scope="microbatch", parent_id=parent["microbatch_id"],
                        microbatch_id=parent["microbatch_id"], window_id=parent["window_id"], attempt_id=None)
        if "attempt" in self.active:
            parent = self.active["attempt"][0]
            return dict(scope="attempt", parent_id=parent["attempt_id"], microbatch_id=None,
                        window_id=parent["window_id"], attempt_id=parent["attempt_id"])
        return dict(scope="epoch_total" if name == "epoch_scheduler" else "epoch_training",
                    parent_id=self.epoch["epoch_id"], microbatch_id=None, window_id=None, attempt_id=None)

    def stage_begin(self, name):
        self._begin("stage:" + name, stage=name, **self._context(name))
        if name == "scaler_step_attempt":
            try:
                self.handles.append(self.trainer.optimizer.register_step_pre_hook(self._optimizer_pre))
                self.handles.append(self.trainer.optimizer.register_step_post_hook(self._optimizer_post))
            except BaseException:
                self.remove_optimizer_hooks()
                raise

    def _optimizer_pre(self, optimizer, args, kwargs):
        self.stage_begin("optimizer_execution")

    def _optimizer_post(self, optimizer, args, kwargs):
        self.stage_end("optimizer_execution")

    def remove_optimizer_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def stage_end(self, name):
        row = super()._end("stage:" + name)
        # Shared pending queue retains this same row until CUDA readout. P19's
        # own ledger stays schema1 and contains no new observation kinds.
        self.rows.pop()
        row.update(kind="stage", status="measured", reason=None)
        self.stages.append(row)
        if name == "scaler_step_attempt":
            self.remove_optimizer_hooks()
            attempt = self.active["attempt"][0]["attempt_id"]
            if not any(r["stage"] == "optimizer_execution" and r["attempt_id"] == attempt
                       for r in self.stages[-2:]):
                self.not_executed("optimizer_execution", "native_scaler_skipped_optimizer")

    def not_executed(self, name, reason):
        self._reserve()
        self.stages.append(dict(kind="stage", stage=name, epoch_id=self.epoch["epoch_id"],
            **self._context(name), start_ns=None, end_ns=None, host_ms=None, cuda_event_ms=None,
            status="not_executed", reason=reason))

    def zero_grad(self, optimizer):
        # Preserve the Trainer's unconditional cleanup even if the recorder fails.
        try:
            self.stage_begin("zero_grad")
        finally:
            optimizer.zero_grad()
        self.stage_end("zero_grad")

    def epoch_end(self):
        # Epoch scheduler events are admitted after training_end's terminal drain.
        self.drain()
        super().epoch_end()

    def close(self):
        try:
            self.remove_optimizer_hooks()
        finally:
            super().close()


def _source(result):
    if type(result) is not TrainingPerformanceResult:
        raise TypeError("source must be a TrainingPerformanceResult")
    data = result._export_data()
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode("utf-8")).hexdigest()
    return data, digest


def _base(source, kind, feature, protocol):
    data, digest = _source(source)
    return dict(schema_version=1, format="hyso.performance." + kind, feature_ids=[feature],
                source_segment_id=data["segment_id"], source_sha256=digest, source_p19=data,
                provenance=data["provenance"], protocol=protocol,
                requested=dict(epochs=data["requested_epochs"]), actual=dict(epochs=data["actual_epochs"]),
                limitations=["instrumented clocks include observation and bounded drain overhead",
                    "no CUPTI kernel attribution or hidden-stream support",
                    "host and CUDA event intervals are separate; inclusive scopes must not be summed"],
                status="measured" if data["actual_epochs"] else "no_new_epochs",
                reason=None if data["actual_epochs"] else "completed_checkpoint")


def _validate_base(data, kind, feature):
    required = {"schema_version", "format", "feature_ids", "source_segment_id", "source_sha256", "source_p19",
                "provenance", "protocol", "requested", "actual", "limitations", "status", "reason"}
    required |= {"observations", "summary", "configuration"} if feature == "P20" else {"epochs", "segment"}
    if set(data) != required:
        raise ValueError("invalid training extension fields")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1 or data["format"] != "hyso.performance." + kind or data["feature_ids"] != [feature]:
        raise ValueError("invalid training extension identity")
    source, digest = _source(TrainingPerformanceResult(data["source_p19"]))
    if data["source_sha256"] != digest or data["source_segment_id"] != source["segment_id"] or data["provenance"] != source["provenance"]:
        raise ValueError("source association/provenance mismatch")
    if data["requested"] != dict(epochs=source["requested_epochs"]) or data["actual"] != dict(epochs=source["actual_epochs"]):
        raise ValueError("invalid extension workload")
    if data["status"] != source["status"] or data["reason"] != source["reason"]:
        raise ValueError("invalid extension outcome")
    return source


def _stage_summary(rows):
    return [dict(stage=name, measured=sum(r["status"] == "measured" for r in rows if r["stage"] == name),
                 not_executed=sum(r["status"] == "not_executed" for r in rows if r["stage"] == name),
                 host_ms=sum(r["host_ms"] for r in rows if r["stage"] == name and r["status"] == "measured")
                    if any(r["stage"] == name and r["status"] == "measured" for r in rows) else None,
                 unit="ms", aggregation="inclusive_per_stage_no_cross_stage_sum") for name in STAGES]


def _validate_stages(data):
    source = _validate_base(data, "training_stages", "P20")
    plan = TrainingStageProtocol(**data["protocol"])
    if set(data["configuration"]) != {"grad_clip"}:
        raise ValueError("invalid stage configuration")
    clip = data["configuration"]["grad_clip"]
    if clip is not None and (type(clip) not in (int, float) or not math.isfinite(clip) or clip <= 0):
        raise ValueError("invalid clipping configuration")
    rows = data["observations"]
    if len(rows) + len(source["observations"]) + len(source["epochs"]) > plan.max_records:
        raise ValueError("combined record budget exceeded")
    parents = {(r["epoch_id"], r["kind"], r.get(r["kind"] + "_id")): r for r in source["observations"]}
    epochs = {e["epoch_id"]: e for e in source["epochs"]}
    seen = set()
    cuda = source["provenance"]["runtime"]["device"].startswith("cuda")
    for row in rows:
        if row["stage"] not in STAGES or row["kind"] != "stage":
            raise ValueError("unknown training stage")
        key = row["epoch_id"], row["scope"], row["parent_id"], row["stage"]
        if key in seen:
            raise ValueError("duplicate stage")
        seen.add(key)
        if row["scope"] in ("microbatch", "attempt"):
            parent = parents.get((row["epoch_id"], row["scope"], row["parent_id"]))
            if parent is None or row["window_id"] != parent["window_id"]:
                raise ValueError("invalid stage parent/window")
            if row["scope"] == "microbatch":
                if row["microbatch_id"] != row["parent_id"] or row["attempt_id"] is not None or row["stage"] not in MICRO_STAGES:
                    raise ValueError("invalid microbatch stage association")
            elif row["attempt_id"] != row["parent_id"] or row["microbatch_id"] is not None or row["stage"] not in ATTEMPT_STAGES:
                raise ValueError("invalid attempt stage association")
            start, end = parent["start_ns"], parent["end_ns"]
        else:
            if row["epoch_id"] not in epochs or row["parent_id"] != row["epoch_id"] or any(row[k] is not None for k in ("microbatch_id", "attempt_id", "window_id")):
                raise ValueError("invalid epoch stage association")
            parent = epochs[row["epoch_id"]]
            if (row["scope"], row["stage"]) not in (("epoch_training", "zero_grad"), ("epoch_total", "epoch_scheduler")):
                raise ValueError("invalid epoch stage")
            start, end = parent["start_ns"], parent["end_ns"] if row["scope"] == "epoch_total" else parent["train_end_ns"]
        if row["status"] == "measured":
            _duration(row, "start_ns", "end_ns", "host_ms", 1e6)
            if not start <= row["start_ns"] <= row["end_ns"] <= end or row["reason"] is not None:
                raise ValueError("stage outside parent scope")
            event = row["cuda_event_ms"]
            if cuda:
                if type(event) not in (float, int) or not math.isfinite(event) or event < 0:
                    raise ValueError("incomplete stage event")
            elif event is not None:
                raise ValueError("CPU interval is not device time")
        elif row["status"] == "not_executed":
            if not row["reason"] or any(row[k] is not None for k in ("start_ns", "end_ns", "host_ms", "cuda_event_ms")):
                raise ValueError("not executed is null, not zero")
        else:
            raise ValueError("invalid stage status")
    by_parent = {}
    for r in rows:
        by_parent.setdefault((r["epoch_id"], r["scope"], r["parent_id"]), {})[r["stage"]] = r
    for parent in source["observations"]:
        if parent["kind"] not in ("microbatch", "attempt"):
            continue
        kind = parent["kind"]
        stages = by_parent.get((parent["epoch_id"], kind, parent[kind + "_id"]), {})
        required = MICRO_STAGES if kind == "microbatch" else ATTEMPT_STAGES
        if set(stages) != set(required):
            raise ValueError("missing stage coverage")
        mandatory = MICRO_STAGES if kind == "microbatch" else ("unscale", "gradient_check_normalization", "scaler_step_attempt", "scaler_update", "zero_grad")
        if any(stages[k]["status"] != "measured" for k in mandatory):
            raise ValueError("mandatory stage not measured")
        order = MICRO_STAGES if kind == "microbatch" else tuple(k for k in ATTEMPT_STAGES if k != "optimizer_execution")
        executed = [stages[k] for k in order if stages[k]["status"] == "measured"]
        if any(a["end_ns"] > b["start_ns"] for a, b in zip(executed, executed[1:])):
            raise ValueError("non-nested stages overlap or execute out of order")
        if kind == "attempt":
            if (stages["gradient_clipping"]["status"] == "measured") != (clip is not None and parent["applied"]):
                raise ValueError("gradient clipping execution disagrees with configuration/overflow")
            if (stages["optimizer_execution"]["status"] == "measured") != parent["applied"]:
                raise ValueError("optimizer execution disagrees with applied ownership")
            if stages["optimizer_execution"]["status"] == "measured":
                a, b = stages["scaler_step_attempt"], stages["optimizer_execution"]
                if not a["start_ns"] <= b["start_ns"] <= b["end_ns"] <= a["end_ns"]:
                    raise ValueError("optimizer execution not nested in scaler attempt")
            for name, expected in (("ema_update", source["provenance"]["training"]["ema"] and parent["applied"]),
                                   ("onecycle_scheduler", source["provenance"]["training"]["scheduler"].endswith(".OneCycleLR") and parent["applied"])):
                if (stages[name]["status"] == "measured") != expected:
                    raise ValueError("EMA/scheduler stage ownership mismatch")
    for e in source["epochs"]:
        for scope, name in (("epoch_training", "zero_grad"), ("epoch_total", "epoch_scheduler")):
            if name not in by_parent.get((e["epoch_id"], scope, e["epoch_id"]), {}):
                raise ValueError("missing epoch boundary stage")
            expected = name == "zero_grad" or source["provenance"]["training"]["scheduler"].rsplit(".", 1)[-1] in ("StepLR", "CosineAnnealingLR", "ReduceLROnPlateau")
            if (by_parent[e["epoch_id"], scope, e["epoch_id"]][name]["status"] == "measured") != expected:
                raise ValueError("epoch stage execution mismatch")
    if data["summary"] != _stage_summary(rows):
        raise ValueError("stage summary disagrees with raw observations")


COUNTS = ("samples", "microbatches", "windows", "attempts", "applied", "skipped")
RATES = {"samples_per_second": "samples", "microbatches_per_second": "microbatches",
         "attempts_per_second": "attempts", "applied_updates_per_second": "applied"}


def _rates(epochs):
    counts = {k: sum(e[k] for e in epochs) for k in COUNTS}
    duration = sum(e["training_s"] for e in epochs)
    status = "measured" if epochs and math.isfinite(duration) and duration > 0 else "unavailable"
    reason = None if status == "measured" else ("no_new_epochs" if not epochs else "nonpositive_training_duration")
    rates = {}
    for name, count in RATES.items():
        value = counts[count] / duration if status == "measured" else None
        good = value is None or math.isfinite(value)
        rates[name] = dict(value=value if good else None, unit=count + "/s", status=status if good else "unavailable",
                           reason=reason if good else "nonfinite_rate")
    return dict(completed=counts, training_wall_seconds=duration, rates=rates,
                status=status, reason=reason)


def _throughput_evidence(source):
    # P19 construction fully revalidates raw sample/window/attempt/counter ledger.
    return [dict(epoch_id=e["epoch_id"], warmup=e["warmup"], **_rates([e])) for e in source["epochs"]], _rates(source["epochs"])


def _validate_throughput(data):
    source = _validate_base(data, "training_throughput", "P22")
    if data["protocol"] != dict(version=1, denominator="sum_completed_epoch_training_wall_seconds",
                                validation="excluded", warmup="included_and_labeled", aggregation="ratio_of_sums"):
        raise ValueError("invalid throughput protocol")
    epochs, segment = _throughput_evidence(source)
    if data["epochs"] != epochs or data["segment"] != segment:
        raise ValueError("throughput arithmetic/ledger disagreement")


@dataclass(frozen=True, init=False)
class _Result:
    __slots__ = ("_payload",)
    _payload: str

    def __init__(self, data):
        data = _plain(data)
        self._validate(data)
        object.__setattr__(self, "_payload", json.dumps(data, separators=(",", ":"), allow_nan=False))

    def _validate(self, data):
        try:
            (_validate_stages if type(self) is TrainingStageResult else _validate_throughput)(data)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("malformed training extension payload") from exc

    def to_dict(self):
        return json.loads(self._payload)

    def _export_data(self):
        data = _plain(self.to_dict())
        self._validate(data)
        return data

    def to_json(self, path, *, overwrite=False):
        _write_export(path, json.dumps(self._export_data(), sort_keys=True, ensure_ascii=False,
                                      allow_nan=False, indent=2) + "\n", overwrite)

    def to_csv(self, path, *, overwrite=False):
        data = self._export_data()
        if type(self) is TrainingStageResult:
            columns = ["stage", "epoch_id", "scope", "parent_id", "microbatch_id", "window_id", "attempt_id",
                       "status", "reason", "start_ns", "end_ns", "host_ms", "cuda_event_ms"]
            rows = [{k: r[k] for k in columns} for r in data["observations"]]
        else:
            columns = ["epoch_id", "warmup", "training_wall_seconds", *COUNTS, *RATES, "status", "reason"]
            rows = [dict(epoch_id=e.get("epoch_id"), warmup=e.get("warmup"), training_wall_seconds=e["training_wall_seconds"],
                         **e["completed"], **{k:v["value"] for k,v in e["rates"].items()}, status=e["status"], reason=e["reason"])
                    for e in [*data["epochs"], data["segment"]]]
        columns = ["source_segment_id", *columns]
        for row in rows:
            row["source_segment_id"] = data["source_segment_id"]
        _write_export(path, _csv_text(columns, rows), overwrite)


class TrainingStageResult(_Result):
    """Immutable P20 observations with embedded canonical source P19 evidence."""
    __slots__ = ()


class TrainingThroughputResult(_Result):
    """Immutable P22 epoch/segment rates; no training runs or new timer calls."""
    __slots__ = ()


def analyze_training_throughput(result):
    """Derive P22 from validated complete P19 evidence, without executing training."""
    data = _base(result, "training_throughput", "P22", dict(version=1,
        denominator="sum_completed_epoch_training_wall_seconds", validation="excluded",
        warmup="included_and_labeled", aggregation="ratio_of_sums"))
    data["epochs"], data["segment"] = _throughput_evidence(data["source_p19"])
    return TrainingThroughputResult(data)


@dataclass(frozen=True)
class DetailedTrainingResult:
    """Three immutable results produced by one actual fit; no hidden second run."""
    timing: TrainingPerformanceResult
    stages: TrainingStageResult
    throughput: TrainingThroughputResult


def measure_training_detailed(trainer, train_loader, val_loader=None, *, epochs=5,
                              protocol=None, stage_protocol=None, save_last_to=None, resume_from=None):
    """Opt-in P19+P20+P22 in a single native fit; ordinary P19 remains unchanged."""
    stage_protocol = TrainingStageProtocol() if stage_protocol is None else stage_protocol
    if type(stage_protocol) is not TrainingStageProtocol:
        raise TypeError("stage_protocol must be TrainingStageProtocol")
    captured = []
    def factory(t, p):
        recorder = _StageRecorder(t, p, stage_protocol)
        captured.append(recorder)
        return recorder
    timing = _measure_training(trainer, train_loader, val_loader, epochs=epochs, protocol=protocol,
        save_last_to=save_last_to, resume_from=resume_from, recorder_factory=factory)
    stages = _stage_result(timing, captured[0], trainer.grad_clip)
    return DetailedTrainingResult(timing, stages, analyze_training_throughput(timing))


def _stage_result(timing, recorder, grad_clip):
    """Build the unchanged P20 wire result from a completed shared recorder."""
    data = _base(timing, "training_stages", "P20", asdict(recorder.stage_protocol))
    data["configuration"] = dict(grad_clip=grad_clip)
    data["observations"] = recorder.stages
    data["summary"] = _stage_summary(data["observations"])
    data["limitations"].extend(["optimizer execution spans public pre/post hooks, including hook dispatch overhead",
                                "scaler_step_attempt includes optimizer_execution; no additive stage total or percentages",
                                "unmeasured instrumentation overhead is unknown, not zero"])
    return TrainingStageResult(data)
