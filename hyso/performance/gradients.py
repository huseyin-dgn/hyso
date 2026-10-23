"""P25: read-only gradient and native scaler observations of one real fit.

No Trainer math, extra backward/step, scaler mutation, tensor CPU copy or reset.
Dense real packed strided gradients are supported; sparse layouts stay explicit
unsupported. Scalar CUDA readouts synchronize and perturb instrumented timing.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time

import torch

from hyso.benchmark.result import _plain, _csv_text, _write_export
from .training import _int, TrainingPerformanceResult, _measure_training
from .training_details import (_StageRecorder, TrainingStageProtocol, TrainingStageResult,
                               _stage_result, _source)


class GradientDiagnosticsError(RuntimeError):
    """Incomplete observation; training may have advanced. No automatic retry."""


@dataclass(frozen=True)
class GradientDiagnosticsProtocol:
    max_attempts: int = 10000
    max_tensor_records: int = 100000
    chunk_elements: int = 65536
    observe_postclip: bool = True

    def __post_init__(self):
        for name, ceiling in (("max_attempts", 100000), ("max_tensor_records", 1000000),
                              ("chunk_elements", 1048576)):
            _int(getattr(self, name), 1)
            if getattr(self, name) > ceiling:
                raise ValueError(name + " exceeds hard ceiling")
        if type(self.observe_postclip) is not bool:
            raise TypeError("observe_postclip must be bool")


def _metric(value, status="measured", reason=None):
    return dict(value=value, status=status, reason=reason)


def _tensor_gradient(gradient, chunk_elements):
    """Stable real L2 with bounded device-local FP64 scratch, scalar reads only.

    Permute packed strides to a contiguous view without copying. Gapped,
    overlapping/expanded, sparse, complex, meta and Tensor subclasses are not
    validated adapters. Never coalesce/densify a sparse gradient.
    """
    info = dict(dtype=str(gradient.dtype), device=str(gradient.device), layout=str(gradient.layout),
                shape=list(gradient.shape), elements=gradient.numel(), finite=None,
                has_nan=None, has_inf=None, global_l2=None, chunks=0, scalar_reads=0,
                status="unsupported", reason="requires ordinary real packed dense CPU/CUDA gradient")
    if (type(gradient) is not torch.Tensor or gradient.layout != torch.strided or
            gradient.device.type not in ("cpu", "cuda") or gradient.dtype not in
            (torch.float16, torch.bfloat16, torch.float32, torch.float64) or
            gradient.is_conj() or gradient.is_neg()):
        return info
    with torch.no_grad():
        ordered = gradient.detach().permute(sorted(range(gradient.ndim), key=lambda d: -gradient.stride()[d]))
        if not ordered.is_contiguous():
            return info
        flat = ordered.view(-1)
        norm, has_nan, has_inf = 0., False, False
        for start in range(0, flat.numel(), chunk_elements):
            values = flat[start:start+chunk_elements].to(dtype=torch.float64)
            nan = bool(torch.isnan(values).any().item())
            inf = bool(torch.isinf(values).any().item())
            has_nan |= nan; has_inf |= inf
            info["chunks"] += 1; info["scalar_reads"] += 2
            if not nan and not inf:
                maximum = values.abs().max().item()
                info["scalar_reads"] += 1
                if maximum:
                    squared = (values / maximum).square().sum(dtype=torch.float64).item()
                    info["scalar_reads"] += 1
                    norm = math.hypot(norm, maximum * math.sqrt(squared))
        finite = not (has_nan or has_inf)
        status = "nonfinite" if not finite else "measured" if math.isfinite(norm) else "unrepresentable"
        reason = {"nonfinite":"observed_nan_or_inf", "measured":None,
                  "unrepresentable":"finite_l2_exceeds_float64_range"}[status]
        info.update(finite=finite, has_nan=has_nan, has_inf=has_inf,
                    global_l2=norm if status == "measured" else None, status=status, reason=reason)
    return info


def _summary(tensors):
    present = [r["gradient"] for r in tensors if r["gradient"] is not None]
    unsupported = sum(g["status"] == "unsupported" for g in present)
    nonfinite = any(g["finite"] is False for g in present)
    if unsupported:
        norm = _metric(None, "unsupported", "incomplete_gradient_coverage")
    elif nonfinite:
        norm = _metric(None, "nonfinite", "observed_nan_or_inf")
    else:
        value = math.hypot(*(g["global_l2"] for g in present)) if all(g["global_l2"] is not None for g in present) else math.inf
        norm = (_metric(value) if math.isfinite(value) else
                _metric(None, "unrepresentable", "finite_l2_exceeds_float64_range"))
    return dict(global_l2=norm, finite=False if nonfinite else None if unsupported else True,
                nan_tensor_count=None if unsupported else sum(g["has_nan"] for g in present),
                inf_tensor_count=None if unsupported else sum(g["has_inf"] for g in present),
                parameters=len(tensors), with_gradient=len(present), without_gradient=len(tensors)-len(present),
                unsupported_gradients=unsupported, gradient_elements=sum(g["elements"] for g in present),
                chunks=sum(g["chunks"] for g in present), scalar_reads=sum(g["scalar_reads"] for g in present))


def _snapshot(optimizer, protocol, budget):
    start = time.perf_counter_ns()
    tensors, seen = [], {}
    for group_id, group in enumerate(optimizer.param_groups):
        for position, parameter in enumerate(group["params"]):
            label = "group." + str(group_id) + ".parameter." + str(position)
            if id(parameter) in seen:
                seen[id(parameter)]["names"].append(label)
                continue
            if len(tensors) >= budget:
                raise GradientDiagnosticsError("gradient tensor record budget exhausted")
            row = dict(parameter_id=len(tensors)+1, names=[label],
                       gradient=None if parameter.grad is None else _tensor_gradient(parameter.grad, protocol.chunk_elements))
            tensors.append(row); seen[id(parameter)] = row
    return dict(query_start_ns=start, query_end_ns=time.perf_counter_ns(), tensors=tensors,
                summary=_summary(tensors))


def _scale_change(before, after):
    for value in (before, after):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise GradientDiagnosticsError("invalid native scaler readout")
    return "increased" if after > before else "decreased" if after < before else "unchanged"


class _GradientRecorder(_StageRecorder):
    def __init__(self, trainer, timing_protocol, stage_protocol, protocol):
        super().__init__(trainer, timing_protocol, stage_protocol)
        self.gradient_protocol = protocol
        self.diagnostics = []
        self.current = None
        self.tensor_records = 0
        self.window_samples = 0
        self.micro_ids = []
        self.closed = False

    def microbatch_begin(self, microbatch_id, new_window):
        if new_window:
            self.window_samples = 0; self.micro_ids = []
        super().microbatch_begin(microbatch_id, new_window)

    def microbatch_end(self, samples, inputs, targets):
        self.micro_ids.append(self.active["microbatch"][0]["microbatch_id"])
        self.window_samples += samples
        super().microbatch_end(samples, inputs, targets)

    def attempt_begin(self, attempt_id, updates):
        if len(self.diagnostics) >= self.gradient_protocol.max_attempts:
            raise GradientDiagnosticsError("gradient attempt record budget exhausted")
        super().attempt_begin(attempt_id, updates)
        self.stage_cursor = len(self.stages)
        self.current = dict(epoch_id=self.epoch["epoch_id"], window_id=self.window_id,
            attempt_id=attempt_id, microbatch_ids=list(self.micro_ids), samples=self.window_samples,
            preclip=None, postclip=None, scale_before=float(self.trainer.scaler.get_scale()),
            scale_after=None, scale_query_ns=0)

    def snapshot(self):
        data = _snapshot(self.trainer.optimizer, self.gradient_protocol,
                         self.gradient_protocol.max_tensor_records-self.tensor_records)
        self.tensor_records += len(data["tensors"])
        return data

    def stage_end(self, name):
        super().stage_end(name)
        if name == "gradient_check_normalization":
            self.current["preclip"] = self.snapshot()
        elif name == "gradient_clipping" and self.gradient_protocol.observe_postclip:
            self.current["postclip"] = self.snapshot()
        elif name == "scaler_update":
            start = time.perf_counter_ns()
            self.current["scale_after"] = float(self.trainer.scaler.get_scale())
            self.current["scale_query_ns"] = time.perf_counter_ns()-start

    def attempt_end(self, updates, samples):
        row = self.current
        if samples != row["samples"] or row["preclip"] is None or row["scale_after"] is None:
            raise GradientDiagnosticsError("incomplete diagnostics window")
        stages = self.stages[self.stage_cursor:]
        execution = next(r for r in stages if r["stage"] == "optimizer_execution")
        clipping = next(r for r in stages if r["stage"] == "gradient_clipping")
        before = self.active["attempt"][0]["update_before"]
        row.update(applied=updates == before+1, skipped=updates == before,
            applied_update_id=updates if updates == before+1 else None,
            optimizer_executed=execution["status"] == "measured",
            clipping=dict(status=clipping["status"], reason=clipping["reason"], threshold=self.trainer.grad_clip),
            postclip_reason=(None if row["postclip"] is not None else "not_requested"
                             if not self.gradient_protocol.observe_postclip else "clipping_not_executed"),
            scale_change=_scale_change(row["scale_before"],row["scale_after"]),
            overflow_evidence=("native_skip_with_nonfinite_gradient" if self.trainer.amp and
                updates == before and row["preclip"]["summary"]["finite"] is False else
                "none_observed" if updates == before+1 and row["preclip"]["summary"]["finite"] is True else "unknown"),
            overflow_cause="unknown")
        self.diagnostics.append(row); self.current = None
        super().attempt_end(updates, samples)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            super().close()
        finally:
            self.current = None
            self.trainer = None


def _hash(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()


def _outcome(rows):
    if not rows:
        return "no_new_attempts", "completed_checkpoint"
    if any(r["preclip"]["summary"]["unsupported_gradients"] or
           (r["postclip"] is not None and r["postclip"]["summary"]["unsupported_gradients"]) for r in rows):
        return "partial", "unsupported_gradient_representation"
    if any(r[k] is not None and r[k]["summary"]["global_l2"]["status"] == "unrepresentable"
           for r in rows for k in ("preclip","postclip")):
        return "partial", "finite_norm_exceeds_float64_range"
    return "measured", None


def _diagnostic_summary(rows):
    recovered, awaiting = [], False
    for row in rows:
        if row["skipped"]:
            awaiting = True
        elif row["applied"] and awaiting:
            recovered.append(row["attempt_id"]); awaiting = False
    return dict(attempts=len(rows),applied=sum(r["applied"] for r in rows),skipped=sum(r["skipped"] for r in rows),
                recovered_after_skip_attempt_ids=recovered,recovery_scope="this_segment_only",
                scale_changes={k:sum(r["scale_change"] == k for r in rows) for k in ("increased","unchanged","decreased")})


def _validate_snapshot(data, protocol):
    if set(data) != {"query_start_ns","query_end_ns","tensors","summary"}:
        raise ValueError("invalid gradient snapshot schema")
    _int(data["query_start_ns"]); _int(data["query_end_ns"])
    if data["query_end_ns"] < data["query_start_ns"]:
        raise ValueError("invalid gradient query clock")
    for index,row in enumerate(data["tensors"]):
        if set(row) != {"parameter_id","names","gradient"} or type(row["parameter_id"]) is not int or row["parameter_id"] != index+1 or not row["names"] or any(type(n) is not str or not n for n in row["names"]):
            raise ValueError("invalid gradient parameter identity")
        g = row["gradient"]
        if g is None:
            continue
        if set(g) != {"dtype","device","layout","shape","elements","finite","has_nan","has_inf","global_l2","chunks","scalar_reads","status","reason"}:
            raise ValueError("invalid gradient tensor schema")
        for key in ("elements","chunks","scalar_reads"):
            _int(g[key])
        for dimension in g["shape"]:
            _int(dimension)
        if g["elements"] != math.prod(g["shape"]):
            raise ValueError("invalid gradient element ledger")
        if g["status"] == "unsupported":
            if any(g[k] is not None for k in ("finite","has_nan","has_inf","global_l2")) or not g["reason"] or g["chunks"] or g["scalar_reads"]:
                raise ValueError("unsupported gradient cannot be zero-filled")
        else:
            if g["dtype"] not in ("torch.float16","torch.bfloat16","torch.float32","torch.float64") or g["layout"] != "torch.strided" or not (g["device"] == "cpu" or g["device"].startswith("cuda:")):
                raise ValueError("unvalidated gradient representation")
            if any(type(g[k]) is not bool for k in ("finite","has_nan","has_inf")) or g["finite"] != (not (g["has_nan"] or g["has_inf"])):
                raise ValueError("invalid finite evidence")
            if g["chunks"] != (g["elements"]+protocol.chunk_elements-1)//protocol.chunk_elements or not 2*g["chunks"] <= g["scalar_reads"] <= 4*g["chunks"]:
                raise ValueError("invalid chunk/scalar read budget")
            if g["status"] == "measured":
                if not g["finite"] or type(g["global_l2"]) not in (float,int) or not math.isfinite(g["global_l2"]) or g["global_l2"] < 0 or g["reason"] is not None:
                    raise ValueError("invalid finite L2")
            elif (g["status"] not in ("nonfinite","unrepresentable") or g["global_l2"] is not None or
                  g["finite"] != (g["status"] == "unrepresentable") or g["reason"] !=
                  ("finite_l2_exceeds_float64_range" if g["finite"] else "observed_nan_or_inf")):
                raise ValueError("invalid unavailable norm")
    if data["summary"] != _summary(data["tensors"]):
        raise ValueError("gradient summary arithmetic mismatch")


def _validate(data):
    required = {"schema_version","format","feature_ids","source_segment_id","source_sha256","source_p19",
                "source_p20","source_p20_sha256","protocol","actual","observations","status","reason",
                "instrumentation","limitations","sha256","summary"}
    if set(data) != required or type(data["schema_version"]) is not int or data["schema_version"] != 1 or data["feature_ids"] != ["P25"] or data["format"] != "hyso.performance.gradient_diagnostics":
        raise ValueError("invalid P25 schema/identity")
    if data["sha256"] != _hash({k:v for k,v in data.items() if k != "sha256"}):
        raise ValueError("P25 evidence checksum mismatch")
    source,digest = _source(TrainingPerformanceResult(data["source_p19"]))
    stages = TrainingStageResult(data["source_p20"])._export_data()
    if source["segment_id"] != data["source_segment_id"] or digest != data["source_sha256"] or stages["source_p19"] != source or _hash(stages) != data["source_p20_sha256"]:
        raise ValueError("P25 source association mismatch")
    protocol = GradientDiagnosticsProtocol(**data["protocol"])
    rows = data["observations"]
    attempts = [r for r in source["observations"] if r["kind"] == "attempt"]
    if len(rows) != len(attempts) or len(rows) > protocol.max_attempts or data["actual"] != dict(epochs=source["actual_epochs"],windows=len(attempts),attempts=len(attempts)):
        raise ValueError("invalid P25 actual attempt ledger")
    if (data["status"],data["reason"]) != _outcome(rows):
        raise ValueError("invalid P25 coverage claim")
    if data["summary"] != _diagnostic_summary(rows):
        raise ValueError("invalid recovery/scale summary")
    by_attempt = {}
    micro_by_window = {}
    for m in source["observations"]:
        if m["kind"] == "microbatch":
            micro_by_window.setdefault(m["window_id"],[]).append(m["microbatch_id"])
    for s in stages["observations"]:
        if s["attempt_id"] is not None:
            by_attempt.setdefault(s["attempt_id"],{})[s["stage"]] = s
    total, previous_scale = 0, None
    for row,attempt in zip(rows,attempts):
        if set(row) != {"epoch_id","window_id","attempt_id","microbatch_ids","samples","preclip","postclip",
                       "scale_before","scale_after","scale_query_ns","applied","skipped","applied_update_id",
                       "optimizer_executed","clipping","postclip_reason","scale_change","overflow_evidence","overflow_cause"}:
            raise ValueError("invalid diagnostic attempt schema")
        for k in ("epoch_id","window_id","attempt_id","samples"):
            _int(row[k],1)
            if row[k] != attempt[k]:
                raise ValueError("diagnostic/source IDs or samples disagree")
        for k in ("applied","skipped","optimizer_executed"):
            if type(row[k]) is not bool:
                raise ValueError("invalid execution flag")
        if any(row[k] != attempt[k] for k in ("applied","skipped","applied_update_id")):
            raise ValueError("diagnostic/source applied ownership mismatch")
        micro = micro_by_window.get(row["window_id"],[])
        if row["microbatch_ids"] != micro:
            raise ValueError("invalid microbatch accumulation links")
        linked = by_attempt[row["attempt_id"]]
        clip = linked["gradient_clipping"]; step = linked["scaler_step_attempt"]
        if row["optimizer_executed"] != (linked["optimizer_execution"]["status"] == "measured") or row["optimizer_executed"] != row["applied"]:
            raise ValueError("native optimizer execution evidence mismatch")
        if row["clipping"] != dict(status=clip["status"],reason=clip["reason"],threshold=stages["configuration"]["grad_clip"]):
            raise ValueError("invalid clipping execution evidence")
        _validate_snapshot(row["preclip"],protocol)
        pre = row["preclip"]
        if not linked["gradient_check_normalization"]["end_ns"] <= pre["query_start_ns"] <= pre["query_end_ns"] <= (clip["start_ns"] if clip["status"] == "measured" else step["start_ns"]):
            raise ValueError("norm is not normalized preclip evidence")
        post = row["postclip"]
        expected_post = protocol.observe_postclip and clip["status"] == "measured"
        if (post is not None) != expected_post or row["postclip_reason"] != (None if expected_post else "not_requested" if not protocol.observe_postclip else "clipping_not_executed"):
            raise ValueError("invalid postclip coverage")
        if post is not None:
            _validate_snapshot(post,protocol)
            if not clip["end_ns"] <= post["query_start_ns"] <= post["query_end_ns"] <= step["start_ns"]:
                raise ValueError("postclip observation is outside clipping boundary")
        total += len(pre["tensors"]) + (len(post["tensors"]) if post else 0)
        _int(row["scale_query_ns"])
        if row["scale_change"] != _scale_change(row["scale_before"],row["scale_after"]):
            raise ValueError("invalid scale classification")
        if previous_scale is not None and row["scale_before"] != previous_scale:
            raise ValueError("discontinuous native scale sequence")
        previous_scale = row["scale_after"]
        if not source["provenance"]["training"]["amp"] and (row["scale_before"] != 1. or row["scale_after"] != 1.):
            raise ValueError("disabled native scaler must report scale one")
        overflow = ("native_skip_with_nonfinite_gradient" if source["provenance"]["training"]["amp"] and
                    row["skipped"] and pre["summary"]["finite"] is False else
                    "none_observed" if row["applied"] and pre["summary"]["finite"] is True else "unknown")
        if row["overflow_evidence"] != overflow or row["overflow_cause"] != "unknown":
            raise ValueError("unproven overflow attribution")
    if total > protocol.max_tensor_records:
        raise ValueError("unbounded gradient records")
    if data["instrumentation"] != _instrumentation(rows,protocol,source):
        raise ValueError("invalid instrumentation accounting")


def _instrumentation(rows,protocol,source):
    snapshots = [r[k] for r in rows for k in ("preclip","postclip") if r[k] is not None]
    return dict(norm="sqrt(sum(abs(unscaled_sample_normalized_gradient)^2))",
        reduction="bounded_device_local_FP64_max_scaled_chunks_and_host_math_hypot",
        parameter_convention="unique_optimizer_parameter_object; distinct_alias_objects_count_separately",
        sparse_policy="unsupported_no_coalesce_no_densify", chunk_elements=protocol.chunk_elements,
        query_ns=sum(s["query_end_ns"]-s["query_start_ns"] for s in snapshots),
        scalar_reads=sum(s["summary"]["scalar_reads"] for s in snapshots),
        scale_after_query_ns=sum(r["scale_query_ns"] for r in rows),
        stream_id=source["instrumentation"]["stream_id"],
        cuda_host_scalar_sync=source["provenance"]["runtime"]["device"].startswith("cuda"),
        total_overhead_ns=None, no_full_gradient_cpu_copy=True,
        scope="actual_attempts_only; diagnostic_queries_between_P20_stages_in_P19_attempt_wall")


@dataclass(frozen=True,init=False)
class GradientDiagnosticsResult:
    __slots__ = ("_payload",)
    _payload: str

    def __init__(self,data):
        copied = _plain(data); self._check(copied)
        object.__setattr__(self,"_payload",json.dumps(copied,allow_nan=False,separators=(",",":")))

    @staticmethod
    def _check(data):
        try:
            _validate(data)
        except (KeyError,TypeError,AttributeError) as error:
            raise ValueError("malformed P25 payload") from error

    def to_dict(self):
        return json.loads(self._payload)

    def _export_data(self):
        data = _plain(self.to_dict()); self._check(data); return data

    def to_json(self,path,*,overwrite=False):
        _write_export(path,json.dumps(self._export_data(),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n",overwrite)

    def to_csv(self,path,*,overwrite=False):
        data = self._export_data()
        columns = ["source_segment_id","epoch_id","window_id","attempt_id","samples","preclip_l2","norm_status",
                   "finite","nan_tensor_count","inf_tensor_count","with_gradient","without_gradient",
                   "scale_before","scale_after","scale_change","optimizer_executed","applied","skipped","applied_update_id"]
        rows = []
        for r in data["observations"]:
            summary = r["preclip"]["summary"]
            row = {k:r.get(k,summary.get(k)) for k in columns}
            row.update(source_segment_id=data["source_segment_id"],preclip_l2=summary["global_l2"]["value"],norm_status=summary["global_l2"]["status"])
            rows.append(row)
        _write_export(path,_csv_text(columns,rows),overwrite)


def measure_gradient_diagnostics(trainer,train_loader,val_loader=None,*,epochs=5,protocol=None,
                                 timing_protocol=None,stage_protocol=None,save_last_to=None,resume_from=None):
    """One opt-in real fit; history/state/RNG advance normally. No rollback/retry.

    P20 native execution hooks and P19 source IDs are reused. Norm queries occur
    after normalization and before clipping; optional postclip is a separate read.
    Non-AMP nonfinite gradients retain Trainer's existing failure, without result.
    """
    protocol = GradientDiagnosticsProtocol() if protocol is None else protocol
    stage_protocol = TrainingStageProtocol() if stage_protocol is None else stage_protocol
    if type(protocol) is not GradientDiagnosticsProtocol or type(stage_protocol) is not TrainingStageProtocol:
        raise TypeError("invalid diagnostics/stage protocol")
    from hyso.training import Trainer
    if type(trainer) is not Trainer or type(trainer.scaler) is not torch.cuda.amp.GradScaler:
        raise TypeError("standard Trainer and native GradScaler required")
    captured = []
    def factory(t,p):
        recorder = _GradientRecorder(t,p,stage_protocol,protocol); captured.append(recorder); return recorder
    try:
        timing = _measure_training(trainer,train_loader,val_loader,epochs=epochs,protocol=timing_protocol,
            save_last_to=save_last_to,resume_from=resume_from,recorder_factory=factory)
        recorder = captured[0]
        stages = _stage_result(timing,recorder,trainer.grad_clip)._export_data()
        source,digest = _source(timing)
        rows = recorder.diagnostics
        status,reason = _outcome(rows)
        data = dict(schema_version=1,format="hyso.performance.gradient_diagnostics",feature_ids=["P25"],
            source_segment_id=source["segment_id"],source_sha256=digest,source_p19=source,
            source_p20=stages,source_p20_sha256=_hash(stages),protocol=asdict(protocol),
            actual=dict(epochs=source["actual_epochs"],windows=len(rows),attempts=len(rows)),
            observations=rows,summary=_diagnostic_summary(rows),status=status,reason=reason,instrumentation=_instrumentation(rows,protocol,source),
            limitations=["CUDA scalar readouts synchronize; norm/scratch allocation and dispatch perturb instrumented timing and memory.",
                "Query time is measured; total instrumentation overhead and scale-before read overhead are not isolated.",
                "No CUPTI/kernel attribution or hidden-stream guarantee; P20/P19 are instrumented source projections.",
                "Sparse/complex/gapped/overlapping/custom tensors have null unsupported norms; no implicit dense conversion.",
                "Non-AMP nonfinite gradients preserve Trainer failure; failed runs return no successful partial ledger.",
                "Scale change is not optimizer execution proof; native P20 hooks and Trainer counters own execution.",
                "Overflow evidence records nonfinite-gradient/skip association, not the root cause of overflow.",
                "Logical norm counts unique parameter objects; separate parameter objects sharing gradient storage count separately.",
                "Observation budgets are not workload memory/time limits; no automatic retry, rollback, GC or allocator reset."])
        data["sha256"] = _hash(data)
        return GradientDiagnosticsResult(data)
    finally:
        try:
            for recorder in captured:recorder.close()
        finally:
            captured.clear()
