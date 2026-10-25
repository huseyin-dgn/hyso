"""Versioned, validated plain-data results and explicit atomic exports."""
from dataclasses import dataclass
from decimal import Decimal
import json
import math

# Internal reuse only: no changes to the legacy result/export contract.
from hyso.benchmark.result import _plain, _csv_text, _write_export


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("expected a nonnegative integer count/timestamp")


def _positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("expected a finite positive measurement")


def _quantile(samples, q):
    ordered = sorted(samples)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _tail_count(n, q):
    # Decimal avoids .99 making exactly ten tail samples appear insufficient.
    return float((Decimal(1) - Decimal(str(q))) * n)


def _latency_metrics(rows, quantiles, policy):
    flat = [v for row in rows for v in row]
    result = []
    for q in quantiles:
        sufficient = len(flat) >= policy["min_samples"] and _tail_count(len(flat), q) >= policy["min_tail_samples"]
        result.append(dict(quantile=q, value=_quantile(flat, q) if sufficient else None,
                           unit="ms", status="derived" if sufficient else "insufficient_samples",
                           reason=None if sufficient else "tail_policy",
                           effective_tail_count=_tail_count(len(flat), q)))
    return result


def _validate(data, kind):
    required = {"format", "schema_version", "result_id", "feature_ids", "identity", "protocol",
                "provenance", "requested", "actual", "measurements", "raw_evidence", "diagnostics"}
    if set(data) != required or type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("invalid performance result schema")
    if data["format"] != "hyso.performance." + kind:
        raise ValueError("unsupported performance format")
    import uuid
    uuid.UUID(data["result_id"])
    if data["feature_ids"] != {"latency_analysis": ["P01"], "throughput": ["P02"],
                               "memory": ["P10"], "footprint": ["P11"], "pipeline": ["P03"],
                               "stage_timing": ["P04"], "cold_start": ["P05"],
                               "sweep": data.get('feature_ids') if data.get('feature_ids') in (["P06"],["P07"],["P08"]) else None,
                               "precision": ["P08"], "capacity": ["P17"],
                               "operator_profile": ["P14"], "layer_profile": ["P13"],
                               "operation_count": ["P12"], "transfer": ["P15"],
                               "utilization": ["P16"], "roofline": ["P18"], "layer_overhead": ["P13"]}[kind]:
        raise ValueError("invalid feature identity")
    if data["identity"].get("experiment_run_id") is not None:
        raise ValueError("associate experiment run externally; result identity is immutable")
    protocol = data["protocol"]
    for key in ("model", "input", "environment"):
        if type(data["provenance"][key]) is not dict or not data["provenance"][key]:
            raise ValueError("missing model/input/runtime provenance")
    if type(protocol["version"]) is not int or protocol["version"] != 1:
        raise ValueError("unsupported protocol version")
    if kind == 'layer_overhead':
        from .layer_overhead import _validate_layer_overhead
        _validate_layer_overhead(data)
        return
    if kind == 'roofline':
        from .roofline import _validate_roofline
        _validate_roofline(data)
        return
    if kind == 'transfer':
        from .transfers import _validate_transfer
        _validate_transfer(data)
        return
    if kind == 'utilization':
        from .utilization import _validate_utilization
        _validate_utilization(data)
        return
    if kind in ('operator_profile','layer_profile','operation_count'):
        from .profiling import _validate_profile
        _validate_profile(data,kind)
        return
    if kind == 'capacity':
        from .capacity import _validate_capacity
        _validate_capacity(data)
        return
    if kind in ('sweep','precision'):
        from .sweeps import _validate_sweep
        _validate_sweep(data,kind)
        return
    if kind in ('pipeline', 'stage_timing', 'cold_start'):
        if kind == 'cold_start':
            from .cold import _validate_cold
            _validate_cold(data)
        else:
            from .pipeline import _validate_pipeline
            _validate_pipeline(data, kind)
        return
    if kind in ("memory", "footprint"):
        from .memory import _validate_memory, _validate_footprint
        (_validate_memory if kind == "memory" else _validate_footprint)(data)
        return
    if kind == "latency_analysis":
        if protocol["id"] != "resident_tail" or protocol["method"] != "linear_(n-1)*q":
            raise ValueError("invalid latency protocol")
        rows = data["raw_evidence"]["latency_samples_ms"]
        if type(rows) is not list or not rows or any(type(row) is not list or not row for row in rows):
            raise ValueError("missing raw latency samples")
        for row in rows:
            for value in row:
                _positive(value)
        n = sum(map(len, rows))
        if data["actual"] != dict(samples=n, repeats=len(rows), samples_per_repeat=[len(row) for row in rows]):
            raise ValueError("inconsistent latency sample counts")
        qs = protocol["quantiles"]
        if not qs or len(set(qs)) != len(qs) or any(type(q) not in (int, float) or not 0 < q < 1 for q in qs):
            raise ValueError("invalid quantiles")
        policy = protocol["tail_policy"]
        _integer(policy["min_samples"], 1)
        _integer(policy["min_tail_samples"], 1)
        if data["measurements"] != _latency_metrics(rows, qs, policy):
            raise ValueError("latency quantiles/status disagree with evidence")
        if data["requested"] != dict(samples=n, repeats=len(rows)):
            raise ValueError("inconsistent requested latency samples")
        if not data["identity"]["source_result_id"] or not data["raw_evidence"]["source_sha256"]:
            raise ValueError("missing latency source identity")
        if data["raw_evidence"]["source_result_id"] != data["identity"]["source_result_id"]:
            raise ValueError("inconsistent latency source identity")
    else:
        from .inference import ThroughputProtocol
        plan = ThroughputProtocol(**protocol["configuration"])
        if protocol["id"] != "continuous_completion_window" or protocol["scope"] != "resident_forward_completion_wall":
            raise ValueError("invalid throughput protocol")
        device = data["provenance"]["environment"]["device"]
        if (device != data["provenance"]["input"]["device"] or
                (plan.device == "cuda" and not device.startswith("cuda:")) or
                (plan.device != "cuda" and device != plan.device)):
            raise ValueError("inconsistent device provenance")
        sync = "synchronous_host" if plan.device == "cpu" else "bounded_group_completion_and_terminal_event"
        if protocol["clock"] != "perf_counter_ns" or protocol["synchronization"] != sync:
            raise ValueError("invalid completion clock/sync strategy")
        for value in data["actual"].values():
            _integer(value)
        windows = data["raw_evidence"]["windows"]
        if type(windows) is not list or len(windows) != plan.repeats:
            raise ValueError("inconsistent repeat windows")
        batch = data["provenance"]["input"]["shape"][0]
        _integer(batch, 1)
        total_batches = total_images = 0
        seconds = []
        for index, row in enumerate(windows):
            if row["repeat_index"] != index:
                raise ValueError("duplicate/missing window")
            for key in ("start_ns", "admission_end_ns", "end_ns", "admitted_batches", "completed_batches",
                        "completed_images", "pending_batches", "max_pending_batches", "completion_groups",
                        "backpressure_syncs", "terminal_drains", "repeat_index"):
                _integer(row[key])
            if not row["start_ns"] <= row["admission_end_ns"] <= row["end_ns"] or row["end_ns"] <= row["start_ns"]:
                raise ValueError("inconsistent window/tail timestamps")
            if (row["completed_batches"] < 1 or row["completed_batches"] != row["admitted_batches"]
                    or row["pending_batches"] != 0 or row["completed_images"] != batch * row["completed_batches"]):
                raise ValueError("incomplete tail or inconsistent completion counts")
            if plan.batch_count is not None and row["admitted_batches"] != plan.batch_count:
                raise ValueError("requested batch count not completed")
            if plan.duration_seconds is not None and row["admission_end_ns"] - row["start_ns"] < int(plan.duration_seconds * 1e9):
                raise ValueError("duration admission window truncated")
            cuda = plan.device != "cpu"
            if (row["terminal_drains"] != int(cuda) or
                    row["backpressure_syncs"] > (row["completed_batches"] // plan.queue_depth if cuda else 0) or
                    row["completion_groups"] != (math.ceil(row["completed_batches"] / plan.queue_depth) if cuda else row["completed_batches"])):
                raise ValueError("inconsistent completion group/sync evidence")
            if not 1 <= row["max_pending_batches"] <= (plan.queue_depth if plan.device.startswith("cuda") else 1):
                raise ValueError("unbounded/invalid pending work")
            if row["completed_id_range"] != [0, row["completed_batches"] - 1]:
                raise ValueError("duplicate or inconsistent completion IDs")
            _integer(row["completion_groups"], 1)
            elapsed = (row["end_ns"] - row["start_ns"]) / 1e9
            tail = (row["end_ns"] - row["admission_end_ns"]) / 1e9
            rate = row["completed_images"] / elapsed
            _positive(elapsed); _positive(rate)
            if row["elapsed_wall_seconds"] != elapsed or row["tail_seconds"] != tail or row["images_per_second"] != rate:
                raise ValueError("rate/tail disagrees with wall timestamps")
            total_batches += row["completed_batches"]
            total_images += row["completed_images"]
            seconds.append(elapsed)
        elapsed = math.fsum(seconds)
        if data["actual"] != dict(repeats=len(windows), completed_batches=total_batches,
                                  completed_images=total_images, pending_batches=0):
            raise ValueError("inconsistent aggregate completions")
        expected = dict(value=total_images / elapsed, unit="images/s", status="measured", reason=None,
                        elapsed_wall_seconds=elapsed, aggregation="total_completed_images_over_total_wall_seconds")
        if data["measurements"] != expected:
            raise ValueError("inconsistent sustained throughput")
        if data["requested"] != dict(batch_count=plan.batch_count, duration_seconds=plan.duration_seconds,
                                     repeats=plan.repeats, warmup_iterations=plan.warmup_iterations):
            raise ValueError("inconsistent requested protocol")


@dataclass(frozen=True, init=False)
class _Result:
    __slots__ = ("_payload",)
    _payload: str

    def __init__(self, data):
        copied = _plain(data)
        try:
            _validate(copied, self._kind)
        except (KeyError, IndexError, TypeError, AttributeError, OverflowError) as error:
            raise ValueError("invalid or missing performance schema fields") from error
        object.__setattr__(self, "_payload", json.dumps(copied, allow_nan=False, separators=(",", ":")))

    def to_dict(self):
        return json.loads(self._payload)

    def to_json(self, path, *, overwrite=False):
        _write_export(path, json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False,
                                      sort_keys=True, indent=2) + "\n", overwrite)

    def to_csv(self, path, *, overwrite=False):
        """One row per quantile/window; JSON is the complete canonical record."""
        data = self.to_dict()
        common = dict(format=data["format"], schema_version=1, result_id=data["result_id"],
                      device=data["provenance"]["environment"]["device"], scope=data["protocol"]["scope"])
        if self._kind in ('transfer','utilization'):
            entries = (data['raw_evidence']['copies'] if self._kind=='transfer' else
                       data['raw_evidence']['cpu_intervals'] + data['raw_evidence']['gpu_samples'])
            rows = [dict(common, row_kind=self._kind, observation_index=i,
                         payload=json.dumps(row, sort_keys=True)) for i,row in enumerate(entries)]
            if not rows: rows=[dict(common,row_kind=self._kind,observation_index=None,payload='null')]
        elif self._kind in ('operator_profile','layer_profile','operation_count'):
            entries=(data['measurements']['operators'] if self._kind=='operator_profile' else
                     data['measurements']['modules'] if self._kind=='layer_profile' else data['raw_evidence']['operations'])
            rows=[dict(common,row_kind=self._kind,observation_index=i,payload=json.dumps(row,sort_keys=True))
                  for i,row in enumerate(entries)]
            if not rows: rows=[dict(common,row_kind=self._kind,observation_index=None,payload='null')]
        elif self._kind == 'capacity':
            rows = [dict(common, **{k:v for k,v in row.items()
                                   if k not in ('memory','sweep','provenance')},
                         memory=json.dumps(row['memory'],sort_keys=True),
                         provenance=json.dumps(row['provenance'],sort_keys=True))
                    for row in data['raw_evidence']['candidates']]
            for row in rows: row['shape']=json.dumps(row['shape'])
        elif self._kind in ('sweep','precision'):
            points=data['raw_evidence']['points'] if self._kind=='sweep' else [dict(
                point_id=data['result_id'],observation_index=0,value=data['protocol']['precision'],
                precision=data['protocol']['precision'],shape=data['provenance']['input']['shape'],
                status='measured',reason=None,result_id=data['result_id'],result=data)]
            rows=[]
            for point in points:
                metrics=point['result']['measurements'] if point['result'] is not None else None
                rows.append(dict(common,point_id=point['point_id'],observation_index=point['observation_index'],
                    value=json.dumps(point['value']),precision=point['precision'],shape=json.dumps(point['shape']),
                    status=point['status'],reason=point['reason'],point_result_id=point['result_id'],
                    latency_mean_ms=metrics['latency_mean_ms']['value'] if metrics else None,
                    sustained_images_per_second=metrics['sustained_images_per_second']['value'] if metrics else None))
        elif self._kind in ('pipeline', 'stage_timing', 'cold_start'):
            if self._kind == 'cold_start':
                rows = [dict(common, trial_index=trial['trial_index'], worker_pid=trial['worker_pid'],
                             trial_status=trial['status'], phase=name, **phase)
                        for trial in data['raw_evidence']['trials']
                        for name,phase in dict(startup_to_ready_receipt=trial['startup_to_ready_receipt'], **trial['phases']).items()]
            elif self._kind == 'pipeline':
                rows = [dict(common, **{k:v for k,v in sample.items() if k != 'stages'})
                        for sample in data['raw_evidence']['samples']]
            else:
                rows = [dict(common, repeat_index=sample['repeat_index'], **stage)
                        for sample in data['raw_evidence']['samples'] for stage in sample['stages']]
                for row in rows:
                    for key in ('input_boundary','output_boundary'):
                        row[key]=json.dumps(row[key],sort_keys=True)
        elif self._kind in ("memory", "footprint"):
            if self._kind == "memory" and data["raw_evidence"]["observations"]:
                rows = [dict(common, status=data['measurements']['current_rss_bytes']['status'], reason=None,
                             unit='byte', **sample) for sample in data['raw_evidence']['observations']]
            else:
                rows = [dict(common, metric=name, **metric) for name, metric in data['measurements'].items()]
        elif self._kind == "latency_analysis":
            rows = [dict(common, source_result_id=data["identity"]["source_result_id"],
                         sample_count=data["actual"]["samples"], **metric) for metric in data["measurements"]]
        else:
            rows = [dict(common, **window) for window in data["raw_evidence"]["windows"]]
            for row in rows:
                row["completed_id_range"] = json.dumps(row["completed_id_range"])
                row.update(unit="images/s", status="measured", reason=None)
        _write_export(path, _csv_text(list(rows[0]), rows), overwrite)


@dataclass(frozen=True, init=False)
class LatencyAnalysis(_Result):
    __slots__ = ()
    _kind = "latency_analysis"


@dataclass(frozen=True, init=False)
class ThroughputResult(_Result):
    __slots__ = ()
    _kind = "throughput"


@dataclass(frozen=True, init=False)
class MemoryResult(_Result):
    __slots__ = ()
    _kind = "memory"


@dataclass(frozen=True, init=False)
class FootprintResult(_Result):
    __slots__ = ()
    _kind = "footprint"


@dataclass(frozen=True, init=False)
class PipelineResult(_Result):
    __slots__ = ()
    _kind = 'pipeline'


@dataclass(frozen=True, init=False)
class StageTimingResult(_Result):
    __slots__ = ()
    _kind = 'stage_timing'


@dataclass(frozen=True, init=False)
class ColdStartResult(_Result):
    __slots__ = ()
    _kind = 'cold_start'


@dataclass(frozen=True, init=False)
class SweepResult(_Result):
    __slots__ = ()
    _kind = 'sweep'


@dataclass(frozen=True, init=False)
class PrecisionResult(_Result):
    __slots__ = ()
    _kind = 'precision'


@dataclass(frozen=True, init=False)
class CapacityResult(_Result):
    __slots__ = ()
    _kind = 'capacity'


@dataclass(frozen=True, init=False)
class OperatorProfileResult(_Result):
    __slots__ = ()
    _kind = 'operator_profile'


@dataclass(frozen=True, init=False)
class LayerProfileResult(_Result):
    __slots__ = ()
    _kind = 'layer_profile'


@dataclass(frozen=True, init=False)
class OperationCountResult(_Result):
    __slots__ = ()
    _kind = 'operation_count'


class _CanonicalEnvelopeExports:
    """Lossless schema1 JSON/CSV for derived evidence envelopes."""
    __slots__ = ()

    def to_json(self,path,*,overwrite=False):
        data=type(self)(self.to_dict()).to_dict()
        _write_export(path,json.dumps(data,ensure_ascii=False,allow_nan=False,sort_keys=True,indent=2)+'\n',overwrite)

    def to_csv(self,path,*,overwrite=False):
        data=type(self)(self.to_dict()).to_dict()
        columns=['format','schema_version','result_id','payload']
        row={k:data[k] for k in columns if k!='payload'};row['payload']=json.dumps(data,allow_nan=False)
        _write_export(path,_csv_text(columns,[row]),overwrite)

    @classmethod
    def from_json(cls,path):
        from pathlib import Path
        path=Path(path)
        if path.stat().st_size>128*1024**2:raise ValueError('oversized canonical evidence')
        return cls(json.loads(path.read_text(encoding='utf-8')))

    @classmethod
    def from_csv(cls,path):
        import csv
        from pathlib import Path
        if Path(path).stat().st_size>128*1024**2:raise ValueError('oversized canonical evidence')
        previous=csv.field_size_limit()
        try:
            csv.field_size_limit(128*1024**2)
            with open(path,encoding='utf-8',newline='') as stream:
                reader=csv.DictReader(stream)
                if reader.fieldnames!=['format','schema_version','result_id','payload']:raise ValueError('invalid canonical CSV columns')
                rows=list(reader)
        finally:csv.field_size_limit(previous)
        if len(rows)!=1:raise ValueError('one canonical envelope required')
        result=cls(json.loads(rows[0]['payload']));data=result.to_dict()
        if any(rows[0][k]!=str(data[k]) for k in ('format','schema_version','result_id')):raise ValueError('CSV identity mismatch')
        return result


@dataclass(frozen=True, init=False)
class RooflineResult(_CanonicalEnvelopeExports,_Result):
    __slots__ = ()
    _kind = 'roofline'


@dataclass(frozen=True, init=False)
class LayerOverheadResult(_CanonicalEnvelopeExports,_Result):
    """Canonical schema1 envelope exports, also used by RooflineResult."""
    __slots__ = ()
    _kind = 'layer_overhead'


@dataclass(frozen=True, init=False)
class TransferResult(_Result):
    __slots__ = ()
    _kind = 'transfer'


@dataclass(frozen=True, init=False)
class UtilizationResult(_Result):
    __slots__ = ()
    _kind = 'utilization'
