"""Version 2 task companion; schema 1 loader and worker artifacts stay intact."""
from dataclasses import dataclass
import json

from hyso.benchmark.result import _plain, _write_export
from .dataloader import DataLoaderAnalysisResult, _hash
from ._loader_task_trace import WorkerTaskDataLoader, ADAPTER, SOURCE_HASH
from ._loader_trace_groups import validate_trace_groups


def _derive(data):
    parent = DataLoaderAnalysisResult(data["source_p21"]).to_dict()
    binding, ledger, packets = data["binding"], data["ledger"], data["packets"]
    config = parent["provenance"]["loaders"]["training"]
    if config["type"] != "hyso.performance._loader_task_trace.WorkerTaskDataLoader":
        raise ValueError("source did not execute the task adapter")
    if (binding["source_p21_sha256"] != _hash(parent)
            or binding["source_segment_id"] != parent["source_segment_id"]
            or binding["adapter"] != ADAPTER or binding["adapter_source_sha256"] != SOURCE_HASH):
        raise ValueError("task source binding mismatch")
    if type(binding["execution_id"]) is not str or not binding["execution_id"]:
        raise ValueError("invalid execution identity")
    if any(type(items) is not list or len(items) > 100000 for items in (ledger, packets, data["worker_states"])):
        raise ValueError("invalid task evidence budget")
    errors, submitted, received, delivered, packet_map = [], {}, {}, {}, {}
    pending = buffered = pending_max = buffered_max = 0
    task_ids = set()
    for sequence, row in enumerate(ledger):
        if type(row["sequence"]) is not int or row["sequence"] != sequence:
            errors.append("ledger sequence mismatch")
        kind = row["kind"]
        receipt = row if kind == "submit" else row["receipt"]
        token = receipt["token"]
        if type(token) is not str or not token:
            raise ValueError("invalid task token")
        target = {"submit": submitted, "receive": received, "deliver": delivered}.get(kind)
        if target is None:
            raise ValueError("invalid lifecycle event")
        if token in target:
            errors.append("duplicate lifecycle token")
        target[token] = row
        if receipt["execution_id"] != binding["execution_id"]:
            errors.append("foreign execution")
        if type(receipt["worker_id"]) is not int or receipt["worker_id"] < 0:
            errors.append("invalid worker identity")
        if any(type(receipt[k]) is not int or receipt[k] < 1 for k in ("epoch_id", "batch_id")):
            errors.append("invalid epoch/batch identity")
        if kind == "submit":
            identity = (receipt["epoch_id"], receipt["batch_id"])
            if identity in task_ids:
                errors.append("duplicate epoch/batch task identity")
            task_ids.add(identity)
            pending += 1
            if not (type(row["start_ns"]) is int and type(row["end_ns"]) is int
                    and type(row["duration_ns"]) is int
                    and row["start_ns"] >= 0 and row["end_ns"] >= row["start_ns"]
                    and row["duration_ns"] == row["end_ns"] - row["start_ns"]):
                errors.append("invalid actual index queue put duration")
        elif kind == "receive":
            if type(row["at_ns"]) is not int or row["at_ns"] < 0:
                errors.append("invalid parent receive clock")
            pending -= 1
            buffered += 1
            if token not in submitted:
                errors.append("receive without submission")
        else:
            if any(type(row[k]) is not int or row[k] < 0 for k in ("demand_ns", "at_ns")):
                errors.append("invalid parent delivery clock")
            buffered -= 1
            if token not in received:
                errors.append("delivery without receipt")
        if pending < 0 or buffered < 0:
            errors.append("negative task conservation")
        pending_max, buffered_max = max(pending_max, pending), max(buffered_max, buffered)
    if pending or buffered or not submitted or set(submitted) != set(received) or set(submitted) != set(delivered):
        errors.append("incomplete task lifecycle")
    persistent = config["persistent_workers"]
    states = {(None if persistent else s["epoch_id"], s["worker_id"]): s for s in data["worker_states"]}
    workers = parent["provenance"]["loaders"]["training"]["num_workers"]
    epochs = {r["epoch_id"] for r in submitted.values()}
    expected_states = {(None if persistent else epoch, worker) for epoch in epochs for worker in range(workers)}
    if (len(states) != len(data["worker_states"]) or set(states) != expected_states
            or any(len({s["pid"] for key,s in states.items() if key[0] == epoch}) != workers
                   for epoch in ({None} if persistent else epochs))
            or any(type(s["worker_id"]) is not int or s["worker_id"] < 0
                   or (not persistent and (type(s["epoch_id"]) is not int or s["epoch_id"] < 1))
                   or type(s["pid"]) is not int or s["pid"] <= 0 or s["alive"] is not False
                   or type(s["exitcode"]) is not int or s["exitcode"] != 0 for s in states.values())):
        errors.append("unclean or invalid process receipts")
    counters = data["trace_counters"]
    if (type(counters) is not dict or set(counters) != {"attempted", "accepted", "rejected"}
            or any(type(v) is not int or v < 0 for v in counters.values())):
        raise ValueError("valid task IPC counters required")
    if (counters["attempted"] != len(submitted) or counters["accepted"] != len(packets)
            or counters["attempted"] != counters["accepted"] + counters["rejected"]
            or counters["rejected"]):
        errors.append("task IPC conservation failure")
    stages, clock_bounds = {}, []
    for packet in packets:
        receipt = packet["receipt"]
        if (type(receipt["worker_id"]) is not int or receipt["worker_id"] < 0
                or type(receipt["worker_pid"]) is not int or receipt["worker_pid"] <= 0
                or any(type(receipt[k]) is not int or receipt[k] < 1 for k in ("epoch_id", "batch_id"))):
            errors.append("invalid worker packet identity types")
        token = receipt["token"]
        if packet["schema_version"] != 2 or type(packet["schema_version"]) is not int or token in packet_map:
            errors.append("invalid or duplicate worker task packet")
        packet_map[token] = packet
        if (packet["clock"] != binding["parent_clock"]
                or packet["clock"]["monotonic"] is not True
                or type(packet["clock"]["resolution_ns"]) not in (int, float)
                or not 0 < packet["clock"]["resolution_ns"] < 1e9):
            errors.append("incompatible clock rate/source metadata")
        submit, receive, deliver = submitted.get(token), received.get(token), delivered.get(token)
        if submit is None or receive is None or deliver is None:
            errors.append("unmatched worker task")
            continue
        if receive["receipt"] != receipt or deliver["receipt"] != receipt:
            errors.append("worker/parent receipt identity mismatch")
        if any(receipt[k] != submit[k] for k in ("execution_id", "epoch_id", "batch_id", "token", "worker_id")):
            errors.append("submitted task identity mismatch")
        state = states.get((None if persistent else receipt["epoch_id"], receipt["worker_id"]))
        if state is None or receipt["worker_pid"] != state["pid"]:
            errors.append("worker PID mismatch")
        collate = packet["collate"]
        times = [receipt["receive_ns"], collate["start_ns"], collate["end_ns"], receipt["finish_ns"]]
        if (any(type(t) is not int or t < 0 for t in times) or times != sorted(times)
                or collate["end_ns"] != receipt["finish_ns"]
                or type(collate["duration_ns"]) is not int
                or collate["duration_ns"] != collate["end_ns"] - collate["start_ns"]
                or collate["outcome"] != "completed"):
            errors.append("invalid collate evidence")
        else:
            stages.setdefault("collate", []).append(collate["duration_ns"] / 1e6)
        if len(packet["samples"]) != len(submit["indices"]):
            errors.append("task sample count mismatch")
        for position, sample in enumerate(packet["samples"]):
            if (position >= len(submit["indices"]) or type(sample["position"]) is not int or sample["position"] != position
                    or sample["sample_id"] != submit["indices"][position]
                    or sample["trace_id"] != token + ":" + str(position)
                    or type(sample["dropped_events"]) is not int or sample["dropped_events"] != 0):
                errors.append("task sample identity/drop mismatch")
            group = validate_trace_groups(sample["events"], expected_traces=1)
            errors.extend(group["errors"])
            for event in sample["events"]:
                if (event["trace_id"] != sample["trace_id"] or event["sample_id"] != sample["sample_id"]
                        or event["worker_id"] != receipt["worker_id"] or event["worker_pid"] != receipt["worker_pid"]
                        or not receipt["receive_ns"] <= event["start_ns"] <= event["end_ns"] <= collate["start_ns"]):
                    errors.append("sample event/task mismatch")
                stages.setdefault(event["stage"], []).append(event["duration_ms"])
        observation = deliver["observation"]
        if (observation is None or observation["epoch_id"] != receipt["epoch_id"]
                or observation["batch_index"] != receipt["batch_id"]
                or observation["samples"] != len(submit["indices"])):
            errors.append("P19/P21 batch binding mismatch")
        if observation is not None and not (
                observation["start_ns"] <= deliver["demand_ns"] <= deliver["at_ns"] <= observation["end_ns"]):
            errors.append("task delivery is outside its source batch clock window")
        if (not submit["sequence"] < receive["sequence"] < deliver["sequence"]
                or not submit["start_ns"] <= receive["at_ns"] <= deliver["at_ns"]
                or not deliver["demand_ns"] <= deliver["at_ns"]
                or type(deliver["ordered_buffer_hit"]) is not bool
                or deliver["ordered_buffer_hit"] != (receive["at_ns"] <= deliver["demand_ns"])):
            errors.append("invalid parent lifecycle timing")
        # A causal bracket, not a midpoint offset estimate or a duration claim.
        low = submit["start_ns"] - receipt["receive_ns"]
        high = receive["at_ns"] - receipt["finish_ns"]
        if low > high:
            errors.append("inconsistent causal clock bounds")
        clock_bounds.append(dict(token=token, offset_low_ns=low, offset_high_ns=high,
                                 uncertainty_width_ns=high-low))
    if set(packet_map) != set(submitted):
        errors.append("missing or unexpected worker packets")
    source_batches = [r for r in parent["observations"] if r["kind"] == "batch" and r["phase"] == "training"]
    delivered_observations = [r["observation"] for r in ledger if r["kind"] == "deliver"]
    if source_batches != delivered_observations:
        errors.append("source batch receipts differ")
    complete = not errors
    submit_times = [r["duration_ns"] / 1e6 for r in submitted.values()]
    return dict(status="complete" if complete else "incomplete", errors=errors,
        association=dict(execution="verified" if complete else "incomplete",
                         epoch="verified" if complete else "incomplete", batch="verified" if complete else "incomplete"),
        task_count=len(submitted), sample_count=sum(len(r["indices"]) for r in submitted.values()),
        stage_timings={k: dict(count=len(v), mean_ms=sum(v)/len(v) if complete else None,
                              status="measured" if complete else "incomplete") for k,v in stages.items()},
        dispatch=dict(scope="parent_index_queue_put_call_excludes_feeder_serialization",
                      status="measured" if complete else "incomplete",
                      count=len(submit_times), mean_ms=sum(submit_times)/len(submit_times) if complete else None),
        lifecycle=dict(status="measured" if complete else "incomplete",
            submitted_not_received_high_water=pending_max if complete else None,
            received_not_delivered_high_water=buffered_max if complete else None,
            ordered_buffer_hits=sum(r["ordered_buffer_hit"] for r in delivered.values()) if complete else None,
            ordered_buffer_hit_fraction=(sum(r["ordered_buffer_hit"] for r in delivered.values()) / len(delivered))
                if complete else None),
        clock_association=dict(status="causal_bounds" if complete else "incomplete", bounds=clock_bounds,
            cross_process_durations="unsupported_no_accuracy_bound"),
        queue_occupancy=dict(status="unsupported", value=None, reason="no_physical_queue_boundary_receipts"),
        prefetch_efficiency=dict(status="unsupported", value=None, reason="buffer_hits_are_not_efficiency"))


@dataclass(frozen=True, init=False)
class WorkerTaskAnalysisResult:
    __slots__ = ("_payload",)
    _payload: str

    def __init__(self, data):
        data = _plain(data)
        self._check(data)
        object.__setattr__(self, "_payload", json.dumps(data, allow_nan=False))

    @staticmethod
    def _check(data):
        if (type(data) is not dict or set(data) != {"schema_version", "format", "source_p21", "binding", "ledger",
                "packets", "trace_counters", "worker_states", "evidence_sha256", "analysis"}
                or type(data["schema_version"]) is not int or data["schema_version"] != 2
                or data["format"] != "hyso.performance.worker_tasks"):
            raise ValueError("invalid worker task schema")
        evidence = {k:v for k,v in data.items() if k not in ("analysis", "evidence_sha256")}
        if data["evidence_sha256"] != _hash(evidence):
            raise ValueError("task evidence hash mismatch")
        try:
            derived = _derive(data)
        except (KeyError, TypeError, IndexError, OverflowError) as exc:
            raise ValueError("malformed task evidence") from exc
        if data["analysis"] != derived:
            raise ValueError("task derived evidence mismatch")

    def to_dict(self):
        return json.loads(self._payload)

    def diagnostics(self):
        """Additional same-clock durations from audited task receipts.

        Retains the v2 artifact schema. Incomplete evidence never yields a
        numeric duration. Worker service includes instrumentation between fetch
        entry and collate completion; it excludes worker queue/serialization
        work. Parent buffer residence begins only at the observed receive call.
        Neither interval estimates physical queue wait or prefetch efficiency.
        """
        data = self.to_dict()
        self._check(data)
        complete = data["analysis"]["status"] == "complete"
        service = [(p["receipt"]["finish_ns"] - p["receipt"]["receive_ns"]) / 1e6
                   for p in data["packets"]] if complete else []
        received = {r["receipt"]["token"]: r["at_ns"] for r in data["ledger"]
                    if r["kind"] == "receive"} if complete else {}
        residence = [(r["at_ns"] - received[r["receipt"]["token"]]) / 1e6
                     for r in data["ledger"] if r["kind"] == "deliver"] if complete else []

        def metric(values, scope):
            return dict(status="measured" if complete else "incomplete", scope=scope,
                        count=len(values), mean_ms=sum(values) / len(values) if values else None)

        stages = data["analysis"]["stage_timings"]
        return dict(status="partial" if complete else "incomplete",
            worker_service=metric(service, "worker_fetch_entry_to_collate_completion_including_trace_overhead"),
            parent_buffer_residence=metric(residence, "parent_receive_to_delivery_excludes_unobserved_queue_wait"),
            stages={stage: stages.get(stage, dict(status="unsupported", count=0, mean_ms=None))
                    for stage in ("dataset_fetch", "decode", "transform", "collate")},
            lifecycle=data["analysis"]["lifecycle"],
            queue_occupancy=data["analysis"]["queue_occupancy"],
            prefetch_efficiency=data["analysis"]["prefetch_efficiency"],
            cross_process_queue_wait=dict(status="unsupported", value=None,
                                          reason="no_accuracy_bound_for_cross_process_clocks"),
            overhead="included_not_separately_calibrated")

    def to_json(self, path, *, overwrite=False):
        data = self.to_dict()
        self._check(data)
        _write_export(path, json.dumps(data, indent=2, allow_nan=False) + "\n", overwrite)


def analyze_worker_tasks(loader_result, loader, *, worker_states=None):
    """Consume final same-run channel receipts; never rerun training.

    Stop workers first using bounded caller-owned cleanup. Calling twice is
    deliberately incomplete: channel evidence is consumed once.
    """
    if type(loader) is not WorkerTaskDataLoader or type(loader_result) is not DataLoaderAnalysisResult:
        raise TypeError("expected task loader and schema1 DataLoaderAnalysisResult")
    if loader.trace_source is None:
        raise ValueError("no completed source measurement binding")
    source = loader_result.to_dict()
    if (_hash(source) != loader.trace_source["source_p21_sha256"]
            or source["source_segment_id"] != loader.trace_source["source_segment_id"]):
        # Reject a cross-run result before consuming the caller's evidence.
        raise ValueError("task source binding mismatch")
    if worker_states is None:
        worker_states = [dict(epoch_id=epoch, worker_id=i, pid=worker.pid,
                             alive=worker.is_alive(), exitcode=worker.exitcode)
                         for epoch, workers in loader.trace_processes
                         for i, worker in enumerate(workers)]
    data = _plain(dict(schema_version=2, format="hyso.performance.worker_tasks",
        source_p21=source, binding=loader.trace_source,
        ledger=loader.trace_ledger, packets=loader.trace_channel.receive_available(max_items=100000),
        trace_counters=loader.trace_channel.trace_counters(), worker_states=worker_states))
    try:
        analysis = _derive(data)
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError("malformed task evidence") from exc
    digest = _hash(data)
    return WorkerTaskAnalysisResult(dict(data, evidence_sha256=digest, analysis=analysis))
