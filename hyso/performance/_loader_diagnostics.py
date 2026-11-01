"""Pure P21 diagnosis of saved evidence. Never operate a Dataset or DataLoader."""
from .dataloader import DataLoaderAnalysisResult, _hash


def _unsupported(reason):
    return dict(status="unsupported", value=None, reason=reason)


def analyze(result, *, worker_tasks=None, worker_traces=None):
    if type(result) is not DataLoaderAnalysisResult:
        raise TypeError("DataLoaderAnalysisResult required")
    if worker_tasks is not None and worker_traces is not None:
        raise ValueError("choose one worker companion; overlapping observations must not be counted twice")
    data = result._export_data()
    source_hash = _hash(data)
    worker = dict(status="unsupported", association=dict(execution="unknown", epoch="unknown", batch="unknown"),
        stages={s: dict(status="unsupported", count=0, mean_ms=None) for s in ("dataset_fetch", "load", "decode", "transform", "collate")},
        service=_unsupported("no_verified_worker_trace"),
        parent_buffer_residence=_unsupported("no_verified_parent_receive_delivery_receipts"))
    task_links = {}
    dispatch = _unsupported("no_verified_parent_index_queue_put_receipts")
    lifecycle = _unsupported("no_verified_task_lifecycle_receipts")
    if worker_tasks is not None:
        from .worker_task_result import WorkerTaskAnalysisResult
        if type(worker_tasks) is not WorkerTaskAnalysisResult:
            raise TypeError("WorkerTaskAnalysisResult required")
        raw = WorkerTaskAnalysisResult(worker_tasks.to_dict()).to_dict()
        if raw["source_p21"] != data or raw["binding"]["source_p21_sha256"] != source_hash:
            raise ValueError("worker task companion belongs to a different P21 source")
        diag = worker_tasks.diagnostics()
        dispatch, lifecycle = raw["analysis"]["dispatch"], raw["analysis"]["lifecycle"]
        worker.update(status=raw["analysis"]["status"], association=raw["analysis"]["association"],
            stages=diag["stages"], service=diag["worker_service"], parent_buffer_residence=diag["parent_buffer_residence"])
        if raw["analysis"]["status"] == "complete":
            packets = {p["receipt"]["token"]: p for p in raw["packets"]}
            received = {r["receipt"]["token"]: r["at_ns"] for r in raw["ledger"] if r["kind"] == "receive"}
            for row in raw["ledger"]:
                if row["kind"] != "deliver":
                    continue
                receipt = row["receipt"]
                packet = packets[receipt["token"]]
                task_links[receipt["epoch_id"], receipt["batch_id"]] = dict(
                    worker_id=receipt["worker_id"], worker_pid=receipt["worker_pid"],
                    worker_service=dict(clock_scope="worker_local", milliseconds=(receipt["finish_ns"]-receipt["receive_ns"])/1e6,
                        scope="fetch_entry_through_collate_completion_including_trace_overhead"),
                    stages=[dict(stage=e["stage"], sample_id=e["sample_id"], duration_ms=e["duration_ms"], clock_scope="worker_local")
                            for sample in packet["samples"] for e in sample["events"]],
                    collate=dict(clock_scope="worker_local", duration_ms=packet["collate"]["duration_ns"]/1e6),
                    parent_buffer=dict(clock_scope="parent_local", receive_ns=received[receipt["token"]],
                        delivery_ns=row["at_ns"], residence_ms=(row["at_ns"]-received[receipt["token"]])/1e6,
                        scope="observed_parent_receive_to_delivery_not_physical_queue_residence"))
    elif worker_traces is not None:
        from .worker_trace_result import WorkerTraceAnalysisResult
        if type(worker_traces) is not WorkerTraceAnalysisResult:
            raise TypeError("WorkerTraceAnalysisResult required")
        raw = WorkerTraceAnalysisResult(worker_traces.to_dict()).to_dict()
        if raw["source_p21"] != data or raw["source_p21_sha256"] != source_hash:
            raise ValueError("worker trace companion belongs to a different P21 source")
        worker.update(status=raw["analysis"]["status"],
            association=raw["analysis"].get("association", worker["association"]))
        for stage in worker["stages"]:
            if stage in raw["analysis"]["stage_timings"]:
                worker["stages"][stage] = raw["analysis"]["stage_timings"][stage]
        # Legacy packets carry sample identity, not verified epoch/batch identity.
        # No retrospective packet-to-batch matching or task timeline is invented.

    device = data["source_p19"]["provenance"]["runtime"]["device"]
    links = {r["microbatch_id"]: r for r in data["stage_links"]}
    timeline = []
    for row in data["observations"]:
        if row["kind"] != "batch":
            continue
        stage_link = links.get(row["microbatch_id"]) if row["phase"] == "training" else None
        compute = {}
        for stage in ("forward", "loss_metrics", "backward"):
            recorded = stage_link["compute"][stage] if stage_link and stage_link["status"] == "linked" else None
            compute[stage] = (dict(status=recorded["status"], host_wall_ms=recorded["host_ms"],
                selected_stream_cuda_event_ms=recorded["cuda_event_ms"]) if recorded else
                dict(status="unsupported", host_wall_ms=None, selected_stream_cuda_event_ms=None))
        preparation = stage_link["input_preparation"] if stage_link and stage_link["status"] == "linked" else None
        cpu_fb = (sum(compute[k]["host_wall_ms"] for k in ("forward", "backward"))
                  if device.startswith("cpu") and all(compute[k]["status"] == "measured" for k in ("forward", "backward")) else None)
        timeline.append(dict(epoch_id=row["epoch_id"], phase=row["phase"], batch_index=row["batch_index"],
            microbatch_id=row["microbatch_id"], samples=row["samples"],
            parent_next=dict(start_ns=row["start_ns"], delivery_ns=row["delivery_ns"], duration_ms=row["host_ms"],
                clock_scope="parent_local", scope="next_call_wall_not_worker_compute_or_queue_residence"),
            input_preparation=(dict(status=preparation["status"], host_wall_ms=preparation["host_ms"],
                selected_stream_cuda_event_ms=preparation["cuda_event_ms"]) if preparation else
                dict(status="unsupported", host_wall_ms=None, selected_stream_cuda_event_ms=None)),
            compute=compute, cpu_forward_backward_wall_ms=cpu_fb,
            worker_task=task_links.get((row["epoch_id"], row["batch_index"])) if row["phase"] == "training" else None))

    findings = []
    core = result.core_metrics()
    for summary in core["summaries"]:
        selected = [r for r in timeline if (r["epoch_id"], r["phase"]) == (summary["epoch_id"], summary["phase"])]
        wait = sum(r["parent_next"]["duration_ms"] for r in selected)
        comparable = bool(selected) and all(r["cpu_forward_backward_wall_ms"] is not None for r in selected)
        compute = sum(r["cpu_forward_backward_wall_ms"] for r in selected) if comparable else None
        preparation = (sum(r["input_preparation"]["host_wall_ms"] for r in selected)
            if selected and all(r["input_preparation"]["status"] == "measured" for r in selected) else None)
        relation = ("observed_batch_acquisition_wait_exceeds_CPU_forward_backward_wall" if comparable and wait > compute else
                    "CPU_forward_backward_wall_at_least_observed_batch_wait" if comparable else
                    "no_comparable_CPU_forward_backward_scope")
        findings.append(dict(epoch_id=summary["epoch_id"], phase=summary["phase"], relation=relation,
            parent_delivered_next_wall_ms=wait, cpu_forward_backward_wall_ms=compute,
            input_preparation_host_wall_ms=preparation,
            iterator_init_ms=summary["iterator_init_ms"], samples_per_second=summary["throughput"]["samples_per_second"],
            throughput_scope="achieved_phase_wall_including_consumer_work_not_standalone_loader_capacity"))
    return dict(format="hyso.performance.dataloader_diagnostics", schema_version=1,
        status="observational", source_p21_sha256=source_hash, device=device,
        core=core, batch_timeline=timeline, findings=findings, worker=worker,
        queue_task_scopes=dict(
            parent_index_queue_put=dispatch, task_lifecycle=lifecycle,
            parent_buffer_residence=worker["parent_buffer_residence"],
            physical_queue_residence=_unsupported("no_verified_queue_boundary_clock_mapping"),
            queue_occupancy=_unsupported("no_physical_queue_boundary_receipts"),
            prefetch_efficiency=_unsupported("buffer_hits_or_next_wait_are_not_efficiency")),
        recommendations=[],
        limitations=[
            "Parent next wait is wall time; worker load/decode/transform are separate worker-local wall intervals, not CPU busy time.",
            "Forward/backward host windows measure CPU execution only for CPU training; CUDA event and host clocks stay separate.",
            "Dataset fetch includes nested load/decode/transform; do not sum overlapping stages or parallel workers into phase wall time.",
            "Worker-local timestamps are never subtracted from parent timestamps; task links use verified identity, not timestamp proximity.",
            "Iterator startup, consumer pauses, trace overhead and OS scheduling can affect observations; no causal speedup is inferred.",
            "No worker-count or prefetch change was benchmarked; no such recommendation is made."])
