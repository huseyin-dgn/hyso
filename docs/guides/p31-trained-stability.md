# P31 trained-model stability

P31 retains the existing `measure_stability`, owned Windows worker, Job cleanup
and hash-chained journal. New workers write `stability_version=2` with exact
successful step brackets `[start_ns, end_ns, completed_images]` in each observed
window. Cumulative completed/steps/work counters, the raw brackets and final
completion boundary must agree. Rehashed gaps, duplicates, ordering errors and
contradictory completion evidence are rejected.

`RuntimeResult.stability_diagnostics()` adds `latency_throughput_windows`, global
`latency` and an actual `completion_window`. Latency is the successful synchronous
workload call, including its output/state validation. It is not isolated kernel
time. Completion-window image throughput divides actual completed images by the
first successful step start to last successful completion span. Window throughput
uses observation boundaries and includes monitoring/journal time. Inference calls
and completed images are distinct: one call completes the recorded batch.

P50/P95 are exact empirical quantiles, linearly interpolated at `(n-1)*q`. P99
requires at least 100 completed calls; otherwise it is null with a reason. This
minimum permits an empirical upper-percentile observation and gives no population
tail confidence guarantee. Empty windows have no fabricated latency. Health gaps
remain visible and are not interpolated. Cumulative mean/stddev and the existing
early/late drift diagnostic remain descriptive; a single slowdown does not prove
persistent drift or thermal throttling.

Process RSS, private commit, CUDA allocated/reserved and physical-device readings
remain separate domains. New `process_commit` observations use Windows
`PROCESS_MEMORY_COUNTERS_EX.PrivateUsage`, which reports process private commit,
not RSS. See [Microsoft's structure definition](https://learn.microsoft.com/en-us/windows/desktop/api/psapi/ns-psapi-process_memory_counters_ex).
The query brackets use the worker's `perf_counter_ns` clock. The parent reuses the
existing OS low-memory notification monitor and terminates only its owned Job on
resource pressure. Default process/job RAM, process-count and allocator quotas
remain unset. The protocol still has finite duration, startup/heartbeat deadlines
and caller-selected journal/sample budgets. Raw step evidence counts against the
evidence budget; it is not silently sampled away or turned into a hardware limit.

The worker records factory, work and close boundaries; the controller records its
own start/process/cleanup clock. No cross-domain duration is computed. Missing
private commit, CPU-only GPU power/temperature and unrecorded sensors are null
with reasons. RSS/private commit growth is not a leak diagnosis. No P32 slope or
allocation lifetime model is added. Execution retry/resume is unsupported; reopen
recovers validated evidence only.

Legacy P31 analysis and flattened CSV export stay compatible. New version-2 P31
CSV uses the existing canonical runtime envelope and supports lossless
`RuntimeResult.from_csv`, including empty arrays and null measurements. Generic
runtime export readers retain their 128 MiB input guard.

`tools/p31/trained_acceptance.py` provides distinct `plan`, `smoke`, `prepare`,
`preflight` and hash-bound `start` actions. It has no default execution action.
The reviewed adapter loads the existing six-epoch Model B checkpoint with the
public weights-only checkpoint loader. Its CPU FP32 batch-256 resident input and
uint8/255 preprocessing match the existing recorded inference conditions; it
performs no training or P17/P28/P29 comparison. One setup warmup is explicitly
outside the observed-run counters. The three-second smoke verifies the adapter
and cleanup only.

The new long-run manifest fixes **7200 observed seconds** on that trained CPU
workload, with 5-second observations and one-row durable chunks. It separately
pins the checkpoint, dataset, source overlay, interpreter, packages and expected
outputs. Start needs the exact approved manifest SHA256 and a fresh one-shot
reservation. The existing outer observer records launcher exit and cleanup;
native errors/interruption never cause retry. Preparation and preflight make no
factory, CUDA or worker calls. The historical 7200-second GPU fixture plan is
preserved and is not authorized or replaced by this CPU plan.

A completed short protocol means only that its requested short window completed.
P31 acceptance remains PARTIAL until the actual long run, successful terminal,
observed duration, sample gaps, failures, exports and both cleanup receipts have
been reviewed. Separate short runs cannot be joined into long-run evidence.
