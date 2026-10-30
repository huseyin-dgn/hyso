# P35 execution-provider evidence

P35 supports ONNX Runtime profiler evidence. It does not infer operator placement
from a PyTorch tensor's device or from a session's enabled-provider list. The
available, requested, enabled and actually observed providers remain distinct.
Available providers describe that ORT build; they do not prove successful
initialization or provide per-provider DLL versions.

```python
from hyso.performance import RuntimeResult

saved = RuntimeResult.from_json("provider-result.json")
view = saved.provider_diagnostics()
assessment = view["evidence_assessment"]
```

This replay performs no inference, runtime initialization or device/sensor query.
New `profile_execution_providers` results declare `provider_analysis_version=2`.
This is independent of `graph_analysis_version=2`, which handles narrowly
validated optimized-graph transfer aliases. RuntimeResult schema v1, public
signatures and canonical lossless JSON/CSV envelopes are unchanged. Old evidence
without the new marker retains its original serialized analysis exactly; the
additional diagnostic view does not rewrite its source.

Requested providers retain priority order. The assessment reports a changed
session order as configuration evidence, not as executed fallback. CPU listed
but unused is not CPU execution. An explicit CPU-only configuration is reported
as such, and mapped CPU execution matching CPU expectations is not an expectation
mismatch. A mapped CPU event where CUDA was expected is observed placement
mismatch, not a proven failure/retry transition.

ORT documents provider priority separately from `disable_fallback()`'s
session.run error-recovery mechanism: [Python API](https://onnxruntime.ai/docs/api/python/api_summary.html),
[execution providers](https://onnxruntime.ai/docs/execution-providers/).
Legacy schema1 `fallback_detected` retains its expectation-mismatch meaning.
`execution_fallback.detected` and `execution_error_retry.detected` remain null
without the required temporal failure/retry evidence. An expectation is never
reported as the node's previous actual provider.

The assessment records:

- Model/input/factory/profile hashes, runtime version/build, session association,
  provider priority, requested options and observed execution providers.
- Source-node mapping and coverage, runtime node IDs, raw trace indices,
  timestamps, event identity and measured/derived distinctions.
- Per-node expectation findings with scope, expected/observed provider,
  operator, evidence source, limitation and a next validation only when needed.
- Recognized optimized-graph transfer operators, their host profiler durations,
  directions and explicit missing byte/device-timing reasons.
- Graph producer/consumer provider boundaries as derived dependency evidence,
  distinct from measured transfer events.
- Same-node provider changes only when recorded events actually show them.
- Partial/aborted status and unknown-tail limitations, without promoting
  completed counters or an exit code to node-execution evidence.

The source graph must be self-contained and pass ONNX validation. Subgraphs and
external tensor data are outside this adapter. Source attribution requires
unique names, disabled optimization and equivalent source/optimized definitions,
apart from the existing narrow transfer aliases. Fusion/renaming stay unknown.
Repeated executions of the same node are legitimate; an exact duplicate profile
event is not a second execution. Conflicting runtime node IDs are rejected.
Missing node IDs/timestamps/providers and incomplete source coverage cannot
establish a complete negative fallback result.
For multiple completed runs, aggregate node counts are insufficient: every
source event must belong to exactly one complete, nonoverlapping ORT `model_run`
window, with one event per source node in each run. Missing windows or a missing
last-run profile leave negative placement conclusions unknown.

New captures record observed input fingerprints before execution, verify that
the feed remains unchanged, and record ordered host intervals for warmup and
measured `session.run` calls. A new capture missing these fields is invalid.
Older captures retain null plus a legacy-context reason. Host perf_counter_ns
and ORT JSON host-relative microseconds are separate clock domains; no alignment
or subtraction between them is assumed. Profiler time is not CUDA kernel time.

Durations with missing samples keep `total_host_node_us=null` and an explicit
`known_duration_sum_us`/coverage. A partial sum is not a total. Overlapping host
Node durations are not session wall time. ORT `model_run` events are reported
as instrumented session observations, not a benchmark speedup or latency penalty.

An actual `MemcpyToHost`/`MemcpyFromHost` event supports a runtime transfer-op
observation. Its host duration is not device DMA time; activation/output tensor
sizes are not a measured DMA byte counter. Provider boundaries alone do not
manufacture transfers. Physical GPU UUID execution identity remains unavailable
when the profiler records only provider names; requested UUID visibility and
device options are selection intent and are checked for internal consistency.

Operator support, complete static graph assignment and provider-specific
library versions are null with reasons when not recorded. This adapter does not
use newer runtime-specific assignment APIs to retrofit older captures. Generic
NotImplementedError during execution is not evidence that a requested provider
was absent. Input validation, provider selection, session initialization and
execution failures have separate phases; unsupported operator identity is not
extracted speculatively from an exception string.

Fallback alone does not quantify its performance effect. The assessment returns
`needs_comparable_experiment`, null latency delta and null percentage when a
placement mismatch lacks a matching counterfactual experiment. No optimization
gain, loss or synchronization-wait amount is invented.

Model A/B checkpoint bytes can be verified without exporting them. Their
PyTorch records do not become ORT node profiles. An absent trained ONNX export
stays unsupported; this feature does not convert checkpoints to obtain a PASS.

Existing owned Windows Job cleanup, finite experiment/evidence budgets,
deadlines and crash-stop are retained. No default RAM/VRAM quota, allocator
fraction or hardware batch cap is introduced. Run-error retry is disabled by
the existing adapter; constructor behavior of older runtime versions is not
claimed to be instrumented. New risky/GPU experiments require a fresh exact
manifest and explicit approval. Old authorizations and P31's deferred manifest
do not authorize a new run.
