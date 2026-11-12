# Selective analysis: selection, preflight and explicit execution

The common API reuses G1 planning, the G2/G3 CPU bridges and native producers.
It is not the full acceptance runner and does not reimplement measurement math.

## Coverage and profiles

| Common execution mode | Connected features | Scope |
|---|---|---|
| trusted_cpu | P01, P02, P10, P11 | Caller-owned, trusted local CPU PyTorch classification models |
| isolated_cpu | P02, P11 | Existing source-hashed trusted factories in Windows Job workers |
| Other features | Planning only | Original performance APIs can exist without a common adapter |

`feature_catalog()` describes all P01–P35 IDs, requirements, capabilities,
dependencies, applicability, limitations and qualitative workload classes.
`supported_devices` describes connected common adapters, not the coverage of all
original native APIs. GPU-required features retain that prerequisite. No CUDA
query, GPU execution, training, camera, energy sensor, Nsight or ONNX invocation
is connected here. Those paths still need separate native resource/protocol
preflight and explicit authorization before future integration.

`plan_analysis(context, features="P11")` selects one feature; `features=("P01",
"P11")` selects a list. Alternatively use exactly one `profile`:

| Profile | Explicitly selected IDs |
|---|---|
| basic | P01, P02, P10, P11 |
| inference | P01–P08 |
| training | P19–P25 |
| memory | P09, P10, P11, P23, P32 |
| comparison | P01, P02, P09, P10, P11, P26, P29 |
| full | P01–P35, **selection only** |

An existing `FeaturePlanRequest` is also accepted unchanged, including G1
single/selected/automatic/full modes. One or two ModelContexts are supported.
Dependencies are visible but never automatically authorized or executed. For
example selecting P32 shows P31 as a dependency-only SKIPPED row. P31 still
requires one uninterrupted 7200-second source; P32 still requires validated
source identity/continuity and memory-series evidence. Neither is connected to
this executor. There is no shortcut from a short benchmark to P31/P32 acceptance.

## Complete tiny CPU example

Selection performs no hardware queries or workload execution. Preflight inspects
model/input metadata, required declarations and OS RAM/commit estimates without
forward, dataset iteration or factory import:

```python
import torch
from torch import nn
from hyso import analysis

torch.manual_seed(0)
model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
inputs = torch.ones(2, 1, 2, 2)
plan = analysis.plan_analysis(
    analysis.ModelContext("toy", input_contract=True), profile="basic")
dry_run = analysis.preflight_analysis(
    plan, (analysis.LiveCPUModel("toy", model, inputs),))
print(dry_run.to_dict())
```

Review selected IDs, dependencies, per-model reasons, counts, limits and estimates.
Then authorize an explicit subset. Here only P11 runs; P01/P02/P10 are SKIPPED:

```python
result = analysis.execute_analysis(
    dry_run, authorized_features=("P11",), explicit_authorization=True)
rows = result.to_dict()["features"]
print([(row["feature_id"], row["status"]) for row in rows])
```

To request all connected basic features, explicitly authorize
`("P01", "P02", "P10", "P11")` instead. A dry-run ticket is one-shot; preparing
another ticket is an explicit new action, never an implicit retry. Model/binding
identities must match. Resource preflight is refreshed before execution; changed
binding metadata or plan requires a new review. Availability can change afterward.

## Status and measurement boundaries

| Status | Meaning |
|---|---|
| SUPPORTED | Adapter exists and the current metadata/resource preflight passed; not a measurement |
| UNSUPPORTED | Selected device/task/backend/OS or native RSS backend is outside this route |
| MISSING_PREREQUISITE | Missing declarations/binding/resources, or requested P01 quantiles lack sufficient native samples |
| NOT_IMPLEMENTED_IN_EXECUTOR | No common adapter; this does not mean the native API is absent |
| COMPLETED | Authorized native result obtained and admitted by its original reader; not acceptance PASS |
| FAILED | Native execution/validation failed; the reason is retained |
| SKIPPED | Dependency-only or supported but not authorized; no adapter called |

`executor_support` is separate from runtime `status`: an unconnected feature may
also lack prerequisites. Ordinary per-row failures do not stop independent rows
or models. KeyboardInterrupt/SystemExit propagate. Trusted arbitrary Python can
have effects outside native model guards; no general rollback promise is made.

P01 uses original BenchmarkRunner samples plus analyze_latency. Native warmup,
preflight forward, model-state/RNG guards and disabled CUDA memory probing remain.
The default policy requests 1000 samples, one warmup, one repeat and quantiles
.5/.95/.99. Native TailPolicy stays min_samples=1000, min_tail_samples=10; smaller
explicit budgets are preserved and can yield MISSING_PREREQUISITE with valid raw
evidence retained. Neither correlated samples nor sufficient tails certify an
independent confidence interval or acceptance result.

P02 delegates to G2/G3 continuous completed throughput, not latency-derived
throughput. P10 invokes `measure_memory()` without a workload: exactly one
process RSS snapshot and OS lifetime HWM, no forward, sampler workload or model
peak attribution. P11 delegates logical parameter/buffer bytes without artifact
serialization or checkpoint access. RSS, HWM, logical bytes and VRAM stay separate.

CPUAnalysisPolicy validates bounded operation counts, model/input logical bytes
and configurable minimum RAM/commit availability. Those estimates neither
reserve memory nor bound forward activation memory or elapsed time. Trusted
in-process models can hang or perform arbitrary Python effects. Use only reviewed
CPU code. The isolated route retains existing hard Windows Job timeout/commit/
child limits and source checks; it never falls back to in-process execution and
is still not a hostile-code security sandbox. G3 policy counts must match the
common plan; no supplied value is silently rewritten. Each eligible model/feature
uses its own worker to retain independent failure handling.

Live P01/P02 inputs must be dense, contiguous, no-grad CPU fp32 NCHW tensors;
floating model tensors must already be CPU fp32. No device transfer, dtype cast
or fallback is silently applied. P11 retains the original CPU footprint profile,
including its supported fp64 logical tensors. Dry-run tickets belong to the
calling thread; concurrent ticket use is outside this profile.

## Evidence, interpretation and optional tracking

```python
index = result.to_evidence_index()
report = result.to_unified_report(context=analysis.AnalysisContext(
    workload_id="toy-example", task="classification",
    model={"id": "toy"}, dataset={"kind": "synthetic"}))
from hyso import advisor
interpretation = advisor.advise(report)  # Separate, read-only interpretation.
# result.to_json("execution.json")       # Explicit persistence; no-clobber default.
# index.to_json("evidence-index.json")
# report.to_json("analysis-report.json")
```

Native payloads, P01 auxiliary raw benchmark evidence, execution receipts and
executor statuses remain distinct. `EvidenceRecord.from_native()` validates
in-memory canonical JSON using existing readers. Its artifact hash/size names
retained JSON bytes, not a file already written to disk. Model IDs are caller
associations; process snapshots do not become model-exclusive measurements.
Unified report native-validation coverage differs from executor status and never
implies acceptance PASS; non-completed executor outcomes are also carried as risks.

ExperimentManager requires no new run type or execution hook: explicitly create
and start a `benchmark` run with `dry_run.tracking_config()` as config, export the result,
then call the existing `run.add_artifact(path, kind="analysis_execution",
ownership="copy", role="output")`, finish the run and close the manager. Tracking
is explicit and does not rerun measurements. Use a dedicated non-synced store.
The Manager config projection carries the plan ID, IDs/device/mode and unchanged
policy. Approval flags and factory kwargs stay in the full execution artifact;
the existing Manager restriction on auth-named snapshot keys is preserved.
Artifact APIs require an existing schema2 store. New Manager stores intentionally
start at schema1; if needed, close handles and perform the existing explicit,
offline `ExperimentManager.migrate_store(root_dir)` before reopening. This
executor never migrates or changes a store automatically. Schema1 can still
persist the benchmark run/config without artifact attachment.
No metric values are manufactured from missing results or aggregated across
different model scopes. Comparison-profile selection does not certify a winner.

G1/G2/G3 canonical import paths and original native protocols remain available.
The historical 70/70 acceptance evidence and runner thresholds are unchanged.
