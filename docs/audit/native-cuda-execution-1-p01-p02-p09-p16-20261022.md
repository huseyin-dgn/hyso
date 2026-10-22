# Native CUDA Execution 1 — P01/P02/P09/P16

2026-10-22, Europe/Istanbul. **Execution PASS; combined requested coverage PARTIAL.**
Exactly one authorized small FP32 Conv2d scenario executed. Baseline0, retry0,
additional CUDA scenarios0, P23 diagnostics0. **B4 OPEN / root cause UNRESOLVED.**

Evidence directory:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-native-cuda-execution-1-20261022`.
Native work directory:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-gpu-policy-redesign-20261022/candidate/native-execution-1`.

## 1. Admission and hard safety gates

Entrypoint was the current `tools/forensics/gpu_scenario_runner.py`, whose hash
matches the sealed source. Parent and worker used its sealed copy. The old
`native_scenario_runner.py` and strict r3 evaluator were not used for admission.
The unchanged r3 module supplies constants/JSON helpers only.

Candidate run ID: `8fca9c14-469e-4413-8f5e-7e34c721bef4`.
Manifest SHA256:
`893da6512f5d3fa41c843ebc6e7c548ba5365f3c90d3214e7387df55897cf7b0`.
All51 sealed source files matched both disk and current repository sources;
all26 runtime identities matched before and after execution. No source,
runtime, manifest or seal was changed to obtain admission. Explicit approval
binds this run/manifest/scenario and maximum_executions1. Exclusive
`execution-consumed.json` and external `launch-consumed.json` prevent reuse.

The warning-only policy admitted the live three-sample capture. Initial GPU
free memory was5337/5337/5337MiB and utilization23/23/23%. Shared GPU processes,
unknown roles and WDDM N/A were retained as informational metadata. No PID was
authorized by its executable name. No user application was closed or process
terminated. The capture later aged beyond30s during setup/import and its values
became null/unavailable; this did not block execution.

Actual Job assignment, exact membership via duplicated query-only Job handle,
independent hard limit readback and parent/worker/token identity checks preceded
worker Torch/CUDA import. Controller/source/runtime checks and the prior dry-run
timeout/cleanup receipts were verified before launch. Actual receipts retained:

| Gate | Outer / parent | Child / worker |
|---|---|---|
| Private commit per process | 4294967296 B | 4294967296 B |
| Job private commit | 12884901888 B | 8589934592 B |
| Concurrent active process limit | 8 | 4 |
| Readback flags | 0x2308 | 0x2308 |
| Kill-on-close | true | true |
| Deadline | 90 s | 60 s |
| Observed elapsed | 29.625 s | 25.641 s |
| Peak process commit | 2077282304 B | 2077282304 B |
| Peak Job commit | 2101088256 B | 2087882752 B |
| Final active processes | 0 | 0 |

Allocator fraction0.15 was applied by the admitted worker before model/input
allocation. The limits are private commit limits, not RSS or physical VRAM caps.
Cumulative Job total_processes10/8 counts sequential helpers across the entire
run; these are not concurrent counts and do not violate active limits8/4.

## 2. Execution identity and actual CUDA initialization

| Role | PID | Parent PID | Exit code |
|---|---|---|---|
| Controller | 10980 | 17884 (external owned launcher) | 0 |
| Parent | 2316 | 10980 | 0 |
| Worker | 16900 | 2316 | 0 |

Controller elapsed33.562s. Worker actually initialized and used CUDA: validated
CUDA-resident input/output, real CUDA event timings and native allocator counters
prove completed CUDA operations. The runner did not record an explicit
`torch.cuda.is_initialized()` boolean; `postflight-identity-cleanup.json` states
this evidence basis rather than inventing an extra observation. Controller and
parent identity checkpoints recorded Torch not imported; their code paths perform
no workload imports. Parent/worker identity receipts are pre-import checkpoints.

Runtime: Python3.10.18, Torch2.5.1+cu121, CUDA build12.1, cuDNN90100, NVIDIA
GeForce RTX3060 Laptop GPU, capability8.6, driver591.74, native allocator,
default stream_id0. Model `Conv2d(1,3,3,padding=1)`, input `(2,1,7,9)`,
CUDA:0, FP32 input/output, eval, inference_mode, autocast disabled.

`torch.manual_seed(71)` preceded model/input construction. The unchanged
BenchmarkRunner configuration metadata reports its default seed0; that argument
is unused for generated input because this runner supplies an existing example
tensor. No seed0 reinitialization occurs in the supplied-input path. The model
and input were created from seed71. cuDNN allow_tf32=true was inherited and is
reported; FP32 tensor dtype is not a claim that internal math forbids TF32.

Sealed sample contract: warmup1, measured3, repeat1. Benchmark protocol additionally
performs one untimed preflight forward and one separate P09 memory-probe forward.
These belong to the same single scenario, not additional baseline executions.

## 3. Raw metrics, units and scopes

Machine-readable complete raw payload: `raw-worker-result.json` (copy of the
unchanged17409-byte worker result). Parsed metrics: `metrics.json`. The sealed
BenchmarkResult schema and sample/statistics/allocator consistency were revalidated
using its stdlib result implementation without another Torch import or workload.

### P01 — measured selected-stream CUDA event latency

Scope `serial_current_stream_forward`, synchronization
`repeat_device_drain_and_end_event_sync`; unit ms. Raw samples:

`[0.17404800653457642, 0.1483200043439865, 0.16230399906635284]`.

Mean0.16155733664830527ms; median0.16230399906635284ms;
min0.1483200043439865ms; max0.17404800653457642ms;
sample standard deviation0.012880242728871349ms.
Status measured, n3, repeat1. This is neither host wall latency nor kernel-only
time; event intervals may include host launch gaps and samples may correlate.

P99: **value null, status insufficient_samples, unit ms, reason tail_policy**.
The existing pure latency policy was reused offline on these same samples:
min_samples1000, min_tail_samples10; effective tail count0.03. No extra model run
or descriptive P99 was generated. Raw legacy p90/p95 values remain in the source
payload as descriptive interpolations; they are not tail-acceptance evidence.

### P02 — derived service rate only; bounded completion window not executed

Value12379.505886221727images/s; status derived;
scope `derived_serial_service_rate` from six images divided by the sum of three
CUDA event intervals. This finite sample run is bounded in count, but it does
**not** execute P02's separate bounded completion-window protocol or its wall
clock/count/pending/terminal conservation ledger.

Actual bounded completion throughput: **value null, status not_executed,
unit images/s, reason sealed_runner_only_uses_BenchmarkRunner**. The source
schema similarly records window_wall_images_per_sec=null, window_status=unsupported,
window_reason=separate_protocol_deferred. No sustained throughput, camera FPS,
or full P02 acceptance is claimed. Additional throughput workload was not added.

### P09 — measured native allocator, separate single forward

Unit bytes; scope owned worker/process/device native CUDA allocator,
`separate_single_forward`. Reset was explicitly enabled and performed.

| Raw counter | Bytes |
|---|---:|
| baseline_allocated | 2048 |
| baseline_reserved | 2097152 |
| previous_peak_allocated | 4096 |
| previous_peak_reserved | 2097152 |
| peak_allocated | 3584 |
| peak_reserved | 2097152 |
| incremental_peak | 1536 |
| allocated_after_probe | 3584 |
| reserved_after_probe | 2097152 |
| allocated_after_output_release | 2048 |
| reserved_after_output_release | 2097152 |

Output release returned allocated bytes to baseline. The reserved pool remains
cached. Previous peaks are retained as prior-counter evidence; reset irreversibly
changed only the owned worker's allocator peaks. These are not total physical
VRAM, another process's memory, or non-allocator CUDA allocations.

### P16 — measured read-only physical-device sensors

Backend nvidia-smi.NVML; domain physical_device_wide. GPU UUID
`GPU-9eec2f80-4c4e-a5cd-7091-03f0d013cf11`, PCI00000000:01:00.0.

| Metric | Before | After | Unit | Status |
|---|---:|---:|---|---|
| GPU busy | 39 | 13 | percent | measured |
| Memory-controller busy | 26 | 1 | percent | measured |
| VRAM used | 871366656 | 892338176 | bytes | measured |
| VRAM free | 5414846464 | 5393874944 | bytes | measured |
| VRAM total | 6442450944 | 6442450944 | bytes | measured |

Sensor wall timestamps:2026/10/22 14:15:53.397 and14:15:53.862 (Europe/Istanbul).
Raw perf-counter query intervals:2077173597800–2077250691800ns and
2077651606700–2077716102500ns. These are before/after samples with vendor sampling
windows, not exact integration over the three forwards. No process GPU,
Tensor Core utilization, energy, or worker-attributed VRAM is inferred.
Unavailable hardware fields remain null/status/reason in the evidence envelope.

## 4. Contention, failure and cleanup

Warning codes: `shared_gpu_processes`, `unknown_or_compute_gpu_process`,
`wddm_identity_or_memory_unsupported`, `gpu_telemetry_stale_or_unavailable`.
Measurement quality `unqualified`; no exclusive-device or controlled-comparison
claim. Low memory or utilization thresholds did not gate execution.

**OOM false, native crash false, timeout false.** Parent/worker monitor outcome
null; external controller watchdog outcome null. There was no retry or fallback.

Controller stdout775bytes/stderr0; parent stdout0/stderr0; worker stdout0/stderr0.
All channels below65536bytes. Worker result17409bytes, below2MiB; all work JSON
files bounded, count below32. No unbounded event stream was produced.

Outer and child Job active_processes0; all observed owned PIDs plus controller
were independently checked with read-only process exit APIs:
`868,2252,2316,4840,10980,11412,11760,13152,14976,16900,20024`.
All absent/exited. Termination occurred only through newly owned Jobs; no
inventory/user PID was passed to a termination API. Native receipt details and
per-PID status are retained in `postflight-identity-cleanup.json`.

## 5. Source protection and quality matrix

Branch `feature/hyso-models-next`; HEAD
`a1864094cf41d242d75f634dd7fc134436fae6d6`; index empty. All original13 dirty
files and10 prior forensic files retain their SHA256/status/mode. All previous
audit records and old r3 evidence remain unchanged. Sealed source/manifest/seal
and26 runtime hashes match before/after. Only expected append-only native work
artifacts and the single execution marker were added to the existing candidate.

Repository changes: this new audit and the additive current quality-matrix
section7.1 in `docs/design/hyso-analysis-optimization-contract.md`. No production,
runner, test, guard, or previous audit source changed. No commit/push/reset/clean/
stash/checkout occurred. Full start/final hashes and statuses are in
`start-protection.json`, `final-protection.json`, `evidence-manifest.json`.

The current17-feature quality matrix now records P01/P09/P16 as PASS only within
these narrow measured scopes, P02 PARTIAL, and the other13 features NOT_EXECUTED
in this run. Backlog implemented/partial/planned counts are not rewritten.

## 6. Decision

**PARTIAL** for the requested combined P01/P02/P09/P16 coverage, because actual
P02 bounded completion throughput was not in the sealed execution contract.
The single authorized native inference and hard safety gates **PASS**; P99 is
correctly insufficient_samples. No second workload was used to fill the gap.

This result does not resolve the historical B4 native crash root cause, enable
P23 hooks-enabled CUDA, or complete17/17 CUDA acceptance. No further CUDA work
is authorized or started by this result.
