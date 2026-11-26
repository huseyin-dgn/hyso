# Targeted CUDA functional completion — pending hardware acceptance

This is an opt-in development validation scope. Existing support-matrix VERIFIED
or SUPPORTED cells are unchanged. No new GPU acceptance is claimed by preparation.
The completed CPU campaign and its 8536 source, 55 infrastructure, 36 package and
59 SupportScope successes remain historical evidence of their original sources.
Changed source requires the targeted checks below; the old checkpoint is read-only.

## APIs and measurement boundaries

| Feature | CPU compatibility | Explicit CUDA implementation | Limits |
|---|---|---|---|
| P21 | Existing schema 1 and iterator/worker semantics retained | Trainer's actual CPU→CUDA `Tensor.to` calls recorded in schema 2, tied to phase/epoch/batch/role, host nanoseconds, bytes, pinned flag, selected stream and completed CUDA events | Blocking call host duration and inclusive stream event time are separate; neither is exclusive DMA time, overlap, worker service, nor GPU idle time. Ungathered worker data stays unavailable. |
| P24 | `measure_time_to_quality` defaults to CPU | New keyword `device='cuda:0'`; fresh ordinary same-device Trainer per seed; identical cohort, training data and configuration checks retained | 2–8 seeds, ≤10 epochs, ≤256 samples; no early stopping/EMA/callback injection from the caller. Unreached targets remain censored with no invented time. Initial weights may differ by recorded seed under the same initialization protocol. |
| P27 | CPU perturbation/cohort and default API preserved | New keyword `device='cuda:0'`; clean and corrupted cohorts evaluated on the same device | Seeded CPU corruption, same sample IDs/targets, owned model copies, mode/gradient/RNG guards; no fallback or user-model rewrite. |
| P29 | Comparison versions 1–3 remain readable; CPU capture API unchanged | New keywords `device='cuda:0'`, `allow_cuda_peak_reset=False`; GPU-resident evaluation/timing, comparison version 4, UUID/runtime/stream/model/cohort/input/precision binding | FP32/AMP FP16 timing. Allocator allocated/reserved peaks require explicit irreversible reset consent in an exclusive owned worker. Logical bytes are not physical VRAM; physical VRAM stays unsupported. Missing/mismatched evidence cannot imply superiority. Advisor imports retained evidence read-only; it does not run these measurements. |
| P30 | CPU frame processor, queue/drop/completion evidence retained | `FrameInferenceProcessor(...,device='cuda:0',gpu_uuid=...,cuda_memory_fraction=.15)`; explicit bounded GPU WorkloadRecipe + StreamProtocol; completed forward event before frame completion | UUID-visible device 0, explicit process/job/process-count caps. Recorded local 8-frame MJPG first; no camera required. Transfer, preprocessing, inference and queue accounting remain distinct. |

Native QualitySource device options must match the analysis binding. Peak reset
also needs current-PID OperationPolicy consent. GPU streaming requires explicit
file-write/CUDA policy and typed budgets. Public defaults remain CPU. This does
not add CLI flags or rewrite TR/EN/DE/ES reporting: existing retained report APIs
remain available in all four languages.

## One PowerShell entry point

```powershell
# Static preparation only: no Torch import, test collection, GPU query or execution.
.\run_final_acceptance.ps1 -CudaScope -PrepareOnly
# User starts the bounded CPU regressions, then seven serial GPU groups.
.\run_final_acceptance.ps1 -CudaScope
# Continue only unfinished groups; terminal FAIL/BLOCKED are not retried automatically.
.\run_final_acceptance.ps1 -CudaScope -Resume
.\run_final_acceptance.ps1 -CudaScope -Status
# Only after correcting the defect: explicitly select terminal failures/holds.
# CPU targets are exact nodeids from plan.json, GPU targets are feature IDs.
.\run_final_acceptance.ps1 -CudaScope -Resume -Revalidate P16,P29
```

Checkpoint/evidence: `outputs/cuda-functional-acceptance-20261126/`.
Each attempt has a unique directory, source/runtime identity, original controller
receipts, fsynced hash-chain journal, per-feature JSON and cleanup-sealed evidence.
Plan changes archive previous plan/results and invalidate affected scope. Resume
reconciles an interrupted attempt only with verified final cleanup and bound raw
evidence; unknown owner/cleanup state blocks execution, never PID-based killing.
No automatic retries. Both CPU and CUDA managers use the existing shared Job lock,
including prepare-only. CPU outcomes are retained per nodeid. Normal resume skips
PASS, FAIL and BLOCKED; explicit revalidation runs only the requested cases and
their necessary fresh shared captures, archiving previous failure evidence.
Current valid PASS cannot be overwritten. Affected source transitions retain old
PASS as historical evidence with `source_valid=False`; those visible holds need
explicit selection too. Unaffected PASS stays valid. Unknown owned cleanup never
permits revalidation. Safety stops require explicit selection of all held features.
Live status reports group/feature (CPU nodeid), CPU and GPU PASS/FAIL/BLOCKED,
heartbeat UTC and elapsed seconds. In-flight counters are provisional until
controller cleanup; source-invalidated results are excluded from current counts.

## Groups

1. GPU admission: exact Job/PID + creation time/image, nonce, GPU UUID/driver,
   loaded nvcuda/torch_cuda hashes, Torch/CUDA runtime and bounded CUDA probe.
2. ModelBuilder/GraphModelBuilder, inference, P01/P02/P03/P04/P05/P09/P10/P11.
3. Batch/resolution/precision sweeps P06/P07/P08, evaluation P26/P28.
4. Shared profiling P12/P13/P14, transfer P15, utilization P16, finite P17 search.
5. Normal Trainer/P19/P20/P22 share one fresh episode; P23 basic memory and
   P25 AMP gradients use separate fresh episodes. No saved-tensor hooks.
6. P21/P24/P27/P29/P30, each with its own small fresh capture.
7. Short P32 memory trend, P34 CPU/CUDA comparison, P35 explicit ORT CUDA runtime.

P05's direct cold implementation remains historical evidence, while its imported
quality/runtime dependency sources changed. The plan records those hash deltas;
group 2 adds one fresh owned GPU interpreter, 60-second timeout, one warmup and two
warm references. The allocator is capped at 15% before device movement. Neither
OS page-cache coldness nor a new artifact-loading claim is inferred. Evaluation
is tested explicitly; Advisor comparison remains offline. FLOPs are logical ATen
cost, not hardware counters; layer events are inclusive, not kernel-exclusive.
Unavailable actual kernel traces/sensors remain BLOCKED. P17 reports only the
finite tested range, not an unbounded maximum. P32 is not P31 long-run acceptance.

## Budgets and acceptance

Pinned RTX 3060 Laptop UUID: `GPU-9eec2f80-4c4e-a5cd-7091-03f0d013cf11`.
Native import requires admitted owned Job and visible UUID. Hard process commit
3 GiB, aggregate Job commit 6 GiB, 48 processes; nested native workers use tighter
existing budgets. Start host commit ≥4 GiB, abort below 1 GiB. Start disk ≥6 GiB,
abort below 2 GiB. GPU start free ≥4096 MiB, running free ≥1536 MiB, stop on global
growth >2048 MiB; monitor every second. Torch allocator cap 15% (~0.9 GiB), ORT
arena 256 MiB. Group timeout 900 s, no-progress timeout 300 s; CPU regressions
1800 s. Global readings include other applications and may conservatively stop
this scope. WDDM per-process VRAM remains unknown, never inferred as zero; neither
allocator caps nor global observations constitute a physical per-process VRAM quota.

PASS requires actual scope assertions, expected device/provenance, complete
hash-bound journal/result/receipt and verified empty Job after cleanup. Normal
feature assertion/API failure is FAIL and independent cases/groups continue.
Missing dependency/kernel trace/sensor is BLOCKED with reason. Resource, CUDA
context corruption, process/source/identity/evidence or cleanup violation stops
the scope. Mock tests are CPU guards and never real GPU evidence.

Worker and post-cleanup validation share per-feature functional PASS gates:
completed samples/events/copies, actual optimizer updates and gradients, quality
evaluation ledgers, or observed runtime/provider traces as appropriate. P16
requires observed completions spanning its 1.2-second sensor window from a
concurrent workload bounded to four seconds; board samples remain non-exclusive
with unknown internal sensor windows. P29 retains both original model weight
hashes before/after capture and binds them to the GPU comparison witnesses.
P32 runs for three seconds with at most 512 KiB extra live allocations, requires
at least three real samples for each allocated/reserved series, and binds each
timeline to its own counter. It performs no OOM, stress, cache flush or peak reset.
P23 explicitly requires `saved_activation_lifetime=False` in its basic capture
and retained evidence; the in-process basic API never enters the B4 adapter.

UUID comparison canonicalizes Torch's bare UUID/UUID object and NVIDIA's GPU-
prefixed display. Preparation hashes the expected nvcuda, torch_cuda and cudart
DLL files without loading them. Runtime evidence must match their actually loaded
paths/hashes, the selected Torch source/version, and interpreter image. Parent
binding accepts the exact direct interpreter or one venv redirector hop, matching
the controller's OS parent and exact Job membership checks. Nested native workers
inherit the outer no-breakaway Job before their own stdlib admission latches;
inner Job limits remain restrictive and cleanup receipts must report no remaining
processes. P30 now retains the actual explicit memory limits in that receipt.

Targeted CPU regression selection is sealed in plan.json from the completed
reviewed inventory, plus the explicitly added offline module and device-option
negative cases. It covers loader/training, quality/P29/precision, affected native
bindings, runtime/streaming and existing B4/basic-memory guards. No full campaign
or package rebuild is implied; the prior package evidence is historical, not
acceptance of the newly changed source.
The runtime-analysis module is selected only by its six reviewed CPU case names;
the two existing isolated-memory CUDA guard tests run with mocked CUDA calls.
No real CUDA test from that module is selected. The exact expanded nodeids and
per-file totals are in `plan.json` (`cpu_inventory`); no live collection occurs
during preparation.

B4 `BLOCKED_B4_UNRESOLVED` remains enforced. P23 saved-activation CUDA, P31 ≥7200 s,
P18 advanced counters, P33 advanced energy/thermal, failfast and allocation stress
are deferred and are never scheduled here. No FULL CUDA ACCEPTANCE promotion.
