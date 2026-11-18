# HYSo 13.5.16 CPU/CUDA performance acceptance

2026-11-16, Europe/Istanbul. **PARTIAL — CPU VERIFIED / CUDA BLOCKED.**
CPU acceptance is PASS within the fixed synthetic CPU fp32 scope. CUDA is
`BLOCKED_SAFETY`, with gate H BLOCKED and gate I NOT_TESTED.

`QA` resolves to `%LOCALAPPDATA%/hyso-stage13-5-16-20261116`. Raw samples,
protocols, temporary files, checkpoints, installations, logs and new SQLite
stores are confined to QA. The six final repository deliverables comprise
the example, its tests and the four requested reports.

## Initial audit and protected state

Repository `C:/Users/hdgn5/OneDrive/Belgeler/GitHub/hyso`, branch
`feature/hyso-models-next`, initial HEAD
`1791814e3190215136c288ea55b1af25fffbd61e`. The initial index was empty.
The original user state contains two tracked modifications and 45 untracked
files. No applicable AGENTS.md was found in the repository or its parents.
The Builder, real-learning, strict-closure and Reporting/ExperimentManager
reports were inspected before measurement. `QA/baseline.json` records all
768 existing repository paths and inherited protection groups.

The closing SHA comparison matches 47/47 unrelated user files, 71/71 native
files, 197/197 data/model/checkpoint files and 6/6 Core/evidence contracts.
All 768 baseline repository paths match. External inherited groups match 6/6
and 11/11, including the manifested Studio SQLite/WAL/SHM/lock files.
Historical QA groups match 7,164/7,164 for 13.5.12, 7,021/7,021 for 13.5.13,
895/895 for 13.5.14, 1,545/1,545 for its closure and 2,494/2,494 for 13.5.15.
Groups overlap. Scope is the manifested paths. User stores have zero write
connections from this task. `QA/preservation.json` retains the path-level result.

## Historical native evidence

The retained final-results index has SHA-256
`86a014ef432ed96a552bfc35a9c65c3bb78714c117e72708ce7b3c8051795bde`.
Its rows record A 35/35 and B 35/35 within their historical declared scopes.
The composite A continuation, Basic-only P23 and provider corrections remain
properties of that retained index. This task performed zero P01–P35 suite
launches and zero long-run executions. Its native aggregate claims retain
historical evidence status. [Historical final acceptance](p01-p35-final-acceptance-20261112.md)

B4 remains OPEN with native exit `3221226505` / `0xC0000409` and an unresolved
root cause. The 2026-10-22 hook-free/reset-enabled diagnostic records one
successful execution with clean owned cleanup. The 2026-11-11 Basic P23 record
has its own parameter/gradient/optimizer/RSS/allocator scope. The inspected
2026-11-11 forensic-readiness snapshot was ineligible. The old 2026-10-22 raw
diagnostic locator currently has no directory, so this stage retains the
versioned report observation for that diagnostic. Its raw result was unavailable
for a new byte-level verification. [Crash audit](../audit/p23-native-crash-root-cause-audit-20261103.md),
[diagnostic report](../audit/m2-p23-b4-r3-wrapper-fix-cuda-diagnostic.md)

## Hardware, software and frozen protocol

Python 3.10.18 from `torch_gpu`, Torch 2.5.1+cu121, CUDA build 12.1,
OS profile `Windows-10-10.0.26300-SP0`. CPU is AMD Ryzen 7 5800H,
8 physical and 16 logical cores. Torch intra-op threads equal 1 and the recorded
interop count equals 8. No affinity, priority, driver or global setting was changed.

The read-only GPU observation records RTX 3060 Laptop GPU, 6,144 MiB total,
5,463 MiB free, driver 591.74 and driver-reported CUDA support 13.1. These
version strings are retained metadata. Live CUDA context compatibility remains
unverified for this stage.

The protocol has SHA-256 `22751e3e1e4b843832ab9822df201b97f50e46ce78f65e905d3c46090915b5dc`.
It fixes seed 13516, CHW 1×8×8, ten classes, batches 1/8/32, CPU fp32,
5 warmups then 3×40 measured inference forwards per cell, and three continuous
40-batch throughput windows per cell. Two fresh child interpreters execute
the same configurations. Each child has a 180-second external deadline and
a 60-second measurement budget after imports. Actual measurement scopes finish
in 19.596819 and 20.966723 seconds.
Parent receipts including startup are 28.338476 and 30.123987 seconds.

Construction has five samples. A fresh model performs one initial forward for
each batch geometry inside an already imported interpreter. A dedicated train
clone uses batch 8, two warmups and twenty measured cross-entropy forward/backward
and SGD-step samples. SGD uses lr .001. Checkpoint save and strict restore each
have five samples. Evaluation uses 32 deterministic synthetic samples in four
batches, repeated three times. Expert report creation plus JSON/Markdown/HTML
export has three samples. Comparison has five samples per batch.

`perf_counter_ns` supplies host monotonic timing. CPU completion is synchronous.
Warmup and measured samples have separate loops. The runtime inference API also
performs one untimed validation forward. Throughput includes completion checks,
loop bookkeeping and terminal completion. The independent benchmark matrix
retains both derived serial service rate and measured completion-window rate.
The [benchmark matrix](stage13-5-16-benchmark-matrix.md) gives every sample count
and distribution, while the [resource inventory](stage13-5-16-resource-measurement-inventory.md)
records actual API and trust boundaries.

## Synthetic models and CPU observations

Sequential uses two 3×3 convolutions with channels 1→4→8, ReLU, flatten and
a ten-class head. Graph uses a 1→4 stem, a 4→4 3×3 branch and a 4→4 pointwise
branch, ReLU, concatenation, flatten and a ten-class head through GraphTensorOutput.
Both parameter sets are fully trainable. Their parameter counts are 5,466 and
5,338, with logical parameter memory 21,864 and 21,352 bytes respectively.
Configurations, architecture hashes, input hashes, output shapes and state
hashes are saved per run. The model-state SHA values below use the QA state_dict
name/shape/dtype/byte fingerprint. Native BenchmarkResult retains its own
weights fingerprint in every raw result.

| Model | Architecture SHA-256 | QA state SHA-256 |
| --- | --- | --- |
| sequential | 3dff80bb0aadbb8975e6a107f0fb975072fe76becd0b45cdbc5566d70f9c43bc | e599268c93e8c0c27120e1a9a7bb8d1801e13c3e0e0051b2812927c99cbf8036 |
| graph | 3fa4fc1c55cdd7e5cd5c1ef2da15efcb8d3d90acc0395ba3b23f433541590b35 | 14c8dea3b60f898da9698ed582141cd1c7b5b3e18c8f27e053000133651cc4d7 |
 The two children reproduce each model's state and
input identity exactly. Digits datasets and retained test partitions received
zero new inference or training calls. [Executable example](../../examples/performance/cpu_resource_acceptance.py)

| Run | Model | Batch | n | Min ms | p50 ms | p90 ms | p95 ms | Mean ms | SD ms | Serial service images/s | Completion images/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | sequential | 1 | 120 | 0.1436 | 0.152 | 0.17488 | 0.187475 | 0.157117 | 0.013398 | 6364.697146 | 5728.82602 |
| 1 | sequential | 8 | 120 | 0.2889 | 0.3041 | 0.42178 | 0.496945 | 0.332112 | 0.063515 | 24088.223117 | 23349.596976 |
| 1 | sequential | 32 | 120 | 0.3251 | 0.35665 | 0.43609 | 0.492485 | 0.373724 | 0.060071 | 85624.647412 | 83308.384208 |
| 1 | graph | 1 | 120 | 0.312 | 0.3403 | 0.42045 | 0.521235 | 0.365613 | 0.066051 | 2735.136244 | 2572.358295 |
| 1 | graph | 8 | 120 | 0.4697 | 0.5178 | 0.58059 | 0.63171 | 0.535465 | 0.065946 | 14940.285546 | 14799.809452 |
| 1 | graph | 32 | 120 | 0.5883 | 0.66995 | 0.76001 | 0.82368 | 0.684873 | 0.057966 | 46723.968422 | 43974.870194 |
| 2 | sequential | 1 | 120 | 0.1509 | 0.1557 | 0.21643 | 0.27148 | 0.172331 | 0.041159 | 5802.79211 | 5763.494983 |
| 2 | sequential | 8 | 120 | 0.2856 | 0.29735 | 0.39306 | 0.443865 | 0.322248 | 0.054946 | 24825.638678 | 23600.03835 |
| 2 | sequential | 32 | 120 | 0.3242 | 0.3589 | 0.40945 | 0.506865 | 0.37205 | 0.056809 | 86009.9449 | 81705.604494 |
| 2 | graph | 1 | 120 | 0.3096 | 0.34395 | 0.40497 | 0.46108 | 0.359319 | 0.060167 | 2783.041075 | 2643.561935 |
| 2 | graph | 8 | 120 | 0.4892 | 0.50925 | 0.57254 | 0.601405 | 0.522118 | 0.035366 | 15322.197075 | 14644.392952 |
| 2 | graph | 32 | 120 | 0.6217 | 0.6408 | 0.72438 | 0.745115 | 0.656423 | 0.039444 | 48749.028828 | 41216.841373 |

There are 1,440 measured inference samples across twelve cells. Quantiles use
linear interpolation at `(n−1)q`, and standard deviation uses ddof 1. These
120-sample p90/p95 values are descriptive. The existing default TailPolicy
requires 1,000 samples and ten tail observations, so the separate native tail
analysis correctly returns insufficient_samples. Effective independent sample
size and statistical superiority remain unestablished.

The measured RSS baselines are the start of each separate 100-forward resource
window after prior workflow operations. The sampler interval is 5 ms with actual
timestamps and a 1,000-sample cap. The sampled maximum can miss transients.
Windows lifetime HWM includes startup and earlier allocations. Model-only peak
RSS, process-entry RSS, CPU utilization and fixture FLOPs/MACs remain NOT_MEASURED.
Checkpoint file sizes remain independent serialized-byte observations.

| Run | Model | RSS n | Baseline RSS byte | Final RSS byte | Delta byte | Sampled max RSS byte | Lifetime HWM byte | Parameter byte | Checkpoint byte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | sequential | 4 | 608448512 | 608464896 | 16384 | 608464896 | 666046464 | 21864 | 24776 |
| 1 | graph | 7 | 610738176 | 610811904 | 73728 | 610811904 | 677081088 | 21352 | 26752 |
| 2 | sequential | 5 | 609673216 | 609734656 | 61440 | 609734656 | 666714112 | 21864 | 24776 |
| 2 | graph | 9 | 611479552 | 611565568 | 86016 | 611565568 | 677912576 | 21352 | 26752 |

Forward/backward, SGD step, construction, checkpoint IO, evaluation and report
overhead have distinct raw samples. Their complete distributions are in the
matrix. Reporting p50 is 3500.2627 ms and
3693.9038 ms for the existing 16-record expert report
creation plus three exports. Report cost is measured as its declared composite
operation. Export-format-specific timing remains NOT_MEASURED.

## Measurement correctness and comparisons

`QA/correctness-proof.json` records 212 assertions. They independently recompute
means, medians, quantiles, sample SD and throughput from saved samples. They check
counts, modes, inference-mode metadata, output shapes, native parameter counts,
stable inputs/states, exact checkpoint output equality, completion ledgers,
resource scopes, report raw-source equality and store verification results.
The controlled-clock regression proves preflight/warmup ordering and timer scope.
The [25 new tests](../../tests/test_cpu_resource_acceptance.py) cover concrete
invalid-data and fixture cases.

Six sequential/Graph comparisons have compatible recorded CPU timing conditions.
The architectures and operator paths differ. Per-cell differences in the matrix
remain descriptive observations. A winner, optimization improvement, general
device ranking and HYSo-versus-PyTorch advantage remain unestablished.
Same-model state equality is an explicit EvidenceScope constraint. The existing
general architecture comparator allows different model identities.

Two setup attempts are preserved with exit 1 and excluded from acceptance.
`cpu-1` failed while assigning the external protocol hash to the native
protocol scope. `cpu-final-1` failed because evaluation requires a standard
DataLoader. The corrections retained the sealed protocol bytes, sample counts,
geometry and optimizer settings. Old scripts and locks remain in QA. The final
repository example removes one trailing blank line caught by the staged
whitespace check. Its AST matches the exact executed QA script, whose original
bytes remain archived in QA/example-source-proof.json.

## CUDA admission

The single read-only observation at 2026-11-16 17:13:56 Europe/Istanbul records
an unidentified compute PID 2000 with Insufficient Permissions and N/A process
memory. That PID is absent from the captured C/G table. Twelve visible C+G
processes also have N/A per-process memory, and current workload independence
has no identity-bound review. GPU utilization observations are 30% and 36%.
Free VRAM alone supplies insufficient admission evidence. The current inventory
gap and open native risk produce `CUDA = BLOCKED_SAFETY`.

CUDA child launches equal zero. Forward/backward, allocator counters, peak-reset
windows, device restore and CPU/GPU numerical parity are NOT_TESTED. The task
performed zero GPU resets, stress tests, installation changes or native retries.
`QA/cuda-observation.json`, the raw CLI captures and `QA/cuda-decision.json`
retain the actual decision inputs.

## Reporting, experiments and negative outcomes

Existing EvidenceRecord/EvidenceIndex adapters carry genuine BenchmarkResult,
ThroughputResult, MemoryResult and FootprintResult payloads. Each child exports
one 16-record analysis and derives TR/EN JSON, Markdown and HTML through existing
language/level parameters from that same measurement. Presentation authority is
false. Custom forward/backward, optimizer, IO and workflow samples remain QA
JSON supplements. Their native typed execution authority remains unclaimed.

Each child creates a fresh QA-only schema1 store, closes it, explicitly migrates
it and reopens schema2. Two benchmark runs complete per child. Model identity,
full settings, device, precision, warmups, repetitions, raw sample references
and limitations are retained. There are 32 registered artifacts per child,
verified before and after reopening, yielding 128 verified hash observations.
Latency and derived service-rate scalars have explicit units and sources.
Continuous throughput and resources retain their typed artifact payloads.

The negative matrix covers twenty requested categories. Source tests and
controller probes reject malformed/empty/partial measurements, unavailable CUDA,
wrong scopes and artifact corruption. CPU-only controller probes observe timeout,
exit 7 and zero exit without a completed summary. All three are rejected.
Wrong-unit authenticity beyond the schema's fixed millisecond convention and
coherently relabeled timing samples remains unsupported. Native crash injection
and live CUDA cleanup remain NOT_TESTED.

## Regression and distribution verification

358 source tests PASS, comprising 25 new and 333 selected existing cases.
There are zero failures, errors or skips and two existing autocast deprecation
warnings. The selection covers state safety, controlled-clock correctness,
result schemas, comparisons, inference, memory and CLI help. The previous
full test pools and locked Digits learning set were left out of this selection.

Runtime source bytes are unchanged. Reused 13.5.15 wheel SHA-256 is
`2a6ae9b6757b9f9e4c8437967039f0cd2655f9702e5bf733f2601c6a32b5e572`
and sdist SHA-256 is
`dd169bcdec2fbea7cc86df2637800c34453fa1cb777284d7c877c8d1428c7c3e`.
Both archives match exactly 164 current runtime source modules. Each is freshly
installed under this QA root using offline, no-dependency, no-build-isolation,
no-compile installation. All installed runtime bytes match source, and every
loaded HYSo module resolves inside the selected target. Existing user-site
pygments supplies the already installed test dependency.

Each installation passes all 25 new tests and a separate two-model benchmark,
strict-checkpoint, TR/EN export and artifact verification smoke. Artifact smokes
also observe the expected mismatch and missing outcomes. Root, analysis,
reports, experiments and compare help entry points exit zero on each target.
Installed repetitions are recorded separately from the 358 source cases.
`QA/package-proof.json`, import proofs, XML and logs retain the checks.

## Acceptance gates and Git closure

| Gate | Decision | Verified scope |
| --- | --- | --- |
| A | PASS | Actual performance, analysis, Advisor, reporting, store and CLI entry points inspected |
| B | PASS | Protocol sealed before timing and unchanged through both complete runs |
| C | PASS | 12 CPU inference cells with 120 samples each and separate completion windows |
| D | PASS | Dedicated clone forward/backward and SGD step windows |
| E | PASS | Native Windows RSS/HWM, logical parameter bytes and checkpoint size within declared scopes |
| F | PASS | 212 independent audit assertions and controlled-clock regression |
| G | PASS | CPU shared conditions checked and different-state regression requires an explicit identity constraint |
| H | BLOCKED | Unresolved current GPU process identity and workload independence |
| I | NOT_TESTED | Zero admitted CUDA child executions and zero CUDA measurements |
| J | PASS | Malformed samples and partial payloads rejected, timeout/exit/missing-summary controller probes rejected |
| K | PASS | Existing typed analysis, TR/EN presentation, four completed QA runs and 128 hash observations |
| L | PASS | Historical P01–P35, Basic P23 and unresolved B4 scopes retained separately |
| M | PASS | 358 source cases, 25 per isolated archive installation and two fixture/API smoke sets |
| N | PASS | All manifested repository, native, Core, user data and historical QA hashes match |
| O | PASS | Exact-path local commits, staged checks, original user status and empty closing index |

The implementation/test commit is `56c1c3710843c7c94b500d048eeac797dffa6867`, parent
`1791814e3190215136c288ea55b1af25fffbd61e`. Its paths are the new CPU example and 25-case tests.
The documentation commit contains exactly the four requested reports. Its own
full SHA, parent and final HEAD are retained in QA/commits.json and the final
handoff. Exact staged paths, cached diffs, whitespace checks, SHA protection and
test/package proofs are checked before commits. Closing records include ancestry,
fsck, an empty index and the original 47-file unrelated user state. Push count is zero.

Open items comprise CUDA admission/execution, unresolved B4 root cause,
unavailable old raw diagnostic locator, short-sample tail confidence,
model-exclusive memory and unmeasured utilization/FLOPs. The measured report
overhead and integration usability findings are recorded in the
[backlog](stage13-5-16-performance-usability-backlog.md). The final decision is
**PARTIAL — CPU VERIFIED / CUDA BLOCKED**. Stage 13.5.17 has zero execution.
