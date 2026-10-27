# P34 Multi-Device Comparison acceptance — 2026-10-27

**P34 = IMPLEMENTED within the recorded Windows native CPU + CUDA scope.**
This is one actual host CPU and one actual physical GPU. No two-GPU, edge,
remote-device, universal backend, quality or energy acceptance is claimed.

## Requirement, implementation gaps and evidence gaps

The original `a0672d5` backlog requires real per-device runs with provenance,
matching protocols, visible precision/backend conditions and no invented missing
device result. The original architecture audit, P34 row and M2-P-B16 gate,
explicitly accept a CPU + CUDA pair; two logical indices of one GPU do not qualify.

The starting `632a297` code already supplied serial owned Windows workers,
model/input/factory hashes, host completion latency, instrumented throughput,
cleanup receipts, immutable JSON/CSV and null quality/energy. Existing native
tests were opt-in. A second successful real device capture was an acceptance
evidence gap, not grounds to replace the whole implementation. Existing
BenchmarkRunner CPU host timings and CUDA event timings have different scopes;
P34 therefore captures both devices using its shared host completion protocol.

Actual code gaps found and corrected:

- A hidden fixed 0.1 CUDA allocator fraction became `cuda_memory_fraction=None`
  by default; an explicit fraction is optional and recorded. P34 Job private
  commit/process-count quotas are also explicit optional protocol fields. Job
  ownership, deadlines, output bounds and cleanup remain mandatory.
- Requested UUID/environment selection alone was not physical execution proof.
  The worker now independently observes `torch.cuda.get_device_properties(0).uuid`,
  verifies one visible device and rejects mismatches. Full canonical UUIDs and
  duplicate normalization prevent alias/index counting. `cuda:0`/`cuda:1` are
  not accepted as distinct physical targets.
- Precision evidence now records actual input/output residency and output dtype,
  runtime, worker PID, thread counts, inference mode, autocast, TF32, matmul,
  deterministic and cuDNN settings. FP32 uses highest matmul precision, disabled
  TF32/autocast, deterministic algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- A device drain precedes every forward outside the latency interval; terminal
  device synchronization lies inside it. Resident H2D initialization is excluded,
  including when warmup is zero. Each finite output is validated and released.
- Separate native allocator allocated/reserved peaks and logical registered
  parameter/buffer bytes accompany process RSS snapshots. Per-axis eligibility
  prevents memory/energy/quality superiority claims. Input and model hashes are
  checked again after inference.

## Actual native comparison

Private raw capture: `C:/Users/hdgn5/hyso-p34-evidence-20261027/native/`.
The reproducible entry point is `tests/p34_runtime_acceptance.py`; no model,
dataset or weight download is used.

| Property | CPU | GPU |
| --- | --- | --- |
| Hardware | AMD Ryzen 7 5800H with Radeon Graphics, 8 cores / 16 logical processors | NVIDIA GeForce RTX 3060 Laptop GPU, compute capability 8.6 |
| Execution target | host CPU, one intra/inter-op thread | physical UUID `GPU-9eec2f80-4c4e-a5cd-7091-03f0d013cf11`, worker logical index 0 |
| Runtime | Python 3.10.18, PyTorch 2.5.1+cu121 | same; CUDA build 12.1, cuDNN 90100, driver 591.74 |
| VRAM | unavailable | runtime total 6,441,926,656 bytes; nvidia-smi reports 6144 MiB |
| Mean batch latency | 2.444385 ms | 0.564840 ms |
| Median batch latency | 2.281700 ms | 0.514300 ms |
| Batch latency min–max | 1.938200–4.246700 ms | 0.466100–1.048600 ms |
| Conditional sample standard error | 0.145520 ms | 0.032039 ms |
| Instrumented completed images/s | 2278.845195 | 3907.432914 |
| Process RSS before / after | 598,659,072 / 598,667,264 bytes | 866,103,296 / 866,115,584 bytes |
| CUDA allocated / reserved window peak | null / null | 38,165,504 / 58,720,256 bytes |
| Logical model bytes | 21,672 | 21,672 |
| Quality / energy | null / null | null / null |

Both workers run the same seeded CNN: Conv2d(3→16, 3×3), ReLU,
Conv2d(16→32, 3×3, stride 2), ReLU, adaptive average pooling, flatten and
Linear(32→10). Input is the same seeded normal tensor, NCHW **[8,3,64,64]**;
output is [8,10]. Parameters, inputs and outputs use **FP32**, eager eval and
inference mode. Seed 34, three warmup forwards and twenty measured forwards
per device give 160 completed measured images on each endpoint.

- Model/source-state SHA256 on both:
  `d0c539afa22a19fe3ba98f5fecfef3337aac9a91d728e60985b9e8323911597f`.
- Input SHA256 on both:
  `54c6c7f0c009910d367de05be1f6c4ac5b7fc1b7b713d7e76dce24afd1687a3b`.
- Factory module SHA256 on both:
  `e9009e38668cd2897b1bd4c7e925a49842f300193f876dac8605777fad440637`.

Latency is **per batch**, measured with perf_counter_ns through terminal device
completion. It includes host launch overhead. Throughput is completed images
divided by the instrumented observation wall window, including validation,
pre-drains and heartbeat writes; it is not batch/mean-forward latency or an
uninstrumented saturation rate. Serial captures are descriptive observations;
environment, clocks and thermal variation do not establish confidence bounds
or a universal speedup. No automatic winner/ranking is returned.

Latency and instrumented throughput are eligible for this pair. Logical model
bytes are eligible and equal. RSS describes whole worker processes and includes
different CPU/CUDA runtime overhead; allocator peaks describe only the GPU
worker's native allocator and include resident baselines/cache. Neither produces
model-memory or physical-VRAM superiority. Allocator peaks cover the measurement
window and validation, not an isolated model-only allocation. Quality has no
labeled evaluation protocol; energy has no integrated power measurement. Both
stay null and ineligible.

The primary comparison uses **no CUDA allocator quota**. Explicit test-only
Windows limits are 4 GiB process private commit, 6 GiB Job private commit and
four concurrent processes; deadlines are 60 s startup, 20 s heartbeat and 45 s
worker runtime. Commit limits are not RSS limits. Both terminal reports completed,
both exit codes were zero, all owned Job processes were cleaned up and their PIDs
were independently absent. Exports reopen identically from JSON and CSV.

## Verification and compatibility

- 77 P34 contract/CPU/export/saved-native tests PASS: matching and incompatible
  runtime/settings/model/input/source/shape contracts, missing device, missing
  and incompatible memory domains, null quality/energy, failed/missing/timeout
  worker, binding/identity corruption, asynchronous completion ordering and
  quota defaults. Modified-copy protocol fixtures are explicitly simulated;
  they are not a second native GPU run.
- 304 existing benchmark CPU/comparison/export/CUDA-protocol and provider tests
  PASS. CUDA protocol doubles are not counted as native inference proof.
- One separate short native opt-in quota/OOM/recovery test PASS: a 0.02 child
  allocator fraction rejects a controlled allocation before consuming it; a
  fresh default worker then performs actual GPU inference successfully.
- **382 distinct final tests PASS.** Repeated/intermediate runs are not added.
  The primary two-device acceptance script additionally passed its native,
  terminal, PID, Job and export assertions. GPU work is limited to that one
  comparison plus the explicit OOM/recovery test; no FP64/long-run GPU sweep.
- Final aggregate verification checks 27 owned-worker receipts from P34 and
  related regressions: every Job has zero active processes, expected terminal
  or timeout/failure evidence is retained, and no incomplete temporary receipt
  remains. Process creation times distinguish an original worker from a later
  process reusing its PID; no original owned worker survives.
- Wheel and sdist build successfully without installing/upgrading dependencies.
  Their production sources match this worktree, and private evidence, tests and
  model artifacts are absent. A separate `--target` installation reopens native
  and legacy exports under `python -I` and runs one short installed CPU worker
  with successful terminal/PID/Job cleanup. The package probe uses the Windows
  API, without a psutil dependency. Existing README/license-classifier packaging
  warnings are nonfatal and outside P34.

RuntimeResult schema_version=1 and CSV envelope columns remain unchanged.
New captures declare `comparison_version=2`; legacy captures without that field
retain exactly their original analysis and source hashes. A real pre-change CPU
JSON/CSV capture was reopened unchanged. Committed legacy schema fixtures remove
local paths and recompute bindings; they are labeled modified fixtures, not
native evidence. Missing/unknown new comparison versions are not silently
promoted. Public API changes are additive optional DeviceProtocol fields.

Production changes are `hyso/performance/devices.py` and the P34-only routing in
`hyso/performance/stability.py`. Tests add the CNN factory, bounded native entry
point, contract regressions and legacy JSON/CSV fixtures; existing deployment
tests label their legacy simulated identity fixture and make the OOM quota
explicit. This report is the only documentation change.

The isolated worktree is `C:/Users/hdgn5/hyso-p34-completion`, detached from
`632a297`; P35's uncommitted acceptance work stays in its own worktree. Hash/status
checks preserve the original checkout and P12/P29/P35 changes. All **92 P31 native
private evidence files** and the other 32 P31 private files remain byte-identical.
Base package inventory remains identical. Host driver/CUDA/PyTorch installations
and system settings were not changed; explicit worker-only runtime flags are
recorded above. No commit or push was made.

Remaining limits: Windows owned-worker adapter, trusted CPU factories, floating
NCHW input and tensor output, FP32/FP64 eager execution; no remote/edge/other
backend execution, no two-physical-GPU acceptance, no model-only physical memory,
quality, power integral or statistical superiority guarantee. Unsupported targets
remain failed/unavailable rather than estimated. P34 ends here. **P31 and the
7200-second run were not started.**
