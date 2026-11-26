# Final user acceptance — frozen product

This is a separate campaign, not a rerun of the 658 CPU / 36 CUDA acceptance.
The adapter reuses the serial CUDA controller, admission, UUID/runtime/DLL pins,
Windows Jobs, no-breakaway, memory limits, journal sealing and verified cleanup.
All previous CUDA files (including failed attempts) and original campaign result
files are hashed read-only in the new plan. Product defect fixes require a
scoped source-impact review; they never rewrite original campaign evidence.

```powershell
.\run_final_acceptance.ps1 -UserAcceptance -PrepareOnly
.\run_final_acceptance.ps1 -UserAcceptance
.\run_final_acceptance.ps1 -UserAcceptance -Status
.\run_final_acceptance.ps1 -UserAcceptance -Resume
.\run_final_acceptance.ps1 -UserAcceptance -Resume -Revalidate U05,U13
```

Revalidate accepts exact failed/BLOCKED CUDA scenario IDs or exact CPU node IDs,
for example `tests/test_user_acceptance_scenarios.py::test_U05`. Valid PASS cannot
be overwritten. Normal resume never retries terminal failures. Changed preparation
archives its previous plan/checkpoint and marks evidence invalid; it never grants
implicit execution. No GPU automatic retries. Clean CPU assertions do not gate
independent CUDA journeys; integrity, ownership, monitoring and cleanup remain gates.

| Group | CPU scenario count | CUDA scenario count |
|---|---:|---:|
| External / four torchvision CNNs / ModelBuilder / GraphBuilder (U01–U04) | 4 | 4 |
| Training / persistence and resume (U05–U06) | 2 | 2 |
| Evaluation and two-model comparison (U07–U08) | 2 | 2 |
| Advisor / localized reports / public CLI (U09–U11) | 3 | 1 |
| Read-only integrity (U12) | 1 | 1 |
| ONNX export and actual provider node execution (U13) | 1 | 1 |
| Invalid user input (U14) | 1 | 0 |
| Total (plus one independent CUDA admission) | 14 | 11 |

CPU journeys run as one selected pytest group with the established admitted CPU
runner; CUDA journeys run in six serial groups following admission. Each journey
has a single terminal result, with subcases retained inside it: four architectures,
three builder convolution paths, single/two-input branch/residual graphs, three
training model families with FP32 and (CUDA only) AMP. Training uses eight fixed
local labeled examples, two batches and one epoch; resume compares two epochs
against one epoch plus restored state (SGD momentum + StepLR). Benchmarks use
one warmup and five timed forwards, throughput three batches, provider capture
one warmup and three measured runs. These are functional checks, not statistically
certified tail latency or model superiority claims.

Models are untrained; torchvision uses weights=None and batch 1 / 32×32 input.
Meta-tensor storage estimates block large architectures before real allocation
when their conservative working estimate exceeds existing limits. In particular
VGG11 may be BLOCKED on the admitted GPU. Limits never increase. Missing optional
dependencies are not installed. Actual unsupported operations remain explicit
FAIL/BLOCKED; no scenario has been executed during preparation.

U13 uses public torch.onnx.export (there is no invented HYSo model exporter) and
HYSo's public profile_execution_providers. CUDA node coverage, no CPU placement
fallback, exact pre-admitted ORT DLL identities and owned nested cleanup are
mandatory. CPU provider execution is a separate CPU journey. CPU/GPU comparisons
are separate same-device paired cohorts, never cross-device superiority claims.
Multi-input graph training is CPU-only in the product; the training journey uses
the supported single-input GraphTensorOutput on both devices.

Results live under outputs/final-user-acceptance-20261126. user-results.json is
an index of every attempt's user rows; checkpoint.json owns current statuses.
Each row includes scenario, public API/CLI, model/device, expected/actual behavior,
PASS/FAIL/BLOCKED, raw evidence path, untriaged product failure and support limit.
Raw data is persisted before assertions. CPU artifact hashes and controller seals
are checked on resume. CUDA runtime and feature artifacts use the existing sealed
journal and parent-side functional verification.

Product performance scope stays frozen. full_cuda_acceptance=false. Deferred:
P18 advanced hardware counters/Roofline, P23 CUDA saved activation/B4, new P31
7200-second CUDA validation, and P33 advanced energy/thermal. Native failfast,
intentional overflow/OOM, allocation stress and long-duration runs are excluded.

The U02 CUDA budget separates host guard/byte-witness copies from the single
resident GPU model. A meta-only forward at batch 1 / 32x32 sums every module
output conservatively; model + input + this activation sum must fit the existing
70% allocator allowance. The other 30% remains workspace headroom. The 3 GiB
host estimate and 15% Torch allocator cap remain unchanged. A budget refusal is
a resource limitation of this capture, not an unsupported architecture.

U05 Graph training permits native FP32/autocast intermediate dtype promotion
only while the matching CPU/CUDA autocast policy is active. Graph input dtype,
node shape/device, and out-of-policy dtype rejection remain strict. Native AMP
skip evidence is retained; every episode still needs a real applied update.

U07 now requests both producer evidence and receiver structure on CUDA. The
existing schema 2 stream hashes describe CPU batches before their CUDA transfer;
schema 3 structure records the actual isolated CUDA forward receiver before and
after inference. Execution source device is never relabeled CPU. Default schema
1 behavior and CPU capture remain compatible. Structure requires producer
evidence and nonempty evaluation. Named two-input Graph evaluation and the
separate Graph receiver-runtime receipt remain CPU-only. Physical GPU identity
comes from the campaign's admitted runtime, not from imported evidence alone.

Benchmark finite validation now scans at most 65,536 logical elements per view.
Non-contiguous views are narrowed without flattening or making a full CUDA
contiguous copy. Host snapshots still retain the complete state required for
read-only verification/rollback; their device-to-host transfers, descriptor
hashing, bit comparisons and output finite checks use bounded chunks. Canonical
weight hashes and bitwise NaN/signed-zero gradient preservation stay unchanged.
U02's GPU admission estimate includes an additional 8 MiB validation scratch
allowance, without increasing the allocator, Job or process limits.

After the sealed U02 OOM, explicit revalidation still requires U02 in the retry
set and verified cleanup. The reviewed recovery runs U07, U05, then U02 in
separate owned workers. Each gets fresh pre-import Job admission, physical GPU
and binary checks, a completed CUDA probe and fresh resource admission. Cached
campaign ADMISSION is not substituted for these per-worker checks. The safety
hold is not bypassed: any new OOM/ownership/cleanup failure stops immediately;
ordinary budget refusal remains BLOCKED and never counts as a successful model.

U07 precision quality permits a native FP32 final output under active FP16/BF16
autocast, for example when a low-precision head result is added to a FP32 buffer.
The result is not cast down to create evidence. Each batch still needs supported
Conv2d/Linear observations with matching active autocast, FP32 weights and actual
target-low outputs on the selected device. Output dtype alone cannot establish
AMP execution. Mixed-output proof v3 records actual final dtypes independently
from the autocast target. Uniform-output v2 and legacy readers remain compatible;
CPU FP32/FP64 profiles remain strict. These module observations do not certify
every functional operator, output dataflow dependency or hardware instruction.
