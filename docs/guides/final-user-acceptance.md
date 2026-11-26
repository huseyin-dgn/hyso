# Final user acceptance — frozen product

This is a separate campaign, not a rerun of the 658 CPU / 36 CUDA acceptance.
The adapter reuses the serial CUDA controller, admission, UUID/runtime/DLL pins,
Windows Jobs, no-breakaway, memory limits, journal sealing and verified cleanup.
All previous CUDA files (including failed attempts) and original campaign result
files are hashed read-only in the new plan. No product files are changed.

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
