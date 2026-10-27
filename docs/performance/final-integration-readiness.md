# HYSo final integration and P31 readiness — 2026-10-27

**Decision: NOT READY for the final long run. Integration is preserved; final
source-pinned manifest/CLI preflight and aggregate PID audit are NOT COMPLETED.**
The user requested safe closure because of usage limits. No new development or
comprehensive test is started after that instruction.
No 7200-second P31 run is authorized or started by this review. P31 remains
PARTIAL. B4 remains OPEN; inference acceptance does not resolve its native crash.

## Source identity and protected inputs

Integration worktree: `C:/Users/hdgn5/hyso-final-integration`, detached at
`1ab52822c39c073eebca1d8c1d773e56c47cb728`. No new commit, staging or push.
This base includes the older P34 base and the accepted prior P13/P14/P15/P18/
P21/P23 work. The uncommitted overlay is part of the source identity, not silently
described as that commit's clean tree.

| Input development area | Base / status at inventory | Modified / new files |
| --- | --- | --- |
| `hyso-p12-scope-completion` | detached `1ab5282`, dirty | 4 / 2 |
| `hyso-p29-memory-completion` | detached `1ab5282`, dirty | 2 / 3 |
| `hyso-p35-runtime-acceptance` | detached `1ab5282`, dirty | 2 / 5 |
| `hyso-p34-completion` | detached `632a297b1b6620454ba193af2d61becffe80a88c`, dirty | 3 / 6 |
| `hyso-p31-launcher-finalization-review` | not a Git repository; five standalone files, launcher declares the `1ab5282` source pin | Git modified/new status is inapplicable; all five files fingerprinted |

Each tracked input delta was checked/applied file by file to the newer base;
only absent new files were copied. `devices.py`, `stability.py` and the deployment
test file had no intervening committed delta between P34's base and `1ab5282`.
Both P34 and P35 hunks in `test_deployment_analysis.py` were retained. No older
whole-file replacement overwrote newer source.

The private evidence root is `C:/Users/hdgn5/hyso-integration-evidence-20261027`.
`integration-inputs.json` and `protected-before.json` record the exact inputs.
`integration-final-inventory.json`, `integration-final.patch` and `source-overlay/`
provide the final file inventory, hashes, tracked delta and all changed/new files.
The source-pinned P31 manifest additionally fingerprints every tracked/untracked
source file and exact Git status; ignored runtime outputs remain outside source.

Protection verification preserves all **1259 pre-existing private files**, including
all **92 historical P31 native evidence files**, all five original Git repository
statuses/changed-file hashes and all five standalone launcher files. The original
checkout remains at `778d26f` with its user's roadmap and untracked files intact.
Existing PyTorch/CUDA/driver installations and base package inventory are unchanged.

## Integrated changes and version contracts

| Feature | Integrated scope | Version retained | Status |
| --- | --- | --- | --- |
| P12 | operation model v3 and profiling derivation; ordinary FLOPs, comparisons, roots and named transcendental operations remain separate | schema 1; operation_model_version 1/2/3 dispatch | IMPLEMENTED in its accepted reference scope |
| P29 | quality/latency/selected-memory evidence and conditional three-axis Pareto | schema 1; comparison_version 2; absent version retains legacy two-axis derivation | IMPLEMENTED in accepted CPU FP32/FP64 scope |
| P35 | narrow optimized-transfer graph analysis with original raw graph/profile binding | schema 1; graph_analysis_version 2; old interpretation preserved without it | IMPLEMENTED in accepted Windows CPU/CUDA provider scope |
| P34 | physical identity verification, matched FP32 protocol, optional quotas and independent memory domains | schema 1; comparison_version 2; legacy JSON/CSV unchanged | IMPLEMENTED in accepted CPU + one GPU scope |
| P31 | original fixed inference workload/protocol; revised interruption/export finalization and outer parent-exit observation | runtime schema 1 unchanged; external preparation manifest explicitly version 2 | PARTIAL; long acceptance still missing |

Production integration touches `_operation_models.py`, `profiling.py`, `quality.py`,
the new `_quality_memory.py`, `providers.py`, `devices.py` and P34-only routing in
`stability.py`. P12/P29/P35/P34 tests, factories, legacy fixtures, acceptance
documents and P29's complete 80-module packaging inventory are included.

Integration gates exposed three existing compatibility/reliability defects:

- Torch 2.2 registers module state in exact `OrderedDict` containers. CPU reference
  collection missed it while accepting plain dicts. `_cpu_layer_memory.py` and
  `_layer_allocator.py` now accept that standard container without admitting
  arbitrary mapping subclasses. The real foreign-NumPy 256-byte reference test
  and its 17-case native CPU group pass on minimum Torch.
- Dynamic invocation parents need not be direct registration parents: `ModuleList`
  containers can be bypassed when calling their registered descendants. The
  `_profile_evidence.py` validator now validates registered descendant paths;
  no uncalled container invocation/timing is invented. Sibling/path-boundary
  rejection and actual installed ModularCNN capture verify the correction.
- An atomic `result.tmp` rename could occur between exists/stat in the isolated
  training controller. `_training_process.py` tolerates only FileNotFoundError
  during that size observation; final bounded result validation remains strict.
  Deterministic race injection retains the real typed worker failure and zero
  active Job processes. No cleanup/performance algorithm or saved-activation
  permission was broadened.

Test/support corrections retain exact legacy example blocks and add a separate
current P12 v3 example; the roofline unknown-op test uses genuinely unmodeled
`sin` rather than now-modeled Norm/SiLU. Controlled OOM fixtures use the exception
namespace available on both Torch versions. Stdlib Job tests use the actual base
interpreter instead of Windows venv redirector PIDs. Aggregate installed-smoke
timeout is 240 seconds, and the deliberately hanging P17 fixture allows 30 seconds
for runtime preparation before timeout; product/long-run deadlines are unchanged.
Known private task-adapter execution remains gated to exact Torch 2.5.1 source;
minimum tests explicitly skip that unsupported execution and test rejection.

Public APIs are preserved, with only the previously accepted additive options.
Old P12 v1/v2 arithmetic is not relabeled as v3; old P29/P34/P35 artifacts retain
their own derivation/version rules. JSON/CSV envelopes and legacy readers remain
compatible. P31 CSV stays a flattened export, not a RuntimeResult.from_csv format.

## Regression and artifact evidence

- Development Python 3.10.18 / Torch 2.5.1+cu121: **3934 PASS, 5 explicit opt-in
  skips, 0 unresolved failures; all 3939 distinct final test identities covered**.
- Minimum Python 3.10.18 / Torch 2.2.2+cpu / torchvision 0.17.2+cpu: **3907 PASS,
  32 explicit skips, 0 unresolved failures; all 3939 final identities covered**.
  Runtime JSON files assert actual scientific-package origins; CPU
  execution is not substituted with development Torch. Build/pip tooling alone
  uses the existing base interpreter; installed scientific smoke runs under minimum.
- Counts are deduplicated by final test identity across recorded segments;
  successful retries/overlapping suites are not added as extra tests. Intermediate
  failures/logs remain private. An editing indentation error was caught by one
  active worker, corrected, and the affected groups rerun. All subsequent native
  captures/builds use the same frozen **80-production-module hash inventory**.
- `test-coverage.json` records final pass/skip/failure and missing-node coverage.
  `frozen-production.json`, native phase before/after checks and
  `package-final-verification.json` bind the actual delivered production files.
- All four packaging tests PASS: real wheel/sdist metadata/content and installed
  smoke for both artifacts. Both archives match every one of the frozen 80 modules.
  Private evidence, tests and model artifacts are excluded. Existing README/license
  classifier warnings are nonfatal and not repaired by changing package metadata.
- Additional installed P12/P29/P34/P35 CPU acceptance passes under both scientific
  profiles, with current version fields, real installed worker package roots,
  unchanged parent RNG and no parent CUDA initialization. New native GPU and old
  legacy artifacts reopen with the installed readers.

## Bounded native acceptance on the frozen source

`final-native/summary.json` records four successful outer-observed phases. All
launcher exits are zero, all outer and inner owned Job observations are empty,
and every real inference has completed terminal/execution evidence. Exit zero
alone is not used as proof.

- ORT **1.23.2 CPU**: actual ReLU execution plus fusion capture. Fusion remains
  unmapped/unknown, as required; a provider list is not execution proof.
- ORT **1.20.2 CUDA**: actual CUDA ReLU and controlled CUDA/CPU partition. CUDA
  ReLU and CPU ML Scaler placement mismatch are observed. Internal transfer
  source attribution and runtime-error/retry fallback remain unknown, not invented.
- P34: actual AMD Ryzen 7 5800H CPU and NVIDIA RTX 3060 Laptop GPU, same seeded
  CNN/weights/input `[8,3,64,64]`, FP32, three warmups/twenty samples. Model/input
  hashes and execution contracts match, and GPU UUID is independently observed.
  The final capture records 5.644040 ms CPU and 2.084200 ms GPU mean batch latency;
  instrumented completed-image throughput is 1006.997/s and 1393.665/s respectively.
  Concurrent CPU regressions affected the host environment; these are descriptive
  protocol/compatibility observations, not isolated speedup or thermal guarantees.
  Latency/throughput are eligible; logical model bytes are equal. RSS, CUDA
  allocated/reserved and physical VRAM are distinct; memory/energy superiority is
  not inferred. Quality/energy remain null. This is not two-GPU/edge acceptance.
- P31: **2.0014298 seconds only**, 235 steps / 470 images, original model hash
  `5031500c33a8a2629c3fad9e52597ca0dc7597edff6b1a4c17e64e62861ccdae`, input
  `[2,3,16,16]`, FP32, seed 0, no allocator quota, real terminal/close and JSON/CSV.
  Parent RNG and uninitialized CUDA state are preserved. No duration extrapolation.
- Saved-native reader/corruption checks plus explicit opt-in CUDA quota OOM/fresh
  worker recovery: **19 PASS**, included in development's deduplicated coverage.

Revised launcher fault injection covers interruption, secondary BaseException,
reopen/export/serialization/fsync/link/status failure, exact original exception
preservation, non-overwrite and forced-zero-exit non-completion. The fresh short
native Windows harness tests three real CPU Job/interruption/export scenarios;
an additional actual outer observer success/timeout retains zero-active cleanup.
Physical console Ctrl+C delivery and disk/power-loss durability are not claimed.

## Open gates and final plan

**B4 OPEN / native 0xC0000409 root cause UNRESOLVED.** CUDA saved-activation is
still rejected before allocator/hook setup; its production guard is unchanged.
No training activation crash reproduction or unconditional enabling occurred.
No new unexpected native crash, state corruption or owned-process leak has been
observed in the completed final checks. Inference passes do not close B4 or
certify arbitrary CUDA training/profiling.

Minimum Torch does not acquire the private 2.5.1 task adapter, minimum CUDA was
not installed/tested, and TensorRT/edge/two-GPU/universal backend support is not
claimed. The historical 10.13% P31 throughput decline remains unexplained; no new
performance, memory-growth or temperature acceptance threshold is invented.

The [integrated final plan](final-integration-plan.md) preserves the original
7200-second work window, model/input/seed/device/runtime, 5-second observations,
12-row chunks, 2000-row/128 MiB bounds, 120/60-second startup/heartbeat deadlines,
unset quotas, no retry/resume, exact work-end/terminal/close requirements, gap
visibility, source binding and independent parent exit/Job cleanup. Its reviewed
entry point is `tools/p31/start_approved.py`, requiring a separate explicit start.

The final CPU suites are COMPLETED; no test remains running in the recorded
test sessions. Counts above are per profile and deduplicated, not a sum of
overlapping retries. Minimum skips include the deliberately unsupported exact
2.5.1 private task adapter and unavailable/opt-in capabilities. Development skips
are explicit opt-in GPU cases; the separate native acceptance remains recorded.

The following closure items are **NOT COMPLETED**, rather than silently passed:

- Generation of a fresh final P31 source-pinning manifest and execution of its
  real `launch_p31.py --preflight` CLI against that manifest.
- The final aggregate inspection of every historical-in-task controller/PID
  receipt. Completed individual native/test Job cleanup checks passed; that is
  not substituted for the missing aggregate audit.

No fresh long-run output directory, STARTED marker or 7200-second worker was
created for final acceptance. The plan and launcher code are preserved for a
later explicitly authorized completion of those remaining preparation gates.
The historical B4 risk remains OPEN, and P31 remains PARTIAL. No commit/push was
made. Work stops here at the user's request.
