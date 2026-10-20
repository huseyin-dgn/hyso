# M2-P-B4-R1 — Native replay crash investigation

2026-10-20, Europe/Istanbul. **B4 NO-GO; B5'e geçilmedi.** Yeni native revalidation
temizdir, ancak orijinal crash'in kök nedeni açıklanmış/giderilmiş değildir.
Sınıflandırma **D — aralıklı/koşula bağlı, henüz kök nedeni kanıtlanamayan failure**.
Başarılı tekrarlar önceki failure'ı kapatmaz.

## 1. Başlangıç ve değişiklik sınırı

Branch `feature/hyso-models-next`, HEAD
`724e6bc63802a2f7a85dba40fb38167c56a9415c`, index boş. B4 production/test/docs
değişiklikleri henüz commit edilmemişti; tamamı korundu. Kullanıcının
`tests/test_gradient_accumulation.py` SHA256 değeri başlangıç ve finalde
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Production, test, ortak harness, packaging, backlog ve eski audit değiştirilmedi.
Repository'deki tek R1 eklemesi bu rapordur; tanılama/runner dosyaları TEMP'tedir.
Commit/push/checkout/reset/rebase, dependency/driver/CUDA/Torch/Windows güncellemesi
ve B5 çalışması yapılmadı. AGENTS.md bulunmadı.

R1 evidence kökü **R**:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-m2pb4-r1-20261020`.
Orijinal B4 evidence **E**:
`C:\Users\hdgn5\AppData\Local\Temp\hyso-m2pb4-20261020`.
`R/initial-files.json` bütün tracked + B4 untracked dosyaları;
`original-evidence-hashes.json` E'deki mevcut üst düzey kanıt dosyalarını korur.

## 2. Orijinal failure ve exact invocation

[B4 FAIL audit](m2-p-b4-sweeps-precision-final-gate.md) tarihsel kayıt olarak aynen
korunur. E/run-cuda.py, E/cuda-runs.json ve E/cuda-script-inventory.json incelendi.
Replay script'i:
`C:\Users\hdgn5\OneDrive\Belgeler\GitHub\hyso\tests\cuda_trainer_replay_conformance.py`.
SHA256 `4eb3216db462bf77ca4bc955b26fecf067f5cb8a40fcd6f4845f2fc08d24509d`.

Orijinal argv:

```text
C:\Users\hdgn5\AppData\Local\Temp\hyso-p01b-tests-40awqikl\Scripts\python.exe
-B
C:\Users\hdgn5\OneDrive\Belgeler\GitHub\hyso\tests\cuda_trainer_replay_conformance.py
--output
C:\Users\hdgn5\AppData\Local\Temp\hyso-m2pb4-20261020\cuda-replay.json
```

Cwd repository root. CUDA_VISIBLE_DEVICES=0, CUBLAS_WORKSPACE_CONFIG=:4096:8,
MPLBACKEND=Agg, PYTHONDONTWRITEBYTECODE=1, TMP/TEMP=E. Launcher inherited other
environment values; E did not snapshot the entire inherited environment, so its
historical CUDA_LAUNCH_BLOCKING/allocator/module-loading values cannot be proved
retroactively. R1's corresponding values are explicitly recorded as unset.

Original order: runtime → imagefolder → modular → trainer-updates → lifecycle →
save → restore → **replay** → benchmark → benchmark-export → Experiment → B1 →
B3 → B4. Launcher uses synchronous subprocess.run(timeout=300), file-backed
stdout/stderr and a fresh interpreter per script; next script starts after wait.
Replay exited after23.515s: unsigned3221226505, signed−1073740791,
hex**0xC0000409**. It was not a timeout. No successful result JSON was published;
stdout empty, stderr only existing GradScaler FutureWarning. First case directory
fp32-ema-False exists but contains no checkpoint/artifact. This bounds progress
to the first case before checkpoint publication; it does not identify the exact
faulting Python/CUDA operation. Other eight unique scenarios were unobserved;
rollback-False/True were already covered by restore in a separate process.

## 3. Windows/WER kanıtı

Original Application Error1000 Record159956:2026-10-20 17:22:13 local,
Python PID0x3D1C, nvcuda64.dll32.0.15.9174, offset0x18fd6e5, exception0xc0000409.
WER Event1001 Record159957 at17:22:26 contains the same report identity
`07926442-bba4-4c3b-88ee-9b83dca77152`. R/original-report.wer additionally records
EventType=BEX64 and **Sig[8] exception data=2**. The loaded-module inventory includes
Torch/CUDA/cuDNN and DriverStore nvcuda64.dll; faulting process image is the real
conda Python, consistent with the venv redirector.

Microsoft documents 0xc0000409 as a noncontinuable fail-fast exception, with a
reason parameter; code alone does not prove a driver root cause.
[__fastfail reference](https://learn.microsoft.com/en-us/cpp/intrinsics/fastfail?view=msvc-170).
Reason2 corresponds to FAST_FAIL_STACK_COOKIE_CHECK_FAILURE in Microsoft's
[fast-fail code explanation](https://devblogs.microsoft.com/oldnewthing/20190108-00/?p=100655).
This narrows the reported termination mechanism; it does not identify who
corrupted state or prove that the reporting module originated the corruption.

WER named a temporary mdmp, but that exact path no longer exists. Archive contains
only Report.wer; no Python CrashDumps file or available cdb/WinDbg/ProcDump command
was found in the inspected locations. No debugger was installed, dump registry
setting changed or driver updated. No native stack/exception context is available
from the original crash. R/dump-availability.json and wer-events.json preserve this
limit. System log query17:20–17:24 returned no Display/nvlddmkm/WHEA matching events
and no query error; this does not exclude a user-mode driver/runtime interaction.

## 4. Actual runtime/device

R/runtime-probe.json and complete-gate/gpu-runtime.json record:

| Field | Observed value |
|---|---|
| Launcher sys.executable | hyso-p01b-tests-40awqikl/Scripts/python.exe |
| Actual process image/base_prefix | C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe |
| Python |3.10.18 Anaconda, MSC v.1929,64-bit|
| OS |Windows10.0.26200|
| Torch / torchvision |2.5.1+cu121 /0.20.1+cu121|
| Torch CUDA build / actual cudart API version |12.1 /12010|
| Driver / cuDriverGetVersion |591.74 /13010|
| cuDNN |90100|
| GPU |RTX3060 Laptop, capability8.6,6144MiB nvidia-smi total|

Driver-reported CUDA capability and actual packaged runtime are recorded
separately; different version numbers alone are not treated as a defect.
Before the clean gate nvidia-smi showed615MiB used/5380MiB free; later probe showed
917MiB used/5078MiB free. These are different global WDDM snapshots, not a leak
measurement. Other WDDM clients exist; per-process memory is not exposed by this
driver mode. Original crash-time contention/quiescence was not captured and
cannot be reconstructed. No unrelated process was terminated.

## 5. Kontrollü denemeler

| Attempt | Scope | Replay PASS | Exit | Replay süre |
|---|---|---:|---|---:|
| Original B4 | Original combined order |0 observed|0xC0000409|23.515s|
| R1 isolated-1 | Same unchanged script/interpreter/flags |10|0|23.391s|
| R1 complete-gate | Same complete runner/order, unchanged source |10|0|21.531s|

First action after evidence/source capture was one isolated native replay, not a
full-suite retry. R/reproduce.py uses original argv/environment configuration;
only output/TEMP roots move to R/isolated-1 to preserve the old evidence. Cwd and
script flags are unchanged; no CUDA_LAUNCH_BLOCKING, inserted synchronization,
GC/cache reset, exception suppression or assertion change was added.
command.json records UTC start/end, launcher PID8272, actual argv/env, wait and
signed/unsigned/hex exit. Original and isolated results were retained separately.

Then original E/run-cuda.py was copied byte-exact to R/complete-gate/run-cuda.py,
with its own OUT directory. Same14 scripts/order and same source versions ran
sequentially. It was not modified to retry failures. Both planned replay attempts
passed; no bounded-loop retries or additional attempts were made to obtain GO.
Source/script SHA equality and output roots are recorded. Two successful
observations do not estimate crash probability or establish a fix.

## 6. Ownership/state/harness incelemesi ve sınıflandırma

Replay and trainer_replay_support, Trainer._update/fit, resume transaction and
checkpoint RNG paths were examined. Each replay compares uninterrupted training
with a coherent checkpoint prefix/restart: exact tensor bytes, model/optimizer/
scheduler/scaler/EMA, per-module modes, Python/NumPy/Torch CPU/CUDA RNG, sample
order and counters. It uses native operations/current stream, without explicit
custom CUDA event/stream scheduler, allocator peak reset or empty_cache. Resume
stages an owned CPU checkpoint payload, validates before mutation, restores
runtime/RNG and has transactional negative rollback tests. Current native tests
pass these existing contracts; this does not prove no memory corruption exists.

Original launcher waited for a nonzero process result and propagated it. No
timeout, swallowed failure, script omission or overlapping gate child invocation
was found. Windows venv redirector versus real Python image difference is normal
for this invocation and did not turn the crash into a successful process result.
Process isolation does not reset global driver/WDDM state; previous script exit
does not by itself prove global quiescence. Retained resources/order/contention
are hypotheses, not established causes.

B4 sweep code is not imported by the inspected replay dependency chain: hyso root
imports models/callbacks/augmentation; replay imports Trainer/checkpoint and test
support, not hyso.performance. R1 did not execute sweeps inside replay. All these
sources match B4 freeze. This rules out a direct sweep call in the inspected
execution path, not every possible environment interaction. No production or
harness defect was demonstrated. **A/B/C are unproven; D is the supported
classification.** No speculative synchronization/GC workaround was patched in.

## 7. Eight missing identities and complete native gate

All eight missing original unique scenarios passed in both R1 observations:
fp32-ema-False/True, fp16-ema-False/True, native-overflow-recovery-ema-False/True,
negative-plan-False/True. rollback-False/True also passed in replay; their identical
restore identities are deduplicated in the complete gate.

| Complete R1 script | PASS executions |
|---|---:|
| runtime / imagefolder / modular |7 /10 /18|
| trainer updates / lifecycle / save |40 /16 /3|
| restore / replay |5 /10|
| Benchmark / export / Experiment |45 /8 /2|
| performance B1 / pipeline B3 / sweeps B4 |15 /8 /11|

**14 processes,198 PASS executions,196 distinct PASS,2 duplicate executions,
FAIL/ERROR/CRASH/SKIP0; all exit0.** All old162 native identities and34 B1/B3/B4
identities are covered on the same fixed final source. Exact identity lists,
source SHAs, argv/env, elapsed times, raw results and stdout/stderr are in
R/complete-gate. R1 isolated10 are not added as new identities: combined R1
workload208 PASS executions,196 unique,12 repeated executions. Historical B4
process crash remains separately recorded and is not included as a passing case.

No new regression test was justified without a reproduced defect/fix. Current
native gate contains Trainer/checkpoint/rollback, P09 allocator and C1 protection;
existing CPU regressions are preserved as described below.

## 8. Full CPU validity

Production and common test/harness source did not change in R1. B4's existing
**2313 PASS/1 expected CUDA-hidden SKIP**, including all1996 old identity/status
entries, remains valid for the identical final source. It was **not rerun** or
presented as a new R1 full CPU execution. SHA comparison includes the95 production/
test files in E/gate-source-freeze.json plus the preserved B4 guide/packaging files.
R/verification-summary.json records reuse and source equality explicitly.

## 9. Package/minimum validity

Packaging unchanged; no rebuild/install was necessary. Existing29-module final
wheel/sdist/source-byte equality, Torch2.2.2+cpu installed smoke and installed B1/
B3/B4 CUDA checks remain the B4 evidence, not new R1 runs. Current production
bytes equal the original archives and installed-final target. Archive SHA256:

- wheel `ab1a0fdbf1ef8603222801c679bea9a022182569363f388a56b8e5badd005e49`
- sdist `41ec2472600f13b049d9c93b6f87f672f856c0791181f8175624436f5085070e`

Minimum-prefix distribution METADATA/RECORD hashes still match B4's recorded
after-state; this is not a full binary environment audit. No dependency/runtime
change was made to make replay pass. Existing package/checkpoint/schema/API
contracts remain untouched.

## 10. Final karar ve sonraki teşhis

**Clean native revalidation PASS; root-cause recovery unresolved; B4 NO-GO.**
Eight missing scenarios are now genuinely observed, but acceptance also requires
an evidenced explanation/fix of the crash and no concealed unresolved risk. That
criterion is not met. Old B4 FAIL audit/backlog statement remains unchanged;
P06/P07/P08 implemented feature statuses remain, without milestone GO.

Next useful diagnostic is **one debugger-controlled reproduction that captures
the native exception record, all-thread stacks and loaded modules at the fail-fast
event**, with case/phase progress, exact PID/UTC/env/GPU snapshots and bounded
attempt count. The missing temporary WER dump cannot be recovered by retrying a
healthy run. Available debugger/dump tooling should be prepared before another
targeted reproduction; then isolate the first case and its pre-checkpoint stages
if the crash recurs. Do not infer a driver fix, change packages, or add blind
synchronization based solely on the faulting module. No update/reboot/background
application change was performed in R1. **B5'e geçilemez.**
