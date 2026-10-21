# M2-P-B10 / P23 Completion R1 — isolation and saved activation

2026-10-21, Europe/Istanbul. **R1 NO-GO; P23 PARTIAL; B10 PARTIAL.**
An unexpected CUDA saved-activation worker crash triggered the B4 stop rule.
No retry, final package acceptance or minimum-runtime execution followed it.
CUDA saved-activation admission is disabled in the final working source.

## 1. Branch, HEAD and preservation

Branch `feature/hyso-models-next`; HEAD
`b0e48e98366bcf8c0054e561434bdf35f358dc74`. All starting uncommitted changes
were preserved. No reset/checkout/clean, stage, commit or push was performed.
The index remains empty. Protected `tests/test_gradient_accumulation.py` SHA256:
`17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`.
Its pre-existing working-tree difference remains user-owned.

## 2. Evidence and chronological gates

Evidence directory E:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-p23r1-20261021`.
`resume-verification-summary.json`, `c-native-crash.json`, XML/logs, frozen B
sources and `resume-protection.json` preserve the exact receipts and checks.
No historical task counts are added to this task's counts.

| Gate | Observed result | Scope and limitation |
|---|---|---|
| CPU CSV fix | 1 PASS, 22.88 s | `resume-export.xml`; handles `cuda=None` |
| RSS/export targeted fix | 2 PASS, 15.07 s | `rss-export-fix.xml` |
| A isolation | 29 PASS, 120.27 s | `a-resume.xml`, before C additions |
| B CUDA matrix | 21/21 PASS, 243.062 s | `b-rss-fixed.json` and receipt; fixed source snapshot |
| Existing regression | 2303 PASS, 607.90 s; no fail/error/skip | `resume-regression.xml`; ran before final C implementation |
| C CPU diagnostics | Partial, no completed final suite | Diagnostic attempts below; cannot aggregate retries as final acceptance |
| C CUDA diagnostic | FAIL: 1 unexpected native worker crash | `c-cuda-fixed.*`, worker exit `0xC0000409` |
| Final all-source regression | NOT RUN | Native-crash stop |
| Final wheel/sdist/install/minimum | NOT RUN | Native-crash stop |

## 3. A — worker recipe and one real fit

Additive `TrainingMemoryRecipe`, `IsolatedTrainingMemoryProtocol`,
`IsolatedTrainingMemoryResult`, `measure_training_memory_isolated`.
Only trusted importable factory names and bounded plain kwargs cross the
boundary. No live parent Trainer/model/tensor/optimizer serialization, factory
execution or extra fit/replay occurs in the measurement API. Factory constructs
the standard Trainer/loaders/optional selected stream inside the child.
Windows support is native-verified; other OS admission is unsupported.

## 4. A — limits, outcomes and cleanup

Stdlib bootstrap waits for parent Job admission before importing HYSo/Torch.
Kill-on-close Job defaults: 4 GiB/process, 8 GiB/Job private commit and 4 active
processes; bounded timeout/log/result monitors and atomic publication.
The worker sets native CUDA allocator fraction .15 before workload construction.
Commit limits are not RSS limits; allocator fraction is not physical VRAM.
Success/unsupported/workload exception/resource limit/timeout/typed OOM/native
crash/invalid receipt remain distinct. All successful receipts require zero
active owned processes. Child/grandchild cleanup, real VirtualAlloc denial,
log/result ceilings, training-forward timeout, constructor/forward failures,
GC-disabled weakrefs, immutable export and corrupted receipts were tested.
The parent does not call its RNG or CUDA allocator APIs to restore state.

## 5. A — CSV and inconsistent Windows RSS

CPU CSV boundary export now treats a null CUDA observation as an empty mapping.
An initial B restart failed with a raw Windows tuple:
RSS `1053388800`, HWM `1053384704`; this is retained in `b-resume.stderr.log`.
The fix retains raw tuples and marks the entire RSS summary unsupported with
`inconsistent_native_windows_rss_hwm; raw_tuple_retained`. It does not clamp,
retry or synthesize a peak. A deterministic injected-tuple test passed.
The final frozen B matrix happened to contain no inconsistent RSS tuples.
Legacy in-process RSS behavior is unchanged.

## 6. B — ownership, actual peaks and fixed code

The originally started `window-cuda-b-final` had no completed result/receipt and
was not counted as success. Restart `b-resume` failed on RSS and was preserved.
Final B runs from E/`b-rss-fixed-frozen`, with SHA256 manifest, not changing
checkout sources. Native RTX3060 Laptop / Torch2.5.1+cu121, deterministic
algorithms, CUBLAS `:4096:8`, cudnn deterministic, TF32 disabled.
18 SGD/Adam/AdamW × FP32/AMP × default/non-default stream/overflow combinations,
one no-reset case, one typed allocator OOM and one timeout: **21 distinct cases**.
Parent allocator stats, Python/NumPy/Torch CPU/CUDA RNG and full state digest
were exact in successful cases. Lazy optimizer allocation, uneven `[4,1]`
windows and overflow skip/recovery were checked. OOM/timeout published no
completed measurement and left zero owned processes.

## 7. B — peak domains and overhead

Only explicit worker `allow_peak_reset` authorizes resets. Selected-stream
completion is synchronized before reset and final reads; wait/query overhead
perturbs the probe and is recorded. Each reset belongs to a worker PID/probe ID
and the real P19 window/attempt/sample ledger. Before-reset current/lifetime
counters, after-reset baseline, actual window allocated/reserved peaks,
incremental peaks and allocation-retry/OOM deltas are separate fields.
Reset is irreversible; later boundary counters have a different peak domain.
No physical/device-wide VRAM, hidden-stream or concurrent attribution is made.
No permission means true-window peaks are null/unsupported, not lifetime minus
baseline. P09 and the legacy P23 never-reset contract are unchanged.

## 8. C — experimental CPU adapter

Opt-in `saved_activation_lifetime` uses `saved_tensors_hooks`. Native packed
slots own original tensors; recorder registries contain weak references and
plain metadata. No clone/detach/CPU-copy occurs in hooks. Records include
dtype/device/shape/stride/logical bytes/storage extents/alias IDs, parameter or
buffer alias flag, epoch/microbatch/window and forward/loss/backward-setup stage.
Events count real pack, unpack and native packed-payload disposal. Scalar AMP
scale multiplication is labeled backward setup. Native payload disposal is a
host event and is not evidence of CUDA storage free.

## 9. C — lifetime, alias and incomplete coverage

Live logical bytes and unique saved storage extents are reconstructed and
validated against events. Multiple aliases may exceed unique storage bytes;
non-parameter storage excludes registered parameter/buffer aliases but still
includes saved inputs/targets. These are not activation peak VRAM. Original
tensor/storage/device-free lifetime and physical activation memory stay
null/unsupported. Total overhead and generalized slowdown stay unknown; hook
query wall time alone is recorded.
Retained graphs cannot publish a completed lifecycle. On exception, weakly
registered owned slots are cleared to break abandoned saved-slot/graph cycles;
cleanup releases are never exported as completed native lifetime evidence.
Custom autograd, activation checkpointing and in-place modules are outside
validated coverage and rejected when detected. Nested hooks, arbitrary custom
execution, graph reuse and hidden streams are not general supported adapters.

## 10. C — CPU diagnostic attempts and exact replay

`c-current.xml`: 6 exact worker state/RNG cases passed (Linear SGD/Adam/AdamW,
tied storage, Conv/ModularCNN, EarlyStopping), then checkpoint test failed because
the fixture used a non-ModularCNN checkpoint profile. Fixture corrected.
`c-cleanup-fixed.xml`: checkpoint interruption/resume/completed and three real
worker GC-disabled success/post-forward/backward cleanup cases passed; the next
direct weakref test failed due to its own `locals()` snapshot retaining the
model. Replaced the snapshot with explicit nullable local ownership.
`c-unit-fixed.xml`: 9 direct lifetime/GC/retained-graph/custom/checkpoint/alias
checks passed; Manager test failed for missing metric axis (`invalid_axis`).
Added explicit `epoch=1`, **not rerun after the native-crash stop**.
Rehashed tamper tests, final full CPU suite, final export/Manager acceptance and
minimum Torch compatibility are not claimed as passed.

## 11. C — unexpected native crash and B4 stop

First CUDA C attempt stopped with a Python unsupported outcome at AMP scaler
setup; no native crash in that attempt. Backward-setup attribution was added.
Second diagnostic `c-cuda-fixed` produced 12 completed results (SGD6 + Adam6)
before the 13th case, AdamW FP32/default stream/no overflow/reset enabled,
crashed. Worker PID **19264**, exit **3221226505 / 0xC0000409**, elapsed **20.672 s**,
no worker receipt/completed measurement. The production outcome is
`native_crash`; Job active processes **0** after termination.
The parent harness exited **1** from its assertion; its receipt elapsed143.266s,
timed_out=false. This is not a parent native crash, an expected timeout, typed
OOM or unsupported workload. The diagnostic source was under active
development, not a release freeze. Post-stop source hashes are not a claim to
reconstruct the crashed process's imported bytes.

**No retry followed. No root cause is inferred from the exit code.**
The prior B4 issue is not asserted to be the same defect. CUDA saved-activation
admission now raises explicit unsupported before CUDA setup/factory construction
because its crash gate is unresolved. This guard itself has only static
inspection after stop, not a subsequent native acceptance execution.

## 12. Regression, packaging and minimum runtime

The 2303 existing CPU regressions covered P09/P10/P11/P17, P19/P20/P21/P22/P23/P25,
Trainer/gradients/scaler/EMA/schedulers/EarlyStopping, checkpoint/resume,
Experiment Manager and benchmark state safety. They are evidence for that
pre-C run, not final C acceptance. `regression-files.txt` records exact selection.
Final native old-feature regressions and the post-C full regression were stopped.
Packaging inventory now includes **47 production modules**. Source inventory
and Python3.10 syntax can be checked statically, but wheel/sdist builds,
source/archive/installed byte equality and Python3.10/Torch2.2.2+cpu installed
smoke **were not run**. No dependency/runtime/driver upgrade or download occurred.
The existing minimum-runtime location was found, not changed.

## 13. Schema, export and Experiment Manager

Legacy P23 schema1 fixture deserialize passed in A. Existing TrainingMemoryResult,
Trainer/checkpoint/Manager schemas and default observation paths are unchanged.
The independent isolated wire type validates source SHA, receipt ownership,
window/source ledgers and opt-in activation events. JSON carries full evidence;
CSV carries boundaries, windows and saved events. Atomic/no-clobber and
canonical integrity checks remain. Manager test uses copied report artifacts,
SHA verification and explicit epoch metric with finish/close/reopen; a tracking
failure must not retrain. Final Manager/tamper/installed acceptance is pending.

## 14. Exact task files

Production (5):

- `hyso/performance/__init__.py`
- `hyso/performance/isolated_training_memory.py`
- `hyso/performance/_training_process.py`
- `hyso/performance/_training_memory_worker.py`
- `hyso/performance/_saved_activation.py`

Tests/fixture (6):

- `tests/test_packaging.py`
- `tests/test_isolated_training_memory.py`
- `tests/test_saved_activation.py`
- `tests/cuda_isolated_training_memory_validation.py`
- `tests/training_memory_factories.py`
- `tests/fixtures/training-memory-p23-schema1.json`

Documentation (4):

- `docs/audit/m2-p-b10-p23-isolation-activation.md`
- `docs/guides/performance-training-memory-isolated.md`
- `docs/guides/performance-training-memory.md`
- `docs/roadmap/hyso-performance-backlog-v1.1.md`

15 task files plus the unchanged user-owned gradient working-tree difference.
No evidence directory or build output was added to package sources.

## 15. Final decisions and remaining gates

| Domain | Decision |
|---|---|
| A worker isolation | PASS in observed Windows scope (29 tests before C); final source release gate pending |
| B true window peaks | PASS in frozen native/no-activation scope (21); final source/install gate pending |
| C saved activation | FAIL native CUDA gate; CPU PARTIAL; CUDA admission disabled |
| P23 | PARTIAL; not eligible for implemented |
| R1 | NO-GO after B4 stop |
| B10 | PARTIAL / no full-scope GO; P21 worker service trace remains partial |

Feature count stays **17 implemented /7 partial /11 planned =35**. P25 remains
implemented. B4 OPEN/root cause UNRESOLVED, B6/B7 PARTIAL and P18 planned/blocked
remain. No facade work was started.

Next work requires a separate authorized native-crash investigation preserving
this failure. After resolution, freeze sources and complete C CPU/tamper/Manager,
source/installed CUDA matrix including Conv/tied/early/resume/GC cases, final
existing/native regressions, minimum installed smoke and archive byte integrity.
Do not replace this failure with a later successful retry in the audit.
