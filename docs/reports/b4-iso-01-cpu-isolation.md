# B4-ISO-01: owned-worker phase isolation, CPU only

2026-11-18. B4 **ROOT_CAUSE_UNRESOLVED**. GPU admission remains blocked;
this preparation grants no GPU execution permission. No CUDA context/workload,
GPU hooks, P31 run, or full suite is part of this task.

Initial local and GitHub `feature/hyso-models-next` both resolve to
`d86a5ab14e125f20633dfecd0ef579b453c8cc6b` (GitHub commit API checked).
This change intentionally creates a local successor; no remote update.

## Two different historical observations

Only the existing [SiLU investigation](../audit/m2-p-b4-r3-native-callpath-investigation.md)
and [P23 Dropout investigation](../audit/m2-p-b4-p23-native-crash-investigation.md),
plus their already recorded source analysis, are used here. No missing-dump
search, live debugger, new binary audit, or GPU identity/admission investigation.

| Evidence | A: pre-P23 B4 | B: P23 B4 |
|---|---|---|
| Worker / exception thread | 15644 / 13196 | 19264 / 21692 |
| Immediate host path | SiLU call/redispatch/structured implementation → torch_cuda internals → cuLibraryLoadData+0x4f → NVIDIA internals → fail-fast | Dropout call/native_dropout/redispatch → torch_cuda internals → cuLibraryLoadData+0x4f → NVIDIA internals → fail-fast |
| Signature | 0xC0000409, parameter 2, nvcuda64+0x18fd6e5 | Same recorded signature |
| Reported driver | 591.74, module 32.0.15.9174 | Same recorded module version/header identity |
| Model/workload | ModularCNN/SiLU, replay first FP32/EMA-off case; ECA, 7 examples, batch 3, accumulation 2, AdamW | tiny Linear(3,4)/Dropout(.2)/Linear(4,2), FP32 AdamW, seed 71, two epochs, default stream, hooks ON/reset ON |
| Scope | Predates the P23 saved-activation implementation | Thirteenth C case, after 12 SGD/Adam results; no failed-worker publication |
| Exact phase/iteration | Exact Python source line and corruption origin UNKNOWN | Epoch/microbatch/attempt and exact Python line UNKNOWN; do not call it the first Dropout |
| Loaded source identity | Crash-time complete loaded-source hash set UNKNOWN | `unverified_source_identity`: mutable checkout, missing pre-crash import hash; post-stop files are not an attestation |
| Binary proof limit | Local PE timestamp/image-size match reported; full crash-time byte/PDB identity UNKNOWN | Same limitation; current version strings/hashes cannot repair historical identity |

The SiLU path is consistent with `ModularCNN.forward` → `ConvNormAct.forward` →
SiLU → Torch dispatcher. The P23 path is consistent with `_launch` → admitted
stdlib worker → `_run_recipe` → tiny factory → `_measure_training` → Trainer.fit
→ model forward → Dropout → Torch dispatcher → CUDA library load. These are
source/stack relationships, not recovered exact Python frames. Neither immediate
stack demonstrates a payload destructor, peak reset, optimizer step or Job
termination as the faulting call. Earlier corruption, driver/library loading,
allocator/reset interactions, and saved-hook interactions remain unproven
hypotheses. Identical fail-fast signatures do not establish identical causes.

## Actual recorder and ownership relationships

`_launch` owns the child and Windows Job, assigns the child before writing its
token/PID admission file, drains bounded pipes, enforces the existing deadline,
and verifies owned-tree cleanup. The stdlib bootstrap checks admission before
importing HYSo/Torch. `_run_recipe` configures the device, builds the factory,
creates `_SavedActivationRecorder` when enabled, and supplies `_IsolatedRecorder`
to the shared one-fit `_measure_training`. Trainer's existing stage callbacks
drive `_MemoryRecorder` and the saved-activation stage context. No training
forward/backward or optimizer algorithm is rewritten.

`_Payload.tensor` **and `_Payload.owner` are strong references**. The reverse
recorder slot entry is `weakref.ref(payload)`; the owner is not a weak reference.
Pack/unpack retains the original tensor identity and checks its version. Native
slot disposal calls `__del__` → `clear` → `release`; `clear` drops owner before
the tensor, allowing reentrant `close` without a second release. An abandoned
output graph may form a native cycle requiring explicit owned cleanup. Forced
cleanup is host payload release, not proof of physical GPU storage release.
Existing tests exercise identity/alias storage, version mutation rejection,
native disposal, reentrant release, abandoned graph, disabled GC, forced cleanup,
exceptions, and hooks leaving scope. No demonstrated failing lifecycle
regression justified changing `_saved_activation.py`; its bytes are unchanged.

## Private, opt-in durable CPU phases

The existing ready receipt observes Job admission. Existing memory stage pairs
cover forward/backward/scaler step in memory, and saved-activation records track
host payload events. These are normally published after the fit; a fail-fast can
lose the entire in-memory ledger. Import, setup, factory, hook scope and cleanup
previously had no durable per-phase record.

The stdlib bootstrap now accepts an internal request key `_phase_trace` with
`path` and optional `max_records`/`max_bytes`. The public protocol and measurement
API expose no switch and never set this key. A trusted, separately authorized
private controller can supply it to `_launch`; it does not grant admission.
The trace opens only after exact token/PID Job admission. The destination must
be absolute, exclusively created, and outside source and transient worker roots.
No existing evidence is overwritten. Default bounds are 512 records/64 KiB;
hard ceilings are 4096 records/1 MiB. The existing log/result budgets are separate
and unchanged. This is instrumentation of a trusted owned process, not a sandbox.

Each JSONL row contains only `phase`, `monotonic_ns`, `pid`, `run_id`, and
`cpu_status` (`begin`, `end`, `error`, `skipped`). No tensors, addresses, model
parameters, exception messages, clones, transfers or detached data are logged.

| Boundary | Durable phase / interpretation |
|---|---|
| A/B | `worker_started` captures bootstrap-entry monotonic time; it is persisted only after `job_admitted`, not a parent OS creation-time attestation |
| C | `torch_import` begin/end brackets HYSo performance import, including Torch; not the exact instant each dependency finishes |
| D | `cuda_setup` begin/end, or CPU `skipped`; existing CUDA saved-hook rejection still precedes availability/allocator setup |
| E | `factory` begin/end brackets construction of model, optimizer, loaders and Trainer together; no finer construction claim |
| F | `saved_hook_context` before/after actual context entry, or `skipped` |
| G/H | Existing `forward` stage begin/end, including the first observed training forward pair |
| I/J | Existing `loss_metrics` and `backward` begin/end |
| K | Existing `scaler_step_attempt` begin/end around scaler.step; not native optimizer-entry or proof a skipped step updated parameters |
| Validation | Existing `training_end`/`validation_end` callbacks bracket the validation segment as `validation` begin/end; no validation forward rewrite |
| L | `recorder_cleanup`, actual `saved_hook_exit`, `activation_cleanup`, workload return/error, and atomic `result_publication` begin/end |

Stage starts are emitted before the existing recorder's pre-stage observations;
ends follow its post-stage observations. They bracket CPU control flow, including
existing recorder work. Validation forward calls do not have these training stage
callbacks and are not relabeled as individual forward observations. The existing
validation segment callbacks distinguish validation from training, without
claiming a validation batch index or exact native call. With tiny's EMA enabled,
validation uses the EMA shadow, another reason not to infer which Dropout invocation
failed from the historical stack. No synchronization or stream
behavior is added or altered. CUDA work can be asynchronous: CPU phase end does
not prove GPU completion, establish corruption origin, or exclude earlier work.

Every row is flushed/fsynced. Native fail-fast may prevent the last record,
Python finally, interpreter teardown, or publication. No last-record guarantee.
Trace overhead changes timing; traced durations are not measurement authority.
The traced worker's P19 projection carries this limitation explicitly as well.
I/O/budget faults latch without interrupting fit/cleanup, then prevent success.
A fault during/after publication yields nonzero worker exit, which the unchanged
parent classification rejects even if a success receipt was written. Owned Job
cleanup, exit-zero requirement, native fail-fast visibility and no-retry behavior
are preserved. Trace files remain outside the disposable worker directory.

## Future isolation matrix — definitions only, no execution

Use exactly `training_memory_factories:tiny`, `modular=False`, `tied=False`,
`early=False`, `nondefault=False`, `amp=False`, `overflow=False`, no failure hook,
AdamW lr .01, FP32, seed 71, epochs 2, default stream, identical loaders (5 samples,
batch 2, train shuffle), accumulation 2, EMA .8, OneCycleLR and grad_clip .1.
Do not substitute the historical SiLU model; that would be a different experiment.

| Cell | Hooks | Peak reset | Current status |
|---|---|---|---|
| CONTROL | OFF | OFF | Requires separate explicit approval and existing strict admission |
| RESET | OFF | ON | Separate subsequent experiment; historical pass is not authorization |
| HOOK | ON | OFF | BLOCKED by existing CUDA saved-activation guard |
| COMBINED | ON | ON | BLOCKED by existing CUDA saved-activation guard |

For every future cell, seal a common manifest of complete workload/dependency
source hashes, factory file hash, explicit model/optimizer/data configuration,
Python/Torch/loaded DLL hashes, CUDA build, driver, device index **and GPU UUID**,
environment/determinism/stream and resource/deadline settings before launch.
Hash the canonical common configuration, then hash each cell configuration
including its two flags separately. Verify the immutable source snapshot and
matching runtime/device manifest inside the owned worker before CUDA allocations;
attach a new run identity and explicit authorization. Current instrumentation
alone does not implement this execution attestation or a GPU permission flow.
GPU UUID and future binary/config attestation are **PENDING**, never inferred
from `cuda:0`, an import path or a version string. Keep known mismatches fail-closed.

The **next single proposed experiment is CONTROL only**, one owned worker, no
retry: tiny/AdamW/seed71/two epochs/default stream, hooks OFF/reset OFF, private
phase trace enabled. It requires fresh explicit user approval, the above sealed
manifest, the existing strict admission, current VRAM/host resource checks and
deadline/cleanup controls. Existing WDDM admission is still blocked; this document
does not bypass it or promise the experiment can currently run. Stop after that
single result or failure. No hook cell can run with the preserved guard. A CONTROL
pass would establish only that cell's outcome, not GPU_PASS, hook safety, or root cause.

## CPU validation and preservation receipt

Only `test_saved_activation.py`, `test_isolated_training_memory.py`, and the new
`test_training_phase_trace.py` are selected. Conftest sets CUDA visibility to -1;
CUDA-path tests use mocks and assert rejection before allocator/hook entry.
New tests compare actual traced/untraced tiny state, exception/unwind cleanup,
durable prefix after an owned CPU `os._exit(23)` simulation (not a native-driver
crash), bounds, admission-before-import, no public opt-in, and late I/O faults.
Final frozen-source run: **115 PASS, 0 FAIL, 0 ERROR, 0 SKIP** (437.63 s).
Final state: **ISOLATION_READY_CPU_ONLY**, never ROOT_CAUSE_RESOLVED or GPU_PASS.
The private trace group contributes 24 cases; existing saved-activation and
isolated-worker cases contribute 91. Earlier development runs are separate
receipts, not added to this final count. Source/config hashes, JUnit XML, logs,
phase prefixes and preservation checks are retained outside the repository at
`C:\Users\hdgn5\AppData\Local\hyso-b4-iso-01-20261118-210352`.

The initial index was empty. The 47 pre-existing user files and 19 artifacts
matched the prior baseline before editing. No changes are authorized to those
files. Only the two unprotected worker/recorder source files, the new test and
this report are intended for a local commit. Historical QA/crash reports remain
unchanged. No reset/restore/clean/rebase/push or dependency/driver changes.
