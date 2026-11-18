# P23 isolated training memory — R1 experimental scope

2026-10-21, Europe/Istanbul. **P23 PARTIAL; R1 release NO-GO.**
The saved-activation CUDA diagnostic encountered an unexpected native worker
exit `0xC0000409`. The B4 stop rule applies. CUDA saved-activation admission is
disabled in the current source; the final source, installed distributions and
minimum runtime have not passed a release gate. See the
[R1 audit](../audit/m2-p-b10-p23-isolation-activation.md).

`measure_training_memory()` and its existing schema1/never-reset contract are
unchanged. The additive API accepts a `TrainingMemoryRecipe`, never a live
Trainer/model/tensor/optimizer. A trusted importable factory constructs all
training objects in a fresh worker. This executes trusted Python, not a security
sandbox. The factory must return a dictionary with `trainer`, `train_loader`,
`val_loader` and optionally `stream`. The Trainer must be standard HYSo; initial
loader admission requires `DataLoader(num_workers=0)`.

For a locally maintained `my_workloads:build` factory:

```python
from hyso.performance import (
    TrainingMemoryRecipe, IsolatedTrainingMemoryProtocol,
    TrainingMemoryProtocol, measure_training_memory_isolated,
)

recipe = TrainingMemoryRecipe(
    "my_workloads:build", device="cuda:0", seed=71, epochs=2,
    factory_search_path="C:/trusted/workloads", kwargs={"optimizer": "adamw"},
)
protocol = IsolatedTrainingMemoryProtocol(
    timeout_seconds=60, allow_peak_reset=True,
    memory_protocol=TrainingMemoryProtocol(device="cuda:0"),
    saved_activation_lifetime=False,
)
result = measure_training_memory_isolated(recipe, protocol=protocol)
data = result.to_dict()
if data["outcome"] == "success":
    result.to_json("isolated-memory.json")
    result.to_csv("isolated-memory.csv")
```

Use an indexed CUDA device. `factory_search_path` is an explicit directory added
to the admitted worker's import path. Only plain JSON-compatible kwargs are
accepted. Factory names cannot be lambdas or `__main__` objects. Recipe seed and
fit budget belong to the child; the parent does not import the factory, pickle
live objects, train, restore RNG or query/reset its CUDA allocator.

On Windows, Job admission precedes Torch/factory imports. Default limits are
4 GiB private commit per process, 8 GiB per Job, 4 active processes, 60 seconds,
16 MiB result transfer and 64 KiB per stdout/stderr prefix. Logs are continuously
drained; an exceeded limit terminates the owned process tree. CUDA native
allocator fraction defaults to .15, configured before factory construction.
Private commit is not RSS, and allocator fraction is not a physical VRAM limit.
Linux is `unsupported` until separately validated.

Outcomes are `success`, `unsupported`, `workload_exception`, `resource_limit`,
`timeout`, `oom`, `native_crash` or `invalid_receipt`. Read outcome and receipt
before reading measurement. A native crash is never relabeled as unsupported or
OOM. A terminated/failed workload has no completed measurement. Success includes
validated PID/token/exit/cleanup evidence and one real fit, without retries.

Peak reset requires `allow_peak_reset=True` and an owned CUDA worker. Selected
stream synchronization precedes reset and final peak reads; recorded wait times
are instrumentation overhead. Window IDs, actual samples and applied/skipped
attempts link to embedded P19. Pre-reset peaks, current allocated/reserved,
actual reset-domain peaks, incremental allocated/reserved peaks, retries and OOM
counter deltas are separate fields. Reset is irreversible. Without permission,
window peaks are null/unsupported. No empty-cache, context reset, hidden-stream
attribution or physical VRAM inference is performed.

Raw native RSS tuples are retained. If Windows returns HWM below current RSS or
a regressing HWM, the reason is explicit and the entire RSS summary becomes
null/unsupported. Values are neither clamped nor retried. Independent training
and allocator domains remain separately validated.

`saved_activation_lifetime=True` is an **experimental CPU-only** opt-in in this
working tree. It records bounded pack/unpack/native packed-payload disposal,
dtype/shape/stride/device/logical bytes, live storage aliases and source
epoch/microbatch/window IDs. Hooks do not clone, detach or copy tensors to CPU.
Only autograd packed slots own payloads; the recorder holds weak references.
Abandoned slots are cleared during exception cleanup without global GC. An
incomplete lifecycle cannot produce a successful activation result.

Its measured lifetime domain is host autograd saved-payload retention. It does
not establish original tensor, storage, GPU completion/free or physical
activation memory lifetime. Logical saved bytes and deduplicated full storage
extents are not activation peak VRAM. Registered parameter/buffer aliases are
marked; other saved data includes inputs, targets, intermediates and AMP scale
operands. `backward_setup` distinguishes scaler multiplication from forward
activation saves. Hook query time is recorded; total instrumentation overhead
and a generalized baseline slowdown remain unknown.

Custom autograd, reentrant/non-reentrant activation checkpointing, retained
graphs, in-place modules and saves outside the admitted forward/loss/backward
setup are rejected when detected. Nested hook replacement, arbitrary custom
execution and hidden streams are outside validated coverage. A retained/reused
graph cannot be interpreted as a completed native lifetime. This is not a
general activation adapter, and the CPU acceptance suite is still incomplete.

Exports are immutable validated JSON and CSV (boundary/window/saved event rows),
atomic and no-clobber by default. SHA checks detect corruption, not malicious
authentication attacks. Export only after measurement. Existing Experiment
Manager report artifacts/metrics require no new kind or migration; never log a
null unsupported measurement as zero. Installed export/Manager acceptance and
minimum-runtime validation remain pending after the native-crash stop.
