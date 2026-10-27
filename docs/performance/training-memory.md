# Training memory analysis (P23)

P23 observes training state and native process/allocator memory in separate
domains. Use it for a training run you intend to execute; it does not infer
training memory from inference or replay a live Trainer automatically.

The [P23 acceptance matrix](p23-acceptance.md) identifies the verified CPU/CUDA
scopes and retained unsupported domains.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from hyso.training import Trainer
from hyso.performance import measure_training_memory, TrainingMemoryProtocol

model = torch.nn.Linear(3, 2)
trainer = Trainer(
    model,
    optimizer=torch.optim.AdamW(model.parameters(), lr=0.01),
    loss_fn=torch.nn.CrossEntropyLoss(),
    device="cpu",
    accum_steps=2,
)
loader = DataLoader(
    TensorDataset(torch.ones(5, 3), torch.tensor([0, 1, 0, 1, 0])),
    batch_size=2,
)
result = measure_training_memory(
    trainer, loader, epochs=1, protocol=TrainingMemoryProtocol(device="cpu")
)
report = result.diagnostics()  # Offline derivation; no additional training.
for component in ("parameter", "gradient", "optimizer"):
    print(component, report["logical"][component])
print("RSS observations:", report["rss"])
```

On CUDA, pass an explicitly indexed device such as `cuda:0` matching the Trainer.
The in-process observer does not reset allocator peaks, empty caches or control
global garbage collection. It adds bounded metadata queries and RSS sampling;
their overhead is included. No external profiler or RAM/VRAM quota is required.

## Read each memory domain separately

| Domain | What is measured | What it does not mean |
| --- | --- | --- |
| Parameters and buffers | Registered tensors, dtype/shape and logical `numel * element_size` bytes | Physical process or device footprint |
| Gradients | Existing `.grad` tensors at each boundary | A permanent allocation equal to parameter size |
| Optimizer | Actual initialized tensor state for native SGD/Adam/AdamW; non-tensor metadata separate | An estimate assuming every optimizer has already allocated its state |
| Storage extents | Deduplicated backing-storage bytes within the observed registered state | A complete ownership graph or an additional component to add to logical bytes |
| CPU RSS | Native current-process working-set samples and process-lifetime high-water marks | Python heap size, DataLoader-worker memory, or an exact per-stage training peak |
| CUDA allocator | Current allocated/reserved bytes, native allocator peak counters and their scope | All physical VRAM, driver/context memory or other processes' memory |
| Optional CPU saved payloads | Native eager autograd saved-for-backward logical bytes and observed storage extents/lifetimes | All forward activations, physical activation peak, or proof of final storage deallocation |

Tied tensors are counted once per component. Distinct views retain their logical
sizes and can share storage. Frozen parameters are a subset of parameters, not
an extra component. Cross-component aliases can also overlap. Do not add logical
components, storage extents, RSS and allocator counters into a single total.
Unsupported representations and custom optimizer state coverage remain explicit.

## Training boundaries and gradient lifecycle

The result links its source training segment, epochs, microbatches, accumulation
windows and optimizer attempts. Native snapshots bracket forward, backward,
scaler-step attempts and zero-grad, with ready/end observations around the fit.
Logical state and CUDA allocator queries share those boundary brackets; the
query start/end timestamps are retained. Background RSS samples have their own
query timestamps and are not an atomic snapshot of every training boundary.

Use `boundary_observations` in `diagnostics()` to follow changes. Gradients can
be absent before backward, persist across accumulation microbatches, then be
cleared by zero-grad. `set_to_none=False` can keep gradient storage. Adam/AdamW
state is allocated lazily; a skipped AMP update can leave state uninitialized.
Observe the actual attempt and gradient state instead of applying fixed memory
multipliers. Ready/end exclude construction and include fit finalization, so
their scope differs from an individual forward or backward interval.

## Peaks and optional owned training windows

The maximum sampled RSS or allocator boundary value can miss short transients.
A native lifetime peak also includes activity before the fit. Neither is an
exact training-window peak; subtracting a baseline lifetime peak does not fix
that attribution. Invalid native RSS/high-water tuples are reported unsupported,
without clamping or fabricating a peak.

For an explicitly owned fresh-process recipe, `measure_training_memory_isolated`
can collect CUDA training-window allocator peaks with `allow_peak_reset=True`.
It synchronizes the selected stream at the owned window boundaries, records the
previous counters, resets peaks and records completed end counters. The reset
is irreversible in that worker. These peaks cover the process/device allocator
during the window, not activation-only memory or all physical VRAM. Other hidden
streams and arbitrary concurrent activity are outside validated attribution.

CPU RSS remains a sampled process observation. An exact CPU per-stage/window
RSS peak is not supplied. `diagnostics()["peaks"]` separates unsupported peaks
from actual CUDA window records; it never promotes a sampled maximum to an exact
training peak.

The isolated API takes a trusted importable construction recipe, not a live
Trainer. The factory returns `trainer`, `train_loader` and `val_loader` (and an
optional supported stream). It performs one fresh recipe fit; choose this mode
only for a run you authorize. Current factory admission requires DataLoader
`num_workers=0`. Check `outcome` before using `measurement`; a failed or
unsupported execution does not provide completed measurements or trigger retry.
Memory and CUDA allocator quotas are unset by default. Deadlines, evidence
bounds and owned-worker cleanup remain independent of optional quotas.

## Optional activation evidence

CPU native eager saved-payload tracing is available in the isolated recipe API:

```python
from pathlib import Path
from hyso.performance import (
    TrainingMemoryRecipe, IsolatedTrainingMemoryProtocol,
    measure_training_memory_isolated,
)

# my_training.build is your importable construction-only factory.
recipe = TrainingMemoryRecipe(
    "my_training:build", device="cpu", factory_search_path=Path.cwd(), epochs=1
)
protocol = IsolatedTrainingMemoryProtocol(saved_activation_lifetime=True)
if __name__ == "__main__":
    result = measure_training_memory_isolated(recipe, protocol=protocol)
    data = result.to_dict()
    if data["outcome"] == "success":
        print(data["measurement"]["activation"]["summary"])
```

Pack/unpack/native-release events belong to the same CPU microbatch ledger.
Saved parameters and buffers can appear in those slots; the nonparameter
storage summary excludes registered parameter/buffer storage. Even that summary
does not prove exclusive ownership by activations. Payload release is a host
autograd event, not evidence that all aliases or physical storage were freed.
Checkpointing, custom autograd, retained/reused graphs and nested hooks are
outside validated saved-payload coverage. Collection is opt-in and adds overhead.

CUDA saved-payload tracing is unavailable and is rejected before CUDA setup.
Keep it disabled; use the hook-free CUDA component/allocator observations for
their documented scopes. They do not isolate CUDA activation bytes. CPU
activation evidence belongs to its CPU run and must not be attached to a
different CUDA run or added to that run's allocator peak.

## Interpret changes without inventing attribution

A measured gradient increase after backward and a later optimizer-state increase
identify those logical state changes. A larger allocator or RSS observation can
also include caching, workspaces, library/runtime allocations and instrumentation;
the remainder is not automatically activation memory or a leak. The report
supports investigating growth, not a guaranteed optimization or memory saving.

Owned CUDA windows also retain native allocation-retry and OOM-counter deltas.
Read them alongside the logical gradient/optimizer timeline and allocator
baseline. They do not identify which component caused an OOM. In the isolated
API, `oom`, an explicit `resource_limit`, `timeout` and `native_crash` are distinct
outcomes; an abnormal exit is not relabeled as OOM. Failed runs have no completed
measurement and are not automatically retried.

Both result types preserve their existing JSON/CSV exports and validate source
association, boundaries and evidence integrity. A raw schema `partial` status
describes that artifact's coverage: an execution succeeding does not make every
domain measured. Read each field's scope/status, not only the top-level outcome.
