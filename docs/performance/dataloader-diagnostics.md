# DataLoader wait and compute diagnostics (P21)

Use `measure_training_with_loader` around the training run you already intend
to execute. It observes your existing PyTorch DataLoader; it does not replay
training, replace the sampler or require an external profiler. Worker tracing
is optional. No benchmark, RAM/VRAM quota or worker-count policy is imposed on
ordinary library use.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from hyso.training import Trainer
from hyso.performance import measure_training_with_loader

model = torch.nn.Linear(3, 2)
trainer = Trainer(
    model,
    optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
    loss_fn=torch.nn.CrossEntropyLoss(),
    device="cpu",
)
loader = DataLoader(
    TensorDataset(torch.ones(5, 3), torch.tensor([0, 1, 0, 1, 0])),
    batch_size=2,
)
measurement = measure_training_with_loader(trainer, loader, epochs=1)
report = measurement.loader.diagnostics()  # Offline: no loader/GPU operation.
for phase in report["findings"]:
    print(phase["samples_per_second"], "samples/s")
    print(phase["parent_delivered_next_wall_ms"], "ms in delivered next() calls")
    print(phase["input_preparation_host_wall_ms"], "ms parent input preparation wall")
    print(phase["cpu_forward_backward_wall_ms"], "ms CPU forward/backward wall")
    print(phase["relation"])
```

The achieved samples/s denominator is the observed training/validation phase
wall time, including consumer work, startup and exhaustion. It is not standalone
loader capacity. `core.configurations` records worker count, prefetch factor,
pin-memory request, persistence and sampler configuration. These are settings,
not evidence of effective pinning or efficient prefetching.

`batch_timeline` contains parent-local `next()` start/delivery timestamps, batch
size, epoch and microbatch identity. It links existing P20 input preparation,
forward, loss/metrics and backward windows. CPU forward/backward wall time is
reported separately from input preparation. Input preparation may include dtype
conversion or copy calls; it is not an isolated H2D completion measurement.
For CUDA training, host wall windows
and selected-stream CUDA events remain separate fields; the CPU comparison is
null. Neither host launch time nor worker wall time is GPU compute time. Missing
P20 evidence (`stages=False`) gives null compute windows, not zeros.

If `relation` is
`observed_batch_acquisition_wait_exceeds_CPU_forward_backward_wall`, batch
acquisition consumed more measured wall time than CPU forward/backward in that
phase. Inspect the first batch and iterator startup separately. This observation
does not identify disk, workers, serialization or scheduling as the cause.
Consumer pauses and compute between calls also affect delivery gaps.

## Optional worker stages

Annotate actual operations where their boundaries are known:

```python
from pathlib import Path
from hyso.performance import trace_data_stage

def read_sample(path, index, decode, transform):
    identity = str(index)  # Same identity as the enclosing dataset[index].
    raw = trace_data_stage("load", Path.read_bytes, Path(path), sample_id=identity)
    image = trace_data_stage("decode", decode, raw, sample_id=identity)
    return trace_data_stage("transform", transform, image, sample_id=identity)
```

Each operation executes exactly once and keeps its return value, exception and
RNG behavior. Without an active trace this is a passthrough. An explicitly
configured `TracedDataset` recorder/channel, or the existing opt-in
`WorkerTaskDataLoader`, activates collection. Choose a bounded diagnostic run;
worker event collection and IPC add overhead. `load` measures only the annotated
operation, including any OS-cache effects, not guaranteed physical disk I/O.
Opaque datasets are not automatically split into load/decode/transform.

Pass an already collected companion from the **same** measurement:

```python
# task_evidence is a validated WorkerTaskAnalysisResult from the existing
# analyze_worker_tasks lifecycle; training is not run again here.
report = measurement.loader.diagnostics(worker_tasks=task_evidence)
```

The task adapter remains explicitly gated to its verified PyTorch 2.5.1 source
and supported map-style, spawn, automatic-batching, pin-memory-disabled scope.
It rejects unsupported scopes; an ordinary DataLoader remains usable without
that adapter. The caller retains worker ownership and performs bounded cleanup
before collecting final task evidence. See the native acceptance script
[`p21_diagnostics_acceptance.py`](../../tests/p21_diagnostics_acceptance.py)
for the complete guarded Windows lifecycle. Its Job limits are test safeguards,
not library resource requirements.

Alternatively, `diagnostics(worker_traces=existing_v1_companion)` accepts the
older sample trace. It provides aggregate worker stages with execution/epoch/
batch association explicitly unknown; it cannot populate per-batch task links.
Do not supply both companions. A foreign parent hash is rejected. Incomplete
task evidence yields null service/stage means and no per-batch worker links.

## Which waits are actually measured?

| Field | Measured scope | Interpretation limit |
| --- | --- | --- |
| `parent_next` | Main-process `next()` call through batch delivery | Includes blocking, deserialization and overhead; not worker compute or physical queue residence |
| `worker.stages` | Worker-local load/decode/transform/fetch, and task collate | Wall intervals, not CPU busy time; fetch includes its nested stages |
| `worker_task.worker_service` | Worker fetch entry through collate completion | Includes trace overhead; excludes unobserved queue and serialization work |
| `parent_buffer` | Observed parent receive through ordered delivery | Parent-local buffer residence; excludes time before the receive observation |
| `parent_index_queue_put` | Actual parent index-queue `put()` call | Excludes feeder serialization; not end-to-end task queue wait |
| `task_lifecycle` | Submitted/received/delivered receipt counts and buffer hits | Logical conservation, not physical queue occupancy or prefetch efficiency |
| `physical_queue_residence`, `queue_occupancy`, `prefetch_efficiency` | Unsupported, null | Never inferred from `next()` or buffer-hit counts |

Worker and parent timestamps are never subtracted to produce a duration. Task
links use verified execution, epoch, batch, PID and token identity, not timestamp
proximity. Do not add nested stage durations or parallel-worker durations to
phase wall time.

Use stage measurements to decide what to investigate. A long annotated
transform merits inspection of that transform. A long `next()` without complete
worker evidence supports only a batch-acquisition wait observation. This report
does not prescribe a worker count, prefetch setting or speedup; those require
controlled comparisons in the user's workload.

The existing loader v1 JSON/CSV and worker v1/v2 JSON exports are unchanged.
`diagnostics()` returns a detached, JSON-serializable derived dictionary. The
legacy loader artifact deliberately retains its `partial` status: worker
coverage is represented by its separate companion, not retroactively inserted
into old raw schemas.
