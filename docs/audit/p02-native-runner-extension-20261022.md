# P02 native runner extension — offline preparation

2026-10-22, Europe/Istanbul. **Runner READY; native P02 NOT EXECUTED.**
CUDA/Torch imports0, native workloads0, worker launches0, retries0.
Exactly13 targeted stdlib/protocol tests executed;13 PASS,0 failure/error.
No full pytest suite or previous matrix was rerun.

## Minimal integration

The existing `tools/forensics/gpu_scenario_runner.py` now supports a second
explicit scenario: `conv2d_fp32_p02_completion`. No independent runner was added.
`gpu_admission_policy.py` and all production API/source files are unchanged.

The admitted native worker will construct an eval
`torch.nn.Conv2d(1,3,3,padding=1)` and one resident contiguous FP32 Tensor of
shape `(2,1,7,9)` on CUDA:0 after `torch.manual_seed(71)`. The input source is
a Tensor, not a DataLoader, factory, callable or generator. The public API
owns its inference-mode context and defensive clone.

It calls the existing public API once:

```python
measure_throughput(model, input_source, protocol=ThroughputProtocol(
    device="cuda:0", precision="fp32", warmup_iterations=1,
    batch_count=5, duration_seconds=None, repeats=1,
    queue_depth=2, concurrency=1))
```

This is a short finite batch-count completion window. Its expected successful
ledger is5 completed batch-forward calls,10 completed images,0 pending batches,
3 completion groups and1 terminal drain. These are configured expectations,
not measurements from this offline task. One batch remains for the final tail
after two full queue groups. Backpressure syncs may range0–2 depending on actual
completion; no fabricated count is required.

The worker revalidates the public `ThroughputResult` and retains its entire
schema under the forensic `throughput` field. No fake BenchmarkResult is
constructed for P02. Raw start/admission-end/end timestamps, actual wall seconds,
completed images/s, completed/pending/group/terminal/drain counters, warmup,
unit/status/reason and provenance remain in the public result. The parent also
checks that the transport matches the fixed sealed P02 contract, including
count conservation, wall-clock-derived rate and default stream_id0.

Measured scope is `resident_forward_completion_wall`, clock `perf_counter_ns`,
synchronization `bounded_group_completion_and_terminal_event`. Preflight,
warmup, clone/transfer, state checks, hashing and export are excluded; finite
checks, queue bookkeeping, backpressure and terminal tail are included. No
CUDA event latency or derived serial service rate supplies the P02 value.

## Scenario sealing and safety

`--scenario` selects a scenario only with `--prepare`; execution uses the sealed
manifest. The old scenario remains the default. P02 has an exact sealed
`p02_protocol` dictionary, including JSON type-sensitive comparison. Unsupported
scenarios and modified counts/queue/warmup are rejected. Explicit approval now
binds to the selected manifest scenario, and the controller validates that the
returned result belongs to that scenario.

Source/runtime verification, Job assignment/ownership/exact membership/readback,
60s worker/90s outer deadlines,4GiB per process,8GiB/4-process child Job,
12GiB/8-process outer Job, allocator fraction0.15, kill-on-close, bounded logs/
results, exclusive execution marker and retry0 remain. The P02 path creates no
P23 activation hooks. GPU sharing/unknown PID/utilization/free VRAM retain the
unchanged warning-only policy; no user process is touched.

## Offline evidence and compatibility

Command:

```text
python -I -S -B -m unittest discover -s tests/forensics -p test_p02_runner_extension.py -v
```

**13/13 PASS**, once. Tests cover scenario dispatch; public signature/arguments
and Tensor-construction code; production result serialization; completion,
pending and terminal conservation; corrupt ledger/rate/unit/status/scope/result
rejection; missing execution receipt; source/runtime/manifest mismatch;
isolation invariants; previous scenario compatibility; missing approval refusal;
warning-only policy; and sealed P02 protocol/unsupported scenario rejection.

Tests load the real production result module and protocol dataclass through
isolated stdlib package stubs, bypassing Torch-bearing package initialization.
Only the measure_throughput call is a spy with the real public signature. No
real model, CUDA import/init, sensor query, Popen or Job worker was executed.
Synthetic ledger values verify protocol handling, not hardware performance.

Compatibility proof compares the pre-edit runner snapshot with the new source.
The Job, gate, supervisor, timeout, owned cleanup, internal bootstrap, identity,
bounded reader/publisher and OOM wrapper functions are AST-identical. Removing
only the new P02 dispatch branch yields the exact previous native worker AST.
The previous P01/P09/P16 benchmark path and protocol are otherwise unchanged.
Old sealed candidates and execution markers are not rewritten or reused.

## Files and protection

Only these repository files change:

- `tools/forensics/gpu_scenario_runner.py` — required scenario integration.
- `tests/forensics/test_p02_runner_extension.py` —13 targeted offline tests.
- This new audit.

Branch `feature/hyso-models-next`; HEAD
`a1864094cf41d242d75f634dd7fc134436fae6d6`; index empty. All263 tracked files,
including the original13 dirty files and current quality matrix, retain bytes
and status. Among26 pre-existing dirty/untracked entries, only the authorized
current runner content changes; its old version is retained in evidence.
All other prior forensic files, audits, historical evidence and execution
markers remain hash/status/mode-identical. No commit/push/reset/clean/stash/
checkout, dependency or driver change occurred.

Evidence directory:
`C:/Users/hdgn5/AppData/Local/Temp/hyso-p02-runner-extension-20261022`.
It contains `start-protection.json`, `gpu_scenario_runner.before.py`,
`runner-extension.diff`, `targeted-tests.log`, `compatibility-proof.json`,
`final-protection.json`, `evidence-manifest.json` and the fresh `p02-candidate`.
The new candidate seals51 source and26 verified runtime files. No native
approval or execution marker was created. Its run ID/manifest hash are retained
in `p02-candidate/seal.json` and the final protection record.

Preparation syntax for a separate new candidate:

```text
python -I -S -B tools/forensics/gpu_scenario_runner.py --prepare NEW_DIRECTORY --r3 R3_DIRECTORY --scenario conv2d_fp32_p02_completion
```

## Decision

**READY for a separately authorized native P02 execution.** This is runner and
offline protocol readiness only. No completed count, wall-clock time or real
images/s was measured here; P02 native acceptance remains PARTIAL in the
existing17-feature quality matrix. B4 remains OPEN/UNRESOLVED. Work stops here;
no CUDA execution, P03/P04 or training follows automatically.
