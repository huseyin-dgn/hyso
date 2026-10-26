# P31 bounded long-run acceptance plan — not started

**Execution status: NOT RUN. P31 remains PARTIAL.** This plan requires a separate
explicit user start. It is not a background task, CI test or automatic retry.

## One fixed workload and one bounded run

Proposed acceptance window: **7200 seconds (two hours)** of actual CUDA model
inference using the existing `runtime_support:make_gpu` factory. This is an
experiment duration chosen to fill the missing long-run evidence, not a
library-wide definition of stability. It qualifies only this workload, device,
runtime and observed duration.

The controlled model is the existing Conv2d(3,4) / ReLU / adaptive pooling /
Linear(4,2) eager inference helper, input shape `[2, 3, 16, 16]`, FP32, seed 0.
Each step synchronizes its CUDA completion, validates output and checks the
model/input guard. It is not a production-model capacity or training test.
Use the physical GPU UUID recorded in the prior P31 result; the worker exposes
only that UUID. Do not change the model, batch, seed or code during the run.
Record the chosen commit and factory-file hash before starting; the journal
also retains factory source, model descriptor and runtime identity.

| Protocol item | Proposed value / scope |
| --- | --- |
| Requested observed work | 7200 s; startup/cleanup are separate |
| Health sample interval | 5 s; actual timestamps and gaps retained |
| Sample bound | 2000, sufficient for this cadence/duration |
| Journal chunk | 12 rows, approximately one minute of samples |
| Startup / heartbeat deadline | 120 s / 60 s |
| Journal byte bound | Existing 128 MiB default; an evidence bound, not RAM/VRAM quota |
| Memory / process-count quotas | Unset; no free-memory or occupancy gate |
| CUDA allocator fraction | Unset; no peak reset, clock/power-limit or driver change |
| Ownership | Existing Windows kill-on-close Job and bounded cleanup |
| Retry / execution resume | None; interruption remains a separate incomplete run |

Monitoring occurs only inside the explicitly requested stability run. CPU/RSS,
allocator and optional GPU/thermal queries have overhead. Throughput includes
that overhead; step time includes the workload's synchronization/validation but
excludes between-step health/journal work. Thermal/utilization readings describe
the physical device, including other applications, not model-exclusive activity.

## Explicit start procedure

Run from the selected isolated repository root using the established compatible
Python environment. Use a **new private output directory outside Git**. The
following is a plan example, not an installed command or an executed script.
Save it outside the repository if the user chooses to start it. Supplying the
start flag is required; importing the example does not launch anything.

```python
import argparse
import json
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-two-hours", action="store_true")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if not args.start_two_hours:
        parser.error("Explicit --start-two-hours is required; nothing started")

    repo = Path.cwd().resolve()
    if not (repo / "tests" / "runtime_support.py").is_file():
        parser.error("Run from the selected isolated repository root")
    sys.path.insert(0, str(repo))
    from hyso.performance import (
        WorkloadRecipe, StabilityProtocol, measure_stability, reopen_stability,
    )
    recipe = WorkloadRecipe(
        "runtime_support:make_gpu", {"gpu_uuid": args.gpu_uuid},
        search_path=str(repo / "tests"), gpu_uuid=args.gpu_uuid,
    )
    protocol = StabilityProtocol(
        duration_seconds=7200, interval_seconds=5, idle_seconds=0,
        startup_timeout_seconds=120, step_timeout_seconds=60,
        max_samples=2000, chunk_rows=12, seed=0,
    )
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=False)
    try:
        result = measure_stability(recipe, root / "journal", protocol=protocol)
    except KeyboardInterrupt:
        if (root / "journal" / "controller.json").is_file():
            reopen_stability(root / "journal").to_json(root / "interrupted.json")
        raise
    result.to_json(root / "result.json")
    result.to_csv(root / "result.csv")
    (root / "diagnostics.json").write_text(
        json.dumps(result.stability_diagnostics(), indent=2) + "\n",
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
```

The user would invoke the saved example with `--start-two-hours`, the recorded
`--gpu-uuid`, and a fresh `--directory`. No step in this review invoked it.
If stopped with Ctrl+C, the existing controller records interruption and cleans
its owned process tree. Keep every committed chunk/controller/error record.
Do not restart into the same directory or combine shorter runs into two hours.

## Review after the execution, not before

1. Reopen the actual journal with `reopen_stability`. Check source/model/device
   identity, chunk hashes/order, terminal manifest and absence of temporary tails.
2. Require actual `analysis.duration_seconds >= 7200`, completed successful work,
   a successful terminal/close and controller exit, and owned active processes 0.
   Controller elapsed time or a configured 7200-second request is insufficient.
3. Inspect the work start/end and sampled span separately. For a gap-free
   observation claim, require no detected gaps under the existing `> 2 * interval`
   rule. A gap, system suspension or interruption must remain visible; never
   interpolate it or use a restart to erase it. Periodic sampling is not
   continuous sensor coverage at every instant.
4. Retain throughput and mean-step time series, step dispersion, every error and
   interruption, RSS, allocated/reserved, temperature/utilization observations
   and missing-query counts. Missing sensors remain unsupported/unavailable;
   do not claim thermal stability for unobserved periods. p95/p99 remain null.
5. Reopen exported JSON and verify existing CSV columns/rows. Report recovery
   truthfully: journal/evidence reopening only; execution recovery not attempted
   and not supported. Do not inject a failure into this uninterrupted run;
   bounded failure-protocol tests are separate evidence.

Only then assess the six requirements in the [acceptance matrix](p31-acceptance.md).
No fixed throughput-drop, memory-growth, utilization or temperature threshold is
introduced here. Report the observed trajectory and limitations; a correlation
does not establish thermal throttling, a leak or a unique failure cause. If the
duration/coverage requirement fails, leave P31 PARTIAL with that actual reason.
