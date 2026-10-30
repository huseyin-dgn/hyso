# P32 memory growth analysis

`analyze_memory_trend` consumes a validated P31 `RuntimeResult`. It analyzes
already-recorded data; it does not start inference, query sensors, collect GC,
reset allocator peaks or modify model/RNG state.

```python
from hyso.performance import RuntimeResult, analyze_memory_trend

source = RuntimeResult.from_json("existing-p31-result.json")
trend = analyze_memory_trend(source, warmup_seconds=0.5)
data = trend.to_dict()["analysis"]
trend.to_json("new-p32-result.json")
trend.to_csv("new-p32-result.csv")
assert RuntimeResult.from_csv("new-p32-result.csv").to_dict() == trend.to_dict()
```

New results carry `evidence.trend_version=2`; legacy P32 JSON analysis and
flattened CSV export remain unchanged. New CSV uses the existing canonical
runtime envelope, preserving nulls, empty lists, raw source evidence and types.
Output paths are non-overwriting by default. The outer runtime format stays v1.

The `rss`, `private_commit`, `allocated` and `reserved` fields are independent
series. Each uses its own native query midpoint/timestamp and bracket. Allocator
samples never borrow RSS timestamps. Old allocator records without native query
brackets use explicitly labelled journal timestamp association. Missing RSS does
not discard recorded allocator, commit or weakref evidence. Optional clock, unit,
PID, device and scope declarations must agree with the owned source domain.

| Field | Domain | Not interchangeable with |
| --- | --- | --- |
| `rss` | Current worker process resident set | Model-only footprint, private commit or physical GPU memory |
| `private_commit` | Current worker process private committed bytes | RSS or total Job commit |
| `allocated` | Owned process/device PyTorch allocator live allocations | All driver/device allocations or physical VRAM |
| `reserved` | Owned process/device allocator reserved memory | Live tensor bytes or proven leakage |
| Retention logical bytes | Explicit watched Tensor object sizes; views may alias | Unique physical storage, owner/referrer census or process RAM |

The P31 exact work start/end define the fit window. A sample's whole query
bracket must lie within it; the post-warmup fit additionally requires its query
start at or after the cutoff. Start/end boundary snapshots are retained under
`boundary_observations` and do not leak into the fit. `full_work_window` reports
the contained pre-filter series; `warmup` compares recorded early and selected
observations. An initial zero-counter weakref snapshot may explicitly associate
with the start boundary when no warmup is requested, so genuine release evidence
is preserved. If the source has no terminal, the labelled committed observation
prefix is analyzed; its warmup origin is not presented as an exact full-run start.

Each metric includes sampled baseline/end, delta, observed sampled peak/minimum,
adjacent increases/decreases, number and span of observations, native timeline,
missing records/reasons, excluded samples and actual sampling gaps. A sampled
maximum is not an absolute or allocator-reset peak. P31 counter series do not
establish an owned peak reset domain: allocator peak values stay null with a
reason. P32 performs no reset. P09's separate single-forward peak probe, P10's
lifetime process HWM, P11's logical model bytes and P23 training boundaries retain
their distinct contracts.

The existing centered OLS byte/second fit is retained. Timestamps/byte values are
validated even when fewer than three points remain. Duplicate/reordered native
timestamps are rejected. No missing sample is filled or extrapolated; a fit may
span explicitly recorded gaps. Its conditional iid OLS standard error is not a
confidence interval or leak bound for correlated native observations. Before/
after warmup changes are descriptive, not causal allocation attribution.

`plateau_range_bytes` and `slope_tolerance_bytes_per_second` are declared analysis
thresholds, not memory quotas. Stable observed values can be a sampled plateau;
an early increase followed by stable selected samples can be a warmup/plateau
pattern. Flat allocated with rising reserved is labelled allocator cache growth,
never a leak. A private-commit change cannot be hidden by a flat RSS series.

Weakrefs never own watched tensors. New analysis validates unique/stable labels
and object IDs, monotonically ordered snapshots, non-disappearing registrations
and no resurrection of dead weakrefs. Object releases are observed identity
transitions; allocator or OS release remains unknown. Without sufficient snapshots,
live-reference growth is null with a reason, not a fabricated zero/false measure.
These witnesses do not locate owning referrers or prove a causal leak.

`leak_proven` stays false. The API reports supported observations and unknowns;
it does not prove long-term leak absence, ownership causality or future stability.
A few seconds of trained-model data qualify only the recorded short window.
CPU-only sources report CUDA metrics as unsupported/null with a not-requested
reason. A successful analysis never changes P31's acceptance status.
