# P30 acceptance — 2026-10-30 (started 2026-10-29)

**P30 remains PARTIAL.** Existing local recorded-video replay, real CPU fixture
inference, both application queue policies and lifecycle/export validation PASS.
Live camera, sensor/display measurements and a suitable trained real-world model
were not validated in this review.

The [final audit](../../outputs/p30-final-audit-20261029/final-report.md) records
the original roadmap criteria, source/model SHA256, separate protocols, complete
225-frame decode, raw journals, per-frame lifecycle/conservation CSVs, regression
receipts and preservation checks. The untrained three-channel model performs
real forwards; its results establish technical measurement behavior only.

The finalized recorded-video run processed all 225 frames with `block`. With
`drop_newest`, it processed 62 and dropped 163 at the application FIFO. Both runs
have zero pending frames. Metadata 25 FPS, observed read completion rates,
processing FPS and inference active-time rates are separate measurements.

See the [streaming guide](../guides/performance-streaming.md) for clocks, metric
definitions, finite experiment plans, removed arbitrary limits and compatibility.
Historical camera evidence does not substitute for a new camera acceptance run.
B4 and the CPU65536 native issue remain OPEN; no new CUDA or camera workload was
started and no earlier authorization was reused.
