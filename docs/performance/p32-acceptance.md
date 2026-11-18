# P32 acceptance — 2026-10-30

**P32 PASS within the verified CPU/offline descriptive-analysis scope.** The
existing trained Model B P31 recording supplies real CPU RSS/private commit data;
archived native CPU weakref retain/release recordings and independent arithmetic
regressions cover lifetime and trend behavior. No new model inference, GPU
workload, long run or final integration test is needed for this scoped acceptance.

See the [P32 audit](../../outputs/p32-memory-growth-20261030/final-report.md) for
the original criteria, recorded Model B checkpoint/input identity, independent
slopes/uncertainty, exact data windows, regression receipts and preservation.
The [memory-trend guide](../guides/performance-memory-trend.md) defines v2 fields
and legacy compatibility.

PASS qualifies the analysis behavior and observed short CPU data. It does not
certify absence of leaks in Model B, a causal leak detector, new native CUDA
measurements, absolute allocator peaks or long-term memory stability. The original
P32 row permits OS RSS and/or CUDA tooling and does not impose a 7200-second run.
Such long/risky verification belongs to the deferred Final Integration phase.

P30 and P31 remain PARTIAL. The P31 manifest with SHA256
`f72f2df3043e7c854fa5b7ed981f7bba2a86c152cbe9f605c33261d76bec6d22` is preserved
as a prepared historical plan, not execution approval. Its old source snapshot
must be reviewed and repinned in a new manifest before any future approved run;
the preserved manifest is not rewritten. B4 and CPU65536 remain OPEN.
