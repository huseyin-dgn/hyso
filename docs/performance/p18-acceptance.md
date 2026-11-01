# P18 acceptance: scoped empirical Roofline

**IMPLEMENTED for scoped empirical-session analysis:** saved native FP32 compute/DRAM
evidence, matching work and memory hierarchy, empirical session references,
Roofline plotting and bounded interpretation. This does not certify absolute
GPU capacity, a causal kernel bottleneck or an achievable speedup.

The separate three-sample calibration protocol **FAILS** its predeclared 10%
compute-rate spread screen: **10.518334% > 10%**. Its qualified stable two-roof
acceptance remains **PARTIAL**, with a qualified compute roof still missing.
The valid native measurements and DRAM reference remain usable in their stated
scopes. The later five-sample compute session does not replace or repair that
experiment. No samples or failed records are discarded and no screen is relaxed.

| Acceptance scope | Status | Limit |
|---|---|---|
| Scoped empirical-session analysis | IMPLEMENTED | Separate observed session bands and conditional reference-model interpretation |
| Three-sample 10% calibration protocol | FAIL | Compute spread 10.518334%; predeclared screen not met |
| Qualified stable two-roof experiment | PARTIAL | Qualified compute roof missing; no stable two-roof acceptance claim |

These scopes must remain distinct. The generic reader has no library-wide 10%
admission rule; it reports observations without certifying reference qualification.
When a reference is absent or unqualified, omit it to retain the measured target
in a `partial` result. No two-roof model is derived until both references are
supplied. Adding this partial API does not complete the failed calibration.

## Measured target

| Quantity | Result |
|---|---:|
| GPU interval | 3936 ns |
| Observed DRAM read / write | 528256 / 4992 byte |
| FP32 FADD / FMUL / FFMA thread instructions | 0 / 0 / 65536 |
| Hardware-equivalent work, FMA=2 | 131072 FLOP |
| Analytic useful-add reference, kept separate | 65536 FLOP |
| Arithmetic intensity | 0.245799328 FLOP/byte |
| Achieved compute rate | 33.300813008 GFLOP/s |
| Observed interval DRAM rate | 135.479674797 GB/s |

The rates use work, traffic and GPU duration associated with the same recorded
kernel interval. Payload, parameter size, allocator bytes and PCIe bandwidth
were not substituted for DRAM counters.

## Empirical references and uncertainty

| Independent session | Samples | Observed range | Median | Range / median |
|---|---:|---|---:|---:|
| FP32 compute A | 3 | 5595.489551–6186.037458 GFLOP/s | 5614.462390 | 10.518334% |
| FP32 compute B | 5 | 7732.736881–9367.467619 GFLOP/s | 8593.455340 | 19.022974% |
| Device DRAM | 3 | 301.505804–309.192173 GB/s | 306.876992 | 2.504707% |

The DRAM reference maximum is **309.192173 GB/s within its recorded scope**.
Compute sessions retain their individual samples, conditions and observed bands;
they are not pooled into a device-wide maximum. Ranges are not confidence
intervals, capacity bounds or guaranteed sustainable rates. Earlier experimental
10% screening failures remain valid historical results; those screens were not
mandatory requirements in the original P18 roadmap (`a0672d5`).

At the target's arithmetic intensity, the DRAM term is smaller in both empirical
reference models. This is a **model branch**, not proof that the actual kernel
is memory-bound. Target clock/thermal history is incomplete, replay passes can
differ, shared DRAM may include other engines, and eventual output writeback is
not fully represented by the interval. Actual bottleneck, hardware headroom,
theoretical maximum and speedup remain **unsupported/null**.

## Verification and delivery

- Existing native reports, receipts and raw CSV counters were cross-checked;
  identity, NVTX, FP32 instruction counts, units and output/completion matched.
- The optional adapter reads saved CSV/receipts only. No new GPU capture or
  calibration was run to complete this integration. Ordinary HYSo use does not
  require Nsight Compute or a new resource quota/benchmark.
- The earlier full-reference verification recorded **45 CPU/protocol tests passed**
  across `tests/test_roofline.py` and
  `tests/test_roofline_counters.py`: existing behavior, native arithmetic,
  separate sessions, attribution/unit/source checks, missing evidence, tamper
  rejection, JSON/CSV roundtrips and CPU plotting.
- Partial-reference integration: **54 CPU/protocol tests passed** across the
  same two files, including all three missing-reference combinations, invalid
  supplied references, partial JSON/CSV roundtrips, derived-evidence tampering
  and explicit partial-plot rejection. A separate saved-native compatibility
  check passed against pre-existing schema-1 JSON/CSV exports: full-reference
  derived fields are unchanged, and partial exports retain the same target.
  These are **55 passing checks** in this integration run, with CUDA uninitialized;
  they do not constitute a new calibration or qualified stable-roof acceptance.
- Native evidence and generated plots/exports remain local. No raw profiler
  reports, machine identifiers, personal paths or private logs accompany this
  summary. No exact hardware-capacity or acceleration claim is published.

See [saved-counter API and plotting](roofline-counters.md) to produce local
JSON/CSV and PNG/SVG results with the observed session bands and explicit limits.
