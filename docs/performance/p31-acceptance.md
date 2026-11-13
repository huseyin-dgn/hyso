# P31 acceptance: long-running stability

2026-10-27 integration update: **P31 remains PARTIAL.** A subsequent historical
attempt has a verified sampled span of 4248.1582207 seconds and lacks its
successful terminal/close, exact work-end, launcher final reports and parent exit
receipt. It does not satisfy 7200 seconds. The earlier short-run review below is
preserved as historical evidence; its largest-run statement predates that attempt.
Use the [integrated final plan](final-integration-plan.md) and source-pinned
launcher for any separately authorized future run. No 7200-second run was
started during integration preparation.

**P31 = PARTIAL.** The harness and bounded reporting work, but a real long-duration
acceptance execution has not been recorded. The original P31 row at
`a0672d5:docs/roadmap/hyso-performance-backlog-v1.1.md` explicitly forbids using
a short smoke as long-run evidence. No long workload was started in this review.

Remote HEAD was verified as
`c386d5a95d0239222ebba4b137f8f7ebac085dd4`, including the requested P23 commit.
Review and protocol verification used a new isolated worktree.

## Acceptance matrix before any further execution

| Original requirement | Result | Evidence and missing scope |
| --- | --- | --- |
| 1. Real long-duration operation | MISSING | All located P31 native recordings are short. The largest observed work window is 2.0313133 seconds. No hours-long execution is present. |
| 2. Uninterrupted observed duration | MISSING for long-run acceptance | The current smoke has a 2.0060935-second sample span and an explicitly detected 0.5114869-second gap at a requested 0.2-second cadence. Controller wall time is 11.109 seconds, including startup/cleanup; it is not observed work time. |
| 3. Throughput and latency time series | PASS, bounded scope | Native completion/step counters produce interval throughput and mean successful-step time. `test_exact_drift_dispersion_thermal_association_without_cause` verifies arithmetic. Step timing includes workload synchronization/validation; p95/p99 remain null, not inferred. No long-run trend has been demonstrated. |
| 4. Failure, interruption and recovery records | PASS for reporting and evidence recovery | Existing owned-worker exception/hang/crash/close/interruption tests are retained. Classification and committed-prefix tests pass; a copied real native journal without its terminal remains aborted. Reopening recovers verified evidence only: execution resume/retry is unsupported, and separate runs cannot be joined into an uninterrupted run. |
| 5. Health, memory and thermal scope | PASS, short observation scope | The current smoke has seven RSS/CPU/allocator/thermal samples. Diagnostics distinguish process RSS, allocator allocated/reserved and physical-device sensors, expose missing observations and gaps, and avoid causal thermal/leak claims. Sustained health/thermal behavior remains unverified. |
| 6. JSON/CSV compatibility and safe cleanup | PASS, bounded scope | Existing JSON and the actual journal reopen through current validators; regenerated P31 CSV is byte-identical. Legacy schema/CSV tests pass. Native controller receipts show zero active owned processes after cleanup. No long-run cleanup acceptance has yet occurred. |

The outstanding mandatory evidence is **one real sustained run and verified
coverage of that same run's long window**. It does not require a new profiler,
automatic recovery engine, hardware occupancy gate or a monitoring redesign.

## Verification performed in this review

- **22 P31 CPU/protocol tests passed** in `test_stability_performance.py`.
  Five native-dependent cases were deselected. Synthetic arithmetic tests are
  protocol evidence, not elapsed native runtime.
- Existing native P31 JSON records were reopened, including the current CUDA
  smoke and archived CPU/GPU/gap cases. Their observed durations were checked;
  none supplies the missing long run.
- The current smoke records 849 successful/attempted steps and 1698 completed
  images, not hours of operation. Its journal was reopened from committed
  chunks and matched the saved result. Re-exported JSON reopens unchanged;
  the legacy 799-row CSV remains byte-identical.
- Removing the terminal from an in-memory copy leaves a committed-prefix,
  aborted result; changing a committed sample is rejected. This is evidence
  recovery validation, not a newly executed interruption/recovery scenario.
- All 63 protected existing files retained their hashes. No GPU workload,
  new owned worker or long-duration test was launched. Production sources and
  existing tests were unchanged.

An artifact's `analysis.status == "completed"` or diagnostic
`core_status == "complete"` means its requested short run completed. It does
not complete P31 acceptance and does not imply gap-free sampling.

See the [explicit-start long-run plan](p31-long-run-plan.md). P31 stays PARTIAL
until that execution's actual evidence is reviewed; preparing the plan does not
authorize or perform the run.
