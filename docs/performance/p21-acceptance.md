# P21 acceptance: DataLoader bottleneck analysis

**P21 = COMPLETED for the original roadmap scope.** The reference is P21 in
`a0672d5:docs/roadmap/hyso-performance-backlog-v1.1.md`: distinguish
load/decode/transform and batch acquisition wait from compute, report ms/batch,
samples/s and timelines, retain worker/RNG/epoch configuration, and separate
worker and main-process measurement scopes.

The final review started from remote feature-branch commit
`f504aa82a33f4f39c18e013689137d3bd8ca7db1` in a separate worktree. It reviewed
the diagnostics introduced by `290c9341` and the configuration/wait-scope
acceptance added by `f504aa82`. No mandatory production-code gap remained.

## Acceptance matrix

| Original requirement | Result | Concrete evidence |
| --- | --- | --- |
| 1. Separate load/decode/transform timing | PASS | Saved `task_images` acceptance reads actual PNG files, decodes with PIL and transforms each sample. All 16 samples have separate load/decode/transform events; six tasks have collate events. `test_explicit_load_group_and_failed_stage` validates stage groups; missing stages remain unsupported. |
| 2. Loader wait distinct from CPU/GPU compute | PASS | `test_parent_diagnostics_no_io_no_cuda_and_preserve_exports` checks parent wait against separately linked CPU forward/backward and input preparation. `test_no_stage_evidence_never_claims_compute_time` keeps missing compute null. Saved CUDA captures retain separate host and selected-stream event fields; their CPU-compute comparison is null. |
| 3. Per-batch wait and samples/s | PASS | `test_exact_iter_next_calls_and_fake_clock` checks exact iterator calls and durations. `test_core_metrics_preserve_legacy_exports_and_do_not_initialize_cuda` checks sample-weighted phase throughput. Four saved real-file scenarios each deliver sizes 2/3/3 over two epochs. |
| 4. Real batch/worker timeline | PASS | Saved `task_images` has six batch links with verified execution/epoch/batch identity, worker PID and task token. `test_task_link_scope_and_incomplete_or_foreign_evidence` rejects foreign association and suppresses incomplete task links. |
| 5. Worker and parent scopes separated | PASS | Worker service/stages use worker-local intervals; acquisition and observed receive-to-delivery residence use parent-local intervals. `test_task_clock_offsets_never_change_same_clock_durations` shifts workers independently without changing derived durations or findings. Legacy sample traces retain unknown epoch/batch association. |
| 6. RNG, epoch, sampler, pinning and prefetch behavior retained | PASS, within documented adapter scope | Paired native baseline/observed runs assert equal batch tensors, model/optimizer/history, RNG and loader-generator state. They retain component identities, the caller's variable batch sampler and persistent worker PIDs. Saved configurations cover workers 0/1/2, prefetch 1/2, persistence and requested pinning. Current CPU replay tests cover shuffle, validation, epochs, stage collection and resume. |
| 7. Native acceptance and JSON/CSV compatibility | PASS | Four saved real-file native runs have exit code 0, no timeout and complete worker cleanup. Archived worker v1/v2 artifacts validate. Saved CUDA records validate through the current loader schema. Current tests and independent exports preserve loader v1 JSON, every CSV observation/column and separate worker schemas. |

## What the measurements establish

Ordinary `DataLoader` observation times the parent's actual `next()` call. This
is batch acquisition wall time, including blocking, deserialization and
overhead. It is not worker CPU execution, physical queue residence, GPU idle
time or physical disk I/O. Phase samples/s includes consumer work, startup and
iterator exhaustion; it is not standalone loader capacity.

Optional stage annotations measure operations with known boundaries. The
real-file acceptance uses a declared 10 ms transform delay to check that stages
are distinguishable. File reads may hit the OS cache. These small functional
cases establish instrumentation behavior, not production disk speed, an
optimal worker count or an achievable speedup. Nested fetch/stage durations and
parallel worker intervals must not be summed into phase wall time.

`parent_buffer` measures only the observed parent receive-to-delivery interval.
Task conservation and index-queue `put()` timings do not establish physical
queue occupancy or prefetch efficiency. Those fields remain unsupported/null.
There is no need for additional IPC or queue instrumentation to satisfy the
original P21 acceptance rule.

## Configuration and compatibility boundaries

Normal DataLoader use needs no custom worker adapter, external profiler or
resource quota. Worker tracing is opt-in. The task adapter retains its explicit
PyTorch 2.5.1 source check and map-style/spawn/automatic-batching,
`pin_memory=False` restrictions; these do not constrain ordinary DataLoaders.

The CPU pinning acceptance distinguishes `pin_memory=True` configuration from
effective pinning: with CUDA hidden, the delivered tensors remain unpinned in
both baseline and observed runs. Saved CUDA acceptance also covers requested
pinning and default/nondefault selected streams. No universal pinning,
prefetch-efficiency or hidden-stream coverage is claimed.

The loader v1 artifact's `partial` status is deliberately preserved: worker
coverage lives in its separate v1/v2 companion. It does not mean that the P21
feature is incomplete when the required companion evidence is supplied.
Incomplete task evidence remains incomplete; older sample-only evidence does
not acquire invented per-batch associations. Earlier saved diagnostic dictionaries
reopen with their existing values preserved and additive derived fields.

## Final verification

- **87 targeted CPU/protocol tests passed**, with six native-dependent tests
  deselected because their saved native evidence was reviewed separately.
  Files: `test_performance_dataloader.py`, `test_dataloader_protocol.py` and
  `test_loader_diagnostics.py`.
- **Eight saved-evidence checks passed**: four real-file integrity checks, two
  loader-configuration checks, foreign/incomplete-task handling and independent
  worker-clock-offset invariance. These revalidated previous native executions;
  they did not execute the loaders again.
- Four archived worker companions reopened: persistent/nonpersistent task
  traces remain complete, saturation remains incomplete, and the older v1
  sample trace retains unknown execution/epoch/batch association. Fourteen
  historical worker lifecycle/failure supervisor receipts retain exit code 0
  and no timeout; controlled worker failures are not relabeled successes.
- **18 saved CUDA loader scenarios and two resume envelopes** validated with
  current schemas. Selected-stream CUDA event values remain distinct from host
  windows. No GPU experiment was run in this review.
- All four real-file loader results were exported to JSON/CSV and independently
  checked for JSON reopening and exact CSV columns and observation values.
  Production-source fingerprints match the prior accepted evidence; 32 protected
  existing files retained their hashes.

Only this user-facing acceptance documentation was needed for final closure.
Native reports, raw measurements, private audits and temporary files remain
outside the commit. No original mandatory requirement remains missing.

See [DataLoader wait and compute diagnostics](dataloader-diagnostics.md) for
usage, scope definitions and the optional stage/tracing lifecycle.
