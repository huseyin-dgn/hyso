# P31 final run plan for the integrated source — execution prohibited in this review

P31 remains PARTIAL. Preparation, preflight, CPU fault injection and two-second
GPU evidence do not satisfy the 7200-second acceptance. No shorter runs may be
joined, resumed or relabeled as the required uninterrupted run.

## Pinned original workload and protocol

Use `tools/p31/prepare_manifest.py` with the unchanged historical native P31
reference and a fresh directory outside Git under `hyso-private-evidence`.
The v2 preparation manifest binds base `1ab52822c39c073eebca1d8c1d773e56c47cb728`,
the complete uncommitted source overlay, Git status, interpreter/packages,
launcher, factory and reference hashes. No commit or old-manifest rewrite is
required. Any source/status change after pinning fails preflight.

| Item | Fixed final-run contract |
| --- | --- |
| Factory | `tests/runtime_support.py:make_gpu`, imported as `runtime_support:make_gpu` |
| Model | Conv2d(3,4,3,padding=1), ReLU, adaptive average pooling, flatten, Linear(4,2) |
| Reference model SHA256 | `5031500c33a8a2629c3fad9e52597ca0dc7597edff6b1a4c17e64e62861ccdae` |
| Input | resident ones, `[2,3,16,16]`, FP32, batch 2 |
| Seed / backend | 0; PyTorch eager eval/inference; completed CUDA steps synchronized and validated |
| Physical device | `GPU-9eec2f80-4c4e-a5cd-7091-03f0d013cf11`, RTX 3060 Laptop GPU; UUID-only visibility |
| Runtime | existing Python 3.10.18 / PyTorch 2.5.1+cu121; CUDA build 12.1; unchanged driver |
| Actual requested work | **7200 seconds**, idle 0; startup and cleanup excluded |
| Health cadence | 5 seconds, retain actual query/timeline brackets and missing values |
| Journal | 12-row chunks; maximum 2000 samples and 128 MiB; atomic hash-chained publication |
| Startup / heartbeat | 120 seconds / 60 seconds |
| Ready-to-exit watchdog | 7265 seconds; cleanup waits remain separately bounded |
| Outer observer watchdog | 7440 seconds including startup/finalization allowance; not work duration |
| Memory / process / allocator quotas | all unset; no occupancy/free-memory admission gate |
| Training / saved activation / peak reset | none; CUDA saved-activation remains disabled behind the OPEN B4 gate |
| Retry / execution resume | none; one-shot STARTED marker, fresh output, never overwrite/reuse |

The integrated two-second rehearsal matched the reference model hash, actual
input/precision, default-unset allocator policy, successful close/terminal,
controller exit and zero active Job processes. It verifies compatibility only.
The sampled span and exact work span remain separate quantities.

## Future explicitly authorized start

The preparation review may invoke only `prepare_manifest.py` and
`launch_p31.py --preflight --evidence-root <fresh-directory>`. Preflight imports
no factory, performs no CUDA queries and starts no owned worker. Its receipt
must state factory_calls=0, cuda_queries=0, owned_workers_started=0 and
long_run_started=false. The directory must have neither `run` nor STARTED.json.

**Only after separate user authorization**, run the source-pinned
`tools/p31/start_approved.py --evidence-root <fresh-directory> --start-two-hours`
with the pinned interpreter. That explicit flag is mandatory. The outer
observer captures launcher stdout/stderr, actual launcher exit code, watchdog
outcome and its owned Job cleanup in an atomic, non-overwriting
`parent-exit.json`. The inner launcher retains its revised interruption,
secondary-error, export and finalization receipts. An interruption can prevent
inner finalization; keep the journal prefix and outer receipt and leave P31
PARTIAL. No automatic retry, background continuation or inferred completion.

## Post-run acceptance, never inferred from configuration or forced exit zero

1. Reopen the actual journal and verify all source/model/input/device/runtime
   associations, chunk order/hashes, terminal manifest and lack of temporary tails.
2. Require completed analysis and terminal, `closed=true`, no failure, and
   `(work_end_ns-work_start_ns)/1e9 >= 7200`. Controller wall time, STARTED metadata,
   configured duration and `TerminateJobObject`'s forced zero code are insufficient.
3. Require controller reason=null, normal exit 0, inner active_processes=0,
   launcher status `RUN_FINISHED_PENDING_REVIEW`, and a successful outer
   `PARENT_EXIT_OBSERVED` receipt with exit 0, reason=null, no cleanup errors and
   active_processes=0. Verify actual process identity/creation time if checking PIDs.
4. Inspect exact work start/end and verified first/last sample span separately.
   For a gap-free observation claim require no existing-rule gaps (`> 2*5 s`).
   Expose initial/final observation offsets, suspension, missing sensors and gaps;
   no interpolation, rounding a shorter span to 7200, or restart-based repair.
5. Retain completed/attempted/failed steps, throughput/mean-step timeline and
   dispersion, native RSS/CPU, allocator allocated/reserved, optional physical GPU
   temperature/utilization and missing-query counts. p95/p99 and unsupported
   attribution stay null. RSS/cache growth alone is not a leak or physical VRAM.
6. Reopen JSON and check the existing flattened P31 CSV columns/rows. P31 CSV is
   not a P34/P35 envelope and is not accepted by RuntimeResult.from_csv.
7. Review failures/interruption/reporting truthfully: journal reopening is evidence
   recovery only. Assess the original six P31 acceptance requirements from this
   one run before changing feature status. Successful process completion is still
   pending review, not an automatic P31 IMPLEMENTED decision.

The historical 4248.1582207-second sampled run remains incomplete; its missing
terminal/close/parent exit cannot be repaired retrospectively. The historical
10.13% throughput decline remains unexplained. No new drop/temperature/memory
threshold, causal diagnosis or device-setting change is introduced. B4's native
0xC0000409 root cause remains unresolved; inference rehearsals do not close it.
