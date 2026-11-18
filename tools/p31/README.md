# Integrated P31 launcher finalization

The integration launcher resolves its own repository root and pins base commit
`1ab5282` plus the complete uncommitted source overlay in a fresh v2 manifest.
It does not require or create a commit. Source additions/deletions/content, Git
status, interpreter, packages, launcher, reference and factory are checked again
before every start. Old private manifests and run directories are never reused.

`prepare_manifest.py --directory <fresh-private-directory> --reference <prior-native-P31.json>`
only prepares a fingerprinted manifest. `launch_p31.py --evidence-root <directory>
--preflight` runs CPU-only checks and does not import the factory or query CUDA.
Neither command authorizes or starts the two-hour run.

After separate explicit user authorization, the reviewed entry point is
`start_approved.py --evidence-root <directory> --start-two-hours`. Its external
observer retains launcher stdout/stderr and an atomic, non-overwriting
`parent-exit.json`, including actual launcher exit code, outer watchdog outcome
and zero-active-process Job cleanup. The 7440-second outer watchdog includes
startup/finalization allowance; it is not observed work duration. Interruption
can force the tree to exit before inner finalization; retain the committed
prefix and outer receipt and leave P31 PARTIAL. There is no retry/resume.

All preparation/testing in this integration is bounded. No invocation of either
two-hour start entry point is part of the readiness review. P31 remains PARTIAL;
B4's CUDA saved-activation crash gate remains OPEN and disabled.

## Historical standalone review (prior evidence, preserved below)

P31 remains PARTIAL. This directory contains a revised copy of the private launcher
and isolated fault-injection tests. The original evidence directory, original launcher,
manifest and HYSo repository sources remain unchanged. This revision has no manifest;
its explicit-start preflight will refuse to run. No launch-ready manifest was prepared.
A future run would require separately authorized preparation with fresh output paths
and fingerprints. Never replace the old launcher or reuse its run directory.

## Changes

- `_record_interruption` writes `launcher-interruption.json` with `export_pending`
  before journal reopening/export, retaining the controller receipt when readable.
- Reopen/export catches secondary BaseException values, records them and still attempts
  `launcher-status.json`. An independent `launcher-finalization-error.json` is attempted
  if status publication fails or the finalizer itself unexpectedly fails.
- The original triggering exception is re-raised through the tested secondary-error
  paths, including when all reporting writes fail. Receipts are serialized first,
  written to a same-directory temporary file, flushed/fsynced and atomically published
  with a non-overwriting hard link. Review found that the former direct exclusive write
  could leave partial JSON; this revision fixes that publication gap. These are
  best-effort durable receipts, not a guarantee
  against disk failure, forced termination or power loss. Status phases describe
  reporting progress, never successful training/inference acceptance.
- Normal result summarization/status publication is inside the guarded block.
- Existing result JSON/CSV methods and formats are unchanged. Interrupted/error states
  stay INTERRUPTED_OR_FAILED / PARTIAL_PENDING_REVIEW. A forced worker exit code zero
  is explicitly not completion evidence. No performance or cleanup algorithm changes.
- Repeated asynchronous interrupts can still arrive between protected Python blocks
  or interrupt controller cleanup before its receipt exists. Forced termination,
  disk failure, unsupported hard links and power loss remain limitations. Publication
  does not promise crash-durable directory metadata. A failed temp-file unlink may
  leave a `.p31-publish-*.tmp`; final JSON is never incrementally exposed.

## Tests (no GPU workload)

Run from this directory:

```powershell
& 'C:\Users\hdgn5\.conda\envs\torch_gpu\python.exe' -B -m unittest -v test_launcher_finalization
```

Tests use temporary directories and mock measurement, reopening, exports and process
ownership boundaries. They cover normal completion, KeyboardInterrupt, reopen error,
JSON/CSV export failures, secondary BaseException, finalizer bugs, receipt/status/disk
failure, aborted analysis with worker exit zero, summary failure, controller cleanup
accounting reaching zero and TerminateJobObject's explicit zero code. Cleanup testing
exercises the real Python controller flow with mocked OS boundaries; it is not new
native Windows Job or two-hour acceptance evidence. The existing native controller
receipt already records active_processes=0. Synthetic export fixtures test launcher
control flow, not the library JSON/CSV serialization implementation.

Code-review regression suite: 19/19 PASS, including serialization/fsync/link failures,
complete-before-publication, non-overwrite and a second KeyboardInterrupt in reopen.

## Short native Windows integration

`native_windows_smoke.py` uses actual Windows subprocesses, nested kill-on-close Job
Objects, the existing `_run_owned`/worker journal/reopen path and launcher `start_once`.
Its test-only recipe adapter substitutes `native_cpu_fixture:make_cpu` (sleep/count,
no inference), a four-second maximum work protocol and fresh temporary output paths.
It does not invoke the GPU preflight/main entry point. The launcher's fixed 7200-second
announcement/STARTED metadata is not this smoke's protocol or duration evidence;
the actual journal protocol is four seconds and every scenario is interrupted early.
No fixture here is a GPU or long-duration acceptance run.

Three scenarios cover successful interruption export, injected JSON export failure
and a secondary KeyboardInterrupt during export. Interruption is injected as a Python
KeyboardInterrupt in the real controller after two committed chunks; physical console
Ctrl+C signal delivery is not tested. Real worker processes and all Job API calls are
unmocked. The outer Job admits the controller before imports and has a 90-second parent
watchdog plus bounded exit waits. Temporary native receipts are retained outside the
original evidence. The successful-export scenario reopens actual JSON and validates
the existing flattened P31 CSV columns/status/cleanup values. P31 is not supported by
the existing `RuntimeResult.from_csv` reader (that reader accepts P34/P35 envelopes);
the first harness attempt incorrectly assumed otherwise and failed at that assertion,
after actual cleanup/export. This was a harness correction, not a library change.

Corrected native run: 3/3 PASS. Receipts:
`C:\WINDOWS\TEMP\hyso-p31-native-finalization-pj4hw3wq`.
Controller PID 8348 exited zero after asserting all scenarios; worker PIDs 13244,
16260 and 25428 each had a forced zero exit code but `analysis.status=aborted`.
Each inner Job and the outer Job reported active_processes=0. Every scenario
created interruption and launcher status receipts; injected export/secondary errors
were retained without replacing the first KeyboardInterrupt. CUDA remained
uninitialized. No long-run acceptance or physical Ctrl+C-delivery claim follows.

Original evidence integrity: the 92-file aggregate SHA-256 remained
`9f3f3a17f64ef363db29f9e6c401620c0c03d7b7699587b0138492515788993b`.
The original P31 worktree was clean before this review; all modifications here are
outside it. The historical 10.13% throughput decline remains unresolved.

## Acceptance still missing

The existing native run does not satisfy 7200 seconds; verified sample span is
4248.1582207 seconds. Its successful worker terminal/close, exact work-end receipt,
launcher final reports and parent exit code are missing. This revision does not create
those historical artifacts, repair the old evidence, or complete P31 acceptance.
