# Autonomous final acceptance

From the HYSo repository in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_final_acceptance.ps1
```

This same command resumes after an interruption. Codex is not needed while it runs.
There are no interactive prompts, automatic product fixes, network package installs,
GPU admission overrides, commits, uploads, shutdowns or permanent system changes.

From another terminal:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_final_acceptance.ps1 -Status
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_final_acceptance.ps1 -Stop
```

`-Stop` creates a request. The manager terminates only its owned Job and verifies
cleanup, then writes partial reports. It never kills a PID just because of its
number. Abrupt manager death closes the non-inheritable Job handle. A later run
checks the recorded process creation time and image; an ambiguous still-live
identity stops recovery safely.

`-PrepareOnly` parses/hashes existing evidence and refreshes preparation reports;
it does **not** import pytest/Torch, launch a Job, collect tests, train or run a
device workload. Changed source after execution requires an explicit preparation
transition. Old results remain in immutable transition history, affected scopes
lose acceptance, and the normal unattended run never approves changed source.

## Execution and safety

The manager reuses the retained pre-import bootstrap and production Windows Job
implementation. All source/collection/build/install/E2E subprocesses are serial,
behind exact Job admission. Startup free host commit must be at least **4 GiB**;
below **1 GiB** during work the active Job stops. Process/Job quotas are **3/6 GiB**
and at most **48** processes. Job limits are read back, including kill-on-close
and absence of breakaway. PID creation time, image, parent and admission nonce
bind the redirector and actual interpreter before product/test imports.

Windows environment keys are matched case-insensitively; mandatory SYSTEMROOT is
validated without inheriting secrets or disabling hash randomization. Launchers
are created suspended and bound to the exact no-breakaway Job with PID creation
time and image before any interpreter/redirector execution. Schema-2 receipts
separate payload admission from cleanup: a Python preinitialization failure can
have proved zero-process cleanup while remaining a failed startup. A test PASS
requires both admitted payload and verified cleanup. Unbound identities, failed
wait/query/termination or any active Job process still stop the campaign.

Every invocation first runs two bounded, owned stdlib startup checks: controlled
exit before the admission latch (exit 73), then normal admitted Python startup
with OS randomness and hash randomization. These are infrastructure operations,
not source test cases. Pytest and the remaining campaign start only after both
checks and the campaign contract regressions pass.

The evidence and TEMP drives need at least **6 GiB** free at admission and **2 GiB**
while running. Workspace growth is bounded at **8 GiB**; stdout and phase journals
have **64/128 MiB** limits per attempt. Bundled input is limited to **1 GiB**.
Resource waiting uses 30/60/120/240/300-second backoff with a cumulative two-hour
budget per invocation and at most two resource continuations per group. These
are safety limits, not a total campaign deadline. Owned per-attempt TEMP is
removed only after declared evidence is harvested. Package TEMP is retained for
checkpointed fixture recovery; it is excluded from the ZIP.

CPU groups allow three hours externally with a 30-minute no-progress watchdog;
tests' internal deadlines do not change. The forensic fixture keeps its **60 s**
internal deadline and separate Torch-free parent. Only its existing CPU recipe
is allowed to import the historically pinned runtime; it cannot initialize CUDA.
The normal source/package profile requires **torch 2.5.1+cpu**. One additional
standalone CPU forensic verification is reserved durably before launch. This is
not an admission for any native failfast/dump or GPU work.

All **13** recorded risky nodes remain BLOCKED. B4 CUDA saved activation remains
`BLOCKED_B4_UNRESOLVED`. No new P31 7200-second run is scheduled. The two old CPU
records retain their original scope and are not current-source measurements.

## Evidence and resume

Every setup/call/teardown/subtest event is fsynced in a unique hash-chained journal.
Call PASS alone is insufficient. Completed phases remain provisional until owned
cleanup is verified. Completed tests survive resource stops and manager death;
functional failures remain recorded, while only incomplete work is resumed.
Normal failures and collection errors do not prevent independent safe groups.
Safety, source-integrity, ownership or unknown-inventory violations stop safely.

The frozen 8,549-node source inventory, 36 clean-package cases, 50 campaign
infrastructure checks and 35 feature decisions have separate denominators.
Legacy `*-completed.json` files are selection masks, not evidence of success.
Legacy evidence lacking capture-time commit metadata is explicitly identified;
the reviewed Git delta does not fabricate provenance. Historic trained A/B
performance comparisons remain INSUFFICIENT_EVIDENCE where metadata is missing.

Pytest parameter IDs are opaque strings. Only the file-path prefix before the
first `::` has its separators normalized. A reviewed one-to-one migration repairs
the former whole-string normalization using the original frozen manifest; raw
collection logs and previous phase evidence are never rewritten. Valid historical
results can be relinked to corrected IDs, and are labeled historical rather than
newly executed tests.

CPU, Torch-free stdlib, campaign infrastructure and installed-package cases use
distinct inventories. Source collection imports only frozen, hash-checked test
files. New file names are discovered statically and quarantined without imports.
Exact known IDs remain executable; unknown/different parameter IDs, duplicate IDs
and missing nodes produce local quarantine/holds. Known unselected cases do not
become unknown tests. No arbitrary new test is admitted by a filename or a marker.
Independent safe groups continue. Changed known source hashes, tampered evidence,
unsafe paths or an ambiguous checkpoint identity merge still stop globally.

Completed collections may be reused after a campaign-only fix when their owned
cleanup/admission, original commit, complete journal and input/dependency hashes
are revalidated. Pytest's exit 5 is expected for collect-only with all test bodies
deselected; it does not prove a source test PASS. Quarantine diagnostics and their
overlap with frozen held nodes are reported separately in `quarantined_tests.json`.

## Evidence writer and acceptance boundary

The evidence writer binds its filesystem, JSON encoder/decoder, hash and clock
primitives before product fixtures can monkeypatch shared standard modules.
Product tests keep their intended mocks: the observer never restores or changes
them. Journal sequence/head advance only after a complete write and real fsync.
A real writer I/O failure poisons that writer; it cannot append a fork using its
old head. Main phase order, skipped/failed subtests and package cleanups are part
of final outcome reconciliation. Package teardown completes after doCleanups.

One proof gate validates request/selection, source and environment identity,
per-test source hashes, OS admission receipts, CPU profile, quotas and owned
cleanup before checkpoint acceptance. Schema-3 controller receipts additionally
seal the whole journal and a parent terminal receipt after writers have exited.
Worker committed-head receipts are checked against that stream. Unknown hash,
identity or safety violations remain global stops. Node-local malformed events
are held individually. Interrupted tails never create missing phases.

The recorded schema-1 source-1 failure is explicitly reviewed: its 1,610-event
prefix is hash-valid; three non-test tail records fork after a product fsync
mock affected the old observer. Only 401 complete PASS cases are recovered.
The unfinished fsync case remains unaccepted. Recovery receipts and the original
bytes are retained; this rule cannot bridge arbitrary corrupt chains or apply
to new schema-2 streams. Recovery is idempotent and does not rerun these cases.

The startup infrastructure suite includes hash-pinned replays of this real
source attempt, its JUnit/ownership records and real forensic subtest records,
plus negative tamper/cleanup/PID and writer-failure cases. The previously
unfinished source fsync test is then run once as a short owned CPU canary before
the remaining groups. No completed source case is selected for that canary.

Live `provisional_new_PASS`, `provisional_completed` and
`provisional_remaining_tests` report verified complete event sequences while
final `new_PASS`/`definitive_remaining_tests` wait for controller/cleanup proof.
`recovered_from_prior_attempts` identifies recovered history separately. The
JSON and HTML views use these same fields. A provisional count is never added
to finalized PASS. Reports keep stream quarantine and functional failures apart.

Current-HEAD distributions are built in owned TEMP from `git archive`, installed
into separate fresh wheel/sdist venvs, checked against the commercial source and
support-data hashes, then exercised from outside checkout. The retained CPU E2E
harness verifies Builder/Trainer, read-only Analyzer/Advisor, API/CLI, native
comparison capture, four-language JSON/Markdown/HTML, CSV and B4 rejection.
Only explicitly owned tiny fixture models may be trained. Original user models,
checkpoints and datasets are hash checked without deserializing them in the
manager. Untracked user files and the dirty forensic guard expectation remain
preserved. No model/venv/cache is bundled.

Reports live at `outputs/release-acceptance-20261125/final-campaign/`.
Read `progress.json` / auto-refreshing `progress.html` while running. The final
upload is `COMPLETE_EVIDENCE.zip`, with `evidence_manifest.json` inside and
`COMPLETE_EVIDENCE.sha256.json` alongside it. PREPARED reports contain only
reconciled old evidence and do not claim that this new campaign has run.

## Prerequisites and limits

Windows, the retained CPU Python, offline dependency wheelhouse, Git and declared
historical fixture paths must still exist. Runtime metadata is checked offline
at preparation; imports, Job behavior and infrastructure regression checks are
validated only when the user starts the command. Missing dependencies are
reported; the manager does not modify the main torch_gpu environment.
If TEMP is removed after interruption, missing fixture state is reported rather
than treated as completed proof. Insufficient resources produce PARTIAL/
BLOCKED_RESOURCE after finite waiting. A full CPU/CUDA release cannot receive
FULL_PASS while B4/P23 CUDA and P31 CUDA admission/evidence remain open.
