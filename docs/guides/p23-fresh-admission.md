# P23 post-import admission

Current policy (2026-11-25): `--mode native` and the direct native controller,
supervisor, spawn and worker entries reject `BLOCKED_B4_UNRESOLVED` before
admission, file writes or process creation. A manifest, prior authorization or
environment override cannot enable CUDA saved-activation diagnostics. Generated
CPU diagnostic adapters retain every production device guard. The admission
workflow below remains a reference for the previous design and supported CPU
fixtures; it does not authorize or enable the disabled native mode.

This changes only the private forensic harness. The production CUDA saved-payload
gate stays disabled and P23 stays PARTIAL. No CUDA execution is authorized by a
CPU test, preflight result, retained review, or previous manifest authorization.

The controller still verifies the exact manifest, sources, runtime and separate
one-attempt authorization, then admits a fresh operator-reviewed initial capture.
The capture and review must both bind to the new run ID, full manifest SHA256,
`controller-preflight` stage, capture nonce and worker PID zero. They cannot be
silently rebound from a different manifest. Ordinary legacy policy callers keep
their existing evaluator contract.

The owned worker checks that initial admission before importing Torch. It imports
Torch and the private adapters, verifies loaded Python/native identities, and
prepares instrumentation and the fixed tiny protocol. CUDA must remain
uninitialized. Immediately before recipe entry, it takes one new host/process/GPU
capture using only stdlib and the installed read-only NVML CLI. Sensor helper
processes inherit the existing bounded worker Jobs and watchdogs.
Both stages use `native_admission.collect_p23()`, which also retains the installed
CLI's C/G process table. A malformed/unavailable table or any table PID outside
the identity-reviewed compute inventory blocks admission, including a pure G PID.

The final capture binds to the same run/manifest, `worker-final` stage, the owned
controller nonce and actual worker PID. Its review binds to that exact context and
inventory. Each current GPU process and its parent, creation identity, executable,
command line, NVML row and explicitly restricted fields must match the initial
review. A new, removed or changed GPU process stops the attempt; no name allowlist
or automatic new desktop classification is used. Workload judgments remain trusted
operator input, not cryptographic attestation. A stable process identity cannot
prove that its internal workload is unchanged.

Both captures must independently satisfy the unchanged policy when used for their
respective admission. The retained initial review is checked structurally without
altering timestamps or treating its old telemetry as fresh. Final admission uses
the real current time and new host/disk/GPU observations. Capture start must be at
most 30 seconds old; available RAM must be at least 4 GiB, disk at least 12 GiB,
and each of three ordered GPU samples must have at least 2048 MiB free on the pinned
GPU. Utilization remains telemetry. Publication time is included in a last freshness
check. There is no collection loop, resampling until eligible, or workload retry.

The controller/worker watchdogs remain 90/60 seconds. Process private commit remains
4 GiB; child/outer Jobs remain 8/12 GiB with 4/8 processes and kill-on-close. Dump
scope, 4 GiB file budget, 20-second collector deadline and output limits are
unchanged. A slow import/capture can still exhaust the fixed watchdog; a stale or
ineligible final capture stops before CUDA. Failed collection retains the phase
ledger and owned cleanup receipts; collected evidence and the final decision are
retained separately when available.

Offline regression entry points are `tests/forensics/test_p23_fresh_admission.py`,
`test_p23_preflight.py` and `test_native_admission_recovery.py`, run with isolated
stdlib-only unittest discovery rather than the repository Torch conftest. Scoped
`OwnedSafety` cases cover exception/timeout/tree cleanup and missing native consent.
CPU simulations do not establish real CUDA lifecycle or fix the historical crash.
