# B4/P23 r3 admission wrapper correction and single CUDA diagnostic

Date: 2026-10-22, Europe/Istanbul. Branch: `feature/hyso-models-next`.
HEAD: `b0e48e98366bcf8c0054e561434bdf35f358dc74` (unchanged).

Evidence directory: `C:/Users/hdgn5/AppData/Local/Temp/hyso-p23-r3-wrapper-fix-20261022T010834`.
Sealed snapshot: `C:/Users/hdgn5/AppData/Local/Temp/hyso-p23-product-policy-20261021T232140/proposed-hook-off-control-r3`.
Run ID: `f6ff1fa5-60d4-41b0-963f-d6459856c479`.
Manifest SHA256: `660a53dbdde6a21901bf6ad2743a3fad9bb51bad57ebd25a5a5dee46ad2444b8`.

## Wrapper correction and preflight

Only the temporary admission wrapper was corrected. Previous-PID membership and its hardcoded PID map were removed. Each explicit operator classification now binds to the current inventory digest and that PID's CIM process, parent and raw NVML evidence. Every current GPU PID must be covered exactly once; names and WindowsApps paths do not automatically authorize a process. A temporary helper only binds supplied decisions to captured facts; it does not classify processes. Sealed controller, evaluator, manifest and production code were unchanged.

Exactly one offline evaluation of the retained previous capture passed, with zero CUDA executions. Its recorded clock was used only for historical validation; the offline branch exits before admission. Exactly one subsequent fresh preflight passed using the live clock and unchanged r3 evaluator. The capture was admitted within the 30-second freshness limit.

All 12 fresh GPU PIDs were individually reviewed against process, command line and parent context; no compute conflict or unknown classification remained. PID 15200 was independently supported as Phone Link COM/UI activity by matching NVML/CIM identity, `-ComServer:Background -Embedding` arguments and its service-host parent, rather than prior-list membership. Per-PID reasons and facts are retained in `live-review.json`; restricted compositor identity fields remain explicitly acknowledged, without claiming binary attestation.

Available host RAM was 7.067 GiB (minimum 4); disk 44.066 GiB (minimum 12). GPU free memory was 5537/5537/5537 MiB (each minimum 2048), on the authorized UUID. Utilization 36/36/20 percent was telemetry only. Source/runtime identity and process inventory requirements passed.

## Single execution, phases and exit

CUDA was started exactly once: AdamW FP32, default stream, hooks OFF/reset ON, seed 71, two epochs, five samples, batch two, accumulation two, existing Linear–Dropout fixture. No baseline, prefix, retry or second scenario ran. Exclusive attempt and child-launch markers were consumed.

Controller PID 23464, parent PID 15784 and worker PID 408 all exited 0. Diagnostic outcome was `success`, scenario count 1, baseline fits 0; parent CUDA initialization remained false. Worker elapsed 18.641 seconds; outer elapsed 45.828 seconds. Neither monitor reported timeout or native crash.

Durable child events prove Job admission before import, identity checks, factory construction, one fit entry/return, six microbatch entries, four attempt entries/exits, recipe completion and worker publication exit. Four reset attempts completed four resets. Hook pack/unpack/release/cleanup counters were all zero. Activation measurement remained unsupported with hooks disabled; this is not activation-accounting acceptance.

## Identity, limits and cleanup

Prelaunch and postflight full rehashes matched all 252 source and 26 runtime files; sealed sources remained read-only. All eight controller/parent/child identity checkpoints verified the same manifest. Runtime was Python 3.10.18 / Torch 2.5.1+cu121; no dependency or driver changes were made.

Worker 60-second and outer 90-second timeouts, 4 GiB per-process cap, child Job 8 GiB/4 processes, outer Job 12 GiB/8 processes and allocator fraction .15 remained unchanged. Job limit readbacks, admission and kill-on-close receipts were retained. Child and outer receipts both reported active processes 0; owned controller/parent/worker PIDs were absent at postflight. Child Job peak commit was 2,549,669,888 bytes; outer peak commit 4,178,354,176 bytes.

Child stdout/stderr were 0/324 bytes (stderr contained a GradScaler deprecation warning); outer stdout/stderr were empty. Worker result was 258,058 bytes and diagnostic result 585,992 bytes, below 2 MiB. Event streams were 209/5/9 rows and 81,708/2,795/6,039 bytes, below the per-process 2,000-row/1 MiB bounds. Last phases were worker publication exit, parent cleanup exit and outer cleanup exit.

All original 38 dirty files retained their bytes and Git statuses; this audit is the only repository addition. The index stayed empty. Protected gradient SHA256 remains `17b1e7ac85a3af92c28def4389fadb942d4777ab353baca29a2834547e78c10d`; production guard SHA256 remains `9bacc67691ab032dfa9b6202f5ea3c4bd8b78e0653aca40d53b2fa6e86f85a0d`. Prior 63 PASS CPU tests were not rerun. No stage, commit or push occurred.

## Open blockers

This narrow hooks-OFF/reset-ON diagnostic passed. It does not resolve the historical native crash root cause or validate hooks-enabled activation accounting. B4 remains OPEN, P23 and B10 remain PARTIAL, and broader R1 remains NO-GO. No further scenario or feature work was started.
