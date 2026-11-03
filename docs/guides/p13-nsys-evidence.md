# P13 CUDA module evidence integration

## Contract and root cause

The historical P13 criterion separates inclusive/exclusive module observations
from profiler overhead and baseline latency. Its current acceptance scope also
requires CPU ownership calibration and actual CUDA kernel/module attribution.
`tools/full_acceptance/handlers.py:layer` previously returned PARTIAL unconditionally
after CPU calibration. The acceptance matrix had no P13 promotion criterion.
`profile_layers` derives its evidence from the existing Torch capture; successful
CUDA Events or CPU ownership observations cannot supply missing CUPTI activity.

The P14 reader correlates same-process CUDA launches with lexical ATen NVTX
stacks. Its operator labels, autograd injection requirement, and rejection of
every outside-scope kernel do not describe the independent P13 module capture.
It remains unchanged. P13 uses a separate reader and separate format with only
`feature_ids: ["P13"]`. Existing `LayerProfileResult` instances are unchanged.

## Minimal implementation

`analyze_nsys_modules(configuration, max_rows=20000)` performs offline analysis.
It does not start Nsight, subprocesses, model forwards, CUDA, training, or hooks.
It opens a closed standalone SQLite export with `mode=ro` and `query_only=ON`;
live WAL/journal sidecars are rejected. Inputs are SHA256-checked before and after
analysis. A 256 MiB SQLite limit, row, instruction, forward-count and ownership complexity budgets
bound analysis. The supported export schema is Nsight Systems major 3 with the
columns explicitly enumerated in `_module_activity.py`.

Configuration JSON has these fields (artifact paths should be absolute):

```json
{
  "sqlite": {"path": "<export.sqlite>", "sha256": "<64 lowercase hex>"},
  "report": {"path": "<capture.nsys-rep>", "sha256": "<64 lowercase hex>"},
  "global_pid": 281879844487168,
  "forward_labels": ["hyso.p13/forward/0", "hyso.p13/forward/1", "hyso.p13/forward/2"],
  "capture": {"path": "<supervisor-receipt.json>", "sha256": "<64 lowercase hex>"}
}
```

The encoded global PID includes the host/process identity; matching just the
24-bit OS PID is insufficient. Kernel context records must also identify that
OS PID/device/context. Target-process CUDA and NVTX injection diagnostics are
required. Target or global warnings prevent complete attribution; warnings for
another encoded process do not invalidate this process's observations.

Every requested forward label must occur exactly once as a complete push/pop
range; forwards must be ordered and disjoint. Module labels use the exact prefix
`hyso.p13.module/`. Measured module ranges must form an unambiguous nested stack
on the forward thread and NVTX domain, even when no launch occurs inside a range.
Repeated module paths remain distinct invocations. Label paths are recorded
instrumentation identities; no model graph or aliases are invented from names.

Kernel ownership requires a unique positive correlation ID to a successful
same-process `cudaLaunchKernel[_vN]` or `cuLaunchKernel[_vN]` host call, within
the same measured forward. The unique innermost enclosing module owns that
launch; ancestors provide context. Kernel timestamps need not fit inside the
module's host NVTX interval. They must fit the synchronized forward interval.
Kernel-before-launch ordering, multiple kernels for one ordinary launch,
unmapped kernels, absent kernel activity for measured launches, empty forwards,
and ambiguous correlations prevent completion. No nearest-time fallback exists.

Outside-scope kernels are retained separately and excluded from measured
coverage. An outside kernel with missing/ambiguous launch provenance also blocks
completion, since its scope cannot be independently established. Durations come
only from CUPTI kernel start/end; NVTX duration is never GPU execution time.

## Capture provenance and acceptance

`capture` may be omitted for preliminary diagnostics; the result then remains
PARTIAL regardless of attribution coverage. A supplied but invalid receipt is
rejected. The pinned supervisor receipt must contain:

```json
{
  "format": "hyso.performance.p13.nsys_capture",
  "version": 1,
  "status": "completed",
  "global_pid": 281879844487168,
  "worker_pid": 24132,
  "forward_labels": ["hyso.p13/forward/0", "hyso.p13/forward/1", "hyso.p13/forward/2"],
  "sqlite_sha256": "<export SHA256>",
  "report_sha256": "<binary report SHA256>",
  "nsys_exit_code": 0,
  "output_equal": [true, true, true],
  "cudnn_tf32_disabled": true,
  "gpu_uuid": "<physical GPU UUID>",
  "authorization": {"approved": true, "basis": "<actual capture authorization>"},
  "supervisor": {
    "owned_process_tree": true,
    "process_terminated": true,
    "cleanup": {"active_processes": 0}
  },
  "checkpoint": {"path": "<checkpoint>", "sha256": "<SHA256>"},
  "input": {"path": "<captured input artifact>", "sha256": "<SHA256>"},
  "worker_source": {"path": "<capture worker source>", "sha256": "<SHA256>"}
}
```

These fields are provenance attestations from the approved capture supervisor.
Hashes detect drift; they are not signatures and cannot prove an attestation is
truthful. In particular, this reader cannot decode `.nsys-rep` to prove its export
relationship, reconstruct output equivalence, or retroactively establish process
ownership/authorization. Do not manufacture missing receipts from a diagnostic
trace or populate this example as if an approved capture had occurred.

For a future prepared acceptance run, pass `--p13-evidence <configuration.json>`
alongside the existing `--preflight` arguments. Preparation pins the configuration
in `optional.p13_evidence`; admission also rechecks its artifact references. This
flag grants no permission to run native work. Existing authorization, manifest
SHA, source, supervision and cleanup gates remain in force.

The layer handler first performs the unchanged CPU profile and paired ownership
calibration. Calibration failure remains FAIL. Missing CUDA evidence is PARTIAL;
invalid supplied evidence is FAIL; valid incomplete evidence is PARTIAL. Only
complete verified CUDA evidence matching acceptance Model B's checkpoint hash
and physical GPU UUID permits the handler's PASS criterion. The matrix additionally
requires an attempted authorized stage, CPU profiling/calibration criteria and
zero active processes in the current stage supervisor cleanup. P14 is not promoted
by this criterion. Prior decisions and historical evidence files are preserved.

PASS is scoped to these CPU calibration and observed CUDA attribution criteria.
It does not certify physical retained memory, profiler-free performance, unseen
inputs/models, or universal GPU coverage.

## Local offline result, 2026-11-03

The supplied artifacts were read without rerunning capture:

| Artifact | SHA256 |
| --- | --- |
| `p13-analysis-a8941ade.sqlite` | `94dca2c950dad673aece3d1f679ae972e73ea4f3597c3ac8aa54b4de63084393` |
| `trained-model-b.nsys-rep` | `8ca0ec6331675c700ff0d666b3bd8fa6fe466fbb6ff4c100d594566ac7b1e538` |

The reader reproduced 102/102 measured kernels mapped across three forward
boundaries, 34 outside-scope kernels, 220 captured module ranges and 55 distinct
module paths. Worker PID is 24132, encoded global PID 281879844487168. No measured
launch lacked a kernel, no outside-scope kernel had unresolved launch provenance,
and no target-process warning blocked attribution. The unrelated process warnings
were not assigned to the worker.

**P13 decision: PARTIAL.** Attribution is complete for these observed scopes, but
the supplied folder contains no pinned capture receipt proving model/input/source,
authorization, equivalence and supervisor cleanup identities. No official CPU
calibration/acceptance stage was executed by this integration task. The reported
exit code and output equivalence are not independently reconstructed from CUDA
activity. The diagnostic JSON/configuration are in the new local directory
`outputs/p13-module-evidence-integration-20261103/`; historical outputs are untouched.

Unsupported cases include CUDA graphs/cooperative or extended launch APIs,
multiple runtime/driver candidates, asynchronous kernels escaping forward scopes,
cross-thread or cross-domain module ownership, incomplete exports, and automatic
module alias reconstruction. Such evidence cannot produce a complete result.

Bounded synthetic SQLite tests exercise complete attribution, missing provenance,
invalid hashes/receipts, PID/thread/context mismatches, missing and duplicate
launches/kernels, ambiguous nesting, boundary crossings, empty forwards, warnings,
outside-scope activity, and budgets. Acceptance tests cover missing/invalid imports,
Model B/device mismatch, authorization, cleanup, and historical decision retention.
Synthetic fixtures are validation of the reader, never native acceptance evidence.

## Changed files and validation

Exact source/document changes:

- `hyso/performance/_module_activity.py` (new read-only validator)
- `hyso/performance/profiling.py` (public wrapper)
- `hyso/performance/__init__.py` (API export)
- `tools/full_acceptance/core.py` (configuration sealing/admission hashes)
- `tools/full_acceptance/cli.py` (`--p13-evidence` preparation option)
- `tools/full_acceptance/handlers.py` (derived layer acceptance decision)
- `tools/full_acceptance/reporting.py` (guarded P13 promotion)
- `tests/test_module_activity.py` (bounded SQLite tests)
- `tools/full_acceptance/tests/test_infrastructure.py` (admission/decision tests)
- `docs/guides/p13-nsys-evidence.md` (this contract and local analysis)

Validation command, using the existing `torch_gpu` environment:

```powershell
& 'C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe' -m pytest -c tests/pytest.ini tests/test_module_activity.py tests/test_operator_activity.py tools/full_acceptance/tests/test_infrastructure.py -q --tb=short
```

Result: **112 passed in 18.88 seconds**. `git diff --check` passed. The default
Miniconda interpreter lacked pytest; no packages were installed. No acceptance
suite, CUDA capture, training, saved-activation hooks or Stability run was executed.
No commit or push was performed.
