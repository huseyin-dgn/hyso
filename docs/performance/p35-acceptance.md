# P35 execution-provider native acceptance

**P35 = IMPLEMENTED for observed node/provider execution and placement fallback
within the recorded Windows CPU/CUDA scope.** This is not execution-error retry
recovery, universal accelerator support, source attribution through arbitrary
fusion, GPU kernel timing, or a performance/speedup claim.

## Original requirement and prior evidence

The original `a0672d5` roadmap requires real graph/session profiling and node
execution/provider evidence. A provider list alone cannot establish fallback.
The previous P35 acceptance recorded missing ONNX Runtime, zero completed runs
and `native_validation_not_performed`; protocol doubles were not native evidence.
That historical result remains unchanged. Most implementation already existed;
native execution proof was missing. This acceptance also found and fixed a real
optimized-graph reader defect described below.

## Runtime isolation and compatibility

The existing torch_gpu environment has Python 3.10.18, PyTorch 2.5.1+cu121,
ONNX 1.20.0 and loaded NumPy 2.2.2. Its PyTorch CUDA build is 12.1 and cuDNN
version query is 90100. No ONNX Runtime was installed in that environment.

- CPU acceptance used **ONNX Runtime 1.23.2**, matching the project's optional
  CPU dependency pin, in an isolated runtime target directory.
- Accelerated acceptance used **onnxruntime-gpu 1.20.2** in a separate virtual
  environment sharing existing scientific packages. The compatible 1.20.x family
  avoids replacing the existing CUDA/cuDNN libraries. A 1.20.1 Windows/Python 3.10
  wheel was unavailable; the available 1.20.2 patch was selected before execution.
- Only pinned ORT distributions and coloredlogs/humanfriendly were installed into
  those isolated locations, without dependency upgrades. Base package inventory
  and protected source/evidence are checked independently.
- The GPU runtime advertised TensorRT/CUDA/CPU, but only actual CUDA/CPU Node
  events count as execution evidence. TensorRT was not validated.

Provider-specific library versions remain unavailable where ORT exposes only its
runtime version/build. GPU selection was explicit and the existing worker received
the UUID visibility setting; the result does not independently attest a physical
UUID through ORT's profiler. No driver, clock, power or sleep settings were changed.

References: [ORT CUDA requirements / PyTorch compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
[provider priority and profiling](https://onnxruntime.ai/docs/api/python/api_summary.html),
[1.20-family kernel registry](https://github.com/microsoft/onnxruntime/blob/v1.20.1/docs/OperatorKernels.md).

## Primary native captures

Each capture completed one real inference, without warmup or automatic execution
retry. These four primary captures are separate sessions, not a combined timing
experiment. CPU regression tests below include additional short CPU sessions and
controlled failures; they are not counted as extra GPU evidence.

| Capture | Runtime | Real observed execution | Interpretation |
| --- | --- | --- | --- |
| CPU ReLU | 1.23.2 | `relu` on CPUExecutionProvider | Same source/optimized node, expected provider matched |
| CPU fusion | 1.23.2 | FusedGemm on CPUExecutionProvider | Three source nodes fused to one; source mapping stays unknown |
| CUDA ReLU | 1.20.2 | `relu` on CUDAExecutionProvider | CUDA execution proven by Node event, not enabled-provider list |
| Mixed CUDA/CPU graph | 1.20.2 | `relu_gpu` on CUDA; `scaler_cpu` on CPU; internal MemcpyToHost event retained | CUDA-priority graph placement fallback and expected/actual mismatch observed |

The controlled mixed graph uses float ReLU followed by `ai.onnx.ml.Scaler`.
Both providers were enabled, CUDA was first in priority, CUDA actually executed
ReLU, and CPU actually executed Scaler. The report identifies that observed CPU
placement where CUDA was expected. It does not invent a CUDA failure, session
retry, or accelerated-to-CPU transition after an error.

Schema1 `fallback_detected` retains its documented expectation-mismatch meaning.
`provider_diagnostics().execution_fallback.detected` remains null because a
failure/retry transition was not observed. The observed placement fallback is
the original roadmap's node/provider scope, not a stronger recovery guarantee.

The older GPU runtime does not support the newer constructor `enable_fallback`
keyword in the same way as 1.23.2. The adapter still calls `disable_fallback()`
before running and rejects lost requested providers before inference. Constructor
retry behavior itself is not claimed to be instrumented or prevented in 1.20.2.

All primary workers exited zero, reported completed execution and cleanup with
zero active owned processes. Their PIDs were independently absent afterward.
Exit zero alone was never used as successful execution proof. ORT's warning about
an inserted transfer is retained; no performance conclusion is drawn from it.

## Reader defect and minimal correction

The mixed graph executed successfully and cleaned up, but its initial result
analysis failed: ORT serialized an internal `MemcpyToHost` in the optimized graph,
which the standard ONNX checker did not recognize in that domain.

The reader now supports explicit **graph_analysis_version=2**:

1. Source ONNX still receives strict validation.
2. An optimized graph permits only narrow, recognized transfer definitions with
   one input/output, no attributes, and unambiguous acyclic aliases.
3. A disposable semantic-Identity copy is checked. Raw graph bytes, hashes,
   provider events and durations are retained unchanged.
4. Transfer-edge aliases are resolved only for source/optimized node comparison.
   All other node definitions must still match; optimization must be disabled
   and source names unique. Fusion/renaming remain unknown.
5. Internal transfer events retain source_node=null. They are not assigned to a
   source operator or erased to manufacture complete coverage.

No second GPU run was launched for this correction. The existing mixed capture
was reopened with an explicitly versioned analysis envelope. Its raw request,
report, profile, optimized graph and controller receipt were not rewritten.
Earlier in-task draft exports were retained separately.

Without the version field, old results retain the original graph interpretation.
A dedicated legacy custom-domain-transfer fixture verifies that previously
unknown attribution is not silently promoted. Invalid graph versions are rejected.
Public API and RuntimeResult schema_version=1 remain unchanged; JSON/CSV envelope
roundtrips retain the full payload. The historical missing-runtime JSON/CSV also
reopens with its original failure unchanged.

## Verification

- **57 PASS:** provider protocol and transfer regressions plus read-only checks
  against all four real primary captures. Synthetic doubles and modified-copy
  fault injection are labeled separately from actual execution.
- **16 PASS:** existing CPU provider regressions using isolated ORT 1.23.2,
  including native session execution, mismatch, timeout/cleanup, corruption,
  mapping, optimization and parent-state checks; 24 unrelated cases deselected.
- **73 distinct final tests PASS, zero final failures.** Intermediate repeated
  test runs are not added to this count. No full-suite/release-matrix claim.
- Missing traces/providers/durations stay unknown or null. Wrong process,
  model/session/optimized graph hashes and trace binding are rejected.
- Real fusion remains unmapped; malformed transfer attributes/arity/cycles and
  ambiguous aliases are rejected. Native JSON and CSV roundtrip exactly.

One existing regression initially omitted `runtime_search_path` on its unknown-
provider request, so the intentionally unchanged base environment correctly
reported missing_dependency. The test now forwards the same optional runtime
path as its other cases; the final sixteen-case run passes.

## Delivery and remaining limits

Production change is limited to `hyso/performance/providers.py`. Test graphs,
bounded capture entry point, transfer/legacy tests, saved-native validation and
the optional-runtime test correction accompany this document. Native reports,
profiles, ONNX artifacts, environment packages and raw logs remain outside Git.

Unsupported scope includes TensorRT/DirectML/other-provider execution, arbitrary
fusion source mapping, complete static placement APIs, per-provider DLL versions,
physical UUID execution attestation, GPU kernel/CUPTI timing and runtime-error
retry recovery. These fields are not filled with guesses. No original mandatory
node/provider acceptance gap remains in the stated CPU/CUDA scope.

P12/P29 worktree changes and user changes are preserved. P31's 92 evidence files
remain unchanged. No commit or push was made. P35 stops here; P34 is not started
and P31's two-hour run remains deferred to final validation after development.
