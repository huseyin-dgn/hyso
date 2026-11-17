# P12 read-only observations (schema 1, adapter 1)

Use the separate opt-in module to inspect an existing P12 `OperationCountResult`
through a validated `EvidenceRecord`. The legacy `NativeMetricResolver`, `advise`,
comparison contracts and their JSON defaults are unchanged. Importing this module
does not register a new legacy Advisor metric or start a workload.

```python
from hyso.analysis import EvidenceRecord
from hyso.advisor.p12_observation import P12Observation, observe_p12

def observe_saved_p12(path, trusted_artifact_sha256):
    record = EvidenceRecord.from_json(
        path, feature_id="P12", expected_sha256=trusted_artifact_sha256,
        origin="reused",
    )
    return observe_p12(record, version=1)

# Use a saved counts file and its previously verified manifest SHA:
# observation = observe_saved_p12(counts_path, trusted_artifact_sha256)
# data = observation.to_dict()
# observation.to_json("p12-observation.json")  # exclusive atomic export
# restored = P12Observation.from_json("p12-observation.json")
```

`version` is required. Only integer `1` is accepted. `P12Observation` has its own
`hyso.advisor.p12_observation` schema; it is not an `AdvisorReport`, a legacy
`ResolvedMetric`, a comparison contract or a `PresentationReport`. Its JSON reader
revalidates the embedded record and rederives every projection. Unknown versions,
extra fields and rehashed changes to counts/authority/claims are rejected. Exports
accept `overwrite=False` by default; source files are never overwritten by reading.
The observation envelope is bounded to 64 MiB.

| Retained evidence | Observation |
|---|---|
| Complete observed operator coverage | `availability=AVAILABLE`; FLOPs/batch and MACs/batch `status=DERIVED` |
| Unknown operator cost | `availability=PARTIAL`; total values null, `status=UNKNOWN`, native reason/coverage retained |
| Optional missing producer/source/state field | Value null and authority `missing`; identity status can be PARTIAL while counts are available |
| Caller-only scope label/state association | `declared_scope` and `scope_bindings=declared`; never copied into a missing native model-state field |
| P14 operator profile, P16 utilization, unavailable P12 or raw result object | Rejected; a native P12 EvidenceRecord is required |
| Missing mandatory native result identity or invalid native counts | Rejected by the original native/record reader; no placeholder identity |

The observation retains input shape/batch/dtype/device, counting convention and
operation-model version, operator coverage/unknown-call count, producer and source
UUID/SHA, trace reference, native protocol, existing model-state/input scope and all
native/declared/missing bindings. The full original native payload is embedded.
Caller labels are not architecture identities. Historical QA config/state hashes
are not substituted for native tensor-state hashes or checkpoint-file hashes.

`producer_identity.native_protocol_sha256` is the existing Analysis scope digest
over the native protocol/configuration/training/optimizer context. It is **not** an
external QA protocol SHA, a trace SHA or an original artifact byte SHA. Recipe
locators, including output receipt paths, remain unchanged. No path is followed to
load a factory, model, checkpoint or trace. `native` means recorded in the validated
producer payload, not authenticity attestation or independent replay of execution.

Counts describe executed recipe/input paths under the retained logical counting
convention. Coverage is the observed ATen-call fraction, not a percentage of FLOPs,
hardware operations or kernel coverage. Special operations/comparisons retain
their native domains; arithmetic counts are not a hardware FLOP counter, FLOP/s,
latency or a performance recommendation. A supported observation is not general
model acceptance.

P12 and P16 remain `NOT_SUPPORTED_IN_COMPARISON` in `compare_models`. This adapter
never manufactures `comparison_capture`, COMPARABLE status, a winner, speedup,
statistical superiority or optimization success. P16 producer/transport is unchanged.

The committed `p12-t05-*-v1.json` regression inputs are exact copies of T05's saved
synthetic CPU/FP32 batch1 Sequential and Graph counts. Their source UUIDs, original
locators and original file SHA values are preserved. The P14/P16 copies exercise
wrong-type rejection only. Tests perform no new model forward or profiler capture.
Observation JSON preserves native **values** and the original artifact byte SHA;
it does not recreate the source file's whitespace. Tests separately check original
fixture bytes remain unchanged and embedded canonical native payloads are exact.
