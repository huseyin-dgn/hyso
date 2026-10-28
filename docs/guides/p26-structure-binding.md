# Trusted P26 structure-to-evaluation binding — 3B.4E.1

```python
from hyso import evaluate_classification, EvaluationResult

# Existing trusted CPU model and finite sequential classification loader.
result = evaluate_classification(
    model, loader, class_names=["negative", "positive"], split="test",
    capture_evidence=True, capture_structure=True,
)
result.to_json("evaluation-with-bound-structure.json")
snapshot = result.structure_snapshot  # Validated ModelStructureSnapshot.
assert snapshot.content_sha256 == result.to_dict()["structure_binding"]["structure_sha256"]
```

`capture_structure` defaults to **False** and must be a bool. Enabling it requires
`capture_evidence=True`; neither flag is silently enabled. Nonempty successful
model inference is required: empty evaluation cannot claim a used forward
receiver. Use standalone `capture_model_structure` for structure without an
evaluation. P26's existing CPU, finite sequential loader and isolation
restrictions remain in place. No new model source is imported or checkpoint
loaded by this option.

## Actual writer path and isolation

P26 retains the original model and makes its existing isolated deepcopy. Its
existing object/storage alias checks, eval/inference mode, model-state checks,
coverage accounting, original-model preservation checks and RNG/generator
restoration are unchanged.

Only the **actual isolated object called by `candidate(inputs.clone())`** is
supplied to `capture_model_structure`. P26 captures its registered metadata:

1. After `candidate.eval()` and the existing state-identity check, before
   starting the loader iteration.
2. After successful sample/batch accounting, before constructing the result.

The full pre/post structure hashes must match. A separate post-capture
model-state hash is observed and must match the evaluated baseline and original
recorded state. A mismatch or capture failure produces an error, with no native
bound result. RNG and the loader generator are restored through the existing
finally path. No independent inference pass, training, extra dataset iteration,
forward hook, model edit or device transfer is added. Structure capture itself
is metadata-only, as specified by [3B.4D.1](model-structure.md).

The original model is **not** used as a surrogate structure source. A custom
deepcopy can return a different type or registered structure while preserving
named-tensor state. Schema 3 therefore records the evaluated receiver type from
its own snapshot and explicitly retains
`original_structure_relation=not_captured_or_equivalence_not_certified`.
`source.model_type` continues to describe the original argument. Equal state
hashes or a deepcopy operation do not certify equal architecture or behavior.

Two boundary observations do not prove structural constancy at every instant.
Transient changes restored inside forward, custom nonregistered Python state,
hooks, callable implementation and actual executed operator graphs are outside
the captured scope. Unsupported/missing snapshot fields retain those statuses;
a native association does not turn them into complete architecture evidence.

## Explicit new schema and identities

With both flags true, the writer emits classification schema **3**. Schema 1
remains the default; `capture_evidence=True` alone still emits exact schema 2.
Their fields, meanings and canonical hashes are unchanged; no old record is
upgraded on read. An old reader rejects schema 3 instead of discarding its data.

Schema 3 retains the unmodified `producer_evidence` v1 contract and adds a
`structure_binding` v1 object with format
`hyso.evaluation.structure_binding`, method
`p26_isolated_forward_receiver_pre_post_v1`. Its scope is
`registered_structure_of_forward_receiver_at_evaluation_boundaries`.

| Identity | Meaning and binding |
|---|---|
| `structure_sha256` | Canonical D.1 structure snapshot hash; full snapshot is embedded at `structure_binding.snapshot` |
| `pre_structure_sha256` / `post_structure_sha256` | The matching observed structure boundary hashes |
| `model_state_sha256` and pre/post state fields | Separate historical P26 named-parameter/buffer state hash; must match native producer evidence and source |
| `checkpoint.status` / `checkpoint.sha256` | Separate optional trusted artifact receipt, exactly matching existing producer evidence; missing stays null/missing |
| `evaluation_id` | The actual P26 result UUID allocated by its writer, matching the containing result |
| `evaluated_stream` | Existing native input/target/pair hashes, method, scope and consumed count; all must match producer evidence |
| `evaluated_model_type` | Snapshot root type of the isolated forward receiver |
| `forward_batches` / `observed_phases` | Successful batch count and the two defined observation boundaries |

P26 `source_sha256` covers the complete new binding and embedded snapshot along
with the existing native evidence. File and canonical payload hashes in
EvidenceRecord/EvidenceRef further retain the source reference. No structure
SHA substitutes for a model-state or checkpoint SHA. The checkpoint receipt
continues to bind loaded bytes to model state, not to Python architecture/code;
the original-model receipt does not certify original/copy structure equivalence.

The feature accepts no caller snapshot, path, hash or model-name matching string.
Model attributes such as `structure_sha256` cannot supply or repair a native
binding. Dataset/split/source_id/preprocessing declarations retain their
existing declared/missing authority.

## Reader, native adapter and resolver

`EvaluationResult` checks schema 3 through the existing metrics/accounting and
producer-evidence readers, then reconstructs the embedded
`ModelStructureSnapshot`. It verifies its hash, native Torch provenance,
boundary equality, receiver type, all peer state/checkpoint/stream identities,
evaluation UUID and accounting. Missing fields, unsupported versions, false
native claims, extra fields and rehashed inconsistent bindings are rejected.
`structure_snapshot` returns a detached validated typed snapshot for schema 3,
and None for schemas 1/2. JSON is the complete interchange; the unchanged CSV
is only the existing quality-metric projection.

The closed evidence adapter explicitly admits P26 schema 3. Existing native
model-state/checkpoint scope semantics match schema 2; the split remains
declared. No field is added to EvidenceScope, avoiding old record/hash changes.
Structure identity stays in the new native binding, separate from other hashes.

`NativeMetricResolver` resolves the recorded schema-3 accuracy/macro-F1 mapping
with an explicit `hyso.evaluation.classification:schema3` protocol reference.
Its `native_context.structure_binding` retains the observation method/scope,
evaluation ID, evaluated type, distinct identities, phases/batch count and paths
to the native binding, snapshot and limitations. The existing EvidenceRef still
binds the quality scalar to its report/record/artifact/canonical payload. The
snapshot is not a new quality, latency or memory metric. Configured `test` is
still only a caller-declared split label, as with existing quality resolution.

Internal consistency of imported JSON does **not** cryptographically attest a
live model's origin or a real producer run. Trust in the original producer and
artifact provenance remains external; the reader never follows type names to
import a model or re-execute a workload. Native binding describes what the
trusted writer observed, not proof against a fabricated self-consistent JSON.

## Deliberate integration boundary

`compare_model_structures` and D.1 snapshot semantics are unchanged. A caller
can explicitly inspect or compare extracted typed snapshots; no combined
quality/latency/memory/structure verdict is generated.

The existing `P26QualityComparisonContract` policy v1 remains schema-2-only.
It does not silently admit schema-3 records; they remain insufficient under
that policy. [3B.4E.2](p26-model-profile-comparison.md) provides a separate
schema-3 policy and unified result; it preserves independent structure/quality
eligibility and never silently extends this schema-2 policy.
The general v1 comparison contract likewise retains its original requirements.
Binding validation is not comparison eligibility, statistical independence or
`VERIFIED_IMPROVEMENT`.

No P11 provenance bridge, automatic Advisor recommendation, new end-user report,
GUI or Architecture Audit is added. Existing TR/EN rendering and old source
JSON/hashes remain unchanged. Additional cost is explicitly opt-in: two bounded
metadata captures, embedded metadata serialization and one extra candidate-state
hash, in addition to the existing P26 producer evidence work.

Focused CPU tests in `tests/test_evaluation_structure_binding.py` exercise real
producer -> serialization -> reader -> native adapter -> resolver, actual
receiver/call counts, modified deepcopy structure, separate checkpoint identity,
pre/post mismatch failure, state preservation, failures/RNG restoration, old
schema byte/hash equality and rejection of incoherent new evidence. Snapshot
capture is guarded against forward/state/value-copy/device/hook operations;
reader/resolver checks block model execution, capture and checkpoint loading.
