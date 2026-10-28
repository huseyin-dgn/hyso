# P26 producer evidence — 3B.4C.1

`evaluate_classification(..., capture_evidence=True)` opts into classification
schema **2**, containing `producer_evidence` version **1**. It runs the existing
isolated CPU evaluation once. No training, benchmark, data reread, model edit,
GPU query/transfer, or preprocessing discovery is added. The default remains
schema 1 with its original fields, model-state hashing, and export behavior.

```python
from hyso import evaluate_classification
from hyso.checkpoint import load_training_checkpoint

# Optional: existing caller-created CPU model and a trusted HYSo full checkpoint.
# Explicit restore is a separate user action, not part of P26 evaluation.
load_training_checkpoint(checkpoint_path, model, capture_evidence=True)

result = evaluate_classification(
    model, loader, class_names=["negative", "positive"], split="test",
    capture_evidence=True,
)
result.to_json("evaluation-with-evidence.json")
```

Omit the load for an in-memory model. P26 still works; checkpoint evidence is
`missing`. Both flags must be actual booleans. The existing finite, sequential,
map-style CPU loader restrictions and RNG/model isolation still apply. Custom
model/dataset code has the same external-side-effect limits as before; nothing
here automatically imports or executes user-supplied model source.

## Captured facts and their sources

| Field in `producer_evidence` | Authority and source |
|---|---|
| `model_state` | Native historical P26 named-parameter/buffer SHA-256; original and isolated evaluation model must agree |
| `checkpoint` | Native only with a matching process-local trusted-load receipt; otherwise missing with reason |
| `evaluated_stream.sample_count` | Native number of validated samples consumed in the successful P26 loop; checked against metrics/accounting |
| `evaluated_stream.inputs_sha256` | Native ordered post-collation input tensor content |
| `evaluated_stream.targets_sha256` | Native ordered original target tensor content, including dtype and per-sample shape |
| `evaluated_stream.paired_sha256` | Native ordered input/target pairing |
| `split` | Declared, from the `split` argument |
| `source_id` | Declared from the argument, or missing |
| `preprocessing` | Missing: actual transform chain is not verified |
| `dataset_origin` | Missing: tensor content and dataset class do not establish origin |

Every entry names its source or missing reason. `source.model_state_sha256`
retains its exact old algorithm (`sha256_named_parameters_and_buffers_v1`),
including historical tensor ordering and JSON header spacing. It describes
named tensors, not all Python model behavior, source code or a checkpoint file.

The new evidence object contains hashes and counts, never copies of inputs or
sample identifiers. Existing P26 `raw` targets/predictions remain unchanged;
this feature adds no raw-data logs. Hashing costs CPU time proportional to data
size and is opt-in. State is constant-size plus one contiguous sample at a time,
in addition to P26's pre-existing retained targets/predictions.

## Exact stream identity method v1

Method: `sha256_ordered_tensor_samples_le_v1`. Scope:
`ordered_post_collation_inputs_and_targets`. It supports dense little-endian
CPU tensors; unsupported encodings fail explicitly when capture is requested.

For each input and target sample (batch dimension removed), compute SHA-256 of:

1. ASCII `hyso.p26.tensor_sample.v1` followed by one NUL byte.
2. Eight-byte unsigned big-endian length of the UTF-8 JSON header, then that
   header: `[str(dtype), list(sample.shape)]`, separators `(',', ':')`.
3. Eight-byte unsigned big-endian tensor byte length, then logical contiguous
   row-major tensor bytes in little-endian dtype representation.

Three incremental SHA-256 streams start with ASCII `hyso.p26.inputs.v1`,
`hyso.p26.targets.v1`, or `hyso.p26.pairs.v1`, each followed by NUL. Append each
32-byte input digest to the input stream and each target digest to the target
stream. Append input then target digest for each sample to the paired stream.
Finalize each with the eight-byte unsigned big-endian consumed sample count.
Export lowercase hex digests. Empty evaluation hashes the empty sequence and
count zero; it retains P26's `empty` status and null metrics.

Batch boundaries, strides and storage layout do not enter identity. The same
ordered logical samples therefore hash identically under different batch
partitions. Sample order, dtype, per-sample shape, exact numeric bits, and target
pairing do enter identity. Binary targets `(N,)` and `(N,1)` have distinct sample
shapes. Reordering distinct samples changes identity; permutations of identical
input/target pairs are content-indistinguishable. No unique source sample IDs,
unordered-set equivalence, preprocessing definition, dataset lineage or
distributional equivalence are claimed. Batch-invariant content identity does
not imply batch-invariant model predictions or comparable runtime conditions.

## Trusted checkpoint link

`load_training_checkpoint(..., capture_evidence=True)` hashes a private snapshot
and passes that same snapshot to `torch.load(..., weights_only=True)`. This avoids
hash/load path replacement races. Memory is bounded to an 8 MiB spooled snapshot
plus a 1 MiB read chunk; larger snapshots spill to a temporary file. The loader
does not rewrite the source checkpoint. Seekable caller streams are snapshotted
from their current position and their cursor is restored. Artifact identity
describes those exact snapshot bytes, not a later read of the path.

Expected tensor byte fingerprints are frozen before any target load hooks.
Only a successful restore whose actual CPU named tensors and state dictionary
match those bytes can receive a process-local weak-reference receipt. Missing
nonpersistent buffers, unsupported/custom state, or hook substitutions leave the
link missing. The receipt binds artifact SHA-256 to a separate model-state
SHA-256 and to the exact original model object. Deepcopies, reconstructed models,
model attributes, paths and caller SHA strings cannot supply receipts. State
changes invalidate the match; a later load without capture clears the receipt.
An attempted restore clears it before target mutation, including failed/rolled
back restores. Validation failures before target mutation leave prior valid
state untouched. Receipts are not serialized into user models.

Restoring is still the existing explicit loader operation and changes its target
as requested by its caller. P26 itself never loads a checkpoint or changes the
original model. No external model-only loader or Trainer/EMA provenance bridge
is added in this substage. A direct `torch.load`/`load_state_dict` call supplies no
native checkpoint receipt.

## Readers, reports and comparison boundary

Updated `EvaluationResult` and the closed P26 evidence adapter validate schemas
1 and 2. Unknown versions, malformed hash encodings, incoherent sample counts,
declared-to-native status changes, supplied-prediction captures and silent
schema downgrades fail validation. The complete evidence object is covered by
P26 `source_sha256`, artifact identity and canonical evidence/report hashes.
Import validation proves internal consistency, not signed capture authenticity;
the original producer remains part of the trust boundary. Imported JSON cannot
recreate a live process-local checkpoint receipt.

The existing EvidenceRecord/Index and Advisor contract versions are unchanged.
For P26 schema 2, model-state and eligible checkpoint hashes project to native
scope bindings. Split projects to `declared`; conflicting split declarations
are rejected. Stream hashes remain in `native_payload.producer_evidence` and
are never relabeled `dataset_id`, `preprocessing_id`, or ComparisonContract's
`cohort_sha256`. All schema-1 projections are preserved to avoid changing old
record IDs, canonical JSON and reports, including their historical split
projection. That legacy label does not establish verified split membership.
Old readers explicitly reject schema 2 rather than silently discard evidence;
keep the default schema 1 for old consumers. The old CSV remains a metric
projection, not a lossless carrier of new evidence; use canonical JSON.

Advisor resolves schema-2 quality metrics descriptively, references schema 2
explicitly and retains the declared-split limitation. 3B.4A/B contracts remain
unchanged: these partial captures still yield `INSUFFICIENT_EVIDENCE`, with no
numerical performance comparison or `VERIFIED_IMPROVEMENT`. Stream hashes do
not fill absent comparison metadata.

3B.4C.2 still needs explicit comparison-policy integration for these ordered
stream semantics and verified split/origin/preprocessing identities where
available. External checkpoint loaders, Trainer/EMA identity propagation and
cross-process receipts remain outside 3B.4C.1.

## Focused verification

`tests/test_evaluation_producer_evidence.py` calls the real
`evaluate_classification` producer and real HYSo checkpoint save/load with tiny
synthetic CPU tensors. Training, benchmarks and GPU entry points are blocked.
It checks state/RNG preservation, deterministic and changed content, order,
batch partitions, one dataset pass, post-collation content, accurate counts,
missing/declared authority, exact snapshot binding, unsafe-pickle rejection,
custom/partial restores, schema validation, previous reader behavior, exact
previous/current default producer JSON/hashes, and fail-closed Advisor behavior.
Retained local MNIST TR/EN views are read-only regression inputs when available.
