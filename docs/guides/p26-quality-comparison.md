# P26 observed-stream quality comparison — 3B.4C.2

`P26QualityComparisonContract` schema 1 explicitly selects policy
`p26_exact_observed_stream_v1`. It compares recorded accuracy or macro-F1 on the
**same observed ordered post-collation input/target tensor stream**. It does not
certify dataset origin, split membership, preprocessing implementation,
statistical significance, independent acceptance or overall model superiority.

The existing `ComparisonContract` v1 and its default behavior are unchanged.
No general latency, throughput, RSS, batch or precision condition is relaxed.
The new contract rejects every metric except `NativeMetric.TEST_ACCURACY` and
`NativeMetric.TEST_MACRO_F1` and permits only the explicit `model` change axis.

[P26 schema 3 structure binding](p26-structure-binding.md) is separate. This
policy v1 stays schema-2-only; binding validity does not silently grant comparison
eligibility to schema 3 or combine structure with a quality/performance verdict.

[3B.4E.2 model profile comparison](p26-model-profile-comparison.md) adds the
separate `P26BoundQualityComparisonContract` / `p26_bound_exact_observed_stream_v1`
policy for validated schema-3 bindings and an immutable unified result with
independent structure, accuracy and macro-F1 eligibility. This schema-2 policy
and its existing JSON/hash semantics remain unchanged.

```python
from hyso.advisor import (
    P26QualityComparisonContract, NativeMetric, compare_records,
    compare_performance,
)

# Existing NativeMetricResolvers and record IDs, built from actual opt-in
# evaluate_classification(..., capture_evidence=True) schema-2 results.
# This API does not accept or run model objects.
for metric in (NativeMetric.TEST_ACCURACY, NativeMetric.TEST_MACRO_F1):
    contract = P26QualityComparisonContract(
        metric=metric, allowed_changes=("model",), min_samples=1,
    )
    result = compare_performance(
        baseline_resolver, baseline_record_id,
        candidate_resolver, candidate_record_id,
        contract=contract,
    )
    # Full policy, source EvidenceRefs, eligibility, numbers or null, and limits.
    result.to_json(metric.value + "-observed-comparison.json")
```

## Eligibility and authority

Both records must pass the existing immutable report/native constructor
validation, belong to P26 classification schema 2, and carry valid producer
evidence v1 from [the producer capture path](p26-native-evidence.md). Both must
be complete, nonempty, meet the explicit native sample minimum, and resolve to
the requested derived fraction metric with valid native references.

The policy requires equality of:

- Native consumed sample count and all three input/target/pair stream hashes.
- Hash method, evidence version, ordered scope and batch-invariance semantics.
- Full configured task, ordered class mapping, split label, zero-division and
  prediction decision rules. Macro-F1 averages all configured classes, including
  absent classes; accuracy uses sample accuracy.
- Native batch-size sequence, CPU execution mode, sampler/worker conditions,
  and recorded Python/Torch/platform provenance.
- Resolved metric path/definition, unit, domain, status, aggregation, protocol
  reference and full metric configuration.

The supported quality metric mapping currently requires the **caller-declared**
split label `test`. This checks configured metric scope only; it does not prove
test membership. Configured class names define the calculation's mapping and
must agree in order. Matching names do not authenticate semantic class lineage.
`source_id` may differ or match; it remains declared and is never a native run ID.
Caller `EvidenceScope` dataset/preprocessing/checkpoint labels cannot repair or
override any of the required native stream/model-state facts.

Model-state method/hash and model type come directly from the native producer.
Known model identity differences without `allowed_changes=("model",)` are
incompatible. Even equal named tensors/type names do not identify all model
architecture or Python behavior. Without explicit model permission, that
remaining behavior gap is insufficient evidence. Version 1 therefore requires
explicit model permission for any eligible pair; it makes no assertion that
model behavior is fixed or that one recorded model identity covers all code.

Checkpoint evidence is **not required** for this narrow quality scope. A missing
artifact receipt remains missing; it is never synthesized from model-state
identity. A real native receipt, when present, remains in its original scope and
reference bindings. Different checkpoint artifacts do not invalidate a comparison
that explicitly permits model changes. Neither checkpoint nor dataset-origin or
preprocessing declarations are used as surrogate content equality.

Batch partition invariance belongs to the stream hash, not model execution:
even an eval-mode model can depend on batch composition. This policy keeps the
native batch partition fixed; changing it is incompatible, even when all three
content hashes agree. Precision and other protocol changes are also not allowed.

## Outcomes and reuse

`COMPARABLE` means the recorded results are eligible for this exact observed
stream scope and explicitly allowed model changes. It does not establish IID
samples, independent confirmation, significance, causality, preprocessing or
dataset equivalence beyond those observed tensor contents.

`INSUFFICIENT_EVIDENCE` covers missing records, unavailable/legacy/schema-1 or
supplied-prediction evidence, unobservable quality metrics, insufficient sample
minimums, and missing model behavior permission when identities otherwise agree.
Malformed or declared critical producer fields are rejected earlier by the
native reader; scope declarations do not make them admissible.

`INCOMPATIBLE` covers known stream/count/class/protocol/runtime/batch differences,
unpermitted known model changes, or reused captures. Known mismatches take
precedence when another condition is missing. Record identity, exact artifact,
canonical native payload and native `evaluation_id` are checked for reuse across
imports and reports. Reusing the same declared `source_id` does not imply reuse;
changing an import timestamp cannot turn the same evaluation into two runs.

Native validation and run IDs prove consistency relative to a trusted producer,
not cryptographic capture authenticity. No policy can certify independence or
detect a deliberately fabricated JSON capture merely from self-declared hashes.
The comparison never reads paths or re-executes model/dataset code.

## Numbers, references and limits

Only eligible pairs can produce `MetricDifference`. Existing 3B.4B arithmetic is
reused: signed difference is candidate minus baseline; percentage-point change
is 100 times the fraction difference; relative percent is separately 100 times
the difference divided by baseline, or null for a zero baseline. For accuracy
0.92 to 0.95 these are +0.03, +3 percentage points and about +3.26087 percent.
Ineligible pairs contain no numerical difference. Results remain per-metric;
accuracy and macro-F1 are separate comparisons, with no combined model verdict.

`PerformanceComparison` retains the complete versioned policy and its hash,
eligibility, both regenerated native `EvidenceRef`s and resolved metric contexts.
Its immutable JSON must retain all general 3B.4B limitations plus exact-stream,
declared-split, unverified-origin/preprocessing, optional-checkpoint and variable
model behavior limitations. Verified dataset origin, split membership,
preprocessing identity, quality acceptance thresholds, independent acceptance
and uncertainty remain missing. P29 acceptance is incomplete. No
`VERIFIED_IMPROVEMENT` is generated, including for a favorable metric difference.

This is an explicit comparison artifact API. It does not change automatic
Advisor recommendations, existing TR/EN report views, or add a new GUI/UX flow.

## Compatibility and verification

Old P26 schema-1 records are not migrated. Existing P26 producer JSON/hashes and
general `ComparisonContract`, `ComparisonResult`, `MetricDifference` and general
`PerformanceComparison` JSON retain their previous bytes/semantics. The new
policy has a distinct registered contract format; old readers reject that
contract rather than silently interpret it as the general v1 policy.

Focused tests in `tests/test_advisor_p26_comparison.py` use actual synthetic CPU
P26 inference for positive multiclass/binary accuracy and macro-F1 comparisons,
then prohibit any model forward during comparison. They cover signed/zero-base
arithmetic, stream/pair/order/count/class/protocol differences, run/artifact reuse,
missing and declared evidence, separate checkpoint links, policy restrictions,
immutable JSON and exact previous general-contract/result canonical bytes.
Existing 3B.4A/B, producer, native-evidence and TR/EN presentation regressions
remain separate checks. No training or performance benchmark is required.
