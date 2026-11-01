# Bound structure and P26 quality comparison (3B.4E.2)

`hyso.advisor.compare_p26_model_profiles` reads two **distinct native P26 schema-3
evaluations** and compares their bound structure and recorded quality. No model,
checkpoint, training, benchmark or device execution occurs during comparison.

```python
from hyso.advisor import (
    P26ModelProfileComparisonContract, UnifiedModelComparison,
    compare_p26_model_profiles,
)

# Resolvers already contain validated analysis reports. Source evaluations use
# evaluate_classification(..., capture_evidence=True, capture_structure=True).
result = compare_p26_model_profiles(
    baseline_resolver, baseline_record_id,
    candidate_resolver, candidate_record_id,
    contract=P26ModelProfileComparisonContract(allowed_changes=('model',)),
)
parameters = next(d for d in result.structure_comparison.summary_differences
                  if d.metric == 'parameter_elements')
accuracy = result.accuracy_comparison
if accuracy.difference is not None:
    pp = accuracy.difference.percentage_point_change
    relative_percent = accuracy.difference.percent_change
result.to_json('comparison.json')  # exclusive; never overwrites
restored = UnifiedModelComparison.from_json('comparison.json')
```

## Versioned policies

* Result: `hyso.unified_model_comparison`, schema **1**.
* Contract: `hyso.advisor.P26ModelProfileComparisonContract`, schema 1,
  policy `p26_bound_model_profiles_v1`.
* Per-metric contract: `hyso.advisor.P26BoundQualityComparisonContract`, schema 1,
  policy `p26_bound_exact_observed_stream_v1`, schema-3 evaluations only.

The per-metric contract also works explicitly with `compare_records` and
`compare_performance`, for `TEST_ACCURACY` or `TEST_MACRO_F1`. The old schema-2
`P26QualityComparisonContract` / `p26_exact_observed_stream_v1`, its serialized
bytes/hash meaning, and general v1 latency/throughput/RSS rules are unchanged.
Old readers do not silently reinterpret the new policy format.

Only `allowed_changes=()` or `('model',)` and positive integer `min_samples` are
accepted. The default grants no model-change permission. Neither equal type/state
hashes nor equal structure hashes establish equal forward behavior; quality
requires explicit model-change permission even when these hashes match.

## Independent sections and identities

`result.baseline` and `.candidate` are immutable `BoundEvaluationReference`
objects. They preserve report ID/canonical SHA, record ID, artifact/native payload
SHA, evaluation UUID/source SHA, structure SHA, model-state method/SHA, optional
checkpoint artifact status/SHA, and native binding method/scope. Checkpoint SHA
never falls back to state or structure SHA. Same architecture with different
weights can have equal structure hashes and different state hashes. A declared
`source_id` is not a run identity.

Reused record, evaluation UUID, artifact or canonical payload is rejected by the
unified API. The per-metric gate reports `INCOMPATIBLE` for capture reuse. Distinct
evaluation IDs do not certify statistical independence.

* Structure has separate `COMPARABLE` eligibility after both native bindings are
  validated. `.structure_comparison` is the existing immutable D.2 object:
  parameter/trainable/frozen totals, separate known subtotals, module/category/
  dtype counts, literal registered paths, supported configuration differences,
  and shared/tied/storage relation differences. Coverage, evidence gaps and
  limitations remain intact. Unsupported totals have no exact difference.
* Accuracy and macro-F1 each retain their own `PerformanceComparison`, policy,
  eligibility/reasons, original `EvidenceRef`s, source contexts, missing evidence
  and limitations. Each wire section states its evidence level and scope.
  Native model-state/count, ordered input/target/pair hashes, algorithm/version/
  scope, class mapping, metric definition, full protocol, batch partition and
  recorded runtime must satisfy the existing exact observed-stream gate.
* Known input/target/class/protocol/batch/runtime mismatches are `INCOMPATIBLE`.
  Missing observable quality evidence or an unmet sample minimum is
  `INSUFFICIENT_EVIDENCE`. These cases have **no numerical quality difference**;
  valid structure results remain available. Corrupt native bindings/snapshots
  are rejected by readers rather than converted into missing evidence.

Quality differences are candidate minus baseline. Fraction change times 100 is
percentage-point change; relative percent divides by baseline separately and is
null for baseline zero. There is no global PASS/FAIL, winner, causality,
significance, `VERIFIED_IMPROVEMENT` certificate or completed P29 acceptance.

## Canonical export and verification

The result is frozen; `to_dict()` returns detached data. Canonical JSON and
`content_sha256` are deterministic for identical retained sources and policy.
Comparison creates no timestamps or random IDs. New producer evaluations retain
their own native UUIDs and consequently have different content identities.

Independently readable JSON retains source reports in a catalog keyed by report
ID; a shared report is stored once. Keeping the complete report preserves original
identities and metric references, so unrelated records in that report can add
size. The aggregate limit is 128 MiB, without truncation or source-path fallback.
Existing snapshot and D.2 comparison size/entry limits also apply.

Snapshots live only in retained native evaluation payloads. The structure wire
section contains the D.2 comparison **projection without `source_snapshots`** and
its original hash; that projection is not a standalone D.2 JSON artifact.
`.structure_comparison` restores the complete validated D.2 object, source
snapshots and unchanged hash. Per-metric comparisons retain small native source
contexts without embedding snapshots again.

The reader validates source reports, records, native P26 evidence, bindings and
snapshots, rebuilds the resolvers, and reruns D.2 comparison plus both quality
gates. It compares the complete rederived result: source identities, references,
scope, eligibility, arithmetic, policy and limitations. Rehashing an envelope
cannot validate substituted sources or invented differences. Duplicate JSON keys,
unsupported versions and extra fields are rejected.

This establishes internal consistency, not cryptographic live-producer
authenticity. A wholly fabricated, self-consistent imported capture cannot be
authenticated from its own hashes. Original producer trust remains external.
Structure describes the isolated forward receiver at recorded boundaries, not
certified original-object behavior or an operator graph. Dataset origin, split
membership, preprocessing implementation, generalization, uncertainty and
independent acceptance remain unverified. Latency, throughput, RSS and P11 are
not joined, and no Architecture Audit or GUI/report redesign is introduced.

## Acceptance example

`tests/test_advisor_model_profiles.py` runs two owned tiny CNNs on four synthetic
CPU samples. Parameters are 18 to 10 (-8, approximately -44.4444%); trainable
counts 14 to 8; frozen 4 to 2. The candidate adds an Identity module. Accuracy is
0.75 to 1 (+25 percentage points, +33.3333% relative); macro-F1 is 11/15 to 1
(+26.6667 percentage points, +36.3636% relative). These are separate observations
with no causal claim.

The full chain covers real producer -> schema 3 -> resolver -> structure and
quality comparisons -> unified JSON -> reader, using both shared and separate
source reports. Tests also cover partial results, reuse, same-structure/different-
state models, checkpoint identity, rehashed corruption, immutability, RNG/model/
buffer preservation, and prohibitions on live execution during comparison.
Schema-1/2, D.1/D.2, general comparison and TR/EN regressions remain separate checks.
