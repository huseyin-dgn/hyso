# Comparison eligibility — 3B.4A

`ComparisonContract` v1 and `compare_records` decide whether **recorded** conditions
are comparable. They never change models/weights, execute measurements, calculate
a performance difference, certify a speedup or claim statistical independence.
The existing `compare_evidence`, reporting and native serializers are unchanged.

```python
from hyso.advisor import ComparisonContract, NativeMetricResolver, compare_records

# Existing validated UnifiedAnalysisReports, not model objects.
left = NativeMetricResolver(old_report)
right = NativeMetricResolver(new_report)
policy = ComparisonContract(allowed_changes=("model", "batch"))
result = compare_records(left, old_record_id, right, new_record_id, contract=policy)
```

Only COMPARABLE / INSUFFICIENT_EVIDENCE / INCOMPATIBLE are outcomes. Structured
reasons carry `code`, `field`, `side`. Results retain source EvidenceRefs and the
contract hash; all contracts support existing immutable/versioned JSON utilities.
Invalid API arguments/schema versions raise AdvisorError rather than minting an
extra comparison outcome. Invalid native reports fail the existing resolver's
validation boundary.

V1 deliberately supports native P01 resident-forward P50/P95/P99 only. Metric,
quantile, unit, scope and native validator come from the existing resolver. Both
native eligibility and contract sample/tail requirements must pass. Same record,
payload/artifact or underlying source run reuse is incompatible, even with a new
analysis ID/import timestamp. Distinct captures do not prove IID timing samples,
independent experimental confirmation, significance or causal improvement.

## Producer capture metadata

Old results generally lack critical dataset/checkpoint/hardware associations and
therefore remain INSUFFICIENT_EVIDENCE. No EvidenceScope declaration is promoted
or used to repair an absent native capture. Producers may supply the following
additional metadata **inside the hashed native payload**, at
`provenance.comparison_capture`; this reader does not add it to old artifacts:

| Field | Required meaning |
|---|---|
| format / schema_version | hyso.native_comparison_capture / integer 1 |
| authority | native_capture; declared/unknown is insufficient |
| run_id | Must equal raw_evidence.source_result_id |
| sample_sha256 | Canonical SHA-256 of retained latency_samples_ms |
| hardware_id | Hardware identity SHA-256; a CPU class name is insufficient |
| checkpoint_sha256 | Actual checkpoint identity, not an alias of the model-state hash |
| dataset_id / split / cohort_sha256 | Explicit captured data/cohort identities |
| preprocessing_sha256 / input_family_sha256 | Captured preprocessing and logical input-family identity |
| workload_id / clock_id | Captured workload and clock-domain identity |

Hash fields are lowercase SHA-256; identifiers cannot be unknown/not-collected.
Metadata is checked for completeness, source-run/sample binding and contradictions
with supplied declarations. Model-state/structure, device/backend, runtime,
precision, batch, input shape/dtype/layout/content and protocol/configuration
are read separately from the native payload. Original bindings remain unchanged.

This producer schema is **not a signed attestation or capture implementation**.
The native_capture label alone is insufficient: the structure and run/sample
bindings must validate. Hashes establish internal integrity, not authenticity of
hardware, data files or producer statements. Trust in the original producer and
actual capture remains external, as with existing native result artifacts.
Supplying caller declarations in place of producer-captured metadata is not a
supported path. No collector or automatic metadata migration is implemented here.

By default all critical conditions must match. Explicit `model` permission covers
model-state, structural descriptor and checkpoint differences. `batch` permits
batch cardinality/content-hash differences while requiring identical nonbatch
dimensions and captured input-family/cohort/preprocessing identities. `precision`
can permit precision/dtype/autocast differences; unrelated settings remain fixed.
Hardware, backend, data identities, runtime and protocol changes are never silently
allowed. Missing knowledge yields insufficient evidence; known disallowed or
incoherent conditions yield incompatible, even if another field is missing.

COMPARABLE means suitable recorded conditions **for the explicitly allowed axes**;
it does not imply per-request latency equivalence across batch sizes, quality
preservation or deployment acceptance. Numerical effects belong to 3B.4B.

Focused checks (synthetic JSON samples, no workloads):

```text
python -B -m unittest discover -s tests -p test_advisor_comparison.py -v
```
