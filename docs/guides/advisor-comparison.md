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

V1 supports native P01 resident-forward P50/P95/P99 and the scalar identities
listed below, when all required native capture conditions are present. Metric,
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

## Descriptive differences (3B.4B)

`compare_performance(left_resolver, left_id, right_resolver, right_id,
contract=policy)` reruns `compare_records`; it never accepts a caller's
COMPARABLE assertion. A `PerformanceComparison` contains the complete contract,
eligibility result and, only for an eligible safe calculation, a
`MetricDifference`. Both resolved metrics retain their native EvidenceRefs,
unit, protocol, scope and descriptive limitations. Frozen versioned contracts
use the existing deterministic JSON/hash utilities. Deserialization checks
integrity/arithmetic but is not fresh source certification.

Supported numerical semantics are resident-forward P50/P95/P99 (ms, lower),
completed resident images/s (higher), baseline/current/sampled-peak process RSS
(byte, lower), logical tensor/artifact bytes (lower), and test accuracy/macro F1
(fraction, higher). Lifetime HWM, pipeline and sweep-point metrics remain
unsupported for this comparison policy. Each metric requires its own contract
and pair of records; results are not aggregated into a model-wide verdict.

For non-P01 records, capture `run_id` binds to native `result_id` or
`evaluation_id`; `sample_sha256` hashes the complete native `raw_evidence` or
classification `raw` object. Source capture identities must be distinct;
identical throughput windows/RSS observations are rejected as reused captures.
The same strict native model/input/environment/configuration identities remain
required. Overlapping native configuration declarations must agree. No caller
scope repairs are made. Sample counts come from native sample/observation/batch
counts, never a caller declaration; tail minimums apply to quantiles only.
Logical/artifact byte captures currently lack an observation count and remain
insufficient rather than inventing repeated evidence. Existing MNIST records
are not upgraded to eligible captures. No producer/collector is added.
P26 schema 1 cannot carry the required capture metadata and model/input
identities. [3B.4C.1 producer capture](p26-native-evidence.md) adds opt-in schema 2
with observed model-state/checkpoint and ordered input/target content evidence.
This is a separate partial-evidence contract, not the complete comparison
capture above. P26 end-to-end comparisons remain INSUFFICIENT_EVIDENCE. Quality
arithmetic does not bypass that eligibility gate; comparison-policy integration
and verified split/preprocessing/origin associations remain future work.

`difference = candidate - baseline` is a signed absolute-unit difference.
`percent_change = 100 * difference / baseline`, or null for zero baseline.
For quality fractions, `percentage_point_change = 100 * difference` is separate:
0.80 to 0.90 is +10 percentage points and +12.5 percent relative change.
Decimal arithmetic with fixed precision avoids ordinary binary cancellation;
nonfinite outputs are withheld. Negative/nonfinite values, wrong units/domains,
metric identities or measurement scopes cannot produce a numerical comparison.
A zero baseline still permits an absolute difference, never a relative percent.

For 10 ms to 7 ms the observed latency decrease is 3 ms / 30 percent.
`metric_effect=favorable` describes only that metric under the recorded allowed
axes. Batch/precision/model permissions are retained in the contract and warning
codes; they do not demonstrate quality preservation, per-request equivalence or
causal optimization success. Multiple metric results may show a speed/quality
tradeoff; no overall success is inferred.

Every result explicitly retains missing paired quality acceptance, independent
acceptance and uncertainty evidence. Statistical independence, significance,
causality and optimization success remain uncertified. P29 independent
validation remains incomplete. No VERIFIED_IMPROVEMENT is produced. Existing
canonical Advisor reports, Turkish/English presentation and serializers are
unchanged. No benchmark, model inference or training is performed.

Focused checks (synthetic JSON samples, no workloads):

```text
python -B -m unittest discover -s tests -p test_advisor_comparison.py -v
```
