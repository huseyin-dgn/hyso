# Advisor data contract — Stage 3B.1

`hyso.advisor` supplies immutable, typed request/report data and static interval
validation. It has **no `advise()`**, evidence resolver, eligibility engine,
optimization rules, scoring, executor, hardware query or model invocation.
Importing the package adds no runtime actions beyond the existing `hyso` import.

## Requests

One `AdvisorRequest` supports three entry styles. They do not select separate
engines. `mode` is a derived presentation label, not a serialized switch.

```python
from hyso.advisor import (
    AdvisorRequest, EvaluationProtocol, Metric, MetricScope,
    ResourceBudget, UserGoal, validate_request,
)

automatic = AdvisorRequest()
priority = AdvisorRequest(priority="speed")

scope = MetricScope(
    phase="inference", device="cpu", backend="torch", precision="fp32",
    batch=1, input_shape=(1, 1, 28, 28),
    workload_id="resident-classification", model_id="my-model",
)
protocol = EvaluationProtocol(
    protocol_id="resident-p95", version=1,
    description="Resident synchronous host forward; linear sample quantile",
    sample_policy="1000 captured samples; independence not asserted",
)
latency = UserGoal(
    goal_id="latency", metric=Metric.RESIDENT_FORWARD_P95,
    operator="<=", target=1.0, unit="ms", constraint="hard",
    scope=scope, protocol=protocol,
)
advanced = AdvisorRequest(
    goals=[latency], priority="memory",
    budgets=[ResourceBudget(
        budget_id="rss", purpose="deployment", domain="process_rss",
        limit=1, unit="GiB", allowed_devices=["cpu"],
    )],
)
validation = validate_request(advanced)
assert validation.goals[0].observed == "unknown"
```

`speed`, `memory`, `preserve_accuracy` and `balanced` only identify a preference.
They do not add targets, weights, score functions, hard limits or guarantees.
Goals and budgets can coexist with any priority. A budget-only request is
advanced. Empty allowed-device/backend/precision tuples mean **unspecified**,
not discovered hardware, an authorization or a capability declaration.

`UserGoal` separates metric identity, operator, finite nonnegative target, unit,
hard/soft intent, measured/estimated basis, scope, protocol and baseline ID.
Zero is a mathematical target, not evidence that zero latency or memory is
achievable. Negative values, booleans, nonfinite values and unknown metrics are
rejected. Tiny positive targets remain valid; no feasibility estimate is made.

Supported metric identities distinguish resident P95 from E2E P95, inference
from training throughput, camera frames/s from images/s, accuracy fraction from
percentage-point drop, process RSS from allocated/reserved CUDA memory and
logical tensor bytes. Power and energy per inference have separate identities.
Accuracy accepts only `fraction` in [0,1]. Drop accepts only `percentage_point`
in [0,100] and requires an explicit baseline ID. No percent/fraction or
FPS/images/s conversion is inferred. Seconds/milliseconds and byte/GB/GiB have
explicit exact scales for static interval comparisons only.

Missing scope/protocol are stored as null. Incomplete scope is `needs_scope`:
all goals need workload identity; model metrics also need model identity.
Accuracy needs dataset/split/preprocessing plus evaluator, averaging and sample
policy. Other metrics need device and sample policy; resident-forward and
training timing/throughput also need backend, precision, batch and input shape.
These are minimum **request completeness** requirements, not evidence
eligibility or proof that protocol descriptions are executable. 3B.2 must inspect
the actual native statistic, sample and protocol semantics. No field is filled
from other goals or from a model-name guess.

## Budgets

`ResourceBudget` distinguishes deployment from experiment. Supported domains
are process RSS, CUDA allocated/reserved, physical device memory, logical tensor
bytes, wall time, energy, power and trial count. Limits must be positive and
finite; trial count is an integer. Memory units are `byte`, decimal `GB` and
binary `GiB`. A time window is optional and measured in seconds.

Applicability and enforcement remain explicitly not evaluated, unavailable or
unsupported/advisory as appropriate. No `enforced` claim is supported in 3B.1.
A budget does not start a process, reserve physical memory, set an allocator
fraction, change a device/precision or prove that a runtime limit is enforceable.
Static interval validation currently evaluates goals, not goal/budget interactions.

## References, findings and independent status axes

`EvidenceRef` contains source report ID, record ID, exact native metric path,
artifact and canonical payload SHA-256, domain, unit, source status and per-field
`EvidenceBinding` (`native`, `declared`, `missing`). Optional sample, point and
protocol references remain opaque strings. Paths support dotted native field
names and numeric list indices. They are not Python expressions and are never
followed by this package.

Reference validation is **syntactic**. Hash-shaped values do not prove existence,
authenticity, association, compatibility or independent experiments. Binding
labels are retained exactly; the contract never upgrades declared to native.
Source status, reference existence and field semantics require 3B.2 validation.

`FindingKind` names OBSERVATION, HYPOTHESIS, DIAGNOSIS, MEASURE_NEXT,
EXPERIMENT_CANDIDATE and VERIFIED_IMPROVEMENT. The first five may be stored as
caller-supplied data; no rules produce them. VERIFIED_IMPROVEMENT is reserved:
construction/deserialization raises `comparison_contract_unavailable` in 3B.1.
`verification="verified"` is also rejected. A category or free-text statement
must not be treated as a verification certificate by consumers.

`GoalEvaluation` keeps these axes separate:

| Axis | States |
|---|---|
| Input validity | valid / invalid / needs_scope |
| Evidence support | supported / insufficient_evidence / unsupported_capability |
| Observed goal | observed_met / observed_missed / unknown |
| Feasibility | demonstrated_in_scope / not_demonstrated / unknown |
| Cross-goal consistency (`RequestValidation`) | no_conflict_identified / potentially_conflicting / logically_inconsistent |

Evidence support defaults to null, meaning **not evaluated**, rather than
asserting insufficiency without an assessment. Observed goal and feasibility
default to unknown. There is no implicit zero, score or calibrated confidence.
Invalid goal construction raises `AdvisorError`; the invalid validity state is
available for intake rejection records and cannot be inserted as an accepted
invalid UserGoal. Explicit assessment values can be serialized only as caller
claims, with references where required; they are not engine certifications.

`validate_request()` computes only request completeness and empty hard-target
intervals for the same complete metric/scope/protocol/basis/baseline. It handles
inclusive/strict bounds and equality, using exact decimal rational arithmetic.
Soft preferences and different domains do not form hard contradictions. For
example P95 <=1 ms and >=2 ms is logically inconsistent; high throughput and low
RSS is not mathematically contradictory without further assumptions. No conflict
identified is not evidence of feasibility. Partial contexts are not paired.

## Separate AdvisorReport schema1

Every contract uses `format="hyso.advisor.<TypeName>"`, `schema_version=1`,
strict field sets and a `content_sha256` over canonical JSON excluding that hash.
Nested contracts carry their own version and hash. SHA-256 establishes integrity
and deterministic identity, not authenticity; callers can recompute hashes.

`AdvisorReport` references the source analysis report ID and canonical hash of
its **entire** `to_dict()` (including its report_id), using UTF-8, sorted keys,
compact separators, non-ASCII retained and finite JSON. This is different from
the source report's own ID calculation, which excludes report_id. This stage
accepts these references from the caller; it does not open the source report.

The request is embedded. Its derived `request_id="sha256:<content_sha256>"` and
`request_sha256` are persisted and verified on reopen. `AdvisorReport.report_id`
is its content hash. Findings, goal evaluations, evidence limitations, risks and
rule ID/version provenance are explicit tuples. Optional `request_validation`
embeds the independently reproducible static result. Defaults are empty lists,
null evidence status and `evaluation_status="not_evaluated"`; they do not mean
all goals passed. `unavailable` is also supported. Explicit external assessments
require `evaluation_status="caller_supplied"`. All reports retain
`assessment_authority="caller_supplied_not_verified"`.

Goal IDs, source report references and finding rule versions must agree within
the document. A report does not independently verify any external reference.
`RuleVersion` records provenance only; no rule registry or plugin import exists.

```python
import hashlib
import json
from hyso.advisor import AdvisorReport

# analysis_report is an existing validated UnifiedAnalysisReport.
source = analysis_report.to_dict()
canonical = json.dumps(source, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False)
result = AdvisorReport(
    source_report_id=source["report_id"],
    source_report_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    request=advanced,
    request_validation=validate_request(advanced),
)
result.to_json("advisor-contract.json")  # Explicit no-clobber export.
assert AdvisorReport.from_json("advisor-contract.json") == result
```

Frozen dataclasses normalize input lists to tuples; to_dict returns detached
plain JSON. JSON is limited to 2 MiB, nesting to 24 contract/list levels,
collections to 1024 items and text fields to 4096 characters. Unknown versions,
fields, types, duplicate JSON keys, invalid UTF-8, malformed/nonfinite JSON and
tampered derived identities are rejected. Exports refuse existing files. Paths
are explicit only; export is not an atomic replacement protocol.

## Compatibility and subsequent gates

No `hyso.analysis` source, native JSON/CSV serializer, Trainer, BenchmarkRunner
or ExperimentManager schema is changed. `UnifiedAnalysisReport` schema1 keeps
its empty not-implemented recommendations; compare_evidence is not relaxed.
P18/P29/P31 acceptance limits and B4 OPEN are not reclassified or cleared.

3B.2 needs a closed native metric resolver (including quantile, unit and domain
checks), source-reference verification, per-claim eligibility and preservation
of native/declared/missing associations. Matching report scope groups are not
proof of compatibility. Report requests contain desired scope, not measured
native provenance. Existing old artifacts cannot be repaired with new claims.

Future batch/precision/model comparisons need a separately versioned
ComparisonContract defining intentional changes, fixed conditions, native
bindings, discovery versus independent validation and quality/resource acceptance.
This contract stage cannot mint VERIFIED results around the existing equality
comparison guard. A validated data object is not a recommendation engine.

NOT IMPLEMENTED: advise(), metric resolution, evidence eligibility, optimization
rules, benefit estimation, scoring/confidence, Markdown generation, hardware
enforcement, measurement plans/execution or independent improvement verification.

Focused check (no pytest conftest or workload fixtures):

```text
python -B -m unittest discover -s tests -p test_advisor_contracts.py -v
```
