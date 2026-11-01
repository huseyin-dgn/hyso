# Native observations and evidence eligibility — Stage 3B.2

`advise(report, request=None)` consumes a validated `UnifiedAnalysisReport` and
returns an `AdvisorReport`. It performs deterministic, source-local description
only. It does not run a model, import a source-declared factory, query a device,
benchmark, train, optimize or enforce resource limits. Native validators may
perform bounded CPU arithmetic on retained evidence. The existing eager `hyso`
package imports are unchanged; qualified names inside payloads are never imported.

```python
from hyso.analysis import UnifiedAnalysisReport
from hyso.advisor import (
    AdvisorReport, AdvisorRequest, NativeMetric, NativeMetricResolver,
    advise, validate_advisor_report,
)

source = UnifiedAnalysisReport.from_json("existing-analysis.json")
advisor = advise(source)  # Automatic; no invented target.
memory_first = advise(source, request=AdvisorRequest(priority="memory"))
advisor.to_json("advisor.json")       # Explicit export; refuses existing files.
advisor.to_markdown("advisor.md")    # Same object; no analysis on render.
reopened = AdvisorReport.from_json("advisor.json")
validate_advisor_report(reopened, source)

resolver = NativeMetricResolver(source)
resolved = resolver.resolve("exact-record-sha256", NativeMetric.FORWARD_P95)
# supported + ResolvedMetric, or unsupported_metric/insufficient_evidence + null.
```

Priority changes only finding order. `speed` puts timing/rate first; `memory`
puts RSS/logical/artifact observations first; `preserve_accuracy` puts quality
first. `balanced` retains the default source order without a score. Finding IDs,
values, evidence, risks and verification status do not change. Request/report
hashes naturally change when priority or serialized order changes.

Stage 3B.3A adds [record-local goal evaluation](advisor-goals.md) for supported
explicit goals. Historical observations_only reports keep the 3B.2 behavior.
Deployment/experiment resource budgets remain unevaluated and unenforced.

## Closed resolver v1

| Native feature/format | Supported metrics | Semantics |
|---|---|---|
| P01 latency_analysis | forward P50/P95/P99 | Explicit native quantile plus resident-tail protocol, method, scope and eligibility; ms |
| P02 throughput | completed images/s | Total completed images / total completion-window wall time; includes batch identity; never camera FPS |
| P03 pipeline | E2E P50/P95/P99 | Serial bounded read-start to post-completion; not forward timing; one ID may contain a batch |
| P08 precision sweep | Per-point mean forward ms/batch and completed images/s | Exact point ID required; point precision, shape/batch, protocol and source status retained |
| P10 memory | Baseline/current RSS, sampled peak RSS, current lifetime HWM | Byte quantities remain process-scoped; sampled peak and lifetime HWM are distinct |
| P11 footprint | Parameter/buffer/combined logical bytes; serialized artifact bytes | Unique tensor-object logical accounting is separate from serialized size and physical memory |
| P26 classification | Test accuracy fraction, test macro F1 fraction | Test split and classification identity required; recorded cohort and averaging retained |

Other formats/metrics explicitly return `unsupported_metric`; missing native
semantics or inadmissible statuses return `insufficient_evidence`. No unit
conversion, percentile substitution, point fallback or missing-to-zero fill is
performed. The legacy raw BenchmarkResult format is not a resolver-v1 format;
P01 requires the native typed latency_analysis quantile record.

`measured` and native-derived values can be descriptive observations. `estimated`,
unsupported, invalid, not-run, unknown, null and insufficient-sample values cannot
become measured observations. P03 P99 may therefore be absent even when P95 exists.

P08's old scalar projection may tag an images/s value as `latency`. Resolver v1
does not change that projection or accepted source hashes. It reads the validated
point's exact metric, unit, completion semantics and protocol, returns `throughput`
for that metric and records the projection disagreement as a limitation. Native
point precision overrides no record binding: original scope and native/declared/
missing labels remain intact, while point context is recorded separately. The
configuration's sweep default precision is not the effective point precision.

## Source and reference validation

`NativeMetricResolver` revalidates the typed source report, its schema, report
identity, record identities, native constructors, payload hashes, projections and
derived fields before retaining immutable JSON snapshots. Caller mutation of
the original object or returned detached dictionaries cannot change the snapshot.

`ResolvedMetric` retains the typed native identity, exact scalar, full EvidenceRef,
original record scope JSON, native protocol JSON, native metric/point context,
measurement origin and limitations. These context JSON strings are canonical,
detached objects exposed through `record_scope`, `protocol` and `native_context`.

`check_reference(ref, metric, required_native=(), artifact_bytes=None)` regenerates
the native resolution and requires the **entire** EvidenceRef to match: report/
record ID, exact native field path, artifact/payload SHA, domain/unit/status,
binding authorities and point/protocol references. `check_metric(resolved)` also
checks the scalar, context, origin, limitations and resolver version. Constructing
or roundtripping either object alone is not evidence validation.

Artifact hashes are bound to the validated source report. Original artifact paths
are not followed, and original bytes are not claimed to have been reread. If the
caller explicitly supplies `artifact_bytes`, their SHA and parsed canonical
payload SHA are checked as well. Hash integrity does not prove authenticity;
rehashed fabricated records cannot become independently trusted merely by being
well formed. Establishing trust in the original capture remains external.

A local descriptive observation does not require every absent provenance field.
`required_native` can require specified associations for a stronger claim; missing
or declared authorities do not satisfy it. No authority is promoted from caller
claims or from context summaries.

`compare(left_ref, left_metric, right_ref, right_metric)` validates both references,
requires matching metric semantics/context and then calls the unchanged 3A
`compare_evidence`. Insufficient/incompatible native associations remain blocked.
This returns compatibility eligibility only, never independent significance or a
verified effect. Different model/precision/batch comparisons still need a future
versioned ComparisonContract; equality checks are not weakened to allow them.

## Reports and compatibility

Generated OBSERVATION entries are `NativeObservation`, a closed, registered
subclass of the unchanged-field `Finding` contract. It adds `resolved_metric`, so
JSON retains structured numbers/scopes/protocols rather than relying on parsing
human statements. Its evidence must equal the resolved metric's exact reference,
and its verification is `not_verified`. The other generated kind is MEASURE_NEXT,
used only for specific conditional evidence gaps. Neither kind authorizes a run.

AdvisorReport accepts the additive `observations_only` status and records
`assessment_authority="native_observation_rules_v1; not_independent_verification"`.
All 3B.1 formats, fields, defaults and caller-supplied report roundtrips remain
supported. No 3A formats or serialized analysis data are changed. Old 3B.1-only
readers do not understand the new NativeObservation format and should reject it;
there is no silent downgrade to an unstructured verified claim.

JSON roundtrip validates contract versions, structure and hashes. To check that
an engine report is actually derived from the stated source, call
`validate_advisor_report(result, source)`: it reproduces deterministic output and
compares the complete document. A hash-shaped reference alone is insufficient.
`to_markdown()` renders the same structured object without resolving evidence
again. Source strings are escaped and exports are explicit and no-clobber.

`evidence_status=supported` means at least one local observation is supported; it
does not certify all features, complete provenance or joint deployment goals.
Coverage gaps and unavailable native statuses remain explicit. P05 synthetic
recipe input is not called a real dataset image. P18 qualified roof limitations,
P29 noncertified improvement, P31 noncertified 7200-second acceptance and B4 OPEN
remain in risks. No raw CPU self-time or utilization becomes a causal diagnosis.

NOT IMPLEMENTED: deployment acceptance, optimization/experiment-candidate rules,
causal hypotheses/diagnosis, benefit estimates, scores/confidence percentages,
verified improvement, hardware enforcement, model mutation or execution.

## Focused validation and retained MNIST acceptance

```text
python -B -m unittest discover -s tests -p "test_advisor*.py" -v
python -B tools/integration/advisor_readonly_acceptance.py --source <existing-analysis.json> --output <new-directory>
```

The focused fixture copies six accepted native payloads without executing their
workloads. P26 unit tests use fixed-label confusion arithmetic, not model inference.
Actual MNIST acceptance reads the existing full report. The standalone acceptance
script blocks model forwards, Trainer/Benchmark calls, device APIs and source
factory/model imports, and records zero forbidden calls. It writes only to an
explicit, new output directory: `mnist-advisor.json`, `mnist-advisor.md` and
`acceptance.json`. Existing source bytes are hashed before/after. No training,
new inference, GPU initialization or stability run is part of these checks.
