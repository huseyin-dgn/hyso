# Unified evidence contract — version 1

`hyso.analysis` implements the pure index/envelope step from
`docs/design/hyso-analysis-optimization-contract.md`, section 6. It does not
implement the future facade, task adapters, optimization rules or Advisor.

Existing Trainer, BenchmarkRunner, native result JSON/CSV, checkpoint format
and ExperimentManager database schemas are unchanged. Native files are inputs,
never rewritten. New formats are `hyso.evidence_record` (record_version=1,
adapter_version=1), `hyso.evidence_index` (schema_version=1) and
`hyso.analysis_report` (schema_version=1).

## Design versus current implementation

The original design already calls for 35 coverage slots, immutable source refs,
separate scopes/clocks, null-with-reason and explicit Manager artifact storage.
Existing typed result constructors already validate their own retained evidence;
Manager already verifies file integrity. Neither previously provided a common
index, fail-closed association check or native unified Markdown/JSON report.

This implementation adds that layer using a closed native-reader list. It does
not expand BenchmarkRunner into a general executor. Measurement invocation,
factory import, model construction/fit/forward, device queries and automatic
configuration changes are absent from the analysis API. Some native validators
perform bounded CPU tensor arithmetic; validation is not free instrumentation.

## Record contract

Each immutable record contains:

- `record_id`: SHA-256 of the canonical record excluding this ID; record/adapter
  versions and original source schema version are separate.
- `source`: original format, allowlisted result type, result/source IDs,
  declared source SHA and independently calculated canonical payload SHA.
- `artifact`: original basename, exact byte SHA-256, size and JSON media type.
  `expected_sha256` is required at import. A matching hash proves integrity
  against that caller-supplied expectation, not capture authenticity.
- `native_payload`: the unchanged parsed native evidence. Native constructors
  revalidate derived values; no CSV or legacy JSON semantics are changed.
- `declared_scope`, `scope`, `bindings`: per-field caller claims versus native
  facts. Bindings are `native`, `declared` or `missing`; caller values cannot
  override contradictory native facts. Native schema omissions stay explicit.
- `origin`: `new`, `reused`, or `none` for unavailable evidence. Stage-2 files
  imported during Stage 3A are **reused**, not new measurements.
- `imported_at_utc`: import time, never substituted for capture time.
  `observed_at_utc` is null when capture UTC is unknown. A timer name alone
  does not establish a clock-domain identity.
- `metrics`: native field references, actual scalar/null value, explicit unit,
  status/reason and measurement domain. Projections are deliberately small;
  unsupported/unprojected native fields remain in `native_payload`.

`EvidenceScope` fields are model ID, native model-state hash, checkpoint file
hash, dataset ID, split, preprocessing ID, device, backend, precision, batch,
input shape, workload ID, protocol ID/hash, phase, input kind, clock-domain ID
and source revision. Different hash encodings are not equated. A model-state
hash is not a checkpoint file hash. A shape or class name is not dataset proof.
Multi-point sweep bundles cannot receive one asserted varying batch/shape/
precision. Select a native point separately before a future point comparison.

All native schema-1 sources retain their own validation/version policies,
including old operation-model versions carried in their payloads. Unknown
native/index/record/report versions fail explicitly. There was no prior public
EvidenceIndex format: no invented v0 migration is claimed. Future migrations
must be explicit, retain original source bytes/hashes and advance adapter version.

Canonical JSON is the admitted interchange. CSV exports remain unchanged:
some are readable projections, others are lossless envelopes. Generic summary
CSV-to-evidence import is intentionally not implemented. Use the corresponding
native canonical JSON; do not fill missing CSV metadata with guesses.

## Scope comparison

`compare_evidence(left, right, domain=...)` performs no measurement or numeric
conversion. A domain must exist in both records. Known disagreement returns
`incompatible`. Missing fields or matching caller-only declarations return
`insufficient_evidence`, not compatible. Only complete native associations can
return `compatible`. Many legacy artifacts lack those associations: this is an
explicit provenance gap, not a reason to fabricate them.

Domains keep process RSS, registered model logical bytes, training-state logical
bytes, CUDA allocator and serialized bytes separate. Latency, sustained
throughput, E2E, cold start, training and inference do not share a denominator
merely because units look convertible. Generated recipe input is not labeled
data. Observed Pareto is not independently verified improvement. Missing is not
measured zero. Neither identical declarations nor an integrity hash certifies
independence, hardware identity or a causal bottleneck.

## Unified report

`AnalysisContext` holds caller-declared model/dataset/training summaries and
reference metadata; it does not promote these into validated measurements.
`EvidenceIndex.analyze(context=...)` returns `UnifiedAnalysisReport` with 35
coverage slots, zero-or-many source records per feature, separate scope groups,
new/reused counts, human explanations, scalar references, known risks and an
explicit empty `recommendations` section marked `not_implemented`.

The report never combines incompatible runs. It performs no automatic ranking,
cross-source sum, speedup or time-to-target inference. P18 qualified roofs,
P29 independent improvement, causal leak and P31 7200-second acceptance are
never certified by this envelope. A native short-run `completed` flag stays
separate from long-run acceptance. B4 remains OPEN.

JSON reports retain the source index and revalidate native payloads and derived
report fields when reopened. Markdown is generated **by HYSo** from the report;
it is a readable view, not a second source of truth. Exports are explicit,
atomic UTF-8 and no-clobber by default. Source/private evidence is not edited.

Manager integration is explicit: export the report, then store as
`kind="report", ownership="copy", role="report"`. New Manager stores start at
schema1; use the existing explicit offline `migrate_store` API for artifact
support, then verify/close/reopen normally. The analysis layer never changes a
Manager store automatically, logs null as a metric or creates a fictitious axis.

## Exact input boundary for a future recommendation engine

The only proposed input is a validated `UnifiedAnalysisReport` schema1, plus
explicit user goals and an externally authorized execution plan. No goals or
rules are implemented here. A future consumer must:

1. Check supported index/record/source/adapter versions and native validation.
2. Reference exact `record_id`, metric field path, artifact/payload SHA, domain,
   unit, status and scope; never accept only a Markdown value.
3. Require `compatible` native association checks for any numerical pairing;
   `declared`, missing, invalid, unsupported and not-run evidence cannot authorize
   a verified improvement or automatic device/configuration change.
4. Preserve scope-specific limitations, budgets, clocks, samples, identity and
   provenance. Independence and expected-versus-measured effect need additional
   evidence; this report does not supply them by implication.
5. Keep any recommendation separate from an authorized mutation/measurement.
   No invocation, retraining, fallback or second trial is authorized by import.

IMPLEMENTED: immutable JSON evidence index, closed native validation, strict
scope comparison, 35-slot report and Markdown/JSON exports. PARTIAL: native
metadata completeness and hand-selected scalar projection coverage for old
sources. NOT IMPLEMENTED: summary CSV import, automatic provenance recovery,
Advisor recommendations, UI/dashboard, optimization executor and new workloads.
