# Two-model comparison of retained evidence

`hyso.analysis.ModelComparisonSource`, `compare_models` and `ModelComparisonResult`
unify existing read-only comparison APIs. Exactly two distinct model labels and
an explicit feature selection are required. The existing FeaturePlanRequest
two-model/known-ID/duplicate-selection validation remains authoritative. Selecting
a feature does not compare dependencies or start missing tests.

## Source and metric coverage

Sources accept an existing UnifiedAnalysisReport, or use
`ModelComparisonSource.from_execution(result, model_id)` for a retained Stage-5
result. No live nn.Module, arbitrary factory, checkpoint loader or measurement
callback is accepted. Model labels are caller associations, not state/checkpoint
hashes. Wrong record labels are rejected. A unique matching record is selected;
multiple captures require `records={"P01": exact_record_id}`. There is no implicit
latest-result policy. Explicit record mappings can associate unlabeled legacy
records, without promoting the label to native evidence.

| Requested feature | Existing comparison authority / supported semantics |
|---|---|
| P01 | compare_performance; P50/P95/P99, ms, lower is favorable |
| P02 | compare_performance; completed images/s, higher is favorable |
| P10 | compare_performance; current/baseline/sampled-peak **process RSS**, lower |
| P11 | compare_performance; separate parameter/buffer/union logical bytes and serialized artifact bytes, lower |
| P26 | Existing exact observed-stream schema-2 or bound schema-3 accuracy/macro-F1 policies, higher |
| Optional structure | compare_model_structures or compare_p26_model_profiles; registered counts/sizes, separate from P11 |
| Other P features | NOT_SUPPORTED_IN_COMPARISON; no numerical difference or automatic execution |

An adapter/semantic entry is not evidence eligibility. Existing general
compare_performance gates require complete, correctly bound native producer
comparison_capture metadata. Default sample requirements remain 1000/10; no
threshold is relaxed here, including general scalar policies. Ordinary Stage-5
captures do not supply all of that metadata. P10 snapshots and ordinary P11
footprints commonly remain INSUFFICIENT_EVIDENCE even when individual values are
observable. This API does not append synthetic capture metadata, invent sample
counts or turn caller dataset/hardware labels into producer proof. A future
native collector/specific validated scalar policy is needed before claiming
broader production comparability.

Native gates check state/structure/checkpoint identities, dataset/cohort/split,
preprocessing and input hashes, shape/batch/layout, device/precision, recorded
hardware/backend/runtime, clock, protocol and configuration as required by the
particular policy. The only allowed variable axis is model change, which is
explicit in the common policy; batch/precision/protocol changes are not permitted.
Known mismatches remain INCOMPATIBLE even when other metadata is missing.
Missing facts are insufficient evidence, never equality of two unknown values.

Optional original BenchmarkResults are passed to existing compare_results only
when P01 is selected. Run ID, samples, source hash and provenance must bind to the
selected latency record. Compatibility diagnostics do not replace the stricter
performance gate. Existing memory eligibility and physical-host/load/cache/
thermals uncertainty are retained; no CUDA runtime calls are made while reading.

P10 process RSS is never labeled model-exclusive memory. P11 logical bytes,
serialized artifact size and P09 GPU allocator bytes remain separate domains.
No P09 numerical adapter is connected here. Lifetime HWM is not a supported
performance-difference metric. Bound P26 structure exposes registered parameter
counts and logical-size differences without relabeling them as a measured P11
or GPU-memory result.

## Example and interpretation

The README and Python API guide define a working `compare_saved_models` function
for two previously exported analysis reports. To select fewer features, call the
public API directly, e.g. `compare_models(source_a, source_b, features="P26")`.
Structure work happens only with `include_structure=True`: either supply two
already-captured `structure=` snapshots or select two native bound P26 results.
Standalone snapshots have declared model associations. Bound P26 references
retain their original state/structure/checkpoint distinctions and native binding.
No live capture is attempted when snapshots are absent.

Result format is `hyso.two_model_comparison`, schema 1. Raw selected sources,
native contracts, eligibility reasons, references and numerical differences are
retained. Values are candidate B minus baseline A. Percent change uses A as the
denominator; a zero baseline yields null percent and retains the absolute change.
Quality fraction differences also have separate percentage-point changes.
Metric effects are favorable/unfavorable/unchanged observations, never a general
winner, causal explanation, statistical superiority or optimization certificate.

For the retained tiny CPU P26 fixture (four synthetic samples per model, **not a
model-quality or speed claim**), A has accuracy .75 and 18 parameters; B has
accuracy 1 and 10 parameters. An abbreviated result is:

```json
{
  "feature_id": "P26",
  "metric": "test_accuracy",
  "status": "COMPARABLE",
  "difference": {
    "difference": 0.25,
    "percent_change": 33.333333333333336,
    "percentage_point_change": 25.0,
    "preferred_direction": "higher",
    "metric_effect": "favorable"
  },
  "claims": {"global_winner": null}
}
```

This is a labeled projection, not a standalone canonical result. The corresponding
Markdown summary includes separate scope-qualified rows:

| Metric | A | B | B−A | Relative % |
|---|---:|---:|---:|---:|
| Observed accuracy, fraction | .75 | 1 | .25 (25 percentage points) | 33.3333 |
| Bound registered parameter elements | 18 | 10 | -8 | -44.4444 |

`result.to_json(path)` and `result.to_markdown(path, language="tr"/"en")` are
explicit no-clobber exports. `to_display_dict(language=...)` uses the existing
Advisor metric labels, number formatting and per-model presentation. Only selected
records enter those per-model observation views. Comparative explanations keep
native metric direction and ineligibility reasons; P10 explanation always retains
process scope. `claims.global_winner` stays null in both languages.

JSON import revalidates source reports/native readers and reruns the unchanged
comparison algorithms. Altered arithmetic, an injected winner, extra fields or
unsupported versions are rejected even if the outer hash is recomputed. The
aggregate bound is 128 MiB with no truncation or file-path fallback. Structure
JSON is a labeled projection without duplicate source snapshots; its original
hash is retained and `.structure_comparison` exposes the complete typed object.
Internal consistency does not authenticate an independently fabricated capture.

## ExperimentManager and boundaries

Use the existing Manager API explicitly: `get_run(id).to_dict()` and
`get_run(id).read_metrics()` read run records/scalars. Scalars alone cannot
reconstruct native evidence, protocols or dataset/model bindings. For comparison,
read validated report/result JSON retained as artifacts; this adapter never loads
a checkpoint or follows a source-declared Python import.

A benchmark run can store `result.tracking_config()` as its config. In an existing
schema2 store, `manager.link_artifact(comparison_run.id, source_artifact_id,
role="input")` links each original artifact; `comparison_run.add_artifact(path,
kind="model_comparison", ownership="copy", role="output")` attaches the exported
comparison. Run lifecycle and manager.close() remain explicit. No store is
migrated automatically. New stores start at schema1; existing offline migration
rules apply if artifact APIs are needed. Do not reconstruct evidence from an
arbitrary scalar metric name or a claimed run status.

The existing `compare_performance_quality` is deliberately **not called**, because
it performs new P29 inference/quality/memory measurement. Native measurement
engines, old comparison algorithms/policies, checkpoint schemas, 70/70 acceptance,
and forensic/debug files are unchanged. Broader P-feature comparison adapters,
verified producer capture binding for ordinary execution outputs, independent
uncertainty and generalization evidence remain future work.
