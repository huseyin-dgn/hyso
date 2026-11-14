# Graph v2 provenance in evaluation, comparison and reports

This opt-in adapter presents two different kinds of evidence together. It does
not move a trusted source receipt to the isolated model used for inference.
Existing evaluate_classification, analysis.compare_models and reporting defaults
retain their signatures, schemas and behavior.

## Evaluate with separate source provenance

First explicitly create/load a supported checkpoint using the
[Graph contract v2 API](graph-checkpoint-evidence.md). Then:

```python
from hyso.analysis.graph_evidence import evaluate_graph_classification

association = evaluate_graph_classification(
    loaded_model, sequential_test_loader,
    class_names=["a", "b"], split="test", source_id="my-dataset",
    context={"dataset_id": "my-dataset", "split": "test",
             "preprocessing_id": "my-explicit-transform-v1"},
    expected_checkpoint_sha256=known_checkpoint_sha,
)
data = association.to_dict()
print(data["verification"]["graph_v2"])  # verified only for this live source
print(data["evaluation"]["producer_evidence"]["checkpoint"]["status"])  # missing for supported Graph
print(data["relation"]["status"])  # limited
```

The adapter calls the existing evaluator once, with native evidence enabled.
Metrics, confusion matrix, model-state hash, ordered evaluated stream and original
source kind remain in an unchanged EvaluationResult, available as
`association.evaluation`. `capture_structure=True` optionally retains the existing
schema3 actual-receiver structure capture; default is schema2. Neither option
adds a v2 runtime-contract observation on the receiver.

`source_receipt` is the source's captured v2 snapshot statement.
`verification` is a dynamic, process-local view. It rechecks the existing receipt
on the same original source object, file/state/configuration and runtime contract.
Keep that source model alive. Source mutation/reload or destruction prevents
current verified status; a missing receipt at capture cannot later be upgraded
without a new evaluation capture. Expected identities only constrain a receipt
that already exists; supplying a SHA cannot create one.

The native evaluator checks that the original and actual deepcopy have matching
named-tensor hashes. It does not expose the full v2 runtime signature of the
isolated receiver. Consequently **cross-evidence binding is limited**, even when
the source v2 proof is verified. `isolated_receiver_receipt=not_transferred` and
`runtime_equivalence=not_observed` remain explicit. Matching weights do not prove
the same activation/topology/Python behavior. No joint trusted inference
checkpoint identity is claimed.

## Retain and reopen statements safely

`GraphEvaluationResult` uses format `hyso.graph_evaluation_association`, version 1.
It embeds checkpoint-contract evidence v2 and existing P26 schema2/3; these version
numbers refer to distinct formats. It validates the native evaluation/record,
context consistency, recorded receipt structure and a statement digest.

```python
from hyso.analysis.graph_evidence import GraphEvaluationResult

association.to_json(fresh_path)
reopened = GraphEvaluationResult.from_json(fresh_path, expected_sha256=file_sha)
assert reopened.to_dict()["verification"]["graph_v2"] == "unverified"
```

JSON, another object, deepcopy and another process do not inherit the original
capture registry or source receipt. Persisted `verified` strings are statements,
not authority. The captured receipt remains inspectable as historical data;
read `verification` to distinguish it from live checks. The statement digest
excludes dynamic verification/relation views; file SHA hashes the actual exported
bytes. Readers ignore persisted authority claims and recompute their view.
No JSON reader creates or restores a checkpoint receipt.

## Compare observed quality without upgrading evidence

```python
from hyso.analysis.graph_evidence import compare_graph_evaluations

comparison = compare_graph_evaluations(association_a, association_b)
print(comparison.to_dict()["verification"])  # each source status separately
print(comparison.to_dict()["status"])
```

This compares only the two retained P26 quality observations, using existing
ModelComparisonSource, EvidenceRecord/EvidenceIndex, UnifiedAnalysisReport and
analysis.compare_models. It computes no new inference, training or benchmark.
The existing native comparison policies enforce observed stream, batch accounting,
runtime, metric semantics and capture identity. The opt-in overlay additionally
requires matching nonempty declared dataset/split/preprocessing IDs. Missing
conditions withhold its differences; mismatches return INCOMPATIBLE. Agreement
of declarations does not independently authenticate dataset origin/transforms.

One or both Graph v2 proofs may be missing while the retained legacy quality
comparison is valid. This does not invalidate the original comparison.
`comparison.legacy_comparison` retains that unchanged result with its own scope;
overlay conditions cannot silently relabel its authority. A recorded comparison
can be reopened via GraphComparisonResult and its source provenance is then
unverified. No global winner, statistical superiority or acceptance is inferred.

Performance evidence (P01/P02/etc.), registered structure comparison and multiple
architecture metadata comparison are different APIs. Use existing
`analysis.compare_models`, `compare_model_structures` and `compare_architectures`
for those explicit operations. The new adapter does not derive speed, resource
costs or parameter counts from v2 verification/configuration text.

## Reporting, Advisor and experiments

```python
from hyso.reporting import create_report

report = create_report(association_a, association_b, comparison, language="en")
report.to_json(json_path)
report.to_markdown(markdown_path)
report.to_html(html_path)
```

The existing renderer exposes source Graph v2, legacy P26, per-source limited or
unavailable binding, dataset-origin limitations and native evaluated stream.
Static HTML is an existing Python report artifact, not a frontend or service.
Reports are presentation-only and not authoritative evidence. A saved report's
status describes checks at report creation; reading its JSON cannot revalidate a
model. Advisor still receives the unchanged legacy observations and limitations;
v2 status does not increase its comparison eligibility or recommendations.

Live views call the existing bounded source-receipt query. That query can build
the profile's <=1M-parameter reference for validation, but does not load a
checkpoint, perform inference or train. Comparison/reporting start no workload.
Imported statement views do not have a source object to query.

Use existing ExperimentManager runs, links and schema2 artifact verification:
export the association and add it as a reference artifact with kind
`graph_evaluation_association`. Record its format/version and declared scope in
metadata. Verify its bytes through `verify_artifact`; reopen through the typed
association reader. A verified artifact hash is not a native receipt. Generic
experiment result restoration does not auto-upgrade this separate artifact into
a live model/receipt; explicit typed reading remains required.

The runnable checkout example uses two tiny models, one CPU fp32 epoch each,
synthetic train/test tensors, strict fresh checkpoint loads, actual evaluations,
comparison, reports and the existing Manager in a new isolated QA directory:

```powershell
python -B -m examples.models.graph_evidence_workflow --output C:/local-qa/fresh-graph-integration
```

It checks mutation, wrong expected identity, JSON trust loss and artifact hash
mismatch without accessing any user SQLite workspace. No real-data quality,
CUDA/native performance, independent dataset lineage, or arbitrary-code sandbox
claim is made.
