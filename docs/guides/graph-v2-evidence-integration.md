# Graph v2 provenance in evaluation, comparison and reports

This opt-in adapter presents two different kinds of evidence together. It does
not move a trusted source receipt to the isolated model used for inference.
Existing calls to evaluate_classification, analysis.compare_models and reporting
keep their default schemas and behavior. Evaluation now also accepts the explicit
CPU-only `capture_receiver_runtime` opt-in described below.

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

## Verify the actual isolated receiver explicitly

13.5.11 adds a separate, opt-in association version 2. Default calls above still
emit version 1 with a limited source association. Existing version 1 files retain
that meaning and are never silently upgraded.

```python
association = evaluate_graph_classification(
    loaded_model, sequential_test_loader,
    class_names=["a", "b"], split="test", source_id="my-dataset",
    context={"dataset_id": "my-dataset", "split": "test",
             "preprocessing_id": "my-explicit-transform-v1"},
    expected_checkpoint_sha256=known_checkpoint_sha,
    capture_receiver_runtime=True,
)
view = association.to_dict()
assert view["schema_version"] == 2
assert view["verification"]["graph_v2"] == "verified"       # live source receipt
assert view["verification"]["receiver_runtime"] == "verified"
assert view["relation"]["status"] == "verified"             # same actual execution
assert view["evaluation"]["producer_evidence"]["checkpoint"]["status"] == "missing"
```

Core runs the existing evaluation loop once. After deepcopy/eval, immediately
before and after each actual candidate call, and before publication, it checks
the same exact receiver, the current source receipt, independent full receiver
Graph contract/state equality, eval modes and storage isolation. There is no
source receipt on the receiver and no second inference. `capture_structure=True`
can also be used; the embedded EvaluationResult stays schema2 or schema3.

This path requires a live supported Graph v2 source receipt, a nonempty complete
sequential CPU evaluation, exact Tensor batches, and the existing closed CPU
fp32/fp64 component profile. It conservatively rejects shared/view tensor storage,
hooks, subclasses, instance forward overrides and observed installed callable or
code changes. Unsupported state or any failed/mismatched observation raises an
error and publishes no receiver authority. The Core option also requires
`capture_evidence=True` when used directly with evaluate_classification; it does
not insert new fields into EvaluationResult.

`receiver_binding` uses its own `hyso.graph_receiver_binding` version 1, containing
opaque Core-issued object/capture tokens, exact evaluation/checkpoint/state
references, independent receiver contract, pre/post commitments and actual
stream/batch accounting. Those are retained statements. The process-local
verification view belongs to the original typed result and actual Core capture.
The adapter keeps that capture/isolated receiver alive for its result lifetime,
holds the source weakly, and revalidates both on a live view. Keep the source alive.
Observed source/receiver mutation makes the live binding unavailable. Copies,
constructors and JSON readers, including another process, have no such authority;
their receiver view is `unverified`, even when retained fields say `native`.

Comparison version 2 supports old/new or new/new associations and reports each
source's actual binding separately. Existing P26 comparability and Advisor policies
remain unchanged. Verified runtime does not authenticate dataset lineage, make
different streams comparable or establish statistical superiority. Reporting and
ExperimentManager artifact SHA verification cannot mint source or receiver trust.

The contract assumes trusted installed Core/PyTorch at import and quiescent
supported objects. It detects observations at the stated boundaries; it does not
attest an adversarial interpreter, concurrent hostile mutation, private-registry
tampering or transient mutate-and-restore behavior entirely between observations.
It proves no CUDA behavior or universal Graph support. Queries rebuild bounded
reference metadata but do not run inference/training.

The short example selects the new path explicitly:

```powershell
python -B -m examples.models.graph_evidence_workflow --output C:/local-qa/fresh-receiver --capture-receiver-runtime
```
