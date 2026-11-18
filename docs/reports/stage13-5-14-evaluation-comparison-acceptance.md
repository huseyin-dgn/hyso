# HYSo 13.5.14 evaluation and model comparison acceptance

Date 2026-11-16, Europe/Istanbul. **Overall decision PARTIAL. Gate G FAIL.**
Independent metric arithmetic, native CPU capture, receiver authority lifetime
and historical schema checks pass. Four ordinary negative acceptance tests fail
the requested stronger comparison conditions. These failures also reproduce on
both installed distributions. The historical P26 policies are preserved under
the stage instructions.

Evidence root `%LOCALAPPDATA%/hyso-stage13-5-14-20261116`, abbreviated QA below.
The run uses Python 3.10.18 from
`C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe` and Torch 2.5.1+cu121 on CPU.
The [compatibility matrix](stage13-5-14-evidence-compatibility-matrix.md) contains
the planned inputs, invariants and individual acceptance test outcomes.

## 1. Starting Git audit and evidence baseline

Branch `feature/hyso-models-next`. Expected and observed HEAD
`1f5667af5b4221f6be1bc4925579a978db284c43`. Initial index empty. Initial user
state contains two modified tracked files and 45 untracked files, totaling 47.
The modified tracked paths are `docs/guides/p23-forensic-plan.md` and
`tools/forensics/native_admission.py`. The full initial status is in
`QA/baseline.json`. Repository and parent-directory AGENTS.md discovery yielded
zero applicable files.

The 13.5.13 real-learning, 13.5.12 Builder compatibility, 13.5.11 receiver closure
and P26 native evidence, quality comparison and structure binding reports/guides
were inspected with their implementation and existing tests. The baseline covers
757 existing tracked/nonignored files and the inherited protected groups. It
also records 7,021 files outside __pycache__ directories under the previous stage QA root. QA datasets,
checkpoint copies, reports, SQLite stores, install targets, pytest temporary
directories and Matplotlib state are outside the repository.

## 2. Inspected APIs and result contracts

| Path/API | Observed role |
| --- | --- |
| `hyso/evaluation.py`, `evaluate_predictions(targets, predictions, *, class_names, split, task='multiclass', prediction_type='labels', zero_division=0, source_id=None)` | Supplied hard labels or finite logits produce schema1 |
| `evaluate_classification(model, loader, *, class_names, split, task='multiclass', zero_division=0, source_id=None, max_samples=1000000, capture_evidence=False, capture_structure=False, device='cpu', capture_receiver_runtime=False)` | Native isolated forward execution, schema1/2/3 and optional separate receiver capture |
| `EvaluationResult(data)`, `to_dict`, `to_json`, `to_csv`, `structure_snapshot` | Frozen typed result, strict reader, detached projections and exports |
| `_metadata`, `_predicted`, `_metrics`, `_result` | Ordered task/class contract, decision rule, confusion arithmetic and result assembly |
| `hyso/_evaluation_evidence.py` | Named-state hash, ordered tensor stream capture, weak checkpoint receipts and producer validation |
| `hyso/_evaluation_structure.py` | Schema3 pre/post registered receiver structure association |
| `EvidenceRecord.from_native(native, feature_id='P26', scope=...)`, `from_json(..., expected_sha256=...)` | Validated native statement and artifact/scope binding |
| `EvidenceIndex(records).analyze(context=AnalysisContext(...))` | Typed unified retained evidence report |
| `ModelComparisonSource(model_id, report, *, records=None, structure=None, benchmark=None)` | Explicit model label and record selection |
| `compare_models(baseline, candidate, *, features=..., include_structure=...)` | Existing P26 eligibility and descriptive metric differences |
| `evaluate_graph_classification`, `GraphEvaluationResult`, `compare_graph_evaluations` | Separate source v2, receiver and context-aware comparison association |
| `compare_architectures(configs, *, names=None, exact_parameters=False)` | Builder architecture metadata and optional bounded CPU parameter count |
| `capture_model_structure`, `compare_model_structures` | Literal registered Torch metadata comparison |
| `create_report(*results, ...)`, `ExperimentManager` | Existing presentation and QA artifact registration APIs |

The actual chain exercised is EvaluationResult → EvidenceRecord → EvidenceIndex
→ UnifiedAnalysisReport → ModelComparisonSource → compare_models. Reporting
accepts the typed analysis/comparison and Graph association results. Direct
EvaluationResult input uses its existing EvidenceRecord/Index integration.

## 3. Field sources, invariants and limits

| Field | Source and calculation | Format/invariant | Boundary and negative coverage |
| --- | --- | --- | --- |
| Accuracy | `_metrics`, confusion diagonal divided by consumed count | Fraction-valued scalar, null for empty | Independent perfect/wrong/collapse/imbalance/ten-class calculations |
| Precision/recall/F1 | Per-class TP, predicted count, support | `metrics.per_class`, class index/name/support plus three scalars | Missing predictions, zero support, zero_division 0/1 |
| Macro | Arithmetic mean over every configured class | `metrics.averages.macro` | Includes absent classes |
| Weighted | Support-weighted class arithmetic | `metrics.averages.weighted` | Supports sum to N |
| Micro | Single-label aggregate, equals accuracy | `metrics.averages.micro` | Every consumed sample has one target and prediction |
| Confusion matrix | True target row, predicted class column | C×C integer matrix | Total N, row supports, column prediction counts |
| Prediction/correct/incorrect counts | Column sums, diagonal sum and N minus diagonal | Derived integer checks | The result has no separate serialized predicted-count or correct/incorrect-count field |
| Raw observations | Actual loop logits and normalized target labels | `raw.predictions`, `raw.targets`, `prediction_type='logits'` | Finite output, exact class width and label range |
| Sample accounting | Dataset length, observed loop/batch lengths | `expected_samples`, ordered `batch_sizes` | Under/over collation, count and batch tampering rejected |
| Ordered stream | Validated post-collation input/original target tensors before forward | Input/target/paired SHA-256, method, scope and native count | Input/order/target/pair/count/dtype changes checked in new and retained tests |
| Class vocabulary/order | Explicit `class_names` | Unique ordered indices 0..C−1, 2..1000 classes | Different names/order incompatible, duplicate/invalid names rejected by existing reader tests |
| Dataset/split/preprocessing | Caller context/split and declared EvidenceScope | Split declared, dataset_origin/preprocessing missing in native producer v1 | Generic policy accepts differing dataset/preprocessing declarations on identical streams, strict probes FAIL |
| Model identity | Original/isolated named parameters and buffers | Separate model-state SHA-256 and model type | Copy alias/state mutation checks, wrong native scope rejected |
| Evaluation identity | Writer UUID | Canonical evaluation_id covered by source hash | Copied native run UUID incompatible |
| Checkpoint | Explicit captured trusted load of exact bytes into exact object/state | Native receipt or missing | Wrong file/scope SHA, changed state, copied object and expiry checked |
| Registered structure | Actual isolated receiver boundary snapshots | Schema3 structure_binding and detached typed snapshot | Rehashed incoherent evaluation/boundary associations rejected |
| Graph association | Supported source receipt plus actual Core receiver observations | Separate versioned association and live registry ownership | Default limited, explicit supported CPU verified, imported/copy/expired unavailable |

Independent scalar assertions use absolute tolerance 1e-15 and relative tolerance
1e-14 where rational-to-float projection needs it. The persisted EvaluationResult
reader recomputes metrics and compares their canonical hashes exactly. A 1e-15
alteration of a stored accuracy fails that reader. Arithmetic test tolerance
therefore supplies no acceptance tolerance for modified stored metric evidence.

`source_sha256` covers the source envelope. Metrics and status are separately
regenerated from raw/configuration data during validation. A self-consistent
retained statement depends on trusted producer provenance. Local content hashes
provide integrity, while Graph live authority requires its separate process-local
registry and exact source/result/receiver objects.

## 4. Independent mathematical and task verification

The new reference constructs confusion cells by independently counting target /
prediction pairs and calculates TP/support/predicted totals with `Fraction`.
The production evaluator is exercised with a real Identity forward receiver and
chosen finite logits. Its raw target/logit records are checked against the
fixtures before its metrics are compared with the reference.

Fourteen cases cover seven selected distributions under both zero-division
policies. They include perfect predictions, all wrong, single-class collapse,
imbalance, a class with support and zero predictions, zero-support classes and
ten classes. All confusion totals, row/column counts, exact correct/incorrect
integers and macro/micro/weighted precision, recall and F1 agree.

Binary task requires one logit per sample in shape `(N,)` or `(N,1)` and floating
hard targets. The threshold is `logit >= 0`. Two logits with `task='binary'`
raise ValueError. Explicit two-class `task='multiclass'` accepts `(N,2)` and uses
first-index argmax ties. Existing binary-vector/threshold regressions complement
the new real-receiver cases.

Wrong width and out-of-range targets fail. Separate tests cover nonfinite inputs
and nonfinite output logits produced from finite inputs. Empty evaluation has
`status='empty'`, zero accounting and null metrics. Incomplete/extra collated
samples fail before a complete result can be published.

## 5. Native evidence, stream compatibility and lifetime

The trusted producer hashes original/isolated named state, observes the same
actual input/target iteration used by inference and records its raw predictions.
Schema2 evidence names the native state/stream source and separately retains
checkpoint, split, source_id, preprocessing and dataset-origin statuses.
Schema3 adds the actual receiver registered structure observation.

Input, target, ordering, count, class vocabulary/order, task, split label,
zero-division and batch partition changes produce INCOMPATIBLE with null metric
differences. The input/target/paired stream hashes are batch-partition invariant.
P26 native comparison v1 independently requires equal batch-size sequences.

Schema1 and supplied-prediction evidence produce INSUFFICIENT_EVIDENCE for the
native P26 comparison. Reader-level incoherence raises ValueError/EvidenceError.
Tests reject changed raw/metric/accounting data, false producer status, wrong
checkpoint/state association and changed schema3 evaluation linkage even with a
recomputed outer source hash. Reused evaluation IDs produce INCOMPATIBLE.

A captured QA checkpoint receipt binds its SHA to its exact loaded object and
state. Deepcopy yields missing, state mutation yields missing, and object
collection removes the weak registry entry. The existing producer regressions
also cover altered deepcopy state, one-pass post-collation observation, safe
checkpoint snapshot loading, failed restores and unsupported custom state.

ModelComparisonSource model_id is a caller label. Native state/checkpoint facts
come from its selected record and conflicting native scope declarations are
rejected. Native JSON supplies a retained statement. Its consistency checks
depend on the original producer trust boundary.

## 6. Comparison results and unresolved acceptance failures

Fresh actual captures from different architectures produce COMPARABLE for both
accuracy and macro-F1. A 0.75→1.0 accuracy change yields +0.25 and +25 percentage
points. Equal accuracy with different per-class outcomes also compares, with
zero accuracy difference. All claims retain `global_winner=null`, statistical
superiority `not_established` and acceptance `NOT_EVALUATED`.

The historical policy deliberately selects exact observed tensors and allows
caller dataset/preprocessing declarations to differ. The requested acceptance
requires stronger declaration consistency and complete suppression of numeric
differences for incompatible evidence. The following four collected tests fail
on source, wheel and sdist.

| Ordinary acceptance test parameter | Actual result | Requested result | Decision |
| --- | --- | --- | --- |
| Generic dataset_id left/right | Both P26 metrics COMPARABLE, numeric differences present | INCOMPATIBLE and null differences | FAIL |
| Generic preprocessing_id left/right | Both P26 metrics COMPARABLE, numeric differences present | INCOMPATIBLE and null differences | FAIL |
| Graph dataset_id mismatch | Top INCOMPATIBLE/null, embedded legacy metrics COMPARABLE/numeric | Suppress incompatible numeric differences throughout result | FAIL |
| Graph preprocessing_id mismatch | Top INCOMPATIBLE/null, embedded legacy metrics COMPARABLE/numeric | Suppress incompatible numeric differences throughout result | FAIL |

Generic fixture values are accuracy 0.75→1.0 and macro-F1 11/15→1.0, giving
differences +0.25 and +0.2666666666666666. The Graph fixtures retain numeric zero
differences inside `legacy.metrics`, which still constitute numeric differences.
`QA/requirement-findings.json` and `QA/requirement-evidence/` retain complete
typed comparison/report JSON and Markdown/HTML evidence.

Root causes are `_derive` in `hyso/analysis/model_comparison.py`, which delegates
P26 to its historical exact-stream contracts, and `_compare` in
`hyso/analysis/graph_evidence.py`, which applies context rejection to outer rows
while retaining a complete legacy comparison. This matches the documented old
policy and exposes a gap against this stage's stronger acceptance requirements.

The four assertions use ordinary pytest collection, with zero skip/xfail. Run
the committed probe module directly to reproduce the nonzero test exit.

```powershell
& 'C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe' -B -m pytest tests/test_evaluation_comparison_requirements.py -q -p no:cacheprovider --basetemp="$env:LOCALAPPDATA/hyso-stage13-5-14-repro"
```

### Narrow exception proposals

1. Review an explicitly selected strict context check in the current comparison
   adapter. Require matching supplied dataset/split/preprocessing declarations,
   give known mismatches precedence and withhold metric differences before
   publication. Preserve the existing P26 policy versions and their archived
   results. Declaration equality retains declared authority.
2. Review a separately versioned Graph comparison envelope that carries the
   effective incompatibility verdict throughout its exported result. Keep the
   legacy artifact's original hash and content in separately identified
   historical evidence, with a value-free reference in the incompatible result.
   The present embedded legacy contract requires an intact typed comparison,
   so arbitrary field redaction would violate its validated schema/hash.

These proposals require comparison policy/schema review. Implementation count
for protected Core/native and existing runtime modules in this stage is zero.
The unresolved gates require a subsequent authorized correction and rerun of
the four existing failing tests.

## 7. Graph v2 source, actual receiver and inference binding

New supported CPU Graph checkpoints and the selected historical Graph checkpoint
were loaded through the existing SHA-guarded API. Explicit receiver capture
verified three distinct observations in the original process.

| Evidence | Actual status |
| --- | --- |
| Source Graph v2 checkpoint contract | graph_v2 verified |
| Actual isolated receiver runtime | receiver_runtime verified |
| Same-execution source-to-receiver inference binding | inference_binding verified |
| Default Graph association | relation limited |
| Graph legacy P26 checkpoint in native evaluation | missing |
| Imported association with verified artifact SHA | graph_v2 unverified, receiver_runtime unverified, relation unavailable |

A second isolated interpreter independently imports the saved association and
reports unverified source/receiver authority and unavailable binding. Source
collection expires the original live association authority. Existing receiver
regressions reject unsupported subclasses, hooks, source/receiver mutation,
topology, activation, output, installed callable/constructor changes, storage
aliasing, failed/partial inference and execution-statement substitution.
The supported trusted CPU profile supplies the scope of the passing gate.

## 8. Architecture, registered structure and quality observations

Different Linear weights have different model-state hashes and different QA
accuracy while their registered structures compare equal. A Linear model and
its Sequential wrapper produce equal accuracy with different per-class records
in the selected comparison fixture. Registered structure observations and
descriptive quality results retain their own fields.

Builder architecture comparison distinguishes add/multiply branch topology and
SiLU/ReLU activation at equal parameter counts. A changed output-head width
also changes its configuration hash and parameter count. Repeated identical
Builder configurations have equal configuration hashes. Structure regressions
separately check container/module type and Conv/Linear field changes.

The architecture API records configuration/geometry/count metadata. The structure
API records registered module/tensor metadata. The quality API records metrics
under its evaluation contract. Live receiver verification has its separate
source/receiver/execution evidence.

## 9. Preserved 13.5.13 learning evidence

The previous QA root is `%LOCALAPPDATA%/hyso-stage13-5-13-20261116`.
All 90 entries in its retained evidence index match their expected SHA-256.
Each of the 67 historical artifact observations matches current bytes under
its learning directory. All 7,021 prior QA files match the initial baseline.
The historical report, committed snapshot, checkpoints, evaluations and model
state artifacts are included in these checks.

The saved prediction/target JSON was read through EvaluationResult and checked
with the independent rational reference. This arithmetic check consumes saved
predictions. New locked-test inference count is zero.

| Historical model | Correct/total | Accuracy | Macro-F1 | Selected checkpoint SHA-256 |
| --- | --- | --- | --- | --- |
| sequential | 263/270 | 0.9740740740740741 | 0.9739754790797491 | cddc0ae202ddaaa7347a6d7879867e34487553cf4b858ef27c6947714526ebcc |
| graph | 263/270 | 0.9740740740740741 | 0.973873618676002 | e9972bd496b442d7e05ec9a4f19f5c098d77317f357c398dbdf82c29e6f4c5e4 |

Both saved streams have paired SHA-256
`1be2da286c129a4c7b9691277184591717d40ca57e2b106cea284f217df692dc`.
Importing the historical Graph association in the present process produces
receiver_runtime unverified. `QA/historical-proof.json` records the 67 artifact
matches, historical metrics and current imported-authority observation.

## 10. Fresh selected-checkpoint QA and Reporting/ExperimentManager

Both selected checkpoints were restored into newly constructed models with
their expected artifact SHA and architecture/settings/selected-epoch metadata
guards. The independent QA input is `arange(256).reshape(4,1,8,8).float()/255`
with labels `[0,1,2,3]`, batch size 2 and ten declared digit classes. It uses
four synthetic tensors and the configured P26 `test` label. Both accuracies
are 0/4. This is a bounded evidence integration exercise.

Fresh schema3 evaluation/comparison, live verified Graph, default limited Graph
and imported unavailable Graph feed the existing Reporting API. Files in
`QA/selected-checkpoint-qa/` include evaluation JSON, association JSON,
comparison JSON, `report.json`, `report.md` and static `report.html`.
Twenty reported scalar values match their source metric paths. The report's
embedded evaluation/comparison payloads match the source typed payloads exactly.
The three association sections display verified, limited and unavailable in
their respective scopes. Presentation authority is false.

A fresh QA-only ExperimentManager store records the evaluation artifacts.
Seven reference artifacts verify. A wrong supplied artifact copy produces
mismatch, followed by verified on the correct path. The new test's separate
QA store also checks five managed-copy artifacts and a reference artifact.
`QA/report-content-proof.json` and the selected-checkpoint summary retain the
values, statuses and artifact observations.

For incompatible Graph contexts, Reporting clears the visible quality values
and charts. Its retained raw Graph comparison still contains the legacy numeric
differences described in section 6. This leaves gate J PARTIAL under the requested
complete export-boundary requirement.

## 11. Schemas and regression results

Schema1 remains the default. Native producer capture emits schema2. Adding
registered structure capture emits schema3. All three round-trip exactly with
frozen results and detached accessors. Invalid artifact SHA and incoherent
rehashed metric/raw/producer/accounting/structure data are rejected. The selected
existing tests also exercise old readers, golden schema payloads and native
P26 policy/schema eligibility boundaries.

| Final evidence group | Passed | Failed | Errors/skips |
| --- | --- | --- | --- |
| new-final | 69 | 0 | 0/0 |
| regression | 317 | 0 | 0/0 |
| source-requirements-final | 0 | 4 | 0/0 |
| wheel-short | 69 | 0 | 0/0 |
| wheel-requirements-final | 0 | 4 | 0/0 |
| sdist-short | 69 | 0 | 0/0 |
| sdist-requirements-final | 0 | 4 | 0/0 |

There are **386 unique successful source cases and four unique failed acceptance
cases**. Installed repeats are counted separately. The selected 317 regressions
cover evaluation, native producer/structure binding, P26 Advisor policy,
ModelComparison, receiver runtime, Graph association and selected structure,
reporting and artifact tests. Exact selectors, XML identities and SHA values
are in `QA/regression-selection.json` and `QA/test-proof.json`.

Initial new-test execution recorded 63 passes and one fixture failure. The
fixture supplied a wrong path to verification of a managed-copy artifact, for
which the API verifies its managed destination. The corrected test registers
a reference artifact before checking the supplied wrong path, and then checks
the correct path. The original failure log/XML are retained. The independent
historical audit also first failed on a Windows path-separator filter in the QA
script. Its corrected filter verifies all 67 historical hashes. Both development
failures concern QA setup. The four final requirement failures remain open.

## 12. Installed distributions and code equivalence

The minimal packaging check reuses the stage 13.5.13 archives after verifying
their recorded hashes, exact 164-module sets and every module's source bytes.
Each archive is installed afresh into this stage's separate QA target with
`--no-index --no-deps --no-build-isolation --no-compile`. Both installed targets
match all 164 source module bytes. Packaging configuration is covered by the
757-file preservation manifest.

| Kind | Archive SHA-256 | Fresh target under QA |
| --- | --- | --- |
| wheel | 93f94711abb4a4c1271f50773df0d443526dcfafafb5860a9ffe08310f31b51c | installed-wheel |
| sdist | b7363d12d53183801bce1990c80d5d8b55cadfeea86f7fff3b0ac04d97a2b774 | installed-sdist |

Isolated Python asserts `hyso.__file__` belongs to each target. All 80 imported
HYSo modules also resolve inside that target. The installed commercial profile
and CPU execution are checked. Existing user-site pygments supplies a test
dependency through the isolated runner. The second-interpreter authority test
also imports the selected installed target.

Each distribution passes the same 69 short tests and reproduces the same four
failed requirement probes. `QA/package-proof.json`, `wheel-import-proof.json`,
`sdist-import-proof.json` and the individual XML/logs retain this evidence.
New training fits, locked-test inference, CUDA benchmarks, long native workloads
and new distribution builds each have execution count zero.

## 13. SHA-256 preservation

| Protected group | Matching baseline SHA-256 |
| --- | --- |
| All initial repository files | 757/757 |
| Unrelated dirty user files | 47/47 |
| Historical native files | 71/71 |
| Historical forensic files | 47/47 |
| Inherited external6 | 6/6 |
| Inherited external11 | 11/11 |
| User data/model files | 197/197 |
| Previous stage QA files | 7021/7021 |

Groups overlap. Every pre-existing runtime/Core/evidence contract, Graph
mechanism and acceptance report is covered by these matching manifests.
Historical P01–P35 evidence and manifested SQLite/WAL/SHM/lock bytes are checked
as files. User stores have zero write connections. The scope is the manifested
paths. `QA/baseline.json`, `preservation.json` and `final-manifest.json` provide
the starting and closing records.

## 14. Git closure

Only the two new stage test modules and two new stage reports are committed.
The test commit is `c195051d483076320a8569cfd021c54b19185675`, parent
`1f5667af5b4221f6be1bc4925579a978db284c43`, with message
`test: validate evaluation metrics and comparison evidence`. Its exact paths are
`tests/test_evaluation_comparison_acceptance.py` and
`tests/test_evaluation_comparison_requirements.py`. The cached diff SHA-256 is
`b1a5d233f4e8e6c451ff1860b4e71435b01fdecedb6b8eb90eebea101e36e1a5`.

The following documentation commit contains exactly this report and
`docs/reports/stage13-5-14-evidence-compatibility-matrix.md`. Its own full
SHA/parent are recorded in `QA/commits.json` and the final handoff. The cached
documentation diff and precommit file hashes are retained separately in QA.

Before each commit, exact staged paths, cached diff, whitespace, protected SHA
matches and test outcome ledgers are checked. The closing audit verifies ancestry,
empty index and equality with the original unrelated 47-file status. Local
commit count is two. Push count and next-stage execution count are zero.

## 15. Independent gates and final decision

| Gate | Decision | Observed scope / open requirement |
| --- | --- | --- |
| A Metric Mathematical Correctness | PASS | Independent integer/Fraction arithmetic, all requested averages |
| B Multiclass / Binary Evaluation | PASS | Explicit task semantics, valid widths and malformed-input rejection |
| C Confusion Matrix / Accounting | PASS | Supports, columns, exact counts, complete batches and empty semantics |
| D Native Producer Evidence | PASS | Trusted producer consistency, object/state/receipt/lifetime boundaries |
| E Stream / Dataset / Split / Vocabulary Integrity | PARTIAL | Native stream/class/protocol checks pass, dataset/preprocessing origin and stronger declaration matching are open |
| F P26 Model Comparison Correctness | PARTIAL | Narrow historical P26 arithmetic/status behavior passes, stronger context requirement is open |
| G Incompatible Evidence Rejection | FAIL | Two generic declaration mismatches accepted, two Graph results retain nested numeric differences |
| H Graph v2 / Actual Receiver Trust Boundaries | PASS | Supported CPU explicit opt-in, default limited, imported/expired unavailable |
| I Architecture / Structure / Quality Separation | PASS | Independent configuration, registered metadata, state and quality observations |
| J Reporting / ExperimentManager Integration | PARTIAL | Exact transfer and artifact checks pass, raw incompatible Graph payload retains legacy differences |
| K Legacy Schema Compatibility | PASS | Schema1/2/3, old readers/goldens and historical Graph P26 missing |
| L Regression / Installed Package Tests | PARTIAL | 386 successful source cases, 69 per install, four acceptance failures reproduce on all targets |
| M Native / User Data Preservation | PASS | All manifested SHA groups match |
| N Git Integrity | PASS | Two scoped local commits and closing ancestry/index/status audit |

**Overall PARTIAL.** Four collected acceptance failures are retained with their
minimal reproductions and narrow proposals. The two comparison-boundary changes
need explicit policy/schema resolution before this stage can obtain PASS.
