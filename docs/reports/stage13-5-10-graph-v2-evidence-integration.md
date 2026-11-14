# HYSo 13.5.10 — Graph v2 evidence integration and Git finalization

Date: 2026-11-14, Europe/Istanbul. Evidence root:
`%LOCALAPPDATA%/hyso-stage13-5-10-20261114` (`QA/` below).

| Independent acceptance | Decision and scope |
| --- | --- |
| A — Graph v2 evaluation integration | **PASS — versioned source provenance with explicitly LIMITED inference association** |
| B — Model comparison evidence integration | **PASS — retained P26 quality observations and declared-condition guards** |
| C — Reporting and ExperimentManager | **PASS — separate trust scopes, presentation and verified artifact bytes** |
| D — Legacy P26 compatibility | **PASS** |
| E — Native evidence trust preservation | **PASS** |
| F — Git history and commit integrity | **PASS** |

**A shared trusted inference-checkpoint runtime binding is NOT VERIFIED.**
Source Graph v2 receipts can be verified while the actual evaluation deepcopy has
no transferred receipt and its full v2 runtime signature remains unobserved.
Legacy Graph P26 schema2/3 checkpoint evidence remains **missing**. PASS above
does not extend beyond these explicit scopes, or imply CUDA/data-quality approval.

## 1. Initial Git state

Expected/observed branch: `feature/hyso-models-next`. Starting HEAD:
`d8691a1a7ab464d14aae4ae9e8fea627982dbf99`. The preceding five commit/parent
relationships matched the completed 13.5.9 audit. Index was empty. There were
**2 modified tracked + 46 untracked = 48** remaining files.

`QA/baseline.json` was created before staging: 737 tracked/nonignored untracked
hashes, raw status/index, 11 external hashes and 197 repository user
database/model/checkpoint hashes. The prior protection, acceptance and independent
audit manifests were read. No unrelated working tree cleanup was attempted.
Python: `C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe`; CPU execution only.

## 2. Original/current 13.5.8 report verification

The existing Git blob `b1404718d113f6c8e573ac94e6a6c872f1ee0af1` was read
using `git cat-file blob`. Its actual bytes hash to:
`f462448e1d61d56fc4ed1f3d114c6b757fe08ac944c9a88442b1506699ee8191`.

The current formatted report hashes to:
`afa43847e8b3869c4a4d5fc49d5c2ed94a7aa00aba8db10d27c80ea67e25cc57`.
Its rendered HTML equals the original under the same CommonMark/table parser.
The prior independent `report-comparison.json` was reviewed: FORMATTING ONLY,
75 table-format lines plus three equivalent emphasis-marker lines; no changed
acceptance, metrics, hashes or limits. The historical manifest was not modified,
and the current copy was not represented as original bytes.

## 3. Original report QA archive

The original **29,281 bytes** are separately archived, outside the repository:
`QA/original-stage13-5-8-report.md`. Its SHA is the exact `f462...` value above.
`QA/report-archive.json` records Git blob, original/current hashes, archive path,
HTML equality and the prior independent audit reference. No original bytes were
guessed or regenerated to match a hash. The original Git object remains intact.

## 4. Docs-only closure commit

Commit: **`b527ad90e536c8371e03f311b4cbe3a1c94261dc`**.
Parent: `d8691a1a7ab464d14aae4ae9e8fea627982dbf99`.
Message: `docs: record verified stage 13.5.8 acceptance report`.

Exactly one file was staged and committed:
`docs/reports/stage13-5-8-real-dataset-p26-investigation.md`. Cached bytes equal
the verified formatted SHA. Cached name/check/stat/diff, single-file scope,
parent/show-stat and empty post-commit index were verified. The other 47 files
stayed outside staging. The previous five commits were not changed.

## 5. Existing evaluation/comparison/reporting contracts

`evaluate_classification` performs actual inference on a deepcopy, checks
nonaliasing and original/isolated named-tensor hashes, and records metrics,
confusion matrix, native model-state and ordered-stream evidence. Schema3 adds
the actual receiver's pre/post structure capture; it does not claim source/copy
behavior or v2 contract equivalence. EvaluationResult remains unchanged.

`analysis.compare_models` compares exactly selected retained sources through
existing NativeMetricResolver and P26 comparability policies. It does not run
workloads or repair missing provenance. Registered structure comparison and
multi-architecture metadata comparison are separate operations and remain so.

Existing Reporting produces presentation-only JSON/Markdown/static HTML from
validated retained results; its integrity digest does not certify authenticity.
Advisor interprets those records conservatively. ExperimentManager stores runs,
relationships and schema2 artifacts; byte verification is not a model receipt.
None of those existing contracts was replaced.

## 6. Graph v2 integration design

New opt-in module: `hyso.analysis.graph_evidence`.

- `evaluate_graph_classification` queries source v2 before/after **one** call to
  the existing evaluator, preserving the original native result and record.
- `GraphEvaluationResult` uses a separate
  `hyso.graph_evaluation_association` schema_version 1 envelope containing the
  v2 source statement, original P26 schema2/3 result, record and declared context.
- `compare_graph_evaluations` delegates numerical quality comparison to the
  existing `compare_models`, with a separate context guard/diagnostics overlay.
- `GraphComparisonResult` uses separate `hyso.graph_evidence_comparison` version
  1; its `legacy_comparison` preserves the original typed Core result.
- The existing Reporting dispatcher has an explicit adapter for these two types.

These envelope versions do not upgrade checkpoint v1/v2 or P26 evaluation
schema1/2/3. No alternate evaluator/checkpoint/report engine was introduced.

## 7. Process-local receipt security boundary

Source v2 proof is queried from the actual existing trusted checkpoint API,
never accepted as a caller-supplied receipt dictionary. Expected snapshot/state
hashes constrain an already-existing receipt and cannot create one. The adapter
retains a weak reference to the same source object plus a process-local capture
identity tied to its immutable statement. Live views recheck exact source receipt
and state/config/runtime identity through the unchanged v2 API.

Plain JSON construction/import, result deepcopy and another interpreter do not
register a capture. Their source status is **unverified**, even if JSON says
verified. Recorded native fields remain inspectable historical statements; the
derived verification view supplies the authority distinction. Statement/file
digests validate integrity, not live capture authenticity. Arbitrary malicious
Python execution inside the interpreter is outside the process-local trust model.

## 8. Deepcopy and runtime identity

No source receipt is installed on the isolated inference receiver. The adapter
does not call private receipt producers to manufacture that relationship.
When the source receipt and observed named-state hashes agree, the association
is only **limited**:

```text
isolated_receiver_receipt = not_transferred
runtime_equivalence = not_observed
scope = source_receipt_and_observed_named_tensors_only
```

Source receipt loss/mutation/reload yields missing; a destroyed source or copied/
imported result cannot retain current verified authority. A source missing at
capture is not silently upgraded by a later load. Schema3 receiver structure does
not change the limited contract. A future full runtime binding would need
explicit observation of the actual receiver, not an assumed copy identity.

Live checks can construct the v2 profile's bounded <=1M-parameter reference, as
the existing receipt query already does, but never load a checkpoint, infer or
train. The source evaluator is invoked only once per requested capture.

## 9. Legacy P26 separation

Supported Graph P26 native checkpoint fields remain **missing**. The source v2
statement is outside those fields, outside EvidenceRecord checkpoint bindings,
and outside existing Advisor eligibility. Dataset/split/preprocessing context
remains declared. Native inference kind, model-state/stream hashes, confusion
matrix and classification metrics remain the original Core evidence.

Imported statements cannot become native by adding expected hashes, matching
labels, recording an artifact or rendering a report. No historical JSON, receipt,
P01–P35 record or checkpoint file was automatically converted.

## 10. New and preserved Python APIs

Public opt-in APIs/types are accessed from `hyso.analysis.graph_evidence`:
`GraphEvaluationResult`, `GraphComparisonResult`, `GraphEvidenceError`,
`evaluate_graph_classification`, `compare_graph_evaluations`.
Evaluation associations expose `evaluation`, `analysis_source(model_id)`,
`to_dict`, `to_json`, `from_json`; comparisons expose `legacy_comparison` and
equivalent serialization helpers. JSON reads have a 32 MiB bound and optional
actual-file SHA check. Constructors validate existing native readers and context/
derivation consistency, but do not create native trust.

Signatures and default outputs of existing evaluation, comparison, Trainer,
checkpoint, reporting, Advisor, CLI and structure APIs are unchanged. Existing
v2 code/contract and protected native modules are byte-identical to 13.5.9.

## 11. Source changes

Implementation commit contains only six explicitly reviewed files:

| File | Change |
| --- | --- |
| hyso/analysis/graph_evidence.py | New bounded opt-in association/comparison/report adapter |
| hyso/reporting/report.py | Three-line exact-type opt-in dispatch; old branches unchanged |
| examples/models/graph_evidence_workflow.py | Fresh-path short CPU workflow using existing APIs |
| tests/test_graph_evidence_integration.py | Independent negative matrix, process boundary and golden coverage |
| commercial-profile.json | Explicit new module entry, 162→163 runtime modules |
| tests/test_packaging.py | Inventory/count updated consistently |

The new guide and this report are a separate docs commit. No protected-file
exception was requested or used. Graph forward, checkpoint.py,
_evaluation_evidence.py, evaluation.py, comparison implementations, renderers,
Advisor and ExperimentManager were not modified.

## 12. Negative matrix and red/green evidence

Final new group: **31/31 pass**, no skip/xfail (`QA/integration-final.xml`).

| Requested scenario | Observed behavior |
| --- | --- |
| A correct source v2 receipt | Verified source, original native evaluation/stream, limited association |
| B legacy Graph P26 | Still missing |
| C no source receipt | Evaluation works; v2 missing, association unavailable |
| D v1 Graph checkpoint | Existing load/evaluation works; no v2 upgrade |
| E wrong checkpoint SHA | Expected-source identity rejected before evaluation |
| F wrong state hash | Expected-source identity rejected |
| G topology with same tensor schema | Source mutation invalidates verified view |
| H activation | Invalidates source trust |
| I output adapter | Invalidates source trust |
| J geometry | Invalidates source trust |
| K dtype/device | fp64 or meta mismatch invalidates; no CUDA launched |
| L parameter mutation | Missing after mutation |
| M persistent buffer mutation | Missing after mutation |
| N deepcopy | Both copied result and copied source cannot inherit authority |
| O other model instance | Actual API query has no original source receipt |
| P different stream | Incompatible; guarded numerical differences withheld |
| Q wrong dataset/split association | Explicit declaration mismatch rejected |
| R one verified/one missing | Valid legacy comparison remains usable; statuses separate |
| S two verified/incompatible conditions | Stream/batch/preprocessing/dataset guards withhold differences |
| T fake verified JSON | Typed read and report show unverified; comparison import also downgraded |
| U schema1/2/3 legacy | Existing regressions plus frozen full legacy comparison/report golden pass |
| V missing/broken metadata | Reader rejects invalid context/version/evaluation/integrity |
| W artifact hash mismatch | Real Manager returns mismatch; bytes never mint trust |
| X valid compatible pair | Actual P26 Core difference and per-source verification retained |

Separate tests execute a new interpreter for JSON import and forbid inference/
checkpoint loads during comparison/reporting. Another test counts one Core
evaluator call per capture. Missing preprocessing context limits only the new
overlay; the original legacy comparison result is retained unchanged.

Before implementation, the independent test module failed collection because the
new adapter API did not exist (`QA/red.xml`); this is an integration gap, not a
claimed legacy security defect. First implementation run: 25 passed, one failed,
one deselected. The failure was the new test omitting the existing Manager's
required config argument; only that test fixture was corrected. Final tests
include additional process-boundary/golden/retained-JSON checks and pass all 31.

## 13. Actual short CPU acceptance

`examples/models/graph_evidence_workflow.py` ran once in `QA/cpu-acceptance`, with
CUDA initialization and socket connections blocked. Synthetic train/test tensors
are explicit and separate, four examples each; no real-data/quality claim.

| Setting/result | Model 0 | Model 1 |
| --- | --- | --- |
| Seed | 13610 | 13611 |
| Graph stem activation | SiLU | ReLU |
| Device / precision | CPU fp32 | CPU fp32 |
| Epochs / train batches / updates | 1 / 2 / 2 | 1 / 2 / 2 |
| Train loss | 0.6423162222 | 0.7676266730 |
| Train accuracy % | 75 | 50 |
| Actual test accuracy | 0.5 | 0.5 |
| Strict fixed-input output equality | PASS | PASS |
| Source v2 at report creation | verified | verified |
| Legacy P26 checkpoint | missing | missing |

Both checkpoints are loaded into fresh matching models with strict existing
load, metadata validator and expected snapshot SHA. Test inference is real,
through the unchanged Core evaluator. No validation/model selection or extra
training followed the test values. The source objects stayed alive through
comparison/report creation; exported JSON represents those checks at that time.

Checkpoint SHA-256:

- Model 0: `0448158a040132f00a42c69800c00d4d79d39ad7cff360e360f0b7aad2377b46`.
- Model 1: `cb87054c9d284be0225d5ddf78bed4ec9f92df756a213f03aa0c03bd5b77774d`.

Source state SHA-256:

- Model 0: `4180690aff6d72d00a22ae6b52df0a16d864c3d81f3a01e9d5d851dd40e7c84e`.
- Model 1: `a88de44ad6370a187f1ee90c27618f6883f6fd0c968c4f979a229f5e68821eeb`.

Shared ordered paired test-stream SHA:
`1c844474e20547665aa9f731f5f59c09e1233b67662678ea8d710dabbd279519`.
Full metrics/confusion matrices/source hashes are in each evaluation.json and
association.json. After report export, parameter mutation yields missing;
reopening association JSON yields unverified; wrong expected snapshot identity
is rejected. No checkpoint file is rewritten by those checks.

## 14. Comparison and Advisor results

The two actual retained P26 observations are **COMPARABLE** under existing exact
ordered-stream, batch and runtime policies. Matching declared dataset/split/
preprocessing context is an additional guard, not independent origin proof.
Accuracy difference is **0.0**, macro F1 difference **0.0**. No new speed/memory
benchmark is implied. Core claims remain global_winner=None,
statistical_superiority=not_established, acceptance=NOT_EVALUATED.

Different stream/batch/context scenarios return incompatible and no overlay
numerical differences. Missing conditions return insufficient evidence. Missing
source v2 alone does not invalidate otherwise valid legacy differences. Advisor
retains its original per-record observations and scope; it receives no promoted
checkpoint binding and no stronger improvement/architecture claim.

## 15. Reporting and ExperimentManager artifacts

Existing create_report rendered TrainingResults, two associations and comparison
as **JSON, Markdown and static HTML**. It shows source v2 separately from legacy
P26, per-source limited/unavailable association, observed stream and declared
dataset-origin limitations. Imported typed JSON is unverified in reports.
Reports remain presentation_only=True, authoritative_evidence=False. Saved report
status is an observation at creation, not a recreated current receipt on import.

Only new QA stores were created/migrated. Existing Manager records **5 completed
runs** (2 training, 2 evaluation, 1 comparison/report), links checkpoint/training/
evaluation/association/report artifacts and verifies **13/13 normal artifacts**.
An explicitly wrong file returns **mismatch**; the canonical file is separately
reverified successfully. Both the failed negative observation and positive file
checks remain in the summary/history. Artifact byte verification is not native
model provenance. Explicit GraphEvaluationResult.from_json is required to reopen
custom association artifacts; generic workflow restoration does not auto-promote
them into live receipts.

Experiment ID: `502fde3c1ab24aadab3c21e18ab31c6f`. Run/artifact IDs and all
verification receipts are in `QA/cpu-acceptance/summary.json`. User Studio SQLite
was never opened or migrated. No replacement store/reporting engine was added.

## 16. Public API/schema compatibility and regressions

| Successful group | Main tests | Extra observations |
| --- | ---: | --- |
| New integration | 31 | no failure/error/skip |
| Selected Core | 751 | 10 additional subtests, 47 existing warnings |
| Build/install/commercial | 10 | no failure/error/skip |
| **Unique main successful cases** | **792** | subtests/repeated package/candidate checks excluded |
| Installed wheel / sdist new matrix | 31 / 31 | repetitions, not added to unique count |
| Independent staged implementation tree | 31 | repetition, not added |

Core covers all previous 38 Graph checkpoint tests, prior P26 diagnostics,
evaluation/producer/structure schema1/2/3, ModelBuilder, Graph builder/adapter,
TrainingSession/checkpoint, comparison/structure/AUDIT-01, Reporting, Advisor,
ExperimentManager and CLI. The first Core launcher named a nonexistent test path
and ran no tests; its log is retained. The corrected selection passed 751 cases.
JUnit's aggregate includes the ten subtests; actual main testcase elements are
751. Existing GradScaler warnings were not suppressed with unrelated changes.

Before source edits, complete legacy ModelComparisonResult and PresentationReport
outputs were captured from frozen retained schema3 fixtures with fixed import
time. After implementation they hash identically:
`eb66ee58d77155529d83db358cb5e0567fa8882c44ce9d7d9ee10e6664cc7b6e`.
Proof: `QA/legacy-golden-before.json`, `legacy-golden-proof.json` and the permanent
golden test. No valid old output is silently upgraded. CLI --help also exits zero.
Ruff and cached/working diff checks pass for the new/changed implementation scope.
No long native/CUDA P01–P35 suite or additional real-data training was launched.

## 17. Fresh wheel/sdist and installed verification

Fresh distributions under `QA/package-temp/packaging1/dist`:

| Archive | SHA-256 |
| --- | --- |
| hyso-0.2.0-py3-none-any.whl | `e0f59599e1e03daac72d4e7b65cc250bdff1b039fa58955fe5582d37b822beb5` |
| hyso-0.2.0.tar.gz | `1bdf9a6151a9f5d61f104f33e6e962acb294fc2007dfe3d99f128eed7881a5cf` |

All **163 allowed runtime module bytes** match current source. The old 162
modules remain; only graph_evidence.py is added. The ten development exclusions,
dependency declarations, version and profile policy remain unchanged. No web/
Studio/static source asset, SQLite, checkpoint or bytecode enters archives.

Both distributions were installed offline without dependencies/build isolation/
bytecode into fresh QA targets. Imports resolve to those targets, outside the
checkout. Each executes the full **31-case new integration matrix**, including
cross-process trust, positive/negative evaluation/comparison/reporting/Manager
and golden compatibility. Existing user-site pytest dependencies were reused;
none were installed. No training is performed by the installed integration
matrix. Exact archive, install and source-match proof: `QA/package-proof.json`.

## 18. SHA-256 protection and isolation

| Set | Result |
| --- | --- |
| 47 forensic/debug files | 47/47 unchanged |
| 71 native files at authorized post-13.5.9 versions | 71/71 unchanged |
| Existing Graph v2/checkpoint/evaluation/comparison/Advisor/Manager code | unchanged |
| 6 external historical checkpoint/acceptance files | 6/6 unchanged |
| 11 external workspace files, including SQLite/WAL/SHM/lock | 11/11 unchanged |
| 197 repository user database/model/checkpoint files | 197/197 unchanged |
| Prior 13.5.x reports, including 13.5.6–13.5.9 | 15/15 unchanged |
| Pre-task tracked/nonignored untracked files | 734/737 unchanged; three explicit integration/profile/test edits |

No initial file was removed. Only commercial-profile.json, reporting/report.py
and test_packaging.py changed among pre-existing files; added code/tests/docs are
stage-owned. Historical P01–P35 protection is established for manifested
documents/matrices/checkpoints, not claimed exhaustively for all unmanifested
ignored external history. SQLite files were hashed as bytes, not connected.
Original report archive/hash and previous QA evidence remain intact.

## 19. Git commit SHA/parent/scopes

| Commit | Full SHA | Parent | Files |
| --- | --- | --- | ---: |
| Verified 13.5.8 report | `b527ad90e536c8371e03f311b4cbe3a1c94261dc` | `d8691a1a7ab464d14aae4ae9e8fea627982dbf99` | 1 |
| 13.5.10 implementation | `bb748e28c1debb0099692edeaef7ac047aa0a0c6` | `b527ad90e536c8371e03f311b4cbe3a1c94261dc` | 6 |
| Guide/acceptance | this report's documentation-only commit; exact SHA in QA ledger/final response | `bb748e28c1debb0099692edeaef7ac047aa0a0c6` | 2 |

The last commit contains only this report and
`docs/guides/graph-v2-evidence-integration.md`. Its own SHA cannot be included
inside its immutable bytes without a circular reference; it is recorded after
creation in `QA/commits.json` and the final response. No amend/rewrite is used.

Every commit used explicit file lists, name/check/stat/full cached-diff review,
protected-path checks, relevant validation and empty-index checks afterward.
The implementation candidate was exported as an independent tree and its 31
integration tests passed without the unrelated working files or future docs.
Original five commit SHAs/parents remain unchanged; there is no push/rebase/reset/
clean/restore or broad add operation.

## 20. Uncommitted user files

Exactly **47** pre-existing unrelated files remain: two modified tracked and 45
untracked. They are the prior 48-file inventory minus only the newly committed
13.5.8 report. All 47 match their starting bytes and old protection manifest.
No patch/helper/native script was run, stage-added, cleaned or committed.
The complete raw remaining inventory is `QA/remaining-git-status.txt`.

## 21. Open limitations

Full evaluated-receiver v2 runtime/checkpoint binding remains **limited, not
verified**. Native named-state equality cannot establish arbitrary Python runtime
equivalence. Source proof remains the supported typed CPU profile; unsupported
components/customizations never gain verified status. Imported data, copied
results and new processes retain statements only. Dataset/preprocessing IDs are
declared; identical observed tensors do not establish lineage/generalization.

The opt-in comparison adapter covers retained P26 quality, not new hardware
performance measurements or automatic architecture counts. Advisor and existing
performance/structure APIs remain separate. Custom experiment artifacts require
explicit typed reopen and do not restore source objects. Existing private v2
queries may allocate bounded reference models during live status checks.
No untrusted-code sandbox, cryptographic authorship, independent dataset quality,
statistical improvement or CUDA certification is added.

## 22. Final acceptance

**A PASS; B PASS; C PASS; D PASS; E PASS; F PASS**, independently within the
scopes in the opening table. Actual two-model CPU execution, negative controls,
legacy golden, protected-source hashes, isolated current packages and explicit
commit scopes support those decisions. The incomplete inference runtime binding
and legacy Graph checkpoint missing status are not hidden by test counts.

Stop after 13.5.10. No frontend/GUI/service, push or 13.5.11 work follows.
