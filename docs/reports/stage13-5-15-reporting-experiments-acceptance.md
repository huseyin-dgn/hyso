# HYSo 13.5.15 Reporting and ExperimentManager acceptance

2026-11-16, Europe/Istanbul. **PASS within the verified CPU, retained-result,
local-store and installed-package scope.** All A–P gates pass under the scopes
listed below. A literal Markdown label defect was reproduced and corrected in
the unprotected renderer. Strict comparison and native receiver contracts retain
their accepted source hashes.

Evidence root `%LOCALAPPDATA%/hyso-stage13-5-15-20261116`, abbreviated QA.
The [format/API matrix](stage13-5-15-reporting-format-matrix.md) and
[experiment/artifact matrix](stage13-5-15-experiment-artifact-matrix.md) give the
detailed interfaces, projections and negative outcomes. The executable
[QA example](../../examples/reporting/reporting_experiment_acceptance.py) and
[48-case acceptance suite](../../tests/test_reporting_experiment_acceptance.py)
use actual small CPU inference and the existing public APIs.

## 1. Initial audit and preservation baseline

Repository `C:/Users/hdgn5/OneDrive/Belgeler/GitHub/hyso`, branch
`feature/hyso-models-next`. Expected and observed starting HEAD
`b1d2f3715bbbc32357356217137de80fbe3f0380`. Initial index empty. The unrelated
user state is two tracked modifications and 45 untracked files, totaling 47.
No applicable repository/parent AGENTS.md was found.

The 13.5.13 learning, 13.5.14 PARTIAL, 13.5.14 strict closure and 13.5.11 receiver
closure reports were read with the Reporting, ExperimentManager, workflow, CLI
and artifact contracts. `QA/baseline.json` records 763 initial repository paths,
the protected groups and all manifested non-__pycache__ files from the preceding
three QA roots. New stores, temporary files, reports, checkpoints, build and
installation directories are confined to QA. User stores have zero write
connections.

## 2. Reporting API and typed integration

The actual public exports are create_report, report_experiment, PresentationReport,
ReportError and feature_explanations. The format matrix contains all accepted
types and the inspected signatures. `QA/api-inventory.json` records 37 public
signatures and help output for reports, experiments, compare and analysis.

EvaluationResult integrates through EvidenceRecord and EvidenceIndex or a
UnifiedAnalysisReport. TrainingResult, ModelComparisonResult, Graph associations/
comparisons, typed Advisor/recommendation results and retained analysis/execution
results use their existing adapters. Untyped dictionaries, direct EvaluationResult
and direct PresentationReport inputs are rejected by create_report. PresentationReport
has its own export/from_json methods. The inspected runtime defines TrainingResult
as its singular training result class.

Presentation schema1 fixes presentation_only=true and authoritative_evidence=false.
The JSON export includes complete raw sources, metric projections, chart data,
source ordering, metadata, scope, limits and its presentation digest. Native
ModelComparison schema1/strict schema2 and Graph comparison schema1/2/strict
schema3 retain their own formats inside that presentation.

## 3. Numeric/source and format correctness

Two actual QA models consume x=[−2,−1,1,2] and labels [0,0,1,1], with two batches
of two. Model A is Linear with logits [0,x−1.5], model B is a Linear/Identity
Sequential with logits [0,x]. Their accuracies are 0.75 and 1.0. The compatible
accuracy difference is +0.25. The source per-class supports, confusion matrices,
macro precision/recall/F1 and weighted F1 retain their exact typed payloads.

The main report has ten ordered source sections. Twenty scalar values from the
two native evaluation sections match their source metric paths exactly. Full
EvaluationResult dictionaries match the embedded native payloads, including
evaluation ID, sample count, class vocabulary, split, model state, registered
structure and checkpoint identity. Each evaluation EvidenceRecord is constructed
from the actual exported file with its expected SHA. Its artifact SHA equals the
corresponding registered/observed file SHA in the new store.

JSON roundtrips through PresentationReport. Expert Markdown JSON blocks and
expert HTML preformatted raw blocks are parsed back to their source dictionaries
and compared exactly. The tests cover tr/en, original source ordering, duplicate
source deduplication, explicit null versus numeric zero, and byte-preserving
default no-clobber exports. The full field projection matrix specifies which
metadata requires expert/raw mode. The main HTML and experiment HTML were also
parsed for safe elements/links and escaping.

Empty evaluation retains status empty, sample count/support zero and null
accuracy/F1. Scalar metric logging rejects None. These are their respective
source contracts. Values are carried with their recorded units, including the
percentage units on retained training history.

## 4. Literal Markdown defect and minimal fix

The public reproducers use titles `Name *literal*` and `Name _literal_`.
Before production edits, both tests fail because Markdown creates emphasis while
JSON and HTML retain literal labels. `QA/markdown-red.xml` and its log preserve
both original FAIL cases. `QA/markdown-original-probe.py` preserves their original
source. Their test-function AST matches the final version.

The correction is confined to `_md` in `hyso/reporting/render.py`, plus a standard
library re import. Asterisks, backslashes, tildes and underscore delimiter
boundaries are encoded as literal character references. Intraword underscores
preserve ordinary names such as retained_scalar.

The first implementation escaped all underscores. The existing pagination/scalar
regression caught this with 213 passes and one failure. The boundary correction
preserves that original assertion and yields 214/214 final regression passes.
Both failures and the earlier passing development runs are retained in QA and
excluded from the final unique-success count. Other renderer function ASTs match
the baseline. JSON data/digests, HTML rendering logic, legacy golden output,
strict comparison and store schemas retain their existing contracts.

## 5. Strict/legacy and Graph authority reporting

The new tests explicitly select both context policies. A dataset declaration
mismatch is COMPARABLE under the historical generic policy and INCOMPATIBLE
under strict selection. Legacy schema1 results receive their existing presentation.
Strict schema2 policy/version is retained in the raw report. Graph strict schema3
uses the effective result and historical reference. Recursive scans cover
comparison/effective/legacy/metrics/raw report and Advisor payloads. Incompatible
strict quality differences, relative changes and percentage-point changes are
null throughout. Registered structure/count metadata retains its separate scope.

The four original legacy-default strict expectations remain documented FAIL
observations in the preserved 13.5.14 QA evidence. This stage tests the actual
mode distinction and the four explicit strict probes. A strict guarantee requires
explicit strict selection in the Python comparison API.

The live Graph fixture uses four 1×4×4 CPU tensors, supported v2 checkpoint
restore and explicit receiver capture. The three direct report sections carry
verified, limited and unavailable binding respectively. Source contract,
receiver runtime, same-execution inference binding, process-local scope and
legacy P26 checkpoint are recorded separately. Graph P26 checkpoint is missing.

The association file passes expected SHA verification and reconstructs with
unverified source/receiver authority. The second interpreter independently yields
unverified receiver and unavailable binding. Reopened PresentationReport keeps a
historical presentation snapshot with authoritative_evidence=false. Reconstructing
its Graph association uses the actual process-local authority view.

## 6. Experiment lifecycle, migration and persistence

The final main run is `QA/acceptance-final/`. A new schema1 manager is closed,
explicitly migrated through migrate_store, and reopened as schema2. Four runs
are completed through the actual lifecycle APIs.

| Run | Measured contents |
| --- | --- |
| A evaluation | Managed evaluation JSON, checkpoint reference |
| B evaluation | Managed evaluation JSON, checkpoint reference |
| Comparison workflow | Strict schema2 result, two explicit evaluation input links, report/manifest/Graph artifacts |
| Retained training workflow | Existing labeled fixture history/result, zero new fit |

All four run snapshots compare exactly after close/reopen. Strict comparison and
retained TrainingResult reopen through the typed workflow reader and match their
source JSON/payloads. Twelve artifacts verify before close and twelve after
reopen, totaling 24 verified observations. The dataset manifest, checkpoint
references and JSON/Markdown/HTML reports are included. `QA/report-content-proof.json`
and the main summary retain the source/file/store identity checks.

report_experiment produces another JSON/Markdown/HTML set from the reopened store.
It reads verified whitelisted workflow results and lists other artifacts with
their current availability. The typed whitelist is TrainingResult,
AnalysisExecutionResult and ModelComparisonResult. Evaluation, Graph and
PresentationReport files use normal artifact registration and their own readers.

Separate created/running/failed cases retain their original state after reopen
and have completed count zero. Invalid lifecycle transitions fail. Recording a
result has its existing explicit lifecycle semantics. Migration, guarded
transactions, backup equality, foreign-writer fencing, storage failures and
recovery are covered by selected existing regressions.

## 7. Artifact and invalid-data acceptance

The artifact matrix records the actual copy/reference contracts. Tests cover
changed/missing bytes, expected SHA persistence, unknown IDs, same-name different
contents, empty files, relative/absolute paths, same-request idempotency/conflicts,
symlink/junction/path traversal, no-clobber publication and streaming failures.
Managed copies use their registered location. References require their retained
locator or an explicit supplied path.

Five new cases register hash-valid but invalid typed payloads. Malformed JSON,
unsupported schema, edited metric, wrong result type and partial evaluation
accounting return invalid_result and result=None. Wrong typed direct inputs and
comparison result/context mismatches fail before artifact writes. Actual byte
corruption yields mismatch, missing files yield missing, and unsupported reader
kinds yield unsupported. Verified opaque checkpoint bytes do not cause checkpoint
deserialization through open_result.

Expected artifact identity is immutable and verification observations append.
Run states and current artifact availability retain their separate meanings.
Presentation readers validate the top-level schema and supplied presentation digest. Independently
rehashed presentation data retains presentation authority only. Native source
validation and independently registered file hashes supply their own checks.
Workflow dataset/device/protocol labels are caller-declared associations. Result
matching enforces its supported stage-specific checks and comparison tracking
identity. Native context stays in the retained source result.

## 8. CLI and public Python workflow

The four inspected CLI groups and exact options are recorded in the format
matrix. The selected CLI tests cover report formats, evidence context, raw
Advisor source validation, no-clobber paths, schema1/schema2 store history,
verification/opening and missing stores. Module-entry probes exercise help,
saved strict report rendering, typed reopening and an actual Graph-format
rejection. Both installations also render the saved strict result through their
module entry point.

CLI supports rendering a saved strict ModelComparisonResult. The compare command
has its historical default and offers no context-policy switch. Graph association/
comparison and PresentationReport input dispatch are absent. Their Python APIs
provide the supported route. These are explicit CLI backlog items. Verification
mismatch exits 3, unsupported artifact kind exits 4, and invalid/unsupported report
input or unsupported strict flag exits 2 in the measured cases.

## 9. Usability findings for the next consolidated review

A ready typed result and existing output directory need four executable lines.

```python
report = create_report(comparison, language="en", level="expert")
report.to_json(output / "report.json")
report.to_markdown(output / "report.md")
report.to_html(output / "report.html")
```

EvaluationResult requires the explicit record/index adaptation and AnalysisContext.
New artifact-capable stores require create/close/migrate/reopen. The full QA
example has 188 physical lines. Its run function has 99 lines and 102 call
expressions, including checkpoints, exports, lifecycle, hash checks and readback.
These counts describe the audit example's scope.

| Backlog | Concrete finding / proposed follow-up |
| --- | --- |
| Report input guide | Publish the closed typed whitelist and short EvaluationResult recipe |
| Schema guide | Explain presentation schema1, comparison strict schema2, Graph strict schema3 and store schema2 separately |
| CLI policy selection | Expose/document explicit strict selection when approved |
| CLI input parity | Evaluate Graph/presentation dispatch using existing readers |
| Migration recipe | Show the explicit close/migrate/reopen sequence with new-store setup |
| Artifact location guide | Clarify redacted references, retained locators and managed-copy path precedence |
| Identity guide | Distinguish canonical result content SHA from serialized artifact byte SHA |
| Failure guidance | Link mismatch/missing/invalid_result and running-state recovery to exact existing APIs |

The earlier 13.5.12 usability inventory retains its historical content. These
findings are additions for a future consolidated review.

## 10. Regression and installed distributions

| Final group | Passed | Failed | Errors/skips |
| --- | --- | --- | --- |
| New acceptance | 48 | 0 | 0/0 |
| Targeted regression | 214 | 0 | 0/0 |
| Fresh packaging | 2 | 0 | 0/0 |
| Installed wheel | 49 | 0 | 0/0 |
| Installed sdist | 49 | 0 | 0/0 |

There are 264 unique successful source/packaging cases. Installed repetitions are
separate. The installed sets include the 48 new cases and the legacy comparison/
Reporting golden. The source regression selection covers Reporting, workflow,
core lifecycle, artifacts, migration, recovery/storage, four strict probes and
the relevant CLI commands. Exact selectors and XML case identities are in
`QA/regression-selection.json` and `QA/test-proof.json`.

The legacy golden passes with SHA
`eb66ee58d77155529d83db358cb5e0567fa8882c44ce9d7d9ee10e6664cc7b6e`.
The two initial Markdown failures and the initial one-case regression failure
have their own retained logs/XML. QA launcher path/import/encoding corrections
are separate setup fixes. The final selected test groups have zero errors/skips.

Fresh wheel and sdist are built from an isolated current-source snapshot. The
renderer edit requires these fresh archives. Both archives and offline isolated
installs match all 164 source module bytes/SHA-256. All 89 loaded HYSo modules
resolve under the selected install target. Public Reporting/ExperimentManager
imports, strict Graph report, typed roundtrip, artifact verification and legacy
golden pass on each. Existing user-site pygments supplies a test dependency.

| Archive | SHA-256 |
| --- | --- |
| wheel | 2a6ae9b6757b9f9e4c8437967039f0cd2655f9702e5bf733f2601c6a32b5e572 |
| sdist | dd169bcdec2fbea7cc86df2637800c34453fa1cb777284d7c877c8d1428c7c3e |

Archives are under `QA/package-release-tmp/packaging0/dist/`. Fresh targets are
`QA/installed-wheel` and `QA/installed-sdist`. Installation uses
`--no-index --no-deps --no-build-isolation --no-compile` and isolated Python.

## 11. Historical results and SHA preservation

Historical 13.5.13 evaluation and training records read through the existing
typed readers. Sequential remains recorded as 263/270, accuracy
0.9740740740740741, macro-F1 0.9739754790797491. Graph remains recorded as
263/270, accuracy 0.9740740740740741, macro-F1 0.973873618676002.
Original checkpoint hashes and the locked paired-stream identity match. The
historical presentation is presentation-only and imported Graph authority is
unverified. Old strict schema2/Graph schema3 closure JSON also reads successfully.

| Protected group | Matching baseline SHA-256 |
| --- | --- |
| Existing repository files outside the renderer edit | 762/762 |
| Unrelated dirty user files | 47/47 |
| Native files | 71/71 |
| User data/model files | 197/197 |
| Six Core/evidence contracts | 6/6 |
| Forensic files | 47/47 |
| Inherited external groups | 6/6 and 11/11 |
| Stage 13.5.13 QA | 7021/7021 |
| Stage 13.5.14 acceptance QA | 895/895 |
| Stage 13.5.14 closure QA | 1545/1545 |

Groups overlap. The scope is the manifested paths, including prior reports,
P01–P35 evidence and manifested SQLite/WAL/SHM/lock files. The one runtime edit is
`hyso/reporting/render.py`, outside the protected native/Core groups. Its raw
before SHA is `c9613c1fe24a4c8c41d911fe9f0711a391422a353f5809c833b2063b1d75a0cb`
and after SHA is `bfb569f75f672f389079e640bb0edae5f97269cba79efb316f864d751220ee42`.
The other renderer function ASTs and the minimal probe AST match their baseline.

New training fits, dataset downloads, locked-test inference, CUDA/native
benchmarks and 13.5.16 execution each have count zero. The recorded training
workflow run retains the labeled fixture's history.

## 12. Git closure and final gates

The implementation/test commit is `55d4ea5ff695b21659003858515965514a9ec781`,
parent `b1d2f3715bbbc32357356217137de80fbe3f0380`, with message
`fix: preserve report labels and validate lifecycle contracts`. Its exact paths
are `hyso/reporting/render.py`, `tests/test_reporting_experiment_acceptance.py`
and `examples/reporting/reporting_experiment_acceptance.py`. The staged diff SHA
is `c54e0a13d08ae25984a21f5fea2ae5a7c944cce0f5eceb09661a6c60668f7f8f`.
The following documentation commit contains only this report and the two matrices.
Full commit/parent/path records are in `QA/commits.json` and the final handoff.
Exact staged paths, cached diff, whitespace, hashes and test/package
proofs are checked before each commit. The final audit checks ancestry, empty
index, git fsck and equality with the initial 47-file unrelated user state.
Push count is zero.

| Gate | Decision | Verified scope |
| --- | --- | --- |
| A Reporting API Inventory | PASS | Actual typed whitelist, 37 signatures and CLI discovery |
| B Typed Result Integration | PASS | Existing adapters, strict/legacy and retained training |
| C JSON Correctness | PASS | Source equality, schema/digest and null/zero semantics |
| D Markdown Correctness | PASS | Parsed expert source equality and literal escaping correction |
| E Static HTML Correctness | PASS | Safe parsed elements/links, source equality and static SVG |
| F Strict/Legacy Presentation | PASS | Explicit scope/version and recursive strict difference suppression |
| G Receiver Authority Preservation | PASS | Live verified, default limited, imported unavailable/unverified |
| H ExperimentManager Lifecycle | PASS | Explicit transitions, migration and four complete QA runs |
| I Artifact Integrity | PASS | 12 artifacts, 24 main checks and targeted corruption/path negatives |
| J Store Reopen/Persistence | PASS | Exact run snapshots and typed result readback |
| K Invalid/Incomplete Rejection | PASS | Invalid typed artifacts and unfinished/failed states retained |
| L Python API/CLI Compatibility | PASS | Supported CLI subset and explicit backlog boundaries |
| M Historical Schema/Evidence | PASS | Old readers/records/golden and historical authority scope |
| N Regression/Installed Packages | PASS | 264 source cases, 49 per fresh installation |
| O Native/User Data Preservation | PASS | Protected SHA groups match |
| P Git Integrity | PASS | Scoped local commits and closing audit |

**PASS for the measured Reporting/ExperimentManager contracts.** CLI input/policy
gaps, presentation-only digest scope, caller-declared associations and explicit
storage recovery retain their documented limits. This acceptance covers finite
CPU QA and local dedicated stores. Long-duration workload, power-loss and hostile
filesystem/interpreter guarantees require separate evidence.
