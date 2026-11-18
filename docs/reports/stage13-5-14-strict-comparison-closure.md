# HYSo 13.5.14 strict comparison closure

2026-11-16, Europe/Istanbul. **PASS — STRICT OPT-IN SCOPE.**

The four acceptance scenarios pass through explicit strict context selection.
Incompatible strict typed results, JSON and Reporting payloads withhold quality
differences recursively. The historical default continues to implement its
original observed-stream policy. Its four original strict-requirement probes
still fail when invoked through that default.

Evidence root `%LOCALAPPDATA%/hyso-stage13-5-14-closure-20261116`, abbreviated QA.
Python 3.10.18, Torch 2.5.1+cu121, CPU, one Torch thread. New workload inputs are
small independent synthetic tensors. The previous [acceptance report](stage13-5-14-evaluation-comparison-acceptance.md),
[matrix](stage13-5-14-evidence-compatibility-matrix.md),
[learning report](stage13-5-13-real-learning-acceptance.md) and
[receiver closure](stage13-5-11-receiver-closure.md) supply the historical contracts.

## Initial audit and reproduced failures

Branch `feature/hyso-models-next`. Expected and observed starting HEAD
`5677336a72b3c34ccb800bf0347c187c5aacc9b8`. The initial index is empty and the
unrelated user state contains two tracked modifications plus 45 untracked files.
The initial SHA baseline covers 761 repository files, inherited protected groups,
7,021 previous learning-QA files and 895 previous evaluation-QA files outside
__pycache__ directories. Repository/parent AGENTS.md discovery found zero
applicable files.

Before implementation, the original probe module produced four FAIL results in
`QA/original-red.xml` and `original-red.log`. Its byte-identical original is
`QA/original/tests/test_evaluation_comparison_requirements.py`, SHA-256
`729f7575a2524b24cf0eb57ae79ad610bee44983fc71645f2f6943666f2ee10e`.
`QA/original-failure-proof.json` retains the IDs, failure messages, source SHA,
original assertion ASTs and actual baseline comparison observations.
Complete pre-change comparison/report JSON is under `QA/before/`.

All following IDs belong to `tests/test_evaluation_comparison_requirements.py`.

| Test ID | Original default before | Default after | Explicit strict after |
| --- | --- | --- | --- |
| `test_requested_generic_declared_mismatch_rejection[dataset_id]` | FAIL, COMPARABLE with differences | FAIL, same policy | PASS, INCOMPATIBLE/null |
| `test_requested_generic_declared_mismatch_rejection[preprocessing_id]` | FAIL, COMPARABLE with differences | FAIL, same policy | PASS, INCOMPATIBLE/null |
| `test_requested_graph_incompatibility_has_no_nested_numeric_difference[dataset_id]` | FAIL, nested legacy difference | FAIL, same policy | PASS, all quality differences null |
| `test_requested_graph_incompatibility_has_no_nested_numeric_difference[preprocessing_id]` | FAIL, nested legacy difference | FAIL, same policy | PASS, all quality differences null |

The generic root cause was selection of P26 exact-observed-stream policy without
checking dataset/preprocessing declarations. The Graph root cause was retaining
a complete comparable legacy artifact inside an effectively incompatible result.
The new implementation uses the existing native comparator and its typed chain.

## Explicit policy and compatible schemas

```python
from hyso.analysis import compare_models
from hyso.analysis.graph_evidence import compare_graph_evaluations

strict = compare_models(left_source, right_source, features="P26", context_policy="strict")
strict_graph = compare_graph_evaluations(left_graph, right_graph, context_policy="strict")
```

Both entry points default to `context_policy='legacy'`. Unknown policy values
are rejected. Generic strict selection requires P26 among the selected features.

| Contract | Default | Explicit strict |
| --- | --- | --- |
| ModelComparisonResult | schema1 | schema2 |
| GraphComparisonResult | schema1/2 according to source associations | schema3 |
| Context policy | historical native gates | `strict_declared_context_v1`, version 1 |
| Required declarations | historical eligibility | dataset_id, split, preprocessing_id |
| Context authority | recorded native/declared fields | caller_declared_consistency |
| Graph association / EvaluationResult schemas | existing versions | same existing versions |

Strict declarations are reconciled across the selected EvidenceRecord scope,
its report's `context.dataset`, and the recorded native evaluation split label.
Every supplied representation must agree. Known cross-source differences or
within-source conflicts produce INCOMPATIBLE. Missing required declarations
produce INSUFFICIENT_EVIDENCE. Known incompatibilities take precedence over
missing information. Invalid non-string/blank declarations raise a validation
error. The native task/class/stream/count/batch checks continue to determine
eligibility independently.

Matching identifiers establish declared consistency. Dataset origin,
preprocessing implementation, split membership and producer authorship retain
their existing evidence limits. Native state/checkpoint/stream bindings retain
their original authority. Strict selection records its policy/version in the
result and the tracking policy string.

## Generic comparison implementation

`hyso/analysis/model_comparison.py` adds one context check around the existing
P26 comparator. When context rejects a pair, each P26 row has both `difference`
and its nested native `comparison` set to null. Eligibility reasons identify the
missing, conflicting or mismatched declaration. Global winner stays null.

With compatible context, the complete native metric rows match the corresponding
legacy rows exactly. A controlled actual Linear inference fixture yields
accuracy 0.75→1.0, difference +0.25 and +25 percentage points under both paths.
Different weights and a different Sequential wrapper architecture both pass
the strict positive cases. Registered structure results retain their own scope.

The schema2 reader rederives its result under strict policy and checks the entire
canonical envelope. Rehashed context, policy, schema, difference or nested
comparison inconsistency is rejected. Existing schema1 records select their
historical derivation on read.

## Graph effective result and historical reference

`hyso/analysis/graph_evidence.py` adds explicit schema3 comparison envelopes.
Their `effective` field holds the existing typed ModelComparisonResult under
strict policy. For incompatible or insufficient quality rows, nested native
comparison and every quality-difference value are null.

The `legacy` field is a versioned `hyso.graph_legacy_comparison_reference`.
It records the original legacy comparison's canonical `comparison_sha256` and
`export='reference_only'`. Its two metric entries are value-free export markers
with `status='NOT_EXPORTED'` and `difference=null`. These markers enumerate
accuracy and macro-F1, allowing the original nonempty nested assertions to run
against the new reference representation. They contain zero legacy observations
or numeric differences. This reference has its own explicit format and schema.

The original legacy typed artifact and hash are saved separately in
`QA/closure-evidence/historical-quality-dataset_id.json` and
`historical-quality-preprocessing_id.json`. The new reference hashes match those
artifacts' original canonical content hashes. File artifact SHA-256 is recorded
separately by ExperimentManager.

For schema3, `legacy_comparison` returns None and `effective_comparison` returns
the validated strict typed result. Reporting delegates that effective result to
the existing renderer. Schema1/2 readers, exports and typed legacy access retain
their existing behavior. Four pre-change serialized default results round-trip
byte-for-byte through the new readers on source and both installs.

## Probe adaptation and negative matrix

The committed probe module adds only explicit strict call helpers around its
original test bodies. Both complete test-function ASTs, including assertions and
decorators, match the original. The adapted module SHA-256 is
`09bf79ab369a0b8252b469309d8a3ad6bc7707142eea099a70def152ce891447`.
`QA/compatibility-proof.json` contains both source hashes and the AST comparison.
The original module also runs after implementation and records four default
FAIL results in `QA/legacy-default-after.xml`.

The [new security suite](../../tests/test_strict_comparison_context.py) contains
57 selected cases. Its input classifier stream is x=[−2,−1,1,2], labels
[0,0,1,1], classes [a,b], batches [2,2]. Graph cases use four 1×4×4 tensors from
arange(64)/64, labels [0,1,0,1] and captured supported CPU checkpoints.

| Scenario | Observed result / invariant |
| --- | --- |
| Known dataset/preprocessing/split mismatch | INCOMPATIBLE, recursive quality differences null |
| Dataset/preprocessing missing on both or one side | INSUFFICIENT_EVIDENCE, recursive differences null |
| Known mismatch plus missing metadata | INCOMPATIBLE precedence |
| Scope/report conflict or invalid declaration value | INCOMPATIBLE or validation rejection |
| Equal declarations, changed input/target/order/count | Native INCOMPATIBLE |
| Equal stream, changed vocabulary/order/task/batch | Native INCOMPATIBLE |
| Same context, changed weights/architecture/metrics | COMPARABLE through original native math |
| Missing producer with matching declarations | INSUFFICIENT_EVIDENCE |
| Dataset/preprocessing JSON edits and outer rehash | Reader rejection |
| Rehashed policy/schema/reference/native comparison inconsistency | Reader rejection |
| Inject full legacy artifact into strict Graph JSON | Reader rejection |
| Graph mismatch/missing context, recursive typed/JSON scan | Every quality difference null, including numeric zero detection |
| Reporting JSON/Markdown/HTML | INCOMPATIBLE retained, quality difference values null, charts empty |
| Registered structure on incompatible quality pair | Structure metadata/counts retained independently |
| Imported Graph strict result | Source/receiver authority unverified |
| Fresh ExperimentManager typed comparison, compatible/incompatible | Exact schema2 result roundtrip, verified artifact, idempotent recording |

Recursive checks inspect comparison, effective, reference, raw report and Advisor
payloads. They explicitly detect the old Graph disclosure before checking the
new result. Registered structure/count fields retain their distinct metadata
scope. The old 69-case arithmetic/evidence suite is also executed.

## Reporting, experiments and native receiver boundaries

`QA/closure-evidence/` contains complete default/strict comparisons and existing
Reporting JSON, Markdown and static HTML outputs for both mismatch fields.
`QA/closure-proof.json` records two scenarios, recursive checks and 20 verified
artifact/result observations in a new QA-only store. These include three strict
typed comparison results, of which one is compatible. A wrong artifact copy
returns mismatch.

Fresh live supported Graph captures report verified source, receiver and
same-execution binding. Default receiver captures report limited. Imported strict
comparison JSON reports unverified source/receiver authority. Graph legacy P26
checkpoint is missing. The native receiver mutation/subclass/hooks/lifetime
regressions pass.

The five protected Core files match their baseline SHA-256. Seven
Graph capture/context/export helper ASTs match their baseline, including
GraphEvaluationResult and evaluate_graph_classification. The Graph exception
is confined to comparison policy, result/export schema and Reporting adaptation.

## Regression, goldens and fresh distributions

| Evidence group | Passed | Failed | Errors/skips |
| --- | --- | --- | --- |
| original-red, historical default | 0 | 4 | 0/0 |
| legacy-default-after, original source copy | 0 | 4 | 0/0 |
| new-final, strict probes + new matrix + prior arithmetic/evidence | 130 | 0 | 0/0 |
| selected regression | 385 | 0 | 0/0 |
| fresh packaging | 2 | 0 | 0/0 |
| installed wheel | 131 | 0 | 0/0 |
| installed sdist | 131 | 0 | 0/0 |

There are 517 unique successful source/packaging cases. Installed repetitions and
the recorded original default failures are reported separately. The source
selection covers EvaluationResult schemas and native producer/structure evidence,
ModelComparison, P26 Advisor policy, Graph associations/receiver, full Reporting,
ExperimentManager workflow, artifact checks and architecture/structure boundaries.
Exact selectors, test IDs, XML and SHA values are under `QA/regression-selection.json`
and `QA/test-proof.json`.

The legacy comparison/Reporting golden is
`eb66ee58d77155529d83db358cb5e0567fa8882c44ce9d7d9ee10e6664cc7b6e`.
It passes on source and both installed targets. Four additional pre-change
default JSON byte goldens pass on all three targets.

Fresh archives are built from an isolated current-source snapshot by the existing
packaging tests. Both archives and both fresh installations contain exactly 164
runtime Python modules whose bytes/SHA-256 match the current source. Installations
use `--no-index --no-deps --no-build-isolation --no-compile` and isolated Python.
All 81 imported HYSo modules resolve inside each target. Existing user-site
pygments supplies a test dependency. Package proof records archive paths and all
164 individual module hashes.

| Archive | Fresh SHA-256 |
| --- | --- |
| wheel | 0ed87557eb62b253e7e78e88698a6eacb7bc7153c28a59e57a6775f851c86cf8 |
| sdist | 5d2ded021d2726aeda8a255265f581e791d20716fed1c07ac3c9216f9b1b741a |

Targets are `QA/installed-wheel` and `QA/installed-sdist`. The same 130 relevant
tests plus the legacy golden run in each installation. The second-interpreter
Graph authority test also selects that installed package.

## Historical learning and SHA preservation

Saved evaluation matrices still account for sequential 263/270 and Graph 263/270.
Both recorded paired streams have SHA-256
`1be2da286c129a4c7b9691277184591717d40ca57e2b106cea284f217df692dc`.
The selected sequential checkpoint SHA is
`cddc0ae202ddaaa7347a6d7879867e34487553cf4b858ef27c6947714526ebcc` and Graph SHA is
`e9972bd496b442d7e05ec9a4f19f5c098d77317f357c398dbdf82c29e6f4c5e4`.
New training fits, locked-test inference, CUDA/native benchmarks and next-stage
execution each have count zero.

| Protected group | Matching baseline SHA-256 |
| --- | --- |
| Existing repository files outside three authorized edits | 758/758 |
| Unrelated dirty user files | 47/47 |
| Native files | 71/71 |
| Forensic files | 47/47 |
| User data/model files | 197/197 |
| Protected Core files | 5/5 |
| Inherited external groups | 6/6 and 11/11 |
| Stage 13.5.13 QA | 7021/7021 |
| Previous stage 13.5.14 QA | 895/895 |

Groups overlap. The baseline includes prior reports, manifested P01–P35 and
SQLite/WAL/SHM/lock files. The scope is the manifested paths. The two authorized
runtime edits have separate before/after raw SHA records in preservation.json.

| Authorized runtime file | Before SHA-256 | After SHA-256 |
| --- | --- | --- |
| model_comparison.py | 5c3f828e39292d6d43507e2d392e81aa9a187eca14d038df1334cf014ad3451d | 8e6604bc65023c55cd6cd2aa4f053b3d1d81b87b0ff941b12b191f916748e4b5 |
| graph_evidence.py | 4cfe06657951513c8b19975fee1ad6868a7c38ec6ab2002787f1e1046e9d7057 | 1f7638c7ed71b36cf303b411f77be26e6a06a167565c67694c2b55775421e7ab |

## Git closure

The implementation commit is `b783b14f53966213f47f47671fe828ba741f7545`, parent
`5677336a72b3c34ccb800bf0347c187c5aacc9b8`, with message
`feat: enforce opt-in strict evaluation comparison context`. Its exact paths are
`hyso/analysis/model_comparison.py`, `hyso/analysis/graph_evidence.py`,
`tests/test_evaluation_comparison_requirements.py` and
`tests/test_strict_comparison_context.py`. The staged diff SHA-256 is
`364dfd41e9d4557e2709a75eae905cda02fdac514938291e7d7343a7f746e5d4`.
The following documentation commit contains only this closure report. Its own
full SHA/parent are retained in `QA/commits.json` and the final handoff.

Each commit is checked for exact paths, cached diff, whitespace and protected
hashes. The closing audit verifies initial ancestry, empty staging, git fsck and
the original 47-file unrelated user status. Push count is zero.

## Gate decisions and limits

| Gate | Closure decision | Scope |
| --- | --- | --- |
| A Metric correctness | PASS | Original native arithmetic and independent references |
| B Binary/multiclass | PASS | Existing explicit task contracts |
| C Confusion/accounting | PASS | Native sample, order and batch invariants |
| D Producer evidence | PASS | Trusted producer and existing native receipt boundary |
| E Context/stream/class integrity | PASS — strict opt-in | Required declared consistency plus native eligibility |
| F Model comparison | PASS — strict opt-in | Existing math and explicit context policy v1 |
| G Incompatible evidence rejection | PASS — strict opt-in | Four original scenarios and recursive difference checks |
| H Graph receiver authority | PASS | Existing supported CPU profile/lifetime |
| I Architecture/structure/quality | PASS | Independent registered metadata scope |
| J Reporting/ExperimentManager | PASS — strict opt-in | Effective payload rendering and typed schema2 storage |
| K Legacy schemas/goldens | PASS | Historical defaults and serialized bytes |
| L Source/installed regression | PASS — strict opt-in | 517 source successes, 131 per fresh install |
| M Data preservation | PASS | Manifested protected SHA groups match |
| N Git integrity | PASS | Scoped local commits and closing audit |

**PASS — STRICT OPT-IN SCOPE.** The legacy generic default supplies its historical
observed-stream guarantee and can still accept differing dataset/preprocessing
declarations. The legacy Graph envelope can still retain its historical nested
numeric comparison. Choosing the new strict policy is required for the stronger
closure guarantee. Dataset/preprocessing authenticity and independent statistical
superiority require their own evidence. The closure supplies declared consistency,
native eligibility and safe strict quality-difference export.
