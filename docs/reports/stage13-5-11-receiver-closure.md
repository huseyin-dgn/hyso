# HYSo 13.5.11 — Actual receiver runtime verification closure

Date: 2026-11-14, Europe/Istanbul. Evidence root (outside the repository):
`%LOCALAPPDATA%/hyso-stage13-5-11-closure-20261114` (`QA/` below).

| Independent acceptance | Decision |
| --- | --- |
| A — Actual receiver identity | **PASS** |
| B — Receiver runtime verification | **PASS — existing supported CPU contract** |
| C — Source-to-receiver inference binding | **VERIFIED — explicit opt-in, actual native execution** |
| D — Legacy P26/v2 compatibility | **PASS** |
| E — Reporting/Comparison | **PASS** |
| F — Git integrity | **PASS — scoped local commits and preserved history/data** |

The new authority is explicitly opt-in and bounded to the existing supported
Graph v2 CPU profile and trusted, quiescent installed Core/PyTorch. Default calls
and old association files retain their LIMITED meaning. Legacy Graph P26
checkpoint evidence remains **missing**.

## 1. Initial state and HEAD

Branch: `feature/hyso-models-next`. Starting HEAD:
`ab9e74c17235eb1ad9e0760a867cb7c04de7acbe`.
Index was empty, with **2 modified tracked + 45 untracked = 47** unrelated files.
`QA/baseline.json` records all **743** tracked/nonignored untracked hashes,
status/index, preceding nine commit/parent records, native/forensic sets, external
files and user data. All 743 matched the previous final manifest before changes.
The dirty/forensic 47 are also separately recorded in `QA/protected47.json`.

The previous PARTIAL report and section 6 proposal were read and compared against
the actual source. No applicable AGENTS.md was found in the repository/parent
chain. Test files, builds, logs, plots, model/checkpoint files, experiments,
TEMP/TMP and Matplotlib state were isolated under QA, with no pytest cache and
`python -B`. User databases were byte-hashed, never connected to or migrated.

## 2. Previous PARTIAL reason

The previous public adapter could query only the source Graph receipt. Core
created a local deepcopy `candidate` and called it for inference without exposing
its full v2 runtime observation. Schema3 described actual registered structure,
which did not establish source/candidate behavior equivalence.

The previous report remains unchanged as historical evidence. This closure adds
the missing actual-candidate observations rather than promoting that old report,
its JSON or another independently prepared model into new authority.

## 3. Reviewed narrow Core proposal and authorization

The closure request explicitly authorizes **hyso/evaluation.py only** among
existing protected Core files. That exception is exercised here. No further
Core/native exception was inferred.

The earlier proposal suggested extracting a shared executor. Actual code allowed
a smaller change: keep its existing single loop and add fixed Core-owned
observations around that same local candidate. There is no caller observer
callback, alternate evaluator, second inference for one result, new Trainer,
checkpoint loader or reporting engine.

Precision about the historical manifest: evaluation.py is not one of the 71
native paths. It is tracked separately under this explicit Core exception;
**all 71 historical native paths remain unchanged**. checkpoint.py,
_evaluation_evidence.py, _evaluation_structure.py, Graph forward and Trainer
remain byte-identical to the starting versions.

## 4. Core change and Python integration

`evaluate_classification` adds one keyword-only option:
`capture_receiver_runtime=False`. Existing parameters and defaults remain usable.
True requires native evidence, CPU execution, a live supported source v2 receipt
and complete nonempty inference. Unsupported or mismatched observations raise
ValueError/NotImplementedError and publish no receiver authority.

The existing loop observes candidate after deepcopy/eval, before and after each
actual forward, and immediately before successful publication. Existing loader,
alias, named-state, eval-mode, structure, accounting and RNG/generator restoration
checks remain. The classification result is constructed once by the existing
_result implementation. Its schema1/2/3 format is unchanged.

EvaluationResult gains only a weak-reference slot so the new capture can expire
with the exact immutable result without a strong ownership cycle. No receipt is
stored in model attributes or added to legacy EvaluationResult JSON.

New private module: `hyso/_graph_receiver.py`. It reuses the unchanged native
_graph_contract, source receipt query and named-state hashing; its additional
logic handles actual-receiver identity, execution lifecycle, storage isolation,
installed-profile guards and process-local ownership. It does not read/load a
checkpoint or populate either existing checkpoint receipt registry.

`evaluate_graph_classification(..., capture_receiver_runtime=True)` calls the
existing evaluator once and produces GraphEvaluationResult association **version
2** with a separate `hyso.graph_receiver_binding` **version 1**. Default adapter
calls still produce association version 1. GraphComparisonResult version 2
supports mixed old/new and new/new associations; all-old remains version 1.

Changed implementation paths: evaluation.py, the Graph evidence adapter,
commercial-profile.json, the existing short example and packaging expectations.
The new module and independent receiver tests are added. No new dependency.

## 5. Actual receiver identity observation

The same local `candidate` used in `candidate(inputs.clone().to(device))` is
supplied to the fixed observer on both sides of every call. The observer retains
that exact object and rejects any different object, even with equal state and
configuration. Source identity is independently held through a checked weak
reference. Published source/receiver tokens are distinct opaque capture-local
identifiers, never caller-provided addresses accepted as authority.

Tests observe the actual Core candidate through scoped test-only instrumentation:
two batches produce before/after/before/after on one receiver distinct from the
source. Its checkpoint query remains missing. This instrumentation never creates
production authority or registers a fake receipt. The CPU acceptance itself uses
the ordinary uninstrumented production path.

## 6. Pre/post inference evidence

The recorded phases are `after_deepcopy_eval`, `before_each_forward`,
`after_each_forward`, `before_publication`. Every boundary independently checks
the current source receipt, the exact receiver, full reconstructed receiver
contract, named state and eval modes. Complete successful stream/sample/batch
accounting is required before the registry writer publishes a capture.

Each binding contains evaluation id and full evaluation digest, checkpoint SHA,
source/receiver named-state SHAs, receiver contract, actual ordered stream,
batch/sample counts and independently checked pre/post contract commitments.
Empty, failed or partial inference cannot publish a binding. The result and its
native producer evidence belong to this same single execution.

Final CPU schema3 observations and complete binding records are retained in
`QA/cpu-release-proof.json` and `QA/cpu-release/`. They were freshly evaluated using
strictly restored checkpoints from this closure, without repeating training.

| Final native evaluation | Source object token | Actual receiver token |
| --- | --- | --- |
| 191ca7c7-4996-4220-bade-be30d0e17430 | 470a3bc5bc1248149fdf381e6aec1d90 | aec0d1947a734bf390d0e50f3e4e61ca |
| 4951ec69-cf6f-4d84-8f2f-ec4be372a489 | 80c8a3e22cad4beb953b4fd1afa9e7ae | 72695789119b46f0ba875be078ea1f99 |

Each has two forward batches/four samples. Its pre/post full-contract SHA is
equal: respectively `b5a16058bc24038a8aea9356642b752cd1efd73c2a04a1243b7b20c193faac45`
and `85e8f85f5f37aad62b1cb5c0dd8ec166928c337e3086ee7c947c6596a16185ef`.

## 7. Runtime equivalence verification

Source and actual receiver must independently match the unchanged v2 full
contract: Graph configuration, topology/node order/connections, constructor
attributes, activation, adapter output, CPU dtype/device, parameter schemas,
persistent serialized state and the supported nonpersistent empty Graph anchor.
Named-tensor equality alone is insufficient.

The observer also conservatively requires contiguous owned tensor storage with
zero offset and exact byte capacity. It compares **byte intervals**, rejecting
internal partial overlap and cross-source/receiver overlap, as well as shared
registered objects. Empty anchors have zero storage bytes. No shared/view storage
profile expansion is made.

Installed callable identity/code/defaults and relevant class behavior constants
are checked relative to trusted import. Unknown subclasses, hooks, instance
forwards and observed installed runtime changes fail closed. Profile guards do
not certify a hostile interpreter or self-modified Core before trusted bootstrap.
The installed constructor table, registry specification tuple and exact installed
module types are pinned independently of a source receipt. A source receipt
obtained after constructor-table tampering cannot redefine the receiver profile.

## 8. Source-to-receiver binding and trust lifetime

Three independent views remain explicit: source Graph v2, actual receiver runtime,
and same-execution inference binding. Source receipt is never transferred to
candidate. Its raw status `native` and the adapter's live `graph_v2=verified` view
refer solely to the source checkpoint contract.

VERIFIED binding requires the live source receipt, exact source object, matching
checkpoint/state, independent actual receiver contract, storage isolation,
successful same-receiver calls and matching native result/stream/accounting.

The exact immutable native EvaluationResult owns a process-local capture through
a weak key. A live Graph association keeps that result/isolated receiver alive;
source ownership stays weak. Current views revalidate both. Source loss or observed
source/receiver mutation makes live binding unavailable. Registry entries and the
isolated model expire with their owning results; lifecycle tests verify cleanup.

Constructors, copies, imported JSON, another model and another interpreter cannot
inherit this registry identity. Retained `native`/`verified` text and integrity
digests are statements, not live authority. A structurally valid imported v2
association shows receiver_runtime=unverified and no verified live binding.

## 9. Legacy P26 and schema compatibility

Graph legacy P26 checkpoint remains **missing** in schema2/3. Source Graph v2
format/profile/receipt API remain unchanged. Association version 1 retains its
old limited/unavailable semantics; old records are read without migration.

The starting Core blob was independently read from Git, checked against its
original SHA and compiled under a separate module name for a compatibility
comparison. With fixed result UUID and identical input/model, **default schema
1/2/3 dictionaries matched exactly**. `QA/default-golden-proof.json` records:

| Schema | Equal-output SHA-256 |
| --- | --- |
| 1 | 88088483169c0c354968b5b5d4a1a76b2e8cbd3ba516c6e5d40914c7ae2fbcb3 |
| 2 | be8413ab66ec31e9f8924cb150a5aa4a1833f479e167d2ec36453e463eec692a |
| 3 | ce16457d27cd98754ae820b1145c4e088d79a04d865be8121b1c4f1a52bae685 |

The existing retained comparison/presentation golden remains:
`eb66ee58d77155529d83db358cb5e0567fa8882c44ce9d7d9ee10e6664cc7b6e`.
Original schema/hash, P26, structure and Advisor policies pass their regressions.
No performance or statistical recommendation policy gains authority from runtime
verification.

## 10. Negative tests and the two old acceptance probes

The exact original probe source/log/XML were copied to QA and preserved. Both
failures were reproduced before production edits (`QA/old-red.xml`). Independent
new tests also failed on the missing opt-in API before implementation.

| Historical probe | Before | Untouched source after | Explicit opt-in assertion |
| --- | --- | --- | --- |
| Receiver capture option exists | FAIL | PASS | PASS |
| Actual receiver binding is verified | FAIL | FAIL: its helper still selects legacy defaults | PASS on actual opt-in receiver capture |

The second literal old test expects the default helper to upgrade version 1 to
verified. That expectation is intentionally not satisfied, because default legacy
behavior is preserved. Its failure is recorded, not skipped or counted as passed.
`QA/test_receiver_acceptance_optin.py` keeps the original two assertions and
explicitly opts in through its capture helper; **both pass**. The production suite
separately asserts that default calls stay version 1 / limited.
`QA/probe-port-proof.json` verifies the original file SHA and equality of both
test-function ASTs; the explicit capture helper is the only semantic port.

The receiver matrix covers actual identity; topology/activation/output/geometry;
fp32/fp64 and unsupported device/input; parameter/persistent/nonpersistent mutation;
source/receiver pre/during/post changes; subclass/local/global hooks; callable
changes; wrong object; complete/failed/partial inference; exact expected SHAs;
stream mismatch; other execution statements; deepcopy, JSON, process and lifetime
authority loss; storage equality/partial overlap; and old schema/P26 behavior.

Security-relevant intermediate failures are retained:

- First publication failed because legacy EvaluationResult lacked weak-reference
  support; the authorized Core slot resolves lifetime ownership.
- Test fixture corrections handle an unresizable empty anchor and the legacy
  FrozenInstanceError on EvaluationResult deepcopy; neither mints authority.
- Exact Tensor subclasses are rejected by the existing stream validator with
  NotImplementedError; the negative test accepts that fail-closed outcome.
- A dedicated partial-overlap probe initially failed with DID NOT RAISE. Distinct
  storage base pointers were insufficient. Byte-interval checks fixed it; earlier
  archives were superseded and final packages rebuilt/tested.
- Two constructor-table probes also initially failed with DID NOT RAISE: the
  unchanged source API could reconstruct a reference from a changed table. The
  receiver layer now independently pins installed constructors/specifications
  and exact module types. Both probes fail closed without changing source v2.
- A QA archive-install harness accidentally shadowed its generation prefix with
  the sdist member prefix. Its log-path failure was corrected in QA; no production
  change or user path was involved. Failed and corrected logs remain available.

| Validation | Successful main cases |
| --- | ---: |
| Existing Core/P26/evaluation/comparison/reporting/Advisor/experiments/builders/training/checkpoint/architecture/AUDIT-01/CLI | 782 |
| Previous QA receiver-boundary suite, unchanged assertions | 13 |
| Final actual receiver security suite | 64 |
| Packaging assertions: inventory/metadata/public typing/installed CPU models | 6 |
| Distinct successful main cases above | **865** |

There are **10 additional passed subtests**, excluded from 865. The two explicit
opt-in acceptance probes also pass and are reported separately rather than added
to the main count. Existing Core has 47 warnings, principally existing AMP
deprecations. All 38 Graph v2 contract and all 31 existing integration tests are
included in the 782; installed repeats are not added again.

QA XML/logs retain original failures and corrected outcomes. JUnit aggregate
counts include subtests, so the summary counts actual testcase identities.

The final storage/profile changes affect only the opt-in observer; evaluation.py
and its default branch are unchanged from the successful 782-case Core run. The
entire final receiver suite and installed package matrix cover that observer.
No CUDA/native P01–P35 long acceptance suite was started.

## 11. Short CPU acceptance

The updated existing example was executed once with the explicit receiver flag.
Two small branching Graph CNNs used CPU fp32, seeds **13610/13611**, four explicit
synthetic training samples and four distinct test samples, batch two, one
TrainingSession epoch and **two optimizer updates per model**. There is no real
dataset-quality or CUDA-performance claim.

Both fresh strict v2 restores checked checkpoint SHA and metadata. Both source,
receiver runtime and inference binding views verified at report creation.
Accuracy was **0.5 / 0.5**, comparison COMPARABLE, no global winner/statistical
superiority, and legacy P26 missing. Source mutation then produced source=missing,
receiver_runtime=missing and inference_binding=unavailable.

Closure checkpoint SHAs:

```text
5f959abad8ec347f5c16208298ddb7e8859f664725008f7bd2c96d395b2caa2d
e8291fe54bed88d75ef55a0cf4f0e9d693c6a627ef64f05bc91593a34268784c
```

After final observer hardening, the same newly trained checkpoints were strictly
restored into fresh models and evaluated with the final observer and schema3.
Both actual bindings verified again, with two actual forward batches each, equal
pre/post full-contract commitments and native ordered stream evidence. This is
a new evaluation with its own ids, not evidence attached to the earlier run.
**Training was not repeated. CUDA remained uninitialized.**

## 12. Comparison, Reporting and ExperimentManager

Comparison delegates quality math to existing retained P26 comparison. New
runtime evidence is shown per source. Two verified receivers on different streams
still yield INCOMPATIBLE with differences withheld. Mixed version 1/2 associations
retain their independent limited/verified/unavailable meanings. Advisor remains
unchanged and cannot infer global superiority or acceptance from these receipts.

Existing Reporting creates consistent JSON, Markdown and static HTML for the
new typed associations/comparison. It remains presentation-only. Imported JSON
does not recreate native authority, even when artifact bytes verify.

The initial CPU workflow used five completed runs and **13 verified artifacts**
in its fresh QA ExperimentManager store. A wrong artifact copy mismatched and the
original reverified. The final no-training revalidation used a separate fresh QA
store with three completed evaluation/report runs and **eight verified artifacts**.
Its final release experiment id is `ee80ccc529f2401cb4acce2fe0520a9e`.
Both use the existing Manager implementation and explicit typed association
artifacts; there is no second experiment engine or automatic receipt restoration.

## 13. Wheel/sdist and installed verification

The commercial profile is explicitly **163 → 164** runtime Python modules. Only
`hyso/_graph_receiver.py` is added; existing exclusions and dependencies stay
unchanged. No Studio/frontend/static web asset, SQLite/checkpoint or bytecode
file is admitted to either distribution.

Final release archives under `QA/package-release/dist/`:

| Archive | SHA-256 |
| --- | --- |
| wheel | 3afec059b9dd470c6a7fd41d363c6d4c5f4f9aae1d636edef697a2a1febf5789 |
| sdist | 4bde2fbc991ca3de6ac5ed6efc38c4efc82a28c60858e8f8693c7855fc0b9e5e |

**95 tests pass on each installed release archive**: all 64 final receiver tests
and 31 retained Graph integration tests. Final archive/configuration/metadata
assertions and all 164 module-byte comparisons pass. The six packaging assertion
bodies were exercised on the post-storage-hardening archives; after the additional
opt-in constructor guard, current content/metadata and installed matrices were
revalidated on freshly built release archives. Default Core/public API paths did
not change during that final observer-only hardening.

Final installations use fresh isolated targets with `--no-index --no-deps
--no-build-isolation --no-compile`. Installed tests assert hyso was imported from
that target, not the checkout. Existing user-site pygments is used only as an
already-installed test dependency, with no package download/install.

Both archive module sets and every runtime module's actual bytes are compared
against current source. The old pre-hardening archives remain QA history and
are not used as final source proof. Packaging metadata/public typing/CPU model
smokes and installed console/module CLI help are validated independently.

## 14. Core/native and user-file preservation

Authorized evaluation.py SHA-256 change:

```text
before b7971fdb6c824e7450f687fdd5dc5823b8acc6834d9571b83646948bd35535d6
after  06f3d3abb17241c482d6ecd00a6cf8e193a645aa34398608f5e0454897922a47
```

All **71 historical native files** retain their authorized starting bytes.
_evaluation_evidence.py and checkpoint.py are unchanged. Other approved source,
example/profile/test changes and the guide are listed by exact Git path; no
unrelated source change is hidden behind the evaluation exception.

Before/after QA manifests separately verify the 47 dirty files, 47 forensic
entries, six external acceptance/checkpoint files, 11 external workspace/raw
files, 197 repository model/checkpoint/database files, historical stage reports
including the original 13.5.11 PARTIAL report, and previous commit ancestry.
No user SQLite/WAL/SHM/lock was connected to, migrated or edited. Historical
P01–P35 preservation is claimed for the manifested files and unchanged history,
not an unbounded filesystem inventory or a new acceptance run.

## 15. Git commits and integrity

Implementation commit: **a1b4d3e17474c5421865489adb3e24ea0c7c559e**.
Parent: `ab9e74c17235eb1ad9e0760a867cb7c04de7acbe`.
Message: `feat: verify actual Graph inference receiver runtime`.
Exactly seven paths:

- commercial-profile.json
- examples/models/graph_evidence_workflow.py
- hyso/_graph_receiver.py
- hyso/analysis/graph_evidence.py
- hyso/evaluation.py
- tests/test_graph_receiver_runtime.py
- tests/test_packaging.py

The unchanged `* text=auto` Git attribute normalizes CRLF in the profile JSON and
packaging test when staging; normalized content was checked against working
bytes. All runtime Python bytes in the release archives match actual current
source bytes, including the final observer SHA:
`98740205a213ea056966ee0adcb7ff94cad6a2756fbd477f3d1ef6ee0a227a3d`.

Implementation/test/example/profile changes are one explicitly staged commit;
the updated guide and this new closure report form a separate documentation
commit. Cached names/check/stat/full diff and preserved-file hashes are checked.
No unrelated file is staged. Full commit/parent/path ledgers and post-commit empty
index checks are recorded in `QA/commits.json`. The documentation commit's own SHA
is recorded there and in the final response, without amending this report.

The earlier PARTIAL report and commit are retained. Remaining working-tree files
are exactly the initial 47 unrelated files with their initial hashes. No reset,
clean, restore, rebase, amend, bulk add or push is performed.

## 16. Explicit limitations

- Authority is process-local and tied to actual immutable result/object ownership;
  JSON/file hashes are not remote attestation or proof of producer authorship.
- Trusted installed Core/PyTorch at import and quiescent supported objects are
  required. Arbitrary private-registry/code tampering and hostile concurrent
  Python are outside the contract.
- Boundary/per-forward observations cannot exclude transient mutate-and-restore
  behavior entirely between observations. No universal Python sandbox is claimed.
- Candidate lifetime is retained while its live association owns the native
  result; the caller must keep the original source alive for current verification.
- Runtime equivalence is the unchanged supported Graph v2 CPU contract with
  conservative additional isolation checks, not universal Graph/CUDA support.
- Default/old associations intentionally remain limited. Checkpoint confidence
  does not authenticate dataset origin, preprocessing, quality or superiority.

## 17. Final acceptance

**13.5.11 technically closed for the explicit supported CPU opt-in path.**
A PASS; B PASS; C VERIFIED; D PASS; E PASS; F PASS, independently and within
their stated scopes. Actual source and receiver contracts were observed on the
same successful native inference execution. Legacy P26 remains missing, old
association defaults remain limited, and retained JSON has no new authority.

The literal historical second probe still fails when selecting the legacy default;
its explicit opt-in counterpart passes. This compatibility distinction is disclosed
in section 10 and is not represented as a passed unchanged historical assertion.

No 13.5.12 work is started and no push is performed.
