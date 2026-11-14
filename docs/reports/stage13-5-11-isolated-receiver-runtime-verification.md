# HYSo 13.5.11 — Isolated evaluation receiver runtime verification

Date: 2026-11-14, Europe/Istanbul. QA root (outside the repository):
`%LOCALAPPDATA%/hyso-stage13-5-11-20261114` (`QA/` below).

**Overall: PARTIAL — feasibility/design closure, no receiver verification implementation.**
The actual evaluation receiver's full Graph runtime contract is not observed by
the existing public API. Source v2 verification remains separate from the
**LIMITED** inference association. No new native authority was issued.

| Independent acceptance | Decision | Scope |
| --- | --- | --- |
| A — Actual receiver identity observation | PARTIAL | Existing schema3 observes the called candidate's registered structure; no exposed live identity/full-runtime observation |
| B — Receiver runtime contract verification | BLOCKED | Requires the narrow Evaluation Core change proposed in section 6 |
| C — Source-to-receiver trusted binding | LIMITED | Named-state/stream association only; full inference-runtime binding NOT VERIFIED |
| D — Legacy P26 and Graph v2 compatibility | PASS | Existing contracts and formats preserved |
| E — Reporting/Comparison integration | NOT CHANGED | Existing limited-source presentation/comparison exercised |
| F — Git integrity | PASS | Documentation-only scope, original history and user files preserved |

## 1. Initial Git state

Branch: `feature/hyso-models-next`. Starting HEAD:
`0e81551143d1494ad740a3700575561637f2dbbe`.
Index empty; **2 modified tracked + 45 untracked = 47** unrelated files.
No reset, clean, restore, rebase, amend, bulk add or push was performed.

The previous acceptance report and Graph v2 integration guide were read, along
with the 13.5.6 protection manifest and 13.5.9/13.5.10 final manifests. Before
repository changes, `QA/baseline.json` recorded all **742** tracked/nonignored
untracked file hashes, dirty paths, branch, HEAD, empty index, the preceding eight
commit/parent records, native/forensic hashes, external files and user data.
All 742 hashes exactly matched the 13.5.10 final manifest. `QA/protected47.json`
separately records the dirty and forensic sets. No applicable AGENTS.md was found
in the repository or its parent chain.

Python: `C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe`, `-B`, CPU visibility,
single CPU thread, Agg rendering, no pytest cache; TEMP/TMP, Matplotlib state,
pytest temporary directories, checkpoints, experiments and reports under QA.
User databases were only byte-hashed, never connected to or migrated.

## 2. Existing Evaluation receiver lifecycle

Code reviewed: `hyso/evaluation.py`, `_evaluation_evidence.py`,
`_evaluation_structure.py`, `checkpoint.py`, `analysis/graph_evidence.py`,
`models/graph.py`, `models/output.py`, `structure.py`, EvidenceRecord/EvidenceIndex,
the reporting dispatcher and their security/regression tests.

The following locations refer to the unchanged starting source:

| Question | Observed implementation |
| --- | --- |
| Where is the receiver created? | `evaluation.py:327`: local `candidate = copy.deepcopy(model)`, inside the RNG/generator restoration scope |
| Which state is copied? | The caller's object graph at deepcopy time; source named parameters/buffers are hashed first at line 326, candidate state is hashed after eval at line 336 |
| Which object executes? | Line 369: `candidate(inputs.clone().to(execution_device))`, once per validated batch |
| Identity/alias checks? | Lines 328–333 reject source/candidate shared registered module/tensor identities and nonempty tensor storage base pointers |
| Pre-inference observations? | `candidate.eval()`, named-state equality when evidence capture is enabled, optional actual-candidate structure capture at lines 340–342 |
| Per-forward integrity? | Named-state equality and all module eval modes checked after each forward at line 377 |
| Post-inference observations? | Complete sample accounting, optional matching pre/post structure and post-state, then source named-state equality at lines 381–392 |
| Topology/activation/adapter/device? | Full v2 metadata can be read from a live supported object by the existing contract primitive; Evaluation does not invoke that primitive on candidate |
| How is actual identity exposed? | It is not returned. Schema3 describes the called receiver's registered metadata; neither the live candidate nor its v2 runtime contract is public |
| What survives? | An EvaluationResult with metrics, confusion matrix, state/stream evidence and optionally a schema3 registered-structure binding |

The candidate remains a local reference and is not retained as a live receiver
by GraphEvaluationResult. The legacy producer is called with the original model
at line 398. Returning equal hashes, metrics or structure snapshots cannot expose
or authenticate the candidate's full runtime identity.

## 3. Source v2 receipt contract

`load_training_checkpoint` hashes a private snapshot, optionally checks the
expected SHA before weights-only deserialization, validates the explicit v2
contract before mutation and records evidence only after successful restore.
`_GRAPH_CHECKPOINTS` uses the exact source object's id and a checked weak reference.
`graph_checkpoint_evidence(source)` rechecks current named tensor state and the
full reconstructed supported Graph contract against that receipt.

The v2 primitive verifies exact supported GraphModel/GraphTensorOutput types,
configuration, ordered registered topology, constructor runtime attributes,
activation settings, selected output, CPU fp32/fp64 anchor, parameter/buffer
schemas, serialized persistent state and bounded reference construction. It
rejects unknown components, subclasses, instance callables, hooks and unsupported
runtime attributes within the existing closed profile. It trusts installed Core
constructors; it is not an arbitrary-code or hostile-process sandbox.

The raw checkpoint API uses `status=native`; the integration's live verification
view presents `graph_v2=verified`. Neither string changes legacy P26 semantics.
Expected caller SHAs constrain an existing trusted load; they cannot mint one.

## 4. Deepcopy security boundary

A deepcopy receives no source receipt. The public source query on another object,
even a separately trusted identical checkpoint load, does not identify that object
as the Evaluation candidate. Matching named-state SHA omits non-tensor behavior;
activation, topology and output selection can differ while named tensors match.

The existing alias guard detects cross-source/candidate registered identities and
equal nonempty storage base pointers. It does not establish all internal
byte-range alias semantics, a runtime execution graph or behavior equivalence.
The v2 contract rejects repeated module/tensor identities, but this report does
not widen that to proof against every storage view/overlap or concurrent mutation.
Schema3 structure explicitly excludes full forward semantics, functional paths
and unregistered Python state. Its snapshot is not a Graph v2 runtime contract.

QA tests use independently created clones solely as negative examples. None is
represented as the actual receiver. No source receipt was transferred, no registry
was populated by caller code and no second inference was attached to an earlier
evaluation. Existing test-only fault injection is not a production observation API.

## 5. Mandatory feasibility gate

**Classification B: a new, opt-in, narrow Evaluation Core contract is required.**

A public-API-only solution cannot obtain candidate at the call site. Alternatives
considered and rejected:

- Validate another deepcopy or fresh checkpoint load: verifies another object.
- Use source/stream/tensor/structure/output equality: does not prove the actual
  receiver's runtime identity and equivalence.
- Interpret schema3 metadata as a full v2 contract: overstates its explicit scope.
- Add a subclass, hook, monkeypatch, frame tracer or alternate evaluator: crosses
  the specified trust boundary or violates the requested execution path.
- Change producer evidence outside Evaluation: its existing argument is the source,
  so it still lacks the actual candidate and same-forward observation.

The task explicitly says: “Gerçek inference receiver'a erişmek için mevcut
Evaluation Core'unu değiştirmek gerekiyorsa” evaluate the Core option, and
“İstisna gerektiren teknik değişiklik için açık teklif hazırla ve uygulamayı durdur.”
This closure applies that Core gate rather than implementing the missing authority.

For precision: **evaluation.py and _evaluation_evidence.py are not members of the
historical 71-file native manifest**. Their treatment here is the task-level
Evaluation Core authority gate, not an assertion that the old manifest lists them.
`checkpoint.py` is in that manifest; the earlier 13.5.9 exception for checkpoint.py
does not authorize a new Evaluation authority path. No protected source exception
was assumed or exercised in this stage. This is an implementation boundary, not a
claim that a secure Core solution is impossible.

## 6. Narrow Core exception proposal — NOT IMPLEMENTED

Proposed existing Core file: **`hyso/evaluation.py` only**.
Starting SHA-256:
`b7971fdb6c824e7450f687fdd5dc5823b8acc6834d9571b83646948bd35535d6`.

Required edit: preserve the exact public `evaluate_classification` signature and
default schema1/2/3 behavior; extract its existing single loop into one shared
private executor. Add a separate opt-in Graph entry to that executor with a fixed
Core-owned observer, without a caller callback or exposed mutable candidate.
The observation sites must be on the exact local candidate after deepcopy/eval,
immediately before and after each actual call, and before successful final result
construction. Preserve validation order, alias checks, RNG restoration and error
handling on every path. No alternate inference loop is proposed.

Supporting stage-owned changes, if authorized in a later implementation:

- A small new private observation/association module can reuse the unchanged
  `_graph_contract` and named-state primitives. It must not implement a second
  contract validator/checkpoint engine or mint checkpoint receipts for candidate.
- Update `analysis/graph_evidence.py` with a separately named opt-in entry and
  explicitly versioned receiver association. Keep old association version 1 and
  imported records at their existing meanings.
- Add independent security tests and current package validation. If a new runtime
  module is used, explicitly update the commercial profile from 163 to 164 and
  retain all exclusions. No dependency or component-profile expansion is proposed.
- Reporting/comparison may consume a new typed result through their existing
  adapters; no metric or statistical recommendation policy gains authority from
  checkpoint verification alone.

No change to `checkpoint.py`, `_evaluation_evidence.py`, Graph forward, Trainer,
BenchmarkRunner, native P01–P35 or existing schema readers is requested by this
minimum proposal. If implementation proves any additional Core edit necessary,
it must be identified separately rather than assumed covered by this proposal.

Proposed receiver association format (a design requirement, **not an available
API or emitted record**) is a distinct `hyso.graph_receiver_binding`, version 1:

- Live source weak identity and valid source receipt; exact checkpoint/state SHAs.
- Core-generated evaluation/capture identity and opaque receiver identity, never
  a caller-provided Python id accepted as proof.
- Independently observed receiver configuration/runtime/named-state/persistent
  state and supported nonpersistent anchor before and after the actual calls.
- Explicit source/receiver contract equality, cross-object nonaliasing and a
  conservative internal storage-alias policy.
- Same Evaluation producer identity, ordered stream, batch accounting and result
  digest; successful nonempty complete execution required.
- Process-local authority bound to the exact returned immutable result. JSON,
  constructor calls, copies and another process cannot restore this authority.

`verified` would require **all** those observations to pass. Missing observation
keeps `limited`/`missing`; observed mismatch is `incompatible` or a fail-closed
error. The source checkpoint receipt remains solely on the source. A capture
identity is an in-process authority mechanism, not cryptographic attestation of
an adversarial interpreter.

## 7. Applied API/code changes

**No runtime Python, production test, public API, dependency, packaging profile,
checkpoint format, Evaluation schema or existing guide was changed.** The sole
repository addition is this report with its reviewable Core proposal.

Audit harnesses and security probes are QA-only. The two deliberately red receiver
acceptance probes remain red; they are not skipped, xfailed or relabeled as a
completed receiver implementation. Existing successful boundary behavior is
tested independently.

## 8. Actual receiver observation mechanism

Existing schema3 is the only exercised native receiver observation. It captures
registered structure from the same local candidate before loader iteration and
after successful iteration. `structure_binding.receiver` is
`isolated_candidate_called_for_forward`; its original/copy relation remains
`not_captured_or_equivalence_not_certified`.

Two fresh supported Graph sources were strictly loaded in
`QA/schema3-acceptance/` and each evaluated once with `capture_structure=True`.
Both produced actual schema3 receiver structure bindings, two forward batches
each and native stream records. Neither exposed a live object identity or full
receiver Graph runtime contract. `QA/schema3-proof.json` retains the exact
observations; no test instrumentation was used to issue authority.

## 9. Runtime equivalence evidence

Full source-to-actual-receiver runtime equivalence: **NOT OBSERVED / NOT VERIFIED**.
Source v2 view: verified at capture/report creation. Existing relation remains:

```text
status=limited
isolated_receiver_receipt=not_transferred
runtime_equivalence=not_observed
scope=source_receipt_and_observed_named_tensors_only
```

The proposed Core path must validate the installed closed constructors and reject
subclasses, hooks, instance/dynamic forwards and unsupported components without
extending the current profile. It must also define how class-level callable
tampering is detected relative to trusted installed code. Merely recording a
callable identity after possible tampering is not sufficient. This stage supplies
no new guarantee for those unimplemented receiver checks.

## 10. Pre/post inference integrity

Current checks provide candidate named-state/eval-mode stability per forward,
source named-state stability at completion and optional matching actual-receiver
registered structure at evaluation boundaries. They do not observe a full v2
runtime contract per actual call. Ephemeral mutate-and-restore behavior between
observations cannot be ruled out by equal boundary hashes alone.

The proposed path must use the same live receiver at every call, validate source
and receiver at the required boundaries, avoid arbitrary callbacks between checks
and execution, check complete sample/stream accounting and publish no verified
result after any failure. A supported trusted, quiescent Core environment must be
explicit; concurrent malicious Python execution is not solved by extra hashes.
Mutation after a historical capture must not silently turn that capture into a
claim of current live receiver health. Retained statements and any live validity
view need distinct semantics and weak identity ownership.

## 11. Negative security matrix

Evidence sources: `C` = existing Graph checkpoint contract tests (38 cases), `G` =
existing Graph integration tests (31 cases), `E` = Evaluation/producer/structure
regressions, `Q` = QA-only receiver boundary probes, `S` = actual schema3 smoke.
“Boundary” below does **not** mean full actual-receiver runtime validation passed.

| # | Required case | Observed coverage / remaining limit |
| --- | --- | --- |
| 1 | Valid source v2 | C/G/CPU: source native receipt and live verified view |
| 2 | Valid actual receiver | S/E: actual registered structure captured; full runtime BLOCKED |
| 3 | Wrong receiver identity | Q: independent clone/fresh trusted object cannot strengthen result; exact receiver binding BLOCKED |
| 4 | Different topology | C/G/Q: source contract invalidates or load rejects; actual receiver contract BLOCKED |
| 5 | Different activation | C/G/Q: equal named weights never imply trusted runtime equivalence |
| 6 | Wrong output adapter | C/G/Q: source proof missing/rejected; actual receiver check BLOCKED |
| 7 | Wrong input geometry | C/G: source/load fails closed; actual full-contract check BLOCKED |
| 8 | FP32/FP64 mismatch | C/G: supported exact dtype required; no CPU-to-CUDA inference claim |
| 9 | Parameter mutation | C/G/E: receipt invalidation or evaluation state-mutation error |
| 10 | Persistent buffer mutation | C/G: source invalidation; E retains named-buffer integrity guard |
| 11 | Nonpersistent mismatch | C/Q: unknown buffer/anchor invalidates source; full receiver anchor check BLOCKED |
| 12 | Custom subclass | C/Q: source/load rejected or missing, no receiver authority |
| 13 | Global/local hook | C: source closed-profile rejects hooks; no new runtime observation hook installed |
| 14 | Same tensor hash, different behavior | Q activation/topology/output negatives; no named-state promotion |
| 15 | Source mutation before evaluation | C/G: live receipt cannot remain verified after observed mutation |
| 16 | Receiver mutation during evaluation | E: named-state/mode/registered-structure guards exercised; full v2 runtime BLOCKED |
| 17 | Receiver mutation after evaluation | No public receiver retention/current-health query; full receiver lifecycle BLOCKED |
| 18 | Copy/deepcopy receipt transfer | C/G/Q: new objects missing, copied result unverified |
| 19 | Receipt on another object | C/Q: another object is not the captured inference receiver |
| 20 | Wrong checkpoint SHA | C/G/CPU: rejected by expected SHA constraints |
| 21 | Wrong model-state SHA | C/G: cannot match source/query/association |
| 22 | Wrong dataset/split | G: declared mismatch rejected or comparison incompatible; origin remains declared |
| 23 | Wrong stream | E/G: ordered native stream mismatch blocks comparison |
| 24 | Forged verified JSON | G/Q: persisted authority fields ignored/recomputed |
| 25 | JSON import authority loss | G/Q/CPU: unverified source, unavailable live association |
| 26 | Different process import | G: actual new interpreter has no capture authority |
| 27 | Legacy P26 missing | C/G/S/CPU: unchanged in schema2/3 |
| 28 | Old association records | Q/S: actual 13.5.10 file reopens unverified; bytes unchanged |
| 29 | Schema1/2/3 compatibility | E/G: legacy schema/hash tests and immutable retained golden |
| 30 | Two valid comparable models | CPU/S/G: COMPARABLE quality observations, both bindings limited |

`QA/red.xml` records **2 deliberate acceptance failures**: no public receiver
runtime capture option, and actual binding `limited` rather than `verified`.
These demonstrate the unimplemented gate. `QA/boundary.xml` records **13 passed**
boundary tests. The red signature probe is an absence check, not a prescribed
future public signature; a separate opt-in Core entry is the proposed design.

The regression selection, final results and harness corrections are recorded in
QA; the final counts are summarized in section 13.

## 12. Short CPU acceptance

The unchanged `examples/models/graph_evidence_workflow.py` was freshly executed
once into `QA/cpu-acceptance/`. Its fixture label retains `stage13-5-10`; this is a
new 13.5.11 execution, not reuse of an old run or modification of historical data.

Two tiny branching Graph CNNs use SiLU/ReLU, CPU fp32, seeds **13610/13611**, four
explicit synthetic training samples, four distinct test samples, batch size two,
one TrainingSession epoch and **two optimizer updates per model**. Fresh strict
v2 loads verify snapshot SHA and metadata. Test accuracy is **0.5 / 0.5**;
comparison is **COMPARABLE**, with no global winner, statistical superiority or
data-quality acceptance claim. Both source views were verified; both legacy
checkpoint fields missing; both inference bindings limited.

Fresh checkpoint SHA-256 values:

```text
0448158a040132f00a42c69800c00d4d79d39ad7cff360e360f0b7aad2377b46
cb87054c9d284be0225d5ddf78bed4ec9f92df756a213f03aa0c03bd5b77774d
```

Deterministic checkpoint bytes happen to match the earlier fixture's hashes;
files were newly created in this stage's QA directory. No old artifact was used
as evidence of this fresh execution. The separate schema3 acceptance reuses
these newly created checkpoints with **no additional training**. It uses a
separate explicit synthetic evaluation stream and does not compare its metrics
against the first stream.

## 13. Legacy P26/schema compatibility and regressions

Legacy Graph P26 checkpoint remains **missing**. Source Graph v2 semantics and
schema1/2/3 remain unchanged. Graph association version 1 remains unchanged.
The golden comparison/presentation SHA remains:
`eb66ee58d77155529d83db358cb5e0567fa8882c44ce9d7d9ee10e6664cc7b6e`.
It was tested in QA and the existing integration suite; repeated golden checks
are not independent receiver-runtime evidence.

| Verification | Result |
| --- | --- |
| Selected existing CPU regressions | **782 distinct main cases passed**, plus **10 subtests** |
| Additional QA-only receiver boundary tests | **13 passed** |
| Total successful distinct main case ids | **795**, excluding subtests and reruns |
| Deliberate unsatisfied receiver acceptance probes | **2 failed**, still outstanding |
| Legacy comparison/report golden | Exact match to the SHA above |
| Git object integrity | `git fsck --full` exit 0; dangling objects reported, no corruption |

The existing regression selection covers GraphModelBuilder/GraphTensorOutput,
all 38 Graph v2 contract cases, all 31 Graph integration cases, Evaluation,
schema1/2/3, P26, retained/structure/architecture comparisons and AUDIT-01,
Reporting, Advisor, ExperimentManager, TrainingSession/config/checkpoint and CLI.
Full paths are retained in `QA/regression-selection.json`.

The first selection launcher failed before pytest because a unittest classname
contained a class suffix; this harness mapping was corrected. The subsequent
complete run recorded **779 passed + 3 failed + 10 passed subtests**, with 47
existing warnings. All three failures were historical-reader `git show`
subprocesses launched from the external QA working directory, where Git could
not find a repository. They did not reach their reader assertions. Only their
working directory was corrected to the checkout, keeping every output/temp path
under QA; the exact three cases then **all passed** (`QA/retry.xml`). No passing
case was added twice, and no source was changed to obtain those passes.
Both original failures and corrected results are retained in the logs/XML.
`QA/test-summary.json` independently merges actual main testcase ids rather than
adding JUnit aggregate counts that include subtests.

The two red receiver acceptance failures are excluded from successful regression
counts and remain outstanding. No CUDA/native P01–P35 acceptance suite or long
training was run. Existing CPU tests may use scoped fault injection for negative
cases; those observations are never exported as new native receiver authority.

## 14. Comparison, Reporting and ExperimentManager

Existing comparison uses retained P26 quality evidence, actual stream/batch
conditions and declared dataset/split/preprocessing guards. Both new CPU pairs
were comparable within their respective streams, with source verification shown
separately from limited binding. Advisor cannot infer statistical superiority or
performance improvement from those receipts.

JSON, Markdown and static HTML were generated through the existing renderer for
schema2 and schema3 associations/comparisons. Imported/fabricated JSON cannot
restore source or receiver authority. The actual earlier 13.5.10 association file
was read through the typed reader, remained byte-identical and reopened as
`unverified` / live association `unavailable`.

Fresh ExperimentManager store: `QA/cpu-acceptance/experiments`.
Experiment id: `117aac9c67b84774851f3136150323c3`.
Five completed runs and **13 artifact byte checks verified**. Wrong artifact
copy returned **mismatch**; the original artifact was reverified independently.
Mutation made source proof **missing**; wrong expected checkpoint was rejected;
JSON reopening was **unverified**. Artifact byte integrity supplies no native
receipt. User SQLite/WAL/SHM/lock files were never opened as databases.

## 15. Wheel/sdist verification

No runtime source, dependency, allowlist or packaging change exists in this
documentation-only closure. Per the task's conditional packaging rule, **no new
wheel/sdist build or installed receiver-runtime acceptance was performed**.
The commercial profile remains **163 runtime Python modules** with existing
exclusions. No Studio/frontend/static web product or service was added.

13.5.10 wheel/sdist validations are historical context only, not current proof
of a receiver implementation. A future Core implementation must build fresh
artifacts, compare every packaged runtime module hash, install offline and run
both valid and negative receiver tests on those installed artifacts.

## 16. Source and user-data hash preservation

Before/after evidence is in `QA/baseline.json` and `QA/preservation.json`.
All **742 initial repository files** must remain byte-identical; this report is
the only added repository file. Protected sets are checked separately:

- 47 dirty/user files and 47 historical forensic entries.
- 71 native files at their authorized post-13.5.9 versions.
- Six external acceptance/checkpoint files and 11 external workspace/raw files.
- 197 repository model/checkpoint/database files, including ignored user data.
- Historical stage reports, including 13.5.6–13.5.10, and prior commit ancestry.

Preservation of historical P01–P35 evidence is claimed for the manifested files
and unchanged source/history; this is not a new acceptance run or a claim of an
unbounded filesystem inventory. No historical evidence was regenerated or edited.

## 17. Git commit and scope

One documentation-only commit is authorized for this feasibility closure:
`docs: define isolated Graph receiver trust requirements`.
Its only path is
`docs/reports/stage13-5-11-isolated-receiver-runtime-verification.md`.
Parent is the exact starting HEAD:
`0e81551143d1494ad740a3700575561637f2dbbe`.

Before committing: exact cached path, cached diff/stat, cached whitespace check,
preserved-source hashes and completed relevant test evidence are inspected.
After committing: full SHA/parent/path scope, empty index, unchanged remaining
47 files and historical ancestry are verified. The self-referential commit SHA
is recorded externally in `QA/commits.json` and the final response; this report
is not amended to insert its own commit identity.

No implementation-success commit is made. No push is authorized or performed.

## 18. Open limitations

- The exact live inference receiver is not available to the public adapter.
- Full candidate Graph runtime pre/post/per-forward contract and source/candidate
  runtime equivalence are not observed. The two acceptance probes remain red.
- Existing schema3 actual-receiver structure is useful but narrower than v2 runtime.
- No guarantee against unobserved transient/concurrent mutation or hostile
  interpreter/code tampering is introduced.
- Source v2 native status is not receiver binding authority. Imported records,
  comparison, rendering and artifact verification cannot strengthen it.
- Synthetic CPU checks certify no real dataset quality or CUDA performance.
- Future implementation and packaging remain dependent on the explicit narrow
  Core exception described in section 6 and its independent security validation.

## 19. Final decision

**13.5.11 PARTIAL — feasibility gate B, documentation closure only.**
A PARTIAL; B BLOCKED; C LIMITED; D PASS; E NOT CHANGED; F PASS for the stated
documentation/history preservation scope. There is **no VERIFIED actual inference
receiver binding**. The specific missing contract and minimum Core edit are
reviewable above; implementation stops at the requested gate. No 13.5.12 work is
started.
