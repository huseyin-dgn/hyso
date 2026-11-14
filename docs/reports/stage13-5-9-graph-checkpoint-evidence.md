# HYSo 13.5.9 — Graph checkpoint contract and trusted evidence

Date: 2026-11-14, Europe/Istanbul. QA root:
`%LOCALAPPDATA%/hyso-stage13-5-9-20261114` (`QA/` below).

| Independent decision | Result |
| --- | --- |
| A — Graph checkpoint contract design | **PASS**, within the documented bounded typed CPU profile |
| B — Secure implementation | **PASS**, within that profile |
| C — Native Graph receipt | **VERIFIED — separate Graph contract evidence v2** |
| D — Legacy P26 compatibility | **PASS** |
| E — Git commit integrity | **PASS**, explicit stage-owned files only |

**The existing P26 schema2/3 v1 checkpoint field for Graph remains `missing`.**
The verified v2 receipt is a separate format/API, not a promotion of old P26
evidence. It binds serialized tensors and explicitly reconstructed Graph metadata
under trusted installed constructors; it does not certify arbitrary Python code,
checkpoint authorship or CUDA behavior.

## 1. Initial Git state

Branch `feature/hyso-models-next`; initial HEAD
`d325c613caa2718ea4010bffbe7560219707843d`; index empty. There were 651 tracked
and 82 nonignored untracked files, including 10 modified tracked files.
`QA/baseline.json` was created before source edits: **733 file hashes**, raw Git
status/index inventories, 197 repository user database/model/checkpoint hashes
and 11 external hashes. Prior dirty/untracked work was retained.

Python: `C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe`. No reset, clean,
restore, rebase, amend, push or force operation was used. All execution outputs,
temporary exports, checkpoints, distributions and test fixtures are outside the
repository in QA. Only source/tests/docs were added or edited in the checkout.

## 2. Prior accepted changes and file ownership

`QA/ownership.json` classifies the starting inventory as A (previous acceptance),
B (new stage work), C (unrelated/uncertain user/common work) or D (protected).
It references the 13.5.6 task/protection and Python-first manifests, 13.5.7
baseline/report, and 13.5.8 baseline/final manifests. Clean shared repository
files are not inferred to be new stage work.

Shared intermediate README, example index, profile and packaging test versions
for 13.5.6 were reconstructed in QA and their **raw SHA-256 exactly matched the
original acceptance task manifest**. The 13.5.7 example was likewise recovered
against the 13.5.8 starting hash. Explicit index blobs staged these intermediate
versions without rewriting working files. Results are in
`QA/snapshot-reconstruction.json` and `QA/commit-snapshots/`.

Each candidate tree was exported into QA for independent checks: 13.5.6 **99
passed**, 13.5.7 **25 passed**, 13.5.8 **14 passed**. Packaging profiles and
dependencies of each staged stage remain coherent. The accepted Graph SCC
diagnostic change and its guide matched the original 13.5.6 task hashes and were
included in the first commit; no new Graph forward change was made here.

One material ownership exception: the starting 13.5.8 report SHA was
`afa43847e8b3869c4a4d5fc49d5c2ed94a7aa00aba8db10d27c80ea67e25cc57`,
different from its recorded final-stage hash
`f462448e1d61d56fc4ed1f3d114c6b757fe08ac944c9a88442b1506699ee8191`.
This is not explained by CRLF conversion. That report was preserved as found,
classified C and **excluded from commits**. The verified 13.5.8 code/tests were
committed independently; uncertain report ownership was not silently included.

## 3. Confirmed P26 root cause

GraphModel's empty nonpersistent `_contract` anchors dtype/device checks in
forward. It is absent from normal state_dict and strict loading does not expect
it. P26 v1 hashes all named parameters/buffers and requires all those names to
exist among frozen checkpoint fingerprints before issuing its full-state receipt.
The unmatched `_contract` therefore keeps the legacy receipt missing even when
strict restore and equal inference pass.

The unchanged nine-case 13.5.8 diagnostic group was executed again, including
actual Graph checkpoint load/evaluation, dtype-contract mismatch and independent
persistent-buffer controls. The old behavior remains explicitly covered.

## 4. Alternatives considered

| Alternative | Security and compatibility | Decision |
| --- | --- | --- |
| A: separately versioned reconstructed Graph contract | Can distinguish serialized state from known runtime metadata, bind configuration/output selection and retain full tensor/snapshot checks; needs strict admission | Selected, bounded profile |
| B: persistent `_contract` | Would alter state keys and strict compatibility of old Graph files; its emptiness alone still would not authenticate topology/activation/adapter | Not implemented |
| C: leave everything unchanged | Strong fail-closed legacy policy, but no Graph contract receipt | Retained for v1/unsupported graphs |

A configuration string or digest alone is insufficient. The implementation
compares live runtime structure with fresh closed constructors, commits exact
serialized tensor fingerprints and requires a real snapshot load before receipt.

## 5. Selected versioned contract

The [usage/contract guide](../guides/graph-checkpoint-evidence.md) defines:

```python
from hyso.checkpoint import (
    save_training_checkpoint, load_training_checkpoint, graph_checkpoint_evidence,
)
save_training_checkpoint(path, source, graph_contract=True)
load_training_checkpoint(path, target, capture_evidence=True, expected_sha256=known_sha)
proof = graph_checkpoint_evidence(target)
```

Default saves remain checkpoint format_version 1. Opt-in Graph saves create
format_version 2 with `model.graph_contract`, format
`hyso.graph_checkpoint_contract`, schema_version 2. The returned proof format is
`hyso.graph_checkpoint_evidence`, schema_version 2. Old readers explicitly reject
v2; old files are neither upgraded nor reserialized.

The contract records exact Graph configuration (including names/topology/node
parameters), selected adapter output, CPU device, fp32/fp64 dtype, Torch version,
runtime signature and **separate serialized-state digest**. The runtime signature
checks recursive exact module classes, order, plain attributes, tensor schemas,
requires_grad and aliasing against fresh constructors. Fresh random reference
weights are never substituted for checkpoint values. The current full named-state
hash still includes the nonpersistent anchor metadata.

Only the listed closed components in the guide are supported; conservative
preflight parameter estimate must be known and at most 1,000,000 before reference
allocation. Parameterless Graphs are supported. Construction preserves CPU RNG
and performs no forward or data reads. Unknown components, subclasses, local or
global hooks, extra state/nonpersistent buffers, aliases or modified behavior
fail closed. Unsupported explicit v2 save/load requests raise; receipt queries
without a matching supported restore return `missing`.

## 6. Security invariants

The existing private spooled snapshot (8 MiB memory threshold, 1 MiB read chunks)
is still the exact object hashed and passed to weights_only=True. Optional
`expected_sha256` rejects a mismatch **before deserialization**, including a
locally rewritten payload with recomputed internal commitments. That expected
hash is an external integrity constraint, not proof of authorship by itself.

Strict tensor keys/shapes/dtypes, exact parameter/persistent-buffer fingerprints,
configuration and runtime contract all must agree. Fingerprints are frozen before
target restore; post-restore values are rechecked. Unknown `_contract` substitutes
are not globally ignored. Exact JSON canonical comparison distinguishes values
such as inner schema version `2` versus `2.0`.

Receipts retain immutable plain metadata internally and a weak reference to the
exact loaded object. Query rechecks current state and runtime contract. Tensor,
topology, activation, output or hook mutations invalidate the match; deepcopy or
another instance has none. An uncaptured reload clears it; failed mutation and
rollback cannot mint a receipt. Pre-mutation validation failures may retain only
the earlier valid receipt for its unchanged earlier artifact, as in legacy policy.

An attacker controlling a whole new artifact and its expected hash can supply a
different internally consistent checkpoint; this is not authentication as the
original artifact. Arbitrary malicious Python execution in the interpreter can
also compromise process-local registries. The trusted installed implementation
and ordinary caller/process integrity remain assumptions, as for existing P26.

## 7. Implementation and authorized protection exception

Implementation files: `hyso/_evaluation_evidence.py` (typed contract capture,
validation and separate weak-reference receipt), `hyso/checkpoint.py` (explicit
v2 opt-in and optional snapshot hash constraint). New test, guide and runnable
CPU example accompany them. No new runtime module, dependency or second engine
was added. Graph forward, state_dict persistence, Trainer/optimizer behavior,
evaluation schemas/producers and legacy P26 named-state hashing are unchanged.

The old 71-file native protection manifest includes `hyso/checkpoint.py`. The
initial task simultaneously required preserving those bytes and implementing
save/load changes. The user **explicitly approved an exception only for this
file** during this stage. Preservation therefore reports **70/71 unchanged, one
authorized change**, rather than falsely claiming all 71 unchanged.

## 8. Negative tests and before/after evidence

`tests/test_graph_checkpoint_contract.py` contains **38 passing cases** in the
final source group. Its initial 30-case contract run failed because the new
opt-in API did not exist (`QA/red.xml`), then those 30 passed (`QA/green.xml`).

| Required scenario | Final observation |
| --- | --- |
| Valid Graph checkpoint | Native v2 receipt, exact file/state link, equal outputs |
| Missing parameter | Rejected before target mutation |
| Modified checkpoint tensor | Frozen serialized commitment mismatch, rejected |
| Modified persistent buffer | Payload rejection and live mutation invalidation covered |
| Wrong checkpoint SHA | Rejected before weights-only deserialization |
| Wrong model-state hash | No matching Graph receipt |
| Different input geometry | Contract mismatch, no receipt |
| Same shapes, different topology | Add versus multiply rejected |
| Same state schema, different activation | SiLU versus ReLU rejected |
| Wrong output selection | Logits versus features rejected/invalidated |
| Wrong classification head | State schema rejects wrong class width |
| `_contract` dtype mismatch | Rejected |
| Device mismatch | CPU versus meta anchor fixture rejected; no CUDA claim |
| Extra nonpersistent buffer | Unknown runtime state rejected |
| Custom subclass | Wrapper and nested Graph subclasses rejected |
| Load-hook state alteration | All such v2 hooks rejected; legacy real hook-substitution controls also pass |
| Failed load/rollback | Controlled one-shot failure restores prior state; receipt missing |
| Post-load mutation | Parameters, persistent buffer, config, activation, output, hooks and extra state invalidate |
| Deepcopy | No receipt |
| Another model instance | No receipt |
| Parameterless Graph | Native v2 receipt on fp32/fp64; unsupported byte order rejected |
| Old P26 v1 evidence | No automatic upgrade; legacy Graph receipt stays missing |
| Schema1/2/3 | Legacy evaluation readers/producers and existing binding regressions pass |
| CPU fp32/fp64 | Actual forward and receipt roundtrips pass for both |

Additional coverage includes detached returned receipts, weak-reference expiry,
global module hooks, pre-allocation budget rejection, late metadata callback
mutation and invalid scalar types inside contract metadata. Only safe bounded
local state payloads are deserialized, always weights-only.

Three development findings are retained rather than hidden:

- A narrowed budget guard initially used exact `parameter_count`; Core reports
  that value unavailable for ConvNormAct despite a known conservative estimate.
  This caused 31 failures/12 passes in `contracts-final.xml`. Admission now uses
  the validated conservative estimate and explicitly rejects unknown estimates;
  an allocation-forbidden over-budget test verifies the limit.
- Core initially had 536 passes/1 failure: a legacy malformed version=2 fixture
  was rejected with the wrong diagnostic after v2 was introduced. A v2 payload
  without explicit Graph contract now retains the required format_version error.
  Its test and the complete existing checkpoint group pass in the final run.
- `type-red.xml` reproduces the new inner version `2.0` acceptance bug before the
  canonical typed comparison fix. The same case now passes as a rejection test.

## 9. Real short CPU acceptance

`examples/models/graph_checkpoint_evidence.py` builds the branched Graph via
GraphModelBuilder and selects classification logits with GraphTensorOutput.
Seed **1359**, CPU fp32, batch 2, SGD lr=0.01, one epoch, **2 applied updates**.
The four explicit synthetic inputs are declared as such; this is not another
real-dataset or model-quality claim.

The new QA checkpoint includes training settings/history and architecture
metadata. A fresh matching model validates saved metadata, snapshot SHA and
strict state; identical fixed inputs yield **bit-identical outputs**. Native v2
evidence is present, and modifying a parameter makes the next receipt missing.
Actual evaluation separately retains the legacy P26 missing status.

Evidence is under `QA/cpu-acceptance/`: `training.json`, `receipt-v2.json`,
`evaluation-legacy-p26.json`, `receipt-after-mutation.json`, `summary.json` and
`graph-v2.pth`. Checkpoint SHA-256:
`6af9e54af1d63e3f70fc11c7e1ba2269568499886939311dca03053cfee1469b`.

PowerShell marked the first redirected launcher non-success because native
GradScaler warning output was routed as an error; the example nevertheless
completed all artifacts/assertions. `QA/cpu-final-verification.json` independently
loads that same unchanged checkpoint through final code, verifies native receipt
and mutation invalidation without repeating training. Its verifier exited zero.

## 10. Native receipt result and exact scope

**VERIFIED** for separately queried Graph contract evidence v2, using the actual
trusted load path and the supported typed CPU profile. This is stronger than a
strict roundtrip alone: file/state/config/runtime commitments and exact object
receipt lifecycle are tested separately.

The original P26 producer field remains **STILL MISSING** for Graph. No new
EvidenceRecord/Advisor projection of v2 into old checkpoint bindings is added.
Consumers must retain and interpret the separate explicitly versioned proof.
No claim is made that a schema2/3 Graph P26 record gained a v1 receipt.

## 11. Legacy P26/checkpoint compatibility

Default v1 save/load schemas and Graph state_dict keys are preserved. Historical
model-state/stream hash vectors, producer evidence, identity/mutation rules and
imported native-record semantics were not rewritten. Exact old reader behavior
is tested against trusted repository source; old readers reject v2 files without
changing them. Legacy Graph checkpoints load but do not gain reconstructed proof.
Legacy non-Graph full-state receipts still work.

## 12. Schema and selected CPU regressions

| Evidence group | Result |
| --- | --- |
| Initial selected Core (`core.xml`) | 536 passed, 1 diagnostic failure subsequently fixed |
| Final existing checkpoint + new Graph + prior P26 diagnostics (`final-contract.xml`) | **81 passed**, no failure/error/skip |
| Full earlier packaging/commercial (`package.xml`) | **10 passed** |
| Final-source archive content/profile (`final-package.xml`) | **4 passed** |
| Independent stage 6/7/8 trees | **99 / 25 / 14 passed** |
| Final independent stage 9 candidate new tests | **37 passed**, 1 Git-reader test intentionally deselected outside a Git checkout |
| Final installed wheel / sdist contract matrices | **37 / 37 passed**, same Git-reader exclusion |

The final 81 includes all 34 existing training checkpoint cases, 38 new Graph
cases and 9 prior P26 diagnostic cases. It recovers the initial Core failure.
Across the source regression and stage-specific groups there are **599 unique
successful test cases**, with no unresolved failed case (`QA/test-summary.json`).
Repeated cases and installed repetitions are not added again. The large Core
group was not misreported as one clean 537-case run.

Coverage includes ModelBuilder, Graph builder/adapter, TrainingConfig/Session,
evaluation/schema1/2/3/structure and producer evidence, checkpoint,
ExperimentManager/artifact workflow, Reporting, Architecture Comparison,
AUDIT-01 and CLI. Existing GradScaler warnings remain. No long native/CUDA
P01–P35 performance suite was launched. Ruff passes with the five pre-existing
E702 statements in the two legacy files excluded; new files pass that same check.
No unrelated style cleanup was performed. Cached diff/whitespace checks pass.

## 13. Wheel/sdist and public imports

Final-source archives are under `QA/final-package-temp/packaging1/dist/`:

| Archive | SHA-256 |
| --- | --- |
| hyso-0.2.0-py3-none-any.whl | `9061971ee4160ae28f1bf8a11deec3b9807c21e4c7e29a469a83f448c95ff711` |
| hyso-0.2.0.tar.gz | `a5a3fefb0267ce4aa0a1c408d7a988326af07aa7774b39760c0f5fb48f19f544` |

All **162 commercial runtime Python module bytes** match final source; no
web/Studio assets, SQLite/checkpoints, bytecode or excluded development modules
are packaged. Profile/version/dependency declarations are unchanged. Installs
use --no-index/--no-deps/--no-build-isolation/--no-compile and fresh QA targets.
Public imports resolve to those targets outside the checkout.

Both installed packages run the real valid/invalid v2 matrix, including strict
fp32/fp64 restore, missing/corrupted state, metadata/type mismatch, hooks, mutation,
identity and parameterless behavior. One source-only trusted Git reader test is
excluded because archives have no Git metadata; it passes in the source run.
The first -I pytest launcher could not find already-installed user-site pygments;
only its existing dependency search path was added, with installed HYSo target
identity asserted. No dependency was installed to fix the launcher. Final proof:
`QA/final-package-proof.json`, `final-*-contract.xml` and installed logs.

## 14. Final protection hashes

`QA/preservation.json` records:

| Set | Result |
| --- | --- |
| Initial tracked/nonignored untracked files | **731/733 unchanged**; only two authorized implementation files changed |
| Forensic/debug | **47/47 unchanged** |
| Native | **70/71 unchanged**; checkpoint.py explicitly excepted by user |
| External historical acceptance/checkpoint files | **6/6 unchanged** |
| All initially snapshotted external files, including SQLite/WAL/SHM/lock | **11/11 unchanged** |
| Repository database/model/checkpoint files | **197/197 unchanged** |
| Historical stage 13.5.x reports | **14/14 unchanged**, including starting 13.5.8 report bytes |

No initial file was removed. User SQLite was never opened through sqlite3;
only raw bytes were hashed. Existing user models/checkpoints and manifested
P01–P35 historical evidence/matrices were not rewritten. Preservation is limited
to manifest coverage; exhaustive coverage of all ignored external history is
not claimed. Git staging/commits do not imply checkout byte changes.

## 15. Remaining limitations

This is a bounded typed CPU profile, not support for every Registry component,
custom Graph subclass/hook, CUDA, hostile concurrent mutation or arbitrary
interpreter modification. Reference reconstruction allocates up to the admitted
model budget, but no inference. Proof does not include train/eval modes,
gradients, dataset provenance, cryptographic authorship or a new optimizer/replay
guarantee. It is process-local; serialized JSON cannot recreate its live trust.
Legacy P26 consumer projection remains absent and is not disguised as resolved.

## 16. Commits and exact file scopes

| Stage | Commit SHA | Scope |
| --- | --- | --- |
| 13.5.6 | `16b7b50bf1217d4725a44aef34b2693e8bba07d2` | 36 verified Python-first/Graph diagnostics/docs/profile/test files |
| 13.5.7 | `0b02dfd8897a1f42abb7ae3bc27c1ae2a3bbda52` | 10 file-backed dataset/workflow/docs/profile/test files |
| 13.5.8 | `8b15f83db38abd1663856fb34246d67c54706079` | 4 verified example/provenance/real-source/P26 test files; uncertain report excluded |
| 13.5.9 implementation | `644d9ead51f62c01dab86be8c1f49de920d421a1` | checkpoint.py, _evaluation_evidence.py, new contract test, CPU example and usage guide |

Full per-commit file lists, cached diffs/stats and SHA values are retained in
`QA/commits.json` and `QA/stage*-cached*`. Each commit used explicit file staging,
cached diff/check/stat inspection, category checks and an independent candidate
tree check. After each commit, show --stat and empty index were verified.
No C/D file was included except the expressly approved checkpoint.py change.

This final report is recorded in a subsequent documentation-only commit.
Its own SHA, necessarily unavailable inside its own immutable content, is added
to `QA/commits.json` after commit and reported to the user. It can also be obtained
with `git log -1 --format=%H -- docs/reports/stage13-5-9-graph-checkpoint-evidence.md`.
No amend/history rewrite is used to insert a self-referential SHA.

## 17. Remaining uncommitted user work

The complete remaining porcelain inventory is retained in
`QA/remaining-git-status.txt`. It includes the pre-existing forensic plan and
native_admission edits, .vscode files, patch/probe/acceptance scripts, P23
guides/reports/forensic tests/tools, and the 13.5.8 report whose starting bytes
could not be attributed to its final manifest. Every remaining C/D file keeps
its initial bytes. These were intentionally not cleaned or staged.

The report-only commit leaves an empty index and this same unrelated working
set; the workspace is not represented as globally clean.

## 18. Final acceptance

**A PASS; B PASS; C VERIFIED (separate Graph v2); D PASS; E PASS.** These
decisions apply to the explicit supported CPU contract and safe commit scopes.
The old Graph P26 v1 checkpoint field is still missing; that fact is neither
overridden by strict equality nor hidden by test counts. Stop at 13.5.9.
No push, web/Studio development or 13.5.10 work follows.
