# HYSo 13.5.8 — Real-source dataset acceptance and Graph P26 investigation

Date: 2026-11-14 (Europe/Istanbul).

| Independent acceptance gate                   | Decision          |
| --------------------------------------------- | ----------------- |
| **A — Real-world-source dataset integration** | **PASS**          |
| **B — Graph P26 root-cause investigation**    | **PASS**          |
| **C — Graph native checkpoint receipt**       | **STILL MISSING** |

A certifies a bounded integration flow, not useful digit recognition. Both models
achieved only **10% test accuracy**. B establishes the cause and security boundary;
it does not resolve C. No native evidence policy or Core implementation changed.

Evidence root: `%LOCALAPPDATA%/hyso-stage13-5-8-20261114`, abbreviated `QA/`.
Python: `C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe`; scikit-learn 1.7.2,
Torch 2.5.1+cu121, torchvision 0.20.1+cu121. All execution in this stage used CPU;
the actual two-model dataset acceptance used fp32, with CUDA initialization and
socket connections explicitly blocked. No dataset was downloaded.

## 1. Starting Git state and scope

Branch: `feature/hyso-models-next`. HEAD:
`d325c613caa2718ea4010bffbe7560219707843d`.

`QA/baseline.json` records all 651 tracked and 78 nonignored untracked file
hashes, the full initial porcelain status, tracked/untracked inventories, 197
additional repository database/model/checkpoint hashes and 11 external hashes.
Ten tracked files were already modified. Previous 13.5.6/13.5.7 Python-first,
Graph, forensic, packaging and documentation work was present and retained.
The initial manifest was written before source edits.

This stage changes one pre-existing file,
`examples/models/image_classification.py`, and adds:

- `examples/models/real_digits_acceptance.py`;
- `tests/test_real_dataset_acceptance.py`;
- `tests/test_graph_p26_investigation.py`;
- this report.

The existing example gains optional declared provenance, deterministic model
configuration/split IDs, and an existing `validate_metadata` load callback. Its
default training settings and the runtime Core public API remain unchanged.
There is no new Trainer, Evaluator, checkpoint format or experiment system.

## 2. Local source, provenance, permission and prior use

The already installed `sklearn.datasets.load_digits()` loaded 1,797 labeled 8×8
images offline. The local description and official reference identify it as a
copy of UCI's original test partition. It contains processed human handwriting,
not random generated shapes. See the [scikit-learn 1.7.2 reference](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_digits.html).

The primary [UCI source record](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
was inspected for provenance, attribution and its **CC BY 4.0** license. Citation:
Alpaydin, E. & Kaynak, C. (1998), _Optical Recognition of Handwritten Digits_,
UCI Machine Learning Repository, [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49).
The PNG copies retain attribution, source links and transformation disclosure in
`QA/acceptance/provenance.json`; they are not redistributed in HYSo archives.

`QA/source-verification.json` records the verification date and local bundled
`sklearn/datasets/data/digits.csv.gz` hash:
`09f66e6debdee2cd2b5ae59e0d6abbb73fc2b0e0185d2e1957e9ebb51e23aa22`.
The complete uint8 source-pixels plus little-endian int64 labels hash is
`65e5b4619795efb40c1082f4c350c049e4ad552200d369b6ae99e21eef5e72c5`.
The installed source description is retained as `QA/digits-local-description.txt`.

Repository examples/tests/docs had no previous `load_digits` use before this
change. A bounded text-artifact search of retained repository outputs and the
13.5.6/13.5.7 QA directories found no `load_digits`, `optdigits` or `C50P49`
references (`QA/prior-use-search.json`). This does not establish that the user or
any external model has never used these public data. Historical external usage
is unknown; globally untouched holdout status is not claimed.

## 3. Original versus converted images

These are already processed 8×8 block-count images, not raw original handwriting
scans. Each 0–16 source pixel is converted using integer arithmetic:

```text
uint8((uint16(source_pixel) * 255 + 8) // 16)
```

This is nearest rounding with ties up, preserving all 17 source levels
injectively. PNG output is grayscale L, 8×8, compression level 9, optimize false,
without added metadata. Files are named `source-NNNN.png`, preserving the
zero-based row index in the bundled UCI-test copy. No writer ID is invented and
no independent byte comparison to a newly fetched original UCI file is claimed.

All 200 entries record source index, label, collection/row identity, original
pixel hash, conversion parameters, relative PNG path/hash, processed tensor hash
and split. Conversion is deterministic for the recorded Pillow/runtime version.
All images and temporary test copies reside under QA, outside the repository
and user dataset/workspace directories.

## 4. Fixed train/validation/test protocol

Selection uses NumPy `default_rng(1358)`, one fixed per-class permutation,
content deduplication before split assignment, then the first 12/4/4 distinct
images for train/validation/test respectively. No duplicate candidates needed
skipping. Every class 0–9 has exactly those counts: **120 / 40 / 40** overall.
The existing 128-samples-per-split default is unchanged and no override is used.

All three splits derive from the **same sklearn copy of UCI's test collection**.
This is a new small integration protocol, not the original UCI benchmark
train/test protocol. It establishes neither independent-writer generalization
nor a different-distribution test. Writer membership is unavailable here.

Dataset manifest ID, shared by both models:
`9e63026bc4cc8e7b99020ec8b4c4c8371c90f669b852a297b9f1618e62e8a7e4`.

| Split | Deterministic manifest-derived split ID                            |
| ----- | ------------------------------------------------------------------ |
| train | `06722af3945a1e033972d59dae3f9b2ed3dfc39a87a3d3551ece5925d1bd2257` |
| val   | `a303e9520cbb820e8d82038f96bd2e403a7d2515d3e5302cf2f2879ddcc58e8f` |
| test  | `1099bb4bc2adef8ce7bb985888ece17c2f90e5ad2e161f9040d4760b153024b8` |

These IDs are application metadata, not native verification of origin/split.
Both model configurations, SGD settings and the two-epoch limit were fixed
before test inference. Neither validation nor test shuffled. No selection,
hyperparameter adjustment or additional real-data training followed the results.

## 5. Existing dataset API, leakage and preservation checks

The flow reuses `load_image_splits`, `ImageClassificationSplits` and
`ImageDatasetError` unchanged. Source mode L is enforced; deterministic 8×8
bilinear Resize/ToTensor yields CPU fp32 CHW samples and NCHW batches, dividing
PNG integers by 255 without normalization or augmentation.

All 200 source indices, source identities, original pixel hashes, PNG hashes and
processed tensor hashes are unique. The unchanged API's raw/pixel/tensor
cross-split gates pass. Explicit copied-real-image and changed-file tests reject
the fixtures. A 129-sample train fixture rejects the default sample budget.
Seeded loaders reproduce their order despite unrelated global RNG consumption;
validation/test samplers are sequential. Existing RGB/L, channel/class mismatch,
resize collision, corruption and fp32 guards also pass in the regression group.
No unbounded validator or new dataset runtime was added.

`QA/acceptance-audit.json` verifies every original acceptance PNG still matches
its provenance hash after training, evaluation and package read smoke tests.
Content uniqueness does not detect semantic similarity or prove writer isolation.

## 6. ModelBuilder training

Architecture: 1→4 ConvNormAct, 3×3/padding1, norm=None, SiLU, average pooled
10-class classification head. Configuration ID:
`692e4a25ed5733d0f0af4a6f3cb1e14994d20823bedcceb2a19511b5f688de42`.

Both models use seed 1358, CPU fp32, batch size 16, SGD lr=0.01,
momentum=0, weight_decay=0, CrossEntropyLoss, no scheduler/AMP/EMA/resume.
Real training images produce `(N,1,8,8)` input and `(N,10)` logits; actual
forward/backward observes finite gradients. Optimizer parameter identities match
the model. TrainingSession/Trainer performs 8 train batches and 3 validation
batches per epoch, **2 epochs / 16 applied updates**, processing 240 training
sample presentations. All four state tensors differ from same-seed initialization.

| Epoch |   Train loss | Train accuracy % | Validation loss | Validation accuracy % |
| ----- | -----------: | ---------------: | --------------: | --------------------: |
| 1     | 2.3423978964 |               10 |    2.3406642914 |                    10 |
| 2     | 2.3415312926 |               10 |    2.3399126053 |                    10 |

Evidence: `QA/acceptance/builder/{summary,training-result}.json`.

## 7. GraphModelBuilder training

Architecture: 1→4 ConvNormAct stem, parallel 4→4 1×1 convolution and identity,
add merge, 10-class classification head; explicit `features`/`logits` outputs.
The unchanged `GraphTensorOutput(..., 'logits', task='multiclass')` shares Graph
parameters/autograd. Configuration ID:
`ea1f16b0c8ebfb2eb586db28e49ea0b7978f4fbf0f5f5c4838d7a46e841b5e3d`.

The same geometry, channels, output width, finite real-image backward and
optimizer identity checks pass. **2 epochs / 16 updates**, 8 train and 3
validation batches per epoch. All six state tensors differ from initialization.

| Epoch |   Train loss | Train accuracy % | Validation loss | Validation accuracy % |
| ----- | -----------: | ---------------: | --------------: | --------------------: |
| 1     | 2.3953509808 |               10 |    2.3926280499 |                    10 |
| 2     | 2.3906698863 |               10 |    2.3884094715 |                    10 |

Evidence: `QA/acceptance/graph/{summary,training-result}.json`. Loss decreases
slightly in both models, while accuracy remains poor. Equal epochs/updates do
not equalize compute; no performance ranking or useful-learning claim is made.
Observer fit times were approximately 1.631 s and 1.722 s respectively, with
validation/observer overhead included, outside a controlled benchmark protocol.

## 8. Actual trained-model test inference

After strict restoration into a fresh matching model, existing
`evaluate_classification` runs real isolated deepcopy/eval/inference-mode CPU
forward passes on the sequential test loader: **40 images / 3 batches** per
model. Evaluation source is `model_inference`, not caller prediction arrays.

| Metric, fraction                 |                   Builder |                     Graph |
| -------------------------------- | ------------------------: | ------------------------: |
| Accuracy                         |                       0.1 |                       0.1 |
| Macro precision / recall / F1    | 0.01 / 0.1 / 0.0181818182 | 0.01 / 0.1 / 0.0181818182 |
| Micro precision / recall / F1    |           0.1 / 0.1 / 0.1 |           0.1 / 0.1 / 0.1 |
| Weighted precision / recall / F1 | 0.01 / 0.1 / 0.0181818182 | 0.01 / 0.1 / 0.0181818182 |

For both models, confusion rows are true classes 0–9, columns predictions 0–9:

```text
[[0,0,0,0,0,0,0,4,0,0],
 [0,0,0,0,0,0,0,4,0,0],
 [0,0,0,0,0,0,0,4,0,0],
 [0,0,0,0,0,0,0,4,0,0],
 [0,0,0,0,0,0,0,4,0,0],
 [0,0,0,0,0,0,0,4,0,0],
 [0,0,0,0,0,0,0,4,0,0],
 [0,0,0,0,0,0,0,4,0,0],
 [0,0,0,0,0,0,0,4,0,0],
 [0,0,0,0,0,0,0,4,0,0]]
```

Every class support is 4. Only class 7 is predicted; its precision/recall/F1 is
0.1/1/0.1818181818, with zero precision/recall/F1 for the other classes.

| Native evidence              | SHA-256                                                            |
| ---------------------------- | ------------------------------------------------------------------ |
| Builder model state          | `eb059069b31d00ab14f8c632b0f5d5299d283a5e3b34703a98d2ab9c23d69631` |
| Graph model state            | `6a64a9442f978d34f00653cc7bc04c47b47707bdc4a6151dc7065c270beb7ac1` |
| Shared ordered test inputs   | `3af1f2c77c0da3691221a3de66cda9bf70d8c1a291d9464d5204a23dc9d690e0` |
| Shared ordered test targets  | `804259823c84bfbab81dd5d467fc49ad10a0fe4a19e87c8c46452f81c90715b0` |
| Shared ordered paired stream | `fbde3a9abbd470562999bea1e962f907564571342853931f65889213c258b8e0` |

Full metrics/raw logits/evidence are retained in each `evaluation.json`.
Schema2 producer evidence verifies state and observed stream; dataset source ID
and split remain declared. Native preprocessing/origin evidence remains missing.
External source verification and manifests are separate supporting evidence.

## 9. Checkpoints, experiments and artifacts

Existing `save_training_checkpoint` stores model state, completed epochs,
history and context containing settings, seed, source/license metadata, dataset
ID, split IDs, architecture and configuration ID. Existing restricted
`load_training_checkpoint` uses weights_only=True and default strict state load;
the example additionally validates full saved context before mutation. Fresh
matching models yield bit-identical logits on the same fixed validation batch.

| Model   | Checkpoint file SHA-256                                            | Strict inference roundtrip | Native receipt |
| ------- | ------------------------------------------------------------------ | -------------------------- | -------------- |
| Builder | `10e328de0f927a4940564c5dd9ccdd226427835293d43294c26468b43771107b` | PASS                       | native         |
| Graph   | `1d33a9915f13fa9caafca14e411ad4b34809388c0010377815f81094ef1f1adf` | PASS                       | **missing**    |

The user explicitly approved an exception to the initial no-new-database rule:
only new ExperimentManager stores inside this isolated QA directory are allowed.
No user SQLite workspace was opened. The existing manager explicitly migrates
its new local stores to artifact schema2, records training/evaluation runs,
links dataset/training/checkpoint inputs and evaluation/JSON/Markdown/static
HTML report outputs, then verifies **7/7 artifacts per model** by actual bytes.
Both runs complete in each store. Static reporting output is only a QA artifact;
no web application/source/service was created.

| Model   | Training run ID                    | Evaluation run ID                  |
| ------- | ---------------------------------- | ---------------------------------- |
| Builder | `72f2f97732c04bdcbbeb55b29dfb5957` | `695ee3871aba4d338cbdaf02b28278f7` |
| Graph   | `27bfc3c9778c4218b2dbaa4000ce6ebe` | `90f4a06e4cfb4ec99db766ec450924a0` |

Full experiment/artifact IDs and verifications are in each `summary.json`.
Graph's artifact SHA relation is a verified experiment association, not a native
P26 trusted-load receipt. These are model-only restore flows; automatic
optimizer/replay resume is not claimed for the example models.

## 10. Real-dataset acceptance tests

`tests/test_real_dataset_acceptance.py` is separate from P26 diagnostics and
contains **5 passing cases** (`QA/real-data-final.xml`): offline local loading,
source/label/PNG conversion and 17-level preservation, balanced disjoint splits,
deterministic conversion/seeded loaders and immutable inputs, real-content
duplicate rejection, changed-file rejection and the retained 128-sample budget.
Some checks are grouped within the same case. No skip/xfail was used locally.
An earlier four-case run passed; the fifth was added to cover the explicit cap.

Reproduction uses the Python above, outside-output paths and no download:

```powershell
python -B -m examples.models.real_digits_acceptance --output "$env:LOCALAPPDATA/hyso-stage13-5-8-20261114/fresh-acceptance"
python -B -m pytest tests/test_real_dataset_acceptance.py -q -p no:cacheprovider --basetemp "$env:LOCALAPPDATA/hyso-stage13-5-8-20261114/fresh-data-tests"
```

The actual run is `QA/acceptance`, recorded in `acceptance-final.log`; a separate
network/CUDA guard wrapped it. The first launcher incorrectly replaced the
socket _type_ before Torch imported SSL and failed during imports. No dataset or
training had begun. The corrected launcher blocks connect/connect_ex after
imports, preserving SSL's type definition; `acceptance.log` retains the failure.
No production change was made for that launcher mistake.

## 11. P26 reproduction and root cause

The independent short CPU reproduction was performed before the negative group,
using fresh QA-only Graph models and checkpoint. `QA/p26-repro.json` records
successful strict restoration, equal outputs and unchanged checkpoint bytes,
but `no_matching_trusted_load_receipt`. The exact unmatched name is
`graph._contract` (or `_contract` on an unwrapped GraphModel).

1. **Purpose/nonpersistence.** `hyso/models/graph.py:287` registers an empty
   floating buffer with `persistent=False`. Source behavior indicates its purpose
   is a module-wide dtype/device anchor, including parameterless graphs; it
   follows ordinary Module `.to()` conversions. It stores no learned weights or
   architecture digest. Nonpersistence keeps this runtime anchor out of serialized
   tensor state. No additional historical author-intent claim is needed.
2. **Behavioral effect.** Graph forward checks input and node output dtype/device
   against it (`graph.py:300`, `314`), alongside shapes from configuration/report.
   An empty payload does not make its metadata irrelevant: changing only the
   anchor to fp64 makes fp32 input fail. The negative test reproduces that.
3. **Expected PyTorch state behavior.** `state_dict()` excludes the buffer;
   strict `load_state_dict()` does not require it. This is directly observed in
   the actual models. Persistent tensors and the anchor have different scopes.
4. **P26's larger scope.** `_evaluation_evidence.py:20` hashes all named parameters
   AND named buffers, including the anchor's name/dtype/shape. At `:66`,
   `remember_checkpoint` requires every such name to exist among frozen loaded
   checkpoint fingerprints. Graph fails this subset gate. Merely matching its
   `state_dict()` cannot satisfy the documented full named-tensor contract.
5. **Security interpretation.** This reproduced absence is a fail-closed coverage
   mismatch, not evidence that the strict weight roundtrip failed or that an
   exploit occurred. Ignoring arbitrary nonpersistent buffers would open a real
   evidence gap: they can affect computation/validation. The dtype-mismatch test
   demonstrates why global exclusion would be unsafe.
6. **Reconstructible metadata, with conditions.** A fresh matching CPU fp32 Graph
   regenerates the empty anchor correctly. That fact alone does not prove that
   _every_ graph or nonpersistent buffer is safe to reconstruct. Configuration,
   runtime dtype/device, shape, class mapping and adapter selection need explicit
   validation; current P26 v1 has no such reconstruction proof.
7. **What strict loading verifies.** `checkpoint.py:162` checks exact state keys,
   tensor shapes and dtypes; `:374` uses default strict loading. Saved parameters
   and persistent buffers restore. Python architecture/report/adapter semantics
   are not generally authenticated by tensor-state equality. The diagnostic
   9×9 graph has the same state schema as the 8×8 graph and can load its tensors,
   then rejects the 8×8 input. Existing metadata validation can reject that
   configuration before mutation, as demonstrated and used by the example.
8. **When native receipt is trustworthy.** Under today's contract: restricted
   restore from the SAME privately hashed snapshot, successful strict load,
   frozen pre-hook fingerprints, exact state-dict AND all named-tensor bytes,
   supported CPU tensors and a live weak-reference receipt on the exact model
   object whose current model hash matches. No caller SHA/path/deepcopy supplies
   it. Graph does not meet the complete named-state requirement, so C stays missing.

This matches the unchanged historical rules in
`docs/guides/p26-native-evidence.md` (trusted checkpoint section) and existing
producer tests for nonpersistent/custom state, frozen pre-hook bytes, snapshot
identity, mutation invalidation and uncaptured reloads. The current loader hashes
the private snapshot, never silently reserializes a source checkpoint.

## 12. P26 negative evidence tests

`tests/test_graph_p26_investigation.py` passes **9 cases**, including three
parameterized invalid-load cases (`QA/p26-final.xml`). Tiny local fixtures use
weights_only=True; no unknown checkpoint or unsafe pickle fallback is used.

| Scenario                                     | Real code path and observed result                                                                                                |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Unmodified valid Graph checkpoint            | Strict load and output equality PASS; actual evaluation receipt remains missing                                                   |
| Wrong class configuration                    | Checkpoint tensor schema rejects 10→9 class mismatch; no receipt                                                                  |
| Missing parameter                            | Explicit local bounded negative payload loses one parameter; key validation rejects; target unchanged, no receipt                 |
| Modified persistent buffer                   | BN control initially has native receipt; running_mean mutation invalidates it; post-load buffer substitution also cannot mint one |
| Wrong model-state hash                       | Live trusted control queried with false state hash returns missing                                                                |
| Wrong checkpoint file SHA                    | Actual producer binds hashed snapshot; false checkpoint EvidenceScope is rejected with EvidenceError                              |
| Strict load failure                          | One-shot failing trusted pre-hook causes TrainingCheckpointError, successful rollback, unchanged target and no receipt            |
| Nonpersistent contract mismatch              | Weights strictly load but fp64 anchor rejects fp32 forward; model hash differs and receipt stays missing                          |
| Same tensor schema, different Graph geometry | Generic load alone accepts state; runtime rejects wrong geometry; explicit metadata validator rejects load before mutation        |

The persistent-buffer control is deliberately a small BN model, so Graph's
already-missing receipt does not mask whether mutation invalidation works.
The file-hash test uses the real producer/evidence binding boundary: the loader
has no caller `expected_sha256` argument to invent. A modified well-formed file
is a distinct snapshot, not proof of authenticity against an external expected
hash. Imported JSON schema consistency is not process-local receipt authenticity.

Initial P26 run: **8 passed / 1 failed**, because the new diagnostic post-hook
accidentally returned a Tensor instead of None, violating PyTorch's hook
contract. That fixture was corrected; no Core change followed. Original failure
is retained in `QA/p26.xml`. This is not a repaired P26 production vulnerability.

## 13. Options, security impact and implementation decision

| Option                                                           | Security/compatibility impact                                                                                                                                    | This stage    |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| Keep existing fail-closed receipt behavior                       | Preserves full named-tensor evidence; Graph link stays unavailable                                                                                               | **Retained**  |
| Globally omit nonpersistent buffers from hashes/coverage         | Could falsely bind behavior-changing uncheckpointed state; changes historical evidence scope                                                                     | Rejected      |
| Make Graph `_contract` persistent                                | Changes Graph state keys/strict legacy loading; still does not authenticate all architecture semantics                                                           | Proposal only |
| Versioned reconstructed-metadata proof                           | Could separately bind exact Graph configuration, adapter, empty-anchor shape/dtype/device and construction policy; requires a new explicit receipt scope/version | Proposal only |
| Persist an architecture/contract digest in a future state format | Could authenticate more semantics but needs schema/version/migration and legacy behavior decisions                                                               | Proposal only |

A future scoped proposal must reject arbitrary nonpersistent buffers, subclasses
and load-hook substitutions; verify exact known construction/configuration and
runtime metadata; retain the existing file snapshot hash, parameter/persistent
buffer fingerprint and object/state identity requirements; and distinguish
serialized bytes from independently reconstructed contract evidence. Necessary
regressions include all scenarios above, parameterless Graphs, dtype/device
conversions, adapter output mismatch, same-schema/different topology/activation,
extra nonpersistent state, source mutation hooks, failed/rolled-back restore,
legacy checkpoint loading, deepcopy/reload invalidation and old schema1/2/3
readers/historical hash vectors.

Those changes would alter the P26 or Graph serialization contract. **None is
automatically implemented here.** No clearly safe Core implementation defect
within the existing contract was identified. Only the example's existing
metadata-validator facility is used more explicitly. No receipt was fabricated,
hash relaxed, strict=False used, historical evidence rewritten or weight changed
to obtain acceptance.

## 14. CPU regression and packaging

| Final independent pytest group              |  Passed | Failed/errors/skips |
| ------------------------------------------- | ------: | ------------------- |
| Real-source dataset (`real-data-final.xml`) |       5 | 0 / 0 / 0           |
| Graph P26 diagnostics (`p26-final.xml`)     |       9 | 0 / 0 / 0           |
| Selected Core regression (`core.xml`)       |     537 | 0 / 0 / 0           |
| Build/install/commercial (`package.xml`)    |      10 | 0 / 0 / 0           |
| **Unique successful final cases**           | **561** | **0 / 0 / 0**       |

Core selection: ModelBuilder, Graph builder, Python-first/GraphTensorOutput,
TrainingConfig/Session, training checkpoint, evaluation/schema1/2/3 producer and
structure binding, experiment artifacts/workflow, Reporting, Architecture
Comparison/AUDIT-01, CLI and existing image dataset tests. Exact selections are
retained in XML/logs. The CPU regression took 252.26 s; build/install tests
310.68 s. Existing GradScaler deprecation warnings remain (50 in Core group).
No long native/CUDA P01–P35 suite or additional real-data epochs were launched.
Ruff checks pass for new/adjusted Python sources; `git diff --check` passes.

Runtime Python bytes are unchanged. Nevertheless fresh wheel/sdist were built
from isolated source snapshots and installed offline with --no-index/--no-deps
into fresh QA targets. Commercial allowlist remains **162 modules**, with
`hyso/datasets.py` included and exact source-byte matches. No web/Studio,
SQLite/checkpoint, excluded development module or bytecode enters the archives.
Dependencies/version/profile are unchanged; scikit-learn was **already** a
declared dependency before this stage, and was not installed or newly added.

Release proof: `QA/package-proof.json`, archives under
`QA/package-temp/packaging1/dist/`:

| Archive                     | SHA-256                                                            |
| --------------------------- | ------------------------------------------------------------------ |
| hyso-0.2.0-py3-none-any.whl | `efd9497c561fe873fcda389d02a4a663ddef526c187291b616e678950c05f903` |
| hyso-0.2.0.tar.gz           | `2ebdcd3929b6eed0c3bbf50f9149bd462ed6ddc467b713878519a98b5522fae3` |

Both installed commercial packages independently read the actual 120/40/40
digits files, reproduce the same dataset manifest ID and yield `(16,1,8,8)`
fp32 test batches. Imports resolve to installed QA targets outside the checkout.
These extra real-file read checks perform **no training**, download or database
access (`QA/installed-real-{wheel,sdist}.log`).

## 15. Source, user data and historical evidence protection

The original 13.5.6 protection manifest and 13.5.7 baseline were read and remain
unchanged. Final comparisons are retained in `QA/preservation.json`.

| Protected set                                                            | Hash result                                                    |
| ------------------------------------------------------------------------ | -------------------------------------------------------------- |
| Pre-stage tracked/nonignored untracked files                             | 728/729 unchanged; only existing example intentionally changed |
| Forensic/debug files from prior manifest                                 | 47/47 unchanged                                                |
| Native files from prior manifest                                         | 71/71 unchanged                                                |
| External historical checkpoint/acceptance files                          | 6/6 unchanged                                                  |
| Initial external files, including user workspace SQLite/WAL/SHM/lock     | 11/11 unchanged                                                |
| Repository user database/model/checkpoint files additionally snapshotted | 197/197 unchanged                                              |
| All pre-existing Core Python sources                                     | 172/172 unchanged                                              |
| 13.5.6 report/acceptance documents                                       | 8/8 unchanged                                                  |
| 13.5.7 report                                                            | 1/1 unchanged                                                  |
| All pre-existing manifested docs                                         | 128/128 unchanged                                              |

User SQLite files were read only as raw bytes for hashing, never opened through
sqlite3 or migrated. The explicitly approved new QA ExperimentManager stores are
separate. P01–P35 protection is demonstrated for manifested historical documents
and external acceptance matrices/checkpoints. Ignored historical output trees
and unmanifested external files are **not** claimed to have exhaustive initial
hash coverage. No historical acceptance suite/checkpoint was regenerated.

Final branch/HEAD match the baseline and staging is empty. No commit, push,
reset, clean or restore was performed. Pre-existing dirty Core/forensic changes
were preserved rather than attributed to this stage.

## 16. Technical limitations

Two epochs on 120 training examples produce chance-level accuracy and a single
predicted class. Decreasing losses and genuine weight updates establish execution,
not adequate learning quality. Small test support gives no useful precision or
generalization guarantee. Source writers are not separated by this new split;
external prior usage is unknown. Exact hashes do not reject semantic duplicates.

P26 native origin/preprocessing verification remains unavailable. Native
model/stream evidence, an experiment artifact association and a trusted-load
receipt have distinct meanings. A strict Graph roundtrip does not close the
full named-state receipt gap. CPU evidence cannot certify CUDA, cross-process
receipt continuity, arbitrary model behavior or optimizer replay for these models.

## 17. Final stage decision

**A — Real-world-source dataset integration: PASS.** Local permitted human-source
images traversed the existing bounded dataset, two real CPU training pipelines,
separate validation/test inference, strict checkpoint reload and verified
ExperimentManager/report artifact flow. Model quality remains poor.

**B — Graph P26 root-cause investigation: PASS.** Independent reproduction,
exact code-path explanation, behavioral contract mismatch, negative controls,
historical policy comparison and explicit future options/regressions are recorded.

**C — Graph native checkpoint receipt: STILL MISSING.** Current policy correctly
withholds it; no P26/Graph serialization contract change was authorized or applied.

Stage 13.5.8 stops here. No web development, commit/push or 13.5.9 work follows.
