# HYSo 13.5.7 — File-backed dataset training and evaluation

Date: 2026-11-14. Decision: **PASS for generated-file integration acceptance**.
No real-world dataset or model-quality acceptance is claimed. The Graph native
checkpoint evidence limitation is retained explicitly in sections 8 and 14.

QA root: `%LOCALAPPDATA%/hyso-stage13-5-7-dataset-20261114` (abbreviated `QA/`
below). Python: `C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe`; Torch
2.5.1+cu121, torchvision 0.20.1+cu121, Pillow 12.0.0. All training used CPU fp32.
Installed CUDA-capable binaries do not imply CUDA execution.

## 1. Initial Git state

Branch `feature/hyso-models-next`, HEAD
`d325c613caa2718ea4010bffbe7560219707843d`. `QA/baseline.json` was created before
source edits, independently of the preserved 13.5.6 manifest. It records 724
existing tracked/untracked file hashes, including 73 untracked files, raw hashes
of five Studio workspace files, the protection results and empty staging.

The checkout already contained uncommitted Python-first, Graph, documentation,
packaging and forensic work. Those changes were not reset. Relative to this
baseline, this task changes only `README.md`, `examples/README.md`,
`commercial-profile.json`, `tests/test_packaging.py` and
`tests/test_commercial_distribution.py`; the existing text/tests are extended.
719/724 pre-task files remain byte-identical. Five files are new: the dataset
module, executable example, dataset tests, usage guide and this report.
No pre-task file is removed. In particular, the pre-existing dirty
`hyso/models/graph.py` and forensic edits remain byte-identical to the baseline.

Final branch/HEAD are unchanged, staging remains empty, and `git diff --check`
passes. Git emits existing LF/CRLF conversion notices for unrelated dirty files;
these are not whitespace failures. No commit, push, reset, clean or restore ran.

## 2. Existing Core capabilities and integration gaps

| Existing API | Supported behavior reused / contract preserved |
| --- | --- |
| ModelBuilder / registry | Validated sequential CNN configuration, classification head, build and state_dict |
| GraphModelBuilder | Validated branching graph with named outputs; Core schema and diagnostics unchanged |
| GraphTensorOutput | Explicit single-input `logits` selection, shared parameters/autograd and strict state_dict |
| TrainingConfig / TrainingSession / Trainer | Existing CPU loaders, SGD, CrossEntropyLoss, one native fit call, validation and TrainingResult |
| Evaluation | Actual isolated eval-mode model inference, class metrics/confusion matrix and schema2 producer evidence |
| BenchmarkRunner | Existing bounded explicit measurements; dataset integration does not require new timing machinery or a benchmark claim |
| Checkpoints | Generic safe model-state save/load for both models; automatic session resume retains its existing ModularCNN-only scope |
| ExperimentManager | Existing runs, metrics, schema2 artifacts, links, explicit migration and byte verification |
| Reporting | Existing TrainingResult and UnifiedAnalysisReport inputs; native evaluation is adapted through EvidenceRecord/EvidenceIndex |
| CLI / Python examples | Existing terminal groups retained; a separate executable Python example needs no new command group |

`augmentation.get_loaders` already uses torchvision ImageFolder, supports
grayscale preprocessing and validation remapping, and makes a random split when
validation is omitted. It has no explicit test split, content-leakage gate,
per-loader shuffle generator or source-content manifest. An actual duplicate
train/validation image was accepted by that unchanged helper
(`QA/legacy-gap.json`). Its legacy behavior was preserved. The new small-data
boundary supplies the missing contracts, reusing ImageFolder, transforms,
TrainingConfig.make_loaders and DataLoader. No second trainer, evaluator,
checkpoint format, experiment store or model schema was introduced.

## 3. Added and adjusted Python functionality

`hyso/datasets.py` exposes `load_image_splits`, `ImageClassificationSplits` and
`ImageDatasetError`. It reads actual PNG/JPEG bytes, requires RGB/L source modes,
sorts the train vocabulary and enforces that same vocabulary in every present
nonempty split. It performs deterministic bilinear resize, ToTensor, explicit
fp32 conversion and optional per-channel normalization. CPU batching uses
independent seeded generators; test ordering stays sequential. Optional splits
remain absent, with no automatic split creation.

Raw, decoded-pixel and processed-tensor hashes reject cross-split copies,
including distinct images that become identical after resize. A subsequent read
checks the original raw hash. The manifest includes relative paths, class map,
split counts, preprocessing/runtime versions and dataset ID. Core preflight runs
before geometry/class-width checks, preserving complete original diagnostics.
Validation is bounded to 128 samples per split by default, 32 MiB/file,
16,777,216 decoded pixels and 1,048,576 processed elements/sample. Invalid or
nonfinite fp32 normalization receives an explicit dataset diagnostic.

`examples/models/image_classification.py` accepts explicit dataset/output paths,
Builder or Graph selection, channels, dimensions, seed, normalization and origin.
It requires a separate test split and a fresh output disjoint from the input.
It checks CPU/dtype, actual image forward/backward, finite observed gradients,
optimizer parameter identity, training, strict checkpoint restoration, real test
inference, reports and artifact verification. No existing output is overwritten.
The [usage guide](../guides/file-image-classification.md), README links and example
index document the entire workflow.

## 4. Dataset provenance and fixture boundary

Repository image discovery found historical generated plots, without an
identified permitted labeled train/val/test dataset. No image dataset was
downloaded. Acceptance therefore uses the expressly allowed **generated
fixture**, created only under `QA/generated-rgb` by the deterministic pixel
formula in `tests/test_image_dataset.py:make_images`.

There are 16 real image files: RGB 12×10 PNG/JPEG, classes `alpha=0`, `zebra=1`,
train 8, validation 4, test 4. Split/class/index-dependent image contents are
distinct under the exact content checks. Preprocessing resizes to 8×8, producing
float32 NCHW batches `(2,3,8,8)` without normalization in the acceptance runs.
Separate generated L-mode files also exercise grayscale loading and complete
Graph training in tests. `QA/fixture-provenance.json` declares the source,
formats, counts and `real_world_dataset=false`.

Dataset ID:
`24cdac46ffcff0a2ca6f23feacc417911fafeacb4c33caae40d14511c8456e72`.
All 16 source-file hashes still match the pre-training manifest after every
acceptance flow. These images establish file reading and computational
integration only; they do not establish scientific dataset independence,
generalization, production accuracy or useful learning.

## 5. Actual ModelBuilder training

Final-source command, using the Python executable above:

```powershell
python -B -m examples.models.image_classification --dataset "$env:LOCALAPPDATA/hyso-stage13-5-7-dataset-20261114/generated-rgb" --output "$env:LOCALAPPDATA/hyso-stage13-5-7-dataset-20261114/accept-builder-final" --model builder --channels 3 --epochs 2 --batch-size 2 --seed 17 --dataset-origin generated_fixture
```

ConvNormAct 3→4, 3×3/padding1, no normalization, SiLU, native classification head
with two logits. CPU fp32, SGD lr=0.01, momentum=0, weight_decay=0,
CrossEntropyLoss, no AMP/scheduler/EMA/accumulation or resume. Actual image probe
had finite gradients and the optimizer references the model's own parameters.
Two completed epochs, four train batches and two validation batches per epoch,
eight applied updates. Observed fit wall time: 0.197289 seconds (session observer
timing, not a performance acceptance measurement).

| Epoch | Train loss | Train accuracy (%) | Validation loss | Validation accuracy (%) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6927183270 | 50 | 0.6891003847 | 50 |
| 2 | 0.6944497675 | 50 | 0.6889622808 | 50 |

Evidence: `QA/accept-builder-final/summary.json` and `training-result.json`.
Training completion is observed; improving or high model quality is not claimed.

## 6. Actual GraphModelBuilder training

Run the same command with `--model graph` and fresh output
`QA/accept-graph-final`. The graph has a 3→4 ConvNormAct stem, parallel 4→4 1×1
convolution and identity branches, add merge, and a two-class classification
head. Outputs are `features` and `logits`; GraphTensorOutput explicitly selects
`logits` with `task="multiclass"`. Input geometry, output class count, shared
optimizer parameters and finite connected gradients are checked on real images.

The same seed, CPU fp32, optimizer, batch and epoch budget apply. Two epochs,
four train/two validation batches per epoch, eight applied updates; observed fit
wall time 0.1927725 seconds under the same non-benchmark timing scope.

| Epoch | Train loss | Train accuracy (%) | Validation loss | Validation accuracy (%) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6962194443 | 50 | 0.6946740746 | 50 |
| 2 | 0.6990166306 | 50 | 0.6940992773 | 50 |

Evidence: `QA/accept-graph-final/summary.json` and `training-result.json`.

## 7. Actual inference and evaluation

After loading each saved model into a fresh matching model, the existing
`evaluate_classification` performs isolated deepcopy/eval/inference-mode forward
passes on the separate sequential test loader: four samples in two batches.
Results identify `source.kind=model_inference`, CPU, ImageFolder, dataset ID,
class map and the actual model-state hash. No ready-made prediction array was
substituted for model execution. Test was not used for training or selection.

| Test metric (fraction) | Builder | Graph |
| --- | ---: | ---: |
| Samples | 4 | 4 |
| Accuracy | 0.5 | 0.5 |
| Macro precision | 0.25 | 0.25 |
| Macro recall | 0.5 | 0.5 |
| Macro F1 | 0.3333333333 | 0.3333333333 |
| Micro precision / recall / F1 | 0.5 / 0.5 / 0.5 | 0.5 / 0.5 / 0.5 |
| Weighted precision / recall / F1 | 0.25 / 0.5 / 0.3333333333 | 0.25 / 0.5 / 0.3333333333 |

Confusion matrix rows are true classes and columns predicted classes, ordered
`alpha,zebra`: Builder `[[2,0],[2,0]]`; Graph `[[0,2],[0,2]]`.
Per-class support is two each. Builder alpha precision/recall/F1 is
0.5/1/0.6666666667 and zebra 0/0/0; Graph has the reverse assignment.
These weak fixture results are reported as observed.

`evaluation.json` preserves native model-state and ordered evaluated-stream
hashes. Split and source ID remain caller-declared. Native preprocessing and
dataset-origin verification remain `missing`; the separate manifest/context
records their declared configuration without promoting it to native evidence.
EvidenceIndex and the existing Reporting API produce JSON, Markdown and static
HTML reports for both actual runs.

## 8. Checkpoint verification and exact supported scope

Both runs use existing `save_training_checkpoint` / `load_training_checkpoint`,
with local model state, epoch history and model/dataset/training metadata. Load
uses the native restricted checkpoint path and strict state matching. A fixed
validation image batch produces **bit-identical outputs** before saving and after
loading in both models. Checkpoint equality is not tested on test data for model
selection. No optimizer/epoch-resume claim is made for these example models.

| Model | Actual checkpoint SHA-256 | Strict inference roundtrip | Native checkpoint receipt |
| --- | --- | --- | --- |
| Builder | `98e28554fc71c29fed4690d48fa9cd5f4e7f4ce4d5d5c17811ddb3855cb4e21a` | PASS | native; file hash matches |
| Graph | `6b2597b61ff0155077665969d55d0dbda47a57522d686608bd7ff7598b44e2d1` | PASS | missing |

GraphModel registers nonpersistent `_contract`; the unchanged native evidence
policy requires named model buffers to be represented in loaded state before
issuing a full-state checkpoint receipt. Consequently Graph evaluation retains
`no_matching_trusted_load_receipt`, with null native checkpoint hash. The model
does load correctly, but its file SHA and experiment relation are caller-recorded
associations. Core policy and graph state were not modified to manufacture a
receipt. Model-state and evaluated-stream evidence are native in both runs.

## 9. ExperimentManager and reporting integration

Each fresh output contains its own explicitly created/migrated experiment store.
The existing manager records two completed runs (training and evaluation), model
configuration, requested/effective training settings, dataset identity/origin,
preprocessing, class map, seed, native training history and actual test metrics.
The evaluation run links dataset manifest, training result and checkpoint as
three inputs; it records evaluation plus JSON/MD/HTML report outputs.

All seven artifacts per flow pass native byte/hash verification with their
explicit local paths. Reference locators retain the existing privacy behavior;
verification is not inferred from an unavailable implicit locator. Exact run and
artifact IDs are in each `summary.json.tracking`. Final training run IDs:
Builder `268ba78ad8a74923aa0417f6273a1c0a`, Graph
`8cf64eb825d7435bb29114aca15a8493`. These stores are isolated QA data, not the
user's Studio SQLite or a new custom tracking implementation.

## 10. Negative tests and before/after evidence

`tests/test_image_dataset.py` has 24 final passing parameterized cases, including:
RGB/L NCHW fp32; stable class indices and seeded shuffle despite global RNG
consumption; model channel/class mismatch; empty dataset/class; corrupt image;
unsupported file; mismatched split vocabulary; raw and resized-tensor split
leakage; changed source; explicit graph classification-output selection;
normalization; no implicit test fallback; no existing-output overwrite; unchanged
inputs and full Core invalid-model diagnostic equality. Three complete CPU
flows cover Builder RGB, Graph RGB and Graph grayscale, including real inference,
checkpoint and seven verified tracking artifacts.

Before implementation, the selected contract run failed 11 cases because the new
dataset API did not exist (`QA/red.xml`, eight deselected). This is an integration
gap, not evidence of 11 defects in legacy Core. The first full integration run
passed 16 and failed three: reference artifacts needed explicit verification
paths, and the initial Graph assumption incorrectly demanded a native receipt.
The example now uses the supported verification contract and preserves the real
Graph missing status. The subsequent 19-case run passed; output-protection and
train-only cases expanded coverage to 21.

Three additional tests were executed before correcting an observed fp32 boundary
defect: global float64 default leaked through ToTensor, tiny std rounded to zero
with a generic error, and huge mean created nonfinite tensors. All three failed
(`QA/dtype-red.xml`). Explicit fp32 conversion, preprocessing diagnostics and a
finite-result gate correct those failures. The final complete run passes **24/24**
(`QA/dataset-final.xml`). No xfail/skip masks these cases; original failed evidence
was retained. Unrelated Core was not edited to suppress failures.

## 11. CPU regression results

| Final successful pytest group | Passed | Errors / failures / skips |
| --- | ---: | --- |
| File-backed dataset and complete flows (`dataset-final.xml`) | 24 | 0 / 0 / 0 |
| Selected Core regressions (`core.xml`) | 374 | 0 / 0 / 0 |
| Build/install/commercial tests (`packaging.xml`) | 10 | 0 / 0 / 0 |
| **Unique successful cases** | **408** | **0 / 0 / 0** |

The selected Core run covers ModelBuilder, Graph builder, TrainingConfig,
checkpoint, Evaluation and producer evidence, experiment artifacts/workflow,
Reporting, Architecture Comparison and the existing augmentation split/remapping
regression. Prior 13.5.6 successful wider regression evidence was inspected and
retained, rather than re-running the full suite. BenchmarkRunner/native/CUDA and
long performance acceptance suites were not launched.

Final-source content tests add three repeated checks, excluded from the unique
count (`package-content-final.xml`); the same three were repeated after README
metadata changes (`release-package-content.xml`). Both final CLI flows completed
two real epochs. Additional isolated installed-package flows perform one epoch
each (section 12), outside pytest counts. The existing Trainer emits a GradScaler
deprecation FutureWarning even with AMP disabled; the 374-case Core run had 45
warnings. No new dependency or CUDA execution was needed. Ruff passes for the
new/adjusted Python files; `git diff --check` passes.

## 12. Packaging verification

The commercial allowlist adds only `hyso/datasets.py`: 161→162 runtime Python
modules. Exclusions, dependency declarations, package version and public exports
are unchanged. Packaging test inventory is adjusted and installed commercial
smoke now reads file-backed images and batches through the new module.

Fresh wheel/sdist under `QA/release-package-temp/packaging1/dist` contain all
162 allowed module bytes exactly matching current source. Sdist additionally
has its existing build script. No Studio service, HTML/CSS/JavaScript, SQLite,
model checkpoint or excluded development module is packaged. Static report HTML
exists only as an explicit run output. The package content/metadata and allowlist
tests pass with final README metadata.

Both release archives are installed offline without dependencies into fresh
isolated QA targets. Imports resolve to those installed targets outside the
checkout. The checkout example is then executed against installed HYSo: Graph
from wheel, Builder from sdist, each one real CPU epoch, four test samples,
matching checkpoint inference and seven verified artifacts. The Graph receipt
remains missing and the Builder receipt native, consistently with checkout runs.
Archive SHA-256 values, source-byte matches, installed paths and flow outcomes are
retained in `QA/release-package-proof.json`; install/smoke logs are alongside it.
No global environment install or source-tree build mutation was performed.

## 13. Source, data and historical evidence preservation

The original protection manifest remains at
`%LOCALAPPDATA%/hyso-stage13-5-6-20261113/protection.json`. Comparisons against its
recorded hashes and the separate 13.5.7 baseline give:

| Protected set | Result |
| --- | --- |
| Forensic/debug files | 47/47 hash-identical |
| Native files | 71/71 hash-identical |
| External acceptance/checkpoint files | 6/6 hash-identical |
| All pre-existing `hyso/*.py` recursively | 171/171 hash-identical |
| Existing audit/report documents | 27/27 hash-identical |
| Existing 13.5.6 report/acceptance files in baseline | 8/8 hash-identical |
| AUDIT-01 closure's recorded task-file hashes | 3/3 match |
| Raw user Studio workspace files | 5/5 hash-identical |
| Generated RGB source image files | 16/16 hash-identical |

P01–P35 historical material covered by these manifests is preserved; no native
acceptance evidence was regenerated or edited. All earlier untracked files
remain present. Original 13.5.6 reports and Studio historical documents were not
deleted. The previous audit established three user models and 12 logical
records. This stage did not open the user SQLite: byte-identical raw workspace
files preserve that prior state, without claiming a fresh SQL recount.

Final proof, source hashes, test XML summaries, Git state and archive proof are
collected under `QA/final-proof.json`. Studio source remains absent. No web/GUI
service, user-store migration, external checkpoint overwrite, new framework,
commit or push occurred. Existing Core API implementations and public exports
are unchanged; the new API is additive under `hyso.datasets`.

## 14. Open technical limitations

- No permitted real-world labeled dataset was available. Generated PNG/JPEG
  acceptance proves the integration chain, with no accuracy/generalization claim.
- Graph strict checkpoint load/inference works, but native checkpoint-bound P26
  provenance is missing because of the existing nonpersistent-buffer policy.
- Split/source declarations and manifest preprocessing/origin are not promoted
  to native provenance; the native evaluation stream only proves tensors seen.
- Exact raw/pixel/processed equality checks cannot detect all semantic duplicates
  or establish independent dataset origin. Source files/transform objects should
  remain unchanged while running.
- This intentionally bounded example supports CPU fp32 multiclass, single-input
  classification, PNG/JPEG RGB/L and small datasets. No multi-input/multi-loss,
  regression, distributed training or automatic Builder/Graph epoch resume is added.
- Existing GradScaler deprecation warnings remain. Metadata minimum dependency
  versions were not exhaustively matrix-tested; observed runtime versions are
  recorded above. No CUDA/native performance conclusions are made.

## 15. Acceptance decision

**PASS — generated-file Python-first integration, within the declared scope.**
Both final-source model paths actually read separate image files, preprocess and
batch them, train using the existing session, save/restore supported checkpoints,
run real test inference, compute native metrics, write native reports and link
verified experiment artifacts. Fresh installed wheel/sdist also execute that
chain. Dataset negatives, Core regressions, package checks and protection hashes
pass. Success is not inferred merely from a green test count.

Real-world dataset/model-quality acceptance and Graph native checkpoint receipt
coverage remain unverified/limited as stated above. Future work should first
validate a caller-authorized real dataset with independent splits and provenance,
then separately design/test Graph receipt support without weakening existing
evidence policy. No such additional work or 13.5.8 work was started.
