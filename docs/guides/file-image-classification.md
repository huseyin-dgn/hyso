# File-backed image classification on CPU

Run from the checkout with HYSo and its existing Torch, torchvision and Pillow
dependencies installed. Supply a local dataset you are allowed to use:

```text
my-images/
  train/
    class_a/  # PNG/JPEG files
    class_b/
  val/        # optional; same class folders
    class_a/
    class_b/
  test/       # required by the end-to-end example
    class_a/
    class_b/
```

Each present split must contain exactly the same nonempty class folders. Train
defines their sorted indices. There is no implicit split or mixing of train and
test. The loader accepts PNG/JPEG only, with actual source mode RGB for
`--channels 3` or L for `--channels 1`; prepare other image modes explicitly.

```powershell
python -m examples.models.image_classification --dataset C:/my-images --output C:/local-hyso/builder-01 --model builder --channels 3 --height 8 --width 8 --epochs 2 --batch-size 2 --seed 17
python -m examples.models.image_classification --dataset C:/my-images --output C:/local-hyso/graph-01 --model graph --channels 3 --height 8 --width 8 --epochs 2 --batch-size 2 --seed 17
```

Output must be a fresh directory disjoint from the dataset. Existing outputs are
rejected. For grayscale files, select `--channels 1`. Optional normalization is
explicit: e.g. `--mean 0.5 0.5 0.5 --std 0.5 0.5 0.5` for RGB. Both vectors must
match the channel count; standard deviations must be positive and the resulting
fp32 tensors finite. Default preprocessing is bilinear resize, ToTensor and an
explicit float32 cast, producing CHW samples / NCHW batches in [0,1] before
normalization. No training augmentation is applied.

The example limits execution to CPU fp32, one or two epochs, batch size 1–16,
zero loader workers and at most 128 images per split. The loading API exposes
`max_samples_per_split` for explicit use elsewhere. Each source file is limited
to 32 MiB and 16,777,216 decoded pixels; resized samples are limited to 1,048,576
elements. The complete small dataset is decoded during validation.

## Existing Python APIs

The example reuses ModelBuilder or GraphModelBuilder, TrainingConfig /
TrainingSession / Trainer, evaluate_classification, the generic checkpoint API,
ExperimentManager, EvidenceIndex and Reporting. `hyso.datasets` adds file/split
validation and delegates loading to torchvision ImageFolder and standard CPU
DataLoader objects:

```python
import torch
from hyso import ModelBuilder, TrainingConfig
from hyso.datasets import load_image_splits

data = load_image_splits("C:/my-images", image_size=(8, 8), channels=3)
builder = ModelBuilder((3, 8, 8), len(data.classes)).add(
    "conv_norm_act", in_channels=3, out_channels=4, norm=None)
data.validate_model(builder.to_config())
settings = TrainingConfig(epochs=1, batch_size=2, validation=data.val is not None,
                          shuffle=True, device="cpu", precision="fp32")
train, val, test = data.make_loaders(settings, seed=17)
model = builder.build().to(device="cpu", dtype=torch.float32)
training = settings.prepare(model, train, val).fit()
```

See the runnable example for checkpoint save/load, inference and reports. Its
graph has two branches merged with add, a classification head, and explicitly
selects `logits` through GraphTensorOutput. Model preflight retains the original
Core diagnostics, then checks input geometry and class width before allocation.

Independent loader generators make shuffle reproducible for a fixed seed and
dataset in the same runtime. Manifests include preprocessing/runtime versions,
class mapping, relative filenames, raw/decoded/processed hashes and dataset ID.
Cross-split checks reject equal file bytes, equal decoded images and equal
processed tensors. They cannot establish independent origin or detect all
semantic near-duplicates. Subsequent reads reject changed source bytes. Do not
modify source files or dataset/transform objects while a run is active.

## Results and evidence limits

Each run writes `dataset-manifest.json`, `training-result.json`, `model.pth`,
`evaluation.json`, `report.json`, `report.md`, `report.html`, `summary.json` and
its own `experiments/` store. HTML is an existing static report export. Open
`report.md` or `report.html` locally; inspect `summary.json` for checkpoint,
tracking and artifact verification results. This creates and explicitly migrates
only a new example store, preserving user workspaces.

The native session performs actual training with SGD, CrossEntropyLoss and
validation when present. Test data is evaluated only after training using actual
restored-model inference. Trainer accuracy uses percent; evaluation accuracy,
precision, recall and F1 use fractions. No hyperparameters are selected on test.
Model-only checkpoints are restored strictly and checked for identical outputs
on a fixed validation (or training) image batch. Automatic epoch resume for these
Builder/Graph models is outside the supported example scope.

ModelBuilder evaluation can carry a native checkpoint receipt. GraphModel's
nonpersistent contract buffer currently prevents the existing evidence system
from issuing that receipt: `producer_evidence.checkpoint.status` remains
`missing`, reason `no_matching_trusted_load_receipt`. The graph's strict load
and inference equality still pass. Saved file SHA and ExperimentManager links
record the caller's association, without promoting it to native checkpoint
evidence. Inspect that status before consuming checkpoint-bound evidence.

No dataset is downloaded or generated by this example. `--dataset-origin` is an
explicit caller declaration, default `caller_supplied`; select
`generated_fixture` when using generated test files. The 13.5.7 acceptance used
generated PNG/JPEG fixtures and establishes working file I/O, training and
inference, with no real-world model quality claim.
