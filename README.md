# HYSo

HYSo is a published PyTorch library for image classification with residual CNNs,
CBAM/SE attention, callbacks and augmentation.

This README describes the **0.2.0 release candidate preparation** on the development
branch. Local package metadata is 0.2.0; this version has not been published to
PyPI, tagged or released. The published 0.1.6 package does not include the full-state
checkpoint/resume features below. See the
[release notes draft](docs/release-notes/0.2.0-draft.md).

## Installation

Install the published package from PyPI:

```bash
pip install hyso
```

Install the checked-out source to use this branch's features:

```bash
git clone https://github.com/huseyin-dgn/hyso.git
cd hyso
pip install .
```

Select the intended branch before installing. Use a matching Torch/torchvision
pair for your platform and CPU/CUDA runtime; HYSo does not install a GPU driver.

## Package structure

```text
hyso/
├── pyproject.toml
├── setup.cfg
├── MANIFEST.in
├── LICENSE
├── README.md
├── hyso/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bcam.py           # EBO training wrapper and ELO CBAM network
│   │   ├── WithSE.py         # CEMO and EMA
│   │   └── fit.py            # CNNSENormalWithFit
│   ├── augmentation/
│   │   ├── __init__.py
│   │   └── augmentation.py
│   └── callbacks/
│       ├── __init__.py
│       └── callbacks.py
├── tests/
└── docs/
```

Public model exports from `hyso.models` are `EBO`, `ELO`, `CEMO`, `EMA` and
`CNNSENormalWithFit`. ELO is the CBAM network; the training example uses EBO.

## Image classification and model-only best checkpoint

Provide your own ImageFolder directories, for example `data/train/cat/*.png`,
`data/train/dog/*.png` and matching `data/val/` classes.
This small CPU example does not require CUDA.

```python
from hyso.models import EBO, ELO, CEMO, EMA, CNNSENormalWithFit
from hyso.augmentation import get_loaders

train_loader, val_loader = get_loaders(
    train_path="data/train",
    val_path="data/val",
    batch_size=3,
    img_size=(16, 16),
    augment="basic",
    channels=3,
    num_workers=0,
)
model_config = dict(
    input_channels=3,
    num_classes=len(train_loader.dataset.classes),
    conv_channels=[4, 8],
    residual_per_block=1,
    device="cpu",
)
model = EBO(**model_config)
model.callbacks(
    optimizer="AdamW",
    lr=0.001,
    scheduler="StepLR",
    scheduler_params={"step_size": 1, "gamma": 0.5},
    model_checkpoint=True,
    earlystop={"patience": 5, "verbose": True},
)
history = model.fit(
    train_loader,
    val_loader=val_loader,
    epochs=2,
    use_amp=False,
    use_ema=True,
    accum_steps=2,
    grad_clip=0.5,
)
```

`callbacks()` configures the optimizer, StepLR, EarlyStopping and ModelCheckpoint.
`fit()` reuses that optimizer/scheduler. Validation is required for these best-loss
callbacks. Do not pass `scheduler`, `early_stopping` or `checkpoint_path` to `fit()`.
When configuring a scheduler directly in `fit()`, use `scheduler_name="steplr"`
and `scheduler_params=...` instead of the callback spelling.

`model_checkpoint=True` writes **`best_bcam_model.pth`**, containing the inner
network's raw `state_dict()`. Load it into a matching `model.model` with
`torch.load(..., map_location="cpu", weights_only=True)` and `load_state_dict`.
It contains no optimizer/scaler/history and cannot be used with `resume_from`.
With EMA validation enabled, best loss is measured using EMA; the legacy callback
still saves the live inner network, not the EMA shadow.

## Full-state checkpoint and EBO resume (development branch)

Use a **fresh EBO without the stateful callbacks above**. These examples reuse
`train_loader`, `val_loader` and `model_config` from the first block.
`save_last_to` atomically replaces a full-state checkpoint after each completed
epoch. The parent directory must already exist.

```python
training_options = dict(
    epochs=5,
    lr=0.001,
    optimizer_name="adamw",
    use_amp=False,
    use_ema=True,
    accum_steps=2,
    grad_clip=0.5,
    scheduler_name="onecycle",
    scheduler_params={"cycle_momentum": False},
    max_lr=0.01,
)
training_model = EBO(**model_config)
training_history = training_model.fit(
    train_loader,
    val_loader=val_loader,
    save_last_to="last_training.pth",
    **training_options,
)
```

`epochs=5` is the **total target**, including already completed epochs. If the
process stops after epoch 3 has been saved, the file has `completed_epochs=3`.
The original call must already have planned five epochs. Resume in a new process
after recreating the same loaders, model configuration and `training_options`:

```python
import torch

saved = torch.load("last_training.pth", map_location="cpu", weights_only=True)
print("Completed epochs:", saved["completed_epochs"])
resumed_model = EBO(**model_config)
resumed_history = resumed_model.fit(
    train_loader,
    val_loader=val_loader,
    resume_from="last_training.pth",
    save_last_to="last_training.pth",
    **training_options,
)
```

A checkpoint with three completed epochs continues at epoch 4 and finishes at 5;
history is preserved and extended. A checkpoint already at 5 performs no more
training. Do not change a three-epoch OneCycle plan into a five-epoch plan:
the scheduler's original total step budget must match. Overflow may skip optimizer
updates; OneCycle advances only for applied updates.

EBO restores model, optimizer moments, scheduler counters, real AMP scaler state,
EMA shadow, history and Python/NumPy/Torch/CUDA RNG as applicable. Components are
created before state restoration; RNG is restored before the next data iterator.
Model/config, optimizer/scheduler settings, AMP/EMA flags, accumulation, data and
loader plan must match. State mismatches are rejected rather than silently ignored.
EarlyStopping, legacy ModelCheckpoint and other stateful/custom callbacks are
**unsupported with `resume_from` or `save_last_to`**. Their best-loss/counter state
is not serialized. CEMO and CNNSENormalWithFit do not have this EBO resume API.

The independent `hyso.checkpoint.save_training_checkpoint` and
`load_training_checkpoint` functions operate on caller-created components.
Full-state files use `format="hyso.training"`, `format_version=1`; loading uses
`weights_only=True` without an unsafe fallback. They are separate from model-only files.

### Supported data and device profile

- Standard EBO/ELO, TensorDataset or PIL ImageFolder, including Subset with unchanged
  base dataset and ordered indices; CPU dataset tensors, `num_workers=0`.
- Standard DataLoader, default collate and BatchSampler; SequentialSampler or
  RandomSampler with `replacement=False`; no explicit loader/sampler generators.
- Supported torchvision v1 operations: Compose, Resize, CenterCrop, ToTensor,
  Normalize, Grayscale, RandomHorizontalFlip/VerticalFlip, RandomRotation,
  ColorJitter and RandomResizedCrop, with standard state and no custom hooks.
  HYSo `augment="basic"` works; the entire `advanced` chain is not supported for resume.
- ImageFolder identity checks classes/class_to_idx, sample order/path/label and a
  **path/size/mtime** manifest. It does not decode each image for fingerprinting or
  cryptographically verify image contents. Same-size edits preserving mtime can
  evade detection. Keep data immutable; moving its root or changing runtime details
  can invalidate the plan. Custom loading/target transforms and symlink samples or
  subdirectories are unsupported.
- CPU resume was verified with disabled AMP, EMA on/off, Adam/AdamW and
  StepLR/OneCycleLR. Single-GPU CUDA FP16 resume was verified with active GradScaler,
  EMA, accumulation, TensorDataset and ImageFolder/Subset, including random transforms.
  For CUDA, create the model with `device="cuda"` and use `use_amp=True` consistently
  from the original plan. Cross-device resume equivalence is not claimed.
- No mid-epoch, multi-worker, custom Dataset/transform/stateful callback, DDP or
  multi-GPU resume support. Random-transform equivalence requires the same RNG
  consumption order. CUDA AdaptiveMaxPool backward has no deterministic kernel;
  observed exact equality on the reference GPU is not a general determinism guarantee.

## Compatibility and verification

The 0.2.0 candidate metadata requires **Python>=3.10, Torch>=2.2.2 and
torchvision>=0.17.2**. These are support minimums; they do not certify every
combination above the lower bounds. Python 3.8/3.9 and Torch 1.x/2.0/2.1 users
must upgrade their runtime for 0.2.0 or retain a suitable older HYSo version.
Published 0.1.6 declares Python>=3.8 and Torch>=1.13.

Use a compatible Torch/torchvision pair, including the CPU/CUDA binary channel:
Torch 2.2.2 pairs with torchvision 0.17.2, and Torch 2.5.1 with torchvision 0.20.1.
The separate lower bounds do not make arbitrary Torch/torchvision pairs compatible.
For the older Torch 2.2.2 profile, the tested NumPy version is 1.26.4; use that
constraint to reproduce it. Older binary extensions can have NumPy 2 ABI issues.
HYSo does not impose a global NumPy<2 cap: the newer reference below uses NumPy 2.2.2.

| Level | Environment / evidence |
|---|---|
| Runtime verified, older CPU profile | Windows x64, Python 3.10.18, Torch 2.2.2+cpu, torchvision 0.17.2+cpu, NumPy 1.26.4; installed-wheel forward/checkpoint/resume |
| Runtime verified, reference CPU | Windows, Python 3.10.18, Torch 2.5.1+cu121, torchvision 0.20.1+cu121, NumPy 2.2.2, Pillow 12.0.0 |
| Previously verified, single GPU (P0.6-C1) | Reference runtime, RTX 3060 Laptop, capability 8.6, driver 591.74, CUDA build 12.1, FP16 AMP; final CUDA rerun planned for P0.6-E |

Python 3.8/Torch 2.0 runtime tests were not run; those versions are outside the
0.2.0 support policy. Python 3.10.18 evidence does not verify every Python 3.10 patch
or later minor release. P0.6-C verified a constrained offline clean dependency
installation for the older CPU profile; candidate wheel target installs reuse
those dependencies and are not new independent clean installs. Unconstrained online
installation, other OS/version/GPU combinations and BF16 have not been verified.

Full-state checkpoint format remains **v1**. OneCycleLR compatibility handles the
trusted legacy annealing callable representation in Torch 2.2.2 and the plain
strategy representation in Torch 2.5.1, without serializing callables or disabling
`weights_only=True`. Unknown scheduler schemas are rejected; compatibility with
every future Torch release is not claimed.

Tests: `python -B -m pytest -c tests/pytest.ini -q` runs the CPU suite (fixtures hide
CUDA). Run `tests/cuda_runtime_validation.py` and `tests/cuda_imagefolder_validation.py`
directly in separate CUDA-visible processes to check actual GPU behavior.
See the [candidate preparation report](docs/audit/p0.6-d-release-candidate.md) and
[support decision](docs/audit/p0.6-c2-minimum-support-decision.md) for evidence and limitations.

## Contributing and license

Issues and pull requests are welcome. HYSo is distributed under the
[MIT License](LICENSE), matching the package metadata.
