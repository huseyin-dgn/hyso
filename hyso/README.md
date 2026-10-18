# Hyso

Hyso is a PyTorch-based deep learning library designed for **CNN + Bottleneck Residual + CBAM** architectures. It simplifies building, training, and evaluating image classification models with integrated callbacks and data augmentation.

## Project Structure

````bash
hyso/
│
├─ hyso/
│ ├─ models/
│ │ ├─ bcam_model.py # CNNWithBottleneckResidualCBAM
│ │ └─ bcam_fit.py # BCAMWithFit (fit + callbacks)
│ │
│ ├─ augmentation/
│ │ └─ get_loaders.py # Data loader and augmentations
│ │
│ ├─ callbacks/
│ │ └─ callbacks.py # EarlyStopping, ModelCheckpoint, LRSchedulerCallback
│ │
│ └─ init.py
│
├─ setup.py # Installation script
└─ README.md # Project documentation


---

## Features

- CNN with **Bottleneck Residual + CBAM** blocks
- `fit()` and `callbacks()` integration for simplified training
- EarlyStopping, ModelCheckpoint, and LR Scheduler support
- Flexible data loading and augmentation via `get_loaders()`
- EMA support for smoother training (optional)

---

## Installation

### From PyPI (future)
```bash
pip install hyso

````

## From GitHub

git clone https://github.com/huseyin-dgn/hyso.git
cd hyso
pip install .

## Usage

```python
from hyso.modelsimport EBO
from hyso.augmentation import get_loaders

# ---------------- Dataset ----------------
train_loader, val_loader = get_loaders(
    train_path="path/to/train",
    val_path="path/to/val",
    batch_size=64,
    img_size=(32,32),
    augment="advanced",
    channels=3
)

# ---------------- Model ----------------
model = EBO(input_channels=3, num_classes=10)

# ---------------- Training ----------------
model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    scheduler="StepLR",
    scheduler_params={"step_size":5,"gamma":0.5},
    early_stopping=5,
    checkpoint_path="best_bcam_model.pth",
    epochs=20
)
```

## Callbacks

- Hyso supports the following callbacks to improve training:

- EarlyStopping: Stops training if validation loss stops improving.

- ModelCheckpoint: Saves the best model weights.

- LRSchedulerCallback: Automatically adjusts the learning rate based on the chosen scheduler.

Example integration:

```python
model.callbacks(
    optimizer="Adam",
    scheduler="StepLR",
    scheduler_params={"step_size":5,"gamma":0.5},
    model_checkpoint=True,
    earlystop={"patience":5, "verbose":True}
)
```

## Model Summary

To view a Keras-style summary of your model:

```python
from torchsummary import summary
summary(model.model, input_size=(3, 32, 32))
```

## Contributing

Pull requests, bug reports, and feature requests are welcome!

## License

No license is set. Free to use.
