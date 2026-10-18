"""Execute the documented code, not a separately maintained equivalent example."""

import ast
from functools import wraps
import inspect
from pathlib import Path
import re

import numpy as np
from PIL import Image
import pytest
import torch

from hyso.models import EBO, bcam
from test_ebo_training_resume import seed, EpochBoundaryStop
from test_training_checkpoint import assert_state


README = Path(__file__).resolve().parents[1] / "README.md"


def blocks():
    return re.findall(r"^```python\n(.*?)^```", README.read_text(encoding="utf-8"), re.M | re.S)


def test_readme_python_syntax():
    examples = blocks()
    assert len(examples) == 3
    for example in examples:
        ast.parse(example, feature_version=(3, 8))
        compile(example, str(README), "exec")


@pytest.fixture
def example(tmp_path, monkeypatch, quiet_training):
    monkeypatch.chdir(tmp_path)
    for split in ("train", "val"):
        for label, name in enumerate(("cat", "dog")):
            directory = tmp_path / "data" / split / name
            directory.mkdir(parents=True)
            y, x = np.mgrid[:32, :32]
            for index in range(4):
                data = np.stack([(x * 7 + index * 31) % 256, (y * 11 + label * 73) % 256,
                                 ((x + y) * 3 + index * 19) % 256], axis=-1).astype(np.uint8)
                Image.fromarray(data).save(directory / (str(index) + ".png"))
    calls = []
    # Bind the actual runtime arguments to the unchanged public signatures.
    def observe(name):
        method = getattr(EBO, name)
        signature = inspect.signature(method)

        @wraps(method)
        def checked(*args, **kwargs):
            signature.bind(*args, **kwargs)
            calls.append(name)
            return method(*args, **kwargs)

        monkeypatch.setattr(EBO, name, checked)

    observe("fit")
    observe("callbacks")
    namespace = {}
    exec(compile(blocks()[0], str(README), "exec"), namespace)
    return namespace, calls


def test_readme_callback_training_and_model_only_checkpoint(example):
    namespace, calls = example
    model = namespace["model"]
    assert calls == ["callbacks", "fit"]
    assert len(namespace["history"]["train_loss"]) == 2
    assert model.lr_scheduler.scheduler.last_epoch == 2
    assert all(np.isfinite(values).all() for values in namespace["history"].values())
    state = torch.load("best_bcam_model.pth", map_location="cpu", weights_only=True)
    assert "format" not in state and all(isinstance(value, torch.Tensor) for value in state.values())
    model.model.load_state_dict(state, strict=True)
    inputs, _ = next(iter(namespace["val_loader"]))
    with torch.no_grad():
        assert torch.isfinite(model.eval()(inputs)).all()
    assert not torch.cuda.is_initialized()


def test_readme_full_state_five_epochs_equals_three_plus_resume(example, monkeypatch):
    namespace, calls = example
    start, resume = blocks()[1:]
    seed(19)
    exec(compile(start, str(README), "exec"), namespace)
    full = torch.load("last_training.pth", map_location="cpu", weights_only=True)
    assert full["completed_epochs"] == 5
    seed(19)
    real_save = bcam.save_training_checkpoint

    def save(*args, **kwargs):
        real_save(*args, **kwargs)
        if kwargs["completed_epochs"] == 3:
            raise EpochBoundaryStop

    with monkeypatch.context() as patch:
        patch.setattr(bcam, "save_training_checkpoint", save)
        with pytest.raises(EpochBoundaryStop):
            exec(compile(start, str(README), "exec"), namespace)
    seed(999)
    exec(compile(resume, str(README), "exec"), namespace)
    assert namespace["saved"]["completed_epochs"] == 3
    assert namespace["saved"]["metrics"]["ebo_fit"]["epochs"] == 5
    assert namespace["saved"]["scheduler"]["state_dict"]["total_steps"] == 10
    actual = torch.load("last_training.pth", map_location="cpu", weights_only=True)
    assert_state(actual, full)  # Parameters, buffers, optimizer, scaler, EMA, history and RNG.
    assert actual["scheduler"]["state_dict"]["last_epoch"] == 10
    assert actual["scaler"]["state_dict"] == {}
    assert len(namespace["resumed_history"]["train_loss"]) == 5
    assert calls == ["callbacks", "fit", "fit", "fit", "fit"]
    assert not torch.cuda.is_initialized()
