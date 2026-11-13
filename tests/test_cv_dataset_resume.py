"""Real PNG epoch replay, bounded dataset identity and pre-C plan compatibility."""

import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms as T

from hyso.augmentation.augmentation import build_transforms, get_loaders
from hyso.checkpoint import TrainingCheckpointError
from hyso.models import bcam
from test_ebo_training_resume import presentation, seed, options, interrupted, loader
from test_training_checkpoint import assert_state
from training_helpers import tiny_ebo


@pytest.fixture
def images(tmp_path):
    root = tmp_path / "images"
    y, x = np.mgrid[:18, :20]
    for label, name in enumerate(("alpha", "beta")):
        (root / name).mkdir(parents=True)
        for index in range(4):
            pixels = np.stack([(x * 11 + index * 31) % 256,
                               (y * 13 + label * 91) % 256,
                               ((x + y) * 7 + index * 19) % 256], axis=-1).astype(np.uint8)
            Image.fromarray(pixels).save(root / name / (str(index) + ".png"))
    return root


def transform(random=False):
    if random:
        return build_transforms(img_size=(16, 16), augment="basic", extra_ops=[
            T.RandomVerticalFlip(), T.RandomResizedCrop(16, scale=(0.8, 1.0)),
            T.Grayscale(num_output_channels=3)])
    return T.Compose([T.Resize((18, 18)), T.CenterCrop(16), T.ToTensor(),
                      T.Normalize([0.5] * 3, [0.5] * 3)])


def image_loader(root, *, subset=False, random=False, shuffle=True, **kwargs):
    dataset = datasets.ImageFolder(root, transform=transform(random))
    if subset:
        dataset = Subset(dataset, [7, 1, 5, 3, 0, 6, 2, 4])
    return DataLoader(dataset, batch_size=3, shuffle=shuffle, **kwargs)


@pytest.mark.parametrize("scheduler", ["steplr", "onecycle"])
@pytest.mark.parametrize("subset", [False, True])
@pytest.mark.parametrize("random", [False, True])
def test_png_five_epochs_equal_three_plus_resume(images, tmp_path, monkeypatch, presentation,
                                                scheduler, subset, random):
    opts = options(optimizer="adam" if subset else "adamw", scheduler=scheduler, ema=True)
    full_path, cut_path, resumed_path = [tmp_path / name for name in ("full.pth", "cut.pth", "resumed.pth")]
    reads = []
    real_open = Image.open

    def record_read(source, *args, **kwargs):
        reads.append(Path(getattr(source, "name", source)).relative_to(images).as_posix())
        return real_open(source, *args, **kwargs)

    monkeypatch.setattr(Image, "open", record_read)

    def train():
        return image_loader(images, subset=subset, random=random, shuffle=not subset)

    def validation():
        return image_loader(images, subset=subset, shuffle=False)

    seed(19)
    full = tiny_ebo()
    history = full.fit(train(), val_loader=validation(), save_last_to=full_path, **opts)
    expected = torch.load(full_path, map_location="cpu", weights_only=True)
    expected_reads = list(reads)
    reads.clear()
    seed(19)
    interrupted(monkeypatch, tiny_ebo(), train(), cut_path, opts, validation=validation())
    cut = torch.load(cut_path, map_location="cpu", weights_only=True)
    assert cut["completed_epochs"] == 3
    if scheduler == "onecycle":
        assert cut["scheduler"]["state_dict"]["total_steps"] == 10
        assert cut["scheduler"]["state_dict"]["last_epoch"] == 6
    seed(999)
    resumed = tiny_ebo()
    presentation.clear()
    result = resumed.fit(train(), val_loader=validation(), resume_from=cut_path,
                         save_last_to=resumed_path, **opts)
    assert presentation == []
    actual = torch.load(resumed_path, map_location="cpu", weights_only=True)
    assert reads == expected_reads  # Actual image access order, including sampler/shuffle and validation.
    assert_state(actual, expected)  # Includes RNG, history, optimizer, scaler, scheduler and EMA.
    assert_state(resumed.state_dict(), full.state_dict())  # Parameters AND BatchNorm buffers.
    assert result == history and all(len(values) == 5 for values in result.values())
    assert actual["completed_epochs"] == 5 and actual["scaler"]["state_dict"] == {}
    assert all(state["step"].item() == 10 for state in resumed.optimizer.state.values())
    assert resumed.lr_scheduler.scheduler.last_epoch == (10 if scheduler == "onecycle" else 5)
    assert_state(resumed.ema.ema_model.state_dict(), full.ema.ema_model.state_dict())
    assert not torch.cuda.is_initialized()


@pytest.fixture
def cv_checkpoint(images, tmp_path, monkeypatch, presentation):
    path = tmp_path / "cut.pth"
    seed(41)
    interrupted(monkeypatch, tiny_ebo(), image_loader(images, subset=True), path, options(), stop=1)
    return path


class StatefulTransform:
    def __init__(self):
        self.calls = 0

    def __call__(self, value):
        self.calls += 1
        return value


@pytest.mark.parametrize("change", [
    "added_image", "removed_image", "sample_order", "label", "class_to_idx", "classes",
    "subset_order", "subset_base", "size", "mtime", "batch_size", "sampler", "workers",
    "custom_transform", "stateful_transform", "transform_config", "transform_state", "transform_hook",
    "target_transform", "custom_loader", "missing_root", "moved_root", "corrupt", "legacy",
])
def test_incompatible_cv_resume_rejected_before_model_rng_mutation(
        cv_checkpoint, images, tmp_path, presentation, change):
    train = image_loader(images, subset=True)
    base = train.dataset.dataset
    path = cv_checkpoint
    if change == "added_image":
        shutil.copyfile(images / "alpha/0.png", images / "alpha/new.png")
    elif change == "removed_image":
        (images / "alpha/0.png").unlink()
    elif change == "sample_order":
        base.samples.reverse()
        base.targets.reverse()
    elif change == "label":
        sample, label = base.samples[0]
        base.samples[0] = (sample, 1 - label)
        base.targets[0] = 1 - label
    elif change == "class_to_idx":
        base.class_to_idx = {"alpha": 1, "beta": 0}
    elif change == "classes":
        base.classes.reverse()
    elif change == "subset_order":
        train.dataset.indices.reverse()
    elif change == "subset_base":
        alternate = tmp_path / "alternate"
        shutil.copytree(images, alternate)
        train.dataset.dataset = datasets.ImageFolder(alternate, transform=transform())
    elif change == "size":
        with (images / "alpha/0.png").open("ab") as output:
            output.write(b"extra")
    elif change == "mtime":
        sample = images / "alpha/0.png"
        stat = sample.stat()
        os.utime(sample, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    elif change == "batch_size":
        train = DataLoader(train.dataset, batch_size=4, shuffle=True)
    elif change == "sampler":
        train = DataLoader(train.dataset, batch_size=3, sampler=WeightedRandomSampler([1.] * 8, 8))
    elif change == "workers":
        train = image_loader(images, subset=True, num_workers=1)
    elif change == "custom_transform":
        base.transform = T.Compose([T.Lambda(lambda value: value), T.ToTensor()])
    elif change == "stateful_transform":
        base.transform = T.Compose([StatefulTransform(), T.ToTensor()])
    elif change == "transform_config":
        base.transform.transforms[-1].mean = [0.4] * 3
    elif change == "transform_state":
        base.transform.transforms[0].calls = 1
    elif change == "transform_hook":
        base.transform.transforms[0].register_forward_hook(lambda *args: None)
    elif change == "target_transform":
        base.target_transform = lambda target: target
    elif change == "custom_loader":
        base.loader = lambda path: Image.open(path).convert("RGB")
    elif change == "missing_root":
        base.root = str(tmp_path / "missing")
    elif change == "moved_root":
        moved = tmp_path / "moved"
        images.rename(moved)
        train = image_loader(moved, subset=True)
    elif change == "corrupt":
        path.write_bytes(b"not a checkpoint")
    elif change == "legacy":
        torch.save(tiny_ebo().state_dict(), path)
    seed(999)
    model = tiny_ebo()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(TrainingCheckpointError):
        model.fit(train, resume_from=path, **options())
    assert_state(model.state_dict(), state)
    assert torch.equal(torch.get_rng_state(), rng)


def test_fingerprint_does_not_decode_or_consume_rng(images, monkeypatch):
    train = image_loader(images, subset=True, random=True)
    rng = torch.get_rng_state().clone()
    def forbidden(*args, **kwargs):
        raise AssertionError("fingerprint must not decode")
    monkeypatch.setattr(Image, "open", forbidden)
    first = bcam._resume_loader_plan(train)
    assert first == bcam._resume_loader_plan(train)
    assert torch.equal(torch.get_rng_state(), rng)
    base = first["dataset"]["base"]["dataset"]
    assert base["classes"] == ["alpha", "beta"] and base["samples"] == 8


def test_manifest_metadata_is_not_a_content_integrity_hash(images):
    train = image_loader(images)
    before = bcam._resume_loader_plan(train)
    path = images / "alpha/0.png"
    stat = path.stat()
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert bcam._resume_loader_plan(train) == before


def test_get_loaders_basic_random_split_and_remapped_validation(images, tmp_path):
    train, validation = get_loaders(str(images), batch_size=3, img_size=16, num_workers=0)
    for item in (train, validation):
        assert "dataset" in bcam._resume_loader_plan(item)
    val_root = tmp_path / "validation"
    (val_root / "beta").mkdir(parents=True)
    shutil.copyfile(images / "beta/0.png", val_root / "beta/0.png")
    train, validation = get_loaders(str(images), val_path=str(val_root), batch_size=3, img_size=16)
    plan = bcam._resume_loader_plan(validation)["dataset"]
    assert plan["class_to_idx"] == {"alpha": 0, "beta": 1}
    assert validation.dataset.targets == [1]


@pytest.mark.parametrize("indices", [[8], [-1], [1.5]])
def test_invalid_subset_indices_rejected(images, indices):
    data = Subset(datasets.ImageFolder(images, transform=transform()), indices)
    with pytest.raises(TrainingCheckpointError, match="Subset indices"):
        bcam._resume_loader_plan(DataLoader(data, batch_size=1))


def pre_c_tensor_plan(train):
    # Frozen P0.4-B schema, deliberately independent of the production helper.
    tensors = []
    for tensor in train.dataset.tensors:
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        tensors.append({"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                        "sha256": hashlib.sha256(memoryview(raw)).hexdigest()})
    return {"batches": len(train), "batch_size": train.batch_size, "drop_last": train.drop_last,
            "sampler": "torch.utils.data.sampler.RandomSampler", "sampler_samples": len(train.sampler),
            "tensors": tensors}


def test_pre_c_tensor_checkpoint_and_subset_identity(tmp_path, monkeypatch, presentation):
    path = tmp_path / "old.pth"
    seed(41)
    interrupted(monkeypatch, tiny_ebo(), loader(), path, options(), stop=3)
    state = torch.load(path, weights_only=True)
    old_plan = pre_c_tensor_plan(loader())
    assert state["metrics"]["ebo_fit"]["train_loader"] == old_plan
    state["metrics"]["ebo_fit"]["train_loader"] = old_plan
    torch.save(state, path)
    model = tiny_ebo()
    history = model.fit(loader(), resume_from=path, **options())
    assert len(history["train_loss"]) == 5
    assert state["format_version"] == 1
    base = loader().dataset
    first = bcam._resume_loader_plan(DataLoader(Subset(base, [1, 0]), batch_size=2))
    second = bcam._resume_loader_plan(DataLoader(Subset(base, [0, 1]), batch_size=2))
    assert first != second
    assert first["dataset"]["base"] == {"tensors": old_plan["tensors"]}
