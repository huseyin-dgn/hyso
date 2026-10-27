"""R06/R07/R08: real ImageFolder, synthetic PNGs, transforms and split links."""

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms

from hyso.augmentation import get_loaders
from regression_guards import (
    R06RandomValidation, R06ValidationVariation, R07SharedTransform,
    R07EmptyValidation, R08UnknownClass, R08SubsetMapping, expect_value,
)


def image_tree(root, classes=("a", "b"), per_class=5):
    yy, xx = np.mgrid[:24, :24]
    pixels = np.stack([(xx * 9) % 256, (yy * 7) % 256, (xx * yy) % 256], axis=-1).astype(np.uint8)
    for name in classes:
        folder = root / name
        folder.mkdir(parents=True)
        for index in range(per_class):
            Image.fromarray(np.roll(pixels, index, axis=0)).save(folder / f"{index}.png")
    return str(root)


def random_ops(dataset):
    return [op for op in dataset.transform.transforms
            if type(op).__name__.startswith("Random") or isinstance(op, transforms.ColorJitter)]


@pytest.fixture
def separate_loaders(tmp_path):
    train = image_tree(tmp_path / "train")
    val = image_tree(tmp_path / "val")
    return get_loaders(train, val, batch_size=2, img_size=16)


def test_r06_current_validation_chain_and_repeat_variation(separate_loaders):
    train, val = separate_loaders
    assert {type(op).__name__ for op in random_ops(train.dataset)} == {
        "RandomHorizontalFlip", "RandomRotation", "ColorJitter",
    }
    readings = [val.dataset[0][0] for _ in range(8)]
    assert all(x.shape == (3, 16, 16) for x in readings)
    assert random_ops(val.dataset) == []
    assert all(torch.equal(readings[0], x) for x in readings[1:])


def test_r06_validation_chain_has_no_random_augmentation(separate_loaders):
    _, val = separate_loaders
    assert len(val.dataset) == 10
    expect_value([type(op).__name__ for op in random_ops(val.dataset)], [],
                 R06RandomValidation, "R06: random validation operators")


def test_r06_validation_repeated_reads_are_identical(separate_loaders):
    _, val = separate_loaders
    readings = [val.dataset[0][0] for _ in range(8)]
    assert all(x.shape == (3, 16, 16) and x.dtype == torch.float32 and torch.isfinite(x).all()
               for x in readings)
    expect_value(all(torch.equal(readings[0], x) for x in readings[1:]), True,
                 R06ValidationVariation, "R06: repeated validation tensor equality")


def test_r07_split_indices_disjoint_complete_and_seed_repeatable(tmp_path):
    path = image_tree(tmp_path / "train")
    torch.random.default_generator.manual_seed(97)
    train, val = get_loaders(path, batch_size=2, img_size=16, augment="advanced")
    first = (list(train.dataset.indices), list(val.dataset.indices))
    torch.random.default_generator.manual_seed(97)
    train2, val2 = get_loaders(path, batch_size=2, img_size=16, augment="advanced")
    assert first == (list(train2.dataset.indices), list(val2.dataset.indices))
    a, b = map(set, first)
    assert a.isdisjoint(b)
    assert a | b == set(range(10))
    assert len(a) == 8 and len(b) == 2
    assert train.dataset.dataset is not val.dataset.dataset
    assert train.dataset.dataset.samples == val.dataset.dataset.samples
    assert any(type(op).__name__ == "RandomResizedCrop"
               for op in train.dataset.dataset.transform.transforms)
    assert random_ops(val.dataset.dataset) == []


def test_r07_split_validation_has_independent_transform(tmp_path):
    path = image_tree(tmp_path / "train")
    train, val = get_loaders(path, batch_size=2, img_size=16, augment="advanced")
    assert len(train.dataset) == 8 and len(val.dataset) == 2
    assert set(train.dataset.indices).isdisjoint(val.dataset.indices)
    expect_value(train.dataset.dataset.transform is val.dataset.dataset.transform, False,
                 R07SharedTransform, "R07: validation owns a separate transform")
    expect_value([type(op).__name__ for op in random_ops(val.dataset.dataset)], [],
                 R07SharedTransform, "R07: validation transform must be deterministic")


def test_r08_current_class_mapping_mismatch(tmp_path):
    train_path = image_tree(tmp_path / "train", classes=("a", "b"))
    val_path = image_tree(tmp_path / "val", classes=("b", "c"))
    with pytest.raises(ValueError, match="Unknown validation classes.*c"):
        get_loaders(train_path, val_path, img_size=16)


def test_r08_unknown_validation_class_is_rejected(tmp_path):
    train_path = image_tree(tmp_path / "train", classes=("a", "b"))
    val_path = image_tree(tmp_path / "val", classes=("b", "c"))
    rejected = False
    try:
        get_loaders(train_path, val_path, img_size=16)
    except ValueError as error:
        require_class_mapping_error(error)
        rejected = True
    expect_value(rejected, True, R08UnknownClass, "R08: unknown class c must be rejected")


def require_class_mapping_error(error):
    message = str(error).lower()
    assert "class" in message
    assert any(word in message for word in ("validation", "mapping", "mismatch", "unknown", "subset"))


def test_r08_known_validation_subset_uses_training_labels_or_is_rejected(tmp_path):
    train_path = image_tree(tmp_path / "train", classes=("a", "b"))
    val_path = image_tree(tmp_path / "val", classes=("b",))
    try:
        train, val = get_loaders(train_path, val_path, img_size=16)
    except ValueError as error:
        require_class_mapping_error(error)
        return
    assert train.dataset.class_to_idx == {"a": 0, "b": 1}
    assert set(val.dataset.class_to_idx) <= set(train.dataset.class_to_idx)
    assert len(val.dataset) == 5
    expected = train.dataset.class_to_idx["b"]
    mapping = {name: train.dataset.class_to_idx[name] for name in val.dataset.class_to_idx}
    targets = [val.dataset[index][1] for index in range(len(val.dataset))]
    assert all(isinstance(target, int) for target in targets)
    expect_value(val.dataset.class_to_idx, mapping, R08SubsetMapping, "R08: subset class_to_idx")
    expect_value(targets, [expected] * 5, R08SubsetMapping, "R08: actual subset target labels")


@pytest.mark.parametrize("count", [1, 4])
def test_r07_current_tiny_split_has_empty_validation(tmp_path, count):
    path = image_tree(tmp_path / "train", classes=("a",), per_class=count)
    if count == 1:
        with pytest.raises(ValueError, match="Validation split.*two samples"):
            get_loaders(path, batch_size=2, img_size=16)
        return
    train, val = get_loaders(path, batch_size=2, img_size=16)
    assert len(train.dataset) == count - 1
    assert len(val.dataset) == len(val) == 1


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_r07_tiny_split_never_silently_returns_empty_validation(tmp_path, count):
    path = image_tree(tmp_path / "train", classes=("a",), per_class=count)
    try:
        train, val = get_loaders(path, batch_size=2, img_size=16)
    except ValueError as error:
        assert any(word in str(error).lower() for word in ("split", "sample", "validation"))
        return
    assert len(train.dataset) > 0
    assert set(train.dataset.indices).isdisjoint(val.dataset.indices)
    assert set(train.dataset.indices) | set(val.dataset.indices) == set(range(count))
    expect_value(len(val.dataset) > 0, True, R07EmptyValidation, "R07: nonempty validation or early red")


def test_r08_subset_sample_metadata_and_loader_targets_match_training(tmp_path):
    train_path = image_tree(tmp_path / "train")
    val_path = image_tree(tmp_path / "val", classes=("b",))
    train, val = get_loaders(train_path, val_path, img_size=16, batch_size=2)
    assert val.dataset.classes == train.dataset.classes == ["a", "b"]
    assert val.dataset.class_to_idx == train.dataset.class_to_idx == {"a": 0, "b": 1}
    assert val.dataset.imgs is val.dataset.samples
    assert [target for _, target in val.dataset.samples] == val.dataset.targets == [1] * 5
    assert torch.cat([targets for _, targets in val]).tolist() == [1] * 5


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("augment", ["basic", "advanced"])
@pytest.mark.parametrize("split", [False, True])
def test_r06_r07_validation_normalization_and_training_ops_are_preserved(
    tmp_path, channels, augment, split,
):
    train_path = image_tree(tmp_path / "train")
    val_path = None if split else image_tree(tmp_path / "val")
    extra = transforms.Lambda(lambda tensor: tensor + 1)
    train, val = get_loaders(train_path, val_path, batch_size=2, img_size=16,
                            channels=channels, augment=augment, extra_ops=[extra])
    train_data = train.dataset.dataset if split else train.dataset
    val_data = val.dataset.dataset if split else val.dataset
    assert extra in train_data.transform.transforms
    assert extra not in val_data.transform.transforms
    training_ops = {type(op).__name__ for op in train_data.transform.transforms}
    assert {"RandomHorizontalFlip", "RandomRotation", "ColorJitter"} <= training_ops
    if augment == "advanced":
        assert {"RandomVerticalFlip", "RandomResizedCrop", "GaussianBlur",
                "RandomPerspective", "RandomErasing"} <= training_ops
    assert isinstance(train.sampler, torch.utils.data.RandomSampler)
    assert isinstance(val.sampler, torch.utils.data.SequentialSampler)
    sample_index = val.dataset.indices[0] if split else 0
    with Image.open(val_data.samples[sample_index][0]) as image:
        image = image.convert("RGB")
        if channels == 1:
            image = transforms.functional.to_grayscale(image, num_output_channels=1)
        pixels = transforms.functional.to_tensor(transforms.functional.resize(image, [16, 16]))
    mean = torch.tensor([0.5] if channels == 1 else [0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.5] if channels == 1 else [0.229, 0.224, 0.225])[:, None, None]
    actual = val.dataset[0][0]
    assert actual.shape == (channels, 16, 16)
    torch.testing.assert_close(actual, (pixels - mean) / std, rtol=0, atol=0)
    assert torch.equal(actual, val.dataset[0][0])
    normalization = train_data.transform.transforms[-1]
    assert torch.equal(torch.tensor(normalization.mean), mean.flatten())
    assert torch.equal(torch.tensor(normalization.std), std.flatten())
