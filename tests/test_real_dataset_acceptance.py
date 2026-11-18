"""Offline real-source tests, separate from synthetic dataset regression tests."""

import hashlib
import shutil
import socket

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import SequentialSampler

from hyso import TrainingConfig
from hyso.datasets import ImageDatasetError, load_image_splits
from examples.models.real_digits_acceptance import prepare, png_bytes


@pytest.fixture
def offline(monkeypatch):
    pytest.importorskip("sklearn")
    def forbidden(*args, **kwargs):
        raise AssertionError("Real-source acceptance attempted network access")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def test_source_conversion_and_disjoint_balanced_splits(tmp_path, offline):
    from sklearn.datasets import load_digits
    source = load_digits()
    provenance = prepare(tmp_path / "digits")
    assert provenance["real_world_source"] is True
    assert provenance["original_uci_partition"] == "test"
    assert len(provenance["samples"]) == 200
    rows = provenance["samples"]
    assert len({r["source_index"] for r in rows}) == 200
    for field in ("png_sha256", "tensor_sha256", "original_pixels_sha256"):
        assert len({r[field] for r in rows}) == 200
    for split, count in (("train", 12), ("val", 4), ("test", 4)):
        assert [sum(r["split"] == split and r["label"] == k for r in rows)
                for k in range(10)] == [count] * 10
    for row in rows:
        path = tmp_path / "digits" / row["path"]
        blob = path.read_bytes()
        assert blob == png_bytes(source.images[row["source_index"]])
        assert hashlib.sha256(blob).hexdigest() == row["png_sha256"]
        assert source.target[row["source_index"]] == row["label"]
        with Image.open(path) as image:
            assert image.mode == "L" and image.size == (8, 8)
            actual = np.asarray(image)
        original = source.images[row["source_index"]].astype(np.uint16)
        assert np.array_equal(actual, ((original * 255 + 8) // 16).astype(np.uint8))
    data = load_image_splits(tmp_path / "digits", channels=1)
    assert data.to_dict()["dataset_id"] == provenance["dataset_id"]
    # All 17 possible source levels survive the integer PNG mapping injectively.
    levels = (np.arange(17, dtype=np.uint16) * 255 + 8) // 16
    assert len(set(levels)) == 17


def test_reproducible_seeded_loaders_and_input_preservation(tmp_path, offline):
    first = prepare(tmp_path / "one")
    second = prepare(tmp_path / "two")
    assert first == second
    data = load_image_splits(tmp_path / "one", channels=1)
    config = TrainingConfig(epochs=2, batch_size=16, validation=True, shuffle=True)
    a, val, test = data.make_loaders(config, seed=1358)
    torch.rand(50)
    b, _, _ = data.make_loaders(config, seed=1358)
    assert list(a.sampler) == list(b.sampler)
    assert isinstance(val.sampler, SequentialSampler)
    assert isinstance(test.sampler, SequentialSampler)
    x, y = next(iter(a))
    assert x.shape == (16, 1, 8, 8) and x.dtype == torch.float32
    assert y.dtype == torch.int64
    for row in first["samples"]:
        assert hashlib.sha256((tmp_path / "one" / row["path"]).read_bytes()).hexdigest() == row["png_sha256"]
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "one")


def test_real_source_duplicate_is_rejected(tmp_path, offline):
    provenance = prepare(tmp_path / "digits")
    row = next(r for r in provenance["samples"] if r["split"] == "train")
    shutil.copyfile(tmp_path / "digits" / row["path"],
                    tmp_path / "digits" / "test" / str(row["label"]) / "duplicate.png")
    with pytest.raises(ImageDatasetError, match="Split leakage"):
        load_image_splits(tmp_path / "digits", channels=1)


def test_real_source_changed_file_is_rejected(tmp_path, offline):
    prepare(tmp_path / "digits")
    data = load_image_splits(tmp_path / "digits", channels=1)
    path = data.train.samples[0][0]
    with open(path, "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ImageDatasetError, match="changed"):
        data.train[0]


def test_default_sample_budget_is_retained_for_real_data(tmp_path, offline):
    provenance = prepare(tmp_path / "digits")
    row = next(r for r in provenance["samples"] if r["split"] == "train")
    source = tmp_path / "digits" / row["path"]
    for index in range(9):
        shutil.copyfile(source, source.parent / f"budget-{index}.png")
    with pytest.raises(ImageDatasetError, match="sample budget 128"):
        load_image_splits(tmp_path / "digits", channels=1)
