"""File-backed classification contracts; all images are generated test fixtures."""

import copy
import hashlib
import importlib
import shutil

from PIL import Image
import pytest
import torch

from hyso import ModelBuilder, GraphModelBuilder, TrainingConfig
from hyso.model_checks import preflight_model


def make_images(root, channels=3):
    for split_index, (split, count) in enumerate(
        [("train", 4), ("val", 2), ("test", 2)]
    ):
        for label, name in enumerate(["alpha", "zebra"]):
            directory = root / split / name
            directory.mkdir(parents=True)
            for index in range(count):
                image = Image.new("RGB" if channels == 3 else "L", (12, 10))
                values = []
                for y in range(10):
                    for x in range(12):
                        v = (
                            17 * x + 9 * y + 41 * index + 67 * split_index + 103 * label
                        ) % 256
                        values.append(
                            (
                                v,
                                (v + 29 * label + 7 * y) % 256,
                                (v + 11 * x + split_index) % 256,
                            )
                            if channels == 3
                            else v
                        )
                image.putdata(values)
                image.save(
                    directory / (str(index) + (".png" if index % 2 == 0 else ".jpg"))
                )
    return root


def hashes(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def images(tmp_path):
    return make_images(tmp_path / "images")


def api():
    return importlib.import_module("hyso.datasets")


@pytest.mark.parametrize("channels", [1, 3])
def test_files_mapping_preprocessing_and_seeded_loaders(tmp_path, channels):
    root = make_images(tmp_path / "images", channels)
    before = hashes(root)
    data = api().load_image_splits(root, image_size=(8, 8), channels=channels)
    assert data.classes == ("alpha", "zebra")
    assert data.to_dict()["class_to_idx"] == {"alpha": 0, "zebra": 1}
    assert data.to_dict()["split_counts"] == {"train": 8, "val": 4, "test": 4}
    settings = TrainingConfig(epochs=1, batch_size=2, shuffle=True, validation=True)
    first, _, test = data.make_loaders(settings, seed=19)
    torch.rand(19)
    second, _, _ = data.make_loaders(settings, seed=19)
    a = list(first)
    b = list(second)
    assert all(torch.equal(x, u) and torch.equal(y, v) for (x, y), (u, v) in zip(a, b))
    assert a[0][0].shape == (2, channels, 8, 8) and a[0][0].dtype == torch.float32
    assert a[0][1].dtype == torch.long
    assert list(test.sampler) == list(range(4))
    assert hashes(root) == before


@pytest.mark.parametrize(
    "problem",
    [
        "empty",
        "empty_class",
        "corrupt",
        "unsupported",
        "channels",
        "classes",
        "leakage",
        "resized_leakage",
    ],
)
def test_invalid_file_datasets_are_diagnosed(images, problem):
    if problem == "empty":
        for p in images.rglob("*"):
            if p.is_file():
                p.unlink()
    elif problem == "empty_class":
        for p in (images / "val/alpha").iterdir():
            p.unlink()
    elif problem == "corrupt":
        (images / "train/alpha/0.png").write_bytes(b"not an image")
    elif problem == "unsupported":
        (images / "train/alpha/note.txt").write_text("unsupported")
    elif problem == "channels":
        Image.new("L", (12, 10), 123).save(images / "train/alpha/0.png")
    elif problem == "classes":
        (images / "test/zebra").rename(images / "test/unknown")
    elif problem == "leakage":
        shutil.copyfile(images / "train/alpha/0.png", images / "test/alpha/0.png")
    elif problem == "resized_leakage":
        Image.new("RGB", (12, 10), (21, 31, 41)).save(images / "train/alpha/0.png")
        Image.new("RGB", (24, 20), (21, 31, 41)).save(images / "test/alpha/0.png")
    with pytest.raises(api().ImageDatasetError):
        api().load_image_splits(images, image_size=(8, 8), channels=3)


def test_source_change_after_scan_is_rejected(images):
    data = api().load_image_splits(images, image_size=(8, 8))
    (images / "train/alpha/0.png").write_bytes(b"changed")
    with pytest.raises(api().ImageDatasetError, match="changed"):
        data.train[0]


@pytest.mark.parametrize("mismatch", ["channels", "classes", "invalid_core"])
def test_model_dataset_contract_preserves_core_diagnostics(images, mismatch):
    data = api().load_image_splits(images, image_size=(8, 8))
    config = ModelBuilder(
        (1 if mismatch == "channels" else 3, 8, 8), 3 if mismatch == "classes" else 2
    ).to_config()
    if mismatch == "invalid_core":
        config["head"] = "invalid"
    original = copy.deepcopy(config)
    with pytest.raises(api().ImageDatasetError) as error:
        data.validate_model(config)
    if mismatch == "invalid_core":
        assert error.value.diagnostics == preflight_model(config)["diagnostics"]
    assert config == original


def test_graph_requires_explicit_classification_output(images):
    data = api().load_image_splits(images, image_size=(8, 8))
    config = (
        GraphModelBuilder({"image": [3, 8, 8]})
        .add("head", "classification_head", ["image"], in_features=3, num_classes=2)
        .outputs(logits="head", features="image")
        .to_config()
    )
    data.validate_model(config, output_name="logits")
    for output in [None, "missing", "features"]:
        with pytest.raises(api().ImageDatasetError):
            data.validate_model(config, output_name=output)


def test_explicit_normalization_and_no_implicit_split(images):
    data = api().load_image_splits(
        images, image_size=(8, 8), mean=[0.5] * 3, std=[0.5] * 3
    )
    assert data.train[0][0].min() >= -1 and data.train[0][0].max() <= 1
    assert data.to_dict()["preprocessing"]["mean"] == [0.5] * 3
    with pytest.raises(api().ImageDatasetError):
        api().load_image_splits(images, std=[0, 0, 0], mean=[0, 0, 0])
    only = images / "train"
    with pytest.raises(api().ImageDatasetError):
        api().load_image_splits(only)


def test_preprocessing_is_fp32_without_changing_global_default(images):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        data = api().load_image_splits(images)
        assert data.train[0][0].dtype == torch.float32
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize(
    "mean,std", [([0.0] * 3, [1e-50] * 3), ([1e50] * 3, [1.0] * 3)]
)
def test_unrepresentable_normalization_has_dataset_diagnostic(images, mean, std):
    with pytest.raises(api().ImageDatasetError, match="[Pp]reprocessing"):
        api().load_image_splits(images, mean=mean, std=std)


def test_existing_outputs_are_not_overwritten(images, tmp_path):
    output = tmp_path / "old-output"
    output.mkdir()
    sentinel = output / "model.pth"
    sentinel.write_bytes(b"existing user-like artifact")
    workflow = importlib.import_module("examples.models.image_classification")
    with pytest.raises(FileExistsError):
        workflow.run(images, output)
    assert sentinel.read_bytes() == b"existing user-like artifact"


def test_train_only_bundle_does_not_reuse_train_for_evaluation(images, tmp_path):
    root = tmp_path / "train-only"
    root.mkdir()
    shutil.copytree(images / "train", root / "train")
    data = api().load_image_splits(root)
    assert data.val is None and data.test is None
    workflow = importlib.import_module("examples.models.image_classification")
    with pytest.raises(api().ImageDatasetError, match="separate test"):
        workflow.run(root, tmp_path / "result")
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    "model_kind,channels", [("builder", 3), ("graph", 3), ("graph", 1)]
)
def test_real_file_training_inference_checkpoint_and_tracking(
    tmp_path, model_kind, channels
):
    root = make_images(tmp_path / "images", channels)
    before = hashes(root)
    workflow = importlib.import_module("examples.models.image_classification")
    result = workflow.run(
        root,
        tmp_path / "result",
        model_kind=model_kind,
        channels=channels,
        image_size=(8, 8),
        epochs=1,
        batch_size=2,
        seed=11,
        dataset_origin="generated_fixture",
    )
    assert result["checkpoint_roundtrip_equal"] is True
    assert result["evaluation"]["source"]["kind"] == "model_inference"
    assert result["evaluation"]["metrics"]["sample_count"] == 4
    binding = result["evaluation"]["producer_evidence"]["checkpoint"]
    if model_kind == "builder":
        assert (
            binding["status"] == "native"
            and binding["sha256"] == result["checkpoint_sha256"]
        )
    else:
        assert binding["status"] == "missing"
        assert binding["reason"] == "no_matching_trusted_load_receipt"
    assert result["evaluation"]["configuration"]["class_names"] == ["alpha", "zebra"]
    assert result["training"]["history"]["val_loss"][0] is not None
    assert (
        result["training_batches_per_epoch"] == 4
        and result["validation_batches_per_epoch"] == 2
    )
    assert result["optimizer_parameters_shared"] and result["finite_probe_gradients"]
    assert result["experiment_runs"] == 2
    assert all(a["status"] == "verified" for a in result["artifact_verification"])
    assert hashes(root) == before
