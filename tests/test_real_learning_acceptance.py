"""Short acceptance protocol checks; never rerun the two 20-epoch main fits."""

import copy
import socket

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from examples.models.real_learning_acceptance import (
    prepare,
    builders,
    fresh,
    validate_partition,
    validate_epoch,
)
from hyso import evaluate_classification, TrainingConfig
from hyso.datasets import ImageDatasetError, load_image_splits


def records():
    return [
        dict(
            split=s,
            label=i,
            source_id=f"{s}-{i}",
            original_pixels_sha256=f"pixel-{s}-{i}",
        )
        for s in ("train", "val", "test")
        for i in range(10)
    ]


@pytest.mark.parametrize(
    "problem", ["empty_train", "empty_val", "empty_test", "ids", "pixels", "vocabulary"]
)
def test_partition_rejects_invalid_protocol(problem):
    rows = records()
    if problem.startswith("empty_"):
        rows = [r for r in rows if r["split"] != problem[6:]]
    elif problem == "ids":
        rows[10]["source_id"] = rows[0]["source_id"]
    elif problem == "pixels":
        rows[20]["original_pixels_sha256"] = rows[0]["original_pixels_sha256"]
    else:
        rows[10]["label"] = 10
    with pytest.raises(ValueError):
        validate_partition(rows)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_step",
        "hook_step",
        "validation_step",
        "nan",
        "inf",
        "partial_train",
        "partial_val",
    ],
)
def test_epoch_rejects_incomplete_or_nonfinite_evidence(problem):
    row = dict(
        train_samples=1257,
        val_samples=270,
        optimizer_updates=20,
        optimizer_hook_updates=20,
        validation_optimizer_updates=0,
        train_loss=1.0,
        train_acc=50.0,
        val_loss=1.1,
        val_acc=45.0,
    )
    if problem == "missing_step":
        row["optimizer_updates"] = 19
    elif problem == "hook_step":
        row["optimizer_hook_updates"] = 19
    elif problem == "validation_step":
        row["validation_optimizer_updates"] = 1
    elif problem == "nan":
        row["train_loss"] = float("nan")
    elif problem == "inf":
        row["val_loss"] = float("inf")
    elif problem == "partial_train":
        row["train_samples"] = 128
    else:
        row["val_samples"] = 128
    with pytest.raises(ValueError):
        validate_epoch(row, 1257, 270, 20)


def test_full_offline_source_and_actual_loader_counts(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    data, profile = prepare(tmp_path / "digits")
    assert profile["split_counts"] == dict(train=1257, val=270, test=270)
    assert profile["exact_duplicate_groups"] == []
    assert len({r["source_id"] for r in profile["samples"]}) == 1797
    assert data.classes == tuple(str(i) for i in range(10))
    assert sum(profile["source_class_counts"]) == 1797
    for kind, builder in builders().items():
        data.validate_model(
            builder.to_config(), output_name="logits" if kind == "graph" else None
        )
    with pytest.raises(ImageDatasetError, match="budget"):
        load_image_splits(tmp_path / "digits", channels=1)


@pytest.mark.parametrize("kind", ["sequential", "graph"])
def test_real_learning_architectures_short_native_session_and_restore(kind, tmp_path):
    from hyso.checkpoint import save_training_checkpoint, load_training_checkpoint

    config = builders()[kind].to_config()
    model = fresh(kind, config)
    x = torch.linspace(0, 1, 8 * 64).reshape(8, 1, 8, 8)
    y = torch.arange(8)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    before = copy.deepcopy(model.state_dict())
    settings = TrainingConfig(
        epochs=1, batch_size=4, optimizer="adamw", learning_rate=0.003
    )
    result = settings.prepare(model, loader).fit()
    assert result.to_dict()["applied_updates"] == 2
    assert any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    path = tmp_path / "model.pth"
    save_training_checkpoint(
        path, model, include_rng=False, graph_contract=kind == "graph"
    )
    restored = fresh(kind, config)
    load_training_checkpoint(path, restored)
    model.eval()
    restored.eval()
    with torch.no_grad():
        assert torch.equal(model(x), restored(x))


@pytest.mark.parametrize(
    "problem", ["vocabulary", "label_range", "nan_input", "inf_input"]
)
def test_actual_evaluation_rejects_invalid_classification_inputs(problem):
    model = fresh("sequential", builders()["sequential"].to_config())
    x = torch.zeros(2, 1, 8, 8)
    y = torch.tensor([0, 1])
    names = [str(i) for i in range(10)]
    if problem == "vocabulary":
        names[1] = names[0]
    elif problem == "label_range":
        y[1] = 10
    elif problem == "nan_input":
        x[0, 0, 0, 0] = float("nan")
    else:
        x[0, 0, 0, 0] = float("inf")
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    with pytest.raises((ValueError, RuntimeError)):
        evaluate_classification(model, loader, class_names=names, split="val")


def test_different_preprocessing_cannot_compare_as_same_observed_stream(tmp_path):
    from PIL import Image
    from hyso import ModelBuilder
    from hyso.analysis import (
        EvidenceRecord,
        EvidenceScope,
        EvidenceIndex,
        AnalysisContext,
        ModelComparisonSource,
        compare_models,
    )

    for si, split in enumerate(("train", "val", "test")):
        for label in range(2):
            directory = tmp_path / split / str(label)
            directory.mkdir(parents=True)
            Image.new("L", (8, 8), 20 + si * 60 + label * 20).save(directory / "a.png")
    ordinary = load_image_splits(tmp_path, channels=1)
    normalized = load_image_splits(tmp_path, channels=1, mean=[0.5], std=[0.5])
    assert ordinary.to_dict()["preprocessing"] != normalized.to_dict()["preprocessing"]
    assert ordinary.to_dict()["dataset_id"] != normalized.to_dict()["dataset_id"]
    model = (
        ModelBuilder((1, 8, 8), 2).add("conv2d", in_channels=1, out_channels=2).build()
    )
    sources = []
    for index, data in enumerate((ordinary, normalized)):
        # Validation fixtures only: the real acceptance test stream stays locked.
        evaluation = evaluate_classification(
            model,
            DataLoader(data.val, batch_size=2),
            class_names=["0", "1"],
            capture_evidence=True,
            split="val",
        )
        name = str(index)
        record = EvidenceRecord.from_native(
            evaluation.to_dict(), feature_id="P26", scope=EvidenceScope(model_id=name)
        )
        analysis = EvidenceIndex([record]).analyze(
            context=AnalysisContext(
                workload_id="negative-preprocessing",
                task="multiclass",
                model={"id": name},
                dataset={},
            )
        )
        sources.append(
            ModelComparisonSource(name, analysis, records={"P26": record.record_id})
        )
    comparison = compare_models(*sources, features="P26").to_dict()
    assert all(row["status"] == "INCOMPATIBLE" for row in comparison["metrics"])
