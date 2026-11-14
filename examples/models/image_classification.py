"""Explicit short CPU training/evaluation from train/val/test image directories.

Run from the checkout with python -m examples.models.image_classification.
No download, implicit split, automatic resume or existing workspace access.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from hyso import ModelBuilder, GraphModelBuilder, TrainingConfig, ExperimentManager
from hyso.datasets import load_image_splits, ImageDatasetError
from hyso.models.output import GraphTensorOutput
from hyso.checkpoint import save_training_checkpoint, load_training_checkpoint
from hyso.analysis import EvidenceRecord, EvidenceIndex, EvidenceScope, AnalysisContext
from hyso.reporting import create_report


def _write(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _builder(kind, channels, size, classes):
    if kind == "builder":
        return ModelBuilder((channels, *size), classes).add(
            "conv_norm_act",
            in_channels=channels,
            out_channels=4,
            kernel_size=3,
            padding=1,
            norm=None,
        )
    if kind != "graph":
        raise ValueError("model_kind must be builder or graph")
    return (
        GraphModelBuilder({"image": [channels, *size]})
        .add(
            "stem",
            "conv_norm_act",
            ["image"],
            in_channels=channels,
            out_channels=4,
            kernel_size=3,
            padding=1,
            norm=None,
        )
        .add(
            "left",
            "conv2d",
            ["stem"],
            in_channels=4,
            out_channels=4,
            kernel_size=1,
            padding=0,
        )
        .add("right", "identity", ["stem"])
        .add("fused", "add", ["left", "right"])
        .add(
            "head", "classification_head", ["fused"], in_features=4, num_classes=classes
        )
        .outputs(logits="head", features="fused")
    )


def run(
    dataset_root,
    output,
    *,
    model_kind="graph",
    channels=3,
    image_size=(8, 8),
    epochs=1,
    batch_size=2,
    seed=17,
    mean=None,
    std=None,
    dataset_origin="caller_supplied",
    dataset_provenance=None,
):
    """Run only on explicit local paths; create a fresh output/experiment store.

    Origin is a caller declaration. Generated fixtures are not real-world data.
    Model-only checkpoints use the existing generic API, not automatic replay.
    """
    if (
        type(epochs) is not int
        or not 1 <= epochs <= 2
        or type(batch_size) is not int
        or not 1 <= batch_size <= 16
    ):
        raise ValueError("Example CPU budget: 1..2 epochs and batch_size 1..16")
    if dataset_origin not in ("caller_supplied", "generated_fixture"):
        raise ValueError("Declare caller_supplied or generated_fixture origin")
    output = Path(output).resolve()
    root = Path(dataset_root).resolve()
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Output and dataset directories must be disjoint")
    if output.exists():
        raise FileExistsError(
            "Use a fresh output directory; no existing artifacts are overwritten"
        )
    data = load_image_splits(
        root, image_size=image_size, channels=channels, mean=mean, std=std
    )
    if data.test is None:
        raise ImageDatasetError(
            "This end-to-end example requires a separate test split; no training data is reused"
        )
    builder = _builder(model_kind, channels, image_size, len(data.classes))
    config = builder.to_config()
    selected = "logits" if model_kind == "graph" else None
    data.validate_model(config, output_name=selected)
    settings = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation=data.val is not None,
        optimizer="sgd",
        learning_rate=0.01,
        device="cpu",
        precision="fp32",
    )
    train, val, test = data.make_loaders(settings, seed=seed)
    torch.set_num_threads(1)
    torch.random.default_generator.manual_seed(seed)
    underlying = builder.build().to(device="cpu", dtype=torch.float32)
    model = (
        GraphTensorOutput(underlying, selected, task="multiclass")
        if selected
        else underlying
    )
    assert all(
        p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()
    )
    # An actual training image probe verifies geometry and the autograd connection.
    probe = torch.stack([data.train[i][0] for i in range(min(2, len(data.train)))])
    logits = model(probe)
    if logits.shape != (len(probe), len(data.classes)):
        raise ValueError("Model logits differ from the training class vocabulary")
    logits.sum().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    finite_gradients = bool(gradients) and all(
        bool(torch.isfinite(g).all()) for g in gradients
    )
    if not finite_gradients:
        raise ValueError("Training image probe produced no finite observed gradients")
    model.zero_grad(set_to_none=True)
    session = settings.prepare(model, train, val)
    shared = {
        id(p)
        for group in session.trainer.optimizer.param_groups
        for p in group["params"]
    } == {id(p) for p in model.parameters()}
    if not shared:
        raise ValueError("Optimizer parameters do not match the model")
    training = session.fit()
    model.eval()
    output.mkdir(parents=True)
    manifest = data.to_dict()
    model_id = model_kind + "-cnn"
    context = dict(
        model_id=model_id,
        architecture=config,
        dataset_id=manifest["dataset_id"],
        dataset_origin=dataset_origin,
        preprocessing=manifest["preprocessing"],
        class_to_idx=manifest["class_to_idx"],
        training=settings.to_dict(),
        seed=seed,
    )
    if dataset_provenance is not None:
        # Optional caller metadata; never promoted to native P26 evidence.
        context["dataset_provenance"] = json.loads(
            json.dumps(dataset_provenance, allow_nan=False)
        )
    context["model_config_id"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    context["split_ids"] = {
        split: hashlib.sha256(
            json.dumps(
                [row for row in manifest["samples"] if row["split"] == split],
                sort_keys=True, separators=(",", ":"),
            ).encode()
        ).hexdigest()
        for split in manifest["split_counts"]
    }
    _write(output / "dataset-manifest.json", manifest)
    _write(output / "training-result.json", training.to_dict())
    checkpoint = output / "model.pth"
    save_training_checkpoint(
        checkpoint,
        model,
        completed_epochs=epochs,
        history=training.history,
        metrics=context,
        include_rng=False,
    )
    restored_base = builder.build().to(device="cpu", dtype=torch.float32)
    restored = (
        GraphTensorOutput(restored_base, selected, task="multiclass")
        if selected
        else restored_base
    )
    def validate_metadata(metadata):
        if metadata["metrics"] != context:
            raise ValueError("Checkpoint model/data/training metadata differs")

    checkpoint_info = load_training_checkpoint(
        checkpoint, restored, map_location="cpu", capture_evidence=True,
        validate_metadata=validate_metadata,
    )
    restored.eval()
    # Validation (or train if validation is absent) checks restoration, never model selection on test.
    roundtrip_source = data.val if data.val is not None else data.train
    fixed = torch.stack(
        [roundtrip_source[i][0] for i in range(min(2, len(roundtrip_source)))]
    )
    with torch.no_grad():
        equal = torch.equal(model(fixed), restored(fixed))
    if not equal:
        raise ValueError(
            "Checkpoint roundtrip changed inference on the fixed image batch"
        )
    from hyso.evaluation import evaluate_classification

    evaluation = evaluate_classification(
        restored,
        test,
        class_names=list(data.classes),
        split="test",
        source_id=manifest["dataset_id"],
        capture_evidence=True,
    )
    evaluation_data = evaluation.to_dict()
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint_binding = evaluation_data["producer_evidence"]["checkpoint"]
    if (
        checkpoint_binding["status"] == "native"
        and checkpoint_binding["sha256"] != checkpoint_sha
    ):
        raise ValueError("Native evaluation did not bind the restored checkpoint")
    _write(output / "evaluation.json", evaluation_data)
    record = EvidenceRecord.from_native(
        evaluation_data, feature_id="P26", scope=EvidenceScope(model_id=model_id)
    )
    analysis = EvidenceIndex([record]).analyze(
        context=AnalysisContext(
            workload_id="file-image-classification",
            task="multiclass",
            model={
                "id": model_id,
                "architecture": config,
                "checkpoint_sha256": checkpoint_sha,
            },
            dataset={
                "id": manifest["dataset_id"],
                "split": "test",
                "classes": list(data.classes),
                "origin": dataset_origin,
                "preprocessing": manifest["preprocessing"],
            },
        )
    )
    report = create_report(training, analysis, language="en", level="technical")
    report.to_json(output / "report.json")
    report.to_markdown(output / "report.md")
    report.to_html(output / "report.html")
    # Explicit calls to the existing Manager; only this new local store is migrated.
    store = output / "experiments"
    manager = ExperimentManager(store)
    manager.close()
    ExperimentManager.migrate_store(store)
    manager = ExperimentManager(store)
    verification = []
    try:
        experiment = manager.create_experiment("File-backed CPU classification")
        training_run = manager.create_run(
            experiment["id"],
            "training",
            config=context,
            provenance={
                "source": "explicit Python image example",
                "dataset_origin": dataset_origin,
            },
        )
        training_run.start()
        dataset_artifact = training_run.add_artifact(
            output / "dataset-manifest.json",
            kind="dataset_manifest",
            ownership="reference",
            role="input",
            metadata={"dataset_id": manifest["dataset_id"]},
        )
        training.log_to(training_run)
        train_artifact = training_run.add_artifact(
            output / "training-result.json",
            kind="training_result",
            ownership="reference",
        )
        checkpoint_artifact = training_run.add_artifact(
            checkpoint,
            kind="checkpoint",
            ownership="reference",
            metadata={
                "model_id": model_id,
                "dataset_id": manifest["dataset_id"],
                "sha256": checkpoint_sha,
            },
        )
        training_run.finish()
        evaluation_run = manager.create_run(
            experiment["id"],
            "evaluation",
            config={
                **context,
                "split": "test",
                "training_run_id": training_run.id,
                "checkpoint_artifact_id": checkpoint_artifact["id"],
            },
            provenance={"source": "evaluate_classification model inference"},
        )
        evaluation_run.start()
        for artifact in (dataset_artifact, train_artifact, checkpoint_artifact):
            manager.link_artifact(evaluation_run.id, artifact["id"], role="input")
        evaluation_artifact = evaluation_run.add_artifact(
            output / "evaluation.json",
            kind="evaluation",
            ownership="reference",
            metadata={
                "split": "test",
                "checkpoint_sha256": checkpoint_sha,
                "dataset_id": manifest["dataset_id"],
            },
        )
        report_artifacts = [
            evaluation_run.add_artifact(
                output / name, kind="report", ownership="reference"
            )
            for name in ("report.json", "report.md", "report.html")
        ]
        for key, value in [
            ("test_accuracy", evaluation_data["metrics"]["accuracy"]),
            ("test_macro_f1", evaluation_data["metrics"]["averages"]["macro"]["f1"]),
        ]:
            evaluation_run.log_metric(
                name=key,
                value=value,
                unit="fraction",
                source="evaluate_classification",
                epoch=epochs,
            )
        evaluation_run.finish()
        artifacts = (
            dataset_artifact,
            train_artifact,
            checkpoint_artifact,
            evaluation_artifact,
            *report_artifacts,
        )
        filenames = (
            "dataset-manifest.json",
            "training-result.json",
            "model.pth",
            "evaluation.json",
            "report.json",
            "report.md",
            "report.html",
        )
        for artifact, filename in zip(artifacts, filenames):
            verification.append(
                manager.verify_artifact(artifact["id"], path=output / filename)
            )
        if not all(item["status"] == "verified" for item in verification):
            raise ValueError("Tracking artifact bytes failed verification")
        history = manager.get_experiment_history(experiment["id"])
        tracking = dict(
            experiment_id=experiment["id"],
            training_run_id=training_run.id,
            evaluation_run_id=evaluation_run.id,
            dataset_artifact_id=dataset_artifact["id"],
            checkpoint_artifact_id=checkpoint_artifact["id"],
            evaluation_artifact_id=evaluation_artifact["id"],
            report_artifact_ids=[a["id"] for a in report_artifacts],
        )
    finally:
        manager.close()
    result = dict(
        dataset_origin=dataset_origin,
        dataset_id=manifest["dataset_id"],
        context=context,
        training=training.to_dict(),
        training_batches_per_epoch=len(train),
        validation_batches_per_epoch=0 if val is None else len(val),
        test_batches=len(test),
        finite_probe_gradients=finite_gradients,
        optimizer_parameters_shared=shared,
        checkpoint_roundtrip_equal=equal,
        checkpoint_sha256=checkpoint_sha,
        native_checkpoint_binding=checkpoint_binding,
        checkpoint_metadata=checkpoint_info,
        evaluation=evaluation_data,
        experiment_runs=len(history["runs"]),
        tracking=tracking,
        artifact_verification=verification,
    )
    _write(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["builder", "graph"], default="graph")
    parser.add_argument("--channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mean", nargs="+", type=float)
    parser.add_argument("--std", nargs="+", type=float)
    parser.add_argument(
        "--dataset-origin",
        choices=["caller_supplied", "generated_fixture"],
        default="caller_supplied",
    )
    args = parser.parse_args()
    result = run(
        args.dataset,
        args.output,
        model_kind=args.model,
        channels=args.channels,
        image_size=(args.height, args.width),
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        mean=args.mean,
        std=args.std,
        dataset_origin=args.dataset_origin,
    )
    print(
        json.dumps(
            {
                "origin": result["dataset_origin"],
                "dataset_id": result["dataset_id"],
                "test_samples": result["evaluation"]["metrics"]["sample_count"],
                "accuracy": result["evaluation"]["metrics"]["accuracy"],
                "checkpoint_roundtrip": result["checkpoint_roundtrip_equal"],
                "report": str(args.output / "report.html"),
            }
        )
    )
