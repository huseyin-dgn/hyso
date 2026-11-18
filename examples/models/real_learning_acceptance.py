"""Offline, bounded real-learning acceptance using existing HYSo APIs.

Run once into a fresh external QA directory. Integrity scans may inspect test
files before training; test inference is permitted once per selected model only
after both validation-only selections have been sealed. No hyperparameter retry.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hyso import ModelBuilder, GraphModelBuilder, TrainingConfig, ExperimentManager
from hyso import evaluate_classification
from hyso.datasets import load_image_splits
from hyso.checkpoint import save_training_checkpoint, load_training_checkpoint
from hyso.models.output import GraphTensorOutput
from hyso.analysis import (
    EvidenceRecord,
    EvidenceScope,
    EvidenceIndex,
    AnalysisContext,
    ModelComparisonSource,
    compare_models,
)
from hyso.analysis.graph_evidence import (
    evaluate_graph_classification,
    GraphEvaluationResult,
)
from hyso.reporting import create_report
from .real_digits_acceptance import png_bytes, SOURCE, CONVERSION


SEED = 13513
TARGETS = dict(validation_accuracy=0.75, test_accuracy=0.70, test_macro_f1=0.65)


def write(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_partition(records):
    """Acceptance preflight for source IDs and exact original-pixel leakage."""
    ids, pixels = {}, {}
    for split in ("train", "val", "test"):
        rows = [r for r in records if r["split"] == split]
        if not rows or set(r["label"] for r in rows) != set(range(10)):
            raise ValueError("nonempty ten-class splits required")
        for row in rows:
            if row["source_id"] in ids:
                raise ValueError("overlapping source IDs")
            ids[row["source_id"]] = split
            key = row["original_pixels_sha256"]
            if key in pixels and pixels[key] != split:
                raise ValueError("exact duplicate leakage")
            pixels[key] = split


def prepare(root):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    import sklearn

    digits = load_digits()
    x, y = digits.images, digits.target
    if x.shape != (1797, 8, 8) or not np.isfinite(x).all():
        raise ValueError("unexpected source shape/nonfinite pixels")
    if not np.equal(x, x.astype(np.uint8)).all() or (x.min(), x.max()) != (0, 16):
        raise ValueError("unexpected source pixel range")
    groups = {}
    for i, image in enumerate(x):
        groups.setdefault(
            hashlib.sha256(image.astype(np.uint8).tobytes()).hexdigest(), []
        ).append(i)
    duplicates = [v for v in groups.values() if len(v) > 1]
    # Predetermined fail-closed policy: a changed source with duplicates requires
    # a separately reviewed group-stratified protocol before any training.
    if duplicates:
        raise ValueError(f"source duplicates require group split: {duplicates}")
    train, holdout = train_test_split(
        np.arange(len(y)), test_size=0.30, stratify=y, random_state=SEED
    )
    val, test = train_test_split(
        holdout, test_size=0.50, stratify=y[holdout], random_state=SEED + 1
    )
    records = []
    for split, indices in [("train", train), ("val", val), ("test", test)]:
        for i in sorted(indices.tolist()):
            relative = f"{split}/{int(y[i])}/source-{i:04d}.png"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(png_bytes(x[i]))
            records.append(
                dict(
                    split=split,
                    label=int(y[i]),
                    source_id=f"{SOURCE}:row:{i}",
                    source_row_zero_based=i,
                    path=relative,
                    original_pixels_sha256=hashlib.sha256(
                        x[i].astype(np.uint8).tobytes()
                    ).hexdigest(),
                    png_sha256=sha(path),
                )
            )
    validate_partition(records)
    data = load_image_splits(
        root, channels=1, image_size=(8, 8), max_samples_per_split=1797
    )
    manifest = data.to_dict()
    # Verify every actual loader tensor against its original source, including
    # the explicitly documented 8-bit conversion error (<= 0.5/255).
    actual = {}
    for split in ("train", "val", "test"):
        ds = getattr(data, split)
        count = 0
        for (path, label), (tensor, target) in zip(ds.samples, ds):
            i = int(Path(path).stem.split("-")[1])
            expected = (
                torch.from_numpy(
                    ((x[i].astype(np.uint16) * 255 + 8) // 16).astype(np.uint8)
                ).float()
                / 255
            )
            assert tensor.shape == (1, 8, 8) and tensor.dtype == torch.float32
            assert torch.equal(tensor[0], expected) and label == target == int(y[i])
            assert (
                tensor[0] - torch.from_numpy(x[i] / 16).float()
            ).abs().max() <= 0.5 / 255 + 1e-7
            count += 1
        actual[split] = count
    assert actual == manifest["split_counts"] and sum(actual.values()) == 1797
    profile = dict(
        source=SOURCE,
        sklearn_version=sklearn.__version__,
        shape=list(x.shape),
        source_class_counts=np.bincount(y).tolist(),
        pixel_range=[float(x.min()), float(x.max())],
        finite=True,
        exact_duplicate_groups=duplicates,
        duplicate_policy="abort before training; require group split",
        source_array_sha256=hashlib.sha256(
            x.astype(np.uint8).tobytes() + y.astype("<i8").tobytes()
        ).hexdigest(),
        split_seed=SEED,
        split_method="stratified 70/15/15, sklearn rounding",
        split_counts=actual,
        split_class_counts={
            s: np.bincount(
                [r["label"] for r in records if r["split"] == s], minlength=10
            ).tolist()
            for s in actual
        },
        conversion=CONVERSION,
        normalization_range=[0, 1],
        tensor_scan_counts=actual,
        independent_writers=False,
        external_generalization=False,
        samples=records,
        historical_baseline=dict(
            train=120, val=40, test=40, epochs=2, accuracy=0.10, class_collapse=True
        ),
    )
    return data, profile


def builders():
    sequential = (
        ModelBuilder.sequential((1, 8, 8), num_classes=10, pooling="none")
        .add("conv2d", in_channels=1, out_channels=16, kernel_size=3, padding=1)
        .add("relu")
        .add("conv2d", in_channels=16, out_channels=32, kernel_size=3, padding=1)
        .add("relu")
        .add("flatten")
    )
    graph = (
        GraphModelBuilder({"image": [1, 8, 8]})
        .add(
            "stem",
            "conv2d",
            ["image"],
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1,
        )
        .add("activation", "relu", ["stem"])
        .add(
            "left",
            "conv2d",
            ["activation"],
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            padding=1,
        )
        .add("left_relu", "relu", ["left"])
        .add("right", "pointwise_conv2d", ["activation"], in_channels=8, out_channels=8)
        .add("right_relu", "relu", ["right"])
        .add("fusion", "concat", ["left_relu", "right_relu"], axis=1)
        .add("flat", "flatten", ["fusion"])
        .add(
            "head",
            "classification_head",
            ["flat"],
            in_features=16 * 8 * 8,
            num_classes=10,
            pooling="none",
        )
        .outputs(logits="head")
    )
    return dict(sequential=sequential, graph=graph)


def fresh(kind, config):
    raw = (
        (GraphModelBuilder if kind == "graph" else ModelBuilder)
        .from_config(config)
        .build()
    )
    return (
        GraphTensorOutput(raw, "logits", task="multiclass") if kind == "graph" else raw
    )


def validate_epoch(row, train_count, val_count, updates):
    if row["train_samples"] != train_count or row["val_samples"] != val_count:
        raise ValueError("incomplete epoch sample accounting")
    if row["optimizer_updates"] != updates or row["optimizer_hook_updates"] != updates:
        raise ValueError("missing optimizer update")
    if row["validation_optimizer_updates"] != 0:
        raise ValueError("validation optimizer update")
    if not all(
        np.isfinite(row[k]) for k in ("train_loss", "train_acc", "val_loss", "val_acc")
    ):
        raise ValueError("nonfinite epoch metrics")


def train_one(kind, builder, data, directory, settings, seed):
    directory.mkdir()
    config = builder.to_config()
    data.validate_model(config, output_name="logits" if kind == "graph" else None)
    builder.to_json(directory / "architecture.json")
    torch.random.default_generator.manual_seed(seed)
    model = fresh(kind, config)
    train, val, _ = data.make_loaders(settings, seed=seed)
    initial = evaluate_classification(
        model, val, class_names=list(data.classes), split="val"
    )
    initial.to_json(directory / "initial-validation.json")
    save_training_checkpoint(directory / "initial.pth", model, include_rng=False)
    best, events = {}, []
    counts = dict(
        train_samples=0,
        val_samples=0,
        train_batches=0,
        val_batches=0,
        steps=0,
        val_steps=0,
    )

    def forward_observer(module, args, output):
        x = args[0]
        assert torch.isfinite(x).all() and torch.isfinite(output).all()
        assert output.shape == (len(x), 10) and x.shape[1:] == (1, 8, 8)
        phase = "train" if module.training else "val"
        counts[phase + "_samples"] += len(x)
        counts[phase + "_batches"] += 1

    def step_observer(optimizer, args, kwargs):
        counts["steps"] += 1
        counts["val_steps"] += int(not model.training)

    def epoch_observer(row):
        row.update(
            {
                k: counts[k]
                for k in (
                    "train_samples",
                    "val_samples",
                    "train_batches",
                    "val_batches",
                )
            }
        )
        row["optimizer_updates"] = row["applied_updates"] - sum(
            r["optimizer_updates"] for r in events
        )
        row["optimizer_hook_updates"] = counts["steps"]
        row["validation_optimizer_updates"] = counts["val_steps"]
        row["finite_inputs_outputs_loss_state"] = all(
            torch.isfinite(t).all().item() for t in model.state_dict().values()
        )
        assert row["finite_inputs_outputs_loss_state"]
        validate_epoch(row, len(data.train), len(data.val), len(train))
        assert row["train_batches"] == len(train) and row["val_batches"] == len(val)
        if not best or row["val_loss"] < best["val_loss"]:
            best.update(
                epoch=row["epoch"],
                val_loss=row["val_loss"],
                val_acc=row["val_acc"],
                state=copy.deepcopy(model.state_dict()),
            )
        events.append(row)
        for k in counts:
            counts[k] = 0
        write(directory / f"epoch-{row['epoch']:02d}.json", row)
        print(
            json.dumps(
                dict(
                    model=kind,
                    epoch=row["epoch"],
                    train_loss=row["train_loss"],
                    val_acc=row["val_acc"],
                )
            ),
            flush=True,
        )

    session = settings.prepare(model, train, val, on_epoch=epoch_observer)
    h = model.register_forward_hook(forward_observer)
    step = session.trainer.optimizer.register_step_post_hook(step_observer)
    try:
        training = session.fit()
    finally:
        h.remove()
        step.remove()
    write(directory / "training.json", training.to_dict())
    model.load_state_dict(best.pop("state"), strict=True)
    model.eval()
    selection = dict(
        **best,
        criterion="minimum validation loss; earliest tie",
        test_inference_count=0,
        last_epoch=events[-1],
        initial_validation=initial.to_dict()["metrics"],
        parameters=sum(p.numel() for p in model.parameters()),
        seed=seed,
        initial_checkpoint_sha256=sha(directory / "initial.pth"),
    )
    write(directory / "selection.json", selection)
    return dict(
        kind=kind,
        model=model,
        training=training,
        selection=selection,
        config=config,
        directory=directory,
        events=events,
    )


def restore(row, data, settings, context):
    kind, model, directory = row["kind"], row["model"], row["directory"]
    metadata = dict(
        architecture=row["config"],
        settings=settings.to_dict(),
        context=context,
        seed=row["selection"]["seed"],
        selected_epoch=row["selection"]["epoch"],
    )
    checkpoint = directory / "selected.pth"
    save_training_checkpoint(
        checkpoint,
        model,
        include_rng=False,
        graph_contract=kind == "graph",
        metrics=metadata,
    )
    expected = sha(checkpoint)
    restored = fresh(kind, row["config"]).eval()

    def guard(payload):
        if payload["metrics"] != metadata:
            raise ValueError("architecture/settings metadata mismatch")

    load_training_checkpoint(
        checkpoint,
        restored,
        expected_sha256=expected,
        capture_evidence=True,
        validate_metadata=guard,
    )
    assert model.state_dict().keys() == restored.state_dict().keys()
    assert all(
        torch.equal(t, restored.state_dict()[k]) for k, t in model.state_dict().items()
    )
    x, _ = next(iter(DataLoader(data.val, batch_size=16, shuffle=False)))
    with torch.no_grad():
        assert torch.equal(model(x), restored(x))
    assert all(not m.training for m in restored.modules())
    assert all(
        t.dtype == torch.float32 and t.device.type == "cpu"
        for t in restored.parameters()
    )
    row.update(
        restored=restored,
        checkpoint=checkpoint,
        checkpoint_sha256=expected,
        metadata=metadata,
    )
    write(
        directory / "restore.json",
        dict(
            checkpoint_sha256=expected,
            strict_state_equal=True,
            fixed_validation_output_equal=True,
            eval_mode=True,
            dtype="torch.float32",
            device="cpu",
            metadata_equal=True,
            optimizer_epoch_resume="not verified",
            graph_v2_requested=kind == "graph",
        ),
    )


def run(output):
    output = Path(output).resolve()
    repo = Path(__file__).resolve().parents[2]
    if output == repo or output.is_relative_to(repo):
        raise ValueError("external QA output required")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data, profile = prepare(output / "digits")
    write(output / "dataset-profile.json", profile)
    write(output / "dataset-manifest.json", data.to_dict())
    settings = TrainingConfig(
        epochs=20,
        batch_size=64,
        optimizer="adamw",
        learning_rate=0.003,
        optimizer_params={"weight_decay": 0.0001},
        shuffle=True,
        validation=True,
        device="cpu",
        precision="fp32",
    )
    plan = dict(
        targets=TARGETS,
        training=settings.to_dict(),
        split_seed=SEED,
        model_seeds=[SEED, SEED + 1],
        architectures={k: b.to_config() for k, b in builders().items()},
        maximum_main_fits=2,
        selection="minimum validation loss; earliest tie",
        early_stopping=False,
        test_policy="one inference per model after BOTH selections sealed",
        chance_baseline=0.1,
        majority_baselines={
            s: max(c) / sum(c) for s, c in profile["split_class_counts"].items()
        },
    )
    write(output / "pretraining-plan.json", plan)
    # Shape/finite smoke uses real training inputs only; no extra training fit.
    smoke_x, _ = next(iter(DataLoader(data.train, batch_size=2)))
    for kind, b in builders().items():
        probe = fresh(kind, b.to_config()).eval()
        with torch.no_grad():
            logits = probe(smoke_x)
            assert logits.shape == (2, 10) and torch.isfinite(logits).all()
    rows = [
        train_one(k, b, data, output / k, settings, SEED + i)
        for i, (k, b) in enumerate(builders().items())
    ]
    write(
        output / "selection-seal.json",
        dict(
            plan_sha256=sha(output / "pretraining-plan.json"),
            selections={
                r["kind"]: sha(r["directory"] / "selection.json") for r in rows
            },
            test_inference_count=0,
        ),
    )
    context = dict(
        dataset_id=data.to_dict()["dataset_id"],
        split="test",
        preprocessing_id=hashlib.sha256(
            json.dumps(data.to_dict()["preprocessing"], sort_keys=True).encode()
        ).hexdigest(),
    )
    for row in rows:
        restore(row, data, settings, context)
    test = DataLoader(data.test, batch_size=64, shuffle=False, num_workers=0)
    sources = []
    associations = []
    for row in rows:
        kwargs = dict(
            class_names=list(data.classes),
            split="test",
            source_id=context["dataset_id"],
            capture_structure=True,
        )
        if row["kind"] == "graph":
            association = evaluate_graph_classification(
                row["restored"],
                test,
                **kwargs,
                context=context,
                expected_checkpoint_sha256=row["checkpoint_sha256"],
                capture_receiver_runtime=True,
            )
            association.to_json(row["directory"] / "association.json")
            assert association.to_dict()["relation"]["status"] == "verified"
            assert (
                GraphEvaluationResult.from_json(
                    row["directory"] / "association.json"
                ).to_dict()["verification"]["receiver_runtime"]
                == "unverified"
            )
            evaluation = association.evaluation
            associations.append(association)
        else:
            evaluation = evaluate_classification(
                row["restored"], test, **kwargs, capture_evidence=True
            )
        row["evaluation"] = evaluation
        evaluation.to_json(row["directory"] / "evaluation.json")
        metrics = evaluation.to_dict()["metrics"]
        cm = metrics["confusion_matrix"]
        predictions = [sum(r[i] for r in cm) for i in range(10)]
        assert metrics["sample_count"] == len(data.test)
        acceptance = dict(
            validation=row["selection"]["val_acc"] / 100
            >= TARGETS["validation_accuracy"],
            test_accuracy=metrics["accuracy"] >= TARGETS["test_accuracy"],
            test_macro_f1=metrics["averages"]["macro"]["f1"]
            >= TARGETS["test_macro_f1"],
            meaningful_loss_drop=row["events"][-1]["train_loss"]
            < 0.7 * row["events"][0]["train_loss"],
            beats_majority=metrics["accuracy"] > plan["majority_baselines"]["test"],
            no_class_collapse=sum(n > 0 for n in predictions) > 1,
        )
        row["acceptance"] = acceptance
        write(
            row["directory"] / "quality.json",
            dict(
                metrics=metrics,
                predicted_class_counts=predictions,
                correct_count=sum(cm[i][i] for i in range(10)),
                acceptance=acceptance,
                test_inference_count=1,
            ),
        )
        record = EvidenceRecord.from_native(
            evaluation.to_dict(),
            feature_id="P26",
            scope=EvidenceScope(model_id=row["kind"]),
        )
        analysis = EvidenceIndex([record]).analyze(
            context=AnalysisContext(
                workload_id="stage13-5-13",
                task="multiclass",
                model={"id": row["kind"]},
                dataset={
                    "id": context["dataset_id"],
                    "split": "test",
                    "origin": "local real UCI digits",
                },
            )
        )
        row["analysis"] = analysis
        sources.append(
            ModelComparisonSource(
                row["kind"], analysis, records={"P26": record.record_id}
            )
        )
    comparison = compare_models(*sources, features="P26")
    assert all(r["status"] == "COMPARABLE" for r in comparison.to_dict()["metrics"])
    comparison.to_json(output / "quality-comparison.json")
    assert (
        rows[1]["evaluation"].to_dict()["producer_evidence"]["checkpoint"]["status"]
        == "missing"
    )
    report = create_report(
        *[r["training"] for r in rows],
        *[r["analysis"] for r in rows],
        *associations,
        comparison,
        language="en",
        level="technical",
    )
    report.to_json(output / "report.json")
    report.to_markdown(output / "report.md")
    report.to_html(output / "report.html")
    store = output / "experiments"
    manager = ExperimentManager(store)
    manager.close()
    ExperimentManager.migrate_store(store)
    manager = ExperimentManager(store)
    artifacts = []
    try:
        experiment = manager.create_experiment("Stage 13.5.13 real CNN learning")
        for row in rows:
            run = manager.create_run(
                experiment["id"], "training", config=row["metadata"]
            )
            run.start()
            row["training"].log_to(run)
            for file in sorted(row["directory"].iterdir()):
                a = run.add_artifact(
                    file,
                    kind="checkpoint" if file.suffix == ".pth" else "report",
                    ownership="reference",
                )
                artifacts.append((a, file))
            run.finish()
        run = manager.create_run(experiment["id"], "evaluation", config=context)
        run.start()
        for file in sorted(output.iterdir()):
            if file.is_file():
                artifacts.append(
                    (run.add_artifact(file, kind="report", ownership="reference"), file)
                )
        run.finish()
        checks = [manager.verify_artifact(a["id"], path=p) for a, p in artifacts]
        assert all(c["status"] == "verified" for c in checks)
    finally:
        manager.close()
    assert not torch.cuda.is_initialized()
    summary = dict(
        models={
            r["kind"]: dict(
                selection=r["selection"],
                acceptance=r["acceptance"],
                test_metrics=r["evaluation"].to_dict()["metrics"],
                checkpoint_sha256=r["checkpoint_sha256"],
            )
            for r in rows
        },
        graph_verification=associations[0].to_dict()["verification"],
        graph_relation=associations[0].to_dict()["relation"],
        comparison_claims=comparison.to_dict()["claims"],
        experiment_id=experiment["id"],
        artifact_verification=checks,
        cuda_initialized=False,
        main_training_fits=2,
        overall_learning_pass=all(all(r["acceptance"].values()) for r in rows),
    )
    write(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output)))
