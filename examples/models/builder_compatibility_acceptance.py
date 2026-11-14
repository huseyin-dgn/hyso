"""Bounded synthetic CPU acceptance of existing Builder, Graph and evidence APIs."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from hyso import ModelBuilder, GraphModelBuilder, TrainingConfig, ExperimentManager
from hyso import (
    evaluate_classification,
    capture_model_structure,
    compare_model_structures,
)
from hyso.models.output import GraphTensorOutput
from hyso.checkpoint import save_training_checkpoint, load_training_checkpoint
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
from hyso.architecture_comparison import compare_architectures
from hyso.model_checks import run_model_checks
from hyso.model_observations import (
    observe_result,
    observe_model_checks,
    diagnose_results,
)
from hyso.reporting import create_report


def write(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def sequential(*, basic=False):
    builder = ModelBuilder((1, 8, 8), 3)
    if basic:
        return builder.add("conv2d", in_channels=1, out_channels=4).add("relu")
    return (
        builder.add("conv_norm_act", in_channels=1, out_channels=4)
        .add("max_pool2d", kernel_size=2, stride=2)
        .add("conv2d", in_channels=4, out_channels=4)
        .add("silu")
    )


def branching():
    return (
        GraphModelBuilder({"image": [1, 8, 8]})
        .add("stem", "conv_norm_act", ["image"], in_channels=1, out_channels=4)
        .add("left", "pointwise_conv2d", ["stem"], in_channels=4, out_channels=4)
        .add("right", "identity", ["stem"])
        .add("merge", "add", ["left", "right"])
        .add("head", "classification_head", ["merge"], in_features=4, num_classes=3)
        .outputs(logits="head", features="merge")
    )


def measured_probe(builder, directory, *, graph=False):
    def fresh():
        raw = type(builder).from_config(builder.to_config()).build()
        return GraphTensorOutput(raw, "logits", task="multiclass") if graph else raw

    model = fresh()
    x = torch.linspace(-1, 1, 128).reshape(2, 1, 8, 8).requires_grad_()
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    optimizer.step()
    assert any(not torch.equal(before[k], v) for k, v in model.named_parameters())
    checkpoint = directory / "probe.pth"
    save_training_checkpoint(checkpoint, model, include_rng=False)
    clone = fresh()
    load_training_checkpoint(
        checkpoint,
        clone,
        expected_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    model.eval()
    clone.eval()
    with torch.no_grad():
        assert torch.equal(model(x), clone(x))
    return dict(
        input_shape=list(x.shape),
        output_shape=list(logits.shape),
        parameters=sum(p.numel() for p in model.parameters()),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        loss=float(loss.detach()),
        forward=True,
        backward=True,
        finite_gradients=True,
        optimizer_updated=True,
        strict_checkpoint=True,
    )


def run(output):
    output = Path(output).resolve()
    repo = Path(__file__).resolve().parents[2]
    if output == repo or output.is_relative_to(repo):
        raise ValueError("Use a fresh QA directory outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    probes = {}
    for name, builder, graph in [
        ("basic", sequential(basic=True), False),
        ("multi_layer", sequential(), False),
        ("branching", branching(), True),
    ]:
        directory = output / name
        directory.mkdir()
        builder.to_json(directory / "architecture.json")
        probes[name] = measured_probe(builder, directory, graph=graph)
        write(directory / "probe.json", probes[name])
    invalid = (
        GraphModelBuilder({"image": [1, 8, 8]})
        .add("a", "identity", ["b"])
        .add("b", "identity", ["a"])
        .outputs(out="a")
    )
    negative = invalid.preview()
    assert not negative["valid"] and negative["cycles"]
    write(output / "negative-architecture.json", negative)

    train = DataLoader(
        TensorDataset(
            torch.linspace(-1, 1, 256).reshape(4, 1, 8, 8), torch.tensor([0, 1, 2, 0])
        ),
        batch_size=2,
    )
    test = DataLoader(
        TensorDataset(
            torch.linspace(1, 2, 256).reshape(4, 1, 8, 8), torch.tensor([0, 1, 2, 0])
        ),
        batch_size=2,
    )
    settings = TrainingConfig(
        epochs=1,
        batch_size=2,
        shuffle=False,
        validation=False,
        optimizer="sgd",
        learning_rate=0.01,
        device="cpu",
        precision="fp32",
    )
    context = dict(
        dataset_id="explicit-synthetic-stage13-5-12",
        split="test",
        preprocessing_id="fixed_fp32_identity",
    )
    manager = ExperimentManager(output / "experiments")
    manager.close()
    ExperimentManager.migrate_store(output / "experiments")
    manager = ExperimentManager(output / "experiments")
    models, trainings, evaluations, analyses, sources, associations, configs = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    artifacts = []
    observations = []
    try:
        experiment = manager.create_experiment(
            "Builder and branching Graph synthetic compatibility"
        )
        for index, (kind, builder) in enumerate(
            [("sequential", sequential()), ("graph", branching())]
        ):
            directory = output / ("e2e-" + kind)
            directory.mkdir()
            torch.random.default_generator.manual_seed(13512 + index)
            config = builder.to_config()

            def fresh():
                raw = type(builder).from_config(config).build()
                return (
                    GraphTensorOutput(raw, "logits", task="multiclass")
                    if kind == "graph"
                    else raw
                )

            model = fresh()
            training = settings.prepare(model, train).fit()
            assert training.to_dict()["applied_updates"] == 2
            model.eval()
            metadata = dict(
                architecture=config,
                settings=settings.to_dict(),
                context=context,
                seed=13512 + index,
                origin="explicit_synthetic",
            )
            checkpoint = directory / "model.pth"
            save_training_checkpoint(
                checkpoint,
                model,
                include_rng=False,
                graph_contract=kind == "graph",
                metrics=metadata,
                completed_epochs=1,
                history=training.history,
            )
            expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            restored = fresh().eval()

            def guard(payload):
                if payload["metrics"] != metadata:
                    raise ValueError("checkpoint metadata differs")

            load_training_checkpoint(
                checkpoint,
                restored,
                expected_sha256=expected,
                capture_evidence=True,
                validate_metadata=guard,
            )
            if kind == "graph":
                association = evaluate_graph_classification(
                    restored,
                    test,
                    class_names=["a", "b", "c"],
                    split="test",
                    source_id=context["dataset_id"],
                    context=context,
                    expected_checkpoint_sha256=expected,
                    capture_structure=True,
                    capture_receiver_runtime=True,
                )
                assert association.to_dict()["relation"]["status"] == "verified"
                association.to_json(directory / "association.json")
                assert (
                    GraphEvaluationResult.from_json(
                        directory / "association.json"
                    ).to_dict()["verification"]["receiver_runtime"]
                    == "unverified"
                )
                evaluation = association.evaluation
                associations.append(association)
            else:
                evaluation = evaluate_classification(
                    restored,
                    test,
                    class_names=["a", "b", "c"],
                    split="test",
                    source_id=context["dataset_id"],
                    capture_evidence=True,
                    capture_structure=True,
                )
            record = EvidenceRecord.from_native(
                evaluation.to_dict(),
                feature_id="P26",
                scope=EvidenceScope(model_id=kind),
            )
            analysis = EvidenceIndex([record]).analyze(
                context=AnalysisContext(
                    workload_id="stage13-5-12",
                    task="multiclass",
                    model={"id": kind},
                    dataset={
                        "id": context["dataset_id"],
                        "split": "test",
                        "origin": "declared synthetic",
                    },
                )
            )
            sources.append(
                ModelComparisonSource(kind, analysis, records={"P26": record.record_id})
            )
            write(directory / "training.json", training.to_dict())
            evaluation.to_json(directory / "evaluation.json")
            diagnostic_model = restored.graph if kind == "graph" else restored
            checks = run_model_checks(config, model=diagnostic_model, model_id=kind)
            assert all(row["status"] == "PASS" for row in checks["cases"])
            write(directory / "model-checks.json", checks)
            observations.append(
                observe_result(
                    evaluation,
                    config=config,
                    model_id=kind,
                    dataset=context["dataset_id"],
                    split="test",
                )
            )
            test_observation = observe_model_checks(checks, model_id=kind)
            assert (
                json.loads(test_observation._json)["authenticity"]
                == "UNVERIFIED_SOURCE"
            )
            tr = manager.create_run(experiment["id"], "training", config=metadata)
            tr.start()
            training.log_to(tr)
            for file, artifact_kind in [
                (checkpoint, "checkpoint"),
                (directory / "training.json", "training_result"),
            ]:
                artifacts.append(
                    (
                        tr.add_artifact(
                            file, kind=artifact_kind, ownership="reference"
                        ),
                        file,
                    )
                )
            tr.finish()
            er = manager.create_run(experiment["id"], "evaluation", config=metadata)
            er.start()
            for filename in ["evaluation.json", "model-checks.json"] + (
                ["association.json"] if kind == "graph" else []
            ):
                file = directory / filename
                artifacts.append(
                    (er.add_artifact(file, kind="report", ownership="reference"), file)
                )
            er.finish()
            models.append(restored)
            trainings.append(training)
            evaluations.append(evaluation)
            analyses.append(analysis)
            configs.append(config)
        comparison = compare_models(*sources, features="P26")
        assert all(
            row["status"] == "COMPARABLE" for row in comparison.to_dict()["metrics"]
        )
        comparison.to_json(output / "quality-comparison.json")
        architectures = compare_architectures(
            configs, names=["sequential", "graph"], exact_parameters=True
        )
        assert [c["parameter_count"] for c in architectures["columns"]] == [
            sum(p.numel() for p in m.parameters()) for m in models
        ]
        write(output / "architecture-comparison.json", architectures)
        structural = compare_model_structures(
            *[capture_model_structure(m) for m in models]
        )
        write(output / "registered-structure-comparison.json", structural.to_dict())
        write(output / "observations.json", diagnose_results(observations))
        report = create_report(
            *trainings,
            analyses[0],
            associations[0],
            comparison,
            language="en",
            level="technical",
        )
        report.to_json(output / "report.json")
        report.to_markdown(output / "report.md")
        report.to_html(output / "report.html")
        rr = manager.create_run(
            experiment["id"],
            "evaluation",
            config={"scope": "retained_quality_not_architecture_superiority"},
        )
        rr.start()
        for name in [
            "quality-comparison.json",
            "architecture-comparison.json",
            "registered-structure-comparison.json",
            "report.json",
            "report.md",
            "report.html",
        ]:
            file = output / name
            artifacts.append(
                (rr.add_artifact(file, kind="report", ownership="reference"), file)
            )
        rr.finish()
        checks = [manager.verify_artifact(a["id"], path=p) for a, p in artifacts]
        assert all(row["status"] == "verified" for row in checks)
        assert (
            associations[0].to_dict()["evaluation"]["producer_evidence"]["checkpoint"][
                "status"
            ]
            == "missing"
        )
        assert not torch.cuda.is_initialized()
        summary = dict(
            probes=probes,
            invalid_graph_rejected=True,
            source="explicit_synthetic",
            seeds=[13512, 13513],
            epochs_per_model=1,
            updates=[r.to_dict()["applied_updates"] for r in trainings],
            accuracy=[r.to_dict()["metrics"]["accuracy"] for r in evaluations],
            parameter_counts=[sum(p.numel() for p in m.parameters()) for m in models],
            graph_verification=associations[0].to_dict()["verification"],
            graph_binding=associations[0].to_dict()["relation"],
            legacy_p26=[
                r.to_dict()["producer_evidence"]["checkpoint"]["status"]
                for r in evaluations
            ],
            comparison_claims=comparison.to_dict()["claims"],
            experiment_id=experiment["id"],
            artifact_verification=checks,
            cuda_initialized=False,
        )
        write(output / "summary.json", summary)
        return summary
    finally:
        manager.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output)))
