"""A-I: explicit, short CPU engineering workflow; synthetic data, no quality claim."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from hyso import (
    ModelBuilder,
    GraphModelBuilder,
    TrainingConfig,
    BenchmarkRunner,
    ExperimentManager,
)
from hyso.evaluation import evaluate_classification
from hyso.models.output import GraphTensorOutput
from hyso.model_checks import run_model_checks
from hyso.architecture_comparison import compare_architectures
from hyso.structure import capture_model_structure
from hyso.structure_comparison import compare_model_structures
from hyso.analysis import EvidenceRecord, EvidenceIndex, EvidenceScope, AnalysisContext
from hyso.advisor import recommend
from hyso.reporting import create_report


def architecture(classes=2):
    return (
        GraphModelBuilder({"image": [1, 4, 4]})
        .add(
            "stem",
            "conv_norm_act",
            ["image"],
            in_channels=1,
            out_channels=2,
            kernel_size=1,
            padding=0,
            norm=None,
        )
        .add(
            "left",
            "conv2d",
            ["stem"],
            in_channels=2,
            out_channels=2,
            kernel_size=1,
            padding=0,
        )
        .add("right", "identity", ["stem"])
        .add("fused", "add", ["left", "right"])
        .add(
            "head", "classification_head", ["fused"], in_features=2, num_classes=classes
        )
        .outputs(logits="head", features="fused")
        .to_config()
    )


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.random.default_generator.manual_seed(0)
    linear = (
        ModelBuilder((1, 4, 4), 2)
        .add("conv2d", in_channels=1, out_channels=2, kernel_size=1, padding=0)
        .build()
    )
    config = architecture()
    graph = GraphModelBuilder.from_config(config).build()
    model = GraphTensorOutput(graph, "logits", task="multiclass")
    features = torch.arange(64, dtype=torch.float32).reshape(4, 1, 4, 4) / 64
    targets = torch.tensor([0, 1, 0, 1])
    settings = TrainingConfig(
        epochs=1, batch_size=2, shuffle=False, optimizer="sgd", learning_rate=0.01
    )
    train, _ = settings.make_loaders(TensorDataset(features, targets))
    training = settings.prepare(model, train).fit()
    loader = DataLoader(TensorDataset(features, targets), batch_size=2, shuffle=False)
    evaluation = evaluate_classification(
        model, loader, class_names=["a", "b"], split="synthetic-demo"
    )
    benchmark = BenchmarkRunner(
        model.eval(),
        device="cpu",
        example_input=features[:2],
        warmup_iterations=1,
        measure_iterations=3,
        repeats=1,
    ).run()
    checks = run_model_checks(config, model=graph, model_id="demo-A")
    other = GraphModelBuilder.from_config(architecture(3)).build()
    architectures = compare_architectures(
        [config, architecture(3)], names=["A", "B"], exact_parameters=True
    )
    structure = compare_model_structures(
        capture_model_structure(graph), capture_model_structure(other)
    )
    context = AnalysisContext(
        workload_id="python-first-demo",
        task="classification",
        model={"id": "A"},
        dataset={"id": "synthetic-demo"},
    )
    evidence = EvidenceRecord.from_native(
        benchmark.to_dict(), feature_id="P01", scope=EvidenceScope(model_id="A")
    )
    report = EvidenceIndex([evidence]).analyze(context=context)
    advice = recommend(report)
    create_report(training, report, advice, language="en", level="technical").to_json(
        output / "report.json"
    )
    benchmark.to_json(output / "benchmark.json")
    evaluation.to_json(output / "evaluation.json")
    store = output / "experiments"
    if store.exists():
        raise FileExistsError("Use a fresh example output directory")
    # Explicit migration of this newly-created example store only; no user workspace.
    manager = ExperimentManager(store)
    manager.close()
    ExperimentManager.migrate_store(store)
    manager = ExperimentManager(store)
    try:
        experiment = manager.create_experiment("Python-first CPU example")
        record = manager.create_run(
            experiment["id"],
            "training",
            config={"architecture": config},
            provenance={"source": "synthetic-python-example"},
        )
        record.start()
        training.log_to(record)
        artifact = record.add_artifact(
            output / "report.json", kind="report", ownership="reference"
        )
        record.finish()
        history = manager.get_experiment_history(experiment["id"])
    finally:
        manager.close()
    result = dict(
        linear_shape=list(linear(features).shape),
        training=training.to_dict()["history"],
        evaluation=evaluation.to_dict()["metrics"],
        benchmark=benchmark.to_dict()["stats"]["pooled"],
        checks=checks,
        architectures=architectures,
        structure=structure.to_dict(),
        experiment_runs=len(history["runs"]),
        artifact=artifact,
        advice=advice.to_dict(),
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output)
    print(
        json.dumps(
            {
                "checks": [(c["name"], c["status"]) for c in result["checks"]["cases"]],
                "experiment_runs": result["experiment_runs"],
            }
        )
    )
