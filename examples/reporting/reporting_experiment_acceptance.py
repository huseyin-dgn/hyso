"""Bounded CPU Reporting/ExperimentManager acceptance using separate QA tensors."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hyso import ExperimentManager, GraphModelBuilder, evaluate_classification
from hyso.analysis import AnalysisContext, EvidenceIndex, EvidenceRecord, EvidenceScope, ModelComparisonSource, compare_models
from hyso.analysis.graph_evidence import GraphEvaluationResult, evaluate_graph_classification, compare_graph_evaluations
from hyso.checkpoint import save_training_checkpoint, load_training_checkpoint
from hyso.models.output import GraphTensorOutput
from hyso.reporting import create_report, report_experiment
from hyso.training import TrainingResult


ROOT = Path(__file__).resolve().parents[2]
CONTEXT = dict(dataset_id="stage13-5-15-small-qa", preprocessing_id="identity-fp32", split="test")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def linear(threshold=0.0, *, wrapped=False):
    model = nn.Linear(1, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.0], [1.0]]))
        model.bias.copy_(torch.tensor([0.0, -threshold]))
    return nn.Sequential(model, nn.Identity()) if wrapped else model


def evaluation(model=None, *, empty=False):
    x = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    y = torch.tensor([0, 0, 1, 1])
    if empty:
        x, y = x[:0], y[:0]
    return evaluate_classification(linear() if model is None else model,
        DataLoader(TensorDataset(x, y), batch_size=2), class_names=["a", "b"], split="test",
        capture_evidence=True, capture_structure=not empty)


def analysis_source(name, result, *, context=None, path=None):
    context = dict(CONTEXT if context is None else context)
    scope = EvidenceScope(model_id=name, **context)
    record = (EvidenceRecord.from_native(result.to_dict(), feature_id="P26", scope=scope) if path is None else
              EvidenceRecord.from_json(path, feature_id="P26", expected_sha256=sha(path), scope=scope))
    if record.to_dict()["native_payload"] != result.to_dict():
        raise ValueError("Evaluation artifact differs from its typed source")
    report = EvidenceIndex([record]).analyze(context=AnalysisContext(
        workload_id="reporting-acceptance-qa", task="classification", model={"id": name}, dataset=context))
    return ModelComparisonSource(name, report, records={"P26": record.record_id}), report


def graph_model():
    graph = (GraphModelBuilder({"image": [1, 4, 4]})
        .add("stem", "conv_norm_act", ["image"], in_channels=1, out_channels=2, norm="batch", activation="silu")
        .add("branch", "conv2d", ["stem"], in_channels=2, out_channels=2, kernel_size=1, padding=0)
        .add("merge", "add", ["stem", "branch"])
        .add("head", "classification_head", ["merge"], in_features=2, num_classes=2)
        .outputs(logits="head").build())
    return GraphTensorOutput(graph, "logits", task="multiclass").eval()


def graph_evaluation(model, *, receiver=True, context=None):
    data = DataLoader(TensorDataset(torch.arange(64).reshape(4, 1, 4, 4).float() / 64,
        torch.tensor([0, 1, 0, 1])), batch_size=2)
    return evaluate_graph_classification(model, data, class_names=["a", "b"], split="test",
        context=dict(CONTEXT if context is None else context), capture_receiver_runtime=receiver)


def saved_training():
    return TrainingResult((ROOT / "examples/reporting/fixtures/training-result-v1.json").read_text(encoding="utf8"))


def run(output):
    output = Path(output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Choose an output directory outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.random.default_generator.manual_seed(13515)
    models = {"A": linear(1.5), "B": linear(wrapped=True)}
    evaluations, sources, reports = {}, {}, {}
    for name, model in models.items():
        checkpoint = output / (name + ".pth")
        save_training_checkpoint(checkpoint, model, include_rng=False)
        load_training_checkpoint(checkpoint, model, capture_evidence=True, expected_sha256=sha(checkpoint))
        evaluations[name] = evaluation(model)
        evaluations[name].to_json(output / (name + "-evaluation.json"))
        sources[name], reports[name] = analysis_source(name, evaluations[name], path=output / (name + "-evaluation.json"))
    strict = compare_models(sources["A"], sources["B"], features="P26", include_structure=True, context_policy="strict")
    legacy = compare_models(sources["A"], sources["B"], features="P26", context_policy="legacy")
    changed, _ = analysis_source("B", evaluations["B"], context=dict(CONTEXT, preprocessing_id="different-declaration"),
                                 path=output / "B-evaluation.json")
    incompatible = compare_models(sources["A"], changed, features="P26", context_policy="strict")
    strict.to_json(output / "strict-comparison.json")
    legacy.to_json(output / "legacy-comparison.json")
    incompatible.to_json(output / "incompatible-comparison.json")
    graph = graph_model()
    checkpoint = output / "graph.pth"
    save_training_checkpoint(checkpoint, graph, include_rng=False, graph_contract=True)
    load_training_checkpoint(checkpoint, graph, capture_evidence=True, expected_sha256=sha(checkpoint))
    verified = graph_evaluation(graph)
    limited = graph_evaluation(graph, receiver=False)
    verified.to_json(output / "graph-association.json")
    imported = GraphEvaluationResult.from_json(output / "graph-association.json", expected_sha256=sha(output / "graph-association.json"))
    different = graph_evaluation(graph, context=dict(CONTEXT, preprocessing_id="different-declaration"))
    graph_strict = compare_graph_evaluations(verified, different, context_policy="strict")
    graph_strict.to_json(output / "graph-strict.json")
    training = saved_training()
    presentation = create_report(reports["A"], reports["B"], strict, legacy, incompatible,
        verified, limited, imported, graph_strict, training, language="en", level="expert",
        title="Stage 13.5.15 separate QA evidence")
    for suffix, method in (("json", presentation.to_json), ("md", presentation.to_markdown), ("html", presentation.to_html)):
        method(output / ("report." + suffix))
    write(output / "dataset-manifest.json", dict(context=CONTEXT, inputs=[[-2], [-1], [1], [2]],
        targets=[0, 0, 1, 1], classes=["a", "b"], batch_sizes=[2, 2], scope="independent synthetic QA"))
    store = output / "store"
    manager = ExperimentManager(store)
    initial_version = manager._schema_version
    manager.close()
    migration = ExperimentManager.migrate_store(store)
    manager = ExperimentManager(store)
    artifacts, snapshots, inputs = [], {}, []
    try:
        experiment = manager.create_experiment("Reporting acceptance QA")
        for name in ("A", "B"):
            current = manager.create_run(experiment["id"], "evaluation", config=dict(model_id=name, context=CONTEXT))
            current.start()
            path = output / (name + "-evaluation.json")
            item = current.add_artifact(path, kind="evaluation", ownership="copy")
            artifacts.append((item, path)); inputs.append(item["id"])
            path = output / (name + ".pth")
            artifacts.append((current.add_artifact(path, kind="checkpoint", ownership="reference", retain_locator=True), path))
            current.finish(); snapshots[current.id] = current.to_dict()
        current = manager.create_workflow_run(experiment["id"], "comparison", model_ids=["A", "B"], comparison=strict, dataset=CONTEXT)
        current.start()
        typed = current.record_result(strict, input_artifact_ids=inputs)
        assert current.record_result(strict, input_artifact_ids=inputs)["id"] == typed["id"]
        artifacts.append((typed, None))
        for name in ("report.json", "report.md", "report.html", "dataset-manifest.json", "graph-association.json", "graph-strict.json"):
            path = output / name
            artifacts.append((current.add_artifact(path, kind="report", ownership="copy"), path))
        current.finish(); snapshots[current.id] = current.to_dict()
        from hyso import TrainingConfig
        current = manager.create_workflow_run(experiment["id"], "training", model_ids=["retained-training-fixture"],
            training_config=TrainingConfig.from_dict(training.to_dict()["settings"]["requested"]))
        current.start(); trained = current.record_result(training); current.finish()
        snapshots[current.id] = current.to_dict(); artifacts.append((trained, None))
        before = [manager.verify_artifact(item["id"], path=path) for item, path in artifacts]
    finally:
        manager.close()
    reopened = ExperimentManager(store)
    try:
        assert all(reopened.get_run(identity).to_dict() == snapshot for identity, snapshot in snapshots.items())
        assert reopened.open_result(typed["id"])["result"].to_json() == strict.to_json()
        assert reopened.open_result(trained["id"])["result"].to_dict() == training.to_dict()
        after = [reopened.verify_artifact(item["id"], path=path) for item, path in artifacts]
        history = reopened.get_experiment_history(experiment["id"])
        experiment_report = report_experiment(reopened, experiment["id"], language="en", level="expert")
        for suffix, method in (("json", experiment_report.to_json), ("md", experiment_report.to_markdown), ("html", experiment_report.to_html)):
            method(output / ("experiment-report." + suffix))
    finally:
        reopened.close()
    assert all(row["status"] == "verified" for row in before + after)
    assert not torch.cuda.is_initialized()
    summary = dict(experiment_id=experiment["id"], initial_schema=initial_version, migration=migration,
        runs=snapshots, history=history, artifacts=[dict(id=a["id"], path=str(p) if p is not None else None) for a, p in artifacts],
        before_checks=before, after_checks=after, strict=strict.to_dict()["context_policy"],
        graph_verified=verified.to_dict()["verification"], limited=limited.to_dict()["relation"],
        imported=imported.to_dict()["verification"], new_training_fits=0, locked_test_inference=0)
    write(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    print(json.dumps(run(parser.parse_args().output), indent=2))
