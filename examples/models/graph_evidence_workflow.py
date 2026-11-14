"""Explicit bounded CPU provenance workflow; synthetic data, no performance claim."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from hyso import GraphModelBuilder, TrainingConfig, ExperimentManager
from hyso.models.output import GraphTensorOutput
from hyso.checkpoint import save_training_checkpoint, load_training_checkpoint
from hyso.analysis.graph_evidence import (
    GraphEvaluationResult, GraphEvidenceError, evaluate_graph_classification, compare_graph_evaluations,
)
from hyso.reporting import create_report


def write(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def run(output):
    output = Path(output).resolve()
    repo = Path(__file__).resolve().parents[2]
    if output == repo or output.is_relative_to(repo):
        raise ValueError("Use fresh isolated QA outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    train_x = torch.arange(64, dtype=torch.float32).reshape(4, 1, 4, 4) / 64
    test_x = torch.arange(64, 128, dtype=torch.float32).reshape(4, 1, 4, 4) / 128
    labels = torch.tensor([0, 1, 0, 1])
    train = DataLoader(TensorDataset(train_x, labels), batch_size=2, shuffle=False)
    test = DataLoader(TensorDataset(test_x, labels), batch_size=2, shuffle=False)
    context = dict(dataset_id="synthetic-stage13-5-10", split="test", preprocessing_id="fixed_fp32_identity")
    settings = TrainingConfig(epochs=1, batch_size=2, shuffle=False, validation=False,
        device="cpu", precision="fp32", optimizer="sgd", learning_rate=0.01)
    models, associations, trainings, paths = [], [], [], []
    store = output / "experiments"
    manager = ExperimentManager(store)
    manager.close()
    ExperimentManager.migrate_store(store)
    manager = ExperimentManager(store)
    artifacts, checks, run_ids = [], [], []
    try:
        experiment = manager.create_experiment("Graph v2 source/evaluation scope")
        for index, activation in enumerate(["silu", "relu"]):
            directory = output / ("model-" + str(index))
            directory.mkdir()
            torch.random.default_generator.manual_seed(13610 + index)
            b = (GraphModelBuilder({"image": [1, 4, 4]})
                .add("stem", "conv_norm_act", ["image"], in_channels=1, out_channels=2,
                     norm="batch", activation=activation)
                .add("left", "conv2d", ["stem"], in_channels=2, out_channels=2, kernel_size=1, padding=0)
                .add("merge", "add", ["left", "stem"])
                .add("head", "classification_head", ["merge"], in_features=2, num_classes=2)
                .outputs(logits="head", features="merge"))
            def fresh():
                return GraphTensorOutput(b.build(), "logits", task="multiclass").float().cpu()
            source = fresh()
            training = settings.prepare(source, train).fit()
            source.eval()
            metadata = dict(seed=13610 + index, architecture=b.to_config(), settings=settings.to_dict(),
                fixture="explicit_synthetic_not_real_dataset", evaluation_context=context)
            cp = directory / "model.pth"
            save_training_checkpoint(cp, source, graph_contract=True, include_rng=False,
                metrics=metadata, history=training.history, completed_epochs=1)
            sha = hashlib.sha256(cp.read_bytes()).hexdigest()
            restored = fresh().eval()
            def validator(saved):
                if saved["metrics"] != metadata:
                    raise ValueError("checkpoint metadata mismatch")
            load_training_checkpoint(cp, restored, capture_evidence=True, expected_sha256=sha,
                validate_metadata=validator)
            with torch.no_grad():
                assert torch.equal(source(test_x), restored(test_x))
            result = evaluate_graph_classification(restored, test, class_names=["a", "b"], split="test",
                source_id=context["dataset_id"], context=context, expected_checkpoint_sha256=sha)
            assert result.to_dict()["verification"]["graph_v2"] == "verified"
            assert result.to_dict()["relation"]["status"] == "limited"
            assert result.evaluation.to_dict()["producer_evidence"]["checkpoint"]["status"] == "missing"
            models.append(restored)
            associations.append(result)
            trainings.append(training)
            paths.append(cp)
            write(directory / "training.json", training.to_dict())
            result.to_json(directory / "association.json")
            result.evaluation.to_json(directory / "evaluation.json")
            tr = manager.create_run(experiment["id"], "training", config=metadata)
            tr.start()
            training.log_to(tr)
            inputs = []
            for file, kind in [(cp, "checkpoint"), (directory / "training.json", "training_result")]:
                artifact = tr.add_artifact(file, kind=kind, ownership="reference")
                inputs.append(artifact)
                artifacts.append((artifact, file))
            tr.finish()
            er = manager.create_run(experiment["id"], "evaluation", config=metadata,
                provenance={"source": "native evaluation; separate source v2; limited copy binding"})
            er.start()
            for artifact in inputs:
                manager.link_artifact(er.id, artifact["id"], role="input")
            for file, kind in [(directory / "association.json", "graph_evaluation_association"),
                               (directory / "evaluation.json", "evaluation")]:
                artifacts.append((er.add_artifact(file, kind=kind, ownership="reference",
                    metadata={"receipt_scope": "retained_statement_bytes_not_process_local_trust"}), file))
            er.finish()
            run_ids.extend([tr.id, er.id])
        compared = compare_graph_evaluations(*associations)
        assert compared.to_dict()["status"] == "COMPARABLE"
        compared.to_json(output / "comparison.json")
        compared.legacy_comparison.to_json(output / "legacy-comparison.json")
        report = create_report(*trainings, *associations, compared, language="en", level="technical")
        for suffix, method in [("json", report.to_json), ("md", report.to_markdown), ("html", report.to_html)]:
            method(output / ("report." + suffix))
        rr = manager.create_run(experiment["id"], "evaluation", config=compared.legacy_comparison.tracking_config())
        rr.start()
        for artifact, file in artifacts:
            manager.link_artifact(rr.id, artifact["id"], role="input")
        for name in ["comparison.json", "legacy-comparison.json", "report.json", "report.md", "report.html"]:
            file = output / name
            artifacts.append((rr.add_artifact(file, kind="report", ownership="reference"), file))
        rr.finish()
        run_ids.append(rr.id)
        for artifact, file in artifacts:
            checks.append(manager.verify_artifact(artifact["id"], path=file))
        assert all(v["status"] == "verified" for v in checks)
        imported = GraphEvaluationResult.from_json(output / "model-0/association.json")
        assert imported.to_dict()["verification"]["graph_v2"] == "unverified"
        bad = output / "negative-artifact-copy.json"
        bad.write_text("{}", encoding="utf8")
        negative = manager.verify_artifact(artifacts[0][0]["id"], path=bad)
        assert negative["status"] == "mismatch"
        assert manager.verify_artifact(artifacts[0][0]["id"], path=artifacts[0][1])["status"] == "verified"
        try:
            evaluate_graph_classification(models[0], test, class_names=["a", "b"], split="test",
                expected_checkpoint_sha256="0" * 64)
        except GraphEvidenceError:
            wrong_identity_rejected = True
        else:
            raise AssertionError("wrong expected checkpoint identity accepted")
        with torch.no_grad():
            next(models[0].parameters()).add_(1)
        mutation = associations[0].to_dict()
        assert mutation["verification"]["graph_v2"] == "missing"
        write(output / "mutation.json", mutation)
        summary = dict(seeds=[13610, 13611], device="cpu", precision="fp32", epochs_per_model=1,
            applied_updates=[t.to_dict()["applied_updates"] for t in trainings],
            accuracy=[v.evaluation.to_dict()["metrics"]["accuracy"] for v in associations],
            comparison_status=compared.to_dict()["status"], claims=compared.legacy_comparison.to_dict()["claims"],
            legacy_p26=[v.evaluation.to_dict()["producer_evidence"]["checkpoint"]["status"] for v in associations],
            graph_v2_at_report_creation=["verified", "verified"], cross_binding="limited;isolated runtime unobserved",
            imported_json_status="unverified", mutation_status="missing", wrong_identity_rejected=wrong_identity_rejected,
            experiment_id=experiment["id"], run_ids=run_ids, artifact_verification=checks,
            negative_artifact_check=negative, checkpoint_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])
        write(output / "summary.json", summary)
        return summary
    finally:
        manager.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output)))
