"""One-epoch CPU Graph v2 checkpoint acceptance with explicit synthetic inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from hyso import GraphModelBuilder, TrainingConfig, evaluate_classification
from hyso.models.output import GraphTensorOutput
from hyso.checkpoint import save_training_checkpoint, load_training_checkpoint, graph_checkpoint_evidence


def run(output):
    output = Path(output).resolve()
    repo = Path(__file__).resolve().parents[2]
    if output == repo or output.is_relative_to(repo):
        raise ValueError("Use an isolated QA output outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.random.default_generator.manual_seed(1359)
    b = (GraphModelBuilder({"image": [1, 4, 4]})
        .add("stem", "conv_norm_act", ["image"], in_channels=1, out_channels=2, norm="batch")
        .add("left", "conv2d", ["stem"], in_channels=2, out_channels=2, kernel_size=1, padding=0)
        .add("merge", "add", ["left", "stem"])
        .add("head", "classification_head", ["merge"], in_features=2, num_classes=2)
        .outputs(logits="head", features="merge"))
    def fresh():
        return GraphTensorOutput(b.build(), "logits", task="multiclass").float().cpu()
    source, target = fresh(), fresh()
    inputs = torch.arange(64, dtype=torch.float32).reshape(4, 1, 4, 4) / 64
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2, shuffle=False)
    settings = TrainingConfig(epochs=1, batch_size=2, shuffle=False, validation=False,
                              device="cpu", precision="fp32", optimizer="sgd", learning_rate=0.01)
    session = settings.prepare(source, loader)
    training = session.fit()
    source.eval()
    metadata = dict(architecture=b.to_config(), seed=1359, fixture="explicit_synthetic_not_real_dataset",
                    settings=settings.to_dict())
    path = output / "graph-v2.pth"
    save_training_checkpoint(path, source, graph_contract=True, include_rng=False,
        metrics=metadata, history=training.history, completed_epochs=1)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    def check(saved):
        if saved["metrics"] != metadata:
            raise ValueError("saved metadata differs")
    load_training_checkpoint(path, target, capture_evidence=True, expected_sha256=sha, validate_metadata=check)
    target.eval()
    target.load_state_dict(source.state_dict(), strict=True)
    with torch.no_grad():
        equal = torch.equal(source(inputs), target(inputs))
    assert equal
    native = graph_checkpoint_evidence(target)
    assert native["status"] == "native"
    evaluation = evaluate_classification(target, loader, class_names=["a", "b"], split="test",
        capture_evidence=True).to_dict()
    assert evaluation["producer_evidence"]["checkpoint"]["status"] == "missing"
    with torch.no_grad():
        next(target.parameters()).add_(1)
    invalidated = graph_checkpoint_evidence(target)
    assert invalidated["status"] == "missing"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    for name, data in {
        "training.json": training.to_dict(), "receipt-v2.json": native,
        "evaluation-legacy-p26.json": evaluation, "receipt-after-mutation.json": invalidated,
        "summary.json": dict(seed=1359, device="cpu", precision="fp32", strict_equal=equal,
            checkpoint_sha256=sha, applied_updates=training.to_dict()["applied_updates"],
            native_graph_receipt=native["status"], legacy_p26_receipt="missing", mutation_receipt="missing"),
    }.items():
        with (output / name).open("x", encoding="utf8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
    return native


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output)))
