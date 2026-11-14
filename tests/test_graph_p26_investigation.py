"""Short CPU diagnostics. A missing native receipt is an expected fail-closed gate."""

import copy
import hashlib

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from examples.models.image_classification import _builder
from hyso.models.output import GraphTensorOutput
from hyso.models.builder import ModelConfigError
from hyso.checkpoint import (
    TrainingCheckpointError, save_training_checkpoint, load_training_checkpoint,
)
from hyso._evaluation_evidence import model_hash, checkpoint_evidence
from hyso.evaluation import evaluate_classification
from hyso.analysis import EvidenceRecord, EvidenceScope, EvidenceError


def graph(classes=10, size=(8, 8)):
    return GraphTensorOutput(_builder("graph", 1, size, classes).build(),
                             "logits", task="multiclass")


def receipt(model):
    return checkpoint_evidence(model, model_hash(model))


def test_unmodified_graph_strict_roundtrip_still_has_no_receipt(tmp_path):
    torch.set_num_threads(1)
    original, loaded = graph().eval(), graph().eval()
    path = tmp_path / "valid.pth"
    save_training_checkpoint(path, original, include_rng=False)
    before = path.read_bytes()
    load_training_checkpoint(path, loaded, capture_evidence=True)
    assert set(dict(loaded.named_buffers())) - set(loaded.state_dict()) == {"graph._contract"}
    assert loaded.graph._non_persistent_buffers_set == {"_contract"}
    loaded.load_state_dict(original.state_dict(), strict=True)
    x = torch.arange(128, dtype=torch.float32).reshape(2, 1, 8, 8) / 128
    assert torch.equal(original(x), loaded(x))
    result = evaluate_classification(loaded, DataLoader(TensorDataset(x, torch.tensor([0, 1])), batch_size=2),
        class_names=[str(i) for i in range(10)], split="test", capture_evidence=True).to_dict()
    assert result["producer_evidence"]["checkpoint"]["status"] == "missing"
    assert result["producer_evidence"]["model_state"]["status"] == "native"
    assert result["producer_evidence"]["evaluated_stream"]["sample_count"] == 2
    assert path.read_bytes() == before


@pytest.mark.parametrize("case", ["wrong_config", "missing_parameter", "strict_failure"])
def test_invalid_checkpoint_paths_never_mint_receipts(tmp_path, case):
    original = graph()
    target = graph(classes=9) if case == "wrong_config" else graph()
    path = tmp_path / (case + ".pth")
    save_training_checkpoint(path, original, include_rng=False)
    if case == "missing_parameter":
        payload = torch.load(path, weights_only=True, map_location="cpu")
        del payload["model"]["state_dict"][next(iter(payload["model"]["state_dict"]))]
        # An explicitly labeled, local safe negative fixture, never a historical checkpoint.
        torch.save(payload, path)
    elif case == "strict_failure":
        def fail_once(*args):
            handle.remove()
            raise RuntimeError("diagnostic strict load failure")
        handle = target.register_load_state_dict_pre_hook(fail_once)
    before = model_hash(target)
    with pytest.raises(TrainingCheckpointError):
        load_training_checkpoint(path, target, capture_evidence=True)
    assert receipt(target)["status"] == "missing"
    assert model_hash(target) == before


def test_persistent_buffer_change_invalidates_native_control_receipt(tmp_path):
    # BN supplies a genuine persistent-buffer control without Graph's prior missing gate.
    model = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 2)).eval()
    path = tmp_path / "persistent.pth"
    save_training_checkpoint(path, model, include_rng=False)
    load_training_checkpoint(path, model, capture_evidence=True)
    assert receipt(model)["status"] == "native"
    with torch.no_grad():
        model[0].running_mean.add_(1)
    assert receipt(model)["status"] == "missing"
    # A trusted source hook substitution is also rejected by frozen byte fingerprints.
    target = copy.deepcopy(model)
    def substitute_buffer(module, keys):
        module[0].running_mean.add_(1)
    target.register_load_state_dict_post_hook(substitute_buffer)
    load_training_checkpoint(path, target, capture_evidence=True)
    assert receipt(target)["status"] == "missing"


def test_wrong_model_state_hash_cannot_match_live_receipt(tmp_path):
    model = nn.Linear(3, 2)
    path = tmp_path / "state.pth"
    save_training_checkpoint(path, model, include_rng=False)
    load_training_checkpoint(path, model, capture_evidence=True)
    assert receipt(model)["status"] == "native"
    assert checkpoint_evidence(model, "0" * 64)["status"] == "missing"


def test_wrong_checkpoint_file_hash_rejected_at_evidence_binding(tmp_path):
    model = nn.Linear(3, 2)
    path = tmp_path / "hash.pth"
    save_training_checkpoint(path, model, include_rng=False)
    load_training_checkpoint(path, model, capture_evidence=True)
    result = evaluate_classification(model,
        DataLoader(TensorDataset(torch.ones(2, 3), torch.tensor([0, 1])), batch_size=2),
        class_names=["a", "b"], split="test", capture_evidence=True).to_dict()
    assert result["producer_evidence"]["checkpoint"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(EvidenceError):
        EvidenceRecord.from_native(result, feature_id="P26",
            scope=EvidenceScope(checkpoint_sha256="0" * 64))


def test_nonpersistent_contract_mismatch_is_behavioral_and_unrepresented(tmp_path):
    original, target = graph(), graph()
    path = tmp_path / "contract.pth"
    save_training_checkpoint(path, original, include_rng=False)
    # Deliberate isolated mismatch: weights remain fp32; contract becomes fp64.
    target.graph._contract = torch.empty(0, dtype=torch.float64)
    load_training_checkpoint(path, target, capture_evidence=True)
    assert all(torch.equal(value, target.state_dict()[key])
               for key, value in original.state_dict().items())
    assert receipt(target)["status"] == "missing"
    assert model_hash(target) != model_hash(original)
    with pytest.raises(ModelConfigError, match="dtype mismatch"):
        target(torch.ones(2, 1, 8, 8))


def test_same_tensor_schema_different_geometry_requires_explicit_metadata_guard(tmp_path):
    original, target = graph(), graph(size=(9, 9))
    path = tmp_path / "geometry.pth"
    config = original.graph.to_config()
    save_training_checkpoint(path, original, include_rng=False, metrics={"architecture": config})
    # Generic strict state loading does not authenticate Python configuration.
    load_training_checkpoint(path, target, capture_evidence=True)
    assert receipt(target)["status"] == "missing"
    with pytest.raises(ModelConfigError):
        target(torch.ones(2, 1, 8, 8))
    def guard(metadata):
        if metadata["metrics"]["architecture"] != target.graph.to_config():
            raise ValueError("architecture metadata mismatch")
    with pytest.raises(TrainingCheckpointError, match="architecture metadata mismatch"):
        load_training_checkpoint(path, target, capture_evidence=True, validate_metadata=guard)
    assert receipt(target)["status"] == "missing"
