"""Graph v2 is separate from historical P26 v1; no unverified path mints it."""
import copy
import hashlib
import json

import pytest
import torch

from hyso import GraphModelBuilder, EvaluationResult, evaluate_classification
from hyso.models.graph import GraphModel
from hyso.models.output import GraphTensorOutput
from hyso import checkpoint as cp
from hyso._evaluation_evidence import checkpoint_evidence, model_hash


def builder(*, size=4, activation="silu", topology="add", classes=2, parameterless=False):
    b = GraphModelBuilder({"image": [1, size, size]})
    if parameterless:
        return b.add("identity", "identity", ["image"]).outputs(features="identity")
    return (b.add("stem", "conv_norm_act", ["image"], in_channels=1, out_channels=2,
                  norm="batch", activation=activation)
        .add("left", "conv2d", ["stem"], in_channels=2, out_channels=2, kernel_size=1, padding=0)
        .add("merge", topology, ["left", "stem"])
        .add("head", "classification_head", ["merge"], in_features=2, num_classes=classes)
        .outputs(logits="head", features="merge"))


def model(**kwargs):
    return GraphTensorOutput(builder(**kwargs).build(), "logits", task="multiclass")


def evidence(m):
    return cp.graph_checkpoint_evidence(m)


def save(path, m):
    cp.save_training_checkpoint(path, m, graph_contract=True, include_rng=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_versioned_receipt_exact_roundtrip_and_legacy_scope(tmp_path, dtype):
    source, target = model().to(dtype).eval(), model().to(dtype).eval()
    p = tmp_path / "valid.pth"
    before_rng = torch.get_rng_state().clone()
    save(p, source)
    assert torch.equal(before_rng, torch.get_rng_state())
    blob = p.read_bytes()
    cp.load_training_checkpoint(p, target, capture_evidence=True,
        expected_sha256=hashlib.sha256(blob).hexdigest())
    receipt = evidence(target)
    assert receipt["status"] == "native" and receipt["schema_version"] == 2
    assert receipt["checkpoint_sha256"] == hashlib.sha256(blob).hexdigest()
    assert receipt["model_state_sha256"] == model_hash(target)
    assert receipt["contract"]["dtype"] == str(dtype)
    assert receipt["contract"]["device"] == "cpu"
    assert receipt["contract"]["adapter_output"] == "logits"
    assert receipt["contract"]["configuration"] == source.graph.to_config()
    x = torch.arange(32, dtype=dtype).reshape(2, 1, 4, 4) / 32
    assert torch.equal(source(x), target(x))
    assert p.read_bytes() == blob
    # No automatic promotion or wider semantics in schema2/3 P26 v1.
    assert checkpoint_evidence(target, model_hash(target))["status"] == "missing"
    from torch.utils.data import DataLoader, TensorDataset
    data = DataLoader(TensorDataset(x, torch.tensor([0, 1])), batch_size=2)
    for structure in (False, True):
        result = evaluate_classification(target, data, class_names=["a", "b"], split="test",
            capture_evidence=True, capture_structure=structure).to_dict()
        assert result["schema_version"] == (3 if structure else 2)
        assert result["producer_evidence"]["checkpoint"]["status"] == "missing"
        assert EvaluationResult(result).to_dict() == result


@pytest.mark.parametrize("change", ["missing_parameter", "checkpoint_tensor", "persistent_buffer", "contract", "contract_scalar_type"])
def test_corrupted_bounded_payload_rejected_before_target_mutation(tmp_path, change):
    source, target = model(), model()
    p = tmp_path / "negative.pth"
    save(p, source)
    payload = torch.load(p, weights_only=True)
    state = payload["model"]["state_dict"]
    if change == "missing_parameter":
        del state[next(iter(state))]
    elif change == "checkpoint_tensor":
        state[next(iter(state))].add_(1)
    elif change == "persistent_buffer":
        state[next(k for k in state if k.endswith("running_mean"))].add_(1)
    elif change == "contract":
        payload["model"]["graph_contract"]["adapter_output"] = "features"
    else:
        payload["model"]["graph_contract"]["schema_version"] = 2.0
    torch.save(payload, p)  # Explicit new QA fixture, never an old/user artifact.
    before = model_hash(target)
    with pytest.raises(cp.TrainingCheckpointError):
        cp.load_training_checkpoint(p, target, capture_evidence=True)
    assert model_hash(target) == before and evidence(target)["status"] == "missing"


def test_wrong_expected_file_hash_and_wrong_query_state(tmp_path):
    source, target = model(), model()
    p = tmp_path / "hash.pth"
    save(p, source)
    with pytest.raises(cp.TrainingCheckpointError, match="SHA-256"):
        cp.load_training_checkpoint(p, target, capture_evidence=True, expected_sha256="0" * 64)
    assert evidence(target)["status"] == "missing"
    cp.load_training_checkpoint(p, target, capture_evidence=True)
    target.checkpoint_sha256 = "0" * 64  # Unknown attributes cannot forge a contract.
    assert evidence(target)["status"] == "missing"


@pytest.mark.parametrize("case", ["geometry", "topology", "activation", "output", "head", "dtype", "device",
                                  "extra_nonpersistent", "subclass", "nested_subclass", "hook", "instance_forward"])
def test_different_or_unverifiable_runtime_never_receives_trust(tmp_path, case):
    source = model()
    target = model(**{"size": 5} if case == "geometry" else
                   {"topology": "multiply"} if case == "topology" else
                   {"activation": "relu"} if case == "activation" else
                   {"classes": 3} if case == "head" else {})
    if case == "output":
        target.output_name = "features"
    elif case == "dtype":
        target.graph._contract = torch.empty(0, dtype=torch.float64)
    elif case == "device":
        target.graph._contract = torch.empty(0, device="meta")
    elif case == "extra_nonpersistent":
        target.graph.register_buffer("outside", torch.ones(1), persistent=False)
    elif case == "subclass":
        class Custom(GraphTensorOutput):
            pass
        target = Custom(target.graph, "logits")
    elif case == "nested_subclass":
        class CustomGraph(GraphModel):
            pass
        target.graph.__class__ = CustomGraph
    elif case == "hook":
        target.register_load_state_dict_post_hook(lambda m, keys: m.graph._contract.resize_(1))
    elif case == "instance_forward":
        target.forward = lambda x: source(x)
    p = tmp_path / "runtime.pth"
    save(p, source)
    with pytest.raises(cp.TrainingCheckpointError):
        cp.load_training_checkpoint(p, target, capture_evidence=True)
    assert evidence(target)["status"] == "missing"


@pytest.mark.parametrize("mutation", ["parameter", "persistent", "geometry", "activation", "output", "hook", "extra"])
def test_mutations_invalidate_graph_receipt(tmp_path, mutation):
    target = model()
    p = tmp_path / "mutation.pth"
    save(p, target)
    cp.load_training_checkpoint(p, target, capture_evidence=True)
    assert evidence(target)["status"] == "native"
    if mutation == "parameter":
        with torch.no_grad():
            next(target.parameters()).add_(1)
    elif mutation == "persistent":
        next(v for n, v in target.named_buffers() if n.endswith("running_mean")).add_(1)
    elif mutation == "geometry":
        config = target.graph.to_config()
        config["inputs"]["image"]["shape"] = [1, 5, 5]
        target.graph._architecture = json.dumps(config, sort_keys=True)
    elif mutation == "activation":
        target.graph.nodes["node_stem"].activation = torch.nn.ReLU()
    elif mutation == "output":
        target.output_name = "features"
    elif mutation == "hook":
        target.register_forward_hook(lambda *a: None)
    else:
        target.graph.register_buffer("unknown", torch.zeros(0), persistent=False)
    assert evidence(target)["status"] == "missing"


def test_object_identity_deepcopy_reload_and_failed_restore(tmp_path, monkeypatch):
    target = model()
    p = tmp_path / "identity.pth"
    save(p, target)
    cp.load_training_checkpoint(p, target, capture_evidence=True)
    assert evidence(target)["status"] == "native"
    assert evidence(copy.deepcopy(target))["status"] == "missing"
    assert evidence(model())["status"] == "missing"
    cp.load_training_checkpoint(p, target)
    assert evidence(target)["status"] == "missing"
    cp.load_training_checkpoint(p, target, capture_evidence=True)
    original = torch.nn.Module.load_state_dict
    calls = []
    def fail_once(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("isolated restore failure")
        return original(self, *args, **kwargs)
    # Trusted global fault injection tests transactional rollback; no model hook bypass.
    monkeypatch.setattr(torch.nn.Module, "load_state_dict", fail_once)
    before = model_hash(target)
    with pytest.raises(cp.TrainingCheckpointError, match="previous state restored"):
        cp.load_training_checkpoint(p, target, capture_evidence=True)
    assert len(calls) == 2 and model_hash(target) == before
    assert evidence(target)["status"] == "missing"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_parameterless_graph_and_legacy_checkpoint_not_upgraded(tmp_path, dtype):
    b = builder(parameterless=True)
    original, target = b.build().to(dtype), b.build().to(dtype)
    p = tmp_path / "empty.pth"
    save(p, original)
    cp.load_training_checkpoint(p, target, capture_evidence=True)
    assert evidence(target)["status"] == "native"
    assert target.state_dict() == {}
    legacy = tmp_path / "legacy.pth"
    cp.save_training_checkpoint(legacy, original, include_rng=False)
    cp.load_training_checkpoint(legacy, target, capture_evidence=True)
    assert evidence(target)["status"] == "missing"
    assert checkpoint_evidence(target, model_hash(target))["status"] == "missing"


def test_wrong_state_hash_cannot_match_graph_receipt(tmp_path):
    from hyso._evaluation_evidence import graph_checkpoint_evidence
    m = model()
    p = tmp_path / "state.pth"
    save(p, m)
    cp.load_training_checkpoint(p, m, capture_evidence=True)
    assert graph_checkpoint_evidence(m, "0" * 64)["status"] == "missing"


def test_global_hook_and_unknown_save_profile_fail_closed(tmp_path):
    m = model()
    handle = torch.nn.modules.module.register_module_forward_hook(lambda *args: None)
    try:
        with pytest.raises(cp.TrainingCheckpointError, match="global module hooks"):
            save(tmp_path / "hook.pth", m)
        assert not (tmp_path / "hook.pth").exists()
    finally:
        handle.remove()
    m.graph.register_buffer("unknown", torch.empty(0), persistent=False)
    with pytest.raises(cp.TrainingCheckpointError):
        save(tmp_path / "unknown.pth", m)
    assert evidence(m)["status"] == "missing"


def test_expected_hash_checked_before_deserialize_even_if_commitment_recomputed(tmp_path, monkeypatch):
    from hyso._evaluation_evidence import _graph_contract
    m = model()
    p = tmp_path / "rewritten.pth"
    save(p, m)
    expected = hashlib.sha256(p.read_bytes()).hexdigest()
    with torch.no_grad():
        next(m.parameters()).add_(1)
    payload = torch.load(p, weights_only=True)
    payload["model"]["state_dict"] = m.state_dict()
    payload["model"]["graph_contract"] = _graph_contract(m)
    torch.save(payload, p)
    def forbidden(*args, **kwargs):
        raise AssertionError("hash mismatch reached deserialization")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(cp.TrainingCheckpointError, match="SHA-256"):
        cp.load_training_checkpoint(p, model(), capture_evidence=True, expected_sha256=expected)


def test_receipt_is_detached_and_weakly_owned(tmp_path):
    import gc
    import weakref
    from hyso._evaluation_evidence import _GRAPH_CHECKPOINTS
    m = model()
    p = tmp_path / "weak.pth"
    save(p, m)
    cp.load_training_checkpoint(p, m, capture_evidence=True)
    receipt = evidence(m)
    receipt["contract"]["adapter_output"] = "forged"
    assert evidence(m)["contract"]["adapter_output"] == "logits"
    key, ref = id(m), weakref.ref(m)
    del m
    gc.collect()
    assert ref() is None and key not in _GRAPH_CHECKPOINTS


def test_legacy_reader_rejects_v2_without_silent_conversion(tmp_path):
    import subprocess
    source = subprocess.check_output(["git", "show",
        "d325c613caa2718ea4010bffbe7560219707843d:hyso/checkpoint.py"], text=True)
    namespace = {"__name__": "hyso._legacy_checkpoint_reader", "__package__": "hyso"}
    exec(compile(source, "<trusted-repository-checkpoint-reader>", "exec"), namespace)
    m = model()
    p = tmp_path / "v2.pth"
    save(p, m)
    blob = p.read_bytes()
    with pytest.raises(namespace["TrainingCheckpointError"], match="format_version"):
        namespace["load_training_checkpoint"](p, m)
    assert p.read_bytes() == blob


def test_reference_budget_fails_before_module_allocation(tmp_path, monkeypatch):
    m = model()
    large = (GraphModelBuilder({"image": [1024, 4, 4]})
        .add("conv", "conv2d", ["image"], in_channels=1024, out_channels=1024,
             kernel_size=1, padding=0).outputs(logits="conv"))
    m.graph._architecture = json.dumps(large.to_config(), sort_keys=True)
    def forbidden(*args):
        raise AssertionError("over-budget reference allocated")
    monkeypatch.setattr(GraphModelBuilder, "build", forbidden)
    with pytest.raises(cp.TrainingCheckpointError, match="budget"):
        save(tmp_path / "budget.pth", m)


def test_late_metadata_callback_mutation_cannot_mint_receipt(tmp_path):
    m = model()
    p = tmp_path / "callback.pth"
    save(p, m)
    cp.load_training_checkpoint(p, m, capture_evidence=True,
        validate_metadata=lambda metadata: setattr(m, "output_name", "features"))
    assert evidence(m)["status"] == "missing"


def test_parameterless_still_requires_supported_byte_order(tmp_path, monkeypatch):
    import sys
    m = builder(parameterless=True).build()
    monkeypatch.setattr(sys, "byteorder", "big")
    with pytest.raises(cp.TrainingCheckpointError, match="little-endian"):
        save(tmp_path / "endian.pth", m)
