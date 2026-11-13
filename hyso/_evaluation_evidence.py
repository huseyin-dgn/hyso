"""P26 opt-in producer evidence v1. No data reads, inference or GPU transfers.

These hashes describe exact ordered post-collation tensor content, not dataset
origin, split membership, preprocessing code or distributional equivalence.
"""
import hashlib
import json
import sys
import weakref

import torch


STREAM_METHOD = "sha256_ordered_tensor_samples_le_v1"
STREAM_SCOPE = "ordered_post_collation_inputs_and_targets"
STATE_METHOD = "sha256_named_parameters_and_buffers_v1"
CHECKPOINT_METHOD = "hyso.load_training_checkpoint.snapshot_sha256.v1"


def model_hash(model):
    # Keep the historical P26 bytes/ordering exactly, including JSON spacing.
    digest = hashlib.sha256()
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.device.type != "cpu" or tensor.layout != torch.strided or tensor.is_complex():
            raise NotImplementedError("dense real CPU model tensors required")
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


# Process-local receipts belong to the exact model object, never its attributes,
# a caller-provided path/hash, a deepcopy, or a deserialized receipt dictionary.
_CHECKPOINTS = {}


def forget_checkpoint(model):
    _CHECKPOINTS.pop(id(model), None)


def checkpoint_state_fingerprints(state):
    """Freeze expected tensor bytes before any target load hooks can mutate them."""
    try:
        if any(type(value) is not torch.Tensor for value in state.values()):
            return None
        return {key: _tensor_sample(value) for key, value in state.items()}
    except (TypeError, ValueError, RuntimeError, NotImplementedError):
        return None


def remember_checkpoint(model, artifact_sha256, loaded_fingerprints):
    """Called only after a successful trusted restore of the hashed snapshot.

    Unsupported/custom state or non-exact load hooks cannot mint a receipt.
    Evidence capture must not make an otherwise supported restore fail.
    """
    forget_checkpoint(model)
    if loaded_fingerprints is None:
        return
    try:
        current = model.state_dict()
        if set(current) != set(loaded_fingerprints):
            return
        named = dict([*model.named_parameters(), *model.named_buffers()])
        # Nonpersistent buffers and custom state_dict substitutions do not
        # establish an artifact link for the full P26 named-tensor state.
        if not set(named).issubset(loaded_fingerprints):
            return
        for key, actual in current.items():
            if _tensor_sample(actual) != loaded_fingerprints[key]:
                return
        for key, actual in named.items():
            if _tensor_sample(actual.detach()) != loaded_fingerprints[key]:
                return
        state = model_hash(model)
    except (TypeError, ValueError, RuntimeError, NotImplementedError):
        return
    key = id(model)
    def expired(ref):
        entry = _CHECKPOINTS.get(key)
        if entry is not None and entry[0] is ref:
            _CHECKPOINTS.pop(key, None)
    _CHECKPOINTS[key] = (weakref.ref(model, expired), state, artifact_sha256)


def checkpoint_evidence(model, state):
    entry = _CHECKPOINTS.get(id(model))
    if entry is not None and entry[0]() is model and entry[1] == state:
        return dict(status="native", sha256=entry[2], model_state_sha256=state,
                    source=CHECKPOINT_METHOD, reason=None)
    return dict(status="missing", sha256=None, model_state_sha256=None,
                source=None, reason="no_matching_trusted_load_receipt")


def _tensor_sample(tensor):
    if (type(tensor) is not torch.Tensor or tensor.device.type != "cpu"
            or tensor.layout != torch.strided or sys.byteorder != "little"):
        raise NotImplementedError("evidence v1 requires dense little-endian CPU tensors")
    header = json.dumps([str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode("utf-8")
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
    digest = hashlib.sha256(b"hyso.p26.tensor_sample.v1\0")
    digest.update(len(header).to_bytes(8, "big")); digest.update(header)
    digest.update(raw.nbytes.to_bytes(8, "big")); digest.update(memoryview(raw))
    return digest.digest()


class EvaluationStreamCapture:
    """O(1) hash state plus one contiguous sample; batch boundaries excluded."""

    def __init__(self):
        self.count = 0
        self.inputs = hashlib.sha256(b"hyso.p26.inputs.v1\0")
        self.targets = hashlib.sha256(b"hyso.p26.targets.v1\0")
        self.pairs = hashlib.sha256(b"hyso.p26.pairs.v1\0")

    def update(self, inputs, labels):
        for index in range(inputs.shape[0]):
            x, y = _tensor_sample(inputs[index]), _tensor_sample(labels[index])
            self.inputs.update(x); self.targets.update(y)
            self.pairs.update(x); self.pairs.update(y)
            self.count += 1

    def finish(self):
        def final(hasher):
            result = hasher.copy()
            result.update(self.count.to_bytes(8, "big"))
            return result.hexdigest()
        return dict(status="native", method=STREAM_METHOD, scope=STREAM_SCOPE,
                    source="evaluate_classification.validated_batches_before_forward",
                    sample_count=self.count, batch_partition_invariant=True,
                    inputs_sha256=final(self.inputs), targets_sha256=final(self.targets),
                    paired_sha256=final(self.pairs))


def producer_evidence(model, state, stream, config, source_id):
    return dict(format="hyso.evaluation.producer_evidence", schema_version=1,
                model_state=dict(status="native", sha256=state, method=STATE_METHOD,
                                 source="evaluate_classification.original_and_isolated_model"),
                checkpoint=checkpoint_evidence(model, state), evaluated_stream=stream.finish(),
                split=dict(status="declared", value=config["split"], source="caller.split"),
                source_id=dict(status="declared" if source_id is not None else "missing",
                               value=source_id, source="caller.source_id" if source_id is not None else None),
                preprocessing=dict(status="missing", reason="transform_chain_not_verified"),
                dataset_origin=dict(status="missing", reason="dataset_origin_not_verified"))


def validate_producer_evidence(evidence, source, config, sample_count):
    """Validate schema/consistency, not authenticity of an imported statement."""
    def require(condition):
        if not condition:
            raise ValueError("invalid P26 producer evidence")
    def sha(value):
        return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    require(type(evidence) is dict and set(evidence) == {
        "format", "schema_version", "model_state", "checkpoint", "evaluated_stream",
        "split", "source_id", "preprocessing", "dataset_origin"})
    require(evidence["format"] == "hyso.evaluation.producer_evidence"
            and type(evidence["schema_version"]) is int and evidence["schema_version"] == 1
            and source["kind"] == "model_inference")
    require(evidence["model_state"] == dict(status="native", sha256=source["model_state_sha256"],
            method=STATE_METHOD, source="evaluate_classification.original_and_isolated_model"))
    checkpoint = evidence["checkpoint"]
    require(type(checkpoint) is dict and set(checkpoint) == {
        "status", "sha256", "model_state_sha256", "source", "reason"})
    if checkpoint["status"] == "native":
        require(sha(checkpoint["sha256"]) and checkpoint["model_state_sha256"] == source["model_state_sha256"]
                and checkpoint["source"] == CHECKPOINT_METHOD and checkpoint["reason"] is None)
    else:
        require(checkpoint == dict(status="missing", sha256=None, model_state_sha256=None,
                                  source=None, reason="no_matching_trusted_load_receipt"))
    stream = evidence["evaluated_stream"]
    require(type(stream) is dict and set(stream) == {
        "status", "method", "scope", "source", "sample_count", "batch_partition_invariant",
        "inputs_sha256", "targets_sha256", "paired_sha256"})
    require(stream["status"] == "native" and stream["method"] == STREAM_METHOD
            and stream["scope"] == STREAM_SCOPE and stream["batch_partition_invariant"] is True
            and stream["source"] == "evaluate_classification.validated_batches_before_forward"
            and type(stream["sample_count"]) is int and stream["sample_count"] == sample_count
            and all(sha(stream[key]) for key in ("inputs_sha256", "targets_sha256", "paired_sha256")))
    require(evidence["split"] == dict(status="declared", value=config["split"], source="caller.split"))
    sid = source["source_id"]
    require(evidence["source_id"] == dict(status="declared" if sid is not None else "missing", value=sid,
                                         source="caller.source_id" if sid is not None else None))
    require(evidence["preprocessing"] == dict(status="missing", reason="transform_chain_not_verified"))
    require(evidence["dataset_origin"] == dict(status="missing", reason="dataset_origin_not_verified"))
