"""P26 opt-in producer evidence v1. No data reads, inference or GPU transfers.

These hashes describe exact ordered post-collation tensor content, not dataset
origin, split membership, preprocessing code or distributional equivalence.
"""
import hashlib
import json
import sys
import weakref
from collections import OrderedDict

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
_GRAPH_CHECKPOINTS = {}


def forget_checkpoint(model):
    _CHECKPOINTS.pop(id(model), None)
    _GRAPH_CHECKPOINTS.pop(id(model), None)


def _contract_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _serialized_state_digest(fingerprints):
    if fingerprints is None:
        raise ValueError("unsupported serialized graph state")
    return _contract_digest({key: value.hex() for key, value in fingerprints.items()})


def _graph_contract(model):
    """Conservative typed CPU profile; reconstruct structure, never execute forward.

    Trusts the installed closed constructors, not arbitrary subclasses, hooks,
    instance callables or caller metadata. Modes/gradients are not serialized.
    Fresh reference allocation is capped and preserves the caller CPU RNG.
    """
    from .models.graph import GraphModel, GraphModelBuilder
    from .models.output import GraphTensorOutput
    if sys.byteorder != "little":
        raise NotImplementedError("Graph evidence v2 requires little-endian CPU")
    import torch.nn.modules.module as module_runtime
    if any(value for name, value in vars(module_runtime).items()
           if name.startswith("_global_") and name.endswith("hooks")):
        raise ValueError("global module hooks unsupported")
    if type(model) is GraphTensorOutput:
        graph = model.graph
        selected = model.output_name
    elif type(model) is GraphModel:
        graph, selected = model, None
    else:
        raise ValueError("exact GraphModel/GraphTensorOutput required")
    if type(graph) is not GraphModel:
        raise ValueError("custom Graph subclasses unsupported")
    anchor = graph._buffers.get("_contract")
    if (type(anchor) is not torch.Tensor or anchor.shape != (0,)
            or anchor.device.type != "cpu" or anchor.dtype not in (torch.float32, torch.float64)):
        raise ValueError("CPU fp32/fp64 empty Graph contract required")
    config = GraphModel.to_config(graph)
    allowed = {"conv_norm_act", "conv2d", "pointwise_conv2d", "identity", "relu", "silu",
               "batch_norm", "global_avg_pool2d", "flatten", "classification_head",
               "add", "multiply", "concat", "weighted_sum"}
    if any(row["component"] not in allowed for row in config["nodes"]):
        raise ValueError("Graph component outside the v2 evidence profile")
    # Metadata-only Core admission supplies a parameter budget before allocation.
    report = GraphModelBuilder.preflight(config)
    if (not report["valid"] or type(report["parameter_estimate"]) is not int
            or not 0 <= report["parameter_estimate"] <= 1_000_000):
        raise ValueError("Graph contract reference exceeds 1M parameter budget")
    with torch.random.fork_rng(devices=[]):
        reference = GraphModelBuilder.from_config(config).build().to(dtype=anchor.dtype, device="cpu")
        if selected is not None:
            reference = GraphTensorOutput(reference, selected)

    def plain(value):
        if value is None or type(value) in (str, int, bool, float):
            return value
        if type(value) in (list, tuple, set):
            items = sorted(value) if type(value) is set else value
            return [type(value).__name__, [plain(v) for v in items]]
        if type(value) in (dict, OrderedDict) and all(type(k) is str for k in value):
            return {k: plain(v) for k, v in value.items()}
        raise ValueError("custom runtime metadata unsupported")

    seen, tensors = set(), set()
    def signature(actual, expected):
        if type(actual) is not type(expected) or id(actual) in seen:
            raise ValueError("custom or aliased Graph module unsupported")
        seen.add(id(actual))
        a, e = vars(actual), vars(expected)
        omit = {"_modules", "_parameters", "_buffers", "training"}
        attrs = plain({k: v for k, v in a.items() if k not in omit})
        if _contract_digest(attrs) != _contract_digest(plain({k: v for k, v in e.items() if k not in omit})):
            raise ValueError("runtime module behavior differs from closed constructor")
        result = dict(type=type(actual).__module__ + "." + type(actual).__qualname__, attributes=attrs)
        for field in ("_parameters", "_buffers"):
            aa, ee = a[field], e[field]
            if type(aa) is not dict or list(aa) != list(ee):
                raise ValueError("registered Graph tensor names differ")
            schema = {}
            for name, expected_tensor in ee.items():
                tensor = aa[name]
                if expected_tensor is None:
                    if tensor is not None:
                        raise ValueError("unexpected Graph tensor")
                    schema[name] = None
                    continue
                if (type(tensor) is not type(expected_tensor) or id(tensor) in tensors
                        or tensor.device.type != "cpu" or tensor.layout != torch.strided
                        or tensor.dtype != expected_tensor.dtype or tensor.shape != expected_tensor.shape
                        or tensor.requires_grad != expected_tensor.requires_grad):
                    raise ValueError("Graph tensor schema/device/dtype/alias mismatch")
                tensors.add(id(tensor))
                schema[name] = [str(tensor.dtype), list(tensor.shape), tensor.requires_grad]
            result[field] = schema
        if type(a["_modules"]) is not dict or list(a["_modules"]) != list(e["_modules"]):
            raise ValueError("Graph module topology differs")
        result["children"] = {name: signature(a["_modules"][name], child)
                              for name, child in e["_modules"].items()}
        return result

    runtime = signature(model, reference)
    state = torch.nn.Module.state_dict(model)
    return dict(format="hyso.graph_checkpoint_contract", schema_version=2,
                configuration=config, adapter_output=selected, device="cpu", dtype=str(anchor.dtype),
                torch_version=str(torch.__version__),
                runtime_sha256=_contract_digest(runtime),
                serialized_state_sha256=_serialized_state_digest(checkpoint_state_fingerprints(state)))


def validate_graph_contract(model, contract, fingerprints):
    """Check saved tensor commitment AND separately reconstructed runtime metadata."""
    actual = _graph_contract(model)
    expected = dict(actual, serialized_state_sha256=_serialized_state_digest(fingerprints))
    if type(contract) is not dict or _contract_digest(contract) != _contract_digest(expected):
        raise ValueError("Graph checkpoint contract/state mismatch")


def remember_graph_checkpoint(model, artifact_sha256, fingerprints, contract):
    try:
        validate_graph_contract(model, contract, fingerprints)
        if _serialized_state_digest(checkpoint_state_fingerprints(model.state_dict())) != contract["serialized_state_sha256"]:
            return
        state = model_hash(model)
    except (TypeError, ValueError, RuntimeError, AttributeError, KeyError, NotImplementedError):
        return
    key = id(model)
    def expired(ref):
        entry = _GRAPH_CHECKPOINTS.get(key)
        if entry is not None and entry[0] is ref:
            _GRAPH_CHECKPOINTS.pop(key, None)
    # Freeze plain metadata; never retain caller-mutable dictionaries.
    frozen = json.dumps(contract, sort_keys=True, allow_nan=False)
    _GRAPH_CHECKPOINTS[key] = (weakref.ref(model, expired), state, artifact_sha256, frozen)


def graph_checkpoint_evidence(model, state=None):
    """Separate v2 proof; never upgrades P26 v1 schema2/3 checkpoint evidence."""
    missing = dict(format="hyso.graph_checkpoint_evidence", schema_version=2, status="missing",
                   checkpoint_sha256=None, model_state_sha256=None, contract=None,
                   source=None, reason="no_matching_graph_contract_receipt")
    entry = _GRAPH_CHECKPOINTS.get(id(model))
    if entry is None or entry[0]() is not model:
        return missing
    try:
        current = model_hash(model)
        contract = _graph_contract(model)
        if (entry[1] != current or (state is not None and state != current)
                or _contract_digest(contract) != _contract_digest(json.loads(entry[3]))):
            return missing
    except (TypeError, ValueError, RuntimeError, AttributeError, KeyError, NotImplementedError):
        return missing
    return dict(format="hyso.graph_checkpoint_evidence", schema_version=2, status="native",
                checkpoint_sha256=entry[2], model_state_sha256=current, contract=contract,
                source="hyso.load_training_checkpoint.graph_contract.v2", reason=None)


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
