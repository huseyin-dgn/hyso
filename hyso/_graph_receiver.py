"""Core-owned, process-local observations of the actual Graph inference receiver.

No checkpoint receipt is transferred or created here. Installed Core/PyTorch at
import is trusted; arbitrary private-state access and concurrent hostile Python
execution are outside this bounded CPU observation contract.
"""

import hashlib
import json
import uuid
import weakref

import torch
from torch import nn
from ._evaluation_evidence import _graph_contract, graph_checkpoint_evidence, model_hash
from .models import (
    blocks,
    components,
    graph_ops,
    advanced,
    sampling,
    graph,
    output,
    registry,
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _environment():
    # Freeze installed callable identity/code without allocating reference models
    # or running any forward. Include inherited Module dispatch and functional ops.
    modules = (
        nn,
        blocks,
        components,
        graph_ops,
        advanced,
        sampling,
        graph,
        output,
        registry,
        torch.nn.functional,
    )
    values, classes = {}, set()
    for module in modules:
        for name, value in vars(module).items():
            if callable(value):
                values[(module, name)] = value
            if isinstance(value, type):
                classes.update(value.__mro__[:-1])
    for cls in classes:
        for name, value in vars(cls).items():
            if (
                name in ("name", "registry_name")
                or callable(value)
                or isinstance(value, (staticmethod, classmethod, property))
            ):
                values[(cls, name)] = value
    for name in ("cat", "flatten"):
        values[(torch, name)] = getattr(torch, name)

    def defaults(value):
        if value is None or type(value) in (str, int, bool, float):
            return (type(value).__name__, value)
        if type(value) in (tuple, list):
            return (type(value).__name__, tuple(defaults(v) for v in value))
        if type(value) is dict:
            return ("dict", tuple((defaults(k), defaults(v)) for k, v in value.items()))
        return ("object", id(value))

    result = {}
    for key, value in values.items():
        fn = value.__func__ if isinstance(value, (staticmethod, classmethod)) else value
        if isinstance(fn, property):
            fn = fn.fget
        result[key] = (
            value,
            getattr(fn, "__code__", None),
            defaults(getattr(fn, "__defaults__", None)),
            defaults(getattr(fn, "__kwdefaults__", None)),
        )
    return result


_INSTALLED = _environment()
_INSTALLED_TYPES = frozenset(
    value[0]
    for value in _INSTALLED.values()
    if isinstance(value[0], type) and issubclass(value[0], nn.Module)
)
_CONSTRUCTOR_TABLE = graph_ops.CONSTRUCTORS
_CONSTRUCTORS = dict(_CONSTRUCTOR_TABLE)
_REGISTRY_SPECS = registry._SPECS
_RESULTS = {}
_PHASES = [
    "after_deepcopy_eval",
    "before_each_forward",
    "after_each_forward",
    "before_publication",
]
_SCOPE = "supported_graph_v2_cpu_quiescent_core_observed_receiver"


def _guard():
    if (
        registry._SPECS is not _REGISTRY_SPECS
        or graph_ops.CONSTRUCTORS is not _CONSTRUCTOR_TABLE
        or _CONSTRUCTOR_TABLE.keys() != _CONSTRUCTORS.keys()
        or any(
            _CONSTRUCTOR_TABLE[key] is not value for key, value in _CONSTRUCTORS.items()
        )
    ):
        raise ValueError("Graph receiver installed constructor table changed")
    current = _environment()
    if current.keys() != _INSTALLED.keys() or any(
        current[key][0] is not value[0]
        or current[key][1] is not value[1]
        or current[key][2:] != value[2:]
        for key, value in _INSTALLED.items()
    ):
        raise ValueError("Graph receiver installed callable changed")


def _typed(model):
    pending, seen = [model], set()
    while pending:
        current = pending.pop()
        if (
            type(current) not in _INSTALLED_TYPES
            or id(current) in seen
            or len(seen) >= 10000
        ):
            raise ValueError("Graph receiver unsupported module identity/type")
        seen.add(id(current))
        children = vars(current).get("_modules")
        if type(children) is not dict:
            raise ValueError("Graph receiver unsupported module registration")
        pending.extend(children.values())


def _storages(model):
    result = []
    for tensor in [*model.parameters(), *model.buffers()]:
        if tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise ValueError("Graph receiver requires dense CPU storage")
        storage = tensor.untyped_storage()
        size = tensor.numel() * tensor.element_size()
        if (
            not tensor.is_contiguous()
            or tensor.storage_offset() != 0
            or storage.nbytes() != size
        ):
            raise ValueError("Graph receiver shared or view storage unsupported")
        if size:
            pointer = storage.data_ptr()
            result.append((pointer, pointer + size))
    result.sort()
    if any(left[1] > right[0] for left, right in zip(result, result[1:])):
        raise ValueError("Graph receiver overlapping storage unsupported")
    return result


class _ReceiverCapture:
    """Created and advanced only by the existing evaluator's opt-in branch."""

    def __init__(self, source):
        _guard()
        _typed(source)
        receipt = graph_checkpoint_evidence(source)
        if receipt["status"] != "native":
            raise ValueError(
                "Graph receiver requires a live source v2 checkpoint receipt"
            )
        _storages(source)
        self.source = weakref.ref(source)
        self.receipt = _json(receipt)
        self.contract = _json(receipt["contract"])
        self.state = receipt["model_state_sha256"]
        self.source_token = uuid.uuid4().hex
        self.receiver_token = uuid.uuid4().hex
        self.capture_id = uuid.uuid4().hex
        self.receiver = None
        self.batches = 0
        self.samples = 0
        self.pending = None
        self.finished = False

    def start(self, receiver):
        if self.receiver is not None:
            raise ValueError("Graph receiver already selected")
        self.receiver = receiver
        self.check(receiver)

    def check(self, receiver):
        _guard()
        source = self.source()
        if source is None or receiver is not self.receiver or receiver is source:
            raise ValueError("Graph receiver exact object identity mismatch")
        _typed(source)
        _typed(receiver)
        if _json(graph_checkpoint_evidence(source)) != self.receipt:
            raise ValueError("Graph receiver source receipt changed")
        if (
            _json(_graph_contract(receiver)) != self.contract
            or model_hash(receiver) != self.state
            or any(module.training for module in receiver.modules())
        ):
            raise ValueError("Graph receiver runtime/state changed")
        a, b = _storages(source), _storages(receiver)
        if any(x[0] < y[1] and y[0] < x[1] for x in a for y in b):
            raise ValueError("Graph receiver source storage alias")
        objects = {
            id(v) for v in [*source.modules(), *source.parameters(), *source.buffers()]
        }
        if any(
            id(v) in objects
            for v in [*receiver.modules(), *receiver.parameters(), *receiver.buffers()]
        ):
            raise ValueError("Graph receiver source object alias")

    def before(self, receiver, inputs, labels):
        if self.finished or self.pending is not None:
            raise ValueError("Graph receiver invalid forward lifecycle")
        if type(inputs) is not torch.Tensor or type(labels) is not torch.Tensor:
            raise ValueError("Graph receiver requires exact Tensor batches")
        self.check(receiver)
        self.pending = inputs.shape[0]

    def after(self, receiver, logits):
        if self.pending is None or type(logits) is not torch.Tensor:
            raise ValueError("Graph receiver incomplete or custom forward output")
        self.check(receiver)
        self.batches += 1
        self.samples += self.pending
        self.pending = None

    def finish(self, receiver, result):
        if self.finished or self.pending is not None or self.batches < 1:
            raise ValueError("Graph receiver incomplete inference")
        self.check(receiver)
        data = result.to_dict()
        stream = data["producer_evidence"]["evaluated_stream"]
        if (
            data["source"]["model_state_sha256"] != self.state
            or len(data["accounting"]["batch_sizes"]) != self.batches
            or data["accounting"]["expected_samples"] != self.samples
            or stream["sample_count"] != self.samples
        ):
            raise ValueError("Graph receiver evaluation execution mismatch")
        contract = json.loads(self.contract)
        statement = dict(
            format="hyso.graph_receiver_binding",
            schema_version=1,
            status="native",
            scope=_SCOPE,
            capture_id=self.capture_id,
            source_object_token=self.source_token,
            receiver_object_token=self.receiver_token,
            evaluation_id=data["evaluation_id"],
            evaluation_sha256=_digest(data),
            checkpoint_sha256=json.loads(self.receipt)["checkpoint_sha256"],
            source_model_state_sha256=self.state,
            receiver_model_state_sha256=self.state,
            receiver_contract=contract,
            pre_contract_sha256=_digest(contract),
            post_contract_sha256=_digest(contract),
            forward_batches=self.batches,
            sample_count=self.samples,
            evaluated_stream=stream,
            observed_phases=list(_PHASES),
            storage_isolation="distinct_objects_and_nonoverlapping_owned_cpu_storage",
        )
        validate_statement(statement, data, json.loads(self.receipt))
        self.finished = True
        self.statement = _json(statement)
        key = id(result)

        def expired(ref):
            entry = _RESULTS.get(key)
            if entry is not None and entry[0] is ref:
                _RESULTS.pop(key, None)

        # The result owns its isolated receiver's lifetime, never the source.
        # The weak result key prevents a cycle; JSON/copies cannot inherit it.
        _RESULTS[key] = (weakref.ref(result, expired), result._payload, self)


def validate_statement(statement, evaluation, source_receipt):
    """Structural consistency of retained bytes; never issues live authority."""
    keys = {
        "format",
        "schema_version",
        "status",
        "scope",
        "capture_id",
        "source_object_token",
        "receiver_object_token",
        "evaluation_id",
        "evaluation_sha256",
        "checkpoint_sha256",
        "source_model_state_sha256",
        "receiver_model_state_sha256",
        "receiver_contract",
        "pre_contract_sha256",
        "post_contract_sha256",
        "forward_batches",
        "sample_count",
        "evaluated_stream",
        "observed_phases",
        "storage_isolation",
    }
    if type(statement) is not dict or set(statement) != keys:
        raise ValueError("invalid Graph receiver statement")
    d, e, r = statement, evaluation, source_receipt
    if (
        d["format"] != "hyso.graph_receiver_binding"
        or type(d["schema_version"]) is not int
        or d["schema_version"] != 1
        or d["status"] != "native"
        or d["scope"] != _SCOPE
        or r["status"] != "native"
        or e["schema_version"] not in (2, 3)
        or e["source"]["kind"] != "model_inference"
        or e["source"]["device"] != "cpu"
        or d["evaluation_id"] != e["evaluation_id"]
        or d["evaluation_sha256"] != _digest(e)
        or d["checkpoint_sha256"] != r["checkpoint_sha256"]
        or d["source_model_state_sha256"] != r["model_state_sha256"]
        or d["receiver_model_state_sha256"] != e["source"]["model_state_sha256"]
        or d["source_model_state_sha256"] != d["receiver_model_state_sha256"]
        or _json(d["receiver_contract"]) != _json(r["contract"])
        or d["pre_contract_sha256"] != _digest(r["contract"])
        or d["post_contract_sha256"] != d["pre_contract_sha256"]
        or type(d["forward_batches"]) is not int
        or d["forward_batches"] < 1
        or d["forward_batches"] != len(e["accounting"]["batch_sizes"])
        or type(d["sample_count"]) is not int
        or d["sample_count"] < 1
        or d["sample_count"] != e["accounting"]["expected_samples"]
        or _json(d["evaluated_stream"])
        != _json(e["producer_evidence"]["evaluated_stream"])
        or d["observed_phases"] != _PHASES
        or d["storage_isolation"]
        != "distinct_objects_and_nonoverlapping_owned_cpu_storage"
    ):
        raise ValueError("Graph receiver statement/evaluation mismatch")
    for name in ("capture_id", "source_object_token", "receiver_object_token"):
        value = d[name]
        if (
            type(value) is not str
            or len(value) != 32
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("invalid Graph receiver opaque identity")
    if (
        len(
            {
                d[k]
                for k in ("capture_id", "source_object_token", "receiver_object_token")
            }
        )
        != 3
    ):
        raise ValueError("Graph receiver identity tokens overlap")


def _entry(result):
    entry = _RESULTS.get(id(result))
    if entry is None or entry[0]() is not result or entry[1] != result._payload:
        return None
    return entry[2]


def receiver_statement(result):
    capture = _entry(result)
    if capture is None:
        raise ValueError("no native Graph receiver observation on this result object")
    return json.loads(capture.statement)


def receiver_verification(result, statement):
    capture = _entry(result)
    if capture is None or capture.statement != _json(statement):
        return "unverified"
    try:
        capture.check(capture.receiver)
    except (
        ValueError,
        TypeError,
        RuntimeError,
        AttributeError,
        KeyError,
        NotImplementedError,
    ):
        return "missing"
    return "verified"
