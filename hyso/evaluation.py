"""P26 finite CPU classification evaluation, independent of training artifacts."""
from dataclasses import dataclass
import copy
import hashlib
import json
import platform
import uuid

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, SequentialSampler, BatchSampler

from .benchmark.result import _plain, _csv_text, _write_export
from .checkpoint import _rng_state, _restore_rng
from ._evaluation_evidence import (model_hash, EvaluationStreamCapture, producer_evidence,
                                   validate_producer_evidence)
from ._evaluation_structure import build_structure_binding, validate_structure_binding

__all__ = ["EvaluationResult", "evaluate_predictions", "evaluate_classification", "evaluation_capabilities"]


def evaluation_capabilities():
    return {"binary": "supported", "multiclass": "supported",
            "detection": "unsupported", "segmentation": "unsupported"}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _metadata(task, class_names, split, zero_division):
    if task not in ("binary", "multiclass"):
        raise NotImplementedError("only binary/multiclass classification is supported")
    if (type(class_names) not in (list, tuple) or len(class_names) < 2
            or len(class_names) > 1000 or any(type(x) is not str or not x.strip() for x in class_names)
            or len(set(class_names)) != len(class_names) or (task == "binary" and len(class_names) != 2)):
        raise ValueError("class_names must define distinct ordered class indices 0..C-1")
    if type(split) is not str or not split.strip():
        raise ValueError("an explicit split name is required")
    if type(zero_division) is not int or zero_division not in (0, 1):
        raise ValueError("zero_division must be integer 0 or 1")
    return dict(task=task, class_names=list(class_names), split=split, zero_division=zero_division,
                binary_rule="logit >= 0", multiclass_rule="argmax_first_index")


def _labels(values, classes):
    if type(values) is not list or any(type(v) is not int or not 0 <= v < classes for v in values):
        raise ValueError("labels must be finite integer class indices")
    return values


def _predicted(raw, config):
    classes = len(config["class_names"])
    targets = _labels(raw["targets"], classes)
    if len(targets) > 1000000:
        raise ValueError("sample budget exceeded")
    if raw["prediction_type"] == "labels":
        predictions = _labels(raw["predictions"], classes)
    elif raw["prediction_type"] == "logits":
        def numeric(value):
            return all(numeric(v) for v in value) if type(value) is list else type(value) in (int, float)
        if type(raw["predictions"]) is not list or not numeric(raw["predictions"]):
            raise ValueError("logits must contain real numeric values, not booleans")
        values = torch.tensor(raw["predictions"], dtype=torch.float64)
        n = len(targets)
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite logits")
        if config["task"] == "binary":
            if tuple(values.shape) not in ((n,), (n, 1)):
                raise ValueError("binary logits require shape (N,) or (N,1)")
            predictions = (values.reshape(n) >= 0).long().tolist()
        else:
            if n == 0 and raw["predictions"] == []:
                return []
            if tuple(values.shape) != (n, classes):
                raise ValueError("multiclass logits require shape (N,C) matching class map")
            predictions = values.argmax(1).tolist()
    else:
        raise ValueError("prediction_type must be labels or logits")
    if len(targets) != len(predictions):
        raise ValueError("target/prediction count mismatch")
    return predictions


def _metrics(raw, config):
    predictions = _predicted(raw, config)
    classes, n = len(config["class_names"]), len(predictions)
    cm = [[0] * classes for _ in range(classes)]
    for true, pred in zip(raw["targets"], predictions):
        cm[true][pred] += 1
    def ratio(a, b):
        return a / b if b else config["zero_division"]
    rows = []
    for c in range(classes):
        tp = cm[c][c]
        support = sum(cm[c])
        predicted = sum(row[c] for row in cm)
        rows.append(dict(class_index=c, class_name=config["class_names"][c], support=support,
            precision=ratio(tp, predicted) if n else None,
            recall=ratio(tp, support) if n else None,
            f1=ratio(2*tp, support+predicted) if n else None))
    averages = {}
    for kind in ("macro", "micro", "weighted"):
        row = dict(support=n)
        for metric in ("precision", "recall", "f1"):
            row[metric] = (None if not n else
                sum(cm[i][i] for i in range(classes))/n if kind == "micro" else
                sum(r[metric] for r in rows)/classes if kind == "macro" else
                sum(r[metric]*r["support"] for r in rows)/n)
        averages[kind] = row
    return dict(sample_count=n, confusion_matrix=cm, accuracy=sum(cm[i][i] for i in range(classes))/n if n else None,
                per_class=rows, averages=averages)


@dataclass(frozen=True, init=False)
class EvaluationResult:
    __slots__ = ("_payload",)
    _payload: str

    def __init__(self, data):
        data = _plain(data)
        self._check(data)
        object.__setattr__(self, "_payload", json.dumps(data, allow_nan=False))

    @staticmethod
    def _check(data):
        try:
            version = data.get("schema_version") if type(data) is dict else None
            fields = {"schema_version", "format", "feature_ids", "evaluation_id",
                    "configuration", "raw", "source", "provenance", "accounting", "training_source", "status", "metrics", "source_sha256"}
            if version in (2, 3):
                fields.add("producer_evidence")
            if version == 3:
                fields.add("structure_binding")
            if (type(data) is not dict or set(data) != fields
                    or type(version) is not int or version not in (1, 2, 3)
                    or data["format"] != "hyso.evaluation.classification" or data["feature_ids"] != ["P26"]
                    or str(uuid.UUID(data["evaluation_id"])) != data["evaluation_id"] or data["training_source"] is not None):
                raise ValueError("invalid evaluation schema")
            c = data["configuration"]
            if c != _metadata(c["task"], c["class_names"], c["split"], c["zero_division"]):
                raise ValueError("invalid evaluation configuration")
            if set(data["raw"]) != {"targets", "predictions", "prediction_type"}:
                raise ValueError("invalid raw evidence")
            derived = _metrics(data["raw"], c)
            if _hash(data["metrics"]) != _hash(derived) or data["status"] != ("complete" if derived["sample_count"] else "empty"):
                raise ValueError("evaluation metrics/evidence mismatch")
            a = data["accounting"]
            if set(a) != {"expected_samples", "batch_sizes"} or type(a["expected_samples"]) is not int or a["expected_samples"] != derived["sample_count"]:
                raise ValueError("incomplete sample accounting")
            if (type(a["batch_sizes"]) is not list or any(type(n) is not int or n < 1 for n in a["batch_sizes"])
                    or sum(a["batch_sizes"]) != a["expected_samples"]):
                raise ValueError("invalid batch accounting")
            if type(data["source"]) is not dict or type(data["provenance"]) is not dict:
                raise ValueError("invalid source/provenance")
            source = data["source"]
            fields = {"kind", "source_id"}
            if source["kind"] == "model_inference":
                fields |= {"model_type", "model_state_sha256", "device", "execution", "dataset_type", "sampler", "num_workers"}
                if (source["device"] != "cpu" or source["sampler"] != "sequential" or type(source["num_workers"]) is not int
                        or source["num_workers"] != 0 or source["execution"] != "isolated_deepcopy_eval_inference_mode"
                        or len(source["model_state_sha256"]) != 64
                        or any(c not in "0123456789abcdef" for c in source["model_state_sha256"])):
                    raise ValueError("invalid model evaluation conditions")
            elif source["kind"] != "supplied_predictions":
                raise ValueError("invalid evaluation source kind")
            if set(source) != fields or (source["source_id"] is not None and (type(source["source_id"]) is not str or not source["source_id"].strip())):
                raise ValueError("invalid source identity")
            if set(data["provenance"]) != {"python", "torch", "platform"} or any(type(v) is not str or not v for v in data["provenance"].values()):
                raise ValueError("invalid evaluation provenance")
            if version in (2, 3):
                validate_producer_evidence(data["producer_evidence"], source, c, derived["sample_count"])
            if version == 3:
                validate_structure_binding(data['structure_binding'], evaluation_id=data['evaluation_id'],
                    source=source, evidence=data['producer_evidence'], accounting=a, provenance=data['provenance'])
            if data["source_sha256"] != _hash({k:v for k,v in data.items() if k not in ("source_sha256", "metrics", "status")}):
                raise ValueError("evaluation source hash mismatch")
        except (TypeError, KeyError, RuntimeError, OverflowError) as exc:
            raise ValueError("malformed evaluation evidence") from exc

    def to_dict(self):
        return json.loads(self._payload)

    @property
    def structure_snapshot(self):
        """Validated structure of the isolated receiver, or None for schema 1/2."""
        data = self.to_dict(); self._check(data)
        if data['schema_version'] != 3:
            return None
        from .structure import ModelStructureSnapshot
        return ModelStructureSnapshot(data['structure_binding']['snapshot'])

    def to_json(self, path, *, overwrite=False):
        data = self.to_dict(); self._check(data)
        _write_export(path, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n", overwrite)

    def to_csv(self, path, *, overwrite=False):
        d = self.to_dict(); self._check(d)
        rows = []
        for kind, row in [("class", r) for r in d["metrics"]["per_class"]] + list(d["metrics"]["averages"].items()):
            rows.append(dict(evaluation_id=d["evaluation_id"], source_sha256=d["source_sha256"],
                split=d["configuration"]["split"], kind=kind, class_index=row.get("class_index"),
                class_name=row.get("class_name"), accuracy=d["metrics"]["accuracy"], **{k:row[k] for k in ("precision","recall","f1","support")}))
        for true, counts in enumerate(d["metrics"]["confusion_matrix"]):
            for pred, count in enumerate(counts):
                rows.append(dict(kind="confusion", class_index=true, class_name=d["configuration"]["class_names"][true],
                    predicted_class_index=pred, predicted_class_name=d["configuration"]["class_names"][pred], count=count))
        metadata = dict(evaluation_id=d["evaluation_id"],source_sha256=d["source_sha256"],
            source_id=d["source"]["source_id"],split=d["configuration"]["split"],task=d["configuration"]["task"],
            zero_division=d["configuration"]["zero_division"],source_kind=d["source"]["kind"],
            model_state_sha256=d["source"].get("model_state_sha256"),**d["provenance"])
        columns = [*metadata,"kind","class_index","class_name","predicted_class_index","predicted_class_name",
                   "count","accuracy","precision","recall","f1","support"]
        _write_export(path, _csv_text(columns,[{k:dict(row,**metadata).get(k) for k in columns} for row in rows]), overwrite)


def _result(raw, config, source, accounting, evidence=None, structure=None):
    metrics = _metrics(raw, config)
    data = dict(schema_version=1, format="hyso.evaluation.classification", feature_ids=["P26"],
        evaluation_id=str(uuid.uuid4()), configuration=config, raw=raw, source=source,
        provenance=dict(python=platform.python_version(), torch=str(torch.__version__), platform=platform.platform()),
        accounting=accounting, training_source=None)
    if evidence is not None:
        data.update(schema_version=2, producer_evidence=evidence)
    if structure is not None:
        data.update(schema_version=3, structure_binding=build_structure_binding(structure,
            evaluation_id=data['evaluation_id'], source=source, evidence=evidence,
            accounting=accounting, provenance=data['provenance']))
    return EvaluationResult(dict(data, source_sha256=_hash(data), metrics=metrics,
        status="complete" if metrics["sample_count"] else "empty"))


def evaluate_predictions(targets, predictions, *, class_names, split, task="multiclass",
                         prediction_type="labels", zero_division=0, source_id=None):
    """Evaluate hard labels or finite logits; class_names order defines 0..C-1.

    Empty evaluation has null metrics and status empty. For nonempty data,
    absent classes participate in macro; zero denominators use 0 or 1 explicitly.
    """
    config = _metadata(task, class_names, split, zero_division)
    if source_id is not None and (type(source_id) is not str or not source_id.strip()):
        raise ValueError("invalid source_id")
    def plain(values):
        if isinstance(values, torch.Tensor):
            if values.device.type != "cpu":
                raise NotImplementedError("CPU evidence required")
            return values.detach().tolist()
        return _plain(values)
    if prediction_type == "logits" and isinstance(predictions,torch.Tensor) and not predictions.is_floating_point():
        raise ValueError("floating tensor logits required")
    raw = dict(targets=plain(targets), predictions=plain(predictions), prediction_type=prediction_type)
    if task == "binary" and type(raw["targets"]) is list:
        # Trainer-compatible floating hard labels; no soft labels or booleans.
        raw["targets"] = [int(v) if type(v) in (int,float) and v in (0,1) else v for v in raw["targets"]]
    n = len(_labels(raw["targets"], len(class_names)))
    return _result(raw, config, dict(kind="supplied_predictions", source_id=source_id),
                   dict(expected_samples=n, batch_sizes=[n] if n else []))


def _model_hash(model):
    return model_hash(model)


def evaluate_classification(model, loader, *, class_names, split, task="multiclass", zero_division=0,
                            source_id=None, max_samples=1000000, capture_evidence=False, capture_structure=False):
    """Opt-in CPU inference on an isolated deepcopy; never trains or moves model.

    Finite standard sequential map-style loaders, num_workers=0, no drop-last.
    RNG and loader generator are restored on success and every failure path.
    User-defined non-tensor Python state/external side effects are not sandboxed.
    capture_evidence=True emits schema 2 producer evidence v1 using this same
    iteration. Hashes identify ordered post-collation tensors independently of
    batch partitioning; they do not verify origin, split or preprocessing code.
    The default retains schema 1 and does not hash dataset content.
    capture_structure=True additionally requires capture_evidence=True and
    nonempty evaluation. Schema 3 binds matching pre/post structure snapshots
    of the actual isolated forward receiver, without extra inference or any
    assertion of original/copy behavior or architecture equivalence.
    """
    config = _metadata(task, class_names, split, zero_division)
    if type(capture_evidence) is not bool:
        raise ValueError("capture_evidence must be bool")
    if type(capture_structure) is not bool:
        raise ValueError("capture_structure must be bool")
    if capture_structure and not capture_evidence:
        raise ValueError("capture_structure requires capture_evidence=True")
    if type(max_samples) is not int or not 1 <= max_samples <= 1000000:
        raise ValueError("invalid finite sample budget")
    if not isinstance(model, nn.Module) or type(loader) is not DataLoader:
        raise TypeError("expected nn.Module and standard DataLoader")
    if (not isinstance(loader.dataset, Dataset) or isinstance(loader.dataset, IterableDataset)
            or type(loader.sampler) is not SequentialSampler or type(loader.batch_sampler) is not BatchSampler
            or loader.batch_size is None or loader.batch_sampler.drop_last
            or loader.drop_last or loader.num_workers != 0 or loader.pin_memory):
        raise NotImplementedError("requires finite sequential map loader, workers=0, no drop_last/pin_memory")
    if source_id is not None and (type(source_id) is not str or not source_id.strip()):
        raise ValueError("invalid source_id")
    rng = _rng_state()
    generator = None if loader.generator is None else loader.generator.get_state().clone()
    try:
        expected = len(loader.dataset)
        expected_batches = len(loader)
        if not 0 <= expected <= max_samples or expected_batches > 100000:
            raise ValueError("finite evaluation budget exceeded")
        if capture_structure and expected == 0:
            raise ValueError('structure binding requires nonempty evaluation')
        state_hash = _model_hash(model)
        candidate = copy.deepcopy(model)
        original_objects = {id(m) for m in model.modules()} | {id(t) for t in model.parameters()} | {id(t) for t in model.buffers()}
        if any(id(obj) in original_objects for obj in [*candidate.modules(), *candidate.parameters(), *candidate.buffers()]):
            raise ValueError("model copy aliases original objects")
        original_storage = {t.untyped_storage().data_ptr() for t in [*model.parameters(), *model.buffers()] if t.numel()}
        if any(t.numel() and t.untyped_storage().data_ptr() in original_storage for t in [*candidate.parameters(), *candidate.buffers()]):
            raise ValueError("model copy aliases original storage")
        candidate.eval()
        baseline = _model_hash(candidate)
        if capture_evidence and baseline != state_hash:
            raise ValueError("isolated model state differs from original")
        structure_before = None
        if capture_structure:
            from .structure import capture_model_structure
            structure_before = capture_model_structure(candidate)
        stream = EvaluationStreamCapture() if capture_evidence else None
        targets, predictions, sizes = [], [], []
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                if batch_index >= expected_batches or not isinstance(batch, (tuple,list)) or len(batch) != 2:
                    raise ValueError("unexpected or invalid evaluation batch")
                inputs, labels = batch
                if (not isinstance(inputs, torch.Tensor) or inputs.device.type != "cpu" or inputs.dtype not in (torch.float32,torch.float64)
                        or inputs.ndim < 1 or not torch.isfinite(inputs).all()):
                    raise ValueError("finite CPU floating inputs required")
                n = inputs.shape[0]
                expected_batch = min(loader.batch_size, expected-len(targets))
                if n < 1 or n != expected_batch or not isinstance(labels,torch.Tensor) or labels.device.type != "cpu":
                    raise ValueError("missing samples or invalid targets")
                if task == "binary":
                    if labels.shape not in ((n,), (n,1)) or not labels.is_floating_point():
                        raise ValueError("binary targets require floating (B,) or (B,1)")
                    normalized = labels.reshape(n).tolist()
                    if any(v not in (0,1) for v in normalized):raise ValueError("invalid binary hard labels")
                    normalized = [int(v) for v in normalized]
                else:
                    if labels.shape != (n,) or labels.dtype != torch.long:raise ValueError("multiclass targets require long (B,)")
                    normalized = labels.tolist()
                _labels(normalized,len(class_names))
                if stream is not None:
                    stream.update(inputs, labels)
                logits = candidate(inputs.clone())
                if (not isinstance(logits,torch.Tensor) or logits.device.type != "cpu"
                        or logits.dtype not in (torch.float32,torch.float64) or not torch.isfinite(logits).all()):
                    raise ValueError("finite CPU tensor logits required")
                raw = dict(targets=normalized, predictions=logits.tolist(), prediction_type="logits")
                _predicted(raw,config)
                if _model_hash(candidate) != baseline or any(m.training for m in candidate.modules()):
                    raise ValueError("model mutated during evaluation")
                targets.extend(normalized); predictions.extend(logits.reshape(n).tolist() if task=="binary" else logits.tolist()); sizes.append(n)
        if len(sizes) != expected_batches or len(targets) != expected or len(loader.dataset) != expected:
            raise ValueError("incomplete evaluation coverage")
        structure = None
        if capture_structure:
            structure_after = capture_model_structure(candidate)
            post_state = _model_hash(candidate)
            if structure_before.content_sha256 != structure_after.content_sha256:
                raise ValueError('evaluated model structure changed during evaluation')
            if post_state != baseline:
                raise ValueError('evaluated model state changed during structure capture')
            structure = (structure_before, structure_after, baseline, post_state)
        if _model_hash(model) != state_hash:
            raise ValueError("original model changed during evaluation")
        return _result(dict(targets=targets,predictions=predictions,prediction_type="logits"),config,
            dict(kind="model_inference",source_id=source_id,model_type=type(model).__module__+"."+type(model).__qualname__,
                 model_state_sha256=state_hash,device="cpu",execution="isolated_deepcopy_eval_inference_mode",
                 dataset_type=type(loader.dataset).__module__+"."+type(loader.dataset).__qualname__,
                 sampler="sequential",num_workers=0),dict(expected_samples=expected,batch_sizes=sizes),
            evidence=producer_evidence(model, state_hash, stream, config, source_id) if stream is not None else None,
            structure=structure)
    finally:
        _restore_rng(rng)
        if generator is not None:loader.generator.set_state(generator)
