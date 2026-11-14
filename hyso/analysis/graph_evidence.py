"""Opt-in Graph v2 source provenance alongside unchanged P26 evaluation evidence.

The isolated inference receiver has no transferred source receipt. Matching named
state is a limited association, never a verified inference-runtime contract.
Imported statements and presentation/artifact hashes cannot restore live trust.
"""
from dataclasses import dataclass
import json
import weakref

from ._codec import canonical, digest, export, plain, read_json, fingerprint
from .evidence import EvidenceIndex, EvidenceRecord
from .report import AnalysisContext
from .model_comparison import ModelComparisonResult, ModelComparisonSource, compare_models
from ..checkpoint import graph_checkpoint_evidence
from ..evaluation import EvaluationResult, evaluate_classification


class GraphEvidenceError(ValueError):
    """Invalid association, declaration or unsupported integration statement."""


_EVALUATIONS = {}
_COMPARISONS = {}
_LIMITS = [
    "source Graph v2 receipt belongs to the original loaded model only",
    "isolated_receiver_receipt=not_transferred;runtime_equivalence=not_observed",
    "matching named-state and native stream do not authenticate isolated runtime behavior",
    "dataset origin and preprocessing identity remain caller-declared",
    "retained JSON, report integrity and experiment artifact hashes are not native receipts",
    "no CUDA, hardware performance, global winner or improvement certification",
]


def _context(value, source_id, split):
    value = {} if value is None else plain(value)
    if type(value) is not dict or set(value) - {"dataset_id", "split", "preprocessing_id"}:
        raise GraphEvidenceError("invalid_declared_context")
    result = dict(dataset_id=value.get("dataset_id", source_id), split=value.get("split", split),
                  preprocessing_id=value.get("preprocessing_id"))
    if any(v is not None and (type(v) is not str or not v.strip()) for v in result.values()):
        raise GraphEvidenceError("invalid_context_value")
    if result["split"] != split or (source_id is not None and result["dataset_id"] != source_id):
        raise GraphEvidenceError("declared_dataset_or_split_mismatch")
    return result


def _receipt_statement(data):
    keys = {"format", "schema_version", "status", "checkpoint_sha256", "model_state_sha256",
            "contract", "source", "reason"}
    if (type(data) is not dict or set(data) != keys
            or data["format"] != "hyso.graph_checkpoint_evidence"
            or type(data["schema_version"]) is not int or data["schema_version"] != 2):
        raise GraphEvidenceError("invalid_graph_receipt_statement")
    if data["status"] == "missing":
        if any(data[k] is not None for k in ["checkpoint_sha256", "model_state_sha256", "contract", "source"]):
            raise GraphEvidenceError("invalid_missing_receipt")
    elif data["status"] == "native":
        fingerprint(data["checkpoint_sha256"])
        fingerprint(data["model_state_sha256"])
        c = data["contract"]
        if (type(c) is not dict or set(c) != {"format", "schema_version", "configuration", "adapter_output",
                "device", "dtype", "torch_version", "runtime_sha256", "serialized_state_sha256"}
                or c["format"] != "hyso.graph_checkpoint_contract" or type(c["schema_version"]) is not int
                or c["schema_version"] != 2 or c["device"] != "cpu"
                or c["dtype"] not in ("torch.float32", "torch.float64") or type(c["configuration"]) is not dict
                or (c["adapter_output"] is not None and type(c["adapter_output"]) is not str)
                or type(c["torch_version"]) is not str
                or data["source"] != "hyso.load_training_checkpoint.graph_contract.v2" or data["reason"] is not None):
            raise GraphEvidenceError("invalid_graph_contract_statement")
        fingerprint(c["runtime_sha256"])
        fingerprint(c["serialized_state_sha256"])
    else:
        raise GraphEvidenceError("invalid_receipt_status")


def _body(data, format_name, fields):
    data = plain(data)
    if type(data) is not dict:
        raise GraphEvidenceError("plain_versioned_statement_required")
    for optional in ("verification", "relation"):
        if optional in data:
            if type(data[optional]) is not dict:
                raise GraphEvidenceError("invalid_view_metadata")
            data.pop(optional)  # Persisted claims never supply process-local authority.
    if (set(data) != fields | {"format", "schema_version", "content_sha256"}
            or data["format"] != format_name or type(data["schema_version"]) is not int
            or data["schema_version"] != 1
            or data["content_sha256"] != digest({k: v for k, v in data.items() if k != "content_sha256"})):
        raise GraphEvidenceError("invalid_association_schema_or_integrity")
    if len(canonical(data).encode()) > 32 * 1024**2:
        raise GraphEvidenceError("association_size_limit")
    return data


def _remember(registry, result, *values):
    key = id(result)
    def expired(ref):
        if registry.get(key, (None,))[0] is ref:
            registry.pop(key, None)
    registry[key] = (weakref.ref(result, expired), result._payload, *values)


class _Export:
    def to_json(self, path=None, *, overwrite=False):
        raw = canonical(self.to_dict()) + "\n"
        if path is not None:
            export(path, raw, overwrite)
        return raw

    @classmethod
    def from_json(cls, path, *, expected_sha256=None):
        data, _, _ = read_json(path, expected_sha256=expected_sha256, max_bytes=32 * 1024**2)
        return cls(data)


@dataclass(frozen=True, init=False)
class GraphEvaluationResult(_Export):
    """Versioned statement; construction/JSON import never mints native trust."""
    _payload: str

    def __init__(self, data):
        d = _body(data, "hyso.graph_evaluation_association",
                  {"evaluation", "source_receipt", "record", "context"})
        evaluation = EvaluationResult(d["evaluation"]).to_dict()
        if evaluation["schema_version"] not in (2, 3) or evaluation["source"]["kind"] != "model_inference":
            raise GraphEvidenceError("native_evaluation_statement_required")
        record = EvidenceRecord(d["record"]).to_dict()
        if record["feature_id"] != "P26" or canonical(record["native_payload"]) != canonical(evaluation):
            raise GraphEvidenceError("record_evaluation_mismatch")
        if canonical(_context(d["context"], evaluation["source"]["source_id"],
                              evaluation["configuration"]["split"])) != canonical(d["context"]):
            raise GraphEvidenceError("invalid_context")
        _receipt_statement(d["source_receipt"])
        if (d["source_receipt"]["status"] == "native"
                and d["source_receipt"]["model_state_sha256"] != evaluation["source"]["model_state_sha256"]):
            raise GraphEvidenceError("source_and_evaluated_named_state_mismatch")
        object.__setattr__(self, "_payload", canonical(d))

    @property
    def evaluation(self):
        return EvaluationResult(json.loads(self._payload)["evaluation"])

    def _verification(self, d):
        v = dict(origin="retained_statement", graph_v2="unverified", evaluation_capture="retained_statement",
                 reason="JSON_or_copy_has_no_process_local_capture")
        entry = _EVALUATIONS.get(id(self))
        if entry is None or entry[0]() is not self or entry[1] != self._payload:
            return v
        v.update(origin="native_process_capture", evaluation_capture="native_in_process")
        model = entry[2]()
        if model is None:
            return dict(v, reason="original_source_object_unavailable")
        current = graph_checkpoint_evidence(model)
        trusted = d["source_receipt"]["status"] == "native" and canonical(current) == canonical(d["source_receipt"])
        return dict(v, graph_v2="verified" if trusted else "missing",
                    reason="original_source_receipt_revalidated" if trusted else "source_receipt_missing_or_changed")

    def to_dict(self):
        d = json.loads(self._payload)
        v = self._verification(d)
        d["verification"] = v
        d["relation"] = dict(status="limited" if v["graph_v2"] == "verified" else "unavailable",
            named_state_match=d["source_receipt"]["status"] == "native",
            isolated_receiver_receipt="not_transferred", runtime_equivalence="not_observed",
            scope="source_receipt_and_observed_named_tensors_only")
        return d

    def analysis_source(self, model_id):
        """Existing retained P26 source; never backfills checkpoint bindings."""
        d = json.loads(self._payload)
        record = EvidenceRecord(d["record"])
        report = EvidenceIndex([record]).analyze(context=AnalysisContext(workload_id="graph-evidence-association",
            task=d["evaluation"]["configuration"]["task"], model={"id": model_id}, dataset=d["context"]))
        return ModelComparisonSource(model_id, report, records={"P26": record.record_id})


def evaluate_graph_classification(model, loader, *, class_names, split, source_id=None, context=None,
        task="multiclass", zero_division=0, max_samples=1000000, capture_structure=False,
        expected_checkpoint_sha256=None, expected_model_state_sha256=None):
    """Call the existing evaluator once; capture only a LIMITED source association.

    No claim is made about a v2 receipt or runtime contract on its deepcopy.
    Expected identities constrain an existing source receipt, never create one.
    """
    declared = _context(context, source_id, split)
    before = graph_checkpoint_evidence(model)
    for field, expected in [("checkpoint_sha256", expected_checkpoint_sha256),
                            ("model_state_sha256", expected_model_state_sha256)]:
        if expected is not None:
            fingerprint(expected)
            if before["status"] != "native" or before[field] != expected:
                raise GraphEvidenceError("expected_source_identity_mismatch:" + field)
    evaluation = evaluate_classification(model, loader, class_names=class_names, split=split,
        source_id=source_id, task=task, zero_division=zero_division, max_samples=max_samples,
        capture_evidence=True, capture_structure=capture_structure)
    after = graph_checkpoint_evidence(model)
    if canonical(before) != canonical(after):
        raise GraphEvidenceError("source_contract_changed_during_evaluation")
    d = dict(format="hyso.graph_evaluation_association", schema_version=1, evaluation=evaluation.to_dict(),
             source_receipt=after, record=EvidenceRecord.from_native(evaluation.to_dict(), feature_id="P26").to_dict(),
             context=declared)
    d["content_sha256"] = digest(d)
    result = GraphEvaluationResult(d)
    _remember(_EVALUATIONS, result, weakref.ref(model))
    return result


def _compare(a, b):
    legacy = compare_models(a.analysis_source("baseline"), b.analysis_source("candidate"), features="P26").to_dict()
    contexts = [json.loads(x._payload)["context"] for x in (a, b)]
    diagnostics = []
    for field in ("dataset_id", "split", "preprocessing_id"):
        x, y = (c[field] for c in contexts)
        if x is None or y is None:
            diagnostics.append("missing_declared_condition:" + field)
        elif x != y:
            diagnostics.append("declared_condition_mismatch:" + field)
    metrics = [dict(metric=row["metric"], status=row["status"], difference=row["difference"], reasons=row["reasons"])
               for row in legacy["metrics"]]
    if diagnostics:
        state = "INCOMPATIBLE" if any("mismatch" in x for x in diagnostics) else "INSUFFICIENT_EVIDENCE"
        metrics = [dict(row, status=state, difference=None, reasons=row["reasons"] + diagnostics) for row in metrics]
    status = ("INCOMPATIBLE" if any(r["status"] == "INCOMPATIBLE" for r in metrics) else
              "COMPARABLE" if all(r["status"] == "COMPARABLE" for r in metrics) else "INSUFFICIENT_EVIDENCE")
    return dict(legacy=legacy, metrics=metrics, status=status, diagnostics=diagnostics)


@dataclass(frozen=True, init=False)
class GraphComparisonResult(_Export):
    _payload: str

    def __init__(self, data):
        d = _body(data, "hyso.graph_evidence_comparison", {"sources", "legacy", "metrics", "status", "diagnostics"})
        if type(d["sources"]) is not dict or set(d["sources"]) != {"baseline", "candidate"}:
            raise GraphEvidenceError("exactly_two_graph_associations_required")
        a, b = [GraphEvaluationResult(d["sources"][key]) for key in ("baseline", "candidate")]
        expected = _compare(a, b)
        if canonical(expected) != canonical({key: d[key] for key in expected}):
            raise GraphEvidenceError("comparison_derivation_mismatch")
        object.__setattr__(self, "_payload", canonical(d))

    @property
    def legacy_comparison(self):
        return ModelComparisonResult(json.loads(self._payload)["legacy"])

    def to_dict(self):
        d = json.loads(self._payload)
        entry = _COMPARISONS.get(id(self))
        original = entry is not None and entry[0]() is self and entry[1] == self._payload
        pair = entry[2:] if original else [GraphEvaluationResult(d["sources"][k]) for k in ("baseline", "candidate")]
        d["verification"] = {key: result.to_dict()["verification"]
                             for key, result in zip(("baseline", "candidate"), pair)}
        return d


def compare_graph_evaluations(baseline, candidate):
    """Retained P26 comparison plus explicit v2 scope; no architecture/performance inference."""
    if type(baseline) is not GraphEvaluationResult or type(candidate) is not GraphEvaluationResult:
        raise GraphEvidenceError("typed_graph_associations_required")
    d = dict(format="hyso.graph_evidence_comparison", schema_version=1,
             sources={"baseline": json.loads(baseline._payload), "candidate": json.loads(candidate._payload)},
             **_compare(baseline, candidate))
    d["content_sha256"] = digest(d)
    result = GraphComparisonResult(d)
    _remember(_COMPARISONS, result, baseline, candidate)
    return result


def report_section(result, language, level, catalog):
    """Use the existing reporting renderer and Advisor; never project v2 as legacy P26."""
    from ..reporting.report import _analysis_section, _comparison_section, _metric
    d = result.to_dict()
    if type(result) is GraphEvaluationResult:
        source = result.analysis_source("graph").to_dict()
        from .report import UnifiedAnalysisReport
        section = _analysis_section(UnifiedAnalysisReport(source["report"]), language, level, catalog)
        views = {"source": d["verification"]}
        legacy = d["evaluation"]["producer_evidence"]["checkpoint"]["status"]
        relation = d["relation"]["status"]
    else:
        section = _comparison_section(result.legacy_comparison, language, level, catalog)
        views, legacy = d["verification"], "see separate P26 source records"
        relation = {side: "limited" if view["graph_v2"] == "verified" else "unavailable"
                    for side, view in views.items()}
        for metric, row in zip(section["metrics"], d["metrics"]):
            if row["status"] != "COMPARABLE":
                metric.update(value=None, status=row["status"], percent_change=None,
                              percentage_point_change=None, reason=row["reasons"], comparison=row)
        if d["status"] != "COMPARABLE":
            section["charts"] = []
    proof = {key: v["graph_v2"] for key, v in views.items()}
    section.update(kind="graph_evidence", raw=d,
                   graph_evidence=dict(graph_v2=proof.get("source", proof), legacy_p26=legacy,
                       cross_evidence_binding=relation, authority="process_local_at_report_creation_or_retained_statement"))
    section["limits"] = [*section["limits"], *_LIMITS, *d.get("diagnostics", [])]
    for key, value in [("Legacy P26 checkpoint", legacy), ("Graph v2 source contract", proof),
                        ("Cross-evidence binding", relation), ("Dataset origin", "declared;not_verified")]:
        section["metrics"].append(_metric(key, key, value, None, "evidence_scope", "recorded",
            dict(verification=views), "Source provenance is separate from isolated inference evidence.", _LIMITS))
    return section
