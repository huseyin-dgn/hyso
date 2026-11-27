"""Opt-in Graph v2 source provenance alongside unchanged P26 evaluation evidence.

The isolated inference receiver has no transferred source receipt. Matching named
state is a limited association, never a verified inference-runtime contract.
Imported statements and presentation/artifact hashes cannot restore live trust.
"""
import typing as _typing
from dataclasses import dataclass
import json
import weakref

from ._codec import canonical, digest, export, plain, read_json, fingerprint
from .evidence import EvidenceIndex, EvidenceRecord
from .report import AnalysisContext
from .model_comparison import (ModelComparisonResult, ModelComparisonSource, compare_models,
                               _STRICT_CONTEXT_POLICY)
from ..checkpoint import graph_checkpoint_evidence
from ..evaluation import EvaluationResult, evaluate_classification
from .._graph_receiver import receiver_statement, receiver_verification, validate_statement


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
_RUNTIME_LIMITS = [
    "source Graph v2 checkpoint receipt is not transferred to the isolated receiver",
    "receiver binding requires explicit native CPU capture on the actual inference object",
    "pre/post and per-forward observations assume trusted quiescent installed Core/PyTorch",
    "no guarantee against concurrent hostile Python or transient mutate-and-restore behavior",
    *_LIMITS[3:],
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


def _contract_statement(c):
    if (type(c) is not dict or set(c) != {'format', 'schema_version', 'configuration', 'adapter_output',
            'device', 'dtype', 'torch_version', 'runtime_sha256', 'serialized_state_sha256'}
            or c['format'] != 'hyso.graph_checkpoint_contract' or type(c['schema_version']) is not int
            or c['schema_version'] != 2 or c['device'] != 'cpu'
            or c['dtype'] not in ('torch.float32','torch.float64') or type(c['configuration']) is not dict
            or (c['adapter_output'] is not None and type(c['adapter_output']) is not str)
            or type(c['torch_version']) is not str):
        raise GraphEvidenceError('invalid_graph_contract_statement')
    fingerprint(c['runtime_sha256'])
    fingerprint(c['serialized_state_sha256'])


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
        _contract_statement(data['contract'])
        if (data["source"] != "hyso.load_training_checkpoint.graph_contract.v2" or data["reason"] is not None):
            raise GraphEvidenceError("invalid_graph_contract_statement")
    else:
        raise GraphEvidenceError("invalid_receipt_status")


def _body(data, format_name, fields, versions=(1,)):
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
            or data["schema_version"] not in versions
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
    """Versioned statement; construction/JSON import never mints native trust.

    Args:
        data: Detached schema-compatible result mapping, including recorded status and integrity fields.
            Required.
    """
    _payload: str

    def __init__(self, data) -> 'None':
        """Initialize GraphEvaluationResult from the supplied settings or evidence.

        Args:
            data: Detached schema-compatible result mapping, including recorded status and integrity fields.
                Required.
        """
        version = data.get("schema_version") if type(data) is dict else None
        fields = {"evaluation", "source_receipt", "record", "context"}
        if version == 2:
            fields.add("receiver_binding")
        d = _body(data, "hyso.graph_evaluation_association",
                  fields, versions=(1, 2))
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
        if version == 2:
            validate_statement(d["receiver_binding"], evaluation, d["source_receipt"])
        object.__setattr__(self, "_payload", canonical(d))

    @property
    def evaluation(self) -> 'EvaluationResult':
        """Evaluation using the explicitly supplied evidence.
        """
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

    def to_dict(self) -> 'dict[str, _typing.Any]':
        """Return detached JSON-compatible evidence without rerunning the source workload.
        """
        d = json.loads(self._payload)
        v = self._verification(d)
        d["verification"] = v
        d["relation"] = dict(status="limited" if v["graph_v2"] == "verified" else "unavailable",
            named_state_match=d["source_receipt"]["status"] == "native",
            isolated_receiver_receipt="not_transferred", runtime_equivalence="not_observed",
            scope="source_receipt_and_observed_named_tensors_only")
        if d["schema_version"] == 2:
            entry = _EVALUATIONS.get(id(self))
            original = entry is not None and entry[0]() is self and entry[1] == self._payload
            runtime = receiver_verification(entry[3], d["receiver_binding"]) if original else "unverified"
            binding = "verified" if runtime == "verified" and v["graph_v2"] == "verified" else "unavailable"
            v.update(receiver_runtime=runtime, inference_binding=binding)
            d["relation"].update(status=binding, runtime_equivalence=runtime,
                scope="source_v2_and_actual_core_observed_receiver_same_evaluation")
        return d

    def analysis_source(self, model_id) -> 'ModelComparisonSource':
        """Existing retained P26 source; never backfills checkpoint bindings.

        Args:
            model_id: Caller label identifying a model within this evidence context; not a weight or
                hardware identity. Required.
        """
        d = json.loads(self._payload)
        record = EvidenceRecord(d["record"])
        report = EvidenceIndex([record]).analyze(context=AnalysisContext(workload_id="graph-evidence-association",
            task=d["evaluation"]["configuration"]["task"], model={"id": model_id}, dataset=d["context"]))
        return ModelComparisonSource(model_id, report, records={"P26": record.record_id})


def evaluate_graph_classification(model, loader, *, class_names, split, source_id=None, context=None,
        task: _typing.Literal['multiclass', 'binary']="multiclass", zero_division=0, max_samples=1000000, capture_structure=False,
        expected_checkpoint_sha256=None, expected_model_state_sha256=None, capture_receiver_runtime=False) -> 'GraphEvaluationResult':
    """Call the existing evaluator once; default is the legacy LIMITED association.

    Explicit capture_receiver_runtime=True creates association v2 from Core's
    observations on the actual isolated receiver, without transferring a receipt.
    Expected identities constrain an existing source receipt, never create one.

    Args:
        model: Caller-owned torch.nn.Module; must meet the operation device, dtype and input contract.
            Required.
        loader: Finite explicit data loader matching the evaluation or training task. Required.
        class_names: Ordered unique names for the complete class vocabulary; order must match score
            columns. Required.
        split: Explicit evaluation split label recorded in evidence; it does not select or partition
            data. Required.
        source_id: Caller-supplied stable source identifier retained in analysis provenance. Default:
            None.
        context: Explicit analysis provenance and workload scope; labels do not establish native
            identity. Default: None.
        task (_typing.Literal['multiclass', 'binary']): Explicit classification task selector; graph
            evaluation does not infer detection or segmentation semantics. Default: 'multiclass'.
        zero_division: Value used for undefined precision/recall; use the evaluator-supported choices.
            Default: 0.
        max_samples: Declared evaluation sample bound; oversized inputs fail rather than silently
            truncating. Default: 1000000.
        capture_structure: Capture registered model structure as separate metadata; no performance
            conclusion follows from structure. Default: False.
        expected_checkpoint_sha256: Expected recorded checkpoint digest to bind evaluation evidence; not
            a structure hash. Default: None.
        expected_model_state_sha256: Expected model-state digest for evidence binding; not a
            model-architecture hash. Default: None.
        capture_receiver_runtime: Explicitly capture bounded CPU receiver-runtime association evidence
            when True; otherwise retain the static association scope. Default: False.
    """
    declared = _context(context, source_id, split)
    if type(capture_receiver_runtime) is not bool:
        raise GraphEvidenceError("capture_receiver_runtime must be bool")
    before = graph_checkpoint_evidence(model)
    for field, expected in [("checkpoint_sha256", expected_checkpoint_sha256),
                            ("model_state_sha256", expected_model_state_sha256)]:
        if expected is not None:
            fingerprint(expected)
            if before["status"] != "native" or before[field] != expected:
                raise GraphEvidenceError("expected_source_identity_mismatch:" + field)
    options = {"capture_receiver_runtime": True} if capture_receiver_runtime else {}
    evaluation = evaluate_classification(model, loader, class_names=class_names, split=split,
        source_id=source_id, task=task, zero_division=zero_division, max_samples=max_samples,
        capture_evidence=True, capture_structure=capture_structure, **options)
    after = graph_checkpoint_evidence(model)
    if canonical(before) != canonical(after):
        raise GraphEvidenceError("source_contract_changed_during_evaluation")
    d = dict(format="hyso.graph_evaluation_association", schema_version=1, evaluation=evaluation.to_dict(),
             source_receipt=after, record=EvidenceRecord.from_native(evaluation.to_dict(), feature_id="P26").to_dict(),
             context=declared)
    if capture_receiver_runtime:
        d.update(schema_version=2, receiver_binding=receiver_statement(evaluation))
    d["content_sha256"] = digest(d)
    result = GraphEvaluationResult(d)
    if capture_receiver_runtime:
        _remember(_EVALUATIONS, result, weakref.ref(model), evaluation)
    else:
        _remember(_EVALUATIONS, result, weakref.ref(model))
    return result


def _compare(a, b, context_policy='legacy'):
    legacy = compare_models(a.analysis_source("baseline"), b.analysis_source("candidate"), features="P26").to_dict()
    if context_policy == 'strict':
        effective = compare_models(a.analysis_source("baseline"), b.analysis_source("candidate"),
                                   features="P26", context_policy='strict').to_dict()
        metrics = [dict(metric=row['metric'], status=row['status'], difference=row['difference'],
                        reasons=row['reasons']) for row in effective['metrics']]
        status = ('INCOMPATIBLE' if any(row['status'] == 'INCOMPATIBLE' for row in metrics) else
                  'COMPARABLE' if all(row['status'] == 'COMPARABLE' for row in metrics) else 'INSUFFICIENT_EVIDENCE')
        # These are explicit value-free export markers for the referenced metrics.
        # The referenced schema1 artifact is retained separately by its caller.
        reference = dict(format='hyso.graph_legacy_comparison_reference', schema_version=1,
            comparison_sha256=legacy['content_sha256'], export='reference_only',
            metrics=[dict(metric=row['metric'], status='NOT_EXPORTED', difference=None) for row in legacy['metrics']])
        diagnostics = sorted({reason for row in metrics for reason in row['reasons']
                              if reason.startswith(('missing_declared_', 'declared_condition_', 'conflicting_declared_'))})
        return dict(legacy=reference, effective=effective, context_policy=plain(_STRICT_CONTEXT_POLICY),
                    metrics=metrics, status=status, diagnostics=diagnostics)
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
    """Retain and validate a detached comparison of graph evaluation evidence.

    Args:
        data: Detached schema-compatible result mapping, including recorded status and integrity fields.
            Required.

    Returns:
        An immutable JSON-backed comparison result.
    """
    _payload: str

    def __init__(self, data) -> 'None':
        """Initialize GraphComparisonResult from the supplied settings or evidence.

        Args:
            data: Detached schema-compatible result mapping, including recorded status and integrity fields.
                Required.
        """
        version = data.get('schema_version') if type(data) is dict else None
        fields = {"sources", "legacy", "metrics", "status", "diagnostics"}
        if version == 3:
            fields |= {'effective', 'context_policy'}
        d = _body(data, "hyso.graph_evidence_comparison", fields, versions=(1, 2, 3))
        if type(d["sources"]) is not dict or set(d["sources"]) != {"baseline", "candidate"}:
            raise GraphEvidenceError("exactly_two_graph_associations_required")
        a, b = [GraphEvaluationResult(d["sources"][key]) for key in ("baseline", "candidate")]
        expected_version = 2 if any(json.loads(x._payload)["schema_version"] == 2 for x in (a, b)) else 1
        if version != 3 and version != expected_version:
            raise GraphEvidenceError("comparison_association_version_mismatch")
        expected = _compare(a, b, context_policy='strict' if version == 3 else 'legacy')
        if canonical(expected) != canonical({key: d[key] for key in expected}):
            raise GraphEvidenceError("comparison_derivation_mismatch")
        object.__setattr__(self, "_payload", canonical(d))

    @property
    def legacy_comparison(self) -> 'ModelComparisonResult | None':
        """Historical schema1/2 typed artifact; schema3 exports a reference only."""
        data = json.loads(self._payload)
        return None if data['schema_version'] == 3 else ModelComparisonResult(data["legacy"])

    @property
    def effective_comparison(self) -> 'ModelComparisonResult | None':
        """Validated effective comparison for the selected envelope policy."""
        data = json.loads(self._payload)
        return ModelComparisonResult(data['effective']) if data['schema_version'] == 3 else self.legacy_comparison

    def to_dict(self) -> 'dict[str, _typing.Any]':
        """Return detached JSON-compatible evidence without rerunning the source workload.
        """
        d = json.loads(self._payload)
        entry = _COMPARISONS.get(id(self))
        original = entry is not None and entry[0]() is self and entry[1] == self._payload
        pair = entry[2:] if original else [GraphEvaluationResult(d["sources"][k]) for k in ("baseline", "candidate")]
        d["verification"] = {key: result.to_dict()["verification"]
                             for key, result in zip(("baseline", "candidate"), pair)}
        return d


def compare_graph_evaluations(baseline, candidate, *, context_policy: _typing.Literal['legacy', 'strict']='legacy') -> 'GraphComparisonResult':
    """Retained P26 comparison plus explicit v2 scope; no architecture/performance inference.

    Args:
        baseline: Reference evidence used as the denominator/origin of candidate differences;
            missing/incompatible evidence is not inferred. Required.
        candidate: Evidence or runtime being compared to the reference; measurements remain bound to
            their recorded conditions. Required.
        context_policy (_typing.Literal['legacy', 'strict']): Policy for checking recorded context
            compatibility before comparing results. Default: 'legacy'.
    """
    if type(baseline) is not GraphEvaluationResult or type(candidate) is not GraphEvaluationResult:
        raise GraphEvidenceError("typed_graph_associations_required")
    if type(context_policy) is not str or context_policy not in ('legacy', 'strict'):
        raise GraphEvidenceError('unsupported_context_policy')
    version = 2 if any(json.loads(x._payload)["schema_version"] == 2 for x in (baseline, candidate)) else 1
    if context_policy == 'strict':
        version = 3
    d = dict(format="hyso.graph_evidence_comparison", schema_version=version,
             sources={"baseline": json.loads(baseline._payload), "candidate": json.loads(candidate._payload)},
             **_compare(baseline, candidate, context_policy))
    d["content_sha256"] = digest(d)
    result = GraphComparisonResult(d)
    _remember(_COMPARISONS, result, baseline, candidate)
    return result


def report_section(result, language: _typing.Literal['tr', 'en'], level, catalog) -> 'dict[str, _typing.Any]':
    """Use the existing reporting renderer and Advisor; never project v2 as legacy P26.

    Args:
        result: Retained HYSo result/evidence record; rendering/advice does not run measurements.
            Required.
        language (_typing.Literal['tr', 'en']): Presentation language: tr, en, de or es. Technical
            keys/status codes remain stable. Required.
        level: Heading level used when embedding a report section. Required.
        catalog: Translation/metric catalog used to render stable recorded fields. Required.
    """
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
        section = _comparison_section(result.effective_comparison, language, level, catalog)
        views, legacy = d["verification"], "see separate P26 source records"
        relation = {side: view.get("inference_binding", "limited" if view["graph_v2"] == "verified" else "unavailable")
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
    if d.get('context_policy') is not None:
        section['graph_evidence']['context_policy'] = d['context_policy']
    limits = _RUNTIME_LIMITS if d["schema_version"] in (2, 3) else _LIMITS
    if d["schema_version"] in (2, 3):
        section["graph_evidence"]["receiver_runtime"] = {key: v.get("receiver_runtime", "not_observed")
                                                        for key, v in views.items()}
    section["limits"] = [*section["limits"], *limits, *d.get("diagnostics", [])]
    for key, value in [("Legacy P26 checkpoint", legacy), ("Graph v2 source contract", proof),
                        ("Cross-evidence binding", relation), ("Dataset origin", "declared;not_verified")]:
        section["metrics"].append(_metric(key, key, value, None, "evidence_scope", "recorded",
            dict(verification=views), "Source provenance is separate from isolated inference evidence.", limits))
    return section
