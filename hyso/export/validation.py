"""Bounded cohort metrics. Numerical and labelled task decisions are independent."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Mapping
import math
from .contracts import ExportError


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float
    allow_dtype_change: bool = False

    def __post_init__(self) -> None:
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (self.atol, self.rtol)):
            raise ExportError('INVALID_TOLERANCE', 'finite nonnegative tolerances required')
        if type(self.allow_dtype_change) is not bool:
            raise ExportError('INVALID_TOLERANCE', 'explicit dtype policy required')

    @classmethod
    def for_precision(cls, precision: str) -> Tolerance:
        """Defaults are numerical screening bounds, never task-quality guarantees."""
        values = {'fp32': (1e-5, 1e-4, False), 'fp16': (1e-3, 1e-2, True), 'int8': (1e-1, 1e-1, True)}
        if precision not in values:
            raise ExportError('PRECISION', 'fp32/fp16/int8 required')
        return cls(*values[precision])


def _arrays(value: Any) -> tuple:
    import numpy as np
    if isinstance(value, Mapping):
        values = tuple(value.values())
    elif isinstance(value, (tuple, list)):
        values = tuple(value)
    else:
        values = (value,)
    if not values or len(values) > 64:
        raise ExportError('OUTPUT_SPEC', 'one to 64 flat outputs required')
    result = []
    total = 0
    for item in values:
        if hasattr(item, 'detach'):
            if item.device.type != 'cpu':
                raise ExportError('CUDA_ADMISSION_REQUIRED', 'resident metrics require CPU outputs')
            item = item.detach().numpy()
        a = np.asarray(item)
        total += a.nbytes
        if a.size < 1 or a.dtype.kind not in 'fiu' or total > 32 * 1024**2:
            raise ExportError('OUTPUT_SPEC', 'bounded nonempty real numeric outputs required')
        result.append(a)
    return tuple(result)


def output_metrics(reference: Any, actual: Any, *, tolerance: Tolerance | None = None,
                   precision: str = 'fp32') -> dict[str, Any]:
    """Compare flat multi-output CPU values without concealing dtype/shape drift."""
    import numpy as np
    tol = tolerance or Tolerance.for_precision(precision)
    if type(tol) is not Tolerance:
        raise TypeError('Tolerance required')
    if isinstance(reference, Mapping) or isinstance(actual, Mapping):
        if not isinstance(reference, Mapping) or not isinstance(actual, Mapping) or list(reference) != list(actual):
            raise ExportError('OUTPUT_SPEC', 'named output keys/order must match')
    refs, vals = _arrays(reference), _arrays(actual)
    if len(refs) != len(vals):
        raise ExportError('OUTPUT_SPEC', 'output arity differs')
    rows = []
    for a, b in zip(refs, vals):
        shape = a.shape == b.shape
        dtype = a.dtype == b.dtype
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        row = dict(shape_match=shape, dtype_match=dtype, reference_shape=list(a.shape), actual_shape=list(b.shape),
                   reference_dtype=str(a.dtype), actual_dtype=str(b.dtype), finite=finite,
                   mae=None, rmse=None, max_absolute_difference=None, max_relative_error=None, cosine_similarity=None)
        matched = False
        if shape and finite:
            aa, bb = a.astype('float64').reshape(-1), b.astype('float64').reshape(-1)
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                error = np.abs(aa - bb)
                scale = max(float(np.max(np.abs(aa))), float(np.max(np.abs(bb))), 1e-300)
                sa, sb = aa / scale, bb / scale
                denom = float(np.linalg.norm(sa) * np.linalg.norm(sb))
                cosine = float(np.dot(sa, sb) / denom) if denom else (1.0 if np.array_equal(aa, bb) else 0.0)
                metrics = dict(mae=float(np.mean(error)), rmse=float(np.sqrt(np.mean((error / scale)**2)) * scale),
                               max_absolute_difference=float(np.max(error)),
                               max_relative_error=float(np.max(error / np.maximum(np.abs(aa), 1e-30))), cosine_similarity=cosine)
            if all(math.isfinite(v) for v in metrics.values()):
                row.update(metrics)
                matched = bool(np.all(error <= tol.atol + tol.rtol * np.abs(aa))) and (dtype or tol.allow_dtype_change)
            else:
                row['metric_status'] = 'unavailable_numeric_overflow'
        row['status'] = 'PASS' if matched else 'FAIL'
        row['classification_prediction_agreement'] = float(np.mean(a.argmax(1)==b.argmax(1))) if shape and finite and a.ndim==2 and a.shape[1]>1 else None
        rows.append(row)
    return dict(format='hyso.export.validation', schema_version=1, precision=precision,
                tolerance=dict(atol=tol.atol, rtol=tol.rtol, allow_dtype_change=tol.allow_dtype_change, relative_floor=1e-30),
                numerical=dict(status='PASS' if all(r['status'] == 'PASS' for r in rows) else 'FAIL', outputs=rows),
                quality=dict(status='NOT_TESTED', reason='No labelled classification cohort supplied'))


def classification_metrics(scores: Any, labels: Any, *, top_k: int = 1) -> dict[str, Any]:
    """Single-label multiclass metrics, macro average over the complete score vocabulary."""
    import numpy as np
    s = _arrays(scores)
    if len(s) != 1:
        raise ExportError('QUALITY_SPEC', 'select one classification output')
    s = s[0]
    y = np.asarray(labels)
    if (s.ndim != 2 or not np.isfinite(s).all() or y.shape != (s.shape[0],) or y.dtype.kind not in 'iu'
            or type(top_k) is not int or not 1 <= top_k <= s.shape[1] or np.any(y < 0) or np.any(y >= s.shape[1])):
        raise ExportError('QUALITY_SPEC', 'finite [batch,classes] scores and valid integer labels/top_k required')
    order = np.argsort(-s, axis=1, kind='stable')
    pred = order[:, 0]
    per_class = []
    for c in range(s.shape[1]):
        tp = int(np.sum((pred == c) & (y == c)))
        fp = int(np.sum((pred == c) & (y != c)))
        fn = int(np.sum((pred != c) & (y == c)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per_class.append(dict(class_id=c, precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.0,
                              support=int(np.sum(y == c))))
    return dict(status='PASS', samples=len(y), top_k=top_k, top1_accuracy=float(np.mean(pred == y)),
                topk_accuracy=float(np.mean(np.any(order[:, :top_k] == y[:, None], axis=1))),
                precision=float(np.mean([r['precision'] for r in per_class])),
                recall=float(np.mean([r['recall'] for r in per_class])), f1=float(np.mean([r['f1'] for r in per_class])),
                average='macro_all_score_classes', zero_division=0, tie_policy='stable_class_index', per_class=per_class)


def validate(reference: Any, candidate: Any, dataset: Iterable, *, precision: str = 'fp32',
             tolerance: Tolerance | None = None, labelled: bool = False, top_k: int = 1,
             classification_output: int = 0, max_batches: int = 64,
             max_accuracy_drop: float = 0.01, max_f1_drop: float = 0.01, dataset_id: str | None = None) -> dict[str, Any]:
    """Execute explicit Runtime handles on a bounded caller-supplied cohort.

    Unlabelled entries are inputs; labelled entries are (inputs, integer labels).
    Native GPU handles fail closed until an owned runtime adapter is selected.
    """
    import numpy as np
    from ..analysis._codec import digest
    if dataset_id is not None and (not isinstance(dataset_id,str) or not dataset_id.strip() or len(dataset_id)>256):
        raise ExportError('QUALITY_SPEC','optional dataset_id must be a nonempty bounded identifier')
    if type(max_batches) is not int or not 1 <= max_batches <= 256:
        raise ExportError('RESOURCE_BUDGET', 'max_batches must be 1..256')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in (max_accuracy_drop, max_f1_drop)):
        raise ExportError('QUALITY_SPEC', 'quality drop bounds must be 0..1')
    rows, refs, vals, ys, identities = [], [], [], [], []
    prior_receipts=(len(reference.receipts),len(candidate.receipts))
    retained = 0
    for index, entry in enumerate(dataset):
        if index >= max_batches:
            raise ExportError('RESOURCE_BUDGET', 'cohort exceeds declared batch limit; no partial PASS')
        if labelled:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ExportError('QUALITY_SPEC', 'labelled entries must be (inputs, labels)')
            inputs, labels = entry
        else:
            inputs, labels = entry, None
        a, b = reference.run(inputs), candidate.run(inputs)
        aa, bb = _arrays(a), _arrays(b)
        rows.append(output_metrics(a, b, tolerance=tolerance, precision=precision))
        identities.append(reference.input_identity(inputs))
        if labelled:
            if type(classification_output) is not int or not 0 <= classification_output < min(len(aa), len(bb)):
                raise ExportError('QUALITY_SPEC', 'classification output index outside outputs')
            ar, br = aa[classification_output], bb[classification_output]
            classification_metrics(ar, labels, top_k=top_k)
            classification_metrics(br, labels, top_k=top_k)
            retained += ar.nbytes + br.nbytes + np.asarray(labels).nbytes
            if retained > 32 * 1024**2:
                raise ExportError('RESOURCE_BUDGET', 'classification cohort exceeds retained budget')
            refs.append(ar.copy()); vals.append(br.copy()); ys.append(np.asarray(labels).copy())
            import hashlib
            identities[-1]=dict(inputs=identities[-1],label_sha256=hashlib.sha256(np.asarray(labels).tobytes()).hexdigest(),
                                label_dtype=str(np.asarray(labels).dtype),label_shape=list(np.asarray(labels).shape))
    if not rows:
        raise ExportError('QUALITY_SPEC', 'nonempty cohort required')
    quality = dict(status='NOT_TESTED', reason='No labels supplied')
    if labelled:
        a, b, labels = np.concatenate(refs), np.concatenate(vals), np.concatenate(ys)
        qa, qb = classification_metrics(a, labels, top_k=top_k), classification_metrics(b, labels, top_k=top_k)
        delta = {key: qb[key] - qa[key] for key in ('top1_accuracy', 'topk_accuracy', 'precision', 'recall', 'f1')}
        quality = dict(status='PASS' if delta['top1_accuracy'] >= -max_accuracy_drop and delta['f1'] >= -max_f1_drop else 'FAIL',
                       reference=qa, candidate=qb, delta=delta, max_accuracy_drop=max_accuracy_drop, max_f1_drop=max_f1_drop,
                       prediction_agreement=float(np.mean(a.argmax(1) == b.argmax(1))))
    return dict(format='hyso.export.validation', schema_version=1, precision=precision, batches=rows,
                dataset_id=dataset_id,cohort_sha256=digest(identities), numerical=dict(status='PASS' if all(r['numerical']['status']=='PASS' for r in rows) else 'FAIL'),
                quality=quality, reference=reference.info, candidate=candidate.info,
                native_receipts=reference.receipts[prior_receipts[0]:]+candidate.receipts[prior_receipts[1]:])
