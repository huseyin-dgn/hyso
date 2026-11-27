"""3B.4D.2: literal registered-path comparison of validated structure snapshots.

Only recorded metadata is read. No model objects, introspection, workloads,
checkpoint/state identities, execution graph or performance verdicts.
"""
import typing as _typing
from dataclasses import dataclass
from decimal import Decimal, localcontext
import json
import math

from .analysis._codec import canonical, digest, export, plain, read_json
from .structure import ModelStructureSnapshot, ModelStructureError, _freeze

__all__ = ['compare_model_structures', 'ModelStructureComparison', 'ModelStructureComparisonError',
           'StructureSnapshotRef', 'ComparedQuantity', 'StructureNumericDifference',
           'StructureObservation', 'StructureFieldDifference', 'StructurePathDifference',
           'StructureTensorDifference', 'StructureDistributionDifference', 'StructureDtypeDifference',
           'RegistrationEndpoint', 'StructureAliasGroup', 'StructureAliasDifference', 'StructureEvidenceGap',
           'StructureComparisonPolicy']

_POLICY = dict(id='literal_registered_path_structure_v1', version=1,
              matching='exact_registered_path_and_registration_name', aliases='within_snapshot_endpoint_partitions')
_MAX_ENTRIES = 500000
_MAX_BYTES = 64 * 1024**2
_LIMITATIONS = (
    'Recorded static structure only; literal paths/registration names are matched without rename or semantic-equivalence guesses.',
    'Local module/tensor/storage IDs are interpreted only within their source snapshot; cross-model object or memory identity is not established.',
    'Alias relations are compared on common observed endpoints; unmatched and unavailable endpoints remain explicit.',
    'Unsupported/missing totals have no global difference or percentage; known subtotals are separate incomplete observations.',
    'Registered module counts are not executed operations/layers; functional calls, execution order, runtime shapes, dynamic control flow, activations and MAC/FLOPs are out of scope.',
    'requires_grad is metadata, not actual training behavior. Logical bytes are not VRAM/RSS, physical unique storage bytes or serialized checkpoint size.',
    'Equal structure or zero differences do not prove equal weights, outputs, live model origin, independent captures or model behavior.',
    'Snapshot hashes identify recorded structure content, not model-state or checkpoint artifacts; caller model labels are not native identity.',
    'Imported JSON validation proves consistency, not producer authenticity. Torch provenance differences do not establish cross-runtime equivalence.',
    'No speed, memory-allocation, accuracy, generalization, optimization success or VERIFIED_IMPROVEMENT conclusion is produced.',
    'No automatic P26/P11 association, architecture audit, model-family certification or user-interface rendering is performed.',
)
_METRICS = (
    ('parameter_elements', 'element'), ('trainable_elements', 'element'), ('frozen_elements', 'element'),
    ('unique_parameter_tensors', 'tensor'), ('parameter_logical_bytes', 'byte'),
    ('unique_buffer_tensors', 'tensor'), ('buffer_elements', 'element'), ('buffer_logical_bytes', 'byte'),
    ('tensor_union_logical_bytes', 'byte'), ('cross_category_tensors', 'tensor'),
    ('unique_modules', 'module'), ('registered_module_paths', 'path'),
)


class ModelStructureComparisonError(ModelStructureError):
    """Invalid evidence/result or explicit comparison limit; no truncation."""


@dataclass(frozen=True)
class StructureSnapshotRef:
    """Store structure snapshot ref evidence as an immutable typed record.

    Args:
        content_sha256 (str): Digest of the exact canonical evidence content; not a model-weight hash.
            Required.
        format (str): Output representation; use the choices declared for this renderer. Required.
        schema_version (int): Recorded serialization schema version. Required.
        producer_method (str): Recorded evidence producer identifier. Required.
        producer_version (int): Recorded evidence producer version. Required.
        classification_version (int): Version of the exact-type structural classification policy.
            Required.
        torch_version (str): Recorded PyTorch provenance; not runtime compatibility certification.
            Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    content_sha256: str
    format: str
    schema_version: int
    producer_method: str
    producer_version: int
    classification_version: int
    torch_version: str


@dataclass(frozen=True)
class StructureComparisonPolicy:
    """Store structure comparison policy evidence as an immutable typed record.

    Args:
        id (str): Stable policy or rule identifier. Required.
        version (int): Version of this evidence policy or rule. Required.
        matching (str): Declared comparison matching policy; literal paths are not semantic equivalence.
            Required.
        aliases (str): Declared policy for comparing within-snapshot registration alias groups.
            Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    id: str
    version: int
    matching: str
    aliases: str


@dataclass(frozen=True)
class ComparedQuantity:
    """Store compared quantity evidence as an immutable typed record.

    Args:
        status (str): Recorded evidence/coverage status; unavailable values must not be interpreted as
            zero. Required.
        value (int | None): Observed or derived value; None records unavailable evidence where allowed
            by the schema. Required.
        known_subtotal (int | None): Sum over supported observed entries only; not a complete total when
            coverage is partial. Required.
        unsupported_tensor_ids (tuple): Local tensor identifiers omitted from a complete quantity
            because metadata is unsupported. Required.
        source (str): Existing source evidence/artifact; verification is separate from execution.
            Required.
        reason (str | None): Recorded explanation for rejection, absence or incomplete evidence; None
            only where the schema allows it. Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    status: str
    value: int | None
    known_subtotal: int | None
    unsupported_tensor_ids: tuple
    source: str
    reason: str | None


@dataclass(frozen=True)
class StructureNumericDifference:
    """Store structure numeric difference evidence as an immutable typed record.

    Args:
        metric (str): Exact recorded metric whose baseline/candidate values are compared. Required.
        unit (str): Unit attached to the numeric quantity or difference. Required.
        baseline (ComparedQuantity): Reference evidence used as the denominator/origin of candidate
            differences; missing/incompatible evidence is not inferred. Required.
        candidate (ComparedQuantity): Evidence or runtime being compared to the reference; measurements
            remain bound to their recorded conditions. Required.
        status (str): Recorded evidence/coverage status; unavailable values must not be interpreted as
            zero. Required.
        difference (int | None): Candidate minus baseline where both quantities are comparable and
            complete. Required.
        percent_change (float | None): Relative candidate-minus-baseline change, or None when
            unavailable/undefined. Required.
        percent_reason (str | None): Reason a percentage cannot be computed; a zero baseline is not
            silently replaced. Required.
        known_subtotal_difference (int | None): Difference of incomplete known subtotals; distinct from
            a complete metric difference. Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    metric: str
    unit: str
    baseline: ComparedQuantity
    candidate: ComparedQuantity
    status: str
    difference: int | None
    percent_change: float | None
    percent_reason: str | None
    known_subtotal_difference: int | None


@dataclass(frozen=True)
class StructureObservation:
    """Store structure observation evidence as an immutable typed record.

    Args:
        status (str): Recorded evidence/coverage status; unavailable values must not be interpreted as
            zero. Required.
        value (object): Observed or derived value; None records unavailable evidence where allowed by
            the schema. Required.
        reason (str | None): Recorded explanation for rejection, absence or incomplete evidence; None
            only where the schema allows it. Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    status: str
    value: object
    reason: str | None


@dataclass(frozen=True)
class StructureFieldDifference:
    """Store structure field difference evidence as an immutable typed record.

    Args:
        field (str): Source metadata field path used by this observation. Required.
        baseline (StructureObservation): Reference evidence used as the denominator/origin of candidate
            differences; missing/incompatible evidence is not inferred. Required.
        candidate (StructureObservation): Evidence or runtime being compared to the reference;
            measurements remain bound to their recorded conditions. Required.
        status (str): Recorded evidence/coverage status; unavailable values must not be interpreted as
            zero. Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    field: str
    baseline: StructureObservation
    candidate: StructureObservation
    status: str


@dataclass(frozen=True)
class StructurePathDifference:
    """Store structure path difference evidence as an immutable typed record.

    Args:
        path (str): Filesystem location read or written by this explicit operation; output parents must
            already exist. Required.
        baseline_type (str): Recorded qualified type at the baseline registration path. Required.
        candidate_type (str): Recorded qualified type at the candidate registration path. Required.
        metadata_changes (tuple): Recorded changes to supported static type/metadata fields. Required.
        configuration_changes (tuple): Recorded changes to supported static constructor settings.
            Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    path: str
    baseline_type: str
    candidate_type: str
    metadata_changes: tuple
    configuration_changes: tuple


@dataclass(frozen=True)
class StructureTensorDifference:
    """Store structure tensor difference evidence as an immutable typed record.

    Args:
        owner_path (str): Exact registered module path owning this tensor/alias endpoint. Required.
        kind (str): Recorded entity/registration kind, such as parameter or buffer. Required.
        name (str): Exact registered component identifier; names outside the active distribution profile
            are rejected. Required.
        changes (tuple): Explicit changed fields between the matched evidence entries. Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    owner_path: str
    kind: str
    name: str
    changes: tuple


@dataclass(frozen=True)
class StructureDistributionDifference:
    """Store structure distribution difference evidence as an immutable typed record.

    Args:
        name (str): Exact registered component identifier; names outside the active distribution profile
            are rejected. Required.
        unique_modules (StructureNumericDifference): Count/difference of distinct registered module
            objects, not executed operations. Required.
        registered_paths (StructureNumericDifference): Count/difference of registered paths, including
            aliases. Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    name: str
    unique_modules: StructureNumericDifference
    registered_paths: StructureNumericDifference


@dataclass(frozen=True)
class StructureDtypeDifference:
    """Store structure dtype difference evidence as an immutable typed record.

    Args:
        kind (str): Recorded entity/registration kind, such as parameter or buffer. Required.
        dtype (str | None): Recorded tensor dtype; no tensor conversion is performed. Required.
        tensor_count (StructureNumericDifference): Count/difference of unique tensors in this dtype
            group. Required.
        elements (StructureNumericDifference): Number of logical tensor elements, if supported.
            Required.
        logical_bytes (StructureNumericDifference): Logical element count times element size; not
            allocator use, RSS or checkpoint file size. Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    kind: str
    dtype: str | None
    tensor_count: StructureNumericDifference
    elements: StructureNumericDifference
    logical_bytes: StructureNumericDifference


@dataclass(frozen=True)
class RegistrationEndpoint:
    """Store registration endpoint evidence as an immutable typed record.

    Args:
        owner_path (str): Exact registered module path owning this tensor/alias endpoint. Required.
        kind (str): Recorded entity/registration kind, such as parameter or buffer. Required.
        name (str): Exact registered component identifier; names outside the active distribution profile
            are rejected. Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    owner_path: str
    kind: str
    name: str


@dataclass(frozen=True)
class StructureAliasGroup:
    """Store structure alias group evidence as an immutable typed record.

    Args:
        kind (str): Recorded entity/registration kind, such as parameter or buffer. Required.
        members (tuple): Registration endpoints forming one observed alias group. Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    kind: str
    members: tuple


@dataclass(frozen=True)
class StructureAliasDifference:
    """Store structure alias difference evidence as an immutable typed record.

    Args:
        kind (str): Recorded entity/registration kind, such as parameter or buffer. Required.
        status (str): Recorded evidence/coverage status; unavailable values must not be interpreted as
            zero. Required.
        matching_scope (str): Endpoints considered comparable by the declared matching policy. Required.
        baseline_groups (tuple): Observed baseline alias groups on the comparison scope. Required.
        candidate_groups (tuple): Observed candidate alias groups on the comparison scope. Required.
        removed_groups (tuple): Baseline alias groups absent from comparable candidate evidence.
            Required.
        added_groups (tuple): Candidate alias groups absent from comparable baseline evidence. Required.
        unmatched_baseline_endpoints (tuple): Baseline registrations without a matching candidate
            endpoint. Required.
        unmatched_candidate_endpoints (tuple): Candidate registrations without a matching baseline
            endpoint. Required.
        unassessed_baseline_endpoints (tuple): Baseline endpoints with unsupported alias metadata.
            Required.
        unassessed_candidate_endpoints (tuple): Candidate endpoints with unsupported alias metadata.
            Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    kind: str
    status: str
    matching_scope: str
    baseline_groups: tuple
    candidate_groups: tuple
    removed_groups: tuple
    added_groups: tuple
    unmatched_baseline_endpoints: tuple
    unmatched_candidate_endpoints: tuple
    unassessed_baseline_endpoints: tuple
    unassessed_candidate_endpoints: tuple


@dataclass(frozen=True)
class StructureEvidenceGap:
    """Store structure evidence gap evidence as an immutable typed record.

    Args:
        side (str): Evidence side, baseline or candidate. Required.
        owner_path (str): Exact registered module path owning this tensor/alias endpoint. Required.
        kind (str): Recorded entity/registration kind, such as parameter or buffer. Required.
        name (str): Exact registered component identifier; names outside the active distribution profile
            are rejected. Required.
        status (str): Recorded evidence/coverage status; unavailable values must not be interpreted as
            zero. Required.
        fields (tuple): Source metadata field paths associated with the evidence gap. Required.
        reason (str): Recorded explanation for rejection, absence or incomplete evidence; None only
            where the schema allows it. Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    side: str
    owner_path: str
    kind: str
    name: str
    status: str
    fields: tuple
    reason: str


class _Budget:
    def __init__(self, maximum): self.maximum, self.used = maximum, 0
    def use(self, count=1):
        self.used += count
        if self.used > self.maximum:
            raise ModelStructureComparisonError('comparison_entry_limit')


def _ref(data):
    p = data['producer']
    return dict(content_sha256=data['content_sha256'], format=data['format'], schema_version=data['schema_version'],
                producer_method=p['method'], producer_version=p['version'],
                classification_version=p['classification_version'], torch_version=p['torch_version'])


def _quantity(value, source):
    if type(value) is int:
        return dict(status='native', value=value, known_subtotal=value, unsupported_tensor_ids=[], source=source, reason=None)
    return dict(status=value['status'], value=value['value'], known_subtotal=value['known_subtotal'],
                unsupported_tensor_ids=value['unsupported_tensor_ids'], source=source,
                reason=None if value['status'] == 'native' else 'incomplete_tensor_metadata')


def _number(metric, unit, a, b):
    computed = a['status'] == b['status'] == 'native'
    difference = b['value'] - a['value'] if computed else None
    percent = None; reason = 'incomplete_total'
    if computed:
        reason = 'zero_baseline' if a['value'] == 0 else None
        if a['value']:
            with localcontext() as ctx:
                ctx.prec = 50
                try:
                    percent = float(Decimal(difference) / Decimal(a['value']) * 100)
                    if not math.isfinite(percent): percent, reason = None, 'relative_percent_not_finite'
                except (OverflowError, ValueError):
                    reason = 'relative_percent_not_finite'
    subtotal = (b['known_subtotal'] - a['known_subtotal']
                if a['known_subtotal'] is not None and b['known_subtotal'] is not None else None)
    return dict(metric=metric, unit=unit, baseline=a, candidate=b,
        status='computed' if computed else 'insufficient_evidence', difference=difference,
        percent_change=percent, percent_reason=reason, known_subtotal_difference=subtotal)


def _observed(value): return dict(status='native', value=value, reason=None)
def _absent(reason='not_registered'): return dict(status='absent', value=None, reason=reason)


def _field(name, a, b, budget):
    if a['status'] in ('unsupported', 'missing') or b['status'] in ('unsupported', 'missing'):
        status = 'unresolved'
    elif a['status'] == b['status'] == 'absent' or canonical(a) == canonical(b):
        return None
    else:
        status = 'added' if a['status'] == 'absent' else 'removed' if b['status'] == 'absent' else 'changed'
    budget.use()
    return dict(field=name, baseline=a, candidate=b, status=status)


def _views(data, side, budget):
    modules = {m['module_id']: m for m in data['modules']}
    tensors = {t['tensor_id']: t for t in data['tensors']}
    paths = {p['path']: modules[p['module_id']] for p in data['paths']}
    endpoints = {}; gaps = []
    for path, m in paths.items():
        budget.use()
        unknown = [f['name'] for f in m['configuration'] if f['status'] != 'native']
        if m['configuration_status'] != 'native':
            budget.use()
            gaps.append(dict(side=side, owner_path=path, kind='module', name='', status='unsupported',
                fields=unknown or ['custom_semantics'], reason='static_configuration_not_fully_captured'))
        for kind in ('parameter', 'buffer'):
            for slot in m[kind + 's']:
                budget.use(); key = (path, kind, slot['name'])
                t = tensors.get(slot['tensor_id'])
                endpoints[key] = (slot, t)
                if t is not None and (t['status'] != 'native' or t['storage_status'] != 'native'):
                    budget.use()
                    fields = [k for k in ('shape', 'dtype', 'device', 'layout', 'requires_grad', 'elements', 'logical_bytes') if t[k] is None]
                    if t['storage_status'] != 'native': fields += ['storage_identity', 'stride', 'storage_offset']
                    gaps.append(dict(side=side, owner_path=path, kind=kind, name=slot['name'],
                        status='unsupported' if t['status'] != 'native' else 'missing', fields=fields,
                        reason=t['reason'] or t['storage_reason']))
    return paths, endpoints, gaps


_TENSOR_FIELDS = ('qualified_type', 'shape', 'dtype', 'device', 'layout', 'requires_grad',
                  'elements', 'element_size', 'logical_bytes', 'stride', 'storage_offset')


def _tensor_observation(tensor, field):
    if tensor is None: return _absent('tensor_not_present')
    value = tensor[field]
    if value is not None: return _observed(value)
    storage = field in ('stride', 'storage_offset')
    status = tensor['storage_status'] if storage else tensor['status']
    return dict(status=status if status != 'native' else 'missing', value=None,
                reason=tensor['storage_reason'] if storage else tensor['reason'] or 'metadata_not_captured')


def _path_differences(left, right, budget):
    changed = []
    for path in sorted(set(left) & set(right)):
        a, b = left[path], right[path]; fields = []; config = []
        comparisons = [('qualified_type', a['qualified_type'], b['qualified_type']), ('category', a['category'], b['category'])]
        for key in ('children', 'parameters', 'buffers'):
            comparisons.append((key + '_registration_order', [s['name'] for s in a[key]], [s['name'] for s in b[key]]))
        for name, av, bv in comparisons:
            delta = _field(name, _observed(av), _observed(bv), budget)
            if delta: fields.append(delta)
        ac, bc = ({f['name']: f for f in m['configuration']} for m in (a, b))
        for name in sorted(set(ac) | set(bc)):
            def observation(mapping):
                f = mapping.get(name)
                return {k: f[k] for k in ('status', 'value', 'reason')} if f else _absent('configuration_field_absent')
            delta = _field(name, observation(ac), observation(bc), budget)
            if delta: config.append(delta)
        if fields or config:
            changed.append(dict(path=path, baseline_type=a['qualified_type'], candidate_type=b['qualified_type'],
                                metadata_changes=fields, configuration_changes=config))
    return changed


def _tensor_differences(left, right, budget):
    result = []
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key), right.get(key); fields = []
        for name, av, bv in (('registration_present', _observed(a is not None), _observed(b is not None)),
                ('tensor_present', _observed(a[1] is not None) if a else _absent(), _observed(b[1] is not None) if b else _absent())):
            delta = _field(name, av, bv, budget)
            if delta: fields.append(delta)
        if key[1] == 'buffer':
            delta = _field('persistent', _observed(a[0]['persistent']) if a else _absent(),
                           _observed(b[0]['persistent']) if b else _absent(), budget)
            if delta: fields.append(delta)
        for name in _TENSOR_FIELDS:
            delta = _field(name, _tensor_observation(a[1] if a else None, name),
                           _tensor_observation(b[1] if b else None, name), budget)
            if delta: fields.append(delta)
        if fields:
            result.append(dict(owner_path=key[0], kind=key[1], name=key[2], changes=fields))
    return result


def _distribution(a, b, key):
    left, right = ({v['name']: v for v in s['summary'][key]} for s in (a, b))
    return [dict(name=name, **{field: _number(key + '.' + field, unit,
        _quantity(left.get(name, {}).get(field, 0), 'summary.' + key + '.' + name + '.' + field),
        _quantity(right.get(name, {}).get(field, 0), 'summary.' + key + '.' + name + '.' + field))
        for field, unit in (('unique_modules', 'module'), ('registered_paths', 'path'))}) for name in sorted(set(left) | set(right))]


def _dtype_differences(a, b):
    result = []
    for kind in ('parameter', 'buffer'):
        buckets = [{v['dtype']: v for v in s['summary'][kind + '_dtypes']} for s in (a, b)]
        unknowns = []
        for s in (a, b):
            ids = {slot['tensor_id'] for m in s['modules'] for slot in m[kind + 's'] if slot['tensor_id'] is not None}
            unknowns.append([t['tensor_id'] for t in s['tensors'] if t['tensor_id'] in ids and t['dtype'] is None])
        for dtype in sorted(set(buckets[0]) | set(buckets[1]), key=lambda v: '' if v is None else v):
            fields = {}
            for field, unit in (('tensor_count', 'tensor'), ('elements', 'element'), ('logical_bytes', 'byte')):
                values = []
                for bucket, unknown in zip(buckets, unknowns):
                    value = bucket[dtype][field] if dtype in bucket else 0
                    q = _quantity(value, 'summary.' + kind + '_dtypes.' + str(dtype) + '.' + field)
                    if dtype is not None and unknown:
                        q.update(status='unsupported', value=None,
                            unsupported_tensor_ids=sorted(set(q['unsupported_tensor_ids']) | set(unknown)), reason='dtype_assignment_incomplete')
                    values.append(q)
                fields[field] = _number(kind + '_dtype.' + field, unit, *values)
            result.append(dict(kind=kind, dtype=dtype, **fields))
    return result


def _endpoint(key): return dict(owner_path=key[0], kind=key[1], name=key[2])


def _alias_differences(left_paths, right_paths, left_endpoints, right_endpoints, budget):
    result = []
    module_maps = [{(p, 'module', ''): m['module_id'] for p, m in paths.items()} for paths in (left_paths, right_paths)]
    tensor_maps = [{k: t['tensor_id'] for k, (_, t) in eps.items() if t is not None} for eps in (left_endpoints, right_endpoints)]
    storage_maps = [{k: t['storage_id'] for k, (_, t) in eps.items() if t is not None and t['storage_status'] == 'native'}
                    for eps in (left_endpoints, right_endpoints)]
    for kind, maps, universe in (('module_object', module_maps, module_maps),
                                ('tensor_object', tensor_maps, tensor_maps), ('storage', storage_maps, tensor_maps)):
        common = set(universe[0]) & set(universe[1])
        assessed = common & set(maps[0]) & set(maps[1])
        def normalize(side, selected):
            mapping = maps[side]
            objects = {}
            # Source-local IDs only group endpoints INSIDE this snapshot.
            for key in sorted(selected):
                if kind == 'storage':
                    storage = mapping[key]; obj = tensor_maps[side][key]
                    objects.setdefault(storage, {}).setdefault(obj, []).append(key)
                else:
                    objects.setdefault(mapping[key], []).append(key)
            normalized = {}
            for value in objects.values():
                if kind == 'storage':
                    members = sorted(tuple(keys) for keys in value.values())
                    shared = len(members) > 1
                else:
                    members = [tuple(value)]; shared = len(value) > 1
                if shared:
                    group = dict(kind=kind, members=[[_endpoint(k) for k in keys] for keys in members])
                    normalized[canonical(group)] = group
            return normalized
        groups = [normalize(i, assessed) for i in range(2)]
        full_groups = [normalize(i, set(maps[i])) for i in range(2)]
        full_rows = [[g[k] for k in sorted(g)] for g in full_groups]
        removed = [groups[0][k] for k in sorted(set(groups[0]) - set(groups[1]))]
        added = [groups[1][k] for k in sorted(set(groups[1]) - set(groups[0]))]
        unmatched = [sorted(set(universe[i]) - common) for i in range(2)]
        unassessed = [sorted(common - assessed) for _ in range(2)]
        budget.use(len(removed) + len(added) + sum(map(len, unmatched)) + sum(map(len, unassessed))
                   + sum(len(member) for side in full_rows for group in side for member in group['members']))
        result.append(dict(kind=kind, status='partial' if any(unmatched + unassessed) else 'complete_common_endpoints',
            matching_scope='common_registered_endpoints', baseline_groups=full_rows[0], candidate_groups=full_rows[1],
            removed_groups=removed, added_groups=added,
            unmatched_baseline_endpoints=[_endpoint(k) for k in unmatched[0]],
            unmatched_candidate_endpoints=[_endpoint(k) for k in unmatched[1]],
            unassessed_baseline_endpoints=[_endpoint(k) for k in unassessed[0]],
            unassessed_candidate_endpoints=[_endpoint(k) for k in unassessed[1]]))
    return result


def _build(a, b, max_entries=_MAX_ENTRIES):
    budget = _Budget(max_entries)
    for field in ('method', 'version', 'classification_version'):
        if a['producer'][field] != b['producer'][field]:
            raise ModelStructureComparisonError('incompatible_structure_producers')
    left_paths, left_tensors, lg = _views(a, 'baseline', budget)
    right_paths, right_tensors, rg = _views(b, 'candidate', budget)
    summary = [_number(name, unit, _quantity(a['summary'][name], 'summary.' + name),
                       _quantity(b['summary'][name], 'summary.' + name)) for name, unit in _METRICS]
    added, removed = sorted(set(right_paths) - set(left_paths)), sorted(set(left_paths) - set(right_paths))
    budget.use(len(added) + len(removed))
    changed = _path_differences(left_paths, right_paths, budget)
    tensors = _tensor_differences(left_tensors, right_tensors, budget)
    aliases = _alias_differences(left_paths, right_paths, left_tensors, right_tensors, budget)
    provenance = []
    delta = _field('producer.torch_version', _observed(a['producer']['torch_version']), _observed(b['producer']['torch_version']), budget)
    if delta: provenance.append(delta)
    observed = (bool(added or removed) or any(d['difference'] not in (None, 0) for d in summary)
        or any(f['status'] != 'unresolved' for p in changed for f in p['metadata_changes'] + p['configuration_changes'])
        or any(f['status'] != 'unresolved' for t in tensors for f in t['changes'])
        or any(d['removed_groups'] or d['added_groups'] for d in aliases))
    result = dict(format='hyso.model_structure_comparison', schema_version=1, policy=_POLICY,
        baseline=_ref(a), candidate=_ref(b), source_snapshots=dict(baseline=a, candidate=b),
        summary_differences=summary, module_type_differences=_distribution(a, b, 'module_types'),
        category_differences=_distribution(a, b, 'categories'), dtype_differences=_dtype_differences(a, b),
        added_paths=added, removed_paths=removed, changed_paths=changed, tensor_differences=tensors,
        alias_differences=aliases, provenance_differences=provenance, evidence_gaps=lg + rg,
        coverage='partial' if lg or rg or any(x['status'] == 'partial' for x in aliases) else 'complete_recorded_metadata',
        has_observed_changes=bool(observed), limitations=_LIMITATIONS)
    result = plain(result); result['content_sha256'] = digest(result)
    return result


def _qview(data):
    return ComparedQuantity(**{k: _freeze(v) for k, v in data.items()})


def _nview(data):
    return StructureNumericDifference(**dict(data, baseline=_qview(data['baseline']), candidate=_qview(data['candidate'])))


def _fview(data):
    def observation(value): return StructureObservation(**{k: _freeze(v) for k, v in value.items()})
    return StructureFieldDifference(data['field'], observation(data['baseline']), observation(data['candidate']), data['status'])


def _aview(data):
    def group(value):
        return StructureAliasGroup(value['kind'], tuple(tuple(RegistrationEndpoint(**e) for e in member) for member in value['members']))
    return StructureAliasDifference(data['kind'], data['status'], data['matching_scope'],
        tuple(group(v) for v in data['baseline_groups']), tuple(group(v) for v in data['candidate_groups']),
        tuple(group(v) for v in data['removed_groups']),
        tuple(group(v) for v in data['added_groups']),
        *(tuple(RegistrationEndpoint(**e) for e in data[key]) for key in ('unmatched_baseline_endpoints',
          'unmatched_candidate_endpoints', 'unassessed_baseline_endpoints', 'unassessed_candidate_endpoints')))


@dataclass(frozen=True, init=False)
class ModelStructureComparison:
    """Inspect observed structural differences between two retained model snapshots.

    Args:
        data (dict[str, _typing.Any]): Detached plain-data payload accepted by the existing
            version/schema validator. Required.

    Returns:
        An instance of ModelStructureComparison with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        ModelStructureComparison(data)

    HYSo Guide: comparison.architecture
    Offline: hyso docs show comparison.architecture --lang en
    Open: hyso docs open comparison.architecture --lang en
    """
    __slots__ = ('_payload',)
    _payload: str

    def __init__(self, data: 'dict[str, _typing.Any]') -> 'None':
        """Initialize ModelStructureComparison with the supplied settings.

        Args:
            data (dict[str, _typing.Any]): Detached plain-data payload accepted by the existing
                version/schema validator. Required.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            ModelStructureComparison(data)

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        try:
            data = plain(data); raw = canonical(data)
            if len(raw.encode('utf-8')) > _MAX_BYTES: raise ModelStructureComparisonError('comparison_byte_limit')
            sources = data['source_snapshots']
            a, b = (ModelStructureSnapshot(sources[k]).to_dict() for k in ('baseline', 'candidate'))
            expected = _build(a, b)
            if raw != canonical(expected): raise ModelStructureComparisonError('comparison_derivation_or_integrity_mismatch')
        except ModelStructureComparisonError:
            raise
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError) as exc:
            raise ModelStructureComparisonError('invalid_structure_comparison') from exc
        object.__setattr__(self, '_payload', raw)

    def to_dict(self) -> 'dict[str, _typing.Any]':
        """Return a detached plain-data view of the stored state.

        Returns:
            A detached plain-data view of this validated object; native units and unavailable values are
            retained.

        Raises:
            No additional validation for a valid object; corrupt retained state or native copy/serialization
            failures propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_dict()

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return json.loads(self._payload)
    @property
    def content_sha256(self) -> '_typing.Any':
        """Return the canonical content digest of this retained record.

        Returns:
            The canonical SHA-256 digest string of this retained content.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.content_sha256

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return self.to_dict()['content_sha256']
    @property
    def baseline(self) -> '_typing.Any':
        """Read the retained baseline from this ModelStructureComparison record.

        Returns:
            The retained baseline value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.baseline

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return StructureSnapshotRef(**self.to_dict()['baseline'])
    @property
    def candidate(self) -> '_typing.Any':
        """Read the retained candidate from this ModelStructureComparison record.

        Returns:
            The retained candidate value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.candidate

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return StructureSnapshotRef(**self.to_dict()['candidate'])
    @property
    def summary_differences(self) -> '_typing.Any':
        """Read the retained summary differences from this ModelStructureComparison record.

        Returns:
            The retained summary differences value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.summary_differences

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return tuple(_nview(d) for d in self.to_dict()['summary_differences'])
    @property
    def module_type_differences(self) -> '_typing.Any':
        """Read the retained module type differences from this ModelStructureComparison record.

        Returns:
            The retained module type differences value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.module_type_differences

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return tuple(StructureDistributionDifference(v['name'], _nview(v['unique_modules']), _nview(v['registered_paths']))
                     for v in self.to_dict()['module_type_differences'])
    @property
    def category_differences(self) -> '_typing.Any':
        """Read the retained category differences from this ModelStructureComparison record.

        Returns:
            The retained category differences value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.category_differences

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return tuple(StructureDistributionDifference(v['name'], _nview(v['unique_modules']), _nview(v['registered_paths']))
                     for v in self.to_dict()['category_differences'])
    @property
    def dtype_differences(self) -> '_typing.Any':
        """Read the retained dtype differences from this ModelStructureComparison record.

        Returns:
            The retained dtype differences value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.dtype_differences

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return tuple(StructureDtypeDifference(v['kind'], v['dtype'], _nview(v['tensor_count']), _nview(v['elements']), _nview(v['logical_bytes']))
                     for v in self.to_dict()['dtype_differences'])
    @property
    def added_paths(self) -> '_typing.Any':
        """Read the retained added paths from this ModelStructureComparison record.

        Returns:
            The retained added paths value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.added_paths

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return tuple(self.to_dict()['added_paths'])
    @property
    def removed_paths(self) -> '_typing.Any':
        """Read the retained removed paths from this ModelStructureComparison record.

        Returns:
            The retained removed paths value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.removed_paths

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return tuple(self.to_dict()['removed_paths'])
    @property
    def changed_paths(self) -> '_typing.Any':
        """Read the retained changed paths from this ModelStructureComparison record.

        Returns:
            The retained changed paths value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.changed_paths

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return tuple(StructurePathDifference(v['path'], v['baseline_type'], v['candidate_type'],
            tuple(_fview(f) for f in v['metadata_changes']), tuple(_fview(f) for f in v['configuration_changes'])) for v in self.to_dict()['changed_paths'])
    @property
    def tensor_differences(self) -> '_typing.Any':
        """Read the retained tensor differences from this ModelStructureComparison record.

        Returns:
            The retained tensor differences value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.tensor_differences

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return tuple(StructureTensorDifference(v['owner_path'], v['kind'], v['name'], tuple(_fview(f) for f in v['changes']))
                     for v in self.to_dict()['tensor_differences'])
    @property
    def alias_differences(self) -> '_typing.Any':
        """Read the retained alias differences from this ModelStructureComparison record.

        Returns:
            The retained alias differences value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.alias_differences

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return tuple(_aview(v) for v in self.to_dict()['alias_differences'])
    @property
    def evidence_gaps(self) -> '_typing.Any':
        """Read the retained evidence gaps from this ModelStructureComparison record.

        Returns:
            The retained evidence gaps value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.evidence_gaps

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return tuple(StructureEvidenceGap(**{k: _freeze(x) for k, x in v.items()}) for v in self.to_dict()['evidence_gaps'])
    @property
    def provenance_differences(self) -> '_typing.Any':
        """Read the retained provenance differences from this ModelStructureComparison record.

        Returns:
            The retained provenance differences value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.provenance_differences

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return tuple(_fview(v) for v in self.to_dict()['provenance_differences'])
    @property
    def coverage(self) -> '_typing.Any':
        """Read the retained coverage from this ModelStructureComparison record.

        Returns:
            The retained coverage value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.coverage

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return self.to_dict()['coverage']
    @property
    def has_observed_changes(self) -> '_typing.Any':
        """Report whether the retained structural comparison contains observed differences.

        Returns:
            True when observed structural differences exist, otherwise False; missing evidence is separate.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.has_observed_changes

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return self.to_dict()['has_observed_changes']
    @property
    def limitations(self) -> '_typing.Any':
        """Read the retained limitations from this ModelStructureComparison record.

        Returns:
            The retained limitations value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.limitations

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return tuple(self.to_dict()['limitations'])
    @property
    def policy(self) -> '_typing.Any':
        """Read the retained policy from this ModelStructureComparison record.

        Returns:
            The retained policy value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.policy

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return StructureComparisonPolicy(**self.to_dict()['policy'])
    @property
    def baseline_snapshot(self) -> '_typing.Any':
        """Read the retained baseline snapshot from this ModelStructureComparison record.

        Returns:
            The retained baseline snapshot value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.baseline_snapshot

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return ModelStructureSnapshot(self.to_dict()['source_snapshots']['baseline'])
    @property
    def candidate_snapshot(self) -> '_typing.Any':
        """Read the retained candidate snapshot from this ModelStructureComparison record.

        Returns:
            The retained candidate snapshot value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.candidate_snapshot

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return ModelStructureSnapshot(self.to_dict()['source_snapshots']['candidate'])

    def to_json(self, path: '_typing.Any'=None) -> '_typing.Any':
        """Serialize this object using its native JSON format and overwrite policy.

        Args:
            path (_typing.Any): Filesystem path for the specific reader/exporter; no configuration code is
                executed. Default: None.

        Returns:
            A JSON string when path is None, otherwise None after writing the file.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_json()

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        if path is not None: export(path, self._payload + '\n')
        return self._payload + '\n'

    @classmethod
    def from_json(cls, path: '_typing.Any') -> 'ModelStructureComparison':
        """Read and validate this type from a local JSON file.

        Args:
            path (_typing.Any): Filesystem path for the specific reader/exporter; no configuration code is
                executed. Required.

        Returns:
            A validated ModelStructureComparison instance reconstructed from the supplied data.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            ModelStructureComparison.from_json(path)

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        data, _, _ = read_json(path, max_bytes=_MAX_BYTES + 1)
        return cls(data)


def compare_model_structures(baseline: '_typing.Any', candidate: '_typing.Any', *, max_entries: '_typing.Any'=_MAX_ENTRIES, max_bytes: '_typing.Any'=_MAX_BYTES) -> '_typing.Any':
    """Compare validated snapshots only; no live model or introspection path.

    Matching is literal registered path/name. Limits may lower the hard bounds;
    exceeding them raises instead of returning an incomplete/truncated result.

    Args:
        baseline (_typing.Any): Validated retained baseline source or predictions required by this
            comparison. Required.
        candidate (_typing.Any): Validated retained candidate source or predictions required by this
            comparison. Required.
        max_entries (_typing.Any): Upper bound on entries retained/accepted by this operation. Default:
            _MAX_ENTRIES.
        max_bytes (_typing.Any): Maximum accepted or retained byte count. Default: _MAX_BYTES.

    Returns:
        The retained compare model structures value, without recomputing measurements.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        compare_model_structures(baseline, candidate)

    HYSo Guide: comparison.architecture
    Offline: hyso docs show comparison.architecture --lang en
    Open: hyso docs open comparison.architecture --lang en
    """
    if type(baseline) is not ModelStructureSnapshot or type(candidate) is not ModelStructureSnapshot:
        raise ModelStructureComparisonError('validated_structure_snapshots_required')
    if (type(max_entries) is not int or not 1 <= max_entries <= _MAX_ENTRIES
            or type(max_bytes) is not int or not 1 <= max_bytes <= _MAX_BYTES):
        raise ModelStructureComparisonError('invalid_comparison_limit')
    try:
        a, b = (ModelStructureSnapshot(s.to_dict()).to_dict() for s in (baseline, candidate))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ModelStructureComparisonError('invalid_source_snapshot') from exc
    data = _build(a, b, max_entries)
    if len(canonical(data).encode('utf-8')) > max_bytes: raise ModelStructureComparisonError('comparison_byte_limit')
    return ModelStructureComparison(data)
