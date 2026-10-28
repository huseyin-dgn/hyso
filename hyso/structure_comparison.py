"""3B.4D.2: literal registered-path comparison of validated structure snapshots.

Only recorded metadata is read. No model objects, introspection, workloads,
checkpoint/state identities, execution graph or performance verdicts.
"""
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
    content_sha256: str
    format: str
    schema_version: int
    producer_method: str
    producer_version: int
    classification_version: int
    torch_version: str


@dataclass(frozen=True)
class StructureComparisonPolicy:
    id: str
    version: int
    matching: str
    aliases: str


@dataclass(frozen=True)
class ComparedQuantity:
    status: str
    value: int | None
    known_subtotal: int | None
    unsupported_tensor_ids: tuple
    source: str
    reason: str | None


@dataclass(frozen=True)
class StructureNumericDifference:
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
    status: str
    value: object
    reason: str | None


@dataclass(frozen=True)
class StructureFieldDifference:
    field: str
    baseline: StructureObservation
    candidate: StructureObservation
    status: str


@dataclass(frozen=True)
class StructurePathDifference:
    path: str
    baseline_type: str
    candidate_type: str
    metadata_changes: tuple
    configuration_changes: tuple


@dataclass(frozen=True)
class StructureTensorDifference:
    owner_path: str
    kind: str
    name: str
    changes: tuple


@dataclass(frozen=True)
class StructureDistributionDifference:
    name: str
    unique_modules: StructureNumericDifference
    registered_paths: StructureNumericDifference


@dataclass(frozen=True)
class StructureDtypeDifference:
    kind: str
    dtype: str | None
    tensor_count: StructureNumericDifference
    elements: StructureNumericDifference
    logical_bytes: StructureNumericDifference


@dataclass(frozen=True)
class RegistrationEndpoint:
    owner_path: str
    kind: str
    name: str


@dataclass(frozen=True)
class StructureAliasGroup:
    kind: str
    members: tuple


@dataclass(frozen=True)
class StructureAliasDifference:
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
    __slots__ = ('_payload',)
    _payload: str

    def __init__(self, data):
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

    def to_dict(self): return json.loads(self._payload)
    @property
    def content_sha256(self): return self.to_dict()['content_sha256']
    @property
    def baseline(self): return StructureSnapshotRef(**self.to_dict()['baseline'])
    @property
    def candidate(self): return StructureSnapshotRef(**self.to_dict()['candidate'])
    @property
    def summary_differences(self): return tuple(_nview(d) for d in self.to_dict()['summary_differences'])
    @property
    def module_type_differences(self):
        return tuple(StructureDistributionDifference(v['name'], _nview(v['unique_modules']), _nview(v['registered_paths']))
                     for v in self.to_dict()['module_type_differences'])
    @property
    def category_differences(self):
        return tuple(StructureDistributionDifference(v['name'], _nview(v['unique_modules']), _nview(v['registered_paths']))
                     for v in self.to_dict()['category_differences'])
    @property
    def dtype_differences(self):
        return tuple(StructureDtypeDifference(v['kind'], v['dtype'], _nview(v['tensor_count']), _nview(v['elements']), _nview(v['logical_bytes']))
                     for v in self.to_dict()['dtype_differences'])
    @property
    def added_paths(self): return tuple(self.to_dict()['added_paths'])
    @property
    def removed_paths(self): return tuple(self.to_dict()['removed_paths'])
    @property
    def changed_paths(self):
        return tuple(StructurePathDifference(v['path'], v['baseline_type'], v['candidate_type'],
            tuple(_fview(f) for f in v['metadata_changes']), tuple(_fview(f) for f in v['configuration_changes'])) for v in self.to_dict()['changed_paths'])
    @property
    def tensor_differences(self):
        return tuple(StructureTensorDifference(v['owner_path'], v['kind'], v['name'], tuple(_fview(f) for f in v['changes']))
                     for v in self.to_dict()['tensor_differences'])
    @property
    def alias_differences(self): return tuple(_aview(v) for v in self.to_dict()['alias_differences'])
    @property
    def evidence_gaps(self):
        return tuple(StructureEvidenceGap(**{k: _freeze(x) for k, x in v.items()}) for v in self.to_dict()['evidence_gaps'])
    @property
    def provenance_differences(self): return tuple(_fview(v) for v in self.to_dict()['provenance_differences'])
    @property
    def coverage(self): return self.to_dict()['coverage']
    @property
    def has_observed_changes(self): return self.to_dict()['has_observed_changes']
    @property
    def limitations(self): return tuple(self.to_dict()['limitations'])
    @property
    def policy(self): return StructureComparisonPolicy(**self.to_dict()['policy'])
    @property
    def baseline_snapshot(self): return ModelStructureSnapshot(self.to_dict()['source_snapshots']['baseline'])
    @property
    def candidate_snapshot(self): return ModelStructureSnapshot(self.to_dict()['source_snapshots']['candidate'])

    def to_json(self, path=None):
        if path is not None: export(path, self._payload + '\n')
        return self._payload + '\n'

    @classmethod
    def from_json(cls, path):
        data, _, _ = read_json(path, max_bytes=_MAX_BYTES + 1)
        return cls(data)


def compare_model_structures(baseline, candidate, *, max_entries=_MAX_ENTRIES, max_bytes=_MAX_BYTES):
    """Compare validated snapshots only; no live model or introspection path.

    Matching is literal registered path/name. Limits may lower the hard bounds;
    exceeding them raises instead of returning an incomplete/truncated result.
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
