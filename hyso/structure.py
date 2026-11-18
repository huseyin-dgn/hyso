"""Versioned static model metadata. No forward graph or state/checkpoint identity."""
from collections import Counter, deque
from dataclasses import dataclass, fields
import json
import math

from .analysis._codec import canonical, digest, export, plain, read_json

__all__ = ['capture_model_structure', 'ModelStructureSnapshot', 'ModelStructureError',
           'StructureQuantity', 'StructureField', 'TensorStructure', 'TensorSlot',
           'ModuleStructure', 'ModuleEdge', 'ModulePath', 'StructureSummary',
           'StructureTypeCount', 'StructureDtypeCount', 'StorageGroup']

_CATEGORIES = ('container', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
    'conv_transpose2d', 'conv_transpose3d', 'linear', 'normalization', 'recurrent',
    'attention', 'transformer', 'embedding', 'activation', 'pooling', 'regularization',
    'reshape', 'identity', 'custom')
_LIMITATIONS = (
    'Registered structure only; no forward/backward, runtime shapes, execution order or dynamic control flow.',
    'Functional operations and unregistered Python state may be absent; registered module counts are not executed layer counts.',
    'Exact supported standard types have static configuration; custom/subclass semantics are unsupported, not inferred.',
    'requires_grad is metadata, not proof of actual training behavior.',
    'Logical bytes count unique tensor objects, not unique storage bytes, GPU allocation, RSS or checkpoint file size.',
    'Storage groups identify observed shared storage, not byte-range overlap or physical memory use.',
    'Structure SHA is neither model-state SHA nor checkpoint SHA; equal structure does not prove equal weights or behavior.',
    'Existing trusted objects only; no source import, checkpoint load or sandbox for hostile/concurrently mutated Python objects.',
    'Imported metadata validation checks consistency, not producer authenticity or a live model association.',
)
_SOURCES = dict(modules='nn.Module registration dictionaries; exact Python type identity',
    configuration='Exact-type allowlist of stored standard module fields; no properties/callables/repr',
    tensors='Standard Tensor/Parameter shape, dtype, layout, device and requires_grad metadata; no values',
    persistence='Per-owner _non_persistent_buffers_set membership',
    aliases='Object identity mapped to deterministic local IDs; dense untyped-storage handle equality',
    summary='Derived from unique registered objects and expanded noncyclic module paths')
_DTYPE_BYTES = {'torch.bool': 1, 'torch.uint8': 1, 'torch.int8': 1, 'torch.int16': 2,
    'torch.int32': 4, 'torch.int64': 8, 'torch.uint16': 2, 'torch.uint32': 4, 'torch.uint64': 8,
    'torch.float16': 2, 'torch.bfloat16': 2, 'torch.float32': 4, 'torch.float64': 8,
    'torch.complex32': 4, 'torch.complex64': 8, 'torch.complex128': 16,
    'torch.float8_e4m3fn': 1, 'torch.float8_e5m2': 1, 'torch.float8_e4m3fnuz': 1, 'torch.float8_e5m2fnuz': 1}


class ModelStructureError(ValueError):
    """Invalid structure or explicit capture/serialization limit; never truncates."""


@dataclass(frozen=True)
class StructureQuantity:
    status: str
    value: int | None
    known_subtotal: int
    unsupported_tensor_ids: tuple


@dataclass(frozen=True)
class StructureField:
    name: str
    status: str
    value: object
    reason: str | None
    source: str


@dataclass(frozen=True)
class TensorSlot:
    name: str
    tensor_id: str | None
    persistent: bool | None


@dataclass(frozen=True)
class ModuleEdge:
    name: str
    module_id: str | None


@dataclass(frozen=True)
class ModulePath:
    path: str
    parent_path: str | None
    module_id: str
    depth: int
    shared: bool
    cycle: bool


@dataclass(frozen=True)
class TensorStructure:
    tensor_id: str
    qualified_type: str
    status: str
    reason: str | None
    shape: tuple | None
    dtype: str | None
    device: str | None
    layout: str | None
    requires_grad: bool | None
    elements: int | None
    element_size: int | None
    logical_bytes: int | None
    stride: tuple | None
    storage_offset: int | None
    storage_id: str | None
    storage_status: str
    storage_reason: str | None


@dataclass(frozen=True)
class ModuleStructure:
    module_id: str
    primary_path: str
    qualified_type: str
    category: str
    configuration_status: str
    configuration: tuple
    children: tuple
    parameters: tuple
    buffers: tuple
    direct_parameter_elements: StructureQuantity
    direct_buffer_elements: StructureQuantity

    @property
    def direct_parameter_tensor_count(self):
        return len({s.tensor_id for s in self.parameters if s.tensor_id is not None})

    @property
    def direct_buffer_tensor_count(self):
        return len({s.tensor_id for s in self.buffers if s.tensor_id is not None})


@dataclass(frozen=True)
class StructureTypeCount:
    name: str
    unique_modules: int
    registered_paths: int


@dataclass(frozen=True)
class StructureDtypeCount:
    dtype: str | None
    tensor_count: int
    elements: StructureQuantity
    logical_bytes: StructureQuantity


@dataclass(frozen=True)
class StorageGroup:
    storage_id: str
    tensor_ids: tuple

    @property
    def distinct_objects_share_storage(self):
        return len(self.tensor_ids) > 1


@dataclass(frozen=True)
class StructureSummary:
    unique_modules: int
    registered_module_paths: int
    unique_parameter_tensors: int
    unique_buffer_tensors: int
    cross_category_tensors: int
    parameter_elements: StructureQuantity
    trainable_elements: StructureQuantity
    frozen_elements: StructureQuantity
    parameter_logical_bytes: StructureQuantity
    buffer_elements: StructureQuantity
    buffer_logical_bytes: StructureQuantity
    tensor_union_logical_bytes: StructureQuantity
    module_types: tuple
    categories: tuple
    parameter_dtypes: tuple
    buffer_dtypes: tuple
    storage_groups: tuple
    storage_alias_coverage: str


def _quantity(tensors, ids, key):
    subtotal, unknown = 0, []
    for identity in ids:
        value = tensors[identity][key]
        if value is None:
            unknown.append(identity)
        else:
            subtotal += value
    return dict(status='unsupported' if unknown else 'native', value=None if unknown else subtotal,
                known_subtotal=subtotal, unsupported_tensor_ids=unknown)


def _paths(modules, max_paths=50000, max_depth=128):
    by_id = {m['module_id']: m for m in modules}
    queue = deque([('', None, 'm0', 0, ())]); result = []; first = {}
    while queue:
        path, parent, identity, depth, ancestors = queue.popleft()
        if depth > max_depth or len(result) + len(queue) >= max_paths:
            raise ModelStructureError('module_path_or_depth_limit')
        if identity not in by_id:
            raise ModelStructureError('unknown_module_reference')
        cycle = identity in ancestors
        first.setdefault(identity, path)
        result.append(dict(path=path, parent_path=parent, module_id=identity, depth=depth,
                           shared=path != first[identity], cycle=cycle))
        if not cycle:
            for child in by_id[identity]['children']:
                if child['module_id'] is not None:
                    if len(result) + len(queue) >= max_paths:
                        raise ModelStructureError('module_path_or_depth_limit')
                    child_path = path + '.' + child['name'] if path else child['name']
                    queue.append((child_path, path, child['module_id'], depth + 1, ancestors + (identity,)))
    if set(first) != set(by_id) or any(m['primary_path'] != first[m['module_id']] for m in modules):
        raise ModelStructureError('unreachable_module_or_primary_path_mismatch')
    return result


def _summary(modules, tensor_rows, paths):
    tensors = {t['tensor_id']: t for t in tensor_rows}
    p = {s['tensor_id'] for m in modules for s in m['parameters'] if s['tensor_id'] is not None}
    b = {s['tensor_id'] for m in modules for s in m['buffers'] if s['tensor_id'] is not None}
    if p | b != set(tensors):
        raise ModelStructureError('unreferenced_or_unknown_tensor')
    # Tensor ID order is the producer's first-registration order, never pointer order.
    position = {t['tensor_id']: i for i, t in enumerate(tensor_rows)}
    ordered = lambda ids: sorted(ids, key=position.__getitem__)
    qty = lambda ids, key: _quantity(tensors, ordered(ids), key)
    def dtype_counts(ids):
        groups = {}
        for identity in ordered(ids):
            groups.setdefault(tensors[identity]['dtype'], []).append(identity)
        return [dict(dtype=d, tensor_count=len(groups[d]),
                     elements=qty(groups[d], 'elements'), logical_bytes=qty(groups[d], 'logical_bytes'))
                for d in sorted(groups, key=lambda d: '' if d is None else d)]
    by_id = {m['module_id']: m for m in modules}
    def type_counts(key):
        unique = Counter(m[key] for m in modules)
        expanded = Counter(by_id[p['module_id']][key] for p in paths)
        return [dict(name=k, unique_modules=unique[k], registered_paths=expanded[k]) for k in sorted(unique)]
    groups = {}
    for t in tensor_rows:
        if t['storage_id'] is not None:
            groups.setdefault(t['storage_id'], []).append(t['tensor_id'])
    return dict(unique_modules=len(modules), registered_module_paths=len(paths),
        unique_parameter_tensors=len(p), unique_buffer_tensors=len(b), cross_category_tensors=len(p & b),
        parameter_elements=qty(p, 'elements'),
        trainable_elements=qty({i for i in p if tensors[i]['requires_grad'] is not False}, 'elements'),
        frozen_elements=qty({i for i in p if tensors[i]['requires_grad'] is not True}, 'elements'),
        parameter_logical_bytes=qty(p, 'logical_bytes'), buffer_elements=qty(b, 'elements'),
        buffer_logical_bytes=qty(b, 'logical_bytes'), tensor_union_logical_bytes=qty(p | b, 'logical_bytes'),
        module_types=type_counts('qualified_type'), categories=type_counts('category'),
        parameter_dtypes=dtype_counts(p), buffer_dtypes=dtype_counts(b),
        storage_groups=[dict(storage_id=k, tensor_ids=v) for k, v in groups.items()],
        storage_alias_coverage='native' if all(t['storage_status'] == 'native' for t in tensor_rows) else 'missing')


def _freeze(value):
    return tuple(_freeze(v) for v in value) if type(value) is list else value


def _static_value(value, depth=0):
    if depth > 8:
        return False
    return (value is None or type(value) in (bool, int, float) or
            type(value) is str and len(value) <= 1024 or type(value) is list
            and len(value) <= 128 and all(_static_value(v, depth + 1) for v in value))


def _tensor_view(data):
    return TensorStructure(**{k: _freeze(v) for k, v in data.items()})


def _module_view(data):
    d = dict(data)
    d['configuration'] = tuple(StructureField(**{k: _freeze(v) for k, v in f.items()}) for f in d['configuration'])
    d['children'] = tuple(ModuleEdge(**e) for e in d['children'])
    for key in ('parameters', 'buffers'):
        d[key] = tuple(TensorSlot(**s) for s in d[key])
    for key in ('direct_parameter_elements', 'direct_buffer_elements'):
        d[key] = StructureQuantity(**{k: _freeze(v) for k, v in d[key].items()})
    return ModuleStructure(**d)


def _summary_view(data):
    d = dict(data)
    for key, value in list(d.items()):
        if type(value) is dict:
            d[key] = StructureQuantity(**{k: _freeze(v) for k, v in value.items()})
    for key in ('module_types', 'categories'):
        d[key] = tuple(StructureTypeCount(**v) for v in d[key])
    for key in ('parameter_dtypes', 'buffer_dtypes'):
        d[key] = tuple(StructureDtypeCount(v['dtype'], v['tensor_count'],
            StructureQuantity(**{k: _freeze(x) for k, x in v['elements'].items()}),
            StructureQuantity(**{k: _freeze(x) for k, x in v['logical_bytes'].items()})) for v in d[key])
    d['storage_groups'] = tuple(StorageGroup(v['storage_id'], tuple(v['tensor_ids'])) for v in d['storage_groups'])
    return StructureSummary(**d)


def _validate(data):
    def require(condition):
        if not condition:
            raise ModelStructureError('invalid_model_structure_snapshot')
    require(set(data) == {'format', 'schema_version', 'producer', 'source', 'modules', 'tensors',
                          'paths', 'summary', 'limitations', 'content_sha256'})
    require(data['format'] == 'hyso.model_structure' and type(data['schema_version']) is int and data['schema_version'] == 1)
    producer = data['producer']
    require(set(producer) == {'method', 'version', 'classification_version', 'torch_version'})
    require(producer['method'] == 'registered_nn_module_metadata' and type(producer['version']) is int
            and producer['version'] == 1 and type(producer['classification_version']) is int
            and producer['classification_version'] == 1 and type(producer['torch_version']) is str and bool(producer['torch_version']))
    require(data['source'] == dict(kind='existing_trusted_torch_nn_module', authority='native', fields=_SOURCES))
    require(data['limitations'] == list(_LIMITATIONS))
    modules, tensors = data['modules'], data['tensors']
    require(type(modules) is list and 1 <= len(modules) <= 10000 and type(tensors) is list and len(tensors) <= 100000)
    tensor_ids = {t['tensor_id'] for t in tensors}
    require(len(tensor_ids) == len(tensors))
    for index, t in enumerate(tensors):
        _tensor_view(t)
        require(set(t) == {f.name for f in fields(TensorStructure)} and t['tensor_id'] == 't' + str(index)
                and type(t['qualified_type']) is str and bool(t['qualified_type']))
        require(t['status'] in ('native', 'unsupported') and t['storage_status'] in ('native', 'missing', 'unsupported'))
        require(t['requires_grad'] is None or type(t['requires_grad']) is bool)
        for key in ('dtype', 'device', 'layout'):
            require(t[key] is None or type(t[key]) is str and bool(t[key]))
        require(t['shape'] is None or type(t['shape']) is list and len(t['shape']) <= 128
                and all(type(n) is int and n >= 0 for n in t['shape']))
        if t['status'] == 'native':
            require(t['shape'] is not None and t['layout'] == 'torch.strided' and t['dtype'] is not None
                    and t['device'] is not None and type(t['requires_grad']) is bool and t['reason'] is None)
            require(type(t['elements']) is int and t['elements'] == math.prod(t['shape'])
                    and type(t['element_size']) is int and t['element_size'] == _DTYPE_BYTES.get(t['dtype'])
                    and type(t['logical_bytes']) is int and t['logical_bytes'] == t['elements'] * t['element_size'])
        else:
            require(type(t['reason']) is str and bool(t['reason']) and t['elements'] is None
                    and t['logical_bytes'] is None and t['element_size'] is None)
        if t['storage_status'] == 'native':
            require(t['status'] == 'native' and t['device'] != 'meta'
                    and type(t['storage_id']) is str and t['storage_id'].startswith('s')
                    and t['storage_id'][1:].isdigit() and t['storage_reason'] is None
                    and type(t['storage_offset']) is int and t['storage_offset'] >= 0
                    and type(t['stride']) is list and t['shape'] is not None and len(t['stride']) == len(t['shape'])
                    and all(type(n) is int for n in t['stride']))
        else:
            require(t['storage_id'] is None and t['storage_offset'] is None and t['stride'] is None
                    and type(t['storage_reason']) is str and bool(t['storage_reason']))
    lookup = {t['tensor_id']: t for t in tensors}
    registrations = 0
    from ._structure_native import standard_schema
    for index, m in enumerate(modules):
        _module_view(m)
        require(set(m) == {f.name for f in fields(ModuleStructure)} and m['module_id'] == 'm' + str(index)
                and type(m['primary_path']) is str and type(m['qualified_type']) is str and bool(m['qualified_type'])
                and m['category'] in _CATEGORIES and m['configuration_status'] in ('native', 'unsupported'))
        require(type(m['configuration']) is list and len(m['configuration']) <= 64)
        require(len({f['name'] for f in m['configuration']}) == len(m['configuration']))
        if m['category'] == 'custom':
            require(m['configuration_status'] == 'unsupported' and m['configuration'] == [])
        else:
            schema = standard_schema(m['qualified_type'])
            require(schema is not None and schema[0] == m['category']
                    and [(f['name'], f['source']) for f in m['configuration']] == schema[1]
                    and (m['configuration_status'] == 'native') == all(f['status'] == 'native' for f in m['configuration']))
        for f in m['configuration']:
            require(f['status'] in ('native', 'unsupported', 'missing') and type(f['source']) is str
                    and type(f['name']) is str and bool(f['name']))
            require(f['reason'] is None and _static_value(f['value']) if f['status'] == 'native'
                    else f['value'] is None and type(f['reason']) is str)
        for kind in ('children', 'parameters', 'buffers'):
            slots = m[kind]; require(type(slots) is list)
            registrations += len(slots)
            require(registrations <= 500000 and len({s['name'] for s in slots}) == len(slots))
            for slot in slots:
                require(type(slot['name']) is str and bool(slot['name']) and '.' not in slot['name'])
                if kind != 'children':
                    require(slot['tensor_id'] is None or slot['tensor_id'] in tensor_ids)
                    require(slot['persistent'] is None if kind == 'parameters' else type(slot['persistent']) is bool)
        for kind in ('parameter', 'buffer'):
            ids = list(dict.fromkeys(s['tensor_id'] for s in m[kind + 's'] if s['tensor_id'] is not None))
            require(canonical(m['direct_' + kind + '_elements']) == canonical(_quantity(lookup, ids, 'elements')))
    require(canonical(data['paths']) == canonical(_paths(modules)))
    require(canonical(data['summary']) == canonical(_summary(modules, tensors, data['paths'])))
    _summary_view(data['summary'])
    require(data['content_sha256'] == digest({k: v for k, v in data.items() if k != 'content_sha256'}))


@dataclass(frozen=True, init=False)
class ModelStructureSnapshot:
    __slots__ = ('_payload',)
    _payload: str

    def __init__(self, data):
        try:
            data = plain(data)
            _validate(data)
            raw = canonical(data)
            if len(raw.encode('utf-8')) > 16 * 1024**2:
                raise ModelStructureError('snapshot_byte_limit')
        except ModelStructureError:
            raise
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError) as exc:
            raise ModelStructureError('invalid_model_structure_snapshot') from exc
        object.__setattr__(self, '_payload', raw)

    def to_dict(self): return json.loads(self._payload)

    @property
    def content_sha256(self): return self.to_dict()['content_sha256']

    @property
    def modules(self): return tuple(_module_view(m) for m in self.to_dict()['modules'])

    @property
    def tensors(self): return tuple(_tensor_view(t) for t in self.to_dict()['tensors'])

    @property
    def paths(self): return tuple(ModulePath(**p) for p in self.to_dict()['paths'])

    @property
    def summary(self): return _summary_view(self.to_dict()['summary'])

    @property
    def limitations(self): return tuple(self.to_dict()['limitations'])

    def to_json(self, path=None):
        if path is not None: export(path, self._payload + '\n')
        return self._payload + '\n'

    @classmethod
    def from_json(cls, path):
        data, _, _ = read_json(path, max_bytes=16 * 1024**2 + 1)
        return cls(data)


def capture_model_structure(model, *, max_modules=10000, max_paths=50000,
                            max_tensors=100000, max_depth=128, max_registrations=500000,
                            max_bytes=16 * 1024**2):
    """Read registered metadata of an existing trusted, quiescent nn.Module.

    Never executes forward, mode/state hooks, value copies or device transfers.
    Limits fail explicitly; shared/cyclic module paths are never silently pruned.
    """
    from ._structure_native import capture
    return capture(model, max_modules=max_modules, max_paths=max_paths, max_tensors=max_tensors,
                   max_depth=max_depth, max_registrations=max_registrations, max_bytes=max_bytes)
