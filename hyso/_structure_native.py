"""Metadata-only producer for model structure v1; exact standard types only."""
from collections import OrderedDict
import math

import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter, UninitializedBuffer

from .analysis._codec import canonical, digest, plain
from .structure import (ModelStructureError, ModelStructureSnapshot, _LIMITATIONS, _SOURCES,
                        _DTYPE_BYTES, _paths, _quantity, _summary)


# Invoke native descriptors directly, bypassing subclass __dict__ properties and
# __getattribute__. This is introspection of trusted objects, not a code sandbox.
_MODULE_DICT = nn.Module.__dict__['__dict__']
_TYPE_MRO = type.__dict__['__mro__']
_TYPE_QUALNAME = type.__dict__['__qualname__']
_TYPE_MODULE = type.__dict__['__module__']


def _qualified_type(obj):
    cls = type(obj)
    module = _TYPE_MODULE.__get__(cls)
    name = _TYPE_QUALNAME.__get__(cls)
    if type(module) is not str or type(name) is not str:
        raise ModelStructureError('invalid_python_type_identity')
    return module + '.' + name


def _inherits(obj, cls):
    # Avoid custom metaclass __instancecheck__ and Tensor __torch_function__.
    return any(base is cls for base in _TYPE_MRO.__get__(type(obj)))


_SPECS = {}
def _spec(classes, category, fields=()):
    for cls in classes:
        _SPECS[cls] = (category, fields)


_spec((nn.Module, nn.Sequential, nn.ModuleList, nn.ModuleDict, nn.ParameterList, nn.ParameterDict), 'container')
for dim in (1, 2, 3):
    _spec((getattr(nn, 'Conv' + str(dim) + 'd'),), 'conv' + str(dim) + 'd',
          ('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'padding_mode', '@bias'))
    _spec((getattr(nn, 'ConvTranspose' + str(dim) + 'd'),), 'conv_transpose' + str(dim) + 'd',
          ('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'padding_mode', 'output_padding', '@bias'))
_spec((nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear), 'linear', ('in_features', 'out_features', '@bias'))
_spec((nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
       nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d), 'normalization',
      ('num_features', 'eps', 'momentum', 'affine', 'track_running_stats'))
_spec((nn.LayerNorm,), 'normalization', ('normalized_shape', 'eps', 'elementwise_affine', '@bias'))
_spec((nn.GroupNorm,), 'normalization', ('num_groups', 'num_channels', 'eps', 'affine'))
_spec((nn.LocalResponseNorm,), 'normalization', ('size', 'alpha', 'beta', 'k'))
_spec((nn.RNN,), 'recurrent', ('input_size', 'hidden_size', 'num_layers', 'bias', 'batch_first', 'dropout', 'bidirectional', 'nonlinearity'))
_spec((nn.LSTM,), 'recurrent', ('input_size', 'hidden_size', 'num_layers', 'bias', 'batch_first', 'dropout', 'bidirectional', 'proj_size'))
_spec((nn.GRU,), 'recurrent', ('input_size', 'hidden_size', 'num_layers', 'bias', 'batch_first', 'dropout', 'bidirectional'))
_spec((nn.RNNCell,), 'recurrent', ('input_size', 'hidden_size', 'bias', 'nonlinearity'))
_spec((nn.LSTMCell, nn.GRUCell), 'recurrent', ('input_size', 'hidden_size', 'bias'))
_spec((nn.MultiheadAttention,), 'attention', ('embed_dim', 'num_heads', 'dropout', 'batch_first', 'add_zero_attn', 'kdim', 'vdim'))
_spec((nn.Transformer,), 'transformer', ('d_model', 'nhead', 'batch_first'))
_spec((nn.TransformerEncoder,), 'transformer', ('num_layers', 'enable_nested_tensor'))
_spec((nn.TransformerDecoder,), 'transformer', ('num_layers',))
_spec((nn.TransformerEncoderLayer, nn.TransformerDecoderLayer), 'transformer', ('norm_first', 'activation'))
_spec((nn.Embedding,), 'embedding', ('num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm', 'norm_type', 'scale_grad_by_freq', 'sparse'))
_spec((nn.EmbeddingBag,), 'embedding', ('num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm', 'norm_type', 'scale_grad_by_freq', 'sparse', 'mode', 'include_last_offset'))
_spec((nn.ReLU, nn.ReLU6, nn.SiLU, nn.Hardswish, nn.Hardsigmoid, nn.SELU), 'activation', ('inplace',))
_spec((nn.Tanh, nn.Sigmoid, nn.Softsign, nn.Tanhshrink, nn.Softmax2d), 'activation')
_spec((nn.Hardtanh,), 'activation', ('min_val', 'max_val', 'inplace'))
_spec((nn.LeakyReLU,), 'activation', ('negative_slope', 'inplace'))
_spec((nn.ELU, nn.CELU), 'activation', ('alpha', 'inplace'))
_spec((nn.PReLU,), 'activation', ('num_parameters',))
_spec((nn.GELU,), 'activation', ('approximate',))
_spec((nn.Softmax, nn.LogSoftmax, nn.Softmin), 'activation', ('dim',))
_spec((nn.Softplus,), 'activation', ('beta', 'threshold'))
_spec((nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d), 'pooling',
      ('kernel_size', 'stride', 'padding', 'dilation', 'return_indices', 'ceil_mode'))
_spec((nn.AvgPool1d,), 'pooling',
      ('kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad'))
_spec((nn.AvgPool2d, nn.AvgPool3d), 'pooling',
      ('kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad', 'divisor_override'))
_spec((nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d), 'pooling', ('output_size',))
_spec((nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d), 'pooling', ('output_size', 'return_indices'))
_spec((nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout), 'regularization', ('p', 'inplace'))
_spec((nn.Flatten,), 'reshape', ('start_dim', 'end_dim'))
_spec((nn.Unflatten,), 'reshape', ('dim', 'unflattened_size'))
_spec((nn.Identity,), 'identity')


def standard_schema(qualified_type):
    """Closed reader schema; never resolve/import a source-declared type name."""
    for cls, (category, fields) in _SPECS.items():
        if _TYPE_MODULE.__get__(cls) + '.' + _TYPE_QUALNAME.__get__(cls) == qualified_type:
            return category, [('bias' if k == '@bias' else k,
                              '_parameters.bias presence' if k == '@bias' else 'module.__dict__.' + k) for k in fields]
    return None


def _static(value, depth=0):
    if depth > 8:
        raise ValueError()
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is str and len(value) <= 1024:
        return value
    if type(value) in (tuple, list) and len(value) <= 128:
        return [_static(v, depth + 1) for v in value]
    raise ValueError()


def _configuration(module, attributes, parameters):
    spec = next((v for cls, v in _SPECS.items() if type(module) is cls), None)
    if spec is None:
        return 'custom', 'unsupported', []
    category, fields = spec
    result = []
    for key in fields:
        source = '_parameters.bias presence' if key == '@bias' else 'module.__dict__.' + key
        name = 'bias' if key == '@bias' else key
        if key != '@bias' and key not in attributes:
            result.append(dict(name=name, status='missing', value=None, reason='field_not_stored', source=source))
            continue
        try:
            value = parameters.get('bias') is not None if key == '@bias' else _static(attributes[key])
            result.append(dict(name=name, status='native', value=value, reason=None, source=source))
        except ValueError:
            result.append(dict(name=name, status='unsupported', value=None, reason='non_plain_static_field', source=source))
    return category, 'native' if all(f['status'] == 'native' for f in result) else 'unsupported', result


def _tensor(tensor, identity, storages):
    row = dict(tensor_id=identity, qualified_type=_qualified_type(tensor), status='unsupported',
        reason='custom_tensor_subclass', shape=None, dtype=None, device=None, layout=None,
        requires_grad=None, elements=None, element_size=None, logical_bytes=None,
        stride=None, storage_offset=None, storage_id=None, storage_status='unsupported',
        storage_reason='unsupported_tensor_metadata')
    if not any(type(tensor) is cls for cls in (torch.Tensor, nn.Parameter, UninitializedParameter, UninitializedBuffer)):
        return row
    row.update(dtype=str(tensor.dtype), device=str(tensor.device), layout=str(tensor.layout),
               requires_grad=bool(tensor.requires_grad))
    if type(tensor) is UninitializedParameter or type(tensor) is UninitializedBuffer:
        row['reason'] = 'uninitialized_lazy_tensor'
        return row
    if tensor.is_nested:
        row['reason'] = 'nested_tensor'
        return row
    row['shape'] = list(tensor.shape)
    if tensor.layout != torch.strided:
        row['reason'] = 'non_strided_tensor'
        return row
    if tensor.is_quantized:
        row['reason'] = 'quantized_tensor'
        return row
    if row['dtype'] not in _DTYPE_BYTES or tensor.element_size() != _DTYPE_BYTES[row['dtype']]:
        row['reason'] = 'unsupported_dtype'
        return row
    count = math.prod(row['shape'])
    row.update(status='native', reason=None, elements=count, element_size=tensor.element_size(),
               logical_bytes=count * tensor.element_size(), storage_status='missing')
    if tensor.device.type == 'meta':
        row['storage_reason'] = 'meta_has_no_materialized_storage'
        return row
    try:
        # Storage handle equality is metadata only: no data_ptr, reads, copies,
        # allocation probing or CUDA APIs. Never export raw handles/addresses.
        storage = torch.Tensor.untyped_storage(tensor)
        if type(storage) is not torch.UntypedStorage:
            raise ValueError()
        key = storage._cdata
        row.update(storage_id=storages.setdefault(key, 's' + str(len(storages))),
                   storage_offset=tensor.storage_offset(), stride=list(tensor.stride()),
                   storage_status='native', storage_reason=None)
    except (ValueError, RuntimeError, NotImplementedError):
        row['storage_reason'] = 'storage_identity_unavailable'
    return row


def capture(model, **limits):
    ceilings = dict(max_modules=10000, max_paths=50000, max_tensors=100000,
                    max_depth=128, max_registrations=500000, max_bytes=16 * 1024**2)
    if any(type(limits[k]) is not int or not 1 <= limits[k] <= cap for k, cap in ceilings.items()):
        raise ModelStructureError('invalid_capture_limit')
    if not _inherits(model, nn.Module):
        raise ModelStructureError('existing_nn_module_required')
    objects = [model]; module_ids = {id(model): 'm0'}; primary = {id(model): ''}; depths = {id(model): 0}
    tensor_ids = {}; tensors = []; tensor_objects = []; lookup = {}; storages = {}; modules = []; observations = []; registrations = 0
    for module in objects:
        attributes = _MODULE_DICT.__get__(module)
        registries = []
        for key in ('_modules', '_parameters', '_buffers'):
            mapping = attributes.get(key)
            if type(mapping) not in (dict, OrderedDict):
                raise ModelStructureError('unsupported_module_registry')
            registries.append(tuple(mapping.items()))
        children, parameters, buffers = registries
        persistence = attributes.get('_non_persistent_buffers_set')
        if type(persistence) is not set or any(type(k) is not str for k in persistence):
            raise ModelStructureError('unsupported_buffer_persistence_registry')
        registrations += sum(map(len, registries))
        if registrations > limits['max_registrations']:
            raise ModelStructureError('registration_limit')
        for registry in registries:
            if any(type(k) is not str or not k or '.' in k for k, _ in registry):
                raise ModelStructureError('invalid_registration_name')
        identity = module_ids[id(module)]; path = primary[id(module)]
        edges = []
        for name, child in children:
            if child is not None:
                if not _inherits(child, nn.Module):
                    raise ModelStructureError('invalid_child_module')
                if id(child) not in module_ids:
                    if len(objects) >= limits['max_modules']:
                        raise ModelStructureError('module_limit')
                    if depths[id(module)] + 1 > limits['max_depth']:
                        raise ModelStructureError('module_path_or_depth_limit')
                    module_ids[id(child)] = 'm' + str(len(objects)); objects.append(child)
                    primary[id(child)] = path + '.' + name if path else name
                    depths[id(child)] = depths[id(module)] + 1
            edges.append(dict(name=name, module_id=module_ids[id(child)] if child is not None else None))
        slots = {}
        for kind, registry in (('parameters', parameters), ('buffers', buffers)):
            slots[kind] = []
            for name, tensor in registry:
                if tensor is not None:
                    if not _inherits(tensor, torch.Tensor) or kind == 'parameters' and type(tensor) is torch.Tensor:
                        raise ModelStructureError('invalid_registered_tensor')
                    if id(tensor) not in tensor_ids:
                        if len(tensors) >= limits['max_tensors']:
                            raise ModelStructureError('tensor_limit')
                        tensor_ids[id(tensor)] = 't' + str(len(tensors))
                        row = _tensor(tensor, tensor_ids[id(tensor)], storages)
                        tensors.append(row); tensor_objects.append(tensor); lookup[row['tensor_id']] = row
                slots[kind].append(dict(name=name, tensor_id=tensor_ids[id(tensor)] if tensor is not None else None,
                                        persistent=name not in persistence if kind == 'buffers' else None))
        category, config_status, config = _configuration(module, attributes, dict(parameters))
        direct = {}
        for kind in ('parameter', 'buffer'):
            ids = list(dict.fromkeys(s['tensor_id'] for s in slots[kind + 's'] if s['tensor_id'] is not None))
            direct['direct_' + kind + '_elements'] = _quantity(lookup, ids, 'elements')
        modules.append(dict(module_id=identity, primary_path=path, qualified_type=_qualified_type(module),
            category=category, configuration_status=config_status, configuration=config,
            children=edges, **slots, **direct))
        # A shallow identity snapshot detects registry replacement/mutation.
        observations.append((module, registries, frozenset(persistence), modules[-1]))
    for module, before, persistent, recorded in observations:
        attributes = _MODULE_DICT.__get__(module)
        for key, old in zip(('_modules', '_parameters', '_buffers'), before):
            current = attributes.get(key)
            if (type(current) not in (dict, OrderedDict) or len(current) != len(old)
                    or any(a != b or x is not y for (a, x), (b, y) in zip(old, current.items()))):
                raise ModelStructureError('registrations_changed_during_capture')
        if type(attributes.get('_non_persistent_buffers_set')) is not set or frozenset(attributes['_non_persistent_buffers_set']) != persistent:
            raise ModelStructureError('registrations_changed_during_capture')
        category, status, config = _configuration(module, attributes, attributes['_parameters'])
        if (category != recorded['category'] or status != recorded['configuration_status']
                or canonical(config) != canonical(recorded['configuration'])
                or _qualified_type(module) != recorded['qualified_type']):
            raise ModelStructureError('configuration_changed_during_capture')
    for tensor, recorded in zip(tensor_objects, tensors):
        if canonical(_tensor(tensor, recorded['tensor_id'], storages)) != canonical(recorded):
            raise ModelStructureError('tensor_metadata_changed_during_capture')
    paths = _paths(modules, limits['max_paths'], limits['max_depth'])
    data = plain(dict(format='hyso.model_structure', schema_version=1,
        producer=dict(method='registered_nn_module_metadata', version=1, classification_version=1, torch_version=str(torch.__version__)),
        source=dict(kind='existing_trusted_torch_nn_module', authority='native', fields=_SOURCES),
        modules=modules, tensors=tensors, paths=paths, summary=_summary(modules, tensors, paths), limitations=_LIMITATIONS))
    data['content_sha256'] = digest(data)
    if len(canonical(data).encode('utf-8')) > limits['max_bytes']:
        raise ModelStructureError('snapshot_byte_limit')
    return ModelStructureSnapshot(data)
