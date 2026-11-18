"""3B.4D.1 metadata-only capture on trusted synthetic CPU modules; no workloads."""
from contextlib import contextmanager
import copy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter

from hyso import capture_model_structure, ModelStructureSnapshot, ModelStructureError
from hyso.analysis._codec import digest
from hyso.checkpoint import _rng_state
from trainer_replay_support import equal


@pytest.fixture(autouse=True)
def no_gpu_training_or_benchmark(monkeypatch):
    from hyso.benchmark import BenchmarkRunner
    from hyso.training import Trainer
    def forbidden(*args, **kwargs): raise AssertionError('GPU/training/benchmark forbidden')
    for name in ('init', 'is_available', 'device_count', 'current_device', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(BenchmarkRunner, 'run', forbidden)
    monkeypatch.setattr(Trainer, 'fit', forbidden)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


@contextmanager
def metadata_only(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('capture touched execution/state/values/hooks/device')
    with monkeypatch.context() as patch:
        for name in ('_call_impl', 'train', 'eval', 'state_dict', 'load_state_dict', 'named_modules',
                     'named_parameters', 'named_buffers', 'parameters', 'buffers', 'children', 'modules',
                     'to', 'cpu', 'cuda', '_apply', 'register_forward_hook', 'register_forward_pre_hook',
                     'register_backward_hook', 'register_full_backward_hook', 'register_state_dict_pre_hook',
                     'register_state_dict_post_hook', 'register_load_state_dict_pre_hook'):
            patch.setattr(nn.Module, name, forbidden)
        for name in ('to', 'cpu', 'cuda', 'numpy', 'tolist', 'item', 'clone', 'copy_', 'detach', 'backward', 'data_ptr'):
            patch.setattr(torch.Tensor, name, forbidden)
        patch.setattr(torch.UntypedStorage, 'data_ptr', forbidden)
        patch.setattr(copy, 'deepcopy', forbidden)
        yield


def cnn():
    return nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))


def snapshot(model, monkeypatch, **kwargs):
    with metadata_only(monkeypatch):
        return capture_model_structure(model, **kwargs)


def category_counts(s):
    return {c.name: c.unique_modules for c in s.summary.categories}


def test_cnn_exact_counts_config_and_container_semantics(monkeypatch):
    model = cnn(); s = snapshot(model, monkeypatch)
    # Conv: 4*3*3*3+4=112; BN: 4+4=8; Linear: 2*4+2=10.
    assert s.summary.parameter_elements.value == 130
    assert s.summary.trainable_elements.value == 130 and s.summary.frozen_elements.value == 0
    assert s.summary.unique_parameter_tensors == 6
    assert s.summary.parameter_logical_bytes.value == 130*4
    assert s.summary.unique_modules == s.summary.registered_module_paths == 7
    assert category_counts(s) == dict(container=1, conv2d=1, normalization=1, activation=1, pooling=1, reshape=1, linear=1)
    conv = s.modules[1]
    assert conv.primary_path == '0' and conv.direct_parameter_elements.value == 112
    assert conv.direct_parameter_tensor_count == 2 and conv.direct_buffer_tensor_count == 0
    config = {f.name: f.value for f in conv.configuration}
    assert config == dict(in_channels=3, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0),
                          dilation=(1, 1), groups=1, padding_mode='zeros', bias=True)
    assert s.modules[0].direct_parameter_elements.value == 0
    assert s.summary.unique_buffer_tensors == 3 and s.summary.buffer_elements.value == 9
    assert s.summary.buffer_logical_bytes.value == 8*4+8


def test_trainable_frozen_dtype_and_shapes(monkeypatch):
    model = nn.Module()
    model.a = nn.Parameter(torch.ones(2))
    model.b = nn.Parameter(torch.ones(3, dtype=torch.float64), requires_grad=False)
    s = snapshot(model, monkeypatch)
    assert (s.summary.parameter_elements.value, s.summary.trainable_elements.value, s.summary.frozen_elements.value) == (5, 2, 3)
    assert s.summary.parameter_logical_bytes.value == 32
    dtypes = {d.dtype: d for d in s.summary.parameter_dtypes}
    assert dtypes['torch.float32'].elements.value == 2
    assert dtypes['torch.float64'].logical_bytes.value == 24
    assert [t.shape for t in s.tensors] == [(2,), (3,)]
    assert [t.requires_grad for t in s.tensors] == [True, False]


def test_tied_parameters_are_unique_objects_not_repeated_slots(monkeypatch):
    model = nn.Module(); model.a = nn.Linear(3, 2); model.b = nn.Linear(3, 2)
    model.b.weight = model.a.weight; model.b.bias = model.a.bias
    s = snapshot(model, monkeypatch)
    assert s.summary.parameter_elements.value == 8 and s.summary.unique_parameter_tensors == 2
    assert [slot.tensor_id for slot in s.modules[1].parameters] == [slot.tensor_id for slot in s.modules[2].parameters]
    assert s.modules[1].direct_parameter_elements.value == s.modules[2].direct_parameter_elements.value == 8
    local = nn.Module(); local.a = model.a.weight; local.b = model.a.weight
    one = snapshot(local, monkeypatch)
    assert len(one.modules[0].parameters) == 2 and one.modules[0].direct_parameter_tensor_count == 1
    assert one.summary.parameter_elements.value == 6


def test_shared_modules_expand_paths_and_keep_unique_module_counts(monkeypatch):
    model = nn.Module(); shared = nn.Sequential(nn.Linear(2, 2))
    model.left = shared; model.right = shared
    s = snapshot(model, monkeypatch)
    assert s.summary.unique_modules == 3 and s.summary.registered_module_paths == 5
    assert [p.path for p in s.paths] == ['', 'left', 'right', 'left.0', 'right.0']
    assert [p.shared for p in s.paths] == [False, False, True, False, True]
    assert s.summary.parameter_elements.value == 6
    linear = next(t for t in s.summary.module_types if t.name.endswith('.Linear'))
    assert linear.unique_modules == 1 and linear.registered_paths == 2


def test_distinct_parameter_views_share_storage_without_object_dedup(monkeypatch):
    backing = torch.arange(8.)
    model = nn.Module(); model.a = nn.Parameter(backing[:4]); model.b = nn.Parameter(backing[2:6])
    s = snapshot(model, monkeypatch)
    assert s.summary.unique_parameter_tensors == 2 and s.summary.parameter_elements.value == 8
    assert s.summary.parameter_logical_bytes.value == 32
    assert len(s.summary.storage_groups) == 1 and s.summary.storage_groups[0].distinct_objects_share_storage
    assert [t.storage_offset for t in s.tensors] == [0, 2]
    assert [t.storage_id for t in s.tensors] == ['s0', 's0']
    # Empty independent storages have the same data_ptr=0; handles still differ.
    empty = nn.Module(); empty.a = nn.Parameter(torch.empty(0)); empty.b = nn.Parameter(torch.empty(0))
    e = snapshot(empty, monkeypatch)
    assert e.summary.parameter_elements.value == e.summary.parameter_logical_bytes.value == 0
    assert len(e.summary.storage_groups) == 2


def test_buffer_alias_persistence_and_cross_category_union(monkeypatch):
    model = nn.Module(); model.child = nn.Module()
    value = torch.arange(3, dtype=torch.int64)
    model.register_buffer('persistent', value)
    model.child.register_buffer('temporary', value, persistent=False)
    s = snapshot(model, monkeypatch)
    assert s.summary.unique_buffer_tensors == 1 and s.summary.buffer_logical_bytes.value == 24
    assert s.modules[0].buffers[0].persistent is True and s.modules[1].buffers[0].persistent is False
    both = nn.Module(); both.p = nn.Parameter(torch.ones(4)); both.register_buffer('alias', both.p)
    u = snapshot(both, monkeypatch)
    assert u.summary.unique_parameter_tensors == u.summary.unique_buffer_tensors == u.summary.cross_category_tensors == 1
    assert u.summary.parameter_logical_bytes.value == u.summary.buffer_logical_bytes.value == u.summary.tensor_union_logical_bytes.value == 16


def test_none_registrations_and_cycles_are_explicit(monkeypatch):
    model = nn.Module(); model.loop = model; model.add_module('absent', None)
    model.register_parameter('unset', None); model.register_buffer('unset_buffer', None, persistent=False)
    s = snapshot(model, monkeypatch)
    assert s.summary.unique_modules == 1 and s.summary.registered_module_paths == 2
    assert s.paths[1].cycle and s.paths[1].shared
    assert s.modules[0].children[1].module_id is None
    assert s.modules[0].parameters[0].tensor_id is None and s.modules[0].buffers[0].persistent is False
    assert s.summary.unique_parameter_tensors == s.summary.unique_buffer_tensors == 0


def test_deterministic_canonical_snapshot_ignores_values_and_addresses(monkeypatch):
    a, b = cnn(), cnn()
    with torch.no_grad():
        b[0].weight.fill_(123)
    sa, sb = snapshot(a, monkeypatch), snapshot(b, monkeypatch)
    assert sa.to_json() == sb.to_json() and sa.content_sha256 == sb.content_sha256
    b[0].weight.requires_grad_(False)
    assert snapshot(b, monkeypatch).content_sha256 != sa.content_sha256
    changed = nn.Sequential(nn.Conv2d(3, 5, 3))
    assert snapshot(changed, monkeypatch).content_sha256 != sa.content_sha256
    assert 'torch_version' in sa.to_dict()['producer']


def test_model_values_versions_modes_gradients_rng_and_hooks_unchanged(monkeypatch):
    model = cnn(); model.train(); model[1].eval()
    for p in model.parameters(): p.grad = torch.ones_like(p)
    hooks_called = []
    handles = [model.register_forward_hook(lambda *a: hooks_called.append('forward')),
               model.register_state_dict_pre_hook(lambda *a: hooks_called.append('state'))]
    def state():
        return ([p.detach().clone() for p in [*model.parameters(), *model.buffers()]],
                [p._version for p in model.parameters()], [p.grad.clone() for p in model.parameters()],
                [m.training for m in model.modules()], _rng_state(),
                [(id(m), tuple(m._forward_hooks), tuple(m._state_dict_pre_hooks)) for m in model.modules()])
    before = state(); snapshot(model, monkeypatch)
    equal(before, state()); assert hooks_called == []
    for handle in handles: handle.remove()


def test_custom_module_no_repr_properties_state_hooks_or_semantics_guessed(monkeypatch):
    class Custom(nn.Module):
        def __init__(self):
            super().__init__(); self.child = nn.Linear(2, 2); self.parameter_count = 999999
        def forward(self, *a): raise AssertionError('forward')
        def state_dict(self, *a, **kw): raise AssertionError('state_dict')
        def __repr__(self): raise AssertionError('repr')
        @property
        def configuration(self): raise AssertionError('property')
    model = Custom(); s = snapshot(model, monkeypatch)
    assert s.modules[0].category == 'custom' and s.modules[0].configuration_status == 'unsupported'
    assert s.modules[0].configuration == () and s.summary.parameter_elements.value == 6
    assert s.modules[1].category == 'linear' and s.modules[0].qualified_type.endswith('.Custom')
    class SubConv(nn.Conv2d): pass
    assert snapshot(SubConv(1, 1, 1), monkeypatch).modules[0].category == 'custom'


def test_custom_dict_descriptor_and_attribute_overrides_are_bypassed(monkeypatch):
    class Static(nn.Module):
        @property
        def __dict__(self): raise AssertionError('custom __dict__ property')
        def __getattribute__(self, name): raise AssertionError('custom attribute lookup')
    # Populate the standard registration dictionaries before enabling access guards.
    model = Static.__new__(Static)
    raw = nn.Module.__dict__['__dict__'].__get__(model)
    raw.update(_modules={}, _parameters={}, _buffers={}, _non_persistent_buffers_set=set(), training=True)
    s = snapshot(model, monkeypatch)
    assert s.summary.unique_modules == 1 and s.summary.parameter_elements.value == 0


def test_non_plain_standard_config_not_coerced_or_evaluated(monkeypatch):
    class Never:
        def __repr__(self): raise AssertionError('repr')
        def __int__(self): raise AssertionError('int')
    model = nn.Linear(2, 2); model.in_features = Never()
    s = snapshot(model, monkeypatch)
    field = next(f for f in s.modules[0].configuration if f.name == 'in_features')
    assert field.status == 'unsupported' and field.value is None
    assert s.summary.parameter_elements.value == 6


@pytest.mark.parametrize('kind', ['rnn', 'gru', 'lstm', 'attention', 'transformer', 'yolo_like'])
def test_recurrent_transformer_and_custom_network_fixtures(monkeypatch, kind):
    if kind in ('rnn', 'gru', 'lstm'):
        model = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[kind](3, 4, num_layers=2, batch_first=True)
        expected = sum(p.numel() for p in model.parameters())
        s = snapshot(model, monkeypatch)
        assert category_counts(s)['recurrent'] == 1 and s.summary.parameter_elements.value == expected
        assert {f.name: f.value for f in s.modules[0].configuration}['hidden_size'] == 4
    elif kind == 'attention':
        model = nn.MultiheadAttention(8, 2, batch_first=True)
        s = snapshot(model, monkeypatch)
        assert category_counts(s)['attention'] == 1 and category_counts(s)['linear'] == 1
        assert s.summary.parameter_elements.value == 288
    elif kind == 'transformer':
        model = nn.TransformerEncoder(nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, batch_first=True), num_layers=1)
        expected = sum(p.numel() for p in model.parameters())
        s = snapshot(model, monkeypatch)
        assert category_counts(s)['transformer'] == 2 and category_counts(s)['attention'] == 1
        assert s.summary.parameter_elements.value == expected
    else:
        class Detector(nn.Module):
            def __init__(self):
                super().__init__(); self.backbone = nn.Sequential(nn.Conv2d(3, 4, 3), nn.SiLU()); self.head = nn.Conv2d(4, 7, 1)
            def forward(self, *args): raise AssertionError('detector forward')
        s = snapshot(Detector(), monkeypatch)
        assert category_counts(s)['conv2d'] == 2 and s.modules[0].category == 'custom'


@pytest.mark.parametrize('kind', ['lazy', 'meta', 'sparse', 'quantized', 'subclass'])
def test_special_tensor_support_is_explicit_without_materialization(monkeypatch, kind):
    if kind == 'lazy': model = nn.LazyLinear(4)
    elif kind == 'meta': model = nn.Linear(3, 2, device='meta')
    elif kind == 'sparse':
        model = nn.Module(); model.p = nn.Parameter(torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([1.]), (2, 2)))
    elif kind == 'quantized':
        model = nn.Module(); model.register_buffer('q', torch.quantize_per_tensor(torch.ones(4), .1, 0, torch.quint8))
    else:
        class Special(nn.Parameter):
            @classmethod
            def __torch_function__(cls, *args, **kwargs): raise AssertionError('tensor subclass dispatch')
        p = torch.Tensor._make_subclass(Special, torch.ones(3), True)
        model = nn.Module(); model._parameters['special'] = p
    s = snapshot(model, monkeypatch)
    if kind == 'meta':
        assert s.summary.parameter_elements.value == 8 and s.summary.parameter_logical_bytes.value == 32
        assert all(t.device == 'meta' and t.storage_status == 'missing' for t in s.tensors)
    else:
        qty = s.summary.buffer_elements if kind == 'quantized' else s.summary.parameter_elements
        assert qty.status == 'unsupported' and qty.value is None and qty.known_subtotal == 0
        assert all(t.status == 'unsupported' for t in s.tensors)
    if kind == 'lazy': assert type(model.weight) is UninitializedParameter


def test_partial_counts_expose_known_subtotal_without_claiming_total(monkeypatch):
    model = nn.Module(); model.good = nn.Parameter(torch.ones(3)); model.lazy = UninitializedParameter()
    s = snapshot(model, monkeypatch)
    q = s.summary.parameter_elements
    assert q.value is None and q.known_subtotal == 3 and q.unsupported_tensor_ids == ('t1',)
    assert s.summary.unique_parameter_tensors == 2


def test_convolution_dimensions_embeddings_and_supported_configuration(monkeypatch):
    model = nn.ModuleDict(dict(c1=nn.Conv1d(1, 2, 3), c2=nn.Conv2d(1, 2, 3), c3=nn.Conv3d(1, 2, 3),
        embedding=nn.Embedding(5, 3), bag=nn.EmbeddingBag(4, 2, mode='sum'),
        norm=nn.GroupNorm(2, 4), maxpool=nn.AdaptiveMaxPool2d((1, 2), return_indices=True)))
    s = snapshot(model, monkeypatch); counts = category_counts(s)
    assert counts['conv1d'] == counts['conv2d'] == counts['conv3d'] == 1
    assert counts['embedding'] == 2 and counts['normalization'] == 1 and counts['pooling'] == 1
    bag = next(m for m in s.modules if m.primary_path == 'bag')
    assert {f.name: f.value for f in bag.configuration}['mode'] == 'sum'
    pool = next(m for m in s.modules if m.primary_path == 'maxpool')
    assert {f.name: f.value for f in pool.configuration}['return_indices'] is True


def test_registered_order_and_modes_have_explicit_hash_semantics(monkeypatch):
    from collections import OrderedDict
    a = nn.Sequential(OrderedDict([('first', nn.ReLU()), ('second', nn.Sigmoid())]))
    b = nn.Sequential(OrderedDict([('second', nn.Sigmoid()), ('first', nn.ReLU())]))
    before = snapshot(a, monkeypatch)
    assert before.content_sha256 != snapshot(b, monkeypatch).content_sha256
    a.eval()  # Caller action, outside capture. Modes are not part of static structure.
    assert before.content_sha256 == snapshot(a, monkeypatch).content_sha256


def test_metadata_changes_during_capture_fail_closed(monkeypatch):
    from hyso import _structure_native as producer
    original = producer._configuration; model = nn.Linear(2, 2); calls = []
    def changed(*args):
        result = original(*args)
        if not calls:
            calls.append(True); model.in_features = 7  # Simulated concurrent caller mutation.
        return result
    monkeypatch.setattr(producer, '_configuration', changed)
    with pytest.raises(ModelStructureError, match='configuration_changed'):
        snapshot(model, monkeypatch)


def test_storage_identity_unavailable_does_not_erase_logical_metadata(monkeypatch):
    model = nn.Linear(2, 2)
    def unavailable(*args): raise RuntimeError('storage metadata unavailable')
    monkeypatch.setattr(torch.Tensor, 'untyped_storage', unavailable)
    s = snapshot(model, monkeypatch)
    assert s.summary.parameter_elements.value == 6 and s.summary.parameter_logical_bytes.value == 24
    assert s.summary.storage_alias_coverage == 'missing'
    assert all(t.storage_status == 'missing' and t.storage_id is None for t in s.tensors)


@pytest.mark.parametrize('limit,value', [('max_modules', 1), ('max_paths', 2), ('max_tensors', 1),
    ('max_depth', 1), ('max_registrations', 1), ('max_bytes', 100)])
def test_limits_fail_explicitly_without_truncated_result(monkeypatch, limit, value):
    model = nn.Sequential(nn.Sequential(nn.Linear(2, 2)))
    with pytest.raises(ModelStructureError, match='limit'):
        snapshot(model, monkeypatch, **{limit: value})


@pytest.mark.parametrize('value', [0, -1, True, 1.5, 10001])
def test_invalid_limits_and_nonmodule_rejected(monkeypatch, value):
    with pytest.raises(ModelStructureError): snapshot(nn.Module(), monkeypatch, max_modules=value)
    with pytest.raises(ModelStructureError): snapshot('user_factory:build', monkeypatch)


def test_immutable_typed_access_and_json_roundtrip(monkeypatch, tmp_path):
    s = snapshot(cnn(), monkeypatch)
    with pytest.raises((FrozenInstanceError, AttributeError)): s._payload = '{}'
    with pytest.raises(FrozenInstanceError): s.modules[1].primary_path = 'changed'
    with pytest.raises(FrozenInstanceError): s.summary.parameter_elements.value = 5
    assert type(s.modules[1].configuration[2].value) is tuple
    detached = s.to_dict(); detached['tensors'].clear()
    assert s.tensors
    path = tmp_path / 'structure.json'; s.to_json(path)
    with metadata_only(monkeypatch): reopened = ModelStructureSnapshot.from_json(path)
    assert reopened.to_json() == s.to_json() and reopened.content_sha256 == s.content_sha256
    with pytest.raises(FileExistsError): s.to_json(path)


@pytest.mark.parametrize('damage', ['count', 'dtype_bytes', 'dtype', 'shape', 'alias', 'path', 'quantity_bool', 'schema', 'producer', 'native_claim', 'config_object', 'category', 'extra'])
def test_rehashed_inconsistent_snapshots_rejected(monkeypatch, damage):
    s = snapshot(cnn(), monkeypatch); d = s.to_dict()
    if damage == 'count': d['summary']['unique_parameter_tensors'] += 1
    elif damage == 'dtype_bytes': d['tensors'][0]['logical_bytes'] += 1
    elif damage == 'dtype': d['tensors'][0]['dtype'] = 'torch.float64'
    elif damage == 'shape': d['tensors'][0]['shape'][0] += 1
    elif damage == 'alias': d['modules'][1]['parameters'][0]['tensor_id'] = 'unknown'
    elif damage == 'path': d['paths'][1]['module_id'] = 'm0'
    elif damage == 'quantity_bool': d['modules'][0]['direct_parameter_elements']['value'] = False
    elif damage == 'schema': d['schema_version'] = True
    elif damage == 'producer': d['producer']['version'] = 2
    elif damage == 'native_claim': d['source']['authority'] = 'declared'
    elif damage == 'config_object': d['modules'][1]['configuration'][0]['value'] = {'claimed': 'native'}
    elif damage == 'category': d['modules'][1]['category'] = 'recurrent'
    else: d['new_field'] = 1
    d['content_sha256'] = digest({k: v for k, v in d.items() if k != 'content_sha256'})
    with pytest.raises(ModelStructureError): ModelStructureSnapshot(d)


def test_no_automatic_p26_p11_schema_or_evidence_adapter_upgrade(monkeypatch, tmp_path):
    from hyso import evaluate_predictions
    from hyso.analysis import EvidenceRecord, EvidenceError
    from hyso.analysis._codec import canonical
    native = evaluate_predictions([0, 1], [0, 1], class_names=['a', 'b'], split='test')
    before = native.to_dict(); snapshot(cnn(), monkeypatch)
    assert native.to_dict() == before and before['schema_version'] == 1
    assert 'structure' not in canonical(before)
    from hyso.performance import FootprintResult
    payload = json.loads((Path(__file__).parent / 'fixtures/advisor-native-v1.json').read_text(encoding='utf-8'))['payloads']['P11']
    canonical_before = canonical(payload)
    assert canonical(FootprintResult(payload).to_dict()) == canonical_before
    s = snapshot(cnn(), monkeypatch); path = tmp_path / 'structure.json'; s.to_json(path)
    import hashlib
    with pytest.raises(EvidenceError, match='unsupported_source_format'):
        EvidenceRecord.from_json(path, feature_id='P11', expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
