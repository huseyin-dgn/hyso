"""Real D.1 CPU snapshots -> metadata-only D.2 comparison; no workloads."""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter

from hyso import (capture_model_structure, ModelStructureSnapshot, compare_model_structures,
                  ModelStructureComparison, ModelStructureComparisonError)
from hyso.analysis._codec import digest
from hyso.structure import _paths, _summary
from test_model_structure import metadata_only, no_gpu_training_or_benchmark


@contextmanager
def comparison_only(monkeypatch):
    import hyso
    import hyso.structure as structure
    import hyso._structure_native as native
    def forbidden(*args, **kwargs): raise AssertionError('comparison re-introspected or executed a model')
    with metadata_only(monkeypatch), monkeypatch.context() as patch:
        patch.setattr(hyso, 'capture_model_structure', forbidden)
        patch.setattr(structure, 'capture_model_structure', forbidden)
        for name in ('capture', '_tensor', '_configuration'): patch.setattr(native, name, forbidden)
        patch.setattr(torch, 'load', forbidden)
        yield


def run(monkeypatch, baseline, candidate, **kwargs):
    a, b = [s if type(s) is ModelStructureSnapshot else capture_model_structure(s) for s in (baseline, candidate)]
    before = (a.to_json(), b.to_json())
    with comparison_only(monkeypatch):
        result = compare_model_structures(a, b, **kwargs)
    assert before == (a.to_json(), b.to_json())
    return result


def metrics(result): return {v.metric: v for v in result.summary_differences}
def aliases(result): return {v.kind: v for v in result.alias_differences}


def cnn(width=4):
    return nn.ModuleDict(dict(features=nn.ModuleDict(dict(conv1=nn.Conv2d(3, width, 3))), head=nn.Linear(width, 2)))


def parameter_model(size, *, dtype=torch.float32, grad=True):
    m = nn.Module(); m.p = nn.Parameter(torch.empty(size, dtype=dtype), requires_grad=grad)
    return m


def tied_model(tied):
    m = nn.ModuleDict(dict(embed=nn.Embedding(4, 3), head=nn.Linear(3, 4, bias=False)))
    if tied: m['head'].weight = m['embed'].weight
    return m


def field_at(result, path, field):
    return next(f for p in result.changed_paths if p.path == path for f in p.configuration_changes if f.field == field)


def test_parameter_and_percentage_arithmetic_and_source_orientation(monkeypatch):
    a, b = capture_model_structure(parameter_model(72)), capture_model_structure(parameter_model(41))
    result = run(monkeypatch, a, b); q = metrics(result)['parameter_elements']
    assert (q.baseline.value, q.candidate.value, q.difference) == (72, 41, -31)
    assert q.percent_change == pytest.approx(-43.05555555555556)
    assert q.known_subtotal_difference == -31 and q.status == 'computed'
    assert result.baseline.content_sha256 == a.content_sha256 and result.candidate.content_sha256 == b.content_sha256
    assert result.baseline.schema_version == result.candidate.schema_version == 1
    assert result.policy.id == 'literal_registered_path_structure_v1' and result.policy.version == 1
    reverse = metrics(run(monkeypatch, b, a))['parameter_elements']
    assert reverse.difference == 31 and reverse.percent_change == pytest.approx(100*31/41)


def test_zero_baseline_keeps_absolute_difference_and_null_percent(monkeypatch):
    result = run(monkeypatch, nn.Module(), parameter_model(4))
    q = metrics(result)['parameter_elements']
    assert q.difference == 4 and q.percent_change is None and q.percent_reason == 'zero_baseline'
    same = capture_model_structure(nn.Module())
    zeros = run(monkeypatch, same, same)
    assert all(q.difference == 0 and q.percent_change is None for q in zeros.summary_differences if q.baseline.value == 0)
    assert not zeros.has_observed_changes


def test_identical_real_snapshots_have_zero_observed_differences(monkeypatch):
    a, b = capture_model_structure(cnn()), capture_model_structure(cnn())
    assert a.content_sha256 == b.content_sha256  # Different live objects, same recorded structure.
    result = run(monkeypatch, a, b)
    assert not result.has_observed_changes and result.coverage == 'complete_recorded_metadata'
    assert not result.added_paths and not result.removed_paths and not result.changed_paths and not result.tensor_differences
    assert all(q.difference == 0 for q in result.summary_differences)
    assert all(not a.removed_groups and not a.added_groups for a in result.alias_differences)
    assert run(monkeypatch, a, a).to_json() == result.to_json()


def test_conv_and_linear_static_fields_and_counts(monkeypatch):
    result = run(monkeypatch, cnn(4), cnn(8))
    q = metrics(result)['parameter_elements']
    assert (q.baseline.value, q.candidate.value, q.difference) == (122, 242, 120)
    assert q.percent_change == pytest.approx(100*120/122)
    conv = field_at(result, 'features.conv1', 'out_channels')
    linear = field_at(result, 'head', 'in_features')
    assert (conv.baseline.value, conv.candidate.value, conv.status) == (4, 8, 'changed')
    assert (linear.baseline.value, linear.candidate.value) == (4, 8)
    out = run(monkeypatch, nn.Linear(3, 2), nn.Linear(3, 4))
    assert (field_at(out, '', 'out_features').baseline.value, field_at(out, '', 'out_features').candidate.value) == (2, 4)


def test_path_add_remove_without_rename_guessing(monkeypatch):
    a = nn.ModuleDict(dict(features=nn.Sequential(nn.Linear(2, 2))))
    b = nn.ModuleDict(dict(backbone=nn.ModuleDict(dict(conv1=nn.Linear(2, 2)))))
    result = run(monkeypatch, a, b)
    assert result.removed_paths == ('features', 'features.0')
    assert result.added_paths == ('backbone', 'backbone.conv1')
    assert not any(p.path in ('features.0', 'backbone.conv1') for p in result.changed_paths)
    assert 'rename' not in result.policy.matching


def test_module_type_and_container_change_are_explicit(monkeypatch):
    a = nn.ModuleDict(dict(features=nn.ModuleDict(dict(act=nn.ReLU()))))
    b = nn.ModuleDict(dict(features=nn.Sequential(OrderedDict([('act', nn.SiLU())]))))
    result = run(monkeypatch, a, b)
    assert not result.added_paths and not result.removed_paths
    changes = {p.path: p for p in result.changed_paths}
    assert any(f.field == 'qualified_type' for f in changes['features'].metadata_changes)
    assert changes['features.act'].baseline_type.endswith('.ReLU') and changes['features.act'].candidate_type.endswith('.SiLU')
    c = {d.name: d for d in result.category_differences}
    assert c['container'].unique_modules.difference == 0


def test_shape_dtype_requires_grad_and_dtype_distribution(monkeypatch):
    result = run(monkeypatch, parameter_model((2, 3)), parameter_model((3, 2), dtype=torch.float64, grad=False))
    q = metrics(result)
    assert q['parameter_elements'].difference == 0 and q['trainable_elements'].difference == -6 and q['frozen_elements'].difference == 6
    assert q['parameter_logical_bytes'].difference == 24
    fields = {f.field: f for t in result.tensor_differences if t.name == 'p' for f in t.changes}
    assert (fields['shape'].baseline.value, fields['shape'].candidate.value) == ((2, 3), (3, 2))
    assert fields['dtype'].candidate.value == 'torch.float64' and fields['requires_grad'].candidate.value is False
    dtypes = {v.dtype: v for v in result.dtype_differences if v.kind == 'parameter'}
    assert dtypes['torch.float32'].tensor_count.difference == -1
    assert dtypes['torch.float64'].elements.difference == 6
    assert dtypes['torch.float64'].elements.percent_change is None


def test_buffer_persistence_bytes_and_counts(monkeypatch):
    a, b = nn.Module(), nn.Module()
    a.register_buffer('stats', torch.ones(2), persistent=True)
    b.register_buffer('stats', torch.ones(3), persistent=False)
    result = run(monkeypatch, a, b); q = metrics(result)
    assert q['unique_buffer_tensors'].difference == 0 and q['buffer_elements'].difference == 1 and q['buffer_logical_bytes'].difference == 4
    change = next(f for t in result.tensor_differences for f in t.changes if f.field == 'persistent')
    assert (change.baseline.value, change.candidate.value) == (True, False)


def test_tied_embedding_linear_not_double_counted(monkeypatch):
    result = run(monkeypatch, tied_model(False), tied_model(True))
    q = metrics(result)
    assert (q['parameter_elements'].baseline.value, q['parameter_elements'].candidate.value, q['parameter_elements'].difference) == (24, 12, -12)
    assert q['unique_parameter_tensors'].difference == -1 and q['parameter_elements'].percent_change == -50.
    obj = aliases(result)['tensor_object']
    assert len(obj.added_groups) == 1 and not obj.removed_groups
    members = obj.added_groups[0].members
    assert len(members) == 1 and {(e.owner_path, e.name) for e in members[0]} == {('embed', 'weight'), ('head', 'weight')}
    assert not aliases(result)['storage'].added_groups  # One object tied, not two objects sharing storage.
    assert not result.tensor_differences  # No fake tensor-ID differences.


def test_shared_module_vs_independent_equal_shape_modules(monkeypatch):
    shared = nn.Linear(2, 2); a = nn.ModuleDict(dict(left=shared, right=shared))
    b = nn.ModuleDict(dict(left=nn.Linear(2, 2), right=nn.Linear(2, 2)))
    result = run(monkeypatch, a, b); q = metrics(result)
    assert q['unique_modules'].difference == 1 and q['registered_module_paths'].difference == 0
    assert q['parameter_elements'].difference == 6 and q['unique_parameter_tensors'].difference == 2
    assert aliases(result)['module_object'].removed_groups and len(aliases(result)['tensor_object'].removed_groups) == 2
    linear = next(d for d in result.module_type_differences if d.name.endswith('.Linear'))
    assert linear.unique_modules.difference == 1 and linear.registered_paths.difference == 0
    assert not result.tensor_differences


def storage_model(shared):
    m = nn.Module()
    if shared:
        backing = torch.empty(8); m.p = nn.Parameter(backing[:4]); m.q = nn.Parameter(backing[2:6])
    else: m.p = nn.Parameter(torch.empty(4)); m.q = nn.Parameter(torch.empty(4))
    return m


def test_distinct_objects_sharing_storage_are_separate_from_tying(monkeypatch):
    result = run(monkeypatch, storage_model(False), storage_model(True))
    assert metrics(result)['parameter_elements'].difference == 0 and metrics(result)['unique_parameter_tensors'].difference == 0
    assert not aliases(result)['tensor_object'].added_groups
    groups = aliases(result)['storage'].added_groups
    assert len(groups) == 1 and len(groups[0].members) == 2
    assert {e.name for member in groups[0].members for e in member} == {'p', 'q'}


def test_common_alias_projection_does_not_guess_untying_on_removed_paths(monkeypatch):
    a, b, c = nn.Module(), nn.Module(), nn.Module()
    a.p = nn.Parameter(torch.empty(3)); a.q = a.p
    b.p = nn.Parameter(torch.empty(3))
    c.p = nn.Parameter(torch.empty(3)); c.q = c.p; c.r = c.p
    removed = aliases(run(monkeypatch, a, b))['tensor_object']
    assert removed.status == 'partial' and not removed.removed_groups
    assert [e.name for e in removed.unmatched_baseline_endpoints] == ['q']
    extended = aliases(run(monkeypatch, a, c))['tensor_object']
    assert not extended.added_groups and not extended.removed_groups and extended.status == 'partial'
    assert [e.name for e in extended.unmatched_candidate_endpoints] == ['r']
    assert extended.matching_scope == 'common_registered_endpoints'
    assert {e.name for e in extended.baseline_groups[0].members[0]} == {'p', 'q'}
    assert {e.name for e in extended.candidate_groups[0].members[0]} == {'p', 'q', 'r'}


def test_unavailable_storage_does_not_claim_alias_disappearance(monkeypatch):
    a = capture_model_structure(storage_model(True)); b = nn.Module()
    b.p = UninitializedParameter(); b.q = nn.Parameter(torch.empty(4))
    result = run(monkeypatch, a, b); storage = aliases(result)['storage']
    assert storage.status == 'partial' and not storage.removed_groups
    assert storage.baseline_groups and not storage.candidate_groups
    assert [e.name for e in storage.unassessed_baseline_endpoints] == ['p']
    assert result.coverage == 'partial' and result.evidence_gaps


def renumber(snapshot):
    d = snapshot.to_dict()
    mids = [m['module_id'] for m in d['modules']]
    mm = dict(zip(mids, [mids[0]] + list(reversed(mids[1:]))))
    tids = [t['tensor_id'] for t in d['tensors']]; tm = dict(zip(tids, reversed(tids)))
    sids = list(dict.fromkeys(t['storage_id'] for t in d['tensors'] if t['storage_id'] is not None)); sm = dict(zip(sids, reversed(sids)))
    for m in d['modules']:
        m['module_id'] = mm[m['module_id']]
        for child in m['children']:
            if child['module_id'] is not None: child['module_id'] = mm[child['module_id']]
        for slot in m['parameters'] + m['buffers']:
            if slot['tensor_id'] is not None: slot['tensor_id'] = tm[slot['tensor_id']]
    for t in d['tensors']:
        t['tensor_id'] = tm[t['tensor_id']]
        if t['storage_id'] is not None: t['storage_id'] = sm[t['storage_id']]
    d['modules'].sort(key=lambda m: int(m['module_id'][1:])); d['tensors'].sort(key=lambda t: int(t['tensor_id'][1:]))
    d['paths'] = _paths(d['modules']); d['summary'] = _summary(d['modules'], d['tensors'], d['paths'])
    d['content_sha256'] = digest({k: v for k, v in d.items() if k != 'content_sha256'})
    return ModelStructureSnapshot(d)


def test_local_id_permutations_do_not_create_structural_differences(monkeypatch):
    model = tied_model(True); model['tail'] = nn.Linear(2, 2)
    a = capture_model_structure(model); b = renumber(a)
    assert a.content_sha256 != b.content_sha256
    result = run(monkeypatch, a, b)
    assert not result.has_observed_changes and not result.tensor_differences and not result.changed_paths
    assert all(not x.removed_groups and not x.added_groups for x in result.alias_differences)
    assert all(x.difference == 0 for x in result.summary_differences)


def test_actual_producer_new_earlier_module_does_not_change_existing_ids_semantically(monkeypatch):
    a = nn.ModuleDict(OrderedDict([('kept', nn.Linear(2, 2))]))
    b = nn.ModuleDict(OrderedDict([('new', nn.Linear(1, 1)), ('kept', nn.Linear(2, 2))]))
    result = run(monkeypatch, a, b)
    assert result.added_paths == ('new',) and not any(d.owner_path == 'kept' for d in result.tensor_differences)
    assert not any(p.path == 'kept' for p in result.changed_paths)


@pytest.mark.parametrize('kind', ['lazy', 'sparse', 'meta', 'custom'])
def test_unsupported_and_missing_totals_are_not_fabricated(monkeypatch, kind):
    if kind == 'lazy': a = nn.LazyLinear(2); b = nn.Linear(3, 2)
    elif kind == 'sparse':
        a = nn.Module(); a.p = nn.Parameter(torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([1.]), (2, 2)))
        b = parameter_model((2, 2))
    elif kind == 'meta': a, b = nn.Linear(3, 2, device='meta'), nn.Linear(3, 2, device='meta')
    else:
        class Custom(nn.Module):
            def __init__(self): super().__init__(); self.layer = nn.Linear(2, 2)
            def forward(self, *args): raise AssertionError('custom forward')
        a, b = Custom(), Custom()
    result = run(monkeypatch, a, b); q = metrics(result)['parameter_elements']
    assert result.coverage == 'partial' and result.evidence_gaps
    if kind in ('lazy', 'sparse'):
        assert q.status == 'insufficient_evidence' and q.difference is None and q.percent_change is None
        assert q.known_subtotal_difference == q.candidate.known_subtotal - q.baseline.known_subtotal
    else:
        assert q.difference == 0 and not result.has_observed_changes
    assert all(f.status == 'unresolved' for p in result.changed_paths for f in p.configuration_changes) if kind == 'custom' else True


def test_unknown_dtype_cannot_be_assumed_absent_from_known_dtype_bucket(monkeypatch):
    class Special(nn.Parameter):
        @classmethod
        def __torch_function__(cls, *a, **kw): raise AssertionError('subclass dispatch')
    b = nn.Module(); b._parameters['p'] = torch.Tensor._make_subclass(Special, torch.empty(3), True)
    result = run(monkeypatch, parameter_model(3), b)
    dtype = next(v for v in result.dtype_differences if v.kind == 'parameter' and v.dtype == 'torch.float32')
    assert dtype.tensor_count.candidate.status == 'unsupported' and dtype.tensor_count.candidate.value is None
    assert dtype.tensor_count.difference is None and dtype.tensor_count.percent_change is None
    assert dtype.tensor_count.known_subtotal_difference == -1


@pytest.mark.parametrize('family', ['cnn', 'gru', 'rnn', 'lstm', 'transformer', 'llm_components', 'detector_like'])
def test_general_registered_structure_without_model_family_certification(monkeypatch, family):
    if family == 'cnn': a, b = cnn(4), cnn(6)
    elif family in ('gru', 'rnn', 'lstm'):
        cls = {'gru': nn.GRU, 'rnn': nn.RNN, 'lstm': nn.LSTM}[family]
        a, b = cls(3, 4, batch_first=True), cls(3, 6, batch_first=True)
    elif family == 'transformer':
        a = nn.TransformerEncoder(nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, batch_first=True), 1)
        b = nn.TransformerEncoder(nn.TransformerEncoderLayer(8, 2, dim_feedforward=24, batch_first=True), 1)
    elif family == 'detector_like':
        class Detector(nn.Module):
            def __init__(self, width):
                super().__init__(); self.backbone = nn.Sequential(nn.Conv2d(3, width, 3), nn.SiLU()); self.head = nn.Conv2d(width, 7, 1)
            def forward(self, *a): raise AssertionError('detector forward')
        a, b = Detector(4), Detector(6)
    else: a, b = tied_model(False), tied_model(True)
    result = run(monkeypatch, a, b)
    assert result.has_observed_changes and metrics(result)['parameter_elements'].difference != 0
    assert result.baseline.producer_method == 'registered_nn_module_metadata'
    assert any('model-family certification' in v for v in result.limitations)


def test_cyclic_registration_relations_are_bounded_and_path_based(monkeypatch):
    a, b = nn.Module(), nn.Module()
    a.loop = a; b.loop = nn.Module()
    result = run(monkeypatch, a, b)
    assert not result.added_paths and not result.removed_paths
    assert metrics(result)['unique_modules'].difference == 1
    assert aliases(result)['module_object'].removed_groups and result.has_observed_changes


def test_torch_provenance_difference_is_not_a_structure_or_runtime_verdict(monkeypatch):
    a = capture_model_structure(cnn()); d = a.to_dict(); d['producer']['torch_version'] = 'different-recorded-runtime'
    d['content_sha256'] = digest({k: v for k, v in d.items() if k != 'content_sha256'}); b = ModelStructureSnapshot(d)
    result = run(monkeypatch, a, b)
    assert not result.has_observed_changes and result.provenance_differences[0].field == 'producer.torch_version'
    assert result.baseline.torch_version != result.candidate.torch_version


def test_immutable_deterministic_canonical_json_and_source_roundtrip(monkeypatch, tmp_path):
    a, b = capture_model_structure(cnn(4)), capture_model_structure(cnn(8))
    result = run(monkeypatch, a, b)
    assert result.to_json() == run(monkeypatch, a, b).to_json()
    assert result.content_sha256 == digest({k: v for k, v in result.to_dict().items() if k != 'content_sha256'})
    with pytest.raises((FrozenInstanceError, AttributeError)): result._payload = '{}'
    with pytest.raises(FrozenInstanceError): result.policy.id = 'other'
    with pytest.raises(FrozenInstanceError): result.summary_differences[0].difference = 7
    with pytest.raises(FrozenInstanceError): result.changed_paths[0].configuration_changes[0].baseline.value = 7
    detached = result.to_dict(); detached['summary_differences'].clear()
    assert result.summary_differences
    path = tmp_path / 'comparison.json'; result.to_json(path)
    with comparison_only(monkeypatch):
        reopened = ModelStructureComparison.from_json(path)
        assert reopened.baseline_snapshot.to_json() == a.to_json() and reopened.candidate_snapshot.to_json() == b.to_json()
    assert reopened.to_json() == result.to_json()
    with pytest.raises(FileExistsError): result.to_json(path)


@pytest.mark.parametrize('damage', ['difference', 'percent', 'ref', 'order', 'version', 'policy', 'paths', 'tensor', 'alias', 'limits', 'extra', 'source'])
def test_rehashed_comparison_tampering_is_rejected(monkeypatch, damage):
    result = run(monkeypatch, cnn(4), cnn(8)); data = result.to_dict()
    if damage == 'difference': data['summary_differences'][0]['difference'] = 999
    elif damage == 'percent': data['summary_differences'][0]['percent_change'] = 999.
    elif damage == 'ref': data['baseline']['content_sha256'] = 'a'*64
    elif damage == 'order': data['baseline'], data['candidate'] = data['candidate'], data['baseline']
    elif damage == 'version': data['schema_version'] = 2
    elif damage == 'policy': data['policy']['version'] = 2
    elif damage == 'paths': data['added_paths'] = ['invented']
    elif damage == 'tensor': data['tensor_differences'][0]['changes'][0]['candidate']['value'] = 'fake'
    elif damage == 'alias': data['alias_differences'][0]['status'] = 'certified_equivalent'
    elif damage == 'limits': data['limitations'] = []
    elif damage == 'extra': data['VERIFIED_IMPROVEMENT'] = True
    else: data['source_snapshots']['baseline']['summary']['unique_parameter_tensors'] += 1
    data['content_sha256'] = digest({k: v for k, v in data.items() if k != 'content_sha256'})
    with comparison_only(monkeypatch), pytest.raises(ModelStructureComparisonError): ModelStructureComparison(data)


@pytest.mark.parametrize('limit,value', [('max_entries', 1), ('max_bytes', 100), ('max_entries', 0), ('max_bytes', True)])
def test_explicit_limits_no_truncation(monkeypatch, limit, value):
    with pytest.raises(ModelStructureComparisonError, match='limit'):
        run(monkeypatch, cnn(), cnn(), **{limit: value})


def test_non_snapshot_inputs_and_corrupt_snapshot_rejected_without_introspection(monkeypatch):
    a = capture_model_structure(cnn())
    for value in (cnn(), a.to_dict(), 'model_factory:build', None):
        with comparison_only(monkeypatch), pytest.raises(ModelStructureComparisonError, match='snapshots_required'):
            compare_model_structures(a, value)
    object.__setattr__(a, '_payload', '{}')
    b = capture_model_structure(cnn())
    with comparison_only(monkeypatch), pytest.raises(ModelStructureComparisonError, match='invalid_source_snapshot'):
        compare_model_structures(a, b)


def test_structure_comparison_is_not_automatic_p26_or_p11_evidence(monkeypatch, tmp_path):
    from hyso.analysis import EvidenceRecord, EvidenceError
    import hashlib
    result = run(monkeypatch, cnn(), cnn(8)); path = tmp_path / 'comparison.json'; result.to_json(path)
    for feature in ('P26', 'P11'):
        with pytest.raises(EvidenceError, match='unsupported_source_format'):
            EvidenceRecord.from_json(path, feature_id=feature, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
