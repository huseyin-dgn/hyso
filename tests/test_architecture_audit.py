"""3B.5A: actual static CPU capture -> versioned audit -> validated reader."""
from collections import OrderedDict
from contextlib import contextmanager
import copy
from dataclasses import FrozenInstanceError
import json
import subprocess

import pytest
import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter

from hyso import (capture_model_structure, ModelStructureSnapshot, audit_model_architecture,
                  ModelArchitectureAudit, ModelArchitectureAuditError)
from hyso.architecture_audit import ArchitectureFindingClass
from hyso.analysis._codec import canonical, digest
from hyso.checkpoint import _rng_state
from test_model_structure import metadata_only, no_gpu_training_or_benchmark
from trainer_replay_support import equal

pytestmark = pytest.mark.usefixtures('no_gpu_training_or_benchmark')


@contextmanager
def audit_only(monkeypatch):
    import hyso
    import hyso.structure as structure
    import hyso._structure_native as native
    def forbidden(*a, **kw): raise AssertionError('audit accessed live model/capture/checkpoint')
    with metadata_only(monkeypatch), monkeypatch.context() as patch:
        for obj, name in [(hyso, 'capture_model_structure'), (structure, 'capture_model_structure'),
                          (torch, 'load'), (native, 'capture'), (native, '_tensor'), (native, '_configuration')]:
            patch.setattr(obj, name, forbidden)
        yield


def run(monkeypatch, model):
    with metadata_only(monkeypatch): snapshot = capture_model_structure(model)
    before = snapshot.to_json()
    with audit_only(monkeypatch): result = audit_model_architecture(snapshot)
    assert snapshot.to_json() == before
    return result


def issues(result): return [f for f in result.findings if f.classification == 'STATIC_INCONSISTENCY']
def codes(result): return [f.code for f in result.findings]
def observed(result, code): return next(f for f in result.findings if f.code == code)


def cnn():
    model = nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(3, 4, 3)), ('activation', nn.ReLU()),
        ('pool', nn.AdaptiveAvgPool2d(1)), ('flatten', nn.Flatten()), ('head', nn.Linear(4, 2))]))
    model.head.bias.requires_grad_(False)
    return model


def at(data, field):
    for part in field.split('.'): data = data[int(part)] if type(data) is list else data[part]
    return data


def test_real_cnn_capture_audit_json_reader_end_to_end(monkeypatch, tmp_path):
    result = run(monkeypatch, cnn())
    source = result.source_snapshot
    path = tmp_path / 'audit.json'; snapshot_path = tmp_path / 'snapshot.json'
    source.to_json(snapshot_path)
    with audit_only(monkeypatch):
        result.to_json(path)
        reopened = ModelArchitectureAudit.from_json(path)
        assert reopened.to_json() == result.to_json()
        assert audit_model_architecture(ModelStructureSnapshot.from_json(snapshot_path)).to_json() == result.to_json()
    assert not issues(result)
    assert result.audit_policy.id == 'registered_static_architecture_v1' and result.audit_policy.version == 1
    assert [r.id for r in result.audit_policy.rules] == ['linear_shapes', 'convolution_shapes',
        'parameter_inventory', 'registration_relationships', 'coverage_gaps']
    assert all(r.version == 1 for r in result.audit_policy.rules)
    coverage = result.coverage
    assert (coverage['total_unique_modules'], coverage['examined_supported_modules'],
            coverage['consistent_modules'], coverage['not_applicable_modules']) == (6, 2, 2, 4)
    assert coverage['status'] == 'partial'
    assert len(coverage['unassessed_module_ids']) == 4
    inventory = observed(result, 'PARAMETER_INVENTORY')
    assert inventory.classification == ArchitectureFindingClass.OBSERVATION
    assert inventory.derived.value['trainable_fraction'] == pytest.approx(120 / 122)
    assert inventory.derived.value['frozen_fraction'] == pytest.approx(2 / 122)
    assert source.summary.parameter_elements.value == 122
    assert inventory.evidence_status == 'native_metadata_observation'
    assert observed(result, 'DIRECT_PARAMETER_CONCENTRATION').derived.value['fraction'] == pytest.approx(112 / 122)
    assert any(f.value == source.to_dict()['summary']['categories'] for f in inventory.observations)
    assert len({f.finding_id for f in result.findings}) == len(result.findings)
    assert result.summary['total_findings'] == len(result.findings)
    assert 'status' not in result.summary and 'PASS' not in result.to_dict()
    for finding in result.findings:
        assert finding.source_snapshot_sha256 == source.content_sha256
        assert finding.limitation and finding.next_review
        assert finding.observations
        for fact in finding.observations:
            assert fact.value == at(source.to_dict(), fact.field)
        if finding.expected:
            assert set(finding.expected.source_fields) <= {f.field for f in finding.observations}
    data = result.to_dict()
    assert result.content_sha256 == digest({k: v for k, v in data.items() if k != 'content_sha256'})
    (tmp_path / 'example-summary.json').write_text(canonical(dict(
        source_snapshot_sha256=source.content_sha256, content_sha256=result.content_sha256,
        parameter_elements=source.summary.parameter_elements.value, coverage=result.coverage,
        summary=result.summary, findings=[dict(code=f.code, classification=f.classification.value,
            path=f.path, evidence_status=f.evidence_status) for f in result.findings])), encoding='utf-8')


@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('shape', [(3, 2), (0, 2), (2, 0)])
def test_exact_standard_linear_consistent_including_zero_dimensions(monkeypatch, bias, shape):
    out, inp = shape
    model = nn.Linear(2, 3, bias=bias)
    model.in_features, model.out_features = inp, out
    model.weight = nn.Parameter(torch.empty(out, inp))
    if bias: model.bias = nn.Parameter(torch.empty(out))
    result = run(monkeypatch, model)
    assert not issues(result)
    assert result.module_assessments[0].status == 'consistent'
    assert result.coverage['examined_supported_modules'] == 1


@pytest.mark.parametrize('damage,code', [('weight', 'LINEAR_WEIGHT_SHAPE_MISMATCH'),
    ('bias', 'LINEAR_BIAS_SHAPE_MISMATCH'), ('in_features', 'LINEAR_WEIGHT_SHAPE_MISMATCH'),
    ('out_features', 'LINEAR_WEIGHT_SHAPE_MISMATCH')])
def test_actual_linear_static_shape_contradictions(monkeypatch, damage, code):
    model = nn.Linear(3, 2)
    if damage == 'weight': model.weight = nn.Parameter(torch.empty(3, 2))
    elif damage == 'bias': model.bias = nn.Parameter(torch.empty(3))
    else: setattr(model, damage, 4)
    result = run(monkeypatch, nn.Sequential(model))
    finding = observed(result, code)
    assert finding.classification == 'STATIC_INCONSISTENCY' and finding.path == '0'
    assert finding.evidence_status == 'observed_static_contradiction'
    assert finding.tensor_id and finding.module_id and finding.expected.method
    assert any(f.field.startswith('tensors.') and f.field.endswith('.shape') for f in finding.observations)
    assert any(f.field.startswith('modules.1.configuration.') for f in finding.observations)
    assert result.module_assessments[1].status == 'inconsistent'
    assert finding.related_paths[0].startswith('0.')


@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('groups,bias', [(1, True), (2, False), (4, True)])
def test_standard_grouped_convolutions_consistent(monkeypatch, dim, groups, bias):
    result = run(monkeypatch, getattr(nn, 'Conv%dd' % dim)(4, 8, (2,) * dim, groups=groups, bias=bias))
    assert not issues(result)
    assert result.module_assessments[0].status == 'consistent'
    weight = next(c for c in result.module_assessments[0].checks if c.code == 'weight_shape')
    assert weight.expected.value == [8, 4 // groups] + [2] * dim


@pytest.mark.parametrize('damage,code', [('weight', 'CONV_WEIGHT_SHAPE_MISMATCH'),
    ('bias', 'CONV_BIAS_SHAPE_MISMATCH'), ('in_channels', 'CONV_WEIGHT_SHAPE_MISMATCH'),
    ('groups_zero', 'CONV_GROUPS_INVALID'), ('groups_negative', 'CONV_GROUPS_INVALID'),
    ('groups_divisibility', 'CONV_IN_CHANNELS_NOT_DIVISIBLE'),
    ('out_divisibility', 'CONV_OUT_CHANNELS_NOT_DIVISIBLE'), ('kernel', 'CONV_WEIGHT_SHAPE_MISMATCH')])
def test_conv2d_known_contradictions_without_guessed_shapes(monkeypatch, damage, code):
    model = nn.Conv2d(4, 8, 3, groups=2)
    if damage == 'weight': model.weight = nn.Parameter(torch.empty(8, 4, 3, 3))
    elif damage == 'bias': model.bias = nn.Parameter(torch.empty(7))
    elif damage == 'in_channels': model.in_channels = 6
    elif damage == 'groups_zero': model.groups = 0
    elif damage == 'groups_negative': model.groups = -1
    elif damage == 'groups_divisibility': model.groups = 3
    elif damage == 'out_divisibility': model.out_channels = 7
    elif damage == 'kernel': model.kernel_size = (1, 1)
    result = run(monkeypatch, model)
    assert code in [f.code for f in issues(result)]
    if 'groups' in damage or damage == 'out_divisibility':
        assert 'CONV_WEIGHT_SHAPE_MISMATCH' not in codes(result)
        assert next(c for c in result.module_assessments[0].checks if c.code == 'weight_shape').expected is None


@pytest.mark.parametrize('kind', ['missing_weight', 'none_weight', 'lazy_weight', 'sparse_weight',
    'custom_tensor', 'missing_config', 'unsupported_config', 'untyped_config', 'bad_kernel_type'])
def test_insufficient_metadata_never_becomes_shape_inconsistency(monkeypatch, kind):
    model = nn.Conv2d(2, 4, 1) if kind == 'bad_kernel_type' else nn.Linear(3, 2)
    if kind == 'missing_weight': del model.weight
    elif kind == 'none_weight': model.weight = None
    elif kind == 'lazy_weight': model.weight = UninitializedParameter()
    elif kind == 'sparse_weight':
        model.weight = nn.Parameter(torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.]), (99, 99)))
    elif kind == 'custom_tensor':
        class Special(nn.Parameter):
            @classmethod
            def __torch_function__(cls, *a, **kw): raise AssertionError('custom tensor dispatch')
        model._parameters['weight'] = torch.Tensor._make_subclass(Special, torch.empty(99, 99), True)
    elif kind == 'missing_config': del model.in_features
    elif kind == 'unsupported_config': model.in_features = object()
    elif kind == 'untyped_config': model.in_features = True
    elif kind == 'bad_kernel_type': model.kernel_size = 1
    result = run(monkeypatch, model)
    assert not issues(result)
    assert result.module_assessments[0].status == 'insufficient_evidence'
    assert any(f.classification == 'COVERAGE_GAP' for f in result.findings)
    if kind in ('lazy_weight', 'sparse_weight', 'custom_tensor'):
        assert 'TENSOR_METADATA_UNSUPPORTED' in codes(result)
        assert result.source_snapshot.summary.parameter_elements.value is None
        assert observed(result, 'PARAMETER_INVENTORY').derived.value['trainable_fraction'] is None


@pytest.mark.parametrize('kind', ['linear_subclass', 'conv_subclass', 'spoofed_name', 'transposed', 'lazy'])
def test_custom_subclass_and_transposed_do_not_receive_wrong_standard_rules(monkeypatch, kind):
    if kind in ('linear_subclass', 'spoofed_name'):
        class Custom(nn.Linear): pass
        if kind == 'spoofed_name': Custom.__module__, Custom.__qualname__ = 'torch.nn.modules.linear', 'Linear'
        model = Custom(3, 2); model.in_features = 100
    elif kind == 'conv_subclass':
        class Custom(nn.Conv2d): pass
        model = Custom(2, 4, 1); model.in_channels = 100
    elif kind == 'transposed': model = nn.ConvTranspose2d(2, 4, 3)
    else: model = nn.LazyLinear(2)
    result = run(monkeypatch, model)
    assert not issues(result)
    assert result.module_assessments[0].status == 'not_applicable'
    assert result.coverage['examined_supported_modules'] == 0
    assert any(f.classification == 'COVERAGE_GAP' for f in result.findings)
    assert ('TRANSPOSED_CONV_RULE_NOT_IMPLEMENTED' if kind == 'transposed' else
            'CUSTOM_MODULE_SEMANTICS_UNSUPPORTED') in codes(result)


def test_frozen_and_partial_parameter_inventory_is_observation(monkeypatch):
    model = nn.Linear(3, 2)
    for p in model.parameters(): p.requires_grad_(False)
    result = run(monkeypatch, model)
    assert not issues(result)
    assert observed(result, 'PARAMETER_INVENTORY').derived.value == dict(
        trainable_fraction=0., frozen_fraction=1., unavailable_reason=None)
    mixed = nn.Module(); mixed.good = nn.Parameter(torch.ones(3)); mixed.lazy = UninitializedParameter()
    partial = run(monkeypatch, mixed)
    inventory = observed(partial, 'PARAMETER_INVENTORY')
    q = next(f.value for f in inventory.observations if f.field == 'summary.parameter_elements')
    assert q['value'] is None and q['known_subtotal'] == 3
    assert inventory.evidence_status == 'partial_metadata_observation'
    assert inventory.derived.value['trainable_fraction'] is None
    empty = run(monkeypatch, nn.Module())
    assert observed(empty, 'PARAMETER_INVENTORY').derived.value['unavailable_reason'] == 'no_parameter_elements'


def test_shared_modules_tied_parameters_storage_and_cycles_are_not_defects(monkeypatch):
    model = nn.Module(); model.a = nn.Linear(3, 2); model.b = nn.Linear(3, 2)
    model.b.weight = model.a.weight; model.b.bias = model.a.bias
    model.reused = model.a; model.loop = model
    backing = torch.arange(8.)
    model.c = nn.Parameter(backing[:4]); model.d = nn.Parameter(backing[2:6])
    result = run(monkeypatch, model)
    assert not issues(result)
    assert result.source_snapshot.summary.parameter_elements.value == 16
    assert codes(result).count('TIED_PARAMETER_REGISTRATIONS') == 2
    assert codes(result).count('SHARED_MODULE_REGISTRATIONS') == 1
    assert codes(result).count('REGISTERED_MODULE_CYCLE') == 1
    assert codes(result).count('SHARED_STORAGE_OBSERVED') == 1
    for f in result.findings:
        if f.rule_id == 'registration_relationships': assert f.classification == 'OBSERVATION'
    tied = observed(result, 'TIED_PARAMETER_REGISTRATIONS')
    assert tied.related_paths == ('a', 'b')  # shared module expansion is not an extra tensor registration
    assert len(tied.observations) == 2
    assert result.coverage['total_unique_modules'] == 3
    assert result.coverage['registered_module_paths'] == 5


def test_parameter_buffer_identity_and_meta_storage_gaps_are_explicit(monkeypatch):
    model = nn.Module(); model.p = nn.Parameter(torch.ones(3)); model.register_buffer('alias', model.p)
    result = run(monkeypatch, model)
    assert codes(result).count('SHARED_TENSOR_REGISTRATIONS') == 1
    assert result.source_snapshot.summary.parameter_elements.value == 3
    meta = run(monkeypatch, nn.Linear(3, 2, device='meta'))
    assert not issues(meta) and meta.module_assessments[0].status == 'consistent'
    assert codes(meta).count('STORAGE_ALIAS_UNAVAILABLE') == 2
    assert meta.coverage['status'] == 'partial'


def test_audit_preserves_model_values_buffers_gradients_hooks_code_modes_and_rng(monkeypatch):
    model = cnn(); model.train(); model.head.eval()
    model.register_buffer('owned_buffer', torch.ones(3))
    for p in model.parameters(): p.grad = torch.ones_like(p)
    invoked = []
    handles = [model.register_forward_hook(lambda *a: invoked.append('forward')),
               model.register_state_dict_pre_hook(lambda *a: invoked.append('state'))]
    def state():
        return ([t.detach().clone() for t in [*model.parameters(), *model.buffers()]],
            [(p._version, p.requires_grad, p.grad.clone()) for p in model.parameters()],
            [(m.training, tuple(m._forward_hooks), tuple(m._state_dict_pre_hooks)) for m in model.modules()],
            _rng_state(), nn.Sequential.forward.__code__.co_code)
    before = state(); result = run(monkeypatch, model)
    with audit_only(monkeypatch): ModelArchitectureAudit(result.to_dict())
    equal(before, state()); assert invoked == []
    for handle in handles: handle.remove()


@pytest.mark.parametrize('damage', ['invented_finding', 'observed_value', 'expected_shape', 'classification',
    'rule_version', 'source_field', 'source_hash', 'coverage', 'summary', 'limitations', 'finding_order',
    'duplicate_finding', 'extra', 'version', 'snapshot'])
def test_reader_rederives_all_findings_even_with_rehashed_envelope(monkeypatch, damage):
    model = nn.Linear(3, 2); model.in_features = 9
    d = run(monkeypatch, model).to_dict()
    if damage == 'invented_finding': d['findings'][0]['code'] = 'MODEL_IS_SLOW'
    elif damage == 'observed_value': d['findings'][0]['observations'][0]['value'] = 1000
    elif damage == 'expected_shape': d['findings'][0]['expected']['value'] = [3, 2]
    elif damage == 'classification': d['findings'][0]['classification'] = 'OBSERVATION'
    elif damage == 'rule_version': d['audit_policy']['rules'][0]['version'] = 2
    elif damage == 'source_field': d['findings'][0]['observations'][0]['field'] = 'unobserved.runtime'
    elif damage == 'source_hash': d['findings'][0]['source_snapshot_sha256'] = 'a' * 64
    elif damage == 'coverage': d['coverage']['examined_supported_modules'] = 100
    elif damage == 'summary': d['summary']['finding_counts']['STATIC_INCONSISTENCY'] = 0
    elif damage == 'limitations': d['limitations'] = []
    elif damage == 'finding_order': d['findings'].reverse()
    elif damage == 'duplicate_finding': d['findings'].append(copy.deepcopy(d['findings'][0]))
    elif damage == 'extra': d['VERIFIED_IMPROVEMENT'] = True
    elif damage == 'version': d['schema_version'] = True
    elif damage == 'snapshot':
        d['source_snapshot']['summary']['parameter_elements']['value'] += 1
        d['source_snapshot']['content_sha256'] = digest({k: v for k, v in d['source_snapshot'].items() if k != 'content_sha256'})
    d['content_sha256'] = digest({k: v for k, v in d.items() if k != 'content_sha256'})
    with audit_only(monkeypatch), pytest.raises(ModelArchitectureAuditError): ModelArchitectureAudit(d)


def test_frozen_detached_canonical_and_exclusive_export(monkeypatch, tmp_path):
    result = run(monkeypatch, nn.Linear(3, 2)); raw = result.to_json()
    with pytest.raises(FrozenInstanceError): result._payload = '{}'
    with pytest.raises(FrozenInstanceError): result.findings[0].code = 'invented'
    with pytest.raises(FrozenInstanceError): result.audit_policy.version = 2
    with pytest.raises(FrozenInstanceError): result.module_assessments[0].status = 'inconsistent'
    data = result.to_dict(); data['findings'].clear()
    result.coverage['module_assessments'].clear()
    result.summary['total_findings'] = 0
    fact = result.findings[0].observations[0]; fact.value['value'] = 0
    assert raw == result.to_json()
    with audit_only(monkeypatch): assert audit_model_architecture(result.source_snapshot).to_json() == raw
    path = tmp_path / 'audit.json'; result.to_json(path)
    with pytest.raises(FileExistsError): result.to_json(path)
    assert path.read_text(encoding='utf-8') == raw
    assert ModelArchitectureAudit.from_dict(result.to_dict()) == result
    path.write_text('{"format":1,"format":2}', encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate_json_key'): ModelArchitectureAudit.from_json(path)


@pytest.mark.parametrize('limit,value', [('max_entries', 1), ('max_bytes', 100), ('max_entries', 0),
    ('max_entries', True), ('max_bytes', 64 * 1024**2 + 1)])
def test_explicit_limits_never_truncate(monkeypatch, limit, value):
    result = run(monkeypatch, nn.Linear(3, 2))
    with audit_only(monkeypatch), pytest.raises(ModelArchitectureAuditError, match='limit'):
        audit_model_architecture(result.source_snapshot, **{limit: value})


def test_only_validated_snapshot_is_accepted_without_loading_or_introspection(monkeypatch):
    snapshot = capture_model_structure(nn.Linear(3, 2))
    for value in (nn.Linear(3, 2), snapshot.to_dict(), 'untrusted.factory:load', 'checkpoint.pt', None):
        with audit_only(monkeypatch), pytest.raises(ModelArchitectureAuditError, match='snapshot_required'):
            audit_model_architecture(value)
    object.__setattr__(snapshot, '_payload', '{}')
    with audit_only(monkeypatch), pytest.raises(ModelArchitectureAuditError): audit_model_architecture(snapshot)


def test_snapshot_and_structure_comparison_previous_canonical_bytes_unchanged(monkeypatch):
    from hyso import compare_model_structures
    a = capture_model_structure(cnn()); b = capture_model_structure(cnn())
    before = (a.to_json(), b.to_json(), compare_model_structures(a, b).to_json())
    with audit_only(monkeypatch):
        audit_model_architecture(a)
        assert before == (a.to_json(), b.to_json(), compare_model_structures(a, b).to_json())
        for module, typename, current, argument in [
            ('structure', 'ModelStructureSnapshot', a.to_json(), a.to_dict()),
            ('structure_comparison', 'ModelStructureComparison', before[2], compare_model_structures(a, b).to_dict())]:
            text = subprocess.run(['git', 'show', '5126559a9c6af9dd9ae43d04f2325164f410adf1:hyso/' + module + '.py'],
                capture_output=True, text=True, check=True).stdout
            namespace = dict(__name__='hyso._previous_' + module, __package__='hyso')
            exec(compile(text, '<trusted-local-previous-' + module + '>', 'exec'), namespace)
            assert namespace[typename](argument).to_json() == current


@pytest.mark.parametrize('kind,code', [('linear_in', 'STATIC_DIMENSION_INVALID'),
    ('linear_out', 'STATIC_DIMENSION_INVALID'), ('kernel', 'CONV_KERNEL_SIZE_INVALID')])
def test_supported_integer_domain_contradictions_are_static(monkeypatch, kind, code):
    model = nn.Conv2d(2, 4, 3) if kind == 'kernel' else nn.Linear(3, 2)
    if kind == 'kernel': model.kernel_size = (-1, 3)
    elif kind == 'linear_in': model.in_features = -1
    else: model.out_features = -1
    result = run(monkeypatch, model)
    assert code in [f.code for f in issues(result)]
    assert all(f.expected is not None for f in issues(result))


def test_unrelated_configuration_gap_retained_alongside_supported_shape_checks(monkeypatch):
    model = nn.Conv2d(2, 4, 1); model.padding_mode = object()
    result = run(monkeypatch, model)
    assert not issues(result) and result.module_assessments[0].status == 'consistent'
    assert 'STATIC_CONFIGURATION_UNAVAILABLE' in codes(result)
    assert result.coverage['status'] == 'partial'


def test_dtype_distribution_and_tied_concentrations_are_not_additive(monkeypatch):
    model = nn.Module(); model.a = nn.Linear(3, 2); model.b = nn.Linear(3, 2)
    model.a.bias = nn.Parameter(torch.ones(2, dtype=torch.float64), requires_grad=False)
    model.b.weight = model.a.weight; model.b.bias = model.a.bias
    result = run(monkeypatch, model)
    assert not issues(result)
    assert result.source_snapshot.summary.parameter_elements.value == 8
    inventory = observed(result, 'PARAMETER_INVENTORY')
    dtypes = next(f.value for f in inventory.observations if f.field == 'summary.parameter_dtypes')
    assert [(d['dtype'], d['elements']['value']) for d in dtypes] == [('torch.float32', 6), ('torch.float64', 2)]
    assert inventory.derived.value['frozen_fraction'] == .25
    shares = [f.derived.value['fraction'] for f in result.findings if f.code == 'DIRECT_PARAMETER_CONCENTRATION']
    assert shares == [1., 1.]  # overlapping ownership, never a new model-wide total


def test_weight_values_are_not_structure_or_behavior_identity(monkeypatch):
    a, b = nn.Linear(3, 2), nn.Linear(3, 2)
    with torch.no_grad(): b.weight.fill_(123)
    left, right = run(monkeypatch, a), run(monkeypatch, b)
    assert left.content_sha256 == right.content_sha256
    assert left.to_json() == right.to_json()
    assert any('model-state' in text for text in left.limitations)


def test_known_contradiction_does_not_hide_an_independent_missing_check(monkeypatch):
    model = nn.Linear(3, 2); model.in_features = 9; model.bias = UninitializedParameter()
    result = run(monkeypatch, model)
    assert result.module_assessments[0].status == 'inconsistent'
    assert 'LINEAR_WEIGHT_SHAPE_MISMATCH' in codes(result)
    assert 'TENSOR_METADATA_UNSUPPORTED' in codes(result)
    assert result.coverage['module_ids_with_insufficient_checks'] == ['m0']
    assert result.coverage['unassessed_module_ids'] == ['m0']
    assert result.coverage['check_counts']['insufficient_evidence'] == 1
    assert result.coverage['status'] == 'partial'
